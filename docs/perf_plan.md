# cp_balance 性能测量与优化方案

> 2026-09-23：zigzag 改为 attention 内部选行（模型主流恢复 replicated），本文中“模型边界切行/固定序归约”的描述已过时，见 `docs/dp1_zigzag_acc_fix.md`。

对象：`vllm-ascend`（分支 `cp_balance`，当前 b64c9569b）相对 `vllm-ascend-base`
（`base-dp1` 分支 = `main` aff1b74b6 + dp=1 门修一行）新增的 cp_balance。
模型 GLM-5.2-W4A8C8，TP16，DSA-CP（dp=1、SP 关闭；见 `docs/scripts_review.md` §4.1/§8）。

已完成：一轮代码精简（第 4 节），diff vs base 从 2151/-513 行降到 1855/-511 行。
未开始：profiling 采集与依据数据的调优（第 1、3、5 节）。

## 1. 一次远端会话要拿到的数据

四个配置串行跑，每个配置一次服务起停（约 10 分钟启动）：

    cd <harness>
    bash tests/run_tests.sh --only perf

| 配置 | 代码树 | CP_BALANCE | 作用 |
| --- | --- | --- | --- |
| prof_cur_cp0 | 当前 | 0 | 与 base 等价的噪声基准 |
| prof_cur_cp1 | 当前 | 1 | zigzag 现状 |
| prof_cur_cp0_repeat | 当前 | 0 | 同路径重跑（噪声地板） |

（表里用的是 A3 的配置名；`CP_BALANCE_FAMILY=a5` 时实际跑 `prof_a5_*`，由 `tests/lib/roles.tsv` 决定。）
| prof_base_cp0 | base | 0 | 原版 DSA-CP 参照 |

`prof_cur_cp0` 与 `prof_base_cp0` 的差异就是测量本身的噪声地板；两者不重合时，
后面所有对比都不可信，先解决测量问题。

`tests/perf`（p10 + p20~p24）= 采集 + 解析 + 对比 + 归因 + 打包。采集部分每个配置：起服务 → 2 条 long prompt
热身（不采集）→ POST /start_profile → 4 条 long prompt（串行，每条一个 prefill
batch）→ POST /stop_profile → 停服务 → 写 prof_<name>.json。profiling 开关由配置里的
`"profiler": {"enabled": true}` 生成 --profiler-config，/start_profile 只在设置该参数后才存在。

三个必须知道的操作事实（依据见 docs/profiling_guide.md）：

1. 环境变量方式（VLLM_TORCH_PROFILER_DIR 等）在当前 vLLM 上已删除，只有
   --profiler-config 有效；不带它时 /start_profile 返回 404。
2. --distributed-executor-backend mp 下 worker 是 daemon 进程，torch_npu 的解析器
   拒绝在 daemon 里解析，所以 /stop_profile 之后不会自动出现 ASCEND_PROFILER_OUTPUT，
   必须像 profile_analyse.py 那样另起进程补跑 analyse()。analyse() 接收父目录并自己
   并行处理所有 *_ascend_pt，不要在一个进程里按 rank 循环调用（ProfilerConfig 是单例）。
3. profiler 输出目录默认就是 `<harness>/<配置名>`（`profiler.dir` 可覆盖）；
   如果要手写 `torch_profiler_dir`，必须是当前用户可读写的绝对路径。data_simplification=True 会在
   解析成功后删掉原始数据，所以 ASCEND_PROFILER_OUTPUT/ 是唯一证据。

要回传的只有每个目录下的 summary.json（几 KB）与 export/（每 rank 的小 CSV 与
communication.json），加上 prof_<name>.json、profile_<name>.log 的指纹行；
*_ascend_pt 原始 trace 与 kernel_details.csv 不用拷。

## 1b. 结论一律来自全量 78 层；6 层只用于快速复看

这一轮以 `tests/perf` 的 78 层四组为准。理由：

- 要回答的三个问题都是“每层开销 vs 每步开销的比值”和“rank 间失衡”，而 6 层会把
  每步固定开销（metadata、embedding、出口 gather、logits）的占比放大约 13 倍。
  用它来判断 cp_balance 划不划算、H0 是不是大头，会得到错的结论。
- `profile_order.py` 的顺序与归因只读一个 rank 的 trace，78 层和 6 层的做法完全一样，
  全量并不额外花时间。
- 全量只比 6 层多约 35 分钟的一次性启动时间，不是瓶颈；而 6 层依赖
  `--hf-overrides` + W4A8 量化权重跳层，这条路还没在真机上验证过，
  万一不成立就白跑一轮。

6 层（`bash perf/profile_l6.sh`）保留的用途只有一个：某处优化落地后，想快速再看一眼
顺序和归因有没有变化。它由启动参数覆盖，不改模型目录：`--hf-overrides` 里
`num_hidden_layers=6`，并把 `indexer_types` 按真实周期裁成
`[full, shared, shared, shared, full, shared]`。第二步不能省：不裁的话前 6 层全是 full，
而真实模型里多数层是 shared，`patch_deepseek_v2.py:57` 会让这些层跳过 indexer / topk，
算子构成就失真了。指纹行里有 `LAYERS=6` / `LAYERS=all`，不会混。

### 全量一轮的四组配置

| 配置 | 代码树 | CP_BALANCE | 回答什么 |
| --- | --- | --- | --- |
| `prof_cur_cp0` | 当前 | 0 | 不开 cp_balance 的基线（也是与 base 的等价性对照） |
| `prof_cur_cp1` | 当前 | 1 | H0：o_proj 全量权重 all_gather 占多少；zigzag 的净收益 |
| `prof_cur_cp0_repeat` | 当前 | 0 | 同路径重跑（噪声地板） |
| `prof_base_cp0` | base | 0 | 噪声地板（与 `prof_cur_cp0` 应几乎重合） |

四个配置共用 `_profile_common.json`，所以 `enforce-eager`、
`VLLM_CUSTOM_SCOPES_FOR_PROFILING`、`debug=0`、`MIN_TOKENS` 都一致，
唯一变量就是上表的 `CP_BALANCE` 那列。

### 算子顺序与代码归因（profile_order.py）

    python3 profile_order.py prof_l6_cur_cp1 --rank rank0
    python3 profile_order.py prof_l6_cur_cp1 --rank rank0 --devices
    python3 profile_order.py prof_l6_cur_cp1 --rank rank0 --trim

输入是 `ASCEND_PROFILER_OUTPUT/trace_view.json`（流式解析，不整文件加载）。输出四类信息：

1. `head`：第一个完整层之前的非重复部分（embedding、通信初始化、出口）。
2. `one decoder-layer cycle`：自动用出现次数 >= 3 的命名区间做周期切分，打印一层里的
   顺序、单事件耗时、累计占比。
3. `cycle: share by name`：一层内按名字汇总，直接看谁占大头。
4. `device time by enclosing host scope`（`--devices`）：把每个 device kernel 归到包住它的
   host `record_function` 区间上，按 scope 汇总 device 时间占比，这是耗时来源最直接的答案。
   同一份 trace 内的 host/device 事件共用时钟；命名区间优先于普通 cpu_op 取最内层。

每个名字后面跟一条来源链：

| 名字形态 | 映射到 |
| --- | --- |
| `SFA-5.3/xx` | 命中 `record_function_or_nullcontext` 或 `_sfa_5_3_scope` 的调用点，给 文件:行号（当前 35 条标签） |
| `torch.ops.vllm.X` | `direct_register_custom_op(op_name=)` 的注册点（当前 16 个） |
| `Hccl*` / `*ReduceScatter*` | 对应集合通信原语的候选调用点清单（当前 11 个原语） |
| `aten::*` / 裸 kernel 名 | 无，这是 torch 内建或 CANN kernel，本身不含代码位置 |

前提：`VLLM_CUSTOM_SCOPES_FOR_PROFILING=1`。没有它 `record_function_or_nullcontext` 退化成
nullcontext（`vllm/v1/utils.py:747`），trace 里没有任何命名区间，只剩裸 kernel 名。`_profile_common.json` 已设置。
`--trim` 会写一份 `order_<rank>.json`（几十 KB），回传它即可，不必拉原始 trace。

### 回传清单（按这一轮）

- `prof_l6_*/summary.json` 与 `prof_l6_*/export/`（几 MB）
- `prof_l6_*/order_rank0.json`（`--trim` 产物，几十 KB）
- `prof_l6_*.json`、`profile_prof_l6_*.log` 的指纹行

## 2. 要看的指标

按重要性排序：

1. step_trace_time.csv 的每步 Computing 在 rank 间的 max/mean 比。cp_balance 是负载
   均衡改造，这个比值从 CP=0 的失衡降到接近 1，才说明改造有效。
2. HCCL 算子总时间与调用次数（compare 输出的 comm 一节）：zigzag 不再改归约，
   两侧的集合通信构成应基本一致。
3. 算子构成差（compare 输出的 delta 一节）。FlashAttentionScore / LightningIndexer
   的 rank 间分布应当变均衡：zigzag 只改 attention 内部的选行/放回。
4. 客户端 mean_elapsed_s（prof_<name>.json）只作交叉验证：profiling 本身有开销，
   绝对值不作为结论。

## 3. 性能假设（按预期影响排序）

以下都从代码读出，待 profiling 证实或证伪。

先说一个决定量级的事实：AscendSFAMetadataBuilder.build() 每个前向步只被调用一次
（model_runner_v1.py:2974，按 attention group 分组，同一 group 的层共享一份
metadata），所以 metadata builder 里的开销是每步一份，而 linear_op.py /
register_custom_ops.py 里的开销是每层一份（78 层）。metadata 侧的冗余因此只是
顺手删掉，真正要测的是 H0。

### H0：基线里可能最大的通信项是每层的 o_proj 全量权重 all_gather（先确认）

DSA-CP prefill 下，SFA 输出与 TP 分片的 o_proj 不兼容，所以每层都要把 TP 的 o_proj
分片 all_gather 成完整权重再用：发起在 sfa_v1.py:1835-1848
（all_gather_async(..., output=self.o_proj_full_gather_pool)），缓冲池按
(device, dtype, gather_dim, full_shape) 复用（sfa_v1.py:1057-1080，池是静态的
AscendSFAImpl.o_proj_full_pools），等待点在 o_proj 使用前（sfa_v1.py:2213 附近）。

量级估算：hidden=6144、o_proj 全量 = 6144x16384，bf16 约 192MiB；W4A8 量化存储按
4bit 折算约 50MiB。也就是每 rank 每个 forward 收到 78x(50~192)MiB ≈ 4~15GiB。

这是 base 与 cp_balance 共有的开销，所以它既可能盖住 zigzag 的收益，也可能是真正
值得优化的地方。第一轮 profiling 必须先区分：trace 里 Hccl...AllGather 的总时间占比
是多少、调用次数是不是约 78x2（权重 + 量化参数）。

若它确实占大头，那么删掉每层权重 all_gather 的收益会远大于调 cp_balance 本身。

### H1：默认归约模式（已作废）

（2026-09-23 起 `reduce_mode` 及其 A/B 已随 zigzag 改法删除，见 `docs/dp1_zigzag_acc_fix.md`。）

### H2：每层多一层 Python 包装（已消除）

原先 patch_deepseek_v2.py 把 DeepseekV2DecoderLayer.forward 换成一份上游 forward 的
拷贝，只为了跳过两个 sequence-parallel 分支。DP=1 下
parallel_config.use_sequence_parallel_moe 要求 data_parallel_size > 1
（vllm/config/parallel.py:653-668），而 zigzag 资格门拒掉 DP>1，所以那两个分支本来
就不会执行，拷贝是多余的。本轮已删（第 4 节 S8）。

> 更正（2026-09-21）：S8 的前提只在「dp=1 且不启用 SP-MoE」时成立。本地已把 DSA-CP 与
> `use_sequence_parallel_moe` 解耦（`docs/scripts_review.md` §4.1），dp=1 下 SP 仍然关，所以 S8 目前有效；
> 但如果走 §8 的出路 (b)（dp=1 也开 SP），这两个上游分支会重新可达，S8 的删除必须重审。
> 行号也变了：`vllm/config/parallel.py` 现在是 699-726。

### H3：zigzag 激活时多算一次 RoPE 查找表（低，每步）

sfa_v1.py:762 用 zigzag 位置算了一份 cos/sin，sfa_v1.py:768 又用连续切片的位置算了一份
cos_continuous/sin_continuous。get_cos_and_sin_mla（ops/rotary_embedding.py:90）在
use_cache=False 时只是一次 _cos_cache[positions] 索引，所以第二份的代价是两次
index_select，不是重算一张表；而且 metadata 每步只建一次。收益太小，暂不动。

第二份只在 zigzag 被回退时才用到，而资格判定已经把 draft / V2 runner / DP>1 全部拒掉，
即回退在正常路径上不会发生（第 4 节 S9 记录了这一冗余）。

### H4：embedding 边界多两次集合通信（已作废）

（2026-09-23 起 embedding 边界切行与 `VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL` 已随 zigzag 改法删除，见 `docs/dp1_zigzag_acc_fix.md`。）

## 4. 已完成的精简（commit 658b836ba）

原则：只删可证明无读取点、或在配置下永不成立的东西，不改数值路径。每改一项用
pyflakes + check_b_path.py 静态门控兜一遍。

| 编号 | 位置 | 处理 |
| --- | --- | --- |
| S1 | cp_zigzag.py can_enable_zigzag_for_batch | 删（45 行，全仓库无调用点） |
| S2 | DSACPContext 15 个无读取点字段 | 删（q_half、split_list、cp_reverse_index、reverse_split_len、prefix_offsets、q/kv_len prev/next、actual_seq_q_*_list、kv_len_*_list、total_q_*_tokens） |
| S3 | _build_zigzag_meta 12 个 device 张量 | 删（每个都是一次 H2D + kernel 启动）；[CP_BALANCE][plan] 改印 CPU 侧 plan list，顺带去掉一次 device→host 同步 |
| S4 | ascend_forward_context 回退函数里的对应复位 | 删 |
| S5 | ZigzagPlan 的 split_list / cp_reverse_index / reverse_split_len / prefix_offsets 及 build_zigzag_plan 里的 O(T) 计算 | 删 |
| S6 | sfa_v1.py _supports_npu_advanced_index / _NPU_INDEX_UNSUPPORTED_FP8_DTYPES | 删（无调用点） |
| S7 | sfa_v1.py _zigzag_gate_reason | 删（只是给 zigzag_ineligible_reason 套了一个恒假判断） |
| S8 | patch_deepseek_v2.py _zigzag_layer_forward + _patched_decoder_layer_forward | 删（约 85 行）；前提是 DP=1 下 use_sequence_parallel_moe 恒假（该前提的更正见 H2 段） |
| S9 | cp_zigzag.py zigzag_shard_positions、zigzag_shard_tensor 的重复 dim 分支 | 删/合并（zigzag_shard_positions 就是 zigzag_shard_tensor(x, 0)） |
| S10 | model_runner_v1.py _pad_for_sequence_parallelism 未使用的 num_scheduled_tokens_np 形参 | 删 |
| P3 | sfa_v1.py 两个 KV writer 各做一次 slot_mapping[zigzag_gather_index] | 改为 metadata 建时算一次，存 DSACPContext.slot_mapping_cp_gathered |

保留但记录在案：

- SGLang 遗留的 effective_query_lens / block_sizes / query_lens 等 plan 字段没有运行期
  成本，留着便于读代码。
- attention/utils.py 的 is_prefilling_cpu 与 is_prefilling 目前是同一个对象，但上游把
  is_prefilling 标注成 torch.Tensor（vllm/v1/attention/backend.py:459），保留一个显式
  声明 host 侧的字段，避免以后有人传设备张量进来。
- 余数分配的最大流实现（cp_zigzag.py 约 200 行）保留：它保证每 rank 行数严格相等这个
  关键不变量，替换成贪心会改变块分布，必须先有首 token 对比的回归
  证据，等 profiling 后再单独一轮做。

## 5. 下一步

1. 远端 git pull 到最新，先跑静态自检与等价性验收（都是秒级）：
   python perf/check_cp_balance_fields.py --repo <cur 树>、
   python accuracy/check_b_path.py --repo <cur 树> --base-repo <base 树>、
   python accuracy/compare_first_token.py --preflight（可选）；
   然后 bash tests/run_tests.sh --only accuracy/a10_matrix_gate（B == base + 噪声地板）。
   注：selftest_plan.py（zigzag plan 的随机自测）已删除。它的默认值 --cases 2000 会让
   脚本跑几十秒到几分钟且没有进度输出，看着像卡死；而它覆盖的那套不变量在上一轮改动里
   没有被动过（只删了 ZigzagPlan 的无读取点字段），并且已经用等价条件跑过 700 组通过。
   真要重做这一步时，把它写成 20 组、带进度输出的一次性脚本即可。
2. 跑 bash tests/run_tests.sh --only perf（78 层四组 + 解析 + 报告）。它一次给出全部结论依据：
   rank 间 step Computing 的 max/mean、HCCL 次数与时间、算子构成差、以及 rank0 的
   算子顺序与 device 归因。回传 summary.json + export/ + order_rank0.json +
   prof_*.json + 服务日志的指纹行。
3. 只有想在两次优化之间快速复看顺序/归因时，才跑 bash perf/profile_l6.sh（6 层，结论不取它）。
4. 依据 H0 的实测结果决定：是否继续动 o_proj 权重 all_gather、以及是否做余数分配重写。
5. 优化落地后重跑 prof_cur_cp0 / prof_cur_cp1 验证收益，并用
   compare_first_token.py compare --require-text 确认数值验收仍 PASS。
