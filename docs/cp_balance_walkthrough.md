> 历史文档（2026-09 上旬的交接/评审期记录）：结论可能已过时，当前状态以 `docs/scripts_review.md` 为准。
>
> 已变事实（2026-09-21 更正）：
> - §13 的命令路径已变：`cp_balance/check_cp_balance_fields.py` → `perf/check_cp_balance_fields.py`、
>   `cp_balance/check_branch.py` → `accuracy/check_branch.py`、`bash cp_balance/run_matrix.sh …` 已删除，
>   改用 `bash tests/run_tests.sh --only accuracy/a10_matrix_gate#b_matrix`。
> - §6 表里“o_proj 未改”只对老线成立：main 线的 zigzag o_proj 出口需要改，已在 6ca45f53e 修好（`scripts_review.md` §4.1b）。
> - `MIN_TOKENS`：env 默认 8192、配置显式 2048（现在在 `configs/_base.json`），以配置为准。

# cp_balance 代码修改走读：跟着一次 forward 走

对象：`vllm-ascend` 的 `cp_balance` 分支，HEAD `658b836ba`；对照 `vllm-ascend-base` 的 `c7990e5e4`。

> 行号说明：本文的行号按 `658b836ba` 读；第二轮精简（`docs/cp_balance_review_round2.md`，删掉 5 个
> 无读者字段、合并 topk/SFA 重复分支、去死别名等）之后，`sfa_v1.py` / `cp_zigzag.py` /
> `ascend_forward_context.py` 的行号有少量偏移，函数名与结构不变。

    git -C vllm-ascend diff c7990e5e4 HEAD

本文按**一次 prefill forward 的执行顺序**走，每一站统一四段：base 做什么 / 当前做什么 / 差在哪 / 为什么与 review 重点。**没被改到的环节也写出来并标注「未改」**——review 时最怕的是不知道哪里没动。

与其它文档分工：`base_dsa_cp_flow.md` 是原版完整链路，`cur_cp_balance_flow.md` 是现版完整链路与候选清单，`profiling_guide.md` 管采集，`perf_plan.md` 管测量结论。

## 1. 全景：一次 prefill forward

     NPUModelRunner.execute_model                       model_runner_v1.py:1698
       ├─ _prepare_inputs                              model_runner_v1.py:821
       ├─ _determine_batch_execution_and_padding
       │    └─ _pad_for_sequence_parallelism           model_runner_v1.py:2613
       ├─ _build_attention_metadata                    model_runner_v1.py:2775
       │    └─ AscendSFAMetadataBuilder.build()        ← 布局在这里定死
       ├─ set_ascend_forward_context                   ascend_forward_context.py:182
       │    └─ zigzag_cp_active                        ← 这一 forward 走哪条路在这里定
       ├─ _model_forward                               model_runner_v1.py:2573
       │    └─ DeepseekV2Model.forward（被 patch）     patch_deepseek_v2.py:305
       │         ├─ embedding + positions（第 4 站）
       │         ├─ 78 × DeepseekV2DecoderLayer.forward（第 5~6 站）
       │         └─ self.norm
       ├─ 出口 gather                                  model_runner_v1.py:2605
       └─ logits → sample

| # | 站 | 是否改了 |
| --- | --- | --- |
| 1 | padding | 行为未改，只清了形参 |
| 2 | metadata builder | **改了**：布局从「连续区间」变成「置换」 |
| 3 | forward context | **改了**：新增逐 forward 的 zigzag 开关 |
| 4 | 模型入口 embedding / positions | **改了**：先 gather 回全长再切 |
| 5 | 逐层 attention（SFA + KV 写回） | **改了**：写回 slot 顺序 + 合并 prev/next 调用 |
| 6 | 逐层 o_proj / MLP / MoE | o_proj 未改；归约与 MoE 输入序**改了** |
| 7 | 出口 gather | **改了**：拼接后多一次逆置换 |
| — | norm / logits / sample | 未改 |

## 2. 第一站：padding（行为未改，但它是前提）

**base**（`vllm-ascend-base/vllm_ascend/worker/model_runner_v1.py:2605`）：`enable_sp` 打开时 `round_up(num_scheduled_tokens, tp_size)`。

**当前**（`vllm-ascend/worker/model_runner_v1.py:2613`）：函数体一字未改，只删掉了没人用的第二个形参 `num_scheduled_tokens_np`。

**为什么值得单列**：不变量 1 要求 `num_tokens_pad % cp_size == 0`，**保证它的地方就是这里**。metadata builder 不做二次 padding，只用 `zigzag_ineligible_reason` 里的 `pad%cp_size!=0` 把不合规的批挡回去。

**review 重点**：以后改对齐规则（比如改成按 `2*cp_size` 对齐），必须同时改 `zigzag_ineligible_reason`，否则 metadata 与 runner 对不上，各 rank 通信形状不一致会直接挂死。

## 3. 第二站：metadata builder —— 布局在这里定死

**base**（`vllm-ascend-base/vllm_ascend/attention/sfa_v1.py` 的 `AscendSFAMetadataBuilder._build`）：

| 位置 | 做什么 |
| --- | --- |
| `:403-404` | `local_start = rank_in_group * num_tokens_per_device` |
| `:419` | `slot_mapping_cp = slot_mapping[local_start:local_end_with_pad]` |
| `:421` | `cos = cos[local_start:local_end_with_pad]` |
| `:461-470` | 把每个请求的 `[global_start, global_end)` 与窗口求交，重算局部视角的 `actual_seq_lengths_query/key` |
| `:474` | 构造 `DSACPContext`，**8 个字段** |

一句话：**布局是一个连续区间**。

**当前**（`vllm-ascend/attention/sfa_v1.py` 同一函数）：

| 位置 | 做什么 |
| --- | --- |
| `:567-577` | 仍然先算连续切片那一套，留着当回退值 |
| `:589` | 调 `zigzag_ineligible_reason(...)` 判定，返回 `None` 才走 zigzag |
| `:611` | `_build_zigzag_meta(...)`（定义在 `:242`）算出计划并落成 device 张量 |
| `:651` | `block_table_zigzag`：每个真实请求的行重复一遍，变成 `2*num_reqs` 行 |
| `:663-667` | `local_positions = input_positions[zigzag_index]` 后直接生成 cos/sin |
| `:698` | `slot_mapping_cp = slot_mapping[zigzag_index]` |
| `:800` | `slot_mapping_cp_gathered`：按 `zigzag_gather_index` 再排一次，给 KV 写回用 |
| `:773` | 构造 `DSACPContext`，**18 个字段**（含 `fallback_slot_mapping_cp` / `fallback_cos` / `fallback_sin`） |

**差在哪**：布局从「一个连续区间」变成「一个置换」——`zigzag_index`（本 rank 拥有的全局位置，长度 `pad/cp_size`）、`zigzag_gather_index`（rank 拼回自然序的置换，长度 `pad`）、`inv_gather_index`（上一条的逆）。长度表也从「每请求一段」变成「prev/next 各一段，拼成 2B 个 batch 供一次算子调用」。另外多留三个 `fallback_*` 张量给第三站的事后否决。

**为什么**：attention 计算量随 causal 前缀增长，连续切片让靠后的 rank 前缀最长；每条序列头尾各给每 rank 一块，各 rank 的前缀长度之和就趋于均匀，而**每 rank 行数不变**，于是 FlashComm 集合通信形状不变。

**review 重点**：
1. `:589` 的判定与第 1 站的 padding 必须同源（现在调的是同一个函数，刻意设计，别分叉）；
2. `fallback_*` 必须齐全，缺一个 `_disable_zigzag_metadata_for_fallback` 会直接抛错，不静默降级；
3. `DSACPContext` 全仓库只有 `:773` 一个构造点，加字段要同步 `check_cp_balance_fields.py` 能覆盖的三方。

## 4. 第三站：forward context —— 这一 forward 走哪条路

**base**（`vllm-ascend-base/vllm_ascend/ascend_forward_context.py`）：没有 zigzag 概念，只建 MoE 通信方式、pad_size、mc2_mask。

**当前**（`vllm-ascend/ascend_forward_context.py`）：

| 位置 | 做什么 |
| --- | --- |
| `:70` `_iter_attn_metadata` | 兼容 `{layer: metadata}` 与 ubatching 的 list-of-dict |
| `:89` `_find_zigzag_cp_context` | 挑出带 `zigzag_index` 的 `DSACPContext` |
| `:105` `_disable_zigzag_metadata_for_fallback` | 事后否决时把 metadata 改回连续切片 |
| `:219-233` | 判定 `zigzag_cp_active`；draft / V2 runner 命中就回退 |
| `:307-310` | DP>1 再回退一次 |
| `:328-332` | 写进 forward context（V2 runner 走 `additional_kwargs`） |
| `:334-342` | `input_ids` 转 int64 并按 `zigzag_gather_index` 重排一次，供所有 MoE 层复用 |
| `:355-361` | 同样重排 `mc2_mask`，与 `input_ids` **同一张置换** |
| `:605` `zigzag_active()` | 所有归约路经的门控，逐 forward 判定 |

**差在哪**：base 没有「每个 forward 可能走不同 token 布局」这个概念；当前引入了一个**逐 forward** 的布尔量。

**为什么**：metadata builder 在 runner 里、比 forward context 早，拿不到「这一批是不是 draft 步」「是不是 V2 runner」「DP 是否大于 1」，所以要在 context 里补一次否决，并把已建好的 metadata 改回连续切片——这正是 `fallback_*` 的用途。

**review 重点**：
1. `zigzag_active()` 是**逐 forward** 的：同一进程里既有 zigzag 批次（长 prefill）也有连续切片批次（decode、短 prompt）。cp_balance 专属改动必须挂在它后面——历史上出过「挂配置开关 `enable_dsa_cp()`」的 bug，导致 `CP_BALANCE=0` 时归约也被改；
2. `input_ids` 与 `mc2_mask` 必须用同一张置换，否则 MoE 路由和激活行对不上。

## 5. 第四站：模型入口 —— embedding 与 positions

**base**（`vllm-ascend-base/vllm_ascend/patch/worker/patch_deepseek_v2.py:314`）：

    hidden_states = self.embed_input_ids(input_ids)   # FlashComm 下 = reduce_scatter 后的 L 行局部切片
    ...
    for idx, layer in enumerate(...):                 # positions 始终是全长 T 行
        hidden_states, residual = layer(positions, hidden_states, residual, llama_4_scaling)

**当前**（`vllm-ascend/patch/worker/patch_deepseek_v2.py`）：

| 位置 | 做什么 |
| --- | --- |
| `:314` | 取 `_EXTRA_CTX.zigzag_cp_active` |
| `:350-362` | embedding 若已是全长就跳过，否则先 `all_gather` 回全长，再 `zigzag_shard_tensor` 切成本 rank 的 `[prev, next]` |
| `:375-376` | `positions = zigzag_shard_tensor(positions)` —— **positions 也切成 L 行** |
| `:401` / `:410` | zigzag 下跳过 aux 与出口的全量拼接（出口交给 runner，见第七站） |

**差在哪**：embedding 多了一次 `all_gather`；`positions` 从全长 T 行变成 L 行。

**为什么**：embedding 权重按 vocab 做 TP 分片，只有**每 rank 喂同样的全量 token ids** 再 all_reduce 才能还原完整 embedding 行；喂 rank-local ids 会把不同 token 的 embedding 加起来（`ops/vocab_parallel_embedding.py:170` 的注释专门写了）。positions 一起切是因为第七站靠形状推断要不要拼接，不切会走错分支。

**review 重点**：
1. 这里是「一次 forward 在 embedding 处走三次集合通信」的来源（reduce_scatter → 丢全长 → all_gather → 切片）。`VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL=1` 是想省掉中间那次，代价是一次全量 hidden 的 all_reduce，且**没有实测收益证据，默认关闭**；
2. `positions` 的切片不是可选的。

## 6. 第五站：逐层 —— attention（SFA 与 KV 写回）

层循环在 `patch_deepseek_v2.py:395-405`；每层内部顺序是 input_layernorm → self_attn → post_attention_layernorm → mlp，**这里只讲 self_attn**。

**base**：

| 位置 | 做什么 |
| --- | --- |
| `sfa_v1.py(base):1756` / `:1766` / `:1776` | KV、indexer 的 k、k_scale 各做一次 `all_gather_async`，汇总成**全量 T 行** |
| `sfa_v1.py(base):1853-1857` | DSA-CP 分支：等 KV 的 all_gather 完成后，用 `slot_mapping_sfa[:num_actual_tokens]` 写 **`num_actual_tokens` 行**（不是全量） |
| `sfa_v1.py(base):2212-2228` 附近 | o_proj 用异步 all_gather 来的**全量权重**做本地 GEMM，输出 L 行（base 与当前一致，详见第六站） |
| `sfa_v1.py(base)` topk / SFA 调用 | 用局部视角的 `actual_seq_lengths_query/key` 各调一次 |

注意 base 里 `sfa_v1.py(base):1825` 那处 scatter 属于 `if self.enable_sparse_sfa_c8 and not self.enable_dsa_cp:` 分支，**本配置开了 `enable_dsa_cp`，不走它**——读代码时很容易把它当成 DSA-CP 的写回路径。

另外：**层内不做 SP 的 gather/scatter**——`DeepseekV2DecoderLayer.forward` 里那两个 sequence-parallel 分支由 `self.use_sequence_parallel_moe` 控制，DP=1 时恒假（`vllm/config/parallel.py:653-668`）。

**当前**（`vllm-ascend/attention/sfa_v1.py`）：

| 位置 | 做什么 |
| --- | --- |
| `:2137` / `:2147` / `:2157` | KV 的 all_gather **未改**，仍然汇总成全量 T 行 |
| `:2244-2245` | KV 写回改用 `dsa_cp_context.slot_mapping_cp_gathered`（每 batch 算一次，不再每层做置换） |
| `:2567-2568` | indexer 的 KV 写回同样改用 `slot_mapping_cp_gathered` |
| `:2552` | zigzag 下关掉 `store_kv_block` 快速写路径，退回 scatter |
| `:2616` | topk：一次 `indexer_select_post_process`，用合并的 `actual_seq_lengths_*_zigzag` 与 `block_table_zigzag`，把 prev/next 当 2B 个 batch |
| `:2659` | sparse attention：同理一次 `_execute_sparse_flash_attention_process` |
| `device/device_op.py` | `get_indexer_post_process` 系列加 `block_table` 参数（默认取 `attn_metadata.block_table`），让 zigzag 能把重复过的 block table 传进去 |

**差在哪**：Q/KV 的**代码**几乎没动——因为输入本来就是 rank-local 序，投影自然就在这个序上算。真正改的是两处：

1. **写回 KV cache 的 slot 顺序**：数据在 `[prev, next]` 序，但 KV cache 的物理位置由全局 token 位置决定，所以 slot 映射要按 `zigzag_gather_index` 重排；
2. **合并调用**：prev/next 在同一个 TND 张量里前后相接，用「2B 个 batch」的描述就能一次算完，省一次 kernel 启动与一套 metadata 组装。**不是**省掉 Q 交换——base 的 `_record_query_gather_context`（`sfa_v1.py(base):1707-1713`）是空实现 `return`，本配置本来就没有每层的 Q 收集（Q 是 rank-local 的，KV 靠 all_gather 全量共享）。

**review 重点**：
1. `slot_mapping_cp_gathered` 必须**保留 padding 行的 -1**，scatter 靠它跳过 padding；
2. 合并 metadata 要求 block table 的行序是 `[所有 prev, 所有 next]`；改 `:651` 就要同步改 `actual_seq_lengths_*_zigzag` 的语义；
3. `:2552` 关掉快速写是**有性能代价**的已知取舍。

## 7. 第六站：逐层 —— o_proj 与 MLP / MoE

**第一步先搞清「哪个算子类落在哪个层上」**，因为归约的门控只盖住了其中一部分。选择逻辑在 `ops/linear_op.py:540-565`：

| 条件 | 选中的类 | 本文关注点 |
| --- | --- | --- |
| `down_proj` 且 `mlp_tp_enable()` 且非 MoE 层 | `MLPRowParallelOp` | 归约**被门控** |
| `o_proj` 且 `oproj_tp_enable()` | `OProjRowParallelOp` | 归约**未门控**（见下） |
| 其余 `o_proj` / `down_proj` / `wo_b` 等且 `enable_sp()` | `SequenceRowParallelOp` | 归约**被门控** |
| `shared_expert` 的 down_proj | 返回 `None`（走普通 TP） | — |

`oproj_tp_enable()` / `mlp_tp_enable()`（`utils.py:845` / `utils.py:853`）只有在 fine-grained TP 配了非零尺寸时才为真；**本配置没配，所以 down_proj（以及非 prefill 的 o_proj）都落到 `SequenceRowParallelOp`**。

注意与归约**配套的还有每层一次 all_gather**：`SequenceColumnParallelOp.apply_impl` 在 GEMM 前调 `maybe_all_gather_and_maybe_unpad`（`ops/linear_op.py:315`，base 对应 `ops/linear_op.py(base):301`）把 L 行拼回 T 行，GEMM 之后再由 `SequenceRowParallelOp` 的 reduce_scatter 切回 L 行。**这一对 all_gather/reduce_scatter cp_balance 都没改**（只换了 reduce_scatter 的实现），对账时要把它算进去。

**o_proj（未改）**：DSA-CP prefill 的 o_proj 实际在 SFA 内部完成（`sfa_v1.py:2217` 起：异步 all_gather 全量权重 → 本地全量 GEMM），base 与当前一致。原因是 DSA-CP prefill 的 attention 输出不是 TP 分片的，用 TP 分片权重会算错。

**MLP / MoE（改了）**：

| 位置 | base | 当前 |
| --- | --- | --- |
| `ops/linear_op.py:203` `MLPRowParallelOp.forward` | `linear_op.py(base):200` 直接 `reduce_scatter` | `zigzag_active()` 时换 `fixed_order_reduce_scatter`，否则原样 |
| `ops/linear_op.py:463` `SequenceRowParallelOp.matmul_and_reduce` | `linear_op.py(base):413` `tensor_model_parallel_reduce_scatter` | 同上，经 `linear_op.py:351` 的 `_fixed_order_reduce_scatter` |
| `ops/register_custom_ops.py:36` `_maybe_pad_and_reduce_impl` | `register_custom_ops.py(base):90` 直接 RS | 同上（embedding 与 MoE finalize 共用） |
| MoE 的 `input_ids` | 自然序 | 用 `set_ascend_forward_context` 重排好的序（`ops/fused_moe/experts_selector.py` 只加了注释） |
| 归约实现 | — | 新增 `distributed/utils.py:95` `fixed_order_reduce_scatter`：`allreduce`（默认，就地）/ `alltoall` / `reducescatter` 三选一 |

**为什么**（不变量 2）：HCCL 的 `ReduceScatter` 累加顺序与「哪个 rank 拿到这块 chunk」有关，而 zigzag 故意把 token 行在 rank 之间搬。改成与 owner 无关的求和顺序后，B/C 的首 token 才一致。

**实测对账**（`data/send` 那轮）：三个门控点里，本配置真正会命中的只有「3 个 dense 层的 down_proj + embedding」共 **4 处/步**。四轮对比里 `reduce_scatter` 的调用数正好差 16 = 4 处 × 4 个 prefill 步——代码读出来的结论和实测数字吻合。

**review 重点**：
1. `OProjRowParallelOp`（`linear_op.py:253-295`，内含 `reduce_scatter`）**没有被门控**。本配置用不到它，但只要有人打开 fine-grained TP，它就是 B/C 逐位一致性的潜在漏洞；
2. `_allreduce_slice_reduce_scatter`（`distributed/utils.py:30`）是**就地** all_reduce，前提是三处调用点传入的都是本层新产生的激活、归约后不再使用——新增调用点必须复核这个前提；
3. `reduce_mode()` 用 `lru_cache` 只读一次环境变量。

## 8. 第七站：出口 —— gather 与还原

**base**：代码里有**两处**「形状不等就拼」，两处都写在那里：

- `patch_deepseek_v2.py(base):348-355`：模型内部，形状不等就 `cat + all_gather + 截断到 positions.shape[0]`；
- `worker/model_runner_v1.py(base):2601`：runner 里 `flash_comm_v1_enabled` 时调 `_all_gather_hidden_states_and_aux`（`model_runner_v1.py(base):2541`）。

**开放问题（未证实）**：base 传给模型的 `positions` 是全长 T 行，而 FlashComm 下 hidden 是 L 行，按形状推导**两处都会成立**；若真的都执行，每步会多做一次 T×hidden 的 all_gather（结果仍对，因为 gather 后各 rank 内容相同，`logits_indices` 落在第一份拷贝里）。我们没有实测确认哪一处真正生效。判定办法：按步统计 `kernel_details.csv` 里 all_gather 类算子的条数，或在两处各加一行日志。

**当前**：

- `patch_deepseek_v2.py:410`：zigzag 下**跳过**这里的拼接（因为 positions 被切成 L 行，形状相等，判断自然为假）；
- `worker/model_runner_v1.py:2605-2608`：`zigzag_cp_active` 时改用 `zigzag_gather_hidden_states_and_aux`，即一次 all_gather + `inv_gather_index` 还原自然序（aux 列表同样处理）。

**差在哪**：连续切片的 rank 拼接结果**就是**自然序；zigzag 的 rank-local 序 `[r0_prev, r0_next, r1_prev, ...]` 不是，必须再乘一次逆置换。

**为什么关键**：logits 只取每个请求的最后一个 token。顺序错了输出就错了，而且不会报错——这是最容易漏测的一站。

**review 重点**：zigzag 路径的出口还原**只有 runner 这一处**（模型内那一处靠 `positions` 被切成 L 行、形状相等而跳过）。base 在 runner 里还会按 `_EXTRA_CTX.pad_size` 截断；zigzag 版本由 `inv_gather_index` 的长度天然保证，不要再截一次。

## 9. 之后：norm / logits / sample（未改）

`patch_deepseek_v2.py:421` 的 `self.norm`、以及 runner 的 logits 与采样路径与 base 完全一致。也就是说：**出口 gather 之后（norm / logits / sample）没有任何 cp_balance 代码**；但这**不代表** zigzag 的影响只止于出口——它还改了第三站（forward context 的输入重排）与第六站（归约实现）。

## 10. 一次 forward 的差异对账表

| 站 | 新增集合通信 | 减少的集合通信 | 变化的数据布局 |
| --- | --- | --- | --- |
| 2 metadata | — | — | 局部行的语义从「区间」变成「置换」 |
| 3 context | — | — | `input_ids` / `mc2_mask` 重排一次 |
| 4 embedding | 1 次全长 all_gather | — | hidden 与 positions 都切成 L 行 |
| 5 SFA | — | 合并调用省一次 kernel 启动与一套 metadata（不是省 Q 交换） | KV 写回从「自然序前 `num_actual_tokens` 行」变成「全 padded 行 + 重排过的 slot」 |
| 6 MLP/MoE | 归约字节数上升（默认 allreduce 约为 RS 的 2 倍） | — | MoE 消费重排后的 `input_ids` |
| 6 每层 all_gather | — | — | 无变化（`SequenceColumnParallelOp` 的 gather 未改） |
| 7 出口 | 1 次 all_gather（base 也有） | — | 多一次逆置换 |

## 11. 三条硬不变量

| # | 不变量 | 谁保证 | 破坏了会怎样 |
| --- | --- | --- | --- |
| 1 | 每 rank 局部行数严格等于 `num_tokens_pad / cp_size` | 第 1 站的 padding + 第 2 站的 `pad%cp_size` gate + `build_zigzag_plan` 的长度断言 | 各 rank 集合通信形状不一致，挂死或结果错乱 |
| 2 | row-parallel 归约的舍入与 token owner 无关 | 第 6 站的 `fixed_order_reduce_scatter` | B/C 首 token 不一致 |
| 3 | 非 zigzag 的 forward 逐位走 base | `zigzag_active()` 门控三处归约 + 第 7 站的出口分支 | `CP_BALANCE=0` 时与 base 不等价 |

任何改动只要碰了其中一条，就必须重做等价性验收。

## 12. 关键耦合点

| 改这里 | 必须同步看 |
| --- | --- |
| `zigzag_ineligible_reason` 的任何 gate | 第 1 站的 padding、第 2 站的断言（不变量 1） |
| `build_zigzag_plan` 的块分布 | 三个索引的长度与拼接序、KV 的 `block_table_zigzag` 行序 |
| `zigzag_gather_index` 的拼接顺序 | MoE 的 `input_ids`、`mc2_mask`、出口的 `inv_gather_index` —— **四处必须同源** |
| `DSACPContext` 字段 | 构造点（只有 `sfa_v1.py:773`）、回退函数、SFA 消费点 |
| 任何 row-parallel 归约 | `zigzag_active()` 门控 + 不变量 2 |
| 第 4 站的 positions 切片 | 第 7 站的形状推断 |

## 13. review 检查清单（都可执行）

| 检查 | 命令 | 判据 |
| --- | --- | --- |
| 三方容器字段对得上 | `python3 cp_balance/check_cp_balance_fields.py --repo <vllm-ascend>` | 末行 `RESULT: PASS` |
| cp_balance 只作用于 zigzag | `python3 cp_balance/check_b_path.py --repo <vllm-ascend> --base-repo <base>` | 末行 `RESULT: PASS` |
| 非 zigzag 逐位等价 base | `bash cp_balance/run_matrix.sh configs/matrix_b_vs_base.json` | 末行 `RESULT: PASS`（含噪声地板） |
| 真的进了 zigzag | `python3 cp_balance/check_branch.py --url ... --log ...` | 短 prompt 只有 `CONTINUOUS`，长 prompt 至少一条 `ZIGZAG` |
| 语法与未定义名 | `python3 -m pyflakes <改过的文件>` | 无新告警 |

## 14. 已知边界与坑

1. `VLLM_ASCEND_CP_BALANCE` **默认是 1**，升级上来默认就开；要复现 base 必须显式设 0。
2. `MIN_TOKENS` 默认 8192，但实测配置用 2048，文档以配置为准。
3. 第 6 站的 `OProjRowParallelOp` 归约未被门控（fine-grained TP 打开时会踩）。
4. 第 4 站的 `EMBED_LOCAL` 是未验证的实验路径，默认关闭。
5. 第 3 站的回退路径是防御性代码，正常配置不会走到；缺 `fallback_*` 时直接抛错而非静默降级（有意）。
6. diff 里 `AGENTS.md` / `CLAUDE.md` 的 -420 行与功能无关，review 先剔掉。
7. `q_up` 融合算子在 diff 里看不到差异：这条分支曾经改坏过，由 `9e462094b` 恢复成与 base 逐字节相同，`check_b_path.py` 的 byte-identical 断言就是钉这个。
8. 改动面：18 文件 / +1853 / -511，剥掉仓库文档（-420）与新增 UT（+46）后，**真正的 cp_balance 代码约 +1807 / -91**。

