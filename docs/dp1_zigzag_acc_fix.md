# dp=1 下 DSA-CP 走的分支 + zigzag 精度修复（2026-09-23）

适用部署：A5（8 卡）、GLM-5.2（SFA + indexer）、TP8 / dp=1、`additional_config.enable_dsa_cp=true`。

## 0. 结论

- **dp=1 + `enable_dsa_cp=true` 时 DSA-CP 并没有被关掉**，但它只活在 SFA 层内部：模型主流始终是
  全量、自然序、各 rank 完全相同的张量；attention 在层内自己切出本 rank 的行，出口再 all_gather 回全量。
  集合通信上的 "CP 轴" 就是 TP 组（cp_size = TP = 8）。所谓"失效"= 它没有把序列状态变成 rank 私有：
  每层 KV/indexer 都 all_gather 成全量写回本 rank 的 cache（没有显存收益），o_proj 用全量权重 + 出口
  all_gather，层间不存在序列并行。真正按 KV 分片的 DCP 需要 `decode_context_parallel_size > 1`（本部署 = 1）。
- 这个"全量 replicated 主流"不是巧合，是上游契约：`ops/vocab_parallel_embedding.py::_forward_origin` 的注释写着
  "vLLM 0.26 model forwards expect the first decoder layer to receive the complete token sequence.
  Sequence parallelism starts only after that layer's attention output"，所以它在 TP 上做的是 all_reduce 而不是 reduce_scatter。
- **cp_balance（zigzag）原来把行布局搬到了模型边界**：`patch_deepseek_v2._patched_forward` 在进层循环前把
  hidden_states/positions 切成 rank-local 的 `[prev,next]` 块，出口才 all_gather 回自然序。而 dp=1 下
  `use_sequence_parallel_moe == False`，层内 MLP/MoE 的 TP/EP 集合通信是 **element-wise** 的（要求每个 rank
  持有完全相同的全量行）——第一次 all_reduce 就把**不同 token 的行**加在了一起。这就是 C 矩阵长 prompt 全错的根因。
- **修复**：把 zigzag 收回 attention 内部。模型主流（embedding / 模型边界 / dense MLP / MoE / shared experts /
  runner）恢复成与 `cp_balance=0` 逐位相同的 replicated 全量行；attention 用 `zigzag_index` 选本 rank 的行
  （替代连续切片），出口用 `zigzag_gather_tensor` 把 all_gather 结果排回自然序。zigzag 的收益（把 causal
  attention 的工作按 head/tail 块均分）保持不变，同时不再需要任何 owner-independent 归约。

## 1. 代码实际走的分支（task 1）

门在哪：

- `vllm_ascend/ascend_config.py:609-672`：`VLLM_ASCEND_ENABLE_FLASHCOMM1=1` 让 FlashComm "显式开"，
  但 `use_sequence_parallel_moe` 要求 `data_parallel_size > 1`（`vllm/config/parallel.py:711-726`），
  所以 dp=1 时只打一行 warning `FlashComm1 is enabled, but the current config does not support sp MoE. Disabling`
  （它并不关任何东西），接着打 `DSA-CP is enabled without sequence-parallel MoE (data_parallel_size=1)`；
- `ascend_config.py:672` 的 DSA-CP 前提只有 `has_indexer`（被测树与参照树都改了这道门，原来 AND 上
  `use_sequence_parallel_moe`），所以 dp=1 下 `enable_dsa_cp` 仍为 True。

选了哪个实现：

- `attention/context_parallel/sfa_cp.py` 的 `resolve_sfa_metadata_builder` / `resolve_sfa_impl`：
  `enable_sfa_dcp_replicated_indexer()` 为 False（要 `decode_context_parallel_size > 1`），于是选中
  **`AscendSFADSACPMetadataBuilder` + `AscendSFADSACPImpl`**（连续切片版）；DCP 的 `AscendSFADSADCP*` 不在链上。
- cp_size / cp_rank 来自 TP 组：`sfa_cp.py` 的 `_prepare_parallel_metadata` 用
  `get_tp_group().world_size / rank_in_group`，`indexer.py` 的 `self.dsa_cp_world_size = tp_size`；
  runner 把 token 数补齐到 TP 的倍数（`worker/model_runner_v1.py:3153-3161`）。

层内实际发生什么（每层一次）：

1. 层入口拿到**全量** hidden_states（embedding 是 all_reduce，不是 reduce_scatter：
   `ops/vocab_parallel_embedding.py::_forward_origin`）；
2. `AscendSFADSACPImpl._prepare_native_hidden_states`：pad 到 `num_tokens_pad`，再取本 rank 的
   L = num_tokens_pad / cp_size 行；
3. KV / indexer 的 k 用 `all_gather_async` 收成全量写回本 rank 的 cache（`sfa_cp.py` 的 KV 写回、
   `indexer.py:363-378`）；
4. Q/KV 全头本地算（`ShardedCPColumnParallelOp` 让 q_b_proj/kv_b_proj 不按 TP 切）；
5. o_proj 用 gather 来的**全量权重**（`enable_dsa_cp_full_o_proj()`，无 kv_transfer 时为 True，
   `utils.py:1524-1541`），`_finalize_o_proj` 在 `o_proj.reduce_results=True`（dp=1 就是）时
   `tp_group.all_gather` 把 L 行还原成全量行。

也就是说 dp=1 的 DSA-CP 是"层内切一刀又收回来"：数上等价于不开 DSA-CP，只是多付了 KV all_gather、
o_proj 权重 gather 和出口 all_gather。真正省显存的 DCP 需要 `decode_context_parallel_size>1`。

zigzag（cp_balance）的门只放行 dp=1：`layers/cp_zigzag.py:492` 的 `zigzag_ineligible_reason`
（`cp_size<=1 / draft / v2_model_runner / dp>1 / dcp_replicated / o_proj_not_full / 注意力状态 /
query_len<2*cp_size / actual<min_tokens(2048) / pad%cp_size!=0`）。少一样就回连续切片。
注意 `VLLM_USE_V2_MODEL_RUNNER=0` 是 load-bearing（V2 runner 不出 zigzag 元数据）。

## 2. 精度测试方法（task 2）

入口（A5 默认）：

```bash
bash verify.sh --family a5                                   # 全量：前置 -> 静态 -> 冒烟 -> 诊断 -> 打包
bash tests/run_tests.sh --only accuracy/a10_matrix_gate#b_matrix   # B 等价性（cur_cp0 vs base_cp0）+ 噪声地板
bash tests/run_tests.sh --only accuracy/a10_matrix_gate#c_matrix   # C 验收（cur_cp1 vs cur_cp0）
bash tests/run_tests.sh --only accuracy/a10_matrix_gate#nomtp_matrix   # C 的关 MTP 版本
bash tests/run_tests.sh --only accuracy/a20_compare_collected          # 复用已采的 json 重跑对比（不起服务）
```

矩阵怎么打分（`harness.json` + `accuracy/run_matrix.py` + `accuracy/compare_first_token.py`）：

- 每个矩阵起停服务、采 `questions.json` 的 40 条 prompt（前 20 条 short ≈ 120-170 字符、后 20 条 long ≈ 4.4k 字符），
  `/v1/completions`、`temperature=0.0`、`max_completion_tokens=50`，配置里 `deterministic=true` 会带 4 个确定性环境变量；
- 首 token：completion 去掉空白后为空 → `(None, "empty")`（就是"吐了一堆换行"），否则走 `/tokenize` 比对 token 串；
- compare：同 `prompt_sha256` + 同首 token 才算 OK；`require_text=true` 时还要前 120 字符相同；
- 矩阵 PASS = 静态门 + 端口空闲 + 服务就绪 + collect rc0 + 指纹匹配 + `require_log` 通过 + 所有 gate compare rc0；
- `require_log`（C 矩阵）：cp1 日志必须出现 `branch=ZIGZAG` 或 `[CP_BALANCE][plan]`，cp0 不许出现 `[CP_BALANCE][plan]`
  —— 这是防"C 空跑"的中间门。

不通过时看什么：`tests/_out/<stamp>_<family>/accuracy/<matrix id>/matrix/` 下的 `summary.txt`、各配置的
`.log/.collect.txt/.json`、`cmp_*.txt`；`a20_compare_collected` 可以直接在旧 json 上重跑 compare。

已知的判读陷阱：C 默认 `require_text=false`（只比首 token，可改成 true 收紧）；`gate:false` 的 compare 失败不算矩阵 FAIL；
driver.log 里回显的 `branch=ZIGZAG` 计数不是证据（要看服务日志）；短 prompt 不进 zigzag（`reason=actual<min(2048)`）是正常的。

## 3. 本轮远端结果（`log/acc_0922_222922`）

| id | 结果 | 说明 |
| --- | --- | --- |
| `accuracy/a10_matrix_gate#b_matrix` | PASS | `cur_cp0` vs `base_cp0` + 噪声地板都 40/40（连续切片路径成立） |
| `accuracy/a10_matrix_gate#c_matrix` | FAIL | cp1 有 zigzag 证据（`branch=ZIGZAG`/`[CP_BALANCE][plan]` 各 160 行），但 short 20/20、**long 0/20** |
| `accuracy/a10_matrix_gate#nomtp_matrix` | FAIL | 同一签名（关掉 MTP 后仍然 long 0/20 → 与 MTP 无关） |
| `accuracy/a20_compare_collected` | FAIL | 复用上面 json 重跑 compare，同一签名 |
| `accuracy/a30_slot_filter_ab` | FAIL | harness 自身问题（硬编码 `/opt/its/...` 路径），与本议题无关 |

失败签名：cp1 的 20 条 long 全部 DIFF，19 条 `A=None source=empty`、1 条 `A='\n'*10`；
collect rc=0、服务日志无 ERROR/Traceback（是 HTTP 200 + 50 个换行），
`SpecDecoding Accepted: 0 tokens, Drafted: 40` —— logits 已坏，greedy 只会吐换行。

## 4. 根因：行布局契约被破坏

1. dp=1 ⇒ `use_sequence_parallel_moe=False`（`vllm/config/parallel.py:711-726`）⇒ 层内 TP/EP 集合通信是
   element-wise 的，正确性前提是**每个 rank 持有完全相同的全量行**（cp0 就是：embedding all_reduce 成全量、
   attention 出口 all_gather 成全量，层间从不变形）。
2. 原来的 zigzag 让层循环里的张量变成 rank-local 的 1/8 行（`patch_deepseek_v2.py` 的 `zigzag_shard_tensor`，
   出口才 `zigzag_gather_hidden_states_and_aux`），层中间没有任何 gather（decoder layer 没被 patch）。
3. 于是每层至少 3 处 element-wise all_reduce 把不同 token 的行加起来（layer 0 就已经错）：
   dense MLP 的 `down_proj`（`vllm/model_executor/layers/linear.py`，dp=1 时 `mlp_tp_enable()=False` 不走 custom op）、
   MoE 出口（`ops/fused_moe/fused_moe.py::_maybe_reduce_final_output` → `register_custom_ops.py::_maybe_all_reduce_tensor_model_parallel_impl`）、
   每层共享专家（`register_custom_ops.py::maybe_all_reduce_shared_expert`）。
4. 为什么只有长 prompt 错：短 prompt 被 `reason=actual<min(2048)` 挡在 zigzag 外，走的是连续切片；
   长 prompt 进 zigzag，所以长 prompt 全错而 short 20/20。
5. 旁证：`summary.txt` 里 `path=fixed_order 0 / path=native 0` —— 为 zigzag 写的
   `MLPRowParallelOp`/`AscendSharedExperts` 归约适配只在 SP/fine-grained-mlp-tp 配置里安装，dp=1 从不触发。

## 5. 修复内容（vllm-ascend，被测树）

思路：**zigzag 只是 attention 内部"哪些行归我算"的问题**，不碰模型主流。

- `attention/context_parallel/sfa_cp.py::AscendSFADSACPImpl._prepare_native_hidden_states`：
  pad 到 `num_tokens_pad` 后，zigzag 用 `hidden_states[context.zigzag_index]` 选行（原来是"模型边界已经切好"直接返回），
  连续切片仍然取 `[local_start, local_end_with_pad)`。
- `attention/context_parallel/sfa_cp.py::AscendSFADSACPImpl._finalize_o_proj`：zigzag 分支改成
  `zigzag_gather_tensor(local_output, <本层 plan 的 inv_gather_index>, output.shape[0])` ——
  全量 o_proj 权重算完本 rank 的行后，一次 rank 拼接序 all_gather + 反排列，
  把 replicated 自然序主流写回 `output`（与连续切片分支同构）。
- **行布局的唯一来源是每层自己的 `DSACPContext`**（`sfa_cp.py` 的 `_get_parallel_forward_context` 把它塞进
  `SFAForwardContext.zigzag_inv_gather_index`，forward 再透传给 `_finalize_o_proj`）。
  为此删掉了全局的 `zigzag_active()` / `_EXTRA_CTX.zigzag_cp_active` / `zigzag_cp_context` 与
  `get_zigzag_cp_context()`：选行与放回都取自同一层的同一份 plan，不存在"元数据连续但全局开关说 zigzag"的组合。
  `set_ascend_forward_context` 只保留 forward 级否决（draft 实例 / V2 runner / dp>1），
  否决时按元数据里的连续切片 fallback 还原。
- `attention/sfa_v1.py`：`SFAForwardContext` 增加 `zigzag_inv_gather_index` 字段并在 `forward` 里透传；
  NATIVE 分支选行后把 `gate_hidden_states` 重新绑定到本 rank 的行（否则带 `g_proj` 的模型会拿全量行点乘本 rank 的
  attn_output）。
- `layers/cp_zigzag.py`：删掉模型边界专用原语（`zigzag_shard_tensor`、`zigzag_gather_hidden_states*`、
  `zigzag_reorder_moe_aux`、`fixed_order_rank_sum`），保留 `zigzag_gather_tensor` 与 plan/资格门；
  模块文档改成"模型主流 replicated、zigzag 只改 attention 工作窗口"。
- `patch/worker/patch_deepseek_v2.py`：`_patched_forward` 恢复成 base 版本（不再切 hidden_states/positions、
  不再在模型出口 gather），只留一段注释说明 zigzag 是 attention 内部的事。
- `ops/linear_op.py`、`ops/fused_moe/shared_experts.py`、`ops/vocab_parallel_embedding.py`、
  `tests/ut/ops/test_vocab_parallel_embedding.py`：整文件恢复到 base（删掉 SP-only 的 owner-independent 归约分支、
  `forward_zigzag_local` 实验入口）。
- `ascend_forward_context.py`：删掉 MoE `mc2_mask` 的 zigzag 重排（MoE 现在拿自然序全量行）与全局 zigzag 开关；
  只保留 forward 级否决（draft 实例、V2 runner、dp>1）+ 元数据回退。
- `distributed/utils.py`：删掉 `fixed_order_reduce_scatter` 三种实现与 `reduce_mode()`；
  `envs.py`：删掉 `VLLM_ASCEND_CP_BALANCE_REDUCE_MODE` 与 `..._EMBED_LOCAL`（都已无读者）。

为什么这样就对了：

- 模型主流（embedding、层间 residual、MLP、MoE）在所有 rank 上都是同一份全量行，与 cp0 完全相同 →
  element-wise 的 all_reduce 逐 token、逐位一致（不再需要 owner-independent 归约）；
- attention 只负责选行/放回：选行用的 `zigzag_index` 与 `slot_mapping_cp`、metadata cos/sin、KV/indexer 写回
  用的是同一套索引；放回用 `inv_gather_index`（plan 里 `gather_positions` 的逆），行数仍满足
  `num_tokens_pad % cp_size == 0`；
- 收益不变：每个 rank 仍然只算自己的 head/tail 块（causal attention 负载均衡），
  只是不再假设"层间的序列状态被切开了"。
- 代价：与 cp0 一样，MoE/MLP 每 rank 仍然算全量行（replicated），zigzag 不会减少 MoE 的计算量；
  这是 dp=1（没有 SP、没有 MoE dispatch）下的唯一正确做法。

harness 侧同步：删掉 `reduce_mode` 字段/环境变量/指纹字段与 3 个 `prof_*_cp1_a2a.json` 配置
（`tests/lib/roles.tsv` 里 `prof_a2a` 改成 `-` → 该 variant 自动 SKIP）；
`accuracy/check_b_path.py` 的静态门改成按新架构断言（模型主流 zigzag-free、模型级 shard/固定序归约原语不得残留、
选行与放回都在 DSA-CP attention 内）；`planlog.py`/`run_matrix.py` 删掉已不存在的 reduce 行统计。

## 6. 远端怎么跑（分两阶段，保证一次只动一个变量）

被测树的历史按“是否可能影响 C 的数字”切成三段，两个 ref 各自承担一次远端跑：

| 阶段 | ref / commit | 改了什么 | 会不会影响 C 的数字 | 远端要跑 |
| --- | --- | --- | --- | --- |
| S1 唯一变量 | `cp_balance` = `d06883a99` | 只改行布局：模型边界恢复 upstream + attention 内选行/放回（含 per-layer plan 透传）+ 纯测试/文档 | **会** | C 矩阵（阶段 1） |
| S2 清理 | `cp_balance_stage2` 里的 `2a14d5c91` | 删本部署不可达的失效路径（fixed_order 归约 + `reduce_mode`/`EMBED_LOCAL`、`mc2_mask` 重排、全局 `zigzag_active()`、`forward_zigzag_local`、边界 helper） | 不会 | 不单独跑 |
| S3 latent | `cp_balance_stage2` 的 tip `f1bb3c915` | `g_proj` 的 gate 行绑定（GLM-5.2 无 `g_proj`，不可达） | 不会 | 不单独跑 |
| S4 出口搬运融合 | `cp_balance_perf` 的 tip `c3b7d2d8b`（= S1+S2+S3 + 1 个提交） | 把出口反排列融进那一次写入（`index_select(..., out=)`，带 fallback） | 不会（写入的是同一批值） | 可与阶段 2 一起跑；若阶段 2 结果与阶段 1 不同，先回退这一个提交 |

S2/S3“不会影响”的依据：2026-09-22 那轮 cp1 日志里 `[CP_BALANCE][reduce]` 0 行、mc2 相关 0 行、
`sequence_parallel_moe`/`mlp_tp`/`g_proj` 0 行；S2 删的代码只在 mlp-tp / SP-only / MC2 / g_proj
这些配置下才会被安装或触发，本部署一个都不满足。

```bash
# 阶段 1（归因）：树上只有“行布局”这一个变量
cd <harness> && git pull                     # 被测树会被 reset 到 origin/cp_balance = S1
bash tests/run_tests.sh --only accuracy/a10_matrix_gate#c_matrix
#   PASS → 长 prompt 全错就是“模型级行布局”造成的（另一半改动不在树里）
#   FAIL → 行布局不是根因，回 docs/scripts_review.md §8 重新取证
# 注意：阶段 1 只跑这一条。不要跑 verify.sh / --tag fast：check_b_path 的 3 条“清理”断言在 S1 按设计不成立
#       （S2 删掉那些代码后才成立）；需要冒烟/诊断就用 bash verify.sh --skip-fast
#       （s06 的字段门与 C 矩阵自身都不受影响）

# 阶段 2（验收 + 证明清理无害）：树切到含全部改动的 tip
#   默认由我 fast-forward cp_balance → cp_balance_stage2（线性历史，S1 是它的祖先）；
#   也可以自取：把 harness.json 的 trees.cur.ref 改成 cp_balance_stage2（kind 保持 tip）
bash tests/run_tests.sh --only smoke/s06_static_fields,smoke/s07_static_b_path   # 两个静态门应 PASS
bash tests/run_tests.sh --only accuracy/a10_matrix_gate#c_matrix
bash tests/run_tests.sh --only accuracy/a10_matrix_gate#b_matrix                 # 顺带确认 B 没坏
```

阶段 2 的 C 结果应与阶段 1 **逐条一致**（并核对两次日志里的 `[CP_BALANCE][plan]`/`[branch]` 行与
配置指纹相同）：一致就同时证明了“清理没有引入变化”；若不一致，回归只可能来自 S2/S3 两段各一个提交。

期望：

- `[CP_BALANCE][branch] ... branch=ZIGZAG` 与 `[CP_BALANCE][plan] ... local=<pad/8>` 照旧出现（说明 C 没空跑），
  但**不再**出现 `[CP_BALANCE][reduce] path=fixed_order`（S1 里这条路径已不可达，S2 起直接删除）；
- `c_vs_b` 的 `first-token match: 40/40`（long 20/20）；
- `max_completion_tokens` 在 pinned vLLM 上可能被忽略，脚本已有 `max_tokens` 兜底。

## 6b. 出口搬运优化（S4，分支 `cp_balance_perf`）

zigzag 的 o_proj 出口原本是三步：all_gather → `gathered[inv]` 生成新张量 → 拷进 `output`。
其中后两步只是把同一批值按自然序搬一遍，属于“为了把行序摆回自然序”付的搬运费。S4 把反排列
融进写入本身（helper 用 `index_select(gathered, 0, inv[:N], out=output)`；拿不到 out 变体时回退
gather+copy 并在日志里 warning 一次），于是：

| 每层出口（T=2784, H=6144, bf16） | 搬运量 | 算子数 |
| --- | --- | --- |
| S1（gather + copy） | all_gather 34MB + 反排列 68MB + 拷贝 68MB ≈ **171MB** | all_gather + index_select + copy |
| S4（融进写入） | all_gather 34MB + 直写 68MB ≈ **103MB** | all_gather + index_select(out=) |
| `cp_balance=0`（连续切片） | all_gather 34MB + 拷贝 68MB ≈ **103MB** | 同 S4 |

即 S4 之后**每层出口与 cp0 完全等价**（少 68MB/层、少 1 个算子/层；78 层约 −5.3GB/forward）。
语义不变：写进 `output` 的仍是同一批值，只是排列方式换了实现（新增
`tests/ut/layers/test_cp_zigzag_exit_write.py` 用真 `build_zigzag_plan` 的排列核对“融合写入 ==
gather+copy”、尾部丢 padding、行数不匹配报错；本地已跑过）。

**仍然没消掉的**：attention 入口那次选行 gather（本 rank 的 `[prev,next]` 行在自然序里是两段，
不是连续区间）。它的量级是 ≈8.6MB/层（读 4.3 + 写 4.3），78 层 ≈ 0.7GB/forward，属于百分点级；
要清零只能把模型主流改成“拼接序”（每层入口退化成连续切片、出口连反排列都不需要），
但那要同时改 embedding 输入、runner/MTP 的取行与 aux 路径，是独立一轮验证的事。

## 7. 已知的残留风险与离线验证

- **离线已验**（用真 `build_zigzag_plan` 跑随机与部署同形状的配置，本仓已固化成
  `vllm-ascend/tests/ut/layers/test_cp_zigzag_plan.py`）：
  每 rank 恰好拿到 `num_tokens_pad / cp_size` 行、各 rank 的行拼起来正好覆盖 padded 流且不重不漏、
  `all_gather(本 rank 行)[inv_gather_index]` 就是自然序。
- pad 行（`num_tokens_pad - num_actual_tokens`，本部署是 TP 对齐的 ≤7 行）只落在最后一个请求的块里；
  单请求时它们就是本 rank 行序的末尾。多请求 + 大 pad 的形状（例如 `[256, 2]` + 大 pad）里，
  某个 rank 的行序中确实可能出现"真实行排在 pad 行之后"，但这不影响正确性：
  每个 query 行的因果窗口恰好到它自己的自然位置（`kv_len_batch - n_batch + 1 + i`），
  而 pad 位置在所有真实 token 之后，所以**没有真实行的窗口会包含 pad 行**，
  多算的 KV 长度只作用在 pad 行自己的、会被丢掉的输出上。
- **latent**：`_disable_zigzag_metadata_for_fallback` 只还原带 `dsa_cp_context` 的元数据；
  indexer 的元数据是它自己建的（没有该字段），一旦真的触发回退会出现 SFA 连续 + indexer 排列的单侧布局。
  当前三个触发条件（draft 实例 / V2 runner / dp>1）在资格门里都会被拒，所以这条路径在本部署不可达；
  真要长期共存，得给 indexer 元数据也加一份 fallback 或改成硬报错。
- vllm-ascend 侧单测：`tests/ut/attention/test_sfa_*` 跟着 `_finalize_o_proj` 的新签名更新（参数透传 + 断言），
  新增的 `tests/ut/layers/test_cp_zigzag_plan.py` 只依赖 `build_zigzag_plan`（纯 Python），
  其断言已在本地用真函数跑过（含 `cp ∈ {2,4,8,16}` 的 2400 组随机形状）；带 torch_npu 的用例本地跑不了。

## 8. 还没在远端确认的点

- 修复后的 cp1 vs cp0 是否**逐位**一致：本设计上模型主流的集合通信与 cp0 完全同构，理论上逐位；
  但 SFA/LI kernel 的行顺序不同、KV 在 cache 里的写者不同，理论上可能带来极小差异 ——
  若要收紧，把 `configs/matrix_a5_c_accept.json` 的 `require_text` 改成 true 再跑一次 C。
- `moe_comm_type` 只有在 `--log-level debug` 才打印（`ascend_forward_context.py` 的 selection 日志是 debug 级），
  本轮没有该证据；按 A5 策略（`CAPACITY_AND_WORLD_SIZE`、`world_size 8 <= top_k 8`）推断是 ALLGATHER，
  而 ALLGATHER 在 dp=1 下 prepare/finalize 都是恒等，正确性完全依赖最后那次 all_reduce —— 与本文结论一致。
- `a30_slot_filter_ab` 目前的 FAIL 是 harness 自身问题（`accuracy/round2_verify.py` 硬编码 `/opt/its/...`），
  与本议题无关，本轮忽略。
