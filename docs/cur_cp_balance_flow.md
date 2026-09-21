> 历史文档（2026-09 上旬的交接/评审期记录）：结论可能已过时，当前状态以 `docs/scripts_review.md` 为准。
>
> 已变事实（2026-09-21 更正）：
> - `VLLM_ASCEND_CP_BALANCE_REDUCE_MODE` “没有读者”不成立：新线 `distributed/utils.py` 走 `ascend_envs`。
> - 资格门现在是 `attention/context_parallel/zigzag_cp.py::zigzag_gate_reason`（没有 `tokens!=pad` 死门），
>   且 `speculative = draft_index is not None or for_draft`（99d03e477）；照本文去 `sfa_v1.py` 找会看不到 draft 修复。
> - `can_enable_zigzag_for_batch` 在当前线已不存在（6.1-1 已完成）。
> - §6 的“可删/可复用/可合并”清单只对 `cp_balance_v0.26.0rc` 有效，勿在本线执行（6.1-1、6.2-4 已完成）。

# 当前分支 cp_balance（model-level zigzag CP）流程与代码路径

审阅对象：`/d/code/cp_balance/vllm-ascend`，分支 `cp_balance`，HEAD `9e462094b`，
对比 base `c7990e5e4`。只做静态阅读，未运行模型、未做 git 写操作。
路径一律相对 `D:\code\cp_balance`（即 `vllm-ascend/...`）。

---

## 1. 开关与资格判定

### 1.1 环境变量定义

| 变量 | 定义位置 | 语义 |
| --- | --- | --- |
| `VLLM_ASCEND_CP_BALANCE` | `vllm-ascend/vllm_ascend/envs.py:85` | 总开关，默认 1 |
| `VLLM_ASCEND_CP_BALANCE_MIN_TOKENS` | `vllm-ascend/vllm_ascend/envs.py:90` | 实际 token 数下限，默认 8192 |
| `VLLM_ASCEND_CP_BALANCE_REDUCE_MODE` | `vllm-ascend/vllm_ascend/envs.py:96-98` | allreduce/alltoall/reducescatter，默认 allreduce |
| `VLLM_ASCEND_CP_BALANCE_DEBUG` | `vllm-ascend/vllm_ascend/envs.py:102-104` | 分支与 plan 日志，默认 0 |
| `VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL` | `vllm-ascend/vllm_ascend/envs.py:109-111` | 实验性 embedding 本地入口，默认 0 |

`envs.py` 用 `__getattr__` 惰性求值（`vllm-ascend/vllm_ascend/envs.py:140-144`），
因此每次属性访问都会重新 `os.getenv` + `int()`（见 7.5）。

`VLLM_ASCEND_CP_BALANCE_REDUCE_MODE` 在 envs 里定义但没有任何读者：
`distributed/utils.py` 自己用 `os.getenv` 再解析一遍（`vllm-ascend/vllm_ascend/distributed/utils.py:119`）。

### 1.2 zigzag_active()：定义与作用域

- 定义：`vllm-ascend/vllm_ascend/ascend_forward_context.py:616-628`。
  读 `_EXTRA_CTX.zigzag_cp_active`（同一文件的代理对象，`vllm-ascend/vllm_ascend/ascend_forward_context.py:552-613`），
  任何异常（无 forward context / 非本模块属性）都返回 False。
- 写入：`vllm-ascend/vllm_ascend/ascend_forward_context.py:338-343`（V2 runner 写 `additional_kwargs`，
  否则写 forward context 属性）。
- 值来源：`vllm-ascend/vllm_ascend/ascend_forward_context.py:230-244`，
  从 `attn_metadata` 里找第一个 `dsa_cp_context.zigzag_index is not None` 的层元数据
  （`vllm-ascend/vllm_ascend/ascend_forward_context.py:89-102`），再叠加三个否决条件：
  `is_draft_model`、`VLLM_USE_V2_MODEL_RUNNER`（`.../ascend_forward_context.py:231-235`）、
  `get_dp_group().world_size > 1`（`.../ascend_forward_context.py:317-321`）。
  后两者触发时会调用 `_disable_zigzag_metadata_for_fallback` 把 cos/sin、slot_mapping
  和整个 zigzag 字段恢复成连续切片态（`.../ascend_forward_context.py:105-163`）。
- 作用域：每个 forward 动态判定，`zigzag_active()` 是**所有**非 metadata 侧 cp_balance 代码的
  唯一门（`vllm-ascend/vllm_ascend/ops/linear_op.py:203`、`.../ops/linear_op.py:463`、
  `vllm-ascend/vllm_ascend/ops/register_custom_ops.py:36`）。
  metadata/attention 内部则用 `dsa_cp_context.zigzag_index is not None`
  （`vllm-ascend/vllm_ascend/attention/sfa_v1.py:2358-2361`、`.../attention/sfa_v1.py:2601-2606`、
  `.../attention/sfa_v1.py:2682-2685`）。

### 1.3 can_enable_zigzag_for_batch 的门（严格按代码顺序）

入口 `vllm-ascend/vllm_ascend/layers/cp_zigzag.py:668-710`，实际逻辑全在
`zigzag_ineligible_reason`（`vllm-ascend/vllm_ascend/layers/cp_zigzag.py:589-665`），
返回 None 表示可走 zigzag，否则返回第一个不满足的门名：

1. `not VLLM_ASCEND_CP_BALANCE` → `flag_off`，`.../layers/cp_zigzag.py:610-611`
2. `cp_size <= 1` → `cp_size<=1`，`.../layers/cp_zigzag.py:612-613`
3. `speculative` → `draft`，`.../layers/cp_zigzag.py:614-615`
4. `v2_model_runner` → `v2_model_runner`，`.../layers/cp_zigzag.py:616-617`
5. `dp_size > 1` → `dp>1`，`.../layers/cp_zigzag.py:618-619`
6. `dcp_replicated` → `dcp_replicated`，`.../layers/cp_zigzag.py:620-621`
7. `not full_o_proj` → `o_proj_not_full`，`.../layers/cp_zigzag.py:622-623`
8. `attn_state.name` 不属于 `{PrefillNoCache, PrefillCacheHit, ChunkedPrefill}` → `state=<name>`，
   `.../layers/cp_zigzag.py:624-626`，状态集合定义在 `.../layers/cp_zigzag.py:36-40`
9. `query_lens is None` → `no_query_lens`，`.../layers/cp_zigzag.py:627-628`
10. 空 batch → `empty_batch`，`.../layers/cp_zigzag.py:630-631`
11. `prefix_lens` 长度与 `query_lens` 不一致 → `prefix_len_mismatch`，`.../layers/cp_zigzag.py:632-636`
12. 任一 `prefix < 0` → `negative_prefix`，`.../layers/cp_zigzag.py:637-638`
13. 任一 `query_len < 2 * cp_size` → `query_len<2*cp_size`，`.../layers/cp_zigzag.py:639-640`
14. `num_actual_tokens > num_tokens_pad` → `actual>pad`，`.../layers/cp_zigzag.py:641-644`
15. `num_actual_tokens < VLLM_ASCEND_CP_BALANCE_MIN_TOKENS` → `actual<min(...)`，`.../layers/cp_zigzag.py:645-646`
16. `num_tokens_pad % cp_size != 0` → `pad%cp_size!=0`，`.../layers/cp_zigzag.py:651-652`
17. `is_prefilling is None`：只有 `PrefillNoCache` 允许放行，否则
    `is_prefilling_missing(<state>)`，`.../layers/cp_zigzag.py:653-659`
18. `len(is_prefilling) != len(query_lens)` → `is_prefilling_len_mismatch`，`.../layers/cp_zigzag.py:661-662`
19. `not all(is_prefilling)` → `not_all_prefilling`，`.../layers/cp_zigzag.py:663-664`
20. 全部通过 → `.../layers/cp_zigzag.py:665`

metadata 侧多一层包装 `_zigzag_gate_reason`（`vllm-ascend/vllm_ascend/attention/sfa_v1.py:276-316`），
在调用上面的谓词之前先判 `num_tokens != num_tokens_pad` → `tokens!=pad`
（`vllm-ascend/vllm_ascend/attention/sfa_v1.py:301-302`）。
调用点传入的 `num_tokens` 就是 `num_input_tokens`，而 `num_tokens_pad = _round_up(num_tokens, tp)`
（`vllm-ascend/vllm_ascend/attention/sfa_v1.py:664-666`），该门在当前调用点永远不成立。
metadata 侧实际传入的门参数：`speculative=draft_index is not None`、`v2_model_runner=envs_vllm.VLLM_USE_V2_MODEL_RUNNER`、
`dp_size=parallel_config.data_parallel_size`、`dcp_replicated=enable_sfa_dcp_replicated_indexer()`、
`full_o_proj=dsa_cp_with_o_proj_tp_for_config()`
（`vllm-ascend/vllm_ascend/attention/sfa_v1.py:687-706`）。

model runner 侧**没有**用这个谓词，只是把 token 数对齐到 `tp_size`
（`vllm-ascend/vllm_ascend/worker/model_runner_v1.py:2613-2629`），
`num_scheduled_tokens_np` 形参未被使用（`.../worker/model_runner_v1.py:2616`）。

---

## 2. zigzag plan 数据结构

### 2.1 输入输出

- 输入：`query_lens`（每请求本次调度 token 数）、`prefix_lens`（radix cache / 已算前缀）、
  `cp_size`、`cp_rank`、`num_tokens_pad`、可选 `num_actual_tokens`
  （`vllm-ascend/vllm_ascend/layers/cp_zigzag.py:413-420`）。
- 输出：冻结 dataclass `ZigzagPlan`（`vllm-ascend/vllm_ascend/layers/cp_zigzag.py:104-145`）。
- 设备侧封装：`_build_zigzag_meta`（`vllm-ascend/vllm_ascend/attention/sfa_v1.py:319-427`）
  把 plan 转成一组 device 张量放进 `DSACPContext`（`vllm-ascend/vllm_ascend/attention/sfa_v1.py:220-273`）。

### 2.2 每条序列怎么切 2*CP 块、rank r 拿哪两段、remainder 怎么分

- 记 T = num_tokens_pad，L = T / cp_size，m = 2 * cp_size（`vllm-ascend/vllm_ascend/layers/cp_zigzag.py:362-363,439`）。
- 全局 SP padding（T - sum(query_lens)）追加到最后一条请求上，等价于 SGLang 的 pad_len
  （`vllm-ascend/vllm_ascend/layers/cp_zigzag.py:365-375`）。
- 每条序列切成 m 个连续块，长度 `base = len // m` 或 `base + 1`
  （`vllm-ascend/vllm_ascend/layers/cp_zigzag.py:377-378,381-396`）。
- rank r 拿第 r 块（prev/head）与第 m-1-r 块（next/tail）；块内保持自然顺序
  （`vllm-ascend/vllm_ascend/layers/cp_zigzag.py:487-492`）。
- rank 局部张量顺序 = 所有序列的 prev 块（按序列号 0..B-1），再接所有序列的 next 块
  （`vllm-ascend/vllm_ascend/layers/cp_zigzag.py:483-492`）。
- remainder 分配：每条序列的 `remainder = len % m` 个“多一个 token”的块按对分配，
  每对最多 2 个（head 一个、tail 一个），全 batch 的多余块总数必须整除 cp_size，
  且每个 rank 恰好分到 `sum(remainder)/cp_size` 个，从而 `rank_tokens == L`
  （`vllm-ascend/vllm_ascend/layers/cp_zigzag.py:245-314`、分配落到 block：
  `.../layers/cp_zigzag.py:384-396`、等行校验 `.../layers/cp_zigzag.py:398-408`）。
  实现用一个 CPU 端 Dinic 最大流（`.../layers/cp_zigzag.py:148-209`）先算聚合容量，
  再用 `_decompose_remainder_rows` 拆回每行（`.../layers/cp_zigzag.py:212-242`），
  失败则退回逐行最大流（`.../layers/cp_zigzag.py:290-291,307-308,317-350`）。

### 2.3 三个 index 的定义与形状

| 名称 | 定义 | 形状 / 类型 |
| --- | --- | --- |
| `zigzag_index` | 本 rank 在自然流中的全局 token 位置，顺序 [prev..., next...] | tuple[int]，长度 L；device int64 [L]（`vllm-ascend/vllm_ascend/layers/cp_zigzag.py:485-497`，`.../attention/sfa_v1.py:402`） |
| `zigzag_gather_index` | 按 rank 拼接后的全量顺序 `[r0_prev, r0_next, r1_prev, ...]`，元素是自然流位置 | tuple[int]，长度 T；device int64 [T]（`.../layers/cp_zigzag.py:499-512`，`.../attention/sfa_v1.py:403`） |
| `inv_gather_index` | `inv_gather_index[p]` = 自然位置 p 在拼接张量中的行号（即 gather_index 的逆） | tuple[int]，长度 T；device int64 [T]（`.../layers/cp_zigzag.py:514-516`，`.../attention/sfa_v1.py:404`） |

三者关系：拼接张量里 rank r 的那一段就等于 rank r 的局部张量，即
`gathered[r*L : (r+1)*L] == local_rank_r`（由 `.../layers/cp_zigzag.py:398-408` 的等行断言保证，也是 `.../distributed/utils.py:37-40` 能“全量 all_reduce 后切本 rank 块”的前提）；
`inv_gather_index` 是 `zigzag_gather_index` 的逆置换（`inv_gather_index[zigzag_gather_index[p]] = p`），
对 rank r 的局部第 i 行有 `inv_gather_index[zigzag_index[i]] = r*L + i`。

### 2.4 half q/kv 长度

- `total_q_prev_tokens` / `total_q_next_tokens`：L 按 prev/next 拆分
  （`vllm-ascend/vllm_ascend/layers/cp_zigzag.py:518-525`）。
- 每请求 `q_len_prev/next` 用 `_cumsum_list` 生成累计值（int32，长度 num_reqs）
  （`vllm-ascend/vllm_ascend/attention/sfa_v1.py:376-378`、`.../attention/sfa_v1.py:429-435`）。
- KV 侧：`kv_len_prev[s] = prefix[s] + sum(blocks[0..r])`，
  `kv_len_next[s] = prefix[s] + sum(blocks[0..m-1-r])`
  （`vllm-ascend/vllm_ascend/layers/cp_zigzag.py:527-538`）。
- 单次 kernel 调用的合并元数据：prev/next 当作 2B 个 batch，
  `actual_seq_lengths_query_zigzag = cumsum(prevs) ++ (prev_total + cumsum(nexts))`，
  `actual_seq_lengths_key_zigzag = kv_len_prev_list ++ kv_len_next_list`
  （`vllm-ascend/vllm_ascend/attention/sfa_v1.py:389-396`）；
  `block_table_zigzag` 是把真实请求行先 `index_select` 再复制两份（int64 [2*num_reqs, blocks]）
  （`vllm-ascend/vllm_ascend/attention/sfa_v1.py:740-752`）。

### 2.5 plan 校验与回退

- 构造期校验（抛 `ValueError`/`AssertionError`）：
  无请求 `.../layers/cp_zigzag.py:430-431`；prefix 长度不匹配 `:432-436`；
  `cp_size<=1` `:437-438`；`T % cp_size != 0` `:447-451`；
  `num_actual_tokens` 越界 `:454-458`、与 `sum(query_lens)` 不等 `:459-463`；
  padding 为负 `:367-371`；块和与预期不符 `:390-395`；某 rank 局部行数不等于 L `:399-408`；
  多余块总数不整除 cp_size `:264-268`；块起点累计不等于 T `:478-481`；
  `zigzag_index` 长度不等于 L `:493-497`；prev+next 不覆盖 L `:522-525`；
  `_individual_remainder_extras` 无解 `:338-341`；`_build_zigzag_meta` 内部 prev+next 校验
  `vllm-ascend/vllm_ascend/attention/sfa_v1.py:354-356`。
- 运行期回退：`AscendSFAMetadataBuilder._build` 捕获 `(ValueError, AssertionError, RuntimeError)`，
  打 warning_once 后本 batch 退回连续切片（`vllm-ascend/vllm_ascend/attention/sfa_v1.py:709-730`，
  门名记为 `plan_error:<Type>`）。
- 已经建好 zigzag metadata 之后才被否决（draft / V2 runner / DP>1）时，
  用 metadata 里保存的连续切片 cos/sin、slot_mapping 覆盖回连续态，
  缺任一 fallback 张量直接 `RuntimeError`（`vllm-ascend/vllm_ascend/ascend_forward_context.py:105-163`，
  保存点 `vllm-ascend/vllm_ascend/attention/sfa_v1.py:929-933`）。
- 模型侧的硬校验：`hidden_states`/`positions` shard 行数不等于 `positions.shape[0] // tp_size`
  直接抛 `RuntimeError`（`vllm-ascend/vllm_ascend/patch/worker/patch_deepseek_v2.py:344-351,370-375,384-389`）。

---

## 3. 端到端调用路径（按顺序）

1. padding：`NPUModelRunner._determine_batch_execution_and_padding` 调
   `_pad_for_sequence_parallelism(num_tokens, num_scheduled_tokens_np)`
   （`vllm-ascend/vllm_ascend/worker/model_runner_v1.py:2693-2695`），实现只有 `round_up(tp_size)`
   （`.../worker/model_runner_v1.py:2625-2629`）。
2. `AscendCommonAttentionMetadata` 组装：`is_prefilling`（CPU 张量）
   `vllm-ascend/vllm_ascend/worker/model_runner_v1.py:2882-2883`，
   `is_prefilling_cpu=is_prefilling`（同一对象）`:2913`，
   `positions` 为全量 padded `:2916`，`num_input_tokens=num_tokens_padded` `:2914`，
   `num_actual_tokens=num_tokens` `:2906`。
3. metadata 构建：`AscendSFAMetadataBuilder.build`（`vllm-ascend/vllm_ascend/attention/sfa_v1.py:559-570`）
   → `_build`（`.../attention/sfa_v1.py:598`）。
   - 取 `block_table/slot_mapping/input_positions`（`.../attention/sfa_v1.py:607-609`）；
   - 从 host `query_start_loc_cpu` 算 `query_lens_cpu`、`prefix_lens_cpu`、`real_req_indices`
     （`.../attention/sfa_v1.py:625-652`，其中 `prefix = seq_lens_cpu[i] - raw_query_len[i]` `:641-644`）；
   - DSA-CP 分支先把 slot_mapping 补齐/截到 `num_tokens_pad`，算连续切片
     `slot_mapping_cp_continuous = slot_mapping[local_start:local_end_with_pad]`
     （`.../attention/sfa_v1.py:662-676`）；
   - 资格门（`.../attention/sfa_v1.py:687-706`）→ `_build_zigzag_meta`（`:710-719`）；
   - `block_table_zigzag`（`:740-752`）；
   - RoPE 表：zigzag 时 cos/sin 直接按 `input_positions[zigzag_index]` 生成（`use_cache=False`），
     另算一份连续切片的 cos/sin 作为 fallback（`.../attention/sfa_v1.py:754-771`）；
     非 zigzag 时保持 base 逻辑（`:772-785`）；
   - `slot_mapping_cp = slot_mapping[zigzag_index]`（`:797`）；
   - 连续切片的 `actual_seq_lengths_query/key` 计算在 zigzag 时仍然执行（`:843-867`）；
   - 组装 `DSACPContext`（`:869-934`）与 `AscendSFAMetadata`（`:971-989`，cos/sin 取 `[:num_input_tokens]` `:982-983`）。
4. forward context：`set_ascend_forward_context`（`vllm-ascend/vllm_ascend/ascend_forward_context.py:192-378`）
   找 ctx、定 active、必要时回退、写 `zigzag_cp_active/zigzag_cp_context`（`:230-244`、`:317-321`、`:338-343`），
   并把 `input_ids` 用 `zigzag_gather_index` 重排成 MoE 侧顺序（`:345-354`）、
   同样重排 `mc2_mask`（`:366-374`）。
5. 模型入口（embedding 处）：`DeepseekV2Model.forward` 的 Ascend patch
   （`vllm-ascend/vllm_ascend/patch/worker/patch_deepseek_v2.py:312-433`）：
   - `zigzag_active = bool(_EXTRA_CTX.zigzag_cp_active)`（`:321`）；
   - 默认路径：`hidden_states = self.embed_input_ids(input_ids)`（`:355`）——
     `AscendVocabParallelEmbedding.forward`→`_forward_origin`→`maybe_pad_and_reduce`
     （`vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py:164-168,283-288`），
     返回的是 TP reduce-scatter 后的连续局部切片；随后若行数不等于 padded 行数就
     `tensor_model_parallel_all_gather` 回全量（`.../patch/worker/patch_deepseek_v2.py:357-368`），
     再 `zigzag_shard_tensor`（`.../patch/worker/patch_deepseek_v2.py:369`；
     实现 `vllm-ascend/vllm_ascend/layers/cp_zigzag.py:713-722`，即 `x[zigzag_index]`）；
   - 实验路径（`VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL=1`）：
     `forward_zigzag_local` = `all_reduce(_embed_partial(input_ids))` 后直接 shard
     （`vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py:170-201`，入口
     `.../patch/worker/patch_deepseek_v2.py:339-353`）；
   - `positions = zigzag_shard_positions(positions)`（`.../patch/worker/patch_deepseek_v2.py:382-389`）；
   - 逐层调用（`:402-412`）→ `_patched_decoder_layer_forward`（`:497-512`）→ `_zigzag_layer_forward`（`:436-491`）：
     attention 直接吃 rank-local 行（不再做层内 all_gather），
     `hidden_states = self.mlp(hidden_states)`（`:486`，**不传** `already_sequence_parallel=True`）。
6. 单层 attention（`AscendSFAImpl.forward`，`vllm-ascend/vllm_ascend/attention/sfa_v1.py:2479-2848`）：
   - `slot_mapping_cp`、连续切片的 `actual_seq_lengths_query/key` 从 ctx 取（`:2513-2518`）；
   - native 预处理分支：`qkv_lora`、`q_c` 直接在该 rank-local hidden 上算（`:2582-2595`）；
   - `zigzag_active` 判定（`:2601-2606`）；
   - indexer 预处理（`:2608-2613`）；
   - KV：`exec_kv(kv_no_split, cos, sin, kv_cache, kv_slots=slot_mapping_cp, ...)`
     （`:2619-2624`，实现 `:1714-1772`，DSA-CP 分支用 `npu_kv_rmsnorm_rope_cache(is_output_kv=True)` `:1746-1759`）；
   - KV all_gather 成 T 行（`:2627-2634` → `:2220-2289`）；
   - Q 投影 + RoPE（`:2636-2637`，`_q_proj_and_k_up_proj` `:1776-1800`）；
   - KV 写回/索引 KV：`_maybe_store_kvcache_for_c8_n_dsacp`（`:2644-2661` → `:2291-2416`），
     zigzag 时用 `scatter_slots = slot_mapping_sfa[zigzag_gather_index]` 配全量 gathered KV
     （`:2358-2372`），C8-SFA 走 `npu_scatter_nd_update_`（`:2373-2378`），
     非 C8 走 `reshape_and_cache`（`:2403-2414`）；
   - indexer cache 写：zigzag 时禁用 `store_kv_block` 快速路径并改用
     `idx_slots = slot_mapping[zigzag_gather_index]` + scatter（`:2670-2726`，判定 `:2675`、重排 `:2682-2691`）；
   - top-k：zigzag 时一次 LightningIndexer 调用，传合并后的
     `actual_seq_lengths_*_zigzag` 与 `block_table_zigzag`（`:2739-2764`）；
   - SFA：zigzag 时同样一次调用 + 合并元数据（`:2782-2799`）；
   - o_proj：DSA-CP 全量权重分支 `_handle_o_proj_weight_switch_and_forward`
     （`:2813-2831` → `:1611-1656`），否则普通 o_proj（`:2840-2841`）；
   - 不做层内 rerange，直接返回 rank-local 行（`:2843-2848`）。
7. 层间 MLP/MoE（同一 rank-local 行序）：
   dense MLP 的 gate_up 是 `SequenceColumnParallelOp`，先 `maybe_all_gather_and_maybe_unpad`
   把 L 行 all_gather 成 T 行（`vllm-ascend/vllm_ascend/ops/linear_op.py:314-316` →
   `.../ops/register_custom_ops.py:68-98`）；
   down_proj 是 `SequenceRowParallelOp`，zigzag 时走 fixed-order 归约（`.../ops/linear_op.py:462-470`）。
   MoE 层：prepare 做 EP/TP all_gather（`vllm-ascend/vllm_ascend/ops/fused_moe/prepare_finalize.py:386-387`），
   finalize 做归约（`.../ops/fused_moe/prepare_finalize.py:523` → `.../ops/register_custom_ops.py:115-124`）。
8. 模型出口：`DeepseekV2Model.forward` 跳过 base 的出口 all_gather
   （条件加了 `not zigzag_active`，`vllm-ascend/vllm_ascend/patch/worker/patch_deepseek_v2.py:408-410,417-423`），
   返回 rank-local hidden；`NPUModelRunner._model_forward` 里
   `zigzag_cp_active` 时调 `zigzag_gather_hidden_states_and_aux`
   （`vllm-ascend/vllm_ascend/worker/model_runner_v1.py:2603-2610`）：
   `tensor_model_parallel_all_gather` + `gathered[inv_gather_index]`
   （`vllm-ascend/vllm_ascend/layers/cp_zigzag.py:731-757`）。
   重排后前 `num_actual_tokens` 行就是自然序真实 token，padding 行留在尾部。
9. logits：`sample_hidden_states = hidden_states[logits_indices]`
   （`vllm-ascend/vllm_ascend/worker/model_runner_v1.py:2073`），`logits_indices < num_scheduled_tokens`，
   因此与尾部 padding 行无关。

---

## 4. 通信清单表

形状记号：T = `num_tokens_pad`，L = T / cp_size，H = hidden_size，B = 请求数。
“group”指实际执行的进程组；DSA-CP 下 CP 就是 TP 组（`get_tp_group()`，`vllm-ascend/vllm_ascend/attention/sfa_v1.py:663`）。

| 位置(file:line) | 原语 | group | 形状 | 触发条件 | 相比 base 多出来的通信 |
| --- | --- | --- | --- | --- | --- |
| `vllm-ascend/vllm_ascend/ops/register_custom_ops.py:119-121`（embedding 经 `.../ops/vocab_parallel_embedding.py:287`） | 默认 allreduce 模式：`dist.all_reduce` + 切本地块（`.../distributed/utils.py:127-128`、`.../distributed/utils.py:37-40`）；reducescatter 模式：`dist.reduce_scatter_tensor`（`.../distributed/utils.py:56-58`） | TP | [T,H] → [L,H] | `zigzag_active()` 且 `dp_metadata is None`（DP=1） | 有：base 只做一次 reduce_scatter；zigzag 默认改成 all_reduce（同量级约 2× 字节），得到全量后又只留 [L,H] |
| `vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py:200` | `tensor_model_parallel_all_reduce` | TP | [T,H] | `VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL=1`（默认关） | 有（实验分支） |
| `vllm-ascend/vllm_ascend/patch/worker/patch_deepseek_v2.py:368` | `tensor_model_parallel_all_gather` | TP | [L,H] → [T,H] | `zigzag_active` 且 embedding 返回局部行（默认路径） | 有：base 不重新聚合 embedding |
| `vllm-ascend/vllm_ascend/ops/fused_moe/prepare_finalize.py:386-387`（经 `.../ops/register_custom_ops.py:74-85`） | `tensor_model_parallel_all_gather` | TP（DP=1 时 EP=TP） | [L,H] → [T,H] | MoE 层 + `enable_sp()` | 无（与 base 相同） |
| `vllm-ascend/vllm_ascend/ops/fused_moe/prepare_finalize.py:523`（经 `.../ops/register_custom_ops.py:119-121`） | 同第一行（fixed-order 或原生 reduce_scatter） | TP/EP | [T,H] → [L,H] | MoE finalize + DP=1 | 有：原语从 reduce_scatter 升级为 all_reduce+切片 |
| `vllm-ascend/vllm_ascend/ops/linear_op.py:314-316`（SequenceColumnParallelOp） | `maybe_all_gather_and_maybe_unpad` → `tensor_model_parallel_all_gather`（`.../ops/register_custom_ops.py:78`） | TP | [L,H] → [T,H] | enable_sp 的 column-parallel（gate_up/qkv 等） | 无（与 base 相同），但 zigzag 下同样是“rank 拼接序”而非自然序 |
| `vllm-ascend/vllm_ascend/ops/linear_op.py:462-470` | zigzag：all_reduce+切片（`.../distributed/utils.py:127-128`）；base：`tensor_model_parallel_reduce_scatter` | TP | [T, out] → [L, out] | `zigzag_active()` | 有：原语升级为 all_reduce（约 2× 字节） |
| `vllm-ascend/vllm_ascend/ops/linear_op.py:203-214`（MLPRowParallelOp，`mlp_tp_enable` 时） | 同上 / `comm_group.reduce_scatter` | MLP-TP | [T,out] → [L,out] | `zigzag_active()` | 有（同上一行，路径互斥） |
| `vllm-ascend/vllm_ascend/attention/sfa_v1.py:2261-2265` | `all_gather_async` | TP | [L, kv_lora+rope(+head_dim)] → [T, ...] | `enable_dsa_cp` | 无（与 base 相同） |
| `vllm-ascend/vllm_ascend/attention/sfa_v1.py:2271-2275`、`:2281-2285` | `all_gather_async` | TP | [L,head_dim] → [T,head_dim] | `enable_sparse_sfa_c8` 或 `enable_sparse_li_c8` | 无（与 base 相同） |
| `vllm-ascend/vllm_ascend/attention/sfa_v1.py:2341-2353` | `all_gather_async`（异步） | TP | o_proj 分片权重 → 全量权重（每层） | `full_gather_o_proj_enabled`（prefill/mixed 且 `enable_dsa_cp_with_o_proj_tp`） | 无（与 base 相同） |
| `vllm-ascend/vllm_ascend/worker/model_runner_v1.py:2608` → `.../layers/cp_zigzag.py:743` | `tensor_model_parallel_all_gather` + `[inv_gather_index]` 重排 | TP | [L,H] → [T,H] → 自然序 | `zigzag_cp_active` | 基本无：base 在 `.../worker/model_runner_v1.py:2610` 也做一次 all_gather；但 base 模型内还有一次出口 all_gather（`.../patch/worker/patch_deepseek_v2.py:417-423`），zigzag 反而少一次 |
| `vllm-ascend/vllm_ascend/ops/fused_moe/comm_utils.py:53,62` | `dist.all_to_all_single`（EP dispatch/combine） | EP | [T,H] 量级 | MoE + EP | 无（未改动） |
| `vllm-ascend/vllm_ascend/attention/sfa_v1.py:1654,1688,1704` | `all_to_all_single` / `reduce_scatter_tensor`（oproj_tp 路径） | OTP group | - | `oproj_tp_enable()`（要求 TP=1，与 DSA-CP 互斥） | 不适用 |
| `vllm-ascend/vllm_ascend/patch/worker/patch_deepseek_v2.py:478` | `tensor_model_parallel_all_reduce` | TP | [L,H] | `use_sequence_parallel_moe and not full_o_proj` | 不执行（zigzag 要求 DP=1，而 `use_sequence_parallel_moe` 要求 DP>1，见 vllm/config/parallel.py:654-666） |

要点：zigzag 并没有新增“层内 attention all_gather/reduce_scatter”这类通信，
真正新增的是 embedding 的“重新聚合”（`.../patch/worker/patch_deepseek_v2.py:368`）
以及把所有 row-parallel 归约从 reduce_scatter 换成 all_reduce+切片（约 2× 字节）。

---

## 5. 与 base 的差异逐条对照表

| 文件/位置 | base | 当前分支 | 类型 |
| --- | --- | --- | --- |
| `vllm-ascend/vllm_ascend/layers/cp_zigzag.py`（整文件 757 行） | 不存在 | 新增：plan、index、shard/gather、fixed_order_rank_sum、max-flow | 新增 |
| `vllm-ascend/vllm_ascend/layers/__init__.py` | 不存在 | 新增包标记（1 行） | 新增 |
| `.../ascend_forward_context.py:70-163` | 无 | `_iter_attn_metadata` / `_find_zigzag_cp_context` / `_disable_zigzag_metadata_for_fallback` | 新增 |
| `.../ascend_forward_context.py:230-244,317-321,338-343` | 无 zigzag 概念 | 找 ctx、定 active、DP>1 否决、写 forward context | 新增 |
| `.../ascend_forward_context.py:345-354,366-374` | 无 | `input_ids` / `mc2_mask` 按 `zigzag_gather_index` 重排 | 新增 |
| `.../ascend_forward_context.py:616-628` | 无 | `zigzag_active()` | 新增 |
| `.../distributed/utils.py:24-136` | 无 | `_allreduce_slice_reduce_scatter` / `_plain_reduce_scatter` / `_all_to_all_fixed_order_reduce_scatter` / `fixed_order_reduce_scatter` | 新增 |
| `.../ops/linear_op.py:202-214` | `output = self.comm_group.reduce_scatter(output_parallel, 0)` | zigzag 时 `fixed_order_reduce_scatter`，否则原语句 | 包裹 |
| `.../ops/linear_op.py:351-384` | 无 | `SequenceRowParallelOp._fixed_order_reduce_scatter` | 新增 |
| `.../ops/linear_op.py:462-470` | `output = tensor_model_parallel_reduce_scatter(output_parallel, 0)` | zigzag 分支 + 原语句 | 包裹 |
| `.../ops/linear_op.py:405-406` | `dsa_cp_attn_out = enable_dsa_cp() and (...)` | 拆出局部变量 `dsa_cp`（纯重构） | 重构 |
| `.../ops/register_custom_ops.py:25-48` | 无 | `_fixed_order_zigzag_reduce_scatter` | 新增 |
| `.../ops/register_custom_ops.py:119-124` | 直接 `tensor_model_parallel_reduce_scatter(x, 0)` | 先试 fixed-order，失败/非 zigzag 再退回原语句 | 包裹 |
| `.../ops/vocab_parallel_embedding.py:262-288` | `_forward_origin` 一体 | 拆出 `_embed_partial`，`_forward_origin` 只做 reduce | 重构 |
| `.../ops/vocab_parallel_embedding.py:170-201` | 无 | `forward_zigzag_local`（实验） | 新增 |
| `.../patch/worker/patch_deepseek_v2.py:319-433` | 无 zigzag | `zigzag_active`、embedding 重聚合+shard、positions shard、跳过入口/出口 all_gather | 包裹/新增 |
| `.../patch/worker/patch_deepseek_v2.py:436-512` | 无 | `_zigzag_layer_forward` + `DeepseekV2DecoderLayer.forward` 替换 | 替换 |
| `.../patch/worker/patch_deepseek_v2.py:515-516` | 只替换 `DeepseekV2Model.forward` | 追加 `DeepseekV2DecoderLayer.forward` | 新增 |
| `.../attention/sfa_v1.py:230-273` | `DSACPContext` 只有连续切片字段 | 新增 zigzag/合并元数据/fallback 字段 | 新增 |
| `.../attention/sfa_v1.py:276-316,319-435` | 无 | `_zigzag_gate_reason` / `_build_zigzag_meta` / `_cumsum_list` | 新增 |
| `.../attention/sfa_v1.py:625-652,686-934` | 直接算 cos/sin + 连续切片 | 新增 CPU 侧 query/prefix 提取、资格门、plan、block_table_zigzag、zigzag cos/sin、slot_mapping_cp、ctx 组装 | 替换/新增 |
| `.../attention/sfa_v1.py:940-960` | 无 | `[CP_BALANCE][branch]` 日志 | 纯日志 |
| `.../attention/sfa_v1.py:798-813` | 无 | `[CP_BALANCE][plan]` 日志 | 纯日志 |
| `.../attention/sfa_v1.py:1776-1800` | base 有 `npu_transpose_batchmatmul` 融合 | 当前分支补齐 `hasattr(torch_npu, "npu_transpose_batchmatmul")` 分支，否则 transpose+bmm | 修复（commit 9e462094b） |
| `.../attention/sfa_v1.py:2060-2122,2123-2163` | `indexer_select_post_process` 一体 | 拆出 `_indexer_qk_proj`，新增 `block_table` 形参 | 重构 |
| `.../attention/sfa_v1.py:2189-2210` | 无 `block_table` 形参 | 新增透传 | 新增 |
| `.../attention/sfa_v1.py:2358-2372,2386-2414` | 用 `slot_mapping_sfa[:num_actual]` + KV 截断 | zigzag 时改用重排 slot + 全量 gathered KV；k_li 取全量 | 替换 |
| `.../attention/sfa_v1.py:2670-2726` | `k_li`/`slot_mapping` 直接 scatter | zigzag 时禁用 reshape 优化 + 重排 slot | 替换 |
| `.../attention/sfa_v1.py:2736-2809` | 单路 topk + SFA | 增加 zigzag 分支（合并元数据 + `block_table_zigzag`） | 新增分支 |
| `.../device/device_op.py:488,495-496` 与 `:1690,1694-1695` | `indexer_select_post_process` 只用 `attn_metadata.block_table` | 新增可选 `block_table`（None 则回退原行为） | 新增 |
| `.../device/device_op.py:560-565`（`execute_sparse_flash_attention_process`） | 已有 `block_table` 形参 | 未改 | 无 |
| `.../attention/utils.py:229-231,296` | 无 | 新增 `is_prefilling_cpu` 字段与切片传递 | 新增（与 `is_prefilling` 重复） |
| `.../worker/model_runner_v1.py:2603-2610` | `flash_comm_v1_enabled` 时 `_all_gather_hidden_states_and_aux` | 增加 `zigzag_cp_active` → `zigzag_gather_hidden_states_and_aux` | 新增分支 |
| `.../worker/model_runner_v1.py:2613-2629` | `_pad_for_sequence_parallelism(num_scheduled_tokens)` | 只加未使用的 `num_scheduled_tokens_np` 形参和注释 | 纯注释/死参数 |
| `.../worker/model_runner_v1.py:2913` | 无 | `is_prefilling_cpu=is_prefilling` | 新增（重复） |
| `.../utils.py:1372-1425` | `enable_dsa_cp` / `enable_dsa_cp_with_o_proj_tp` 各自读全局 config | 抽出 `enable_dsa_cp_for_config` / `dsa_cp_with_o_proj_tp_for_config`，原函数变缓存包装 | 重构 |
| `.../ops/fused_moe/experts_selector.py:250-252` | `input_ids = forward_context.input_ids.to(torch.int64)` | 仅加注释 | 纯注释 |
| `tests/ut/ops/test_vocab_parallel_embedding.py:204-249` | 无 | `forward_zigzag_local` 单测 | 新增测试 |

---

## 6. 精简候选清单

### 6.1 可删

1. `vllm-ascend/vllm_ascend/layers/cp_zigzag.py:668-710` `can_enable_zigzag_for_batch`。
   仓库内无任何调用点（只有注释与自身定义提及）。风险：无；如有外部脚本 import 需同步改名。
2. `vllm-ascend/vllm_ascend/layers/cp_zigzag.py:148-209,212-242,317-350` 的 `_MaxFlow`、
   `_decompose_remainder_rows`、`_individual_remainder_extras`（约 200 行）。
   只是把每条序列 ≤ 2cp-1 个多余块均分到 cp 个 pair 上，可用贪心轮转（按 remainder 排序轮流补齐）
   替换。风险：块分布会变，需重跑首 token 对比（B/C 布局同时变，
   自洽性由 `.../layers/cp_zigzag.py:398-408` 的等行断言保证）。
3. `vllm-ascend/vllm_ascend/attention/sfa_v1.py:87-101` `_NPU_INDEX_UNSUPPORTED_FP8_DTYPES` /
   `_supports_npu_advanced_index`，无调用点。风险：无。
4. SGLang 兼容但无人读的字段：`split_list`、`cp_reverse_index`、`reverse_split_len`、`prefix_offsets`、
   `q_len_prev`、`q_len_next`、`kv_len_prev`、`kv_len_next`、`actual_seq_q_prev_list`、
   `actual_seq_q_next_list`、`kv_len_prev_list`、`kv_len_next_list`、`q_half`、`total_q_prev_tokens`、
   `total_q_next_tokens`。定义/写入见 `vllm-ascend/vllm_ascend/layers/cp_zigzag.py:139-141,540-562,574-586`、
   `.../attention/sfa_v1.py:239-258,409-424,885-917`、`.../ascend_forward_context.py:142-158`。
   其中 q/kv half 长度仅被 `.../attention/sfa_v1.py:808-811` 的 debug 日志读取，
   `total_q_prev_tokens`/`total_q_next_tokens`/`q_half` 连日志都不读（只在 `.../attention/sfa_v1.py:354` 做一次 plan 内部校验）。风险：删时要同步那条日志。
5. `vllm-ascend/vllm_ascend/distributed/utils.py:43-59` `_plain_reduce_scatter`（等价于 base 的
   `reduce_scatter`）与 `:62-81` `_all_to_all_fixed_order_reduce_scatter`（含
   `vllm-ascend/vllm_ascend/layers/cp_zigzag.py:52-76` `fixed_order_rank_sum`）——
   只有 `REDUCE_MODE=reducescatter/alltoall` 的 A/B 调试会走到。风险：失去 A/B 手段，可移到独立调试模块。
6. `vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py:170-201` `forward_zigzag_local`
   + `vllm-ascend/vllm_ascend/envs.py:109-111` `VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL`
   + `vllm-ascend/tests/ut/ops/test_vocab_parallel_embedding.py:204-249`：实验开关默认 0。
   风险：如果后续决定用它替换默认路径，应先修默认路径（见 6.3-1）。
7. `vllm-ascend/vllm_ascend/ascend_forward_context.py:338-343` 的 V2-runner 分支：
   V2 runner 下 `zigzag_cp_active` 恒为 False（`.../ascend_forward_context.py:231-235`），
   写进去的永远是 None。风险：低，需保留属性名以兼容读取方。
8. `vllm-ascend/vllm_ascend/patch/worker/patch_deepseek_v2.py:471-478`
   `use_sequence_parallel_moe and not full_o_proj` 的 all_reduce 分支：
   zigzag 要求 DP=1（`.../layers/cp_zigzag.py:618-619`），而 `use_sequence_parallel_moe`
   要求 `data_parallel_size > 1`（`vllm/config/parallel.py:654-666`），永不执行。风险：无（除非上游改语义）。

   > 更正（2026-09-21）：上面这条判断的前提已被本地改动推翻——被测树把 `enable_dsa_cp` 改成只判 `has_indexer`，
   > 所以 dp=1 下 DSA-CP 会开、而 SP（`use_sequence_parallel_moe`）仍然关。真正的问题也因此不同：
   > 不是这条死分支，而是**层内 MoE 的布局**——zigzag 给 MoE 的是 rank-local 切片行，而 SP 关时的 MoE
   > （no-DP-EP：每 rank 用自己的专家对全部行算 partial，再做一次 all_reduce）需要**全量行**。
   > 结论与三条出路见 `docs/scripts_review.md` §8；此处"风险：无"不再成立。
9. `vllm-ascend/vllm_ascend/worker/model_runner_v1.py:2616` 形参 `num_scheduled_tokens_np` 未被使用。
   风险：无。
10. `vllm-ascend/vllm_ascend/attention/utils.py:229-231` `is_prefilling_cpu` 字段与
    `:296` 的切片传递：`vllm-ascend/vllm_ascend/worker/model_runner_v1.py:2913` 传的就是
    `is_prefilling` 同一对象（`:2882` 由 CPU 张量比较得到），
    `.../attention/sfa_v1.py:645-652` 读它是多此一举（直接读 `is_prefilling` 即可）。
    风险：无。
11. `vllm-ascend/vllm_ascend/layers/cp_zigzag.py:713-722` 中重复的 `dim` 分支
    （`x.ndim > 1 and dim == 0` 与 `dim == 0` 两段代码完全相同）。风险：无。

### 6.2 可复用 base

1. `vllm-ascend/vllm_ascend/attention/sfa_v1.py:276-316` `_zigzag_gate_reason`：只比
   `zigzag_ineligible_reason` 多一个在当前调用点恒假的 `num_tokens != num_tokens_pad`
   （`.../attention/sfa_v1.py:301-302`；调用处 `num_tokens == num_input_tokens == num_tokens_pad`，
   `.../attention/sfa_v1.py:664-665,687-690`）。可直接调 base 谓词。
2. `vllm-ascend/vllm_ascend/layers/cp_zigzag.py:725-728` `zigzag_shard_positions` 与
   `:713-722` `zigzag_shard_tensor` 同体；`:747-748` `zigzag_gather_hidden_states_list` 与
   `vllm-ascend/vllm_ascend/worker/model_runner_v1.py:2542-2549`
   `_all_gather_hidden_states_and_aux` 结构重复。可合成“一个带置换参数的 gather/shard helper”。
   风险：低，注意 base 路径要保留 `pad_size` 截断语义。
3. 三处同样的“zigzag_active + rows%ws==0 → fixed-order，否则原生”逻辑且异常处理不一致：
   `vllm-ascend/vllm_ascend/ops/linear_op.py:202-214`（`except Exception`）、
   `:351-384`（`except (RuntimeError, NotImplementedError, TypeError, ValueError)`）、
   `vllm-ascend/vllm_ascend/ops/register_custom_ops.py:25-48`（`except Exception`）。可收敛成一个 helper。
   风险：低，统一异常类型时别把 `TypeError` 吞掉导致掩盖真 bug。
4. `vllm-ascend/vllm_ascend/distributed/utils.py:119` 用 `os.getenv` 重解析，
   而 `vllm-ascend/vllm_ascend/envs.py:96-98` 已定义同一变量且无读者。应改用 envs 访问器并缓存到模块级。
   风险：无。
5. `vllm-ascend/vllm_ascend/patch/worker/patch_deepseek_v2.py:436-512`
   `_zigzag_layer_forward` + `_patched_decoder_layer_forward`：
   去掉 8 号候选里那条死分支后，与 base 的 `DeepseekV2DecoderLayer.forward` 等价
   （base 的 `use_sequence_parallel_moe` 分支在 DP=1 下同样不执行，
   `vllm/model_executor/models/deepseek_v2.py:1279-1336`），可以直接不 patch。
   风险：中；如果上游把 `use_sequence_parallel_moe` 的成立条件改成不依赖 DP>1，
   层内 all_gather/reduce_scatter 会破坏 zigzag 行序，建议在资格门里显式断言。

   > 更正（2026-09-21）：这里担心的方向还没发生，但**相反的情形已经发生**——本地把 `enable_dsa_cp`
   > 与 SP 解耦后，dp=1 下 DSA-CP 开、SP 关，zigzag 的切片行会直接送进需要全量行的 MoE。
   > 见 `docs/scripts_review.md` §8。
6. `vllm-ascend/vllm_ascend/distributed/utils.py:37` `tensor.contiguous().clone()`
   可直接用调用方的 `output_parallel`（新分配的张量）做 in-place all_reduce，省一次整张量拷贝；
   若要保守则可只对非 contiguous 输入做一次拷贝。风险：中，需确认四处调用点传入的张量都不再被复用
   （`.../ops/linear_op.py:208,377`、`.../ops/register_custom_ops.py:41`）。

### 6.3 可合并

1. embedding 入口两条路径合并（默认 `.../patch/worker/patch_deepseek_v2.py:355-369` 与实验
   `.../ops/vocab_parallel_embedding.py:170-201`）：默认路径先做一次归约又立刻 all_gather 回来，
   而 allreduce 模式的全量结果本来就在手（`vllm-ascend/vllm_ascend/distributed/utils.py:37-40` 切掉了它）。
   合并为“embedding 返回全量 T 行 + 模型边界 shard”可省两次集合通信。
   风险：中，需确认非 zigzag/非 DSA-CP 路径仍走 `_forward_origin` 原语义。
2. 两处 slot_mapping 重排在 `DSACPContext` 里预算一次：
   `vllm-ascend/vllm_ascend/attention/sfa_v1.py:2366-2368` 与 `:2689-2691` 用的是同一张量
   （`_get_sfa_kv_slot_mapping` 直接返回 `attn_metadata.slot_mapping`，`.../attention/sfa_v1.py:2418-2422`）。
   风险：低，注意 metadata 是跨层共享对象，预算是 per-batch 而非 per-layer。
3. 先定 active 再建 metadata：`vllm-ascend/vllm_ascend/ascend_forward_context.py:105-163`
   的逐字段清零 + `vllm-ascend/vllm_ascend/attention/sfa_v1.py:929-933` 保存 fallback，
   本质是“判定顺序倒置”的补丁；把 draft/V2/DP 判定提前到 metadata 构建即可省掉这两块。
   风险：高（改动面横跨 runner 与 builder），收益是删掉两份 fallback 张量的保存/清零。
4. `vllm-ascend/vllm_ascend/layers/cp_zigzag.py:747-757` 与
   `vllm-ascend/vllm_ascend/worker/model_runner_v1.py:2543-2549` 的 aux 列表处理合并为一份。
   风险：低。

---

## 7. 性能可疑点清单

1. `vllm-ascend/vllm_ascend/patch/worker/patch_deepseek_v2.py:368`：每次 zigzag forward 多一次
   TP all_gather（[L,H] → [T,H]，T 行 bf16）；它只是把 `.../ops/vocab_parallel_embedding.py:287`
   刚 reduce 掉的行再拼回来。
2. 默认路径 embedding 一次 forward 走 3 个集合通信：
   `.../ops/register_custom_ops.py:119-121`（all_reduce+切片，全量结果被丢）
   → `.../distributed/utils.py:40` 只留 [L,H] → `.../patch/worker/patch_deepseek_v2.py:368` 再 all_gather。
3. `vllm-ascend/vllm_ascend/distributed/utils.py:37` 每次归约都 `contiguous().clone()`，
   复制整张 [T, N]（embedding 时 N=H，row-parallel 时 N=该层输出分片宽）；dense MLP + MoE finalize 每层各一次。
4. allreduce 模式把每个 row-parallel 归约的通信量放大到 reduce_scatter 的约 2 倍
   （`vllm-ascend/vllm_ascend/distributed/utils.py:127-128`），每层 dense MLP / MoE finalize 各一次。
5. 每层都重新读环境变量：`vllm-ascend/vllm_ascend/ops/linear_op.py:212,468`、
   `.../ops/register_custom_ops.py:122`、`.../attention/sfa_v1.py:798,940`；
   `envs.py` 的惰性求值（`vllm-ascend/vllm_ascend/envs.py:140-144`）使每次访问都 `os.getenv` + `int()`；
   `.../distributed/utils.py:119` 再解析一次字符串。
6. `vllm-ascend/vllm_ascend/attention/sfa_v1.py:2366-2368` 与 `:2689-2691`：每个 attention 层各做一次
   `slot_mapping[zigzag_gather_index]`（同一张量、同一重排），每层 2 次设备端 gather。
7. `vllm-ascend/vllm_ascend/attention/sfa_v1.py:762-771`：每个 group 建 metadata 时做
   `input_positions[zigzag_index]`、`input_positions[local_start:...]` 两次高级索引/切片，
   并调用两次 `get_cos_and_sin_mla(..., use_cache=False)`
   （`vllm-ascend/vllm_ascend/ops/rotary_embedding.py:93-96`，每次都是两次 index + 两次 unsqueeze 复制）；
   其中 continuous 那份只在 fallback 时才会用到。
8. `vllm-ascend/vllm_ascend/attention/sfa_v1.py:797`：每个 group 建 metadata 时再做一次
   `slot_mapping[zigzag_index]`。
9. `vllm-ascend/vllm_ascend/attention/sfa_v1.py:740-752`：每次建 metadata 做
   `index_select` + `zeros_like` + 两次 `torch.cat` 造 2*num_reqs 行的 block_table。
10. `vllm-ascend/vllm_ascend/attention/sfa_v1.py:409-426`：`_build_zigzag_meta` 从 Python list
    造约 10 个小设备张量（H2D + kernel 启动），其中大半（见 6.1-4）无人使用。
11. `vllm-ascend/vllm_ascend/attention/sfa_v1.py:2675`：zigzag 下禁用 `store_kv_block` 快速写，
    indexer KV 每层退回 `npu_scatter_nd_update_`（`:2703-2707`、`:2721-2725`）。
12. `vllm-ascend/vllm_ascend/attention/sfa_v1.py:2366-2372` 与 `:2394`：zigzag 时 scatter /
    `reshape_and_cache` 传的是全 T 行（含 padding 行，靠 -1 slot 跳过），base 传 `num_actual_tokens` 行；
    多出的行数 ≤ tp-1，可忽略但内核规模变大。
13. `vllm-ascend/vllm_ascend/ascend_forward_context.py:352-354`：每 forward 复制一份 T 行 int64 的
    `input_ids`（`zigzag_reorder_moe_aux` 的高级索引）；`:371-373` 每 forward 复制一份 T 元素
    `mc2_mask`。两者只在 MoE hash 路由（`.../ops/fused_moe/experts_selector.py:252-256`）用到，
    对 dense 模型是纯浪费。
14. `vllm-ascend/vllm_ascend/ops/fused_moe/experts_selector.py:253`：每个 MoE 层再执行一次
    `forward_context.input_ids.to(torch.int64)`，而 `.../ascend_forward_context.py:352` 已经转过。
15. `vllm-ascend/vllm_ascend/attention/sfa_v1.py:370-372,402-404`：`zigzag_index`/`gather_index`/
    `inv_gather_index` 都用 int64（`:370-372` 的 `_int64_tensor`），T、L 远小于 2^31，int32 足够，
    高级索引的索引带宽与转换开销更小。
16. `vllm-ascend/vllm_ascend/layers/cp_zigzag.py:212-350`：每次建 metadata 跑 CPU 最大流与
    `_decompose_remainder_rows` 里的 `while` + `sorted` 循环（O(行数 × cp) 级），属于每 batch 的 host 开销。
17. `vllm-ascend/vllm_ascend/patch/worker/patch_deepseek_v2.py:329-331,344-351,370-375,384-389`：
    每次 forward 取 `tp_rank`、算 `expected_local_tokens` 并做多次 Python 层形状判断/异常构造准备；
    属于每 forward 的 host 开销（小，但可以改成 assert）。
18. `vllm-ascend/vllm_ascend/layers/cp_zigzag.py:616-628` `zigzag_active()` 每次经
    `_ExtraForwardContextProxy.__getattr__` + `check_extra_attr` + `get_forward_context()` + try/except，
    被每个 row-parallel 层（`.../ops/linear_op.py:203,463`）和每个 `maybe_pad_and_reduce`
    （`.../ops/register_custom_ops.py:36`）调用。
19. `vllm-ascend/vllm_ascend/ops/linear_op.py:419-460`（mmrs_fusion 的
    `npu_mm_reduce_scatter_base`）没有接 fixed-order：MoE 模型下 `mmrs_fusion` 被置为 False
    （`vllm-ascend/vllm_ascend/ascend_forward_context.py:279-282`），但非 MoE 的 DSA 模型且 tp ≤ 8 时
    该分支会在 zigzag forward 里做 owner 相关归约；这既是性能项也是 B/C 逐位一致性风险。
20. `vllm-ascend/vllm_ascend/ascend_forward_context.py:105-163`：一旦触发回退，
    `.../attention/sfa_v1.py:768-771` 里为 fallback 多算的那份 cos/sin 才有用，
    否则（DSA-CP + DP=1 + 非 draft 的常见路径）它是纯多余计算。

---

计数：精简候选 11（可删） + 6（可复用 base） + 4（可合并） = 21 条；性能可疑点 20 条。
