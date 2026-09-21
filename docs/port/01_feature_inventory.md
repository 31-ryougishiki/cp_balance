> 历史文档（2026-09-18 移植期取证/评审）：结论可能已过时；dp>1 那道门与 zigzag 的现状见 `docs/scripts_review.md` §4.1/§4.1c/§8。
>
> 已变事实（2026-09-21 更正）：
> - 本文 pinned 旧树 23ff2c23c（移到了 `cp_balance_v0.26.0rc`）；现行 `cp_balance` = b64c9569b（基于 aff1b74b6 的移植），file:line/类名不可引用，行为语义可参考。
> - §2 的 `VLLM_ASCEND_CP_BALANCE_MIN_TOKENS`：现场 2048 现在 `configs/_base.json`（不是 `_common.json`），envs 默认 8192 是未决定案项（`scripts_review.md` §7）。
> - 用「`use_sequence_parallel_moe` 要求 dp>1 且 zigzag 门拒 dp>1」当 S8 删除前提已不成立（见 §4.1/§8）。
> - 「`branch=ZIGZAG` 从来没打过」已经改了：现树在 zigzag 成功时补了 `[CP_BALANCE][branch] branch=ZIGZAG`（b64c9569b）。

# 01 — Feature inventory: model-level zigzag CP balance (`cp_balance`) for DSA prefill

**Source of truth / pinning.** The repo ref `cp_balance` was moved while this doc was written (it was reset to
`upstream/main` and the feature commits were re-homed on `cp_balance_v0.26.0rc`). Everything below is read from
git objects at pinned SHAs, never from the working tree (which currently contains an in-progress conflicted merge):

| role | SHA | note |
| --- | --- | --- |
| feature tree (all `file:line` in this doc) | `23ff2c23c` | = `cp_balance` at task start = `cp_balance_v0.26.0rc` now |
| old base seen by the feature | `dd13a7f29` | fork point of releases/v0.26.0rc |
| base + release cherry-picks | `ae9c13a42` | parent of `d22ddb2ef`; `ae9c13a42..23ff2c23c` = the 11 feature commits only |
| original DSA-CP reference tree | `vllm-ascend-base` `c7990e5e4` | `base_layer3` |

`git diff dd13a7f29 23ff2c23c` also contains ~30 unrelated release cherry-picks (mla_prolog_v3, quantization,
kv-pool, balance-scheduler …): **ignore every file not listed in §3**. `AGENTS.md` / `CLAUDE.md` deletions
(commit `d22ddb2ef`, −420 lines) are repo docs, not feature code.

---

## 1. Feature summary

1. `cp_balance` is a **model-boundary zigzag layout** for DSA prefill under context parallelism. CP is the TP
   group (`cp_size = get_tp_group().world_size`, `cp_rank = rank_in_group`); there is no separate CP group.
2. Instead of each rank owning one contiguous slice of the SP-padded token stream, every sequence is cut into
   `2*cp_size` consecutive blocks and rank `r` owns block `r` (head/prev) and block `2*cp-1-r` (tail/next), so
   per-rank token counts stay exactly equal (equal-shaped FlashComm collectives) while causal attention work
   spreads evenly across ranks (`vllm_ascend/layers/cp_zigzag.py:339-396`).
3. The rank-local token order becomes `[all prev blocks by sequence, all next blocks by sequence]`
   (`cp_zigzag.py:469-483`); the rank-concatenating all-gather order is
   `[r0_prev, r0_next, r1_prev, r1_next, …]` (`cp_zigzag.py:485-498`).
4. Three index vectors carry the whole layout: `zigzag_index` (global positions owned by this rank, len `L=T/cp`),
   `zigzag_gather_index` (all-gather order, len `T`), `inv_gather_index` (inverse of the former, len `T`).
5. Scope of the change (one forward): metadata layout (permutation instead of interval) → RoPE tables and KV
   write slots follow the local order → embedding is gathered back to full length and sharded → per-layer KV
   write-back uses the reordered slot mapping → top-k + SFA run once on merged `2*B`-batch metadata → MoE
   `input_ids`/`mc2_mask` are reordered once per forward → row-parallel reductions become owner-independent →
   the model exit gathers once and re-applies `inv_gather_index` (the model itself returns rank-local rows).
6. Gating is **per forward**, not per config: `zigzag_ineligible_reason()` (`cp_zigzag.py:542-618`) decides in the
   SFA metadata builder, and `set_ascend_forward_context` (`ascend_forward_context.py:219-233,306-310`) may still
   veto (draft model, V2 runner, DP>1) and then restores continuous-slice metadata from saved `fallback_*` tensors.
7. Only the eligible prefill batches take the new layout; decode/short/ineligible batches must be **bit-identical
   to the old code**, which is why every non-metadata change is gated on the per-forward `zigzag_active()`
   (`ascend_forward_context.py:604-616`).
8. Extra cost: one full-length TP all-gather at the embedding entry (the default entry re-aggregates what
   `maybe_pad_and_reduce` just reduced) plus the default `allreduce` reduction mode, which doubles the reduction
   bytes vs `reduce_scatter`. The exit gather replaces the base one (the in-model gather is skipped, so the count
   is unchanged).
9. Two orthogonal things rode along in the same big commit and are **not** part of the zigzag algorithm: (a) GLM-5.2
   checkpoint layer-slice weight filtering (`patch_deepseek_mtp.py`), (b) `@_sfa_5_3_scope` profiler decorators.
10. No NPU/runtime test covers the plan math; the two added/updated tests cover only `forward_zigzag_local` and the
    layer-slice filter (§3.18, §3.19).

---

## 2. Env vars / config flags

All are plain env vars read through `vllm_ascend/envs.py` (lazy `__getattr__` → `os.getenv` on every attribute
access, `envs.py:140-147`). There is **no** `additional_config`/ascend-config key for this feature.

| env var | default | semantics | defined | read at |
| --- | --- | --- | --- | --- |
| `VLLM_ASCEND_CP_BALANCE` | `1` (`bool(int(...))`) | master switch. `0` ⇒ `zigzag_ineligible_reason` returns `flag_off` and nothing else in the process changes. Upgrade enables it by default. | `envs.py:85` | `cp_zigzag.py:563` (only reader) |
| `VLLM_ASCEND_CP_BALANCE_MIN_TOKENS` | `8192` | minimum `num_actual_tokens` (sum of scheduled query lens) for a batch to be eligible; below it ⇒ `actual<min(N)`. Field configs used 2048. | `envs.py:90` | `cp_zigzag.py:598-599` (twice), log at `sfa_v1.py:850` |
| `VLLM_ASCEND_CP_BALANCE_REDUCE_MODE` | `allreduce` (`.strip().lower()`) | `allreduce` \| `alltoall` \| `reducescatter`; selects the owner-independent row-parallel reduction implementation. `reducescatter` restores the pre-feature owner-dependent collective (A/B debugging only). | `envs.py:96-98` | `distributed/utils.py:24-27` (`reduce_mode()`, `lru_cache(maxsize=1)`) → `fixed_order_reduce_scatter` `distributed/utils.py:130-147` |
| `VLLM_ASCEND_CP_BALANCE_DEBUG` | `0` | emits `[CP_BALANCE][plan]` / `[CP_BALANCE][branch]` / `[CP_BALANCE][reduce]` log lines; inside the plan log there are `.tolist()` calls (device→host syncs). | `envs.py:102-104` | `sfa_v1.py:714,831`, `distributed/utils.py:131`, `linear_op.py:212,467`, `register_custom_ops.py:122` |
| `VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL` | `0` | experimental alternative embedding entry: full token ids → vocab-sharded partials → TP all-reduce → local zigzag rows. Unvalidated; default path is gather+shard. | `envs.py:109-111` | `patch_deepseek_v2.py:319` |

Non-env gates that must also hold (all inside `zigzag_ineligible_reason`, in this exact order):
`enable_dsa_cp` must already be true (the caller only enters the DSA-CP branch), `cp_size > 1`, not a draft build,
not `VLLM_USE_V2_MODEL_RUNNER`, `data_parallel_size == 1`, `not enable_sfa_dcp_replicated_indexer()`,
`dsa_cp_with_o_proj_tp_for_config()` true, attention state ∈ `{PrefillNoCache, PrefillCacheHit, ChunkedPrefill}`,
per-request `query_len >= 2*cp_size`, `num_actual_tokens >= MIN_TOKENS`, `num_tokens_pad % cp_size == 0`, and
either state `PrefillNoCache` or an all-true `is_prefilling` mask.

---

## 3. File-by-file inventory

Legend: **[NEW]** new file · **[MOD]** modified file · **[HOOK]** small hook in existing code ·
**[CORE]** new feature code. Line numbers are from `23ff2c23c`.

### 3.1 `vllm_ascend/layers/cp_zigzag.py` — [NEW], 665 lines, the whole algorithm

Imports: `torch`; `vllm.distributed.get_tensor_model_parallel_world_size`,
`vllm.distributed.tensor_model_parallel_all_gather`; `vllm_ascend.envs`.
Lazy imports: `vllm_ascend.ascend_forward_context._EXTRA_CTX` (`:43`), `vllm.forward_context.get_forward_context`
(`:637`). No import of `sfa_v1` (avoids a cycle).

* `_PURE_PREFILL_ATTENTION_STATES` `:35-39` **[CORE]** — `{PrefillNoCache, PrefillCacheHit, ChunkedPrefill}`
  (string names compared with `getattr(state, "name", state)`).
* `get_zigzag_cp_context()` `:42-48` **[CORE]** — returns `_EXTRA_CTX.zigzag_cp_context` or `None` on **any**
  exception (also `None` outside a forward).
* `fixed_order_rank_sum(parts)` `:51-75` **[CORE]** — clone `parts[0]`, `add_` the rest in list order, raising on
  empty input and on shape mismatch. Local half of the layout-independent reduction; all parts must have identical
  row order.
* `zigzag_reorder_moe_aux(x, ctx=None)` `:78-100` **[CORE]** — `x[ctx.zigzag_gather_index]`; raises `RuntimeError`
  if `x.shape[0] != len(zigzag_gather_index)`; returns `x` unchanged when the ctx has no `zigzag_gather_index`.
  Used for both `input_ids` and `mc2_mask` so they cannot drift.
* `ZigzagPlan` (frozen dataclass) `:103-131` **[CORE]** — fields `num_tokens, num_reqs, cp_size, zigzag_index,
  zigzag_gather_index, inv_gather_index, total_q_prev_tokens, total_q_next_tokens, q_len_prev_list, q_len_next_list,
  kv_len_prev_list, kv_len_next_list` (tuples of int) + property `local_tokens = num_tokens // cp_size`.
* `_MaxFlowEdge` `:134-141`, `_MaxFlow` `:143-195` **[CORE]** — small CPU Dinic max-flow
  (`add_edge`, `max_flow(source, sink, limit)`); no device work.
* `_decompose_remainder_rows(remainder, row_count, cp_size, column_caps)` `:198-228` **[CORE]** — splits aggregate
  pair capacities back into per-sequence rows, ≤2 extra blocks per row, returns `None` if infeasible.
* `_allocate_remainder_extras(remainders, cp_size)` `:231-300` **[CORE]** — asserts `sum(remainders) % cp_size == 0`
  (`:250-254`), builds `source → remainder-groups → cp columns → sink` capacities, falls back to
  `_individual_remainder_extras` on infeasible/undecomposable flow (`:277, :294`).
* `_individual_remainder_extras(remainders, cp_size)` `:303-336` **[CORE]** — exact per-row max-flow; raises
  `AssertionError("unable to balance zigzag remainder extras …")` when impossible.
* `_balanced_zigzag_blocks(query_lens, cp_size, num_tokens_pad)` `:339-396` **[CORE]** — appends global padding to
  the **last** request (`:358-361`), `base = qlen // (2*cp)`, `remainder = qlen % (2*cp)`, extra block `k` of rank
  `r` goes to `blocks[r]` and (for 2 extras) `blocks[2cp-1-r]` (`:367-382`), asserts per-rank total equals
  `num_tokens_pad // cp_size` (`:384-394`). Returns `(effective_query_lens, block_sizes)`.
* `build_zigzag_plan(query_lens, prefix_lens, cp_size, cp_rank, num_tokens_pad, num_actual_tokens=None)`
  `:399-539` **[CORE]** — validates non-empty batch, equal prefix/query lengths, `cp_size > 1`,
  `num_tokens_pad % cp_size == 0`, `num_actual_tokens == sum(query_lens) ∈ [0, num_tokens_pad]`
  (`:433-449`); builds `block_starts` (`:455-467`), `zigzag_index` = this rank's prev blocks then next blocks
  (`:471-483`), `gather_positions` = per-rank prev+next blocks (`:486-498`), `inv_positions`
  (`:500-502`), prev/next q splits (`:504-511`), and per-request causal KV lens
  `kv_prev[s] = prefix[s] + sum(blocks[:rank+1])`, `kv_next[s] = prefix[s] + sum(blocks[:2cp-rank])` (`:513-524`).
* `zigzag_ineligible_reason(attn_state, num_tokens_pad, cp_size, query_lens, prefix_lens=None, is_prefilling=None,
  num_actual_tokens=None, *, speculative=False, v2_model_runner=False, dp_size=1, dcp_replicated=False,
  full_o_proj=True)` `:542-618` **[CORE]** — the single eligibility predicate. Returns `None` or the **first**
  failing gate name: `flag_off`, `cp_size<=1`, `draft`, `v2_model_runner`, `dp>1`, `dcp_replicated`,
  `o_proj_not_full`, `state=<name>`, `no_query_lens`, `empty_batch`, `prefix_len_mismatch`, `negative_prefix`,
  `query_len<{2*cp_size}`, `actual>pad`, `actual<min(N)`, `pad%cp_size!=0`, `is_prefilling_missing(<state>)`,
  `is_prefilling_len_mismatch`, `not_all_prefilling` (lines `:563-618`).
* `zigzag_shard_tensor(x, dim=0)` `:621-627` **[CORE]** — natural-order full tensor → `x[ctx.zigzag_index]`;
  asserts ctx/index present; `dim != 0` raises `NotImplementedError`.
* `zigzag_gather_tensor(x)` `:630-652` **[CORE]** — `tensor_model_parallel_all_gather(x, 0)` (skipped when TP==1)
  → `[inv_gather_index]` → drops the trailing `get_forward_context().pad_size` rows, mirroring
  `NPUModelRunner._all_gather_hidden_states` (`model_runner_v1.py:2527-2533`).
* `zigzag_gather_hidden_states_list(list)` `:655-656`, `zigzag_gather_hidden_states_and_aux(hidden_states)`
  `:659-665` **[CORE]** — same for `(hidden, [aux…])` tuples.

Invariants: indices are CPU-computed then moved to device (`sfa_v1.py:315-317`, dtype `int64`); all three vectors
must come from one `build_zigzag_plan` call; padding rows are part of the permutation and are skipped downstream by
`slot_mapping == -1` / `mc2_mask == False`; `zigzag_reorder_moe_aux` requires the full padded length `T`.

### 3.2 `vllm_ascend/attention/sfa_v1.py` — [MOD], +739/−? (the biggest change)

New module-level code:
* `_sfa_5_3_scope(name)` `:88-103` **[CORE, profiler-only]** — decorator wrapping with
  `record_function_or_nullcontext(name)` (a no-op unless `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1`). Applied to
  `rope_single:1396`, `_handle_o_proj_weight_switch_and_forward:1502`, `_forward_o_proj_tp:1549`, `exec_kv:1604`,
  `_q_proj_and_k_up_proj:1666`, `_v_up_proj:1693`, `indexer_select_pre_process:1901`,
  `indexer_select_post_process:2013`, `_execute_sparse_flash_attention_process:2079`,
  `_maybe_gather_kv_for_dsacp:2111`, `_maybe_store_kvcache_for_c8_n_dsacp:2182`, `forward:2370`,
  and module-level `custom_kv_rmsnorm_rope:2718`; four inline `record_function_or_nullcontext` scopes were also added
  inside `forward`: `SFA-5.3/01_fused_qkv_a_proj_split_qnorm` `:2480`, `SFA-5.3/06_indexer_cache_write` `:2563`,
  `SFA-5.3/07_topk` `:2646`, `SFA-5.3/09_o_proj_default` `:2707`.
* `DSACPContext` `:203-239` (existing dataclass, 8 original fields) — **[CORE]** 10 added zigzag fields, all
  `torch.Tensor | None = None`: `zigzag_index:219`, `zigzag_gather_index:220`, `inv_gather_index:221`,
  `actual_seq_lengths_query_zigzag:226`, `actual_seq_lengths_key_zigzag:227`, `block_table_zigzag:228`,
  `slot_mapping_cp_gathered:231`, `fallback_slot_mapping_cp:237`, `fallback_cos:238`, `fallback_sin:239`.
* `_build_zigzag_meta(num_tokens, cp_size, cp_rank, query_lens, prefix_lens, device, num_actual_tokens=None,
  num_reqs_meta=None)` `:242-328` **[CORE]** — calls `build_zigzag_plan` (`:271-278`), re-asserts the prev/next split
  (`:279-282`), right-pads the four length lists with zeros so their length equals the block-table row count
  (`:287-291`), and returns the meta dict with device tensors: three `int64` index vectors (`:315-317`) plus
  `actual_seq_lengths_query_zigzag` (int32, cumulative over `[all prevs, all nexts]`) and
  `actual_seq_lengths_key_zigzag` (int32, raw per-batch KV lengths) (`:301-314`), plus the CPU lists for the debug
  log (`:319-327`). Inner helper `_int64_tensor` `:293-294`.
* `_cumsum_list(values)` `:331-336` **[CORE]** — Python cumsum, no device work.

`AscendSFAMetadataBuilder` (builder, i.e. metadata side):
* `build(..., **kwargs)` `:461-478` **[HOOK]** — pops `for_draft` from `**kwargs` (`:474`) and forwards it to
  `_build` (`:477`). `build_draft_attn_metadata` calls `build()` directly, so the sentinel `draft_index` cannot
  identify it; `for_draft=True` is therefore passed by the spec-decode proposer.
* `build_for_drafting(...)` `:480-492` — unchanged behaviour, still passes `draft_index` (`:490`), which the gate
  maps to `speculative=True`.
* `_build_with_metadata_view` `:494-504` — **[HOOK, no change]** but is the extension point overridden by
  `AscendSFADCPMetadataBuilder._build_with_metadata_view` (`attention/context_parallel/sfa_cp.py:285-360`), which
  **rewrites `dsa_cp_context.slot_mapping_cp` afterwards** (`sfa_cp.py:250-269`). Zigzag is excluded in that path
  only by the `dcp_replicated` gate.
* `_build(..., draft_index=None, for_draft=False)` `:506-880` **[CORE]**:
  * CPU per-request extraction `:534-569`: `query_lens_cpu` / `prefix_lens_cpu` / `is_prefilling_cpu` /
    `real_req_indices` from `query_start_loc_cpu`, `seq_lens_cpu`, `is_prefilling_cpu`; any length mismatch sets
    `real_req_indices = []` so the gate reports `no_query_lens` (never indexes out of range).
  * DSA-CP branch `:579-825`: keeps the old continuous slice as `slot_mapping_cp_continuous` `:593`; gate call
    `:604-622` (passes `speculative=draft_index is not None or for_draft` `:617`, `v2_model_runner=` `:618`,
    `dp_size=` `:619`, `dcp_replicated=enable_sfa_dcp_replicated_indexer(...)` `:620`,
    `full_o_proj=dsa_cp_with_o_proj_tp_for_config(...)` `:621`); plan build wrapped in
    `try/except (ValueError, AssertionError, RuntimeError)` → `warning_once` + `zigzag=None` +
    `zigzag_gate=f"plan_error:{type(exc).__name__}"` `:636-646`; `block_table_zigzag` `:648-668`
    (`index_select(0, real_req_indices)` → append `zeros_like` rows for padded slots → `cat([t, t])` = `2*num_reqs`
    rows); zigzag RoPE via `get_cos_and_sin_mla(input_positions[zigzag_index], use_cache=False)` plus a second
    continuous copy kept as fallback `:670-701`; `slot_mapping_cp = slot_mapping[zigzag_index]` `:713`; debug
    `[CP_BALANCE][plan]` log `:714-732`; non-zigzag branch keeps the original cos/sin pad+slice `:688-701,733-736`;
    the pre-existing shape asserts `:738-749`; `actual_seq_lengths_query/key` are still computed for the continuous
    layout `:751-786`; `DSACPContext(...)` construction `:788-825` (only construction site in the repo).
  * `[CP_BALANCE][branch]` log `:831-851` (rank / branch / reason / state / pad / actual / reqs / real / min_qlen /
    local / min_tokens) — one line per branch, so "which path ran" is answered by the log itself.
  * `store_kv_block_metadata` call `:853-860` unchanged.
* `AscendSFAMetadata` `:340-379` — not modified; `cos`/`sin` (`:361-362`) and `dsa_cp_context` (`:371`) are the
  fields the feature reads/mutates externally.

`AscendSFAImpl` (impl, i.e. per-layer forward):
* `_indexer_qk_proj(q_c, cos, sin, output_dtype)` `:1951-2011` **[CORE, refactor]** — extracted from
  `indexer_select_post_process` so the continuous and zigzag top-k paths share the projection/RoPE/quant code.
* `indexer_select_post_process(..., block_table: torch.Tensor | None = None)` `:2014-2053` **[CORE]** — computes
  `kw/weights`, calls `_indexer_qk_proj` (`:2036-2038`), forwards `block_table=block_table` to
  `DeviceOperator.indexer_select_post_process` (`:2052`).
* `_execute_sparse_flash_attention_process(..., block_table=None)` `:2080-2101` **[HOOK]** — forwards the kwarg to
  `DeviceOperator` (`:2100`).
* `_maybe_store_kvcache_for_c8_n_dsacp(...)` `:2183-2308` **[CORE]** — inside the `enable_dsa_cp` branch
  (`:2227-2306`): if `dsa_cp_context.zigzag_gather_index is not None` (`:2249-2252`) use
  `scatter_slots = slot_mapping_cp_gathered` and the **full** gathered/padded `fused_kv_no_split` (`:2259-2261`);
  otherwise the original `[…][:num_actual_tokens]` truncation (`:2262-2264`). Then the C8 scatter (`:2265-2272`),
  the non-indexer split (`:2273-2277`), the `k_li` split which deliberately keeps the **full** tensor
  (`:2283-2289`, comment) and the non-C8 `DeviceOperator.reshape_and_cache(key=k_nope, value=k_pe,
  slot_mapping=scatter_slots)` now passing untruncated tensors (`:2295-2306`).
* `forward(...)` `:2371-2715` **[CORE]**:
  * `slot_mapping_cp`, `actual_seq_lengths_query/key` come from the ctx before the branch `:2406-2413`;
  * `zigzag_active = self.enable_dsa_cp and attn_metadata.dsa_cp_context is not None and
    …dsa_cp_context.zigzag_index is not None and bool(_EXTRA_CTX.zigzag_cp_active)` `:2493-2498` (set only in the
    native preprocess branch, and prefill forces `fused_type = NATIVE` at `:2432-2438`, so every eligible batch
    reaches it);
  * indexer index-cache write `:2562-2616`: `use_li_c8_reshape_optim = self._use_li_c8_reshape_optim() and not
    zigzag_active` (`:2567`) and `idx_slots = slot_mapping` replaced by `slot_mapping_cp_gathered` when the ctx has
    `zigzag_gather_index` (`:2571-2581`), then `npu_scatter_nd_update_` (`:2593-2597, 2611-2615`);
  * merged single-call metadata `:2626-2644`: when `zigzag_active`, `query_lens_arg =
    ctx.actual_seq_lengths_query_zigzag`, `key_lens_arg = ctx.actual_seq_lengths_key_zigzag`,
    `block_table_arg = ctx.block_table_zigzag` (asserts `:2633-2637`), else the continuous tensors and
    `block_table_arg = None`;
  * `indexer_select_post_process(..., block_table=block_table_arg)` `:2653-2663` and
    `_execute_sparse_flash_attention_process(..., block_table=block_table_arg)` `:2667-2676`;
  * o_proj unchanged; the comment at `:2694-2695,2710-2711` records that zigzag output rows stay rank-local and the
    model boundary does the single gather+rerange.
  * `npu_transpose_batchmatmul` fused `q_nope` path in `_q_proj_and_k_up_proj` `:1666-1692` is **byte-identical to
    base** after commit `9e462094b` (the doc `docs/cp_balance_walkthrough.md` §8.7 refers to this).

Contracts: reads `common_attn_metadata` fields `num_reqs, num_actual_tokens, num_input_tokens, block_table_tensor,
slot_mapping, positions, query_start_loc, query_start_loc_cpu, _seq_lens_cpu` (or `seq_lens_cpu`),
`is_prefilling_cpu, attn_state, group_len/group_key_idx/group_key_cache_idx` (`:512-517, 522-532, 541-569, 605`);
writes `_EXTRA_CTX` read-only (`:2497`); calls `vllm_ascend.layers.cp_zigzag.{build_zigzag_plan,
zigzag_ineligible_reason}`, `vllm_ascend.ops.rotary_embedding.get_cos_and_sin_mla`,
`vllm_ascend.utils.dsa_cp_with_o_proj_tp_for_config`, `vllm.envs.VLLM_USE_V2_MODEL_RUNNER` (as `envs_vllm:618`).

### 3.3 `vllm_ascend/ascend_forward_context.py` — [MOD] (decisive per-forward state)

* `import … zigzag_reorder_moe_aux` `:16` **[HOOK]**.
* `_iter_attn_metadata(attn_metadata)` `:70-86` **[CORE]** — yields metadata objects from a `{layer: md}` dict, a
  list of such dicts (ubatching) or a bare object.
* `_find_zigzag_cp_context(attn_metadata)` `:89-102` **[CORE]** — first metadata whose `dsa_cp_context.zigzag_index`
  is not None.
* `_disable_zigzag_metadata_for_fallback(attn_metadata)` `:105-151` **[CORE]** — for every zigzag metadata object:
  raises `RuntimeError` if any of `fallback_slot_mapping_cp/fallback_cos/fallback_sin` is missing (`:126-134`),
  otherwise restores `ctx.slot_mapping_cp`, `meta.cos`, `meta.sin` and clears the 7 zigzag ctx fields plus the three
  `fallback_*` fields (`:136-151`).
* `set_ascend_forward_context(...)` `:182-366` **[CORE hooks]**:
  * `:219-233` compute `zigzag_cp_context = _find_zigzag_cp_context(attn_metadata)` and
    `zigzag_cp_active = ctx is not None and not is_draft_model and not VLLM_USE_V2_MODEL_RUNNER`; if inactive but a
    ctx was found, call the fallback restorer for `attn_metadata` **and** every entry of `draft_attn_metadatas`
    (`:229-233`). Note this runs before the metadata is used but after it was built.
  * `:306-310` second veto: `get_dp_group().world_size > 1` → restore fallback, `zigzag_cp_active = False`.
  * `:330-331` publish `forward_context.zigzag_cp_context` (write-only) and `forward_context.zigzag_cp_active`.
  * `:333-342` when active: `input_ids = input_ids.to(torch.int64)` then
    `zigzag_reorder_moe_aux(input_ids, ctx)` and store in `forward_context.input_ids` (the model still receives the
    natural-order ids argument); `:354-361` reorder `mc2_mask` with the same permutation (inside
    `reserved_mc2_mask[:padded_num_tokens]`, after `mc2_mask[:num_actual_tokens] = True`).
* `_ExtraForwardContextProxy.extra_attrs` `:543-569` **[HOOK]** — adds `"zigzag_cp_active"`, `"zigzag_cp_context"`
  (`:567-568`) so the proxy lets them through (`check_extra_attr` rejects unknown names).
* `zigzag_active()` `:604-616` **[CORE]** — `bool(_EXTRA_CTX.zigzag_cp_active)` with `except Exception → False`.
  This is the **only** gate for every non-metadata cp_balance change.

Contracts: `vllm.forward_context.{BatchDescriptor, get_forward_context, set_forward_context}`; reads
`envs_vllm.VLLM_USE_V2_MODEL_RUNNER` (`:223`), `get_dp_group()`, `get_tensor_model_parallel_world_size()`; writes
on the vllm `ForwardContext` dataclass (which has `additional_kwargs: dict`, `vllm/forward_context.py:188`):
`input_ids`, `mc2_mask`, `zigzag_cp_active`, `zigzag_cp_context`; reads `pad_size`/`padded_num_tokens` it sets
itself (`:282-285, 311-318, 348`).

Invariants: `padded_num_tokens == num_tokens_padded == len(mc2_mask) == len(input_ids) == plan.num_tokens`
(DP>1 is excluded, so `ceil(max_tokens_across_dp/tp)*tp == num_tokens`); the fallback path must never be reached
without all three `fallback_*` tensors (hard error, no silent degradation); `input_ids` and `mc2_mask` must use the
same permutation.

### 3.4 `vllm_ascend/distributed/utils.py` — [MOD], +133 lines

* `reduce_mode()` `:24-27` **[CORE]** — `@lru_cache(maxsize=1)` wrapper over
  `ascend_envs.VLLM_ASCEND_CP_BALANCE_REDUCE_MODE` (read once per process).
* `_allreduce_slice_reduce_scatter(tensor, group)` `:30-52` **[CORE]** — **in-place**
  `dist.all_reduce(summed, group.device_group)` on the caller's tensor (contiguous copy only if needed), then
  returns the local `[rank*chunk:(rank+1)*chunk].contiguous()`.
* `_plain_reduce_scatter(tensor, group)` `:54-71` **[CORE]** — the original `dist.reduce_scatter_tensor`
  (owner-dependent rounding) + output buffer; baseline debugging only.
* `_all_to_all_fixed_order_reduce_scatter(tensor, group)` `:73-92` **[CORE]** — `all_to_all_single` of chunks then
  `fixed_order_rank_sum([recv[src] for src in range(world_size)])`; lower volume, A/B only.
* `fixed_order_reduce_scatter(tensor, group)` `:95-148` **[CORE]** — returns `tensor` for `world_size<=1` or 0 rows;
  raises `ValueError` when `rows % world_size != 0`; dispatches on `reduce_mode()` with aliases
  (`allreduce|all_reduce|ar`, `alltoall|all_to_all|a2a`, `reducescatter|reduce_scatter|rs`) and raises
  `ValueError` for anything else; `[CP_BALANCE][reduce]` `info_once` log at `:130-136`.
* imports `fixed_order_rank_sum` from `vllm_ascend.layers.cp_zigzag` `:11` **[HOOK]**.

Invariants: every rank must arrange the tensor in the **same row order** and own exactly `rows/world_size` rows;
the tensor must not be read again by the caller (in-place all-reduce); `chunk` slicing assumes contiguous rows.

### 3.5 `vllm_ascend/ops/linear_op.py` — [MOD], +62 lines

* imports `zigzag_active` (`:58`) and `fixed_order_reduce_scatter` (`:64`) **[HOOK]**.
* `MLPRowParallelOp.apply_impl` `:196-217` **[CORE]** — `output=None`; if `zigzag_active()` try
  `fixed_order_reduce_scatter(output_parallel, self.comm_group)` inside `except Exception → None` (`:203-210`); on
  `None` log `path=native site=mlp` (debug) and run the original `self.comm_group.reduce_scatter(output_parallel, 0)`
  (`:211-214`).
* `SequenceRowParallelOp._fixed_order_reduce_scatter(output_parallel)` `:351-384` **[CORE]** — guards
  `tp_size <= 1` or `rows % tp_size != 0` → original `tensor_model_parallel_reduce_scatter`; otherwise
  `fixed_order_reduce_scatter` with `except (RuntimeError, NotImplementedError, TypeError, ValueError) →
  warning_once + original`.
* `SequenceRowParallelOp.matmul_and_reduce` `:386-471` **[CORE hook]** — the unquantized `else` branch
  (`:460-469`) uses the fixed-order helper when `zigzag_active()`, otherwise the original reduce_scatter.
  **Not** routed: `mmrs_fusion` `npu_mm_reduce_scatter_base` branches `:418-459` and the
  `not flash_comm_v1_enabled` early return at `:401-403` (`tensor_model_parallel_all_reduce`).
* `OProjRowParallelOp.apply_impl` `:261-295` — **unchanged**, bare `self.comm_group.reduce_scatter(output_parallel,
  dim=0)` at `:289`; only reachable with fine-grained OTP (`oproj_tp_enable()`), which is mutually exclusive with
  this DSA-CP config but remains an owner-dependent reduction on the zigzag path if enabled.

### 3.6 `vllm_ascend/ops/register_custom_ops.py` — [MOD], +36/−?

* `_fixed_order_zigzag_reduce_scatter(x)` `:25-48` **[CORE]** — returns `None` (never raises) when not
  `zigzag_active()`, when `x.shape[0] % tp_world_size != 0`, or on any exception (`warning_once`);
  otherwise `fixed_order_reduce_scatter(x, get_tp_group())`. Lazy import of `distributed.utils` (`:34`) to avoid a
  cycle.
* `_maybe_pad_and_reduce_impl(x, is_ep_comm=False)` `:101-124` **[CORE hook]** — after the existing `F.pad(x, …)`
  (`:117-118`) tries the helper (`:119-121`), logs `path=native site=pad_and_reduce` (debug) and falls back to
  `tensor_model_parallel_reduce_scatter(x, 0)` (`:124`). This single site serves both the embedding
  (`vocab_parallel_embedding._forward_origin:287`) and MoE finalize.

### 3.7 `vllm_ascend/ops/vocab_parallel_embedding.py` — [MOD], +45/−?

* `AscendVocabParallelEmbedding.forward_zigzag_local(input_)` `:170-201` **[CORE, experimental]** —
  raises `NotImplementedError` when `self.forward_type == "embed_tp"` (`:190-194`); TP==1 safety net (`:197-199`);
  else `tensor_model_parallel_all_reduce(self._embed_partial(input_))` then `zigzag_shard_tensor(output)`
  (`:200-201`, lazy import at `:195`). Exists as an A/B against the default gather+shard entry.
* `_embed_partial(input_)` `:262-281` **[CORE, refactor]** — vocabulary-shard mask + `quant_method.embedding` +
  `masked_fill_`, no TP reduction.
* `_forward_origin(input_)` `:283-288` **[HOOK]** — now `_embed_partial` + `torch.ops.vllm.maybe_pad_and_reduce`
  (unchanged semantics).
* New import `tensor_model_parallel_all_reduce` from `vllm.distributed` (`:23`).

### 3.8 `vllm_ascend/patch/worker/patch_deepseek_v2.py` — [MOD], +80/−?

Patches only `DeepseekV2Model.forward` (`_patched_forward:305`, assignment `:437`). `DeepseekV2DecoderLayer.forward`
is **deliberately not patched** any more (see the comment at `:429-436`: the upstream SP branches are guarded by
`use_sequence_parallel_moe`, which requires `data_parallel_size > 1`, and the zigzag gate rejects DP>1).

* `:311-314` local import of `_EXTRA_CTX`, `zigzag_active = bool(_EXTRA_CTX.zigzag_cp_active)` (direct read; no
  exception guard — unlike `zigzag_active()`).
* `:316-324` `use_embed_local = zigzag_active and VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL and
  hasattr(self.embed_tokens, "forward_zigzag_local")`; `tp_size`, `tp_rank`,
  `expected_local_tokens = positions.shape[0] // tp_size` (asserted after every shard).
* `:335-353` experimental entry: requires `input_ids.shape[0] == positions.shape[0]` (else `RuntimeError`),
  calls `self.embed_tokens.forward_zigzag_local(input_ids)`, sets `hidden_is_zigzag_local`.
* `:350-368` default entry: `hidden_states = self.embed_input_ids(input_ids)`; when zigzag and the result is not
  already full length (`shape[0] != positions.shape[0]`) → `tensor_model_parallel_all_gather(hidden_states, 0)`
  (the FlashComm embedding returned a reduce-scattered local slice) → `zigzag_shard_tensor(hidden_states)`; row
  count checked against `expected_local_tokens`.
* `:375-381` `positions = zigzag_shard_tensor(positions)` (mandatory: the exit path infers "already gathered" from
  `hidden.shape[0] == positions.shape[0]`); row count checked.
* `:401` `if not zigzag_active and aux_hidden_state.shape[0] != positions.shape[0]` and `:410`
  `if not zigzag_active and hidden_states.shape[0] != positions.shape[0]` — the two in-model gather sites are
  skipped under zigzag (the runner does the single gather+rerange instead).
* `:419-426` `self.norm` unchanged; returns rank-local tensors.

Imports: `vllm.distributed.{get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather}`,
`vllm.distributed.parallel_state.get_tp_group` (`:12`), `vllm_ascend.envs`, `vllm_ascend.layers.cp_zigzag.
zigzag_shard_tensor` (`:37`).

### 3.9 `vllm_ascend/worker/model_runner_v1.py` — [MOD], small hooks only

* import `zigzag_gather_hidden_states_and_aux` (`:185`) **[HOOK]**.
* `_model_forward` exit gather `:2602-2609` **[CORE hook]** — condition becomes `flash_comm_v1_enabled or
  getattr(forward_context, "zigzag_cp_active", False)`; when zigzag → `zigzag_gather_hidden_states_and_aux(
  hidden_states)`, else the original `self._all_gather_hidden_states_and_aux(...)`.
* `_pad_for_sequence_parallelism(num_scheduled_tokens)` `:2612-2623` — **[HOOK, comment only]** the rule stays
  `round_up(num_scheduled_tokens, tp_size)` when `enable_sp(...)`. This is the site that guarantees
  `num_tokens_padded % cp_size == 0`; the doc `docs/cp_balance_walkthrough.md` §2 claim that the feature "deleted an
  unused second parameter" is **wrong for this commit set** — the signature is identical in `dd13a7f29`,
  `ae9c13a42`, `c7990e5e4` and `23ff2c23c`.
* `AscendCommonAttentionMetadata(...)` construction `:2905` **[HOOK]** — adds `is_prefilling_cpu=is_prefilling`
  (the same CPU tensor built at `:2874`, `num_computed_tokens_cpu < num_prompt_tokens_cpu`, tail forced False).

### 3.10 `vllm_ascend/attention/utils.py` — [MOD], +6 lines

* `AscendCommonAttentionMetadata.is_prefilling_cpu: torch.Tensor = None` `:229-232` **[CORE field]** — host-side
  copy of `is_prefilling`, documented to avoid a device→host sync in the metadata hot path.
* `unpadded()` `:296` **[HOOK]** — `is_prefilling_cpu=_slice_reqs(self.is_prefilling_cpu)` (missing this slice makes
  draft/unpadded batches report `is_prefilling_missing(...)` and silently fall back to continuous).

### 3.11 `vllm_ascend/device/device_op.py` — [MOD], +39/−?

* `_sfa_5_3_scope(name)` `:49-60` **[CORE, profiler-only]** — same decorator as in `sfa_v1.py`.
* `BaseDeviceAdaptor.indexer_select_post_process(..., block_table: torch.Tensor | None = None)` `:476` (decorator
  `:475`, default `:488`) **[CORE]** — `if block_table is None: block_table = attn_metadata.block_table`
  (`:489-490`), and all three kernel calls (`npu_lightning_indexer` `:511`, the two lightning-indexer variants
  `:526, :539`) now receive `block_table=block_table` instead of `attn_metadata.block_table`.
* `BaseDeviceAdaptor.execute_sparse_flash_attention_process(...)` `:549` (decorator `:548`) **[HOOK]** — already had a
  keyword-only `block_table: torch.Tensor | None = None` (`:560`) with `if block_table is None: block_table =
  attn_metadata.block_table` (`:564-565`); only the decorator (`:548`) is new.
* A5 adaptor mirrors: `indexer_select_post_process` `:1678` (decorator `:1677`) with `block_table` default `:1690` and
  the three call sites updated (`:1713, :1728, :1741`); decorators on `_execute_kv_quant_sparse_flash_attention` (`:620`, `:1024`) and the A5
  indexer post-process (`:1677`).

### 3.12 `vllm_ascend/ops/fused_moe/experts_selector.py` — [MOD], comment only

* `_select_experts_with_fusion_ops` `:249-255` **[HOOK, comment]** — documents that `forward_context.input_ids`
  has already been converted to `int64` and reordered; no behaviour change. This is the only consumer of
  `forward_context.input_ids` (hash routing, `scoring_func == "sqrtsoftplus"`, `tid2eid is not None`).

### 3.13 `vllm_ascend/utils.py` — [MOD], refactor only

* `enable_dsa_cp_for_config(vllm_config)` `:1372-1390` **[CORE, refactor]** — body of the old `enable_dsa_cp`, but
  takes an explicit config and calls `enable_sp(vllm_config)` (`:1386, :1390`).
* `enable_dsa_cp()` `:1393-1398` — now an `lru_cache` wrapper reading the current config.
* `dsa_cp_with_o_proj_tp_for_config(vllm_config)` `:1401-1415` **[CORE, refactor]** — the "prefill node keeps the
  full TP o_proj weight" predicate (`kv_transfer_config is None or is_kv_producer`); used by the zigzag gate.
* `enable_dsa_cp_with_o_proj_tp()` `:1418-1425` — `lru_cache` wrapper.
No other semantics changed.

### 3.14 `vllm_ascend/envs.py` — [MOD], +32 lines

Five new entries in `env_variables` (`:85, :90, :96-98, :102-104, :109-111`), see §2. Uses the module's existing
lazy `__getattr__` (`:140-147`), i.e. every read re-parses the environment.

### 3.15 `vllm_ascend/layers/__init__.py` — [NEW], 1 line

`# SPDX-License-Identifier: Apache-2.0` — makes `vllm_ascend.layers` a package so `layers.cp_zigzag` is importable.

### 3.16 `vllm_ascend/spec_decode/llm_base_proposer.py` — [MOD], 9 lines

* `build_draft_attn_metadata` fallback branch `:2132-2140` **[CORE hook]** — `builder.build(0, common_attn_metadata,
  self.runner.get_model(), for_draft=True, **extra_attn_metadata_args)`; comment explains that the drafter's token
  tensors do not follow the target batch's plan. The `draft_index=1` branch above (`:2129`) is already covered by
  the `speculative` gate.

### 3.17 `vllm_ascend/patch/worker/patch_deepseek_mtp.py` (+ `vllm_ascend/patch/__init__.py`) — [MOD], co-shipped, NOT zigzag

Shipped inside `d22ddb2ef` to allow running a reduced GLM-5.2 (`num_hidden_layers=3`) against the full checkpoint
(used for single-node debugging of this feature). It is independent of the zigzag layout:

* `_BACKBONE_LAYER_RE = re.compile(r"^(?:model\.)?layers\.(\d+)\.")` `:17`.
* `_EXTRA_LAYER_FILTER` contextvar `:19-21`; `_make_extra_layer_predicate(num_hidden_layers)` `:24-29`
  (`idx >= num_hidden_layers` → skip); `_filter_extra_checkpoint_layers(weights, n)` generator `:32-50`;
  `_patched_should_skip_weight` `:53-67` extending `vllm.model_executor.model_loader.weight_utils.should_skip_weight`
  and monkey-patched at module import `:70`.
* `AscendGlmMoeDsaForCausalLM.load_weights` `:130-143` — filters weights and installs/removes the contextvar
  around `AutoWeightsLoader(self, skip_prefixes=[MTP_ROT_WEIGHT_NAME]).load_weights(weights)`.
* `vllm_ascend/patch/__init__.py:628-648` — documentation-only update for that patch.

### 3.18 `tests/ut/ops/test_vocab_parallel_embedding.py` — [MOD], +46 lines (new test at `:204-249`)

`test_forward_zigzag_local_reduces_the_full_row_set` pins the `VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL` entry:
with `layer.tp_size = 2` and a 4-row full token-id stream it asserts (a) the vocab-sharded embedding lookup is
called **once with all 4 rows** (`embedding_inputs == [(4,)]`), (b) `tensor_model_parallel_all_reduce` is called
**once with shape `(4, embedding_dim)`** (patched at
`vllm_ascend.ops.vocab_parallel_embedding.tensor_model_parallel_all_reduce`), (c) `zigzag_shard_tensor` (patched at
`vllm_ascend.layers.cp_zigzag.zigzag_shard_tensor`, which works because the production import is lazy) is called
once and the result has 2 rows. It is the only regression net for the invariant "never all-reduce partials of
rank-local ids".

### 3.19 `tests/ut/patch/worker/test_patch_deepseek_mtp.py` — [NEW], 49 lines, 3 tests

Pure-CPU tests of §3.17 only: `test_layer_predicate_skips_only_unconfigured_layers` `:11-19` (skips
`model.layers.3.…`, `model.layers.78.…`, `layers.3.…`; keeps `model.layers.2.…`, `model.embed_tokens.weight`,
`rot.weight`), `test_filter_keeps_weight_order_and_non_layer_weights` `:22-37` (order-preserving generator),
`test_patched_should_skip_weight_respects_layer_slice_context` `:40-48` (contextvar active/reset semantics).
Nothing here touches zigzag.

### 3.20 Not part of the feature

`AGENTS.md` (−419) and `CLAUDE.md` (−1) are deleted in `d22ddb2ef` but are repository documentation.

### 3.21 Gap: no test covers the layout math

`build_zigzag_plan`, `_allocate_remainder_extras` / `_MaxFlow`, the identity
`inv_gather_index[zigzag_gather_index[p]] == p`, `slot_mapping_cp[zigzag_index]` and the merged
`2*B` metadata have **no unit test** anywhere in the repo. The only automated local checks are the static,
harness-side AST scripts in `cp_balance/`: `cp_balance/perf/check_cp_balance_fields.py` (ZigzagPlan ↔ meta dict ↔
DSACPContext field agreement), `cp_balance/accuracy/check_b_path.py` (asserts exactly two `if zigzag_active():` in
`linear_op.py`, the pad_and_reduce gate, that bare `VLLM_ASCEND_CP_BALANCE` is read only in `envs.py` /
`layers/cp_zigzag.py`, and that the fused `q_up_proj` block is byte-identical to the base branch), and
`cp_balance/accuracy/check_branch.py` (parses the `[CP_BALANCE][branch]` log lines). A re-implementation should
either port `check_cp_balance_fields.py` or write the missing unit tests for the plan.

---

## 4. Cross-module data contracts

### 4.1 `DSACPContext` (`sfa_v1.py:203-239`)

| field | set where (23ff2c23c) | read where |
| --- | --- | --- |
| `num_tokens, num_tokens_pad, local_start, local_end, local_end_with_pad, slot_mapping_cp, actual_seq_lengths_query, actual_seq_lengths_key` | `sfa_v1.py:788-796` (only constructor); `slot_mapping_cp` also mutated by `sfa_cp.py:269` (DCP path) and by the fallback `ascend_forward_context.py:136` | `sfa_v1.py:2406-2413, 2622-2623`; `sfa_cp.py:261-269, 614-639, 813` |
| `zigzag_index` | `sfa_v1.py:797` | `ascend_forward_context.py:100,120,142`; `sfa_v1.py:2496`; `cp_zigzag.py:626` |
| `zigzag_gather_index` | `sfa_v1.py:798-800` | `cp_zigzag.py:92` (via `zigzag_reorder_moe_aux`), `sfa_v1.py:2251, 2574` |
| `inv_gather_index` | `sfa_v1.py:801-803` | `cp_zigzag.py:640,645` |
| `actual_seq_lengths_query_zigzag` | `sfa_v1.py:804-808` | `sfa_v1.py:2635,2638` |
| `actual_seq_lengths_key_zigzag` | `sfa_v1.py:809-813` | `sfa_v1.py:2636,2639` |
| `block_table_zigzag` | `sfa_v1.py:814` (built `:648-668`) | `sfa_v1.py:2637,2640` |
| `slot_mapping_cp_gathered` | `sfa_v1.py:815-819` | `sfa_v1.py:2259-2260, 2580-2581` |
| `fallback_slot_mapping_cp, fallback_cos, fallback_sin` | `sfa_v1.py:820-824` | `ascend_forward_context.py:123-138` (cleared `:149-151`) |

Shapes/dtypes: `slot_mapping_cp` `[L]` (same dtype as `slot_mapping`); `slot_mapping_cp_gathered` `[T]`;
`zigzag_index` `[L]` int64; `zigzag_gather_index`/`inv_gather_index` `[T]` int64; the two merged length tensors
`[2*num_reqs]` int32; `block_table_zigzag` `[2*num_reqs, num_blocks]` (dtype of `block_table_tensor`).
Notation: `T = num_tokens_pad`, `L = T // cp_size`, `B = number of real requests`.

### 4.2 vLLM `ForwardContext` attributes used (`vllm/forward_context.py:132-188`)

| attribute | written where | read where |
| --- | --- | --- |
| `input_ids` | `ascend_forward_context.py:217` (natural order), overwritten `:342` (reordered, int64) | `experts_selector.py:254` |
| `mc2_mask` | `:362` (reordered `:359-361`) | MoE communication paths (unchanged by this feature) |
| `zigzag_cp_active` | `:331` (reset `:309`) | `ascend_forward_context.py:614`, `patch_deepseek_v2.py:314`, `sfa_v1.py:2497`, `model_runner_v1.py:2604,2606` |
| `zigzag_cp_context` | `:330` | **no reader** in the tree (write-only compatibility field) |
| `pad_size` | `:282,285,318` | `cp_zigzag.py:649` (exit truncation), `register_custom_ops.py:116` |
| `padded_num_tokens` | `:348` | MoE comm (unchanged) |
| `flash_comm_v1_enabled` | `:280` | `model_runner_v1.py:2603`, `linear_op.py:389` (both pre-existing) |

`_EXTRA_CTX` (`_ExtraForwardContextProxy`, `:540-601`) is the only supported access path for the two zigzag flags;
`extra_attrs` (`:543-569`) is an allow-list and `check_extra_attr` raises `AttributeError` for anything else.

### 4.3 `AscendCommonAttentionMetadata` (`attention/utils.py:199-366`)

| field | written where | read where |
| --- | --- | --- |
| `is_prefilling_cpu` (new) | `model_runner_v1.py:2905` (CPU tensor); sliced `attention/utils.py:296` | `sfa_v1.py:554-559, 565-569` |
| `query_start_loc_cpu` | `model_runner_v1.py:2884` (pre-existing field of vLLM's `CommonAttentionMetadata`) | `sfa_v1.py:541-546` |
| `query_start_loc`, `seq_lens`, `_seq_lens_cpu`/`seq_lens_cpu` | `model_runner_v1.py:2883,2885,2890,2893` | `sfa_v1.py:522-532, 769-770, 781` |
| `num_reqs`, `num_actual_tokens`, `num_input_tokens` | `model_runner_v1.py:2897-2898, 2906` (`num_reqs` is **padded** req count) | `sfa_v1.py:512-514, 634, 660, 846` |
| `block_table_tensor`, `slot_mapping`, `positions`, `attn_state` | `:2901-2902, 2908, 2910` | `sfa_v1.py:516-518, 605` |

### 4.4 Changed call signatures / extension points (anything overriding them must follow)

| symbol | change | consequence |
| --- | --- | --- |
| `AscendSFAMetadataBuilder.build(..., **kwargs)` | accepts `for_draft` (`sfa_v1.py:474`) | draft metadata builders that call `build()` must pass `for_draft=True` (`llm_base_proposer.py:2136`) |
| `AscendSFAMetadataBuilder._build(..., for_draft=False)` | new kwarg `:510` | subclasses overriding `_build` must accept it |
| `AscendSFAImpl.indexer_select_post_process(..., block_table=None)` | `:2014-2025` (default `:2024`) | `AscendSFADCPImpl` has no override of this method, so it inherits the new signature |
| `AscendSFAImpl._execute_sparse_flash_attention_process(..., block_table=None)` | `:2080-2090` | **`AscendSFADCPImpl._execute_sparse_flash_attention_process` (`sfa_cp.py:733-742`) does not accept it while `sfa_v1.py:2667-2676` always passes it as a keyword → `TypeError` on the DCP path (`enable_sfa_dcp_replicated_indexer()` true). Fixed only by "never take that path" here; a port must update the override.** |
| `DeviceOperator.indexer_select_post_process(..., block_table=None)` | `device_op.py:488, 1690` | any adaptor override must accept it |
| `DeviceOperator.execute_sparse_flash_attention_process(..., *, block_table=None, …)` | `device_op.py:560` | keyword-only |
| `AscendVocabParallelEmbedding.forward_zigzag_local(input_)` / `_embed_partial(input_)` | `vocab_parallel_embedding.py:170, 262` | `hasattr(self.embed_tokens, "forward_zigzag_local")` is the dispatch test (`patch_deepseek_v2.py:320`) |
| `vllm_ascend.utils.enable_dsa_cp_for_config(config)` / `dsa_cp_with_o_proj_tp_for_config(config)` | `utils.py:1372, 1401` | config-explicit predicates for paths without a config context |
| `vllm_ascend.distributed.utils.fixed_order_reduce_scatter(tensor, group)` / `reduce_mode()` | `distributed/utils.py:95, 25` | owner-independent reduction |
| `vllm_ascend.ascend_forward_context.zigzag_active()` | `:604-616` | the only gate allowed for cp_balance code |
| `vllm_ascend.layers.cp_zigzag` public symbols | `get_zigzag_cp_context, fixed_order_rank_sum, zigzag_reorder_moe_aux, ZigzagPlan, build_zigzag_plan, zigzag_ineligible_reason, zigzag_shard_tensor, zigzag_gather_tensor, zigzag_gather_hidden_states_list, zigzag_gather_hidden_states_and_aux` | imported by `sfa_v1.py:46-49`, `ascend_forward_context.py:16`, `distributed/utils.py:11`, `patch_deepseek_v2.py:37`, `vocab_parallel_embedding.py:195`, `model_runner_v1.py:185` |
| `vllm.model_executor.model_loader.weight_utils.should_skip_weight` | monkey-patched at import (`patch_deepseek_mtp.py:70`) | upstream renames break the layer-slice filter (unrelated to zigzag) |

### 4.5 vLLM symbols the feature depends on

`vllm.forward_context.{get_forward_context, set_forward_context, BatchDescriptor}` and `ForwardContext` attribute
assignment (`additional_kwargs` exists at `vllm/forward_context.py:188`);
`vllm.distributed.{get_tp_group, get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather,
tensor_model_parallel_reduce_scatter, tensor_model_parallel_all_reduce}`;
`vllm.distributed.parallel_state.{GroupCoordinator, get_tp_group}`;
`vllm.v1.utils.record_function_or_nullcontext`; `vllm.logger.logger`;
`vllm.v1.attention.backend.CommonAttentionMetadata` (`query_start_loc_cpu`, `_seq_lens_cpu`, `is_prefilling` —
present in both vLLM `568afb3a13` and `84030bbe3d`);
`vllm.v1.attention.backend.{AttentionBackend, AttentionCGSupport, MLAAttentionImpl}`;
`vllm.v1.kv_cache_interface.AttentionSpec`; `vllm.model_executor.layers.attention.mla_attention.MLACommonMetadataBuilder`;
`vllm.envs.VLLM_USE_V2_MODEL_RUNNER`; `vllm.model_executor.model_loader.weight_utils.should_skip_weight`.

---

## 5. Invariants and failure modes

**I1 — per-rank local row count is exactly `num_tokens_pad / cp_size`.**
Enforced by (a) `_pad_for_sequence_parallelism` rounding to `tp_size` (`model_runner_v1.py:2612-2623`),
(b) the `num_tokens_pad % cp_size == 0` gate (`cp_zigzag.py:604-605`), (c) the per-rank assertion in
`_balanced_zigzag_blocks` (`cp_zigzag.py:384-394`), (d) the `len(zigzag_index) == L` assertion (`:479-483`).
If it breaks: unequal collective shapes → hang or wrong results in every FlashComm collective.

**I2 — the merged `2*B` metadata must match the rank-local row order.** `actual_seq_lengths_query_zigzag` is a
cumulative sum over `[prev of all real reqs, next of all real reqs]`, `actual_seq_lengths_key_zigzag` is raw
per-batch, `block_table_zigzag` rows are `[real rows in that order, real rows again]`; changing the block
distribution in `_balanced_zigzag_blocks` or the row order in `sfa_v1.py:648-668` without re-deriving all three
breaks attention silently.

**I3 — one permutation, four consumers.** `zigzag_gather_index` is used for KV/indexer slot reordering
(`sfa_v1.py:815-819`), MoE `input_ids` and `mc2_mask` (`ascend_forward_context.py:341,359`), and its inverse for
the exit gather (`cp_zigzag.py:645`). Any change to the gather order must change all four together.

**I4 — padding rows are part of the permutation and must stay skippable.** `slot_mapping` padding is `-1`
(`model_runner_v1.py:2853`, `sfa_v1.py:588-590`); KV scatter/`reshape_and_cache` are called with the **full**
`T`-row tensors plus the reordered slots (`sfa_v1.py:2259-2306, 2593-2615`). If padding slots are ever non-negative,
scatter writes garbage into the KV cache.

**I5 — the exit gather must run exactly once and in the runner.** `patch_deepseek_v2.py:401,410` skip the in-model
gathers when zigzag; `model_runner_v1.py:2606-2607` does all-gather + `inv_gather_index` + `pad_size` truncation
(`cp_zigzag.py:641-651`). `positions` must be sharded too (`patch_deepseek_v2.py:375`), otherwise the
`shape[0] != positions.shape[0]` heuristic fires and the tensor is gathered twice.

**I6 — owner-independent reduction.** HCCL `ReduceScatter` rounding depends on the receiving chunk, and zigzag moves
tokens between owners, so every row-parallel reduction on the zigzag path must go through
`fixed_order_reduce_scatter` (default `allreduce`, in-place, rows divisible by world size). Known gaps:
`OProjRowParallelOp.apply_impl` (`linear_op.py:289`, fine-grained OTP only), the `mmrs_fusion`
`npu_mm_reduce_scatter_base` branches (`linear_op.py:418-459`, off for MoE models and `tp_size > 8`), and
`not flash_comm_v1_enabled` early returns (`linear_op.py:401-403`, `register_custom_ops.py:111-112`).

**I7 — non-zigzag forwards are bit-identical to the old code.** Every cp_balance change outside the metadata
builder is gated by `zigzag_active()`; the eligibility predicate is the only reader of the bare
`VLLM_ASCEND_CP_BALANCE` switch. Adding an ungated cp_balance collective is the classic regression this feature
already hit once (fixed by `c11cefb80`).

**I8 — metadata is built before the last vetoes are known.** Draft / V2-runner / DP>1 can only be decided in
`set_ascend_forward_context`, so every zigzag metadata build must also store
`fallback_slot_mapping_cp`/`fallback_cos`/`fallback_sin`; the restorer raises `RuntimeError` when any is missing
(deliberate: no silent degradation, `ascend_forward_context.py:126-134`).

**I9 — shape chain that must hold for a zigzag forward:**
`num_input_tokens == num_tokens_padded == num_tokens_pad == T == positions.shape[0] == input_ids.shape[0] ==
len(mc2_mask) == len(zigzag_gather_index) == len(inv_gather_index)`, and `T % cp_size == 0`. Note `num_reqs` in
`AscendCommonAttentionMetadata` is the **padded** request count; real requests are identified by
`query_start_loc_cpu` deltas > 0 (`sfa_v1.py:541-547`), and padded request slots get zero q/kv lengths and zero
block-table rows (`sfa_v1.py:287-291, 660-665`).

**I10 — dtype/device assumptions.** The three index vectors are `int64` **device** tensors indexed on the device
(`sfa_v1.py:293-294, 315-317`); merged length tensors are `int32`; `block_table_zigzag` keeps the block-table dtype
(int32 in practice); `zigzag_index` is used both as a device index (`slot_mapping[zigzag_index]`,
`input_positions[zigzag_index]`) and, in `zigzag_shard_tensor`, on the model's `hidden_states`/`positions`
(`x[ctx.zigzag_index]`) — so it must be an on-device index vector for the same device as the activations.

**I11 — plan failures degrade to continuous, but only for the plan itself.** `ValueError/AssertionError/
RuntimeError` inside `_build_zigzag_meta` are caught (`sfa_v1.py:636-646`) and the batch runs as continuous DSA-CP;
anything else (`IndexError`, `TypeError`, …) escapes. The DCP-path failure in §4.4 is of that class, but happens
later, in `forward`, where nothing is caught.

**I12 — `VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL=1` is unvalidated** and requires the full token-id stream
(`patch_deepseek_v2.py:339-344`); with `forward_type == "embed_tp"` it raises `NotImplementedError`
(`vocab_parallel_embedding.py:190-194`). Keep it off.

**Hardest porting risks (ranked):**
1. `sfa_v1.py::_build` is a rewrite of the metadata path, not an insertion — the continuous-slice values must keep
   being produced (they are the fallback) while the zigzag plan takes over `slot_mapping_cp`, `cos`/`sin` and
   `DSACPContext`. Dropping any `fallback_*` tensor turns a legal DP>1/draft batch into a `RuntimeError`.
2. The merged `2*B` metadata triple (`actual_seq_lengths_query_zigzag`, `actual_seq_lengths_key_zigzag`,
   `block_table_zigzag`) plus the full-`T` KV writes: they only work together, and a mismatch is silent (wrong
   tokens / wrong KV slots) rather than a crash.
3. The per-forward `zigzag_active()` gate on three reduction sites plus the exit gather: if the target base has a
   different set of row-parallel reduction sites (or new MoE/eplb paths), the owner-independent reduction will be
   incomplete and non-zigzag batches may stop being bit-identical.
4. `_allocate_remainder_extras` (CPU max-flow + deterministic decomposition) is required whenever
   `sum(query_len % (2*cp)) % cp != 0`; a naive greedy replacement is fine numerically but changes the block
   layout and therefore every index vector.
5. Coupling with the model runner padding contract: the gate only requires `num_tokens_pad % cp_size == 0`, which
   is a property of `_pad_for_sequence_parallelism` (round-up to `tp_size`) — if the target base pads differently
   (e.g. to `2*cp_size`, or pads per-request), the gate and the plan must be re-derived together.

---

## 6. Feature commits (11, in order; everything else on the old branch is a release cherry-pick)

| commit | one-line |
| --- | --- |
| `d22ddb2ef` | The feature: `layers/cp_zigzag.py` (plan + max-flow + shard/gather), zigzag metadata in `sfa_v1._build`, `DSACPContext` fields, forward-context activation/fallback, embedding gather+shard, KV write-back reorder, merged `2*B` top-k/SFA calls, MoE `input_ids`/`mc2_mask` reorder, owner-independent `fixed_order_reduce_scatter` + gated call sites, exit gather; also ships the GLM layer-slice weight filter and the `_sfa_5_3_scope` profiler decorators. |
| `7f4f7f34f` | Make owner-independent reduction possible: `distributed/utils.py` gains `_allreduce_slice_reduce_scatter`/`_all_to_all_fixed_order_reduce_scatter` and `fixed_order_reduce_scatter` with an inline `os.getenv` mode read (default `allreduce`); the `linear_op`/`register_custom_ops` call sites are added; `[CP_BALANCE][plan]` debug log. |
| `e5e86f878` | `block_table_zigzag` re-selects rows by the same real-request indices used for `query_lens_cpu` (padded request slots are not guaranteed to be at the tail). |
| `884c79556` | Wrap the plan build in `try/except` and fall back to continuous DSA-CP when the plan is inconsistent (kills the worker otherwise). |
| `78b63b2d9` | Add `_plain_reduce_scatter` and `REDUCE_MODE=reducescatter` (exact pre-feature baseline) for A/B debugging; update the env-var doc. |
| `f2d487a2c` | Add the `[CP_BALANCE][branch]` log (branch + first refusal reason) and rework the plan logging. |
| `c11cefb80` | Gate the three reduction sites on the per-forward `zigzag_active()` instead of the config-level DSA-CP flag, so `CP_BALANCE=0` (and every non-zigzag batch) is bit-identical to base. |
| `9e462094b` | Restore the fused `npu_transpose_batchmatmul` `q_nope` path in `_q_proj_and_k_up_proj` to be byte-identical with base. |
| `658b836ba` | Drop dead zigzag metadata and the `DeepseekV2DecoderLayer` patch (upstream SP branches cannot fire because the gate rejects DP>1); remove the intermediate `zigzag_shard_positions`/gate wrapper; move the reduce-mode read behind the cached `reduce_mode()` helper in `envs.py`. |
| `8bec7da45` | Drop dead plan fields/aliases and deduplicate the topk/SFA branches (`_indexer_qk_proj` extraction, single `query_lens_arg`/`key_lens_arg`/`block_table_arg`). |
| `23ff2c23c` | Keep the MTP draft path out of zigzag (`for_draft=True` from the proposer) and align the exit gather with base (`pad_size` truncation in `zigzag_gather_tensor`). |
