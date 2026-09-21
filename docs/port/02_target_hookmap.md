> 历史文档（2026-09-18 移植期取证/评审）：结论可能已过时；dp>1 那道门与 zigzag 的现状见 `docs/scripts_review.md` §4.1/§4.1c/§8。
>
> 已变事实（2026-09-21 更正）：
> - 本文的 T = `upstream/main` aff1b74b6；移植已落到 fork 的 `cp_balance`(b64c9569b)，上游 main 已到 d05255207。
> - §1「DSA-CP validation requires index_topk + use_sequence_parallel_moe」：两个实际使用的树都已改成只判 `has_indexer`（base 走 `patches/dsa_cp_dp1.patch`）。
> - §3 建议补回的 `dp_world_size > 1` clause 已补回（`ascend_forward_context.py`），但 dp 门最终按「dp=1 是成立配置」定案，与 §1 结论相反。
> - 「上游 PCP 才是 DSA-CP 的长期归宿」仍然成立（`context_parallel.md` 的迁移说明）。

# 02 — Target hook map for the cp_balance (zigzag DSA-CP) port

Scope: where each of the 10 feature areas lives in **the target tree** `vllm-ascend@upstream/main = aff1b74b6`
(2026-09-18), with file:line, symbol, signature, control flow, the exact zigzag insertion points, and what
differs from the old tree. All target line numbers were read read-only via
`git show 'upstream/main:<path>'` / `git archive` into a scratch dir (no checkout, no commit).

Tree labels used below:

| label | repo / ref | note |
|---|---|---|
| **T** | `vllm-ascend` `upstream/main` = aff1b74b6 | port target |
| **O** | `vllm-ascend` `cp_balance` = 23ff2c23c | feature-carrying old tree |
| **B** | `vllm-ascend-base` (base_layer3) = c7990e5e4 | pre-feature baseline; `diff -rq B O` = the actual feature delta (15 files + new `vllm_ascend/layers/`) |
| **U** | `vllm` 84030bbe3d | upstream vLLM the target tracks |

Whole-feature facts needed to read the map:

* The old feature is `diff -rq /tmp/base /tmp/old`: 15 modified files + new `vllm_ascend/layers/cp_zigzag.py`
  (O `layers/cp_zigzag.py`, 665 lines: `build_zigzag_plan` 399, `zigzag_ineligible_reason` 542,
  `zigzag_shard_tensor` 621, `zigzag_gather_tensor` 630, `zigzag_gather_hidden_states_and_aux` 659,
  `zigzag_reorder_moe_aux` 78, `fixed_order_rank_sum` 51, `get_zigzag_cp_context` 42).
* Feature-side global switch in O: `vllm_ascend/ascend_forward_context.py:604 zigzag_active()`,
  `:216-235 zigzag_cp_active` computation, `:67-160` metadata walkers, `:461-470` A5 MoE selector change.
* **T contains zero occurrences of "zigzag"** (`grep -rn zigzag vllm_ascend/` = 0 hits). T does however contain
  the *contiguous-slice* DSA-CP machinery (`DSACPContext`, `AscendSFADSACPMetadataBuilder`, `AscendSFADSACPImpl`)
  plus three newer CP flavours: DCP (decode CP), PCP (prefill CP) and DSA-CP+DCP/PCP compositions.

---

## 1. DSA-CP prefill token sharding (contiguous slice, `num_tokens_per_device` / `local_start` / `local_end_with_pad`)

**Target sites**

* `T vllm_ascend/attention/context_parallel/sfa_cp.py:281`
  `AscendSFADSACPMetadataBuilder._prepare_parallel_metadata(self, common_attn_metadata, cos, sin, slot_mapping,
  cum_query_lens, seq_lens, draft_index) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]` — the real sharding math:
  `sfa_cp.py:300-306`
  ```
  global_tp_size = get_tp_group().world_size
  num_tokens = common_attn_metadata.num_input_tokens
  num_tokens_pad = _round_up(num_tokens, global_tp_size)
  num_tokens_per_device = num_tokens_pad // global_tp_size
  local_start = get_tp_group().rank_in_group * num_tokens_per_device
  local_end_with_pad = local_start + num_tokens_per_device
  local_end = min(local_end_with_pad, common_attn_metadata.num_actual_tokens)
  ```
  then `:308-324` pads cos/sin (`nn.functional.pad`) and slot_mapping (`-1`) to `num_tokens_pad` and slices all three
  to `[local_start:local_end_with_pad]`; `:335-344` calls `get_cp_local_query_key_lens(...)` into
  `self.dsa_cp_actual_seq_lengths_query/key`; `:346-355` publishes
  `extra["dsa_cp_context"] = DSACPContext(...)`.
* `T sfa_cp.py:191-200` `@dataclass DSACPContext`: `num_tokens, num_tokens_pad, local_start, local_end,
  local_end_with_pad, slot_mapping_cp, actual_seq_lengths_query, actual_seq_lengths_key`.
* `T sfa_cp.py:204-207` `AscendSFADSACPMetadata(AscendSFAMetadata)` with `dsa_cp_context: DSACPContext | None`.
* `T sfa_cp.py:248-279` `__init__(kv_cache_spec, layer_names, vllm_config, device, metadata_cls=None,
  supports_dcp_with_varlen=False)` pre-allocates graph-stable `dsa_cp_actual_seq_lengths_query/key` buffers
  (+ per-draft-step lists when `speculative_config`).
* `T sfa_cp.py:358-377` `_update_parallel_slot_mapping(metadata, slot_mapping, num_input_tokens)` re-derives only
  `dsa_cp_context.slot_mapping_cp` after an outer layout wrapper swapped the slot mapping.
* `T vllm_ascend/attention/context_parallel/common_cp.py:11` `get_cp_local_query_key_lens(query_start_loc,
  cum_query_lens, seq_lens, local_start, local_end) -> (local_query_lens, local_key_lens)` — the vectorised
  clamp/cumsum/offset math used by every CP flavour.
* **Second, independent copy of the sharding math** for the indexer/lightning-indexer cache:
  `T vllm_ascend/attention/indexer.py:753 _build_dsa_cp_slot_mapping(slot_mapping, num_input_tokens, buffer_key)`,
  `:779 _get_dsa_cp_seq_buffers(buffer_key)`, `:794 _build_dsa_cp_parallel_metadata(common_attn_metadata, cos, sin,
  buffer_key)` (recomputes `num_tokens_pad`, `num_tokens_per_device`, `local_start/end` and local cos/sin itself).
* DSA (non-SFA) flavour: `T vllm_ascend/attention/context_parallel/dsa_cp.py:1097`
  `@staticmethod _local_token_range(num_input_tokens) -> (local_start, local_end, tokens_per_rank, num_tokens_pad)`;
  `:1104 _build_local_token_metadata(num_reqs, num_input_tokens, query_start_loc, seq_lens, local_query_start_loc=None,
  local_seq_lens=None, start_pos_out=None, is_noncausal=False)`; consumed from `:844 build_req_metadata(...)`.
* PCP-DSA flavour: `T dsa_cp.py:2133 AscendDSAPCPMetadata`, `:2164 AscendDSAPCPMetadataBuilder`
  (`_local_token_range`-free; uses `pcp_context.global_batch`), `:2447 AscendDSAPCPImpl`.
* Selection/gating: `T sfa_cp.py:1558 resolve_sfa_metadata_builder(vllm_config=None)` / `:1576 resolve_sfa_impl(...)`
  pick the builder/impl from `enable_dsa_cp()` (`T vllm_ascend/utils.py:1491` → `get_ascend_config().enable_dsa_cp`)
  × `enable_sfa_dcp_replicated_indexer()` × `parallel_config.prefill_context_parallel_size > 1`;
  `T attention/sfa_v1.py:276-283 AscendSFABackend.get_builder_cls()` and `:296-303 get_impl_cls()` are the callers.
  DSA-CP config validation lives in `T vllm_ascend/ascend_config.py:611-664` (implies flashcomm, requires
  `index_topk` + `use_sequence_parallel_moe`, forbidden together with PCP).

**Zigzag insertion points (area 1)**
* The natural host is `_prepare_parallel_metadata` at `sfa_cp.py:281`: it already owns `num_tokens`,
  `num_tokens_pad`, `local_start/end`, the padded slot_mapping and cos/sin. A zigzag plan
  (`build_zigzag_plan(query_lens, prefix_lens, cp_size, cp_rank, num_tokens_pad, num_actual_tokens)`) has all its
  inputs at hand right there (`common_attn_metadata.query_start_loc`, `query_start_loc_cpu`, `seq_lens`), and it is
  where O built the plan too (O `attention/sfa_v1.py:595-825`).
* Extended state belongs on `DSACPContext` (`sfa_cp.py:191`) — because `AscendSFADSADCPMetadata`
  (`sfa_cp.py:239-242`) inherits `dsa_cp_context` from both parents, one field set covers the DSA-CP, DSA-CP+DCP
  and DSA-CP+PCP compositions.
* The indexer copy at `indexer.py:794/753` **must** be given the same plan (it writes the LI cache and builds its
  own local cos/sin + slot mapping). A port that only touches `sfa_cp.py` will silently corrupt the indexer cache.
* Alternative, cleaner registration: add a new subclass next to `AscendSFADSACPMetadataBuilder` and extend
  `resolve_sfa_metadata_builder` (`sfa_cp.py:1558-1573`) / `resolve_sfa_impl` (`:1576-1591`).

**Changed vs O**
* O had the whole thing inline inside `AscendSFAMetadataBuilder._build` (O `attention/sfa_v1.py:575-825`,
  `num_tokens_per_device` at O:583-586, `DSACPContext(...)` at O:788-825 with 10 extra zigzag fields).
  T has a **separate builder class** and a small `DSACPContext`; the metadata-object shape changed
  (`slot_mapping_cp_gathered`, `fallback_slot_mapping_cp`, `fallback_cos`, `fallback_sin`, `block_table_zigzag` do
  not exist in T).
* T `_round_up(...)` replaces the inline `(x + w - 1) // w * w`.
* `T get_cp_local_query_key_lens` in `common_cp.py` is a shared helper; O inlined equivalent vectorised math into
  `_build` (O:764-786).
* The indexer is a **second sharding site** in T; in O the indexer was inlined into `AscendSFAImpl`
  (`indexer_select_pre_process` O:1902, `_indexer_qk_proj` O:1951, `indexer_select_post_process` O:2014) so a
  single patch covered both.

---

## 2. SFA attention metadata builder

**Target sites (`T vllm_ascend/attention/sfa_v1.py`)**

* `:310-359` `@dataclass AscendSFAMetadata` (fields: `num_actual_tokens, slot_mapping, seq_lens, seq_lens_cpu,
  cum_query_lens, block_table, sin, cos, num_input_tokens=0, pcp_slot_mapping=None, attn_mask, attn_state,
  num_decodes/num_decode_tokens/num_prefills, positions, query_start_loc, max_query_len/max_seq_len,
  group_len/group_key_idx/group_key_cache_idx, smla_metadata ...`).
  `:365-373` `@dataclass SFAForwardContext(actual_seq_lengths_query, actual_seq_lengths_key, kv_slot_mapping,
  topk_num_tokens, gather_full_o_proj=False)`.
* `:376-421` `class AscendSFAMetadataBuilder(MLACommonMetadataBuilder[AscendSFAMetadata])`,
  `__init__(self, kv_cache_spec, layer_names, vllm_config, device, metadata_cls: type[AscendSFAMetadata] | None =
  None, supports_dcp_with_varlen: bool = False)`.
* **The extension hooks introduced for CP**:
  `:423-434 _prepare_parallel_metadata(common_attn_metadata, cos, sin, slot_mapping, cum_query_lens, seq_lens,
  draft_index) -> (cos, sin, slot_mapping, dict)` (base = identity),
  `:436-443 _update_parallel_slot_mapping(metadata, slot_mapping, num_input_tokens) -> None` (base = no-op),
  `:497-507 _build_with_metadata_view(common_attn_metadata, build_metadata: Callable[[], AscendSFAMetadata])`.
* Control flow of a build: `:470 build(common_prefix_len, common_attn_metadata, fast_build=False, **kwargs)` →
  `_build_with_metadata_view(...)` → `:509 _build(common_attn_metadata, draft_index=None)`:
  `block_table = block_table_tensor[:num_reqs]` (:525), `pcp_slot_mapping = slot_mapping` (:526),
  `slot_mapping = pcp_slot_mapping[:num_input_tokens]` (:527), `input_positions` (:528), `cum_query_lens` (:532),
  `seq_lens`+`seq_lens_cpu` (:533-546), `cos, sin = get_cos_and_sin_mla(input_positions, use_cache=(draft_index is
  None))` (:551), **`_prepare_parallel_metadata(...)` (:553)**, `self.metadata_cls(...)` (:563-583, note `sin/cos`
  are re-truncated to `num_input_tokens` at :575-576), NoPE branch (:584-597).
  `:483 build_for_drafting(common_attn_metadata, draft_index, **kwargs)`, `:600 build_for_cudagraph_capture`,
  `:611 build_for_graph_capture(common_attn_metadata, attn_state=DecodeOnly)`.
* Downstream overrides of the hooks in T: `sfa_cp.py:281` (DSA-CP), `sfa_cp.py:358` (slot refresh),
  `sfa_cp.py:617 AscendSFADCPMetadataBuilder` (`__init__` :621,
  `_get_dcp_local_seq_lens` :710, `_build_block_table_replicated_view` :753, `_build_slot_mapping_replicated_view`
  :775, `_build_compact_kv_gather_metadata` :796, `_build_with_metadata_view` :814, `build_for_graph_capture`
  :883), `sfa_cp.py:904/907/993` (`AscendSFAPCPDCPMetadataBuilder`),
  `sfa_cp.py:1529/1535` (`AscendSFADSADCPMetadataBuilder`), `T attention/sfa_kv_offload.py:148 build_for_drafting`.
* **v2 runner equivalent**: `T vllm_ascend/worker/v2/attn_utils.py:223 build_attn_metadata(*, attn_groups, num_reqs,
  num_actual_reqs=None, num_tokens, query_start_loc_gpu, query_start_loc_cpu, max_query_len, seq_lens, max_seq_len,
  block_tables, slot_mappings, kv_cache_config, dcp_local_seq_lens=None, seq_lens_np=None,
  seq_lens_cpu_upper_bound=None, num_computed_tokens_cpu=None, positions=None, attn_state=None, graph_pad_size=-1,
  num_actual_tokens=None, num_input_tokens=None, is_prefilling=None, pcp_context: AscendPCPAttentionContext | None =
  None, model_specific_attn_metadata=None, for_cudagraph_capture=False, causal=True) -> dict[str, Any]`;
  builder dispatch at `attn_utils.py:320-363` (`attn_group.get_metadata_builder(0)`,
  `isinstance(..., AscendSFAMetadataBuilder)`, `pcp_context` injected for SFA/DSA builders at :341-345,
  `build(...)` at :353). Entry point: `T worker/v2/model_states/default.py:44 prepare_attn(...)` → `:97
  build_attn_metadata(...)`; the whole module is swapped in by `T worker/v2/attn_utils.py:1233
  build_attn_metadata_wrapper()`.

**Zigzag insertion points (area 2)**
* Metadata build: `_prepare_parallel_metadata` (`sfa_v1.py:423` override point; concrete impl to copy:
  `sfa_cp.py:281`). Everything O had to patch in `_build` — cos/sin layout, `slot_mapping_cp`, per-request
  `actual_seq_lengths_query/key`, `block_table_zigzag` — is constructible here and returned through the `extra`
  dict that is splatted into `metadata_cls(**parallel_metadata)` at `sfa_v1.py:582`.
* Graph/draft stability: per-draft-step buffers live in the builder `__init__` (`sfa_cp.py:265-279`), mirroring O's
  `spec_actual_seq_lengths_query/key`.
* Post-build mutation of an outer layout wrapper must go through `_update_parallel_slot_mapping`
  (`sfa_cp.py:358`) — O instead mutated `meta.cos/sin` and `ctx.*` from the forward context
  (O `ascend_forward_context.py:_disable_zigzag_metadata_for_fallback`).

**Changed vs O**
* O's builder was patched by editing `AscendSFAMetadataBuilder._build` itself (O `_build` at `sfa_v1.py:506`,
  zigzag gate at O:575-646, metadata construction O:797-825). T's `_build` is shared by 5 subclasses; the
  supported extension point is the two hook methods.
* `pcp_slot_mapping` (:526/570) is a new always-present field (PCP keeps the scheduler-global mapping; the local
  one is a view). `attn_state`, `attn_mask` are now pre-computed inside `_build` (`:572`,
  `AttentionMaskBuilder` at `:421`).
* No v2-specific metadata *class*: v2 reuses `AscendSFAMetadata`/`AscendSFAMetadataBuilder` and only differs in
  who calls `build()` and what kwargs are injected (`attn_utils.py:320-357`).

---

## 3. Per-forward "which layout is active" switch

**Target sites**

* V1 producer: `T vllm_ascend/ascend_forward_context.py:115-251
  set_ascend_forward_context(attn_metadata, vllm_config, num_tokens=0, num_tokens_across_dp=None,
  in_profile_run=False, num_actual_tokens=None, aclgraph_runtime_mode=CUDAGraphMode.NONE,
  batch_descriptor=None, model_instance=None, is_draft_model=False, skip_compiled=False, max_tokens_across_pcp=0,
  draft_attn_metadatas=None, device_metadata_executor=None, has_sinks=False, eplb_heat_collection_status=False)`.
  Body: `sync_v2_extra_kwargs` (:144) → `set_forward_context(**kwargs)` (:154) → `draft_attn_metadatas` (:156) →
  MoE comm method + `model_comm_methods` (:159-184) → `padded_length` (:229-233) → `max_tokens_across_pcp` (:228) →
  `mc2_mask` (:242-247) → `yield`. Called from `T worker/model_runner_v1.py:2497` and the proposers:
  `spec_decode/llm_base_proposer.py:844,1215`, `dflash_proposer.py:254`, `dspark_proposer.py:394`,
  `step3p5.py:293,473`, `medusa_proposer.py:33`.
* V2 producer (new): `T vllm_ascend/platform.py:81 class NPUPlatform`, `:527-658
  set_additional_forward_context(cls, attn_metadata, vllm_config, dp_metadata, num_tokens=0,
  num_tokens_across_dp=None, cudagraph_runtime_mode=None, batch_descriptor=None, ubatch_slices=None) ->
  dict[str, Any]`, returning `{"moe_comm_type", "moe_comm_method", "capturing", "mmrs_fusion", "num_tokens",
  "padded_length", "max_tokens_across_dp", "max_tokens_across_pcp", "mc2_mask", "is_draft_model",
  "is_draft_model_prefill", "in_profile_run", "padded_num_tokens", "sinks", "dynamic_mx_quant_scale_alg"}` (:642-658);
  V1 short-circuits at `:586-587`. Called by U `vllm/forward_context.py:318` into
  `forward_context.additional_kwargs`.
* Accessor: `T ascend_forward_context.py:443-471 class _ExtraForwardContextProxy` with
  `extra_attrs = ("capturing", "moe_comm_type", "moe_comm_method", "is_decode_only_node", "use_mega_moe",
  "mmrs_fusion", "num_tokens", "padded_length", "num_tokens_across_dp", "mc2_mask", "is_draft_model",
  "is_draft_model_prefill", "draft_moe_quant_type", "prefetch_mlp_gate_up_proj", "prefetch_mlp_down_proj",
  "model_instance", "layer_idx", "max_tokens_across_dp", "max_tokens_across_pcp", "num_accept_tokens",
  "in_profile_run", "padded_num_tokens", "sinks", "eplb_heat_collection_status")`; `__getattr__`/`__setattr__`
  (:484-499) route to `ctx.additional_kwargs` when the module-global `_USE_V2_EXTRA_KWARGS` (declared :26-28, set
  by `sync_v2_extra_kwargs(vllm_config)` :31-38) is true, else to plain attributes. Singleton at `:503`.
* There is **no per-forward layout flag in T**. Layout is expressed on the metadata object and read by the impl:
  `sfa_cp.py:1093 AscendSFADCPImpl._has_prefill(attn_metadata)` (`attn_metadata.num_prefills > 0`),
  `sfa_v1.py:1488 / sfa_cp.py:423 _get_parallel_forward_context(...)` reading `attn_metadata.dsa_cp_context` /
  `dcp_context`, `sfa_cp.py:1668` (`need_gather_q_kv and ... not is_decode`).
  Absent-attribute reads are tolerated by `_EXTRA_CTX` returning `None` for unknown-but-listed names only.

**Zigzag insertion points (area 3)**
* Re-add a `zigzag_active()` helper (O `ascend_forward_context.py:604-616`) next to `_EXTRA_CTX` (T `:503`) and
  register the two names in `extra_attrs` (`T :446-471`) — otherwise `_EXTRA_CTX.zigzag_cp_active` raises
  `AttributeError` by design (`check_extra_attr`, T :473-478).
* V1: compute the flag inside `set_ascend_forward_context` **before** `yield` (T `:154-248`), from
  `attn_metadata` (`_find_zigzag_cp_context` equivalent) + `is_draft_model` (T signature already has it, :125).
  There is no `dp_world_size > 1` branch any more (`T :220-225` only computes `max_tokens_across_dp`), so the old
  "disable on DP>1" clause has to be reintroduced explicitly.
* V2: add the flag to the returned dict of `set_additional_forward_context` (T `platform.py:642-658`); it lands in
  `forward_context.additional_kwargs` automatically. **But** `_USE_V2_EXTRA_KWARGS` is a module global folded by
  Dynamo (T `:26-28` comment) synced from eager setup — a value that only becomes known per batch is fine
  (the dict is rebuilt per call) as long as no compiled graph branches on it.

**Changed vs O**
* O's `set_ascend_forward_context` had an explicit `if envs_vllm.VLLM_USE_V2_MODEL_RUNNER:` branch writing into
  `ctx.additional_kwargs` (O `ascend_forward_context.py:585-597`); T removed that branch and replaced it with the
  module-global `_USE_V2_EXTRA_KWARGS`. `envs_vllm` is no longer imported by `ascend_forward_context.py` in T;
  T uses `vllm_ascend.mrv2_utils.use_v2_model_runner` (`T ascend_forward_context.py:19`).
* Attributes present in O and deleted in T: `flash_comm_v1_enabled`, `pad_size` (0 hits in T;
  O `ascend_forward_context.py:169-184, 208-210`, O `model_runner_v1.py:2603`). New in T: `is_decode_only_node`,
  `use_mega_moe`, `draft_moe_quant_type`, `max_tokens_across_pcp`.
* O's `_iter_attn_metadata` / `_find_zigzag_cp_context` / `_disable_zigzag_metadata_for_fallback`
  (O `ascend_forward_context.py:67-160`) have no counterpart in T.

---

## 4. Model entry: embedding + positions under FlashComm/SP

**Target sites**

* `T vllm_ascend/patch/worker/patch_deepseek_v2.py:301-362
  _patched_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds=None)`; installed by
  `:365 DeepseekV2Model.forward = _patched_forward` (module-import side effect; module is imported from
  `T patch/worker/__init__.py:45`). Body: embed `:308-315`, `llama_4_scaling` `:321-330`, aux-hidden gather
  `:337-342`, decoder loop `:333-343`, PP-intermediate return `:345-346`, **SP exit gather `:348-354`**, final norm
  `:359`, aux tuple return `:360-362`.
* GLM-5.2 entry: `T vllm_ascend/models/__init__.py:73` registers `"GlmMoeDsaForCausalLM"` →
  `vllm_ascend.models.deepseek_mtp:AscendGlmMoeDsaForCausalLM`; `T models/deepseek_mtp.py:74-81` subclasses
  U `GlmMoeDsaForCausalLM` (`U vllm/model_executor/models/deepseek_v2.py:1982`), which reuses
  `U DeepseekV2Model.forward` (`U .../deepseek_v2.py:1481`) — i.e. the patch above **is** the GLM-5.2 forward.
* Vocab-parallel embedding: `T vllm_ascend/ops/vocab_parallel_embedding.py:50
  class AscendVocabParallelEmbedding(VocabParallelEmbedding)`,
  `:57 __init__(num_embeddings, embedding_dim, params_dtype=None, org_num_embeddings=None, padding_size=...,
  quant_config=None, prefix="", *, disable_tp=False)` (comm-group choice `:86-94`: `disable_tp` →
  `get_replicated_group()`, `lmhead_tp_enable() and "head" in prefix` → `get_lmhead_tp_group()`,
  `embedding_tp_enable() and "embed_tokens" in prefix` → `get_embed_tp_group(), forward_type="embed_tp"`,
  else `get_tp_group()`); `:190 forward(input_)` → `:195 _forward_embed_tp` or `:254 _forward_origin`.
  `_forward_origin` ends with `torch.ops.vllm.all_reduce(output_parallel, tp_group.unique_name)` (`:285`) and
  explicitly documents that the regular TP embedding must **not** reduce-scatter (comment `:279-285`).
  `:288 AscendParallelLMHead(ParallelLMHead)`, `:330 lmhead_all_to_all(...)`,
  `:373 AscendLogitsProcessor(LogitsProcessor)` with `:387 _get_logits`, `:407 _get_logits_lmheadtp`,
  `:428 _get_logits_normal`.
* MTP-layer detection helper used by the patch: `T vllm_ascend/utils.py:1617
  is_mtp_layer(hf_config, layer_name) -> bool`.

**Zigzag insertion points (area 4)**
* `_patched_forward` is the only model-boundary hook for the SFA/DeepseekV2 family: shard `positions` and the
  embedding *between* `:314` and `:333`, and replace the exit gather at `:348-354` with the zigzag
  gather+rerange. O did exactly this (O `patch_deepseek_v2.py:305-360`: `zigzag_shard_tensor(positions)`,
  `tensor_model_parallel_all_gather(hidden_states)` + `zigzag_shard_tensor`, and skipped the exit gather while
  the runner did it).
* The experimental local-embedding path belongs in `AscendVocabParallelEmbedding` as an extra
  `forward_type`/method next to `_forward_embed_tp` (`:195`) / `_forward_origin` (`:254`); O added
  `forward_zigzag_local(input_)` (O `:170-201`) plus an `_embed_partial` split of `_forward_origin`
  (O `:262-281`).
* Aux-hidden-states path (`:337-342`) also needs the zigzag branch: O guarded it with `and not zigzag_active`.

**Changed vs O**
* T's `_forward_origin` uses `torch.ops.vllm.all_reduce(output_parallel, tp_group.unique_name)` (`:285`);
  O/B used `torch.ops.vllm.maybe_pad_and_reduce(output_parallel)` (O `:277-289`) — the SP semantics moved, so the
  old "embedding already returns a rank-local reduce-scattered slice" assumption no longer holds.
* `patch/worker/patch_deepseek_v2.py:38-58 _should_skip_indexer_init` now calls `is_mtp_layer(config, prefix)`
  (T `utils.py:1617`) instead of the inline `num_hidden_layers` test O used (`O patch_deepseek_v2.py:44-52`).
* New in T: `disable_tp` embedding path (`:67-94`) and lmhead-TP logits split (`:407-427`), both absent in O.
* `T models/deepseek_mtp.py` (MTP classes) replaces part of the deleted O `patch/worker/patch_deepseek_mtp.py`.

---

## 5. Attention layer internals (SFA forward)

**Target sites (`T vllm_ascend/attention/sfa_v1.py`, `class AscendSFAImpl(MLAAttentionImpl)` at `:628`)**

* `:1622-1836 forward(self, layer_name, hidden_states: Tensor, kv_cache: tuple[Tensor, ...], attn_metadata: M,
  output: Tensor | None = None) -> Tensor`. Flow:
  `:1631-1633` profiling early-out; `:1635-1641` NoPE token clamp; `:1644 _compose_sfa_kv_cache(kv_cache)`;
  `:1648-1651` `cos/sin` from metadata + `slot_mapping_sfa = self._get_sfa_kv_slot_mapping(attn_metadata)` +
  `indexer_attn_metadata = self._get_indexer_attn_metadata()`;
  `:1654-1661 num_input_tokens` + `parallel_context = self._get_parallel_forward_context(...)` →
  `actual_seq_lengths_query/key`;
  `:1663-1703` fused preprocess branch (`PROLOG_V3` → `_sfa_preprocess_prolog_v3` `:1688`, else
  `_sfa_preprocess_mlapo` `:1696`);
  `:1705-1768` native branch: `:1708 fused_qkv_a_proj`, `:1714 q_a_layernorm`,
  `:1726 exec_kv(kv_no_split, cos, sin, kv_cache, parallel_context.kv_slot_mapping, attn_metadata)`
  (**KV write-back**), `:1739 _prepare_kv_for_parallel`, `:1746 _q_proj_and_k_up_proj`,
  `:1748 rope_single`, `:1749 _record_query_gather_context`, `:1755 _store_parallel_kv`;
  `:1770-1796` indexer/top-k: `self.indexer(hidden_states, q_c, k_hidden_states, indexer_attn_metadata,
  compute_topk=not self.skip_topk)` at `:1780`, shared-index fallbacks `:1787-1794`;
  `:1809 _execute_sparse_flash_attention_process(ql_nope, q_pe, kv_cache, topk_indices, attn_metadata,
  actual_seq_lengths_query, actual_seq_lengths_key)`;
  `:1819 _v_up_proj`, `:1820-1822` attention gate, `:1828 _finalize_o_proj`.
* Overridable hooks (all no-op/reference impls in the base class): `:1436 _prepare_kv_for_parallel(k_pe, k_nope,
  knope_scale, full_gather_o_proj_enabled)`, `:1449 _store_parallel_kv(k_pe, k_nope, knope_scale,
  fused_kv_no_split, kv_ag_handles, kv_cache, slot_mapping_sfa, attn_metadata, full_gather_o_proj_enabled)`,
  `:1424 _record_query_gather_context(ql_nope, q_pe, attn_metadata)`, `:1432 _parallel_query_gather_dim()`,
  `:1488 _get_parallel_forward_context(attn_metadata, num_input_tokens, hidden_states) -> SFAForwardContext`,
  `:1501 _prepare_native_hidden_states(hidden_states, attn_metadata)`,
  `:1508 _finalize_o_proj(attn_output, output, gather_full_o_proj)`,
  `:1517 _get_sfa_kv_slot_mapping(attn_metadata) -> Tensor`, `:1083 exec_kv(kv_no_split, cos, sin, kv_cache, slots,
  attn_metadata)`, `:1399 _execute_sparse_flash_attention_process(...)`, `:1523 _compose_sfa_kv_cache(kv_cache)`.
  Concrete overrides live in `sfa_cp.py:63 AscendSFAPCPImpl` (`exec_kv` :172, `_get_parallel_forward_context` :98,
  `_finalize_o_proj` :117, `_get_sfa_kv_slot_mapping` :165), `sfa_cp.py:380 AscendSFADSACPImpl`
  (`_get_parallel_forward_context` :423, `exec_kv` :448, `_prepare_kv_for_parallel` :477,
  `_store_parallel_kv` :504, `_apply_o_proj_full_weight` :545, `_finalize_o_proj` :548) and
  `sfa_cp.py:1039 AscendSFADCPImpl` (`_store_parallel_kv` :1335, `_execute_sparse_flash_attention_process` :1367,
  `_remap_sparse_indices` :1179, `_merge_dcp_outputs` :1228).
* Indexer extracted to its own class: `T vllm_ascend/attention/indexer.py:92
  class AscendSFAIndexerBackend(nn.Module, AttentionBackend)` (dual interface: static backend contract
  `:139-162`, per-layer `nn.Module` impl from `:166 __init__(vllm_indexer, qk_rope_head_dim)`);
  `:220 write_cache(k_li, k_li_scale, slot_mapping, indexer_attn_metadata=None)`,
  `:279 forward_k(hidden_states, cos, sin)`, `:328 _gather_cache_inputs(...)`,
  `:373 forward(hidden_states, q_c, k_hidden_states, indexer_metadata, compute_topk=True)`;
  metadata builder `:471 AscendSFAIndexerMetadataBuilder` (`:912 build`, `:941 build_for_drafting`,
  `:956 build_for_graph_capture`, `:753/779/794` DSA-CP helpers).
  SFA reaches it through `sfa_v1.py:1594 _get_indexer_attn_metadata()` (looks up
  `forward_context.attn_metadata[<indexer k-cache prefix>]`, raises if absent).
* Device-level kernels: `T vllm_ascend/device/device_op.py:409
  BaseDeviceAdaptor.execute_sparse_flash_attention_process(cls, sfa_impl, ql_nope, q_pe, kv_cache, topk_indices,
  attn_metadata, actual_seq_lengths_query, actual_seq_lengths_key, block_table: torch.Tensor | None = None)`
  (`:424-425` falls back to `attn_metadata.block_table`), `:480 _execute_kv_quant_sparse_flash_attention`,
  `:338 indexer_select_post_process` (A5 twin at `:1308`).

**Zigzag insertion points (area 5)**
* Add a new impl subclass beside `AscendSFADSACPImpl` (`sfa_cp.py:380`) and register it in `resolve_sfa_impl`
  (`sfa_cp.py:1576`). Override: `_get_parallel_forward_context` (`sfa_v1.py:1488`, swap in
  `actual_seq_lengths_*_zigzag` + `block_table_zigzag`), `_get_sfa_kv_slot_mapping` (`:1517`, return the zigzag
  slot order), `_prepare_native_hidden_states` (`:1501`, skip the SP all-gather), `_finalize_o_proj` (`:1508`),
  and — for the LI C8 reshape-store optimisation — `_store_parallel_kv` (`:1449`) / `exec_kv` (`:1083`).
* The old "reorder the indexer slot mapping once per batch" trick (O `sfa_v1.py:2572-2584`, using
  `dsa_cp_context.slot_mapping_cp_gathered`) maps onto `indexer.py:794 _build_dsa_cp_parallel_metadata` in T —
  the indexer no longer receives `attn_metadata.dsa_cp_context` directly, it gets its **own** metadata object
  (`indexer.py:58 AscendSFAIndexerMetadata`) built by `indexer.py:471`.
* Sparse-attention call: pass the zigzag `block_table` via the `block_table=` kwarg of
  `execute_sparse_flash_attention_process` (`device_op.py:409/420`), which already exists in T.

**Changed vs O**
* O had the indexer path **inside** `AscendSFAImpl`: `indexer_select_pre_process` (O:1902), `_indexer_qk_proj`
  (O:1951), `indexer_select_post_process` (O:2014), `_maybe_gather_kv_for_dsacp` (O:2112),
  `_maybe_store_kvcache_for_c8_n_dsacp` (O:2183), `indexer_attn_metadata` obtained without a separate lookup.
  **None of those names exist in T** — the feature's zigzag edits at O:2486-2600 have no direct counterpart and
  must be re-expressed on `AscendSFAIndexerBackend` (`indexer.py:92-470`).
* T's `forward` is a template with 8 documented extension hooks; O's `forward` (O:2371-2717) had the DSA-CP
  branching inline (`zigzag_active` computed at O:2492-2498, `kv_slots = slot_mapping_cp` at O:2523-2526,
  `slot_mapping_cp_gathered` for the indexer at O:2568-2584).
* `exec_kv` in T takes `slots` explicitly (`sfa_v1.py:1083`; DSA-CP override `sfa_cp.py:448-476`) and calls
  U `_gather_prefill_cache_inputs` (`sfa_cp.py:182`) for PCP — the O approach of choosing `slot_mapping_cp` vs
  `slot_mapping_sfa` inside `forward` is gone.
* `block_table` is now an explicit optional parameter of the LI kernel call in T (already upstreamed from O).

---

## 6. o_proj / MLP / MoE TP reductions

**Target sites**

* `T vllm_ascend/ops/linear_op.py`: `:61 CustomLinearOp`, `:103 CustomColumnParallelOp`, `:113
  CustomRowParallelOp` (incl. `:126 apply`, `:133 get_input_parallel`), `:152 MLPColumnParallelOp`,
  `:174 MLPRowParallelOp` — `:182 apply_impl(input_) -> (output, output_bias)` ending in
  `:188 output = self.comm_group.reduce_scatter(output_parallel, 0)`; `:194 DSV4OProjColumnParallelOp` /
  `:210 DSV4OProjRowParallelOp` (otp group), `:227 OProjRowParallelOp` (`:235-270` all-to-all+reduce on the otp
  group), `:276 ShardedCPColumnParallelOp`; factories `:299 _get_column_parallel_op(prefix, layer)`,
  `:311 _get_row_parallel_op(prefix, layer)`, `:321 get_parallel_op(disable_tp, prefix, layer, direct)`,
  `:360 get_replicated_op(disable_tp, prefix, layer)`.
  **`SequenceColumnParallelOp` / `SequenceRowParallelOp` do not exist in T** (`grep -c` = 0); they are the
  classes O patched (O `linear_op.py:302/327`, zigzag edits at O:199-215 and O:459-467).
* SP ops are now **EP** ops: `T vllm_ascend/ops/register_custom_ops.py:57
  _maybe_all_gather_and_maybe_unpad_impl(x)` (dp_metadata + `get_ep_group()`, padding helper `:46
  _pad_to_ep_local_size`, chunk-size helper `:18 _get_ep_local_sizes`), `:89 _maybe_pad_and_reduce_impl(x)`
  (uses `_EXTRA_CTX.padded_length` :117, `_EXTRA_CTX.is_draft_model` :93). Registered as custom ops at
  `:226/234`; call sites are `torch.ops.vllm.maybe_all_gather_and_maybe_unpad(...)` /
  `torch.ops.vllm.maybe_pad_and_reduce(...)`.
* MoE: `T vllm_ascend/ops/fused_moe/prepare_finalize.py:401 _prepare_with_ep_group(...)` →
  `maybe_all_gather_and_maybe_unpad` at `:420-426`; `:519 _finalize_with_ep_group(hidden_states)` →
  `maybe_pad_and_reduce` at `:529`; `:533 _finalize_with_dp_group(hidden_states, reduce_results)` (PCP
  `get_pcp_group().reduce_scatter` at `:543-545`); `:466-470 max_tokens_across_pcp` padding.
* Router / hash-routing (the old `experts_selector.py` hook):
  `T vllm_ascend/ops/fused_moe/router/fused_topk_router.py:139 _compute_routing(self, hidden_states,
  router_logits, indices_type, *, input_ids: torch.Tensor | None = None)`; `:158-175` sqrtsoftplus branch with
  `tid2eid`: `input_ids.to(torch.int64)` (`:162`), `pad_and_split_input_ids` /
  `all_gather_input_id_with_dp_group` (`:164-168`), `sequence_parallel_chunk(...)` (`:174`), `-1 → 0` (`:175`),
  then `torch.ops._C_ascend.moe_gating_top_k_hash(...)` (`:194`). Also `router/grouped_topk_router.py:92`,
  `router/router_factory.py:27`.
  **`T vllm_ascend/ops/fused_moe/experts_selector.py` does not exist** (O had it; the feature's entire edit was
  the comment at O `experts_selector.py:247-253` noting that `forward_context.input_ids` was pre-reordered).
* `T vllm_ascend/distributed/utils.py` is 29 lines: `:7 get_decode_context_model_parallel_world_size`,
  `:12 get_decode_context_model_parallel_rank`, `:17 all_gather_async`, `:29 split_tensor_along_first_dim`.
  The old feature added `reduce_mode()`/`_allreduce_slice_reduce_scatter`/`_plain_reduce_scatter`/
  `_all_to_all_fixed_order_reduce_scatter`/`fixed_order_reduce_scatter` here (O `distributed/utils.py:24-150`) —
  **none of it exists in T**.

**Zigzag insertion points (area 6)**
* Owner-independent reduction helper: re-add to `T distributed/utils.py` and call from
  `linear_op.py:188` (`MLPRowParallelOp.apply_impl`) and from `_maybe_pad_and_reduce_impl`
  (`register_custom_ops.py:89`, the TP reduce-scatter site reachable from MoE finalize). O's call sites were
  O `linear_op.py:199-215` (MLP) and O `register_custom_ops.py:116-121` (pad_and_reduce). In T the latter is
  inside `_finalize_with_ep_group` (`prepare_finalize.py:519/529`) — the helper must be handed the right group
  (`get_tp_group()` vs `get_ep_group()`), which is *not* a straight substitution: O's version summed across the
  TP group because FlashComm owned rows per TP rank.
* MoE hash-routing reorder: T passes `input_ids` **explicitly** as a kwarg
  (`fused_topk_router.py:145`), not via `forward_context.input_ids` (O `experts_selector.py` read
  `get_forward_context().input_ids`). The reorder therefore belongs at the router call site / in
  `router/fused_topk_router.py:_compute_routing` before `:164`, or must be published through
  `forward_context.input_ids` again.
* Anything touching `_EXTRA_CTX.pad_size` must be rewritten: T has `_EXTRA_CTX.padded_length` (`:454`) and
  `_EXTRA_CTX.max_tokens_across_pcp` (`:465`).

**Changed vs O**
* Feature classes deleted: `SequenceColumnParallelOp`, `SequenceRowParallelOp` (O `linear_op.py:302/327`) — the
  exact two `apply_impl`s the feature patched.
* `ops/fused_moe/` was regrouped: T has `dataclass/` (`fused_experts, moe_mlp, moe_quant, prepare_finalize,
  router_input, shared_experts, token_dispatcher`), `router/`, `routed_experts.py`, `token_dispatcher.py`;
  O had `comm_utils.py`, `experts_selector.py`, `moe_runtime_args.py`, `moe_stage_contracts.py`,
  `moe_stage_params.py`.
* `maybe_*` custom ops changed meaning (SP → EP), so any zigzag variant must be re-derived, not copied.

---

## 7. Exit gather of hidden states before logits

**Target sites**

* **There is no gather in the runner any more.** `T worker/model_runner_v1.py:3091-3141
  _model_forward(self, num_tokens_padded, input_ids=None, positions=None, intermediate_tensors=None,
  inputs_embeds=None, **model_kwargs)` builds `model_inputs` (:3104-3110), optionally engines Engram inputs
  (:3112-3122), runs `run_model()` (:3129/:3131) and returns `hidden_states` verbatim (:3141).
  `NPUModelRunner._all_gather_hidden_states` / `_all_gather_hidden_states_list` /
  `_all_gather_hidden_states_and_aux` (O `model_runner_v1.py:2526-2548`) **do not exist in T**
  (`grep -rn all_gather_hidden_states` → only stale doc strings in `T profiling_config.py:286-289`).
* The gather moved into the model forward: `T patch/worker/patch_deepseek_v2.py:348-354`
  (`combined_states = torch.cat([hidden_states, residual], -1)` → `tensor_model_parallel_all_gather(..., 0)` →
  `[: positions.shape[0]]`), aux variant `:337-342`.
* V1 logits path: `T worker/model_runner_v1.py:2577 sample_hidden_states = hidden_states[logits_indices]` →
  `:2578 self.model.compute_logits(sample_hidden_states)` (PP-broadcast branch `:2583-2598`).
* `T worker/model_runner_v1.py:3153-3161
  _pad_for_sequence_parallelism(self, num_scheduled_tokens) -> int`:
  `round_up(num_scheduled_tokens, tp_size)` iff `enable_sp(self.vllm_config) or enable_dsa_cp()`; called from
  `_determine_batch_execution_and_padding` at `:3222`.
  `:3164-3193 sync_and_slice_intermediate_tensors(num_tokens, intermediate_tensors, sync_self)` (SP-slices PP
  intermediates using `enable_sp()` at `:3176/:3189`), `:3195 sync_and_gather_intermediate_tensors` (alias).
* V2 runner: `T worker/v2/model_runner.py:648-702 sample(self, hidden_states, input_batch, grammar_output)` →
  `:671 sample_hidden_states = hidden_states[input_batch.logits_indices]` → `:674 self.model.compute_logits(...)`;
  `:639-646 _lmhead_tp_max_num_logits()`; `:704-738 _dummy_run(...)` joins the LM-head collectives.
  The upstream producer of those hidden states is `U vllm/v1/worker/gpu/model_runner.py:879`.
  For PCP the restore step is `T worker/v2/pcp_manager.py:355-392 restore_hidden_states(hidden_states)`
  (delegating to `U vllm/v1/worker/gpu/pcp_manager.py:637`), driven by
  `AscendPCPAttentionContext.hidden_restore_idx` / `padded_gather_idx` (`T worker/v2/pcp_manager.py:36-48`).

**Zigzag insertion points (area 7)**
* The zigzag shard/gather must live in the model forward (`patch/worker/patch_deepseek_v2.py:314-359`), because
  the runner-visible hidden states are already the full-natural-order tensor that `sample()` indexes. O did the
  opposite (skipped the gather in the model and gathered in the runner, O `model_runner_v1.py:2601-2609`) —
  the port must invert that.
* V1 padding site: `_pad_for_sequence_parallelism` (`model_runner_v1.py:3153-3161`) already pads for
  `enable_dsa_cp()`, which is exactly the alignment zigzag needs; O additionally consulted `enable_sp_by_pass`
  (a function in O; in T it is an `AscendConfig` field `ascend_config.py:521`, set at `:785`).
* V2/PCP: the zigzag restore belongs next to `AscendPCPManager.restore_hidden_states`
  (`worker/v2/pcp_manager.py:355`) and/or in `U .../pcp_manager.py:261-333 _build_batch_layout`, which already
  produces `hidden_restore_idx` / `padded_gather_idx` / `gathered_kv_write_mask`.
* Prompt-logprobs path (`T model_runner_v1.py:3036-3039 _get_prompt_logprobs_dict(hidden_states
  [:num_scheduled_tokens], ...)`) also consumes the *full* hidden states and needs the same treatment.

**Changed vs O**
* `forward_context.flash_comm_v1_enabled` and `forward_context.pad_size` were **deleted** (0 hits in T;
  O `ascend_forward_context.py:169-184/208-210`); the whole `if flash_comm_v1_enabled ... _all_gather...` block in
  O `model_runner_v1.py:2601-2609` is gone. `T ascend_forward_context.py` now exposes only `padded_length`
  (`:229-233`, `:454`) and the V2 dict (`platform.py:642-658`).
* `sync_and_slice_intermediate_tensors` in T is a plain override using `enable_sp()` (comments at `:3163-3178`);
  O carried the "flashcomm1 does not scatter the residual" logic.
* T `_pad_for_sequence_parallelism` keys off `enable_dsa_cp()` directly, i.e. the DSA-CP alignment requirement is
  already upstreamed — the feature's equivalent clause (O `model_runner_v1.py:2612-2623`) is redundant in T.
* New: `T worker/v2/model_runner.py:648` and its lmhead-TP padding (`:671-675`) which zigs the "exit row
  ordering" question for the V2 path.

---

## 8. MTP / speculative decoding draft path

**Target sites**

* `T vllm_ascend/spec_decode/llm_base_proposer.py:116 class AscendSpecDecodeBaseProposer(SpecDecodeBaseProposer)`;
  `:2464-2542 build_draft_attn_metadata(self, common_attn_metadata, num_input_tokens, num_actual_tokens)`:
  iterates `self.draft_attn_groups` (`:2494`), dspark → `builder.build_for_drafting(common_attn_metadata,
  draft_index=1, **extra_attn_metadata_args)` (`:2522-2524`), otherwise →
  `builder.build(0, common_attn_metadata, self.runner.get_model(), **extra_attn_metadata_args)` (`:2528-2530`);
  result keyed per layer `:2534-2535`, returned as a 1-step list `:2538-2542 `. Other relevant methods:
  `:683 dummy_run`, `:887 _propose`, `:1285 compute_draft_token_ids`, `:2395 _update_full_graph_params`.
  Every `build_for_drafting` call site in this file: `:813` (in `dummy_run`, `:683`), `:1184` (in `_propose`,
  `:887`), `:2084` (in `attn_update_stack_num_spec_norm`, `:1880`), `:2522` (in `build_draft_attn_metadata`,
  `:2464`).
* `T vllm_ascend/spec_decode/mtp.py` is 50 lines and holds only `:14 compact_mtp_topk_indices(...)` — the MTP
  proposer/model moved out.
* MTP model: `T vllm_ascend/models/deepseek_mtp.py:16 class AscendDeepSeekMTP(DeepSeekMTP)` (`:23 forward(...,
  spec_step_idx=0)`, `:36 _maybe_set_own_lm_head`, `:56 load_weights`, `:67 _rewrite_spec_layer_name`) and
  `:74 class AscendGlmMoeDsaForCausalLM(GlmMoeDsaForCausalLM)`; registered at `T models/__init__.py:73`.
  **`T vllm_ascend/patch/worker/patch_deepseek_mtp.py` does not exist** (O had it: O
  `patch/worker/patch_deepseek_mtp.py:86/122/130` held `AscendDeepSeekMultiTokenPredictorLayer`,
  `AscendDeepSeekMTP`, `AscendGlmMoeDsaForCausalLM`; O `patch/worker/__init__.py` imported it).
* SFA draft metadata: `T attention/sfa_v1.py:483 build_for_drafting(common_attn_metadata, draft_index, **kwargs)`
  → `:509 _build(common_attn_metadata, draft_index=draft_index)`; `:593-597` per-draft `SparseMLAMetadataState`.
  DSA twin: `T attention/context_parallel/dsa_cp.py:442 build_for_drafting`, `:550
  build_req_metadata_for_drafting`, `:753 _num_compressor_metadata_rows`; indexer twin: `T attention/indexer.py:941
  build_for_drafting`.
* V2 speculator tree: `T worker/v2/spec_decode/mtp/speculator.py:21
  class AscendMTPSpeculator(AscendAutoRegressiveSpeculator, MTPSpeculator)` (29-line file);
  `T worker/v2/spec_decode/autoregressive/speculator.py:269-300 propose(...)` wraps
  `disable_target_pcp_for_replicated_draft(self)`, `build_attn_metadata_wrapper()`, `torch_gather_wrapper()`
  around `super().propose(...)`; `:556-595` `_update_draft_attn_metadata` returns early for DSA/SFA
  (`:561-562`); helpers `T worker/v2/attn_utils.py:1244 build_draft_attn_metadata_factory(positions, pad,
  is_prefilling)`, `:1233 build_attn_metadata_wrapper()`.
* MTP layer identification: `T vllm_ascend/utils.py:1617 is_mtp_layer(hf_config, layer_name)`.

**Zigzag insertion points (area 8)**
* The eligibility gate must key on the **draft flag already threaded** into the metadata hooks: `_build(...,
  draft_index)` (`sfa_v1.py:509`) → `_prepare_parallel_metadata(..., draft_index)` (`sfa_v1.py:423`,
  `sfa_cp.py:281`). Return the continuous layout whenever `draft_index is not None`. O instead had to add a
  `for_draft=True` kwarg through `builder.build(...)` in the proposer (O `llm_base_proposer.py:2132-2139`); in T
  the MTP path goes through `build_for_drafting` (`sfa_v1.py:483`) or plain `build` with a dummy third argument
  (`llm_base_proposer.py:2528`), so both entry points must be covered.
* The deleted O `patch_deepseek_mtp.py` logic is split across `T models/deepseek_mtp.py` (model + weight
  rewriting/`skip_prefixes`) and `T patch/worker/patch_deepseek_v2.py:38-58 _should_skip_indexer_init`
  (`is_mtp_layer`-based). Any MTP-specific zigzag exclusion has to be re-added to those two places, not to a
  patch file.
* V2 has a **second, independent draft loop** (`worker/v2/spec_decode/autoregressive/speculator.py:269`) →
  a second gate site (or a shared predicate consulted from `_prepare_parallel_metadata`).

**Changed vs O**
* `patch/worker/patch_deepseek_mtp.py` deleted; classes relocated to `models/deepseek_mtp.py` and registered in
  `models/__init__.py`.
* Draft metadata is now an explicit API (`build_for_drafting`) instead of a kwarg on `build`; the V1 proposer
  passes `self.runner.get_model()` as the third positional arg to `build(...)` (`llm_base_proposer.py:2528`,
  which T's SFA `build` ignores as `fast_build`, `sfa_v1.py:474-481`).
* New V2 speculator hierarchy (`worker/v2/spec_decode/{mtp,eagle,dflash,dflash2,dspark}/…`) with
  `build_attn_metadata_wrapper` / `disable_target_pcp_for_replicated_draft` — no counterpart in O.

---

## 9. Env-flag registry and patch registration

**Target sites**

* `T vllm_ascend/envs.py` (103 lines): `:31 def _strict_binary_env(name, default="0") -> bool`,
  `:38 env_variables: dict[str, Callable[[], Any]] = {` … `:92 # end-env-vars-definition`,
  `:95 def __getattr__(name)` (lazy evaluation of `env_variables[name]()`), `:102 def __dir__()`.
  New flag = one commented dict entry, e.g. `"VLLM_ASCEND_X": lambda: int(os.getenv("VLLM_ASCEND_X", "<d>"))`
  or `_strict_binary_env("VLLM_ASCEND_X")`. No code change to `__getattr__` needed.
  **All five feature flags are absent from T** (`grep VLLM_ASCEND_CP_BALANCE` → 0 hits; O `envs.py:85`
  `VLLM_ASCEND_CP_BALANCE`, `:90 _MIN_TOKENS`, `:96 _REDUCE_MODE`, `:102 _DEBUG`, `:109 _EMBED_LOCAL`).
* Patch registration is **module-import based**; there is no `_PATCHES` dict and no `maybe_apply` anywhere in T
  (grep = 0). The chain: `T vllm_ascend/utils.py:638 adapt_patch(is_global_patch=False)` imports
  `vllm_ascend.patch.worker` (worker start) or `vllm_ascend.patch.platform` (pre-worker);
  `T patch/worker/__init__.py` (77 lines) is a flat list of `import vllm_ascend.patch.worker.patch_X  # noqa`
  under `if HAS_TRITON:` / `if get_current_hardware_profile().supports(HardwareCapability.STANDARD_WORKER_PATCHES):`
  guards (`:45` for `patch_deepseek_v2`); `T patch/platform/__init__.py` (102 lines) is the analogous global list.
  `T patch/__init__.py` (1386 lines) is a **documentation-only index** of numbered
  `# ** N. File: worker/patch_x.py**` sections with `Why:` / `How:` / `Future Plan:` blocks (32 entries).
* Applied patches monkey-patch at import time, e.g. `T patch/worker/patch_deepseek_v2.py:298
  DeepseekV2MLAAttention.__init__ = _deepseek_v2_mla_attention_init` and `:365 DeepseekV2Model.forward =
  _patched_forward`.

**Zigzag insertion points (area 9)**
* Env vars: append the 5 dict entries in `envs.py` inside the `env_variables` block (before `:92`) — mechanically
  unchanged from O. Keep the names (`VLLM_ASCEND_CP_BALANCE`, `_MIN_TOKENS`, `_REDUCE_MODE`, `_DEBUG`,
  `_EMBED_LOCAL`) if the run scripts/`docs/*.md` are to stay valid.
* Patches: if the port needs a new patch module (e.g. a zigzag variant of a `vllm/*` function), add
  `import vllm_ascend.patch.worker.patch_<name>  # noqa` to `T patch/worker/__init__.py` **and** a numbered doc
  block in `T patch/__init__.py`. No registry object to update.
* Feature-global runtime knobs (`VLLM_ASCEND_CP_BALANCE_REDUCE_MODE`) are read through `vllm_ascend.envs`, which
  is imported as `from vllm_ascend import envs as ascend_envs` in O; T has no `envs_vllm` re-export in
  `envs.py` (T imports `vllm.envs` directly where needed, e.g. `mrv2_utils.py:19`).

**Changed vs O**
* `envs.py` pattern identical; O only had 148 lines because of the feature's additions.
* O's `patch/__init__.py` was 1236 lines and contained the same doc-index style; T rewrote/extended it to 1386
  lines. Neither has a `maybe_apply`-style registry, so the task's assumption of `_PATCHES` dict entries does not
  match either tree (explicit negative finding).
* `T patch/worker/__init__.py` gained hardware-profile branching and the `patch_v2/*` family; O's tail imports
  (`patch_draft_quarot`, `patch_npugraph_ex_triton`, `patch_deepseek_mtp`) are gone.

---

## 10. Custom-op registration and device ops

**Target sites**

* `T vllm_ascend/ops/register_custom_ops.py` (280 lines). Pattern: define `_x_impl`, define
  `_x_impl_fake` (or pass a lambda fake), then
  `direct_register_custom_op(op_name="x", op_func=_x_impl, fake_impl=_x_impl_fake, mutates_args=[],
  dispatch_key="PrivateUse1")` (`from vllm.utils.torch_utils import direct_register_custom_op`, `:10`).
  Existing registrations: `:226` `maybe_all_gather_and_maybe_unpad`, `:234` `maybe_pad_and_reduce`,
  `:242` `maybe_all_reduce_tensor_model_parallel`, `:250` `maybe_all_reduce_shared_expert`, `:258` `quantize`,
  `:266` `npu_rotary_embedding`, `:274` `muls_add`. Imported once by `T vllm_ascend/ops/__init__.py:23`.
  Call convention elsewhere: `torch.ops.vllm.<op_name>(...)` (e.g. `prepare_finalize.py:420/529`,
  `vocab_parallel_embedding.py:285`).
* `T vllm_ascend/device/device_op.py` (1536 lines): `:40 class BaseDeviceAdaptor` (static/class methods),
  `:793 class A5DeviceAdaptor(BaseDeviceAdaptor)`, `:1480 class Ascend310PDeviceAdaptor(BaseDeviceAdaptor)`,
  `:1536 DeviceOperator: type["BaseDeviceAdaptor"] = get_device_adaptor()`. Relevant kernels:
  `:409 BaseDeviceAdaptor.execute_sparse_flash_attention_process(cls, sfa_impl, ql_nope, q_pe, kv_cache,
  topk_indices, attn_metadata, actual_seq_lengths_query, actual_seq_lengths_key, block_table=None)`,
  `:480 _execute_kv_quant_sparse_flash_attention`, `:338 indexer_select_post_process` (A5 `:1308`),
  `:519 dsa_kv_compress_scatter`, `:534 indexer_quant_scatter`, `:632 unpack_dsa_indexer_kv_cache`,
  `:640 unpack_dsa_forward_kv_cache`.
  Adding a device op = add the static/classmethod to `BaseDeviceAdaptor` (+ A5/310P override only when the
  hardware differs); `DeviceOperator` resolves the class.

**Zigzag insertion points (area 10)**
* No new custom op is required for the zigzag plan itself (O's `layers/cp_zigzag.py` used plain torch/npu ops and
  index_select). If a zigzag reduce-scatter must be graph-safe it should become a `direct_register_custom_op`
  entry here (T `:226+`), mirroring `maybe_pad_and_reduce`.
* The block-table override needed by the zigzag sparse-attention call **already exists** as the optional
  `block_table=` parameter of `execute_sparse_flash_attention_process` (`T device_op.py:409/420-425`).
  The old feature had to add it (`O`-only change vs B, see B→O diff of `device_op.py`); it must be re-added only
  for `indexer_select_post_process` (`T device_op.py:338` and A5 `:1308`), which still hard-codes
  `block_table=attn_metadata.block_table` (`:372/387/400`, A5 `:1342`).

**Changed vs O**
* `device_op.py` shrank 1914 → 1536 lines; `mla_preprocess_only_decode` / `sfa_preprocess_with_mlapo` style
  helpers were removed from the adaptor (O still had `sfa_preprocess_with_mlapo`), the preprocess logic now lives
  in the attention impls (`sfa_v1.py:1313 _sfa_preprocess_mlapo`, `:1199 _sfa_preprocess_prolog_v3`).
* New `clipped_swiglu` (`:311`), `apply_dsa_q_rms` (`:614`), `pad_dsa_decode_slot_mapping` (`:658`) etc.
* `register_custom_ops.py` went 259 (B) → 293 (O) → 280 (T); the ops the feature patched
  (`maybe_pad_and_reduce`, O `:116-121`) now have completely different semantics in T (EP, not SP).

---

## Cross-cutting: what makes the old hook untransplantable as-is (ranked)

1. **V2 model runner is the default for the GLM-5.2 architecture.**
   `T mrv2_utils.py:33-45 DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES` contains `"GlmMoeDsaForCausalLM"`,
   `"DeepseekV32ForCausalLM"`, `:83 is_default_v2_model_runner_model`, `:178-200 use_v2_model_runner` (explicit
   `VLLM_USE_V2_MODEL_RUNNER` still wins). O hard-disabled zigzag whenever
   `envs_vllm.VLLM_USE_V2_MODEL_RUNNER` (O `ascend_forward_context.py:216-235`, comment "zigzag is never active
   under the V2 model runner"). That escape hatch cannot survive; the port either implements zigzag in the V2 path
   or forces `VLLM_USE_V2_MODEL_RUNNER=0` (and then loses default-GLM-5.2 support/tests).
2. **DSA-CP sharding is duplicated in two builders, and the old single patch point is gone.**
   `T sfa_cp.py:281` (SFA) and `T indexer.py:753/794` (indexer) both compute `num_tokens_pad` /
   `local_start` / `local_end` independently; O computed it once inside `AscendSFAMetadataBuilder._build`
   (O `sfa_v1.py:577-600`) and derived the indexer slot order from the same `dsa_cp_context`
   (O `sfa_v1.py:2568-2584`). A port that hooks only `sfa_cp.py` will write the indexer cache to the wrong slots.
3. **The exit hidden-state gather no longer exists in the model runner.**
   `T worker/model_runner_v1.py:3091-3141 _model_forward` returns hidden states untouched;
   `_all_gather_hidden_states*` deleted (only stale doc references at `T profiling_config.py:286-289`), and
   `flash_comm_v1_enabled` / `pad_size` deleted from the forward context (0 grep hits;
   O `ascend_forward_context.py:179-184`, O `model_runner_v1.py:2601-2609`). The gather now lives in
   `T patch/worker/patch_deepseek_v2.py:348-354`, so the O design (skip in model, gather in runner) must be
   inverted.
4. **The indexer was extracted from `AscendSFAImpl` into its own backend+impl+builder.**
   `T attention/indexer.py:92 AscendSFAIndexerBackend` (impl `:166/220/279/373`, builder `:471/753/779/794`),
   invoked from `T sfa_v1.py:1780` with its own metadata fetched by `:1594 _get_indexer_attn_metadata()`.
   O patched `indexer_select_pre_process` / `_indexer_qk_proj` / `indexer_select_post_process` /
   `_maybe_gather_kv_for_dsacp` / `_maybe_store_kvcache_for_c8_n_dsacp` **inside** `AscendSFAImpl`
   (O `sfa_v1.py:1902/1951/2014/2112/2183`) — none of those symbols exist in T.
5. **The SP/TP reduction plumbing the feature patched was deleted or repurposed.**
   `SequenceColumnParallelOp` / `SequenceRowParallelOp` removed from `T ops/linear_op.py` (0 hits; O
   `linear_op.py:302/327` were the patched classes), `maybe_pad_and_reduce` / `maybe_all_gather_and_maybe_unpad`
   became EP collectives (`T register_custom_ops.py:57/89`, `T ops/fused_moe/prepare_finalize.py:420/529`),
   `distributed/utils.py` lost the feature's whole fixed-order reduction block (T file = 29 lines;
   O `distributed/utils.py:24-150`), and `ops/fused_moe/experts_selector.py` was deleted (router split into
   `T ops/fused_moe/router/*` with `input_ids` passed explicitly at `fused_topk_router.py:145`).
6. **MTP/draft handling changed shape twice over.**
   `T patch/worker/patch_deepseek_mtp.py` deleted (O `patch/worker/patch_deepseek_mtp.py:86/122/130`), MTP model
   classes moved to `T models/deepseek_mtp.py:16/74` and registered in `T models/__init__.py:73`; draft metadata
   is now built by an explicit `build_for_drafting` (`T attention/sfa_v1.py:483`, called from
   `T spec_decode/llm_base_proposer.py:2522`) instead of a `for_draft=True` kwarg (O
   `llm_base_proposer.py:2132-2139`); and the V2 runner has its own draft loop
   (`T worker/v2/spec_decode/autoregressive/speculator.py:269-300`, `T worker/v2/spec_decode/mtp/speculator.py:21`).
   V1 `build_draft_attn_metadata` even passes the model object as `fast_build`
   (`T llm_base_proposer.py:2528`) — any new kwarg must tolerate that call shape.

## Upstream PCP as the intended long-term home (recommended reading for the port)

T/U already ship a generic prefill-CP whose chunking is a per-request zigzag:

* `U vllm/v1/worker/gpu/pcp_manager.py:195-229 _iter_rank_chunks` — `num_chunks = 2 * pcp_world_size`,
  rank *r* takes chunk *r* and chunk *2W-1-r* (docstring shows the PCP=4 layout), decodes replicated.
* `U .../pcp_manager.py:231-259 _get_rank_segments` / `:164-193 _reorder_segments` (pure prefills pushed last).
* `U .../pcp_manager.py:261-333 _build_batch_layout` builds `hidden_restore_idx`, `padded_gather_idx`,
  `gathered_kv_write_mask` — functionally the same trio as O's `inv_gather_index` / `zigzag_gather_index` /
  padded row mask (`O layers/cp_zigzag.py:399-541`, consumed at O `sfa_v1.py:797-825`).
* Ascend overrides: `T worker/v2/pcp_manager.py:36-48 AscendPCPAttentionContext`, `:240-354 partition_batch`,
  `:355 restore_hidden_states`, `:428 prepare_slot_mappings`, `:468 build_attention_context`;
  SFA consumes it via `T attention/context_parallel/sfa_cp.py:63 AscendSFAPCPImpl` /
  `:245 AscendSFAPCPDCPMetadataBuilder` and `pcp_slot_mapping` (`T sfa_v1.py:336/526/570`).
* A cp_balance port could therefore be expressed as a **new PCP chunking policy** (balanced across requests
  instead of per-request ceil-chunks) instead of a new parallel layout — but note the two limitations already
  encoded in U (`U .../pcp_manager.py:150-161`): sparse-MLA PCP rejects CUDA graphs
  (`NotImplementedError` for `hf_text_config.index_topk` + cudagraphs) and requires PIECEWISE graphs.

## Uncertainty / open items (explicitly not verified)

* `T worker/v2/model_runner.py` inherits `sample()` from `NPUModelRunner(GPUModelRunner)`
  (`T worker/v2/model_runner.py:84`, `U vllm/v1/worker/gpu/model_runner.py:1461`); I did not trace the whole V2
  execute path, so the exact place where PCP/V2 restores hidden states before `sample()` is inferred from
  `T worker/v2/pcp_manager.py:355 restore_hidden_states` and `U .../model_runner.py:879`, not read end-to-end.
* `T spec_decode/llm_base_proposer.py:2528` passes `self.runner.get_model()` into `build()`'s third slot
  (`fast_build`); T's SFA `build` ignores it (`sfa_v1.py:474-481`), but I did not verify every other builder that
  this path can reach (DSA/dflash/eagle) tolerates it.
* I did not check the 310P lane (`T _310p/**`) for any of the 10 areas; all findings are for the 910(A5/A3) path.
