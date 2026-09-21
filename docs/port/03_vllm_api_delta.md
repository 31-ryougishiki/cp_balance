> 历史文档（2026-09-18 移植期取证/评审）：结论可能已过时；dp>1 那道门与 zigzag 的现状见 `docs/scripts_review.md` §4.1/§4.1c/§8。
>
> 已变事实（2026-09-21 更正）：
> - §0 的「不要用本地 `cp_balance` 分支（被 force-reset、mid-merge 冲突）」已过期：现 b64c9569b、工作区干净。
> - §4 的「Ordered hard-break checklist for the porter」1~11 项已在移植时执行完；harness 的 vllm pin 若从 84030bbe3d 上移（上游 main=9a70c233cd）需按新 pin 重做 delta。

# 03 — Upstream vLLM API delta for the cp_balance port

Scope: which *upstream vLLM* APIs/behaviour changed between the commit the cp_balance feature was
written against and the commit the target vllm-ascend branch is verified against, restricted to the
areas the feature touches (attention metadata/backends, model runner v1/v2 + worker plumbing,
DCP/CP, distributed collectives, custom-op registration, quantization/linear base classes,
`VocabParallelEmbedding`, `vllm.envs`, `vllm.config`, spec-decode/MTP, sampler/logits).

## 0. Refs, method, notation

* `OLD` = **568afb3a13** (`refs/tags/v0.26.0`, `refs/remotes/origin/releases/v0.26.0`, 2026-07-26) in `D:\code\cp_balance\vllm`.
* `NEW` = **84030bbe3d** (vLLM main @ 2026-09-11) in `D:\code\cp_balance\vllm`. 2241 commits between the two.
* `OLD_FEAT` = **23ff2c23c** — the *old* vllm-ascend feature tip. ~~**Do not use the local `cp_balance`
  branch for this**~~ **（这条警告已过期：现行 `cp_balance` = b64c9569b 的移植版、工作区干净；
  只有需要看 pre-port 实现时才去 `cp_balance_v0.26.0rc` = `23ff2c23c`。）**
* `TGT` = **aff1b74b6** (`refs/remotes/upstream/main`, the port target) — dual-lane: it still carries
  gates for the v0.28.0 release lane and the vLLM-main lane (`vllm_ascend/utils.py:706` `vllm_version_is()`;
  `vllm_ascend/worker/npu_input_batch.py:39-44` "main2main compat"; `vllm_ascend/core/kv_cache_interface.py:114`
  "vLLM main removed this field from AttentionSpec").
* Line numbers are `path:line` and were read with `git show <ref>:<path>`; all paths are relative to the
  vllm repo root unless prefixed. Hard = symbol deleted/renamed/signature change (import-time or
  call-time failure). Soft = additive kwarg, new optional hook, or behaviour/default change.
* Method: `git diff --stat 568afb3a13 84030bbe3d -- <dir>` (416 files, +69k/−15k) → signature-level
  extraction per file (class/method signature diff, dataclass field diff) → cross-check against what
  `vllm-ascend@23ff2c23c` actually imports/calls (`git grep`).

The single most important structural fact for this port: **the feature was written against a
single-lane vLLM (0.26.0) while `TGT` is a two-lane tree (v0.28.0 + main)**. Almost every hard break
below already has an `if vllm_version_is("0.28.0")` precedent in `TGT`, so the porter should decide
per-site whether to add a gate or drop the 0.26.0 path.

## 1. Summary table

| area | old API (file:line @ OLD) | new API (file:line @ NEW) | breaking? | what the porter must do |
|---|---|---|---|---|
| attention backend iface | `AttentionBackend.get_kv_cache_shape` `vllm/v1/attention/backend.py:90` | **removed** (only stray `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py:144`); zero call sites left in vLLM | **HARD** | Ascend backends may keep their own `get_kv_cache_shape` for their own callers, but nothing in vLLM calls it; KV shape/layout now comes from the spec (`customize_spec`/`create_kv_cache_views`) |
| attention backend iface | `get_kv_cache_block_dim` `:100`, `get_kv_cache_stride_order` `:120`, `indexes_kv_by_block_stride` `:206` | all **removed** | **HARD** | move Ascend block-stride/layout knowledge into the Ascend `KVCacheSpec` (as `TGT` already does) |
| attention backend iface | `get_required_kv_cache_layout` `:396` | `supported_kv_cache_layouts` `:360` | **HARD** | rewrite Ascend backend layout declarations; layout is now engine-resolved (`config/cache.py:363` `get_resolved_kv_cache_layout`) |
| attention backend iface | (none) | `customize_spec(cls, spec)` `:136`; `supports_device_cpu_query_lens_mismatch` `:208`; `supports_dcp` `:230`; `supports_non_causal_dcp` `:237` | SOFT (new hooks) | implement `supports_dcp`/`customize_spec` where the Ascend CP path needs them |
| attention backend iface | `validate_configuration(...)` `:320` | `:270` + `use_adaptive_verification`, `use_dcp` `:288-289`; raises if `use_non_causal and use_dcp and not supports_non_causal_dcp()` `:326` and if `use_dcp and not supports_dcp()` `:334` | SOFT but semantic | any Ascend backend used with DCP must set `supports_dcp` truthy or startup validation now fails |
| attention impl default | `AttentionImplBase.supports_dcp = True` `:823` | `= False` `:822` | SOFT default flip, **behaviour-breaking** for DCP | audit every Ascend impl that relied on the default |
| attention metadata | `AttentionMetadata` `:404` (no `max_logits_per_req`) | `:370` + `max_logits_per_req: int \| None = None` `:411`, + `replayssm_decode_base_cpu` `:450` | SOFT | nothing to change unless slicing metadata (`as_slice` passes both, `:562`,`:569`) |
| attention metadata | `CommonAttentionMetadata` `:412`; `token_to_req_indices` from **CPU** `np.repeat` | `:378`; now from **device** `query_start_loc` (`torch.repeat_interleave`, `:525-533`) | SOFT, behaviour | any Ascend builder that assumed `query_start_loc_cpu` boundaries must be re-validated (adaptive-verification trimming) |
| metadata builder | `AttentionMetadataBuilder.__init__` `:630`, `kv_cache_spec: "AttentionSpec"` | `:611`, `kv_cache_spec: "KVCacheSpec"`, `self.kernel_block_size` `:622` | SOFT | annotation/typing only; update Ascend builders' type hints |
| metadata builder | (none) | `requires_block_table_width` `:605`, `supports_draft_decode_metadata_update` `:608`, `set_kernel_block_size()` `:624`, `update_draft_decode_metadata()` `:737` | SOFT | implement `update_draft_decode_metadata` if the Ascend MTP/eagle draft path wants one-build-per-draft-step |
| builder construction | `AttentionGroup.create_metadata_builders` `vllm/v1/worker/utils.py:243-277` (4 positional args) | `:257-…` passes `block_table_width=` when `requires_block_table_width` and uses `MLAAttentionSpec.storage_block_size` | SOFT | Ascend builders constructed via `AttentionGroup` must accept the new kwarg or keep `requires_block_table_width=False` |
| KV layout type | `KVCacheLayoutType = Literal["NHD","HND"]` `vllm/v1/attention/backends/utils.py:42` | `class KVCacheLayout(Enum)` in **new module** `vllm/v1/kv_cache_layout.py:15` | **HARD** (module move + type change) | switch imports; string layout names now go through `resolve_kv_cache_layout` |
| KV layout helpers | `is_valid_kv_cache_layout` `vllm/v1/attention/backends/utils.py:78`, `get_kv_cache_layout` `:83`, `set_kv_cache_layout` `:112`, `subclass_attention_metadata` `:774` | **removed**; new `get_supported_kv_cache_layouts` `:190`, `record_kv_cache_layout` `:228`, `resolve_kv_cache_layout` `:240`, `get_flashinfer_layout_string` `:165` | **HARD** | replace call sites; `subclass_attention_metadata` has no replacement |
| KV spec fields | `AttentionSpec.indexes_kv_by_block_stride` `vllm/v1/kv_cache_interface.py:182` (used `:306`,`:426`) | **removed** | **HARD** | already mirrored in Ascend spec (`TGT vllm_ascend/core/kv_cache_interface.py:114-117`) |
| KV spec fields | `KVCacheSpec.storage_block_size` `:119`; `FullAttentionSpec.real_page_size_bytes` `:328`; `MLAAttentionSpec.storage_block_size` `:394`, `real_page_size_bytes` `:398`; `UniformTypeKVCacheSpecs.get_page_sizes` `:866`, `get_num_layer_tuples` `:869` | `KVCacheSpec.state_content_size_bytes` `:173`, `num_states` `:187`, `get_num_kernel_states` `:190`; `AttentionSpec.state_content_size_bytes` `:416`, `real_page_size_bytes` `:434`; `UniformTypeKVCacheSpecs.get_max_layers_per_page_size` `:1144` | **HARD** for the page-size/layout math | re-derive Ascend page-size accounting for the new names; `real_page_size_bytes` no longer exists on `FullAttentionSpec`/`MLAAttentionSpec` |
| KV tensors | `KVCacheTensor.shared_by` | `KVCacheTensor.layers` `:1258` + **new required** `layer_stride` `:1259` | **HARD** | `TGT vllm_ascend/utils.py:733` `get_kv_cache_tensor_layers()` already bridges this; keep the pattern |
| cache config | `CacheConfig.calculate_kv_scales` `vllm/config/cache.py:111` (+validator `:262`) | **removed**; `kv_cache_layout` `:89`, `get_resolved_kv_cache_layout()` `:363` | **HARD** | delete `calculate_kv_scales` reads; `TGT vllm_ascend/platform.py:689-693` already does it defensively (`getattr`) |
| KV cache views | (none) | `create_kv_cache_views`, `compute_layout_strides`, `compute_layer_kv_cache_shape_bytes`, `group_kernel_blocks` `vllm/v1/kv_cache_interface.py` (new); `vllm/v1/worker/gpu/attn_utils.py:318 init_kv_cache` | SOFT (new entry points) | use these instead of hand-rolled `_allocate_kv_cache_tensors`/`_reshape_kv_cache_tensors` |
| MLA/SFA metadata | `MLACommonPrefillMetadata.ChunkedContextMetadata` `vllm/model_executor/layers/attention/mla_attention.py:1331` (`seq_tot`,`max_seq_lens`,`cu_seq_lens_lst`,`chunk_size`,`prefill_tokens_with_context`,`padded_local_chunk_seq_lens`,`workspace`,`chunk_total_token`,`has_empty_context`) | `:1533` restructured: `context_lens`, `workspace`, `chunks: list[ContextChunk]`, `context_lens_list`, `empty_token_slices`, `dcp_manager` | **HARD** for SFA/DSA-CP metadata | `AscendSFAMetadataBuilder` + `sfa_cp.py` must be rewritten against the chunk plan API; this is the core of the port |
| MLA/SFA metadata | `MLACommonMetadata` `:1377` | `:1599` + `query_lens_cpu`, `use_dense_mha`, `topk_mask_workspace`, `causal` | SOFT additive | non-zigzag path unchanged; zigzag must respect `causal` |
| MLA builder | `MLACommonMetadataBuilder` `:1666`, `__init__:1761`, `build():1884` | `:2158`, `__init__:2283`, `build():2427` (same signature), + `supports_non_causal_multi_token_decode` `:2175`, `supports_non_causal_multi_token_dcp` `:2178`, `_validate_dspark_dcp_support()` `:2187` | SOFT sig / **HARD** body | subclass hooks that used `ChunkedContextMetadata` internals break; DSpark+DCP now raises unless a flag is set |
| MLA helpers | `build_mla_chunked_context_metadata` `:1487` | `:1932` + new `plan_mla_context_chunks` `:1841`, `align_mla_chunked_context_workspace_size` `:1917`, `init_mla_context_partial` `:2687`, `accumulate_mla_context_chunk` `:2714`, `neutralize_empty_context_partials` `:2671` | **HARD** for CP chunking | these are the sanctioned CP hooks; the zigzag feature should hook here instead of patching builder internals |
| MLA backend/layer | `MLACommonBackend.get_kv_cache_shape` `:1301`, `get_kv_cache_stride_order` `:1311`; `MLAAttention.calc_kv_scales` `:1048` | removed; `MLACommonBackend.customize_spec` `:1502`; `MLAAttention.bind_kv_cache` `:693`, `_use_sparse_mha` `:1116` | **HARD** | see KV-spec row; drop `calc_kv_scales` overrides |
| attention op | `torch.ops.vllm.maybe_calc_kv_scales` registered `vllm/model_executor/layers/attention/attention.py:696,714,724`; `Attention.calc_kv_scales` `:584` | **removed entirely** (no `calc_kv_scales` anywhere in vLLM at NEW) | **HARD** (op disappears from the dispatcher) | remove/guard any `torch.ops.vllm.maybe_calc_kv_scales` call; `TGT` keeps only the config-side `getattr` guard |
| attention layer base | `AttentionLayerBase` `vllm/model_executor/layers/attention_layer_base.py:12` (35 lines) | `:14` (45 lines) + `bind_kv_cache(kv_cache)` hook `:26` | SOFT | implement on Ascend attention layers that want the vLLM binding path; + `is_deferred_attention_layer` `vllm/model_executor/layers/attention/__init__.py:22` |
| PCP module | `vllm/model_executor/layers/attention/pcp.py` (92 lines, `_gather_prefill_cache_inputs:11`, `maybe_gather_mla_latent_cache_inputs:48`, `maybe_gather_indexer_k:69`, `finalize_mla_pcp_decode:83`) | **moved verbatim** to `vllm/v1/attention/ops/pcp.py` (same lines) | **HARD** (module move) | `TGT` already dual-imports (`vllm_ascend/attention/attention_v1.py:69-71`, `utils.py:699-700`); reuse that shim |
| DCP ops | `vllm/v1/attention/ops/dcp_alltoall.py` (461 lines, `dcp_a2a_lse_reduce:392`) and CP helpers in `vllm/v1/attention/ops/common.py` (`CPTritonContext`, `correct_attn_out`, `cp_lse_ag_out_rs`, `cp_lse_ag_out_ar`) | **removed file**; merged into `vllm/v1/attention/ops/dcp.py` (1642 lines: `correct_attn_out:358`, `CPTritonContext:345`, `cp_lse_ag_out_rs:452`, `cp_lse_ag_out_ar:485`, `dcp_a2a_lse_reduce:922`, `mask_dcp_empty_shards_:71`, `get_dcp_workspace_max_num_tokens:996`, `DCPCombine:1427`, `MLADCPManager:1438`) + new `vllm/v1/attention/ops/cp_common.py` (`direct_cp_enabled:60`, `direct_cp_multicast_enabled:79`, `DirectCPWorkspace:90`) | **HARD** (imports) | vllm-ascend does **not** import these, but the pointer/OOB machinery (`MLADCPManager`, `get_dcp_workspace_max_num_tokens`) is the new sanctioned CP interface to mirror |
| DCP/CP envs | — | `VLLM_USE_DIRECT_DCP_A2A` `vllm/envs.py:2181`, `VLLM_USE_DIRECT_DCP_Q_GATHER` `:2184`, `VLLM_USE_DIRECT_DCP_KV_GATHER` `:2187`, `VLLM_DCP_Q_REPLICATE` `:1580`, `VLLM_REPLICATE_EMBED` `:633` | SOFT | informational; do not collide with the new `VLLM_ASCEND_CP_BALANCE*` names |
| DCP comm backend | (none) | `envs.VLLM_USE_DIRECT_DCP_A2A` + `ParallelConfig.dcp_comm_backend` path `vllm/config/parallel.py:555-560` | SOFT | check `dcp_comm_backend == "a2a"` validation still passes for the zigzag layout |
| DCP linear | (none) | `DCPGroupColumnParallelLinear` `vllm/model_executor/layers/linear.py:626` (uses `parallel_group`) | SOFT (new class) | useful precedent for group-scoped sharding |
| distributed | `get_dcp_group` `vllm/distributed/parallel_state.py:1376`; `get_dcp_group()` unchanged | `:1566`; new `get_etp_group` `:1557`, `suspend_device_comms` `:183`, `resume_device_comms` `:188`, `checkpoint_prepare_distributed_state` `:2241` | SOFT | `get_dcp_group` is safe; the `vllm_ascend/distributed/utils.py` docstring claim "helper removed on main" is about the *v0.21.0 `get_decode_context_model_parallel_*` wrappers*, not `get_dcp_group` |
| distributed | `GroupCoordinator.__init__` `:358-…`; `device_index` from `_WORLD.device_index` fallback `:400-407` | `:421-…`; +`use_all2all=False` `:458`; `self.device_index = local_rank` unconditional `:466`; +`isend_object` `:524`, `_pending_isends` `:576` | SOFT sig, **semantic** | any code relying on `world.device_index` inheritance changes; `all_gather`/`reduce_scatter`/`all_reduce`/`gather` signatures are unchanged |
| distributed | `vllm/distributed/communication_op.py` (43 lines) | identical (43 lines) | none | `tensor_model_parallel_all_gather/reduce_scatter/all_reduce`, `get_tp_group`, `divide` are safe |
| model runner v1 | `GPUModelRunner` `vllm/v1/worker/gpu_model_runner.py:452`; `AsyncGPUModelRunnerOutput:258` | `:498`; `AsyncGPUModelRunnerOutput:289` (class body 7381→7132 lines) | **HARD** for overrides | method removal list in §3.5 — 11 methods the Ascend runner may `super()`-call |
| model runner v1 | `post_kv_cache_wake_up:976`, `init_fp8_kv_scales:980`, `_init_xdrope_positions:1670`, `_calc_xdrope_positions:2774`, `_freeze_gc:6469`, `_allocate_kv_cache_tensors:7238`, `_reshape_kv_cache_tensors:7290`, `_has_mixed_attention_kv_layout:7419`, `_update_hybrid_attention_mamba_layout:7445`, `_get_attention_kv_cache_gid:7621`, `_bind_routed_experts_capturer:7677` | all **removed**; `initialize_kv_cache_tensors` `:7309`, `initialize_kv_cache` `:7381` remain | **HARD** | delete Ascend references; xdrope support is gone upstream so `TGT` keeps it only behind `vllm_version_is("0.28.0")` |
| model runner v1 | `GPUModelRunner.__init__`: `self.calculate_kv_scales`, `self.uses_xdrope_dim` | `self.mrope_num_dims`, `self.jit_warmup_registry`, `self.cp_kv_cache_interleave_size`, `self._mamba_state_copy_funcs` | **HARD** (attribute names) | rename; existing `TGT` `model_runner_v1.py:1417` pattern shows the gate |
| input batch | `CachedRequestState.xdrope_positions` `vllm/v1/worker/gpu_input_batch.py:49` | **removed** | **HARD** | same as above |
| input batch | `InputBatch.__init__` `:93` (`num_spec_tokens:103`, `cp_kv_cache_interleave_size:105`, `slot_mapping_modes` last) | `:90`; `use_replayssm: bool = False` inserted at `:107` **before** `slot_mapping_modes` | **HARD** if positional | `NPUInputBatch` already accepts both (`TGT worker/npu_input_batch.py:39-70`); never pass these positionally |
| runner v2 / worker plumbing | `WorkerBase.get_kv_cache_spec` `vllm/v1/worker/worker_base.py:98`; `vllm/v1/worker/gpu/model_runner.py:408,411,710` | `get_kv_cache_spec:103`; +`get_supported_kv_cache_layouts:107`, `set_kv_cache_layout:112`, `synchronize_device:128`, `supports_draft_weight_updates:148`; `initialize_kv_cache:561` (sig changed), `capture_model(*, profile_only=False):952`, `pcp_manager_cls:2209` | **HARD** for worker overrides | implement the new abstract-ish hooks; NPU worker must not rely on the removed "post-KV-cache wake" hook |
| runner v2 model states | `ModelState.get_mm_embeddings` `vllm/v1/worker/gpu/model_states/interface.py` | `prepare_inputs_embeds`; +`get_additional_cg_support`, `execute_mm_encoder` | **HARD** | rename overrides; new modules `model_states/{encoder_only,prompt_embeds,recoverssm}.py` |
| runner v2 attention utils | `vllm/v1/worker/gpu/attn_utils.py:50 get_kv_cache_spec`, `:520 init_kv_cache`; `_allocate_kv_cache`, `_reshape_kv_cache`, `_update_hybrid_attention_layout` | `:122 get_kv_cache_spec`, `:318 init_kv_cache`, `:269 get_attn_cg_support`, `FastPrefillHelper:71`; reshape helpers removed | **HARD** | Ascend v2 KV allocation must move to `init_kv_cache`/`create_kv_cache_views` |
| worker utils | `bind_kv_cache(kv_caches, forward_context, runner_kv_caches, num_attn_module=1)` `vllm/v1/worker/utils.py:482` | `:593` + `kv_cache_groups: Sequence[KVCacheGroupSpec] | None = None` | SOFT | pass groups where Ascend needs multi-group binding |
| worker utils | `AttentionGroup` `:243`, `select_common_block_size` `:282` | `:257`, `:328`; +`KVBlockZeroer.warmup`, `supports_draft_decode_metadata_update`, `update_draft_decode_metadata`, `allocate_kv_cache`, `clear_layer_kv_caches`, `get_uniform_decode_token_count`, `is_residual_scattered_for_sp` unchanged `:738` | SOFT | `is_residual_scattered_for_sp` (SP padding contract) is byte-identical — the zigzag SP-alignment assumption still holds |
| ubatch | `vllm/v1/worker/ubatch_utils.py` (265 lines) | (368) +`SMControlContextManager`, `create_sm_control_context`, `get_num_ubatches` | SOFT additive | none for the feature |
| custom op reg | `direct_register_custom_op` `vllm/utils/torch_utils.py:901` | `:1042` — **byte-identical signature and body** | none | no action; all `torch.ops.vllm.<ascend op>` registrations keep working |
| custom op classes | `PluggableLayer` `vllm/model_executor/custom_op.py:32`, `CustomOp:103`, `op_registry` `:19` | same lines/classes; only `assert`→`ValueError` at `:287` and `:302`(old)/`:306`(new) | none (soft) | `CustomOp`-based Ascend ops unaffected; note new parallel module `vllm/model_executor/hw_agnostic/custom_op.py` (318 lines) gated by `VLLM_USE_HW_AGNOSTIC` `vllm/envs.py:1225` |
| quantization base | `QuantizationConfig.is_mxfp4_quant` `vllm/model_executor/layers/quantization/base_config.py:262` (+`mxfp4.py:97`, `quark/quark.py:551`) | **removed**; +`get_checkpoint_weight_mapper`, module-level `resolve_quant_method` | **HARD** upstream, **no impact** (vllm-ascend never calls it) | keep Ascend MXFP4 detection local |
| linear methods | `UnquantizedLinearMethod` `vllm/model_executor/layers/linear.py:182`; `adjust_bitsandbytes_4bit_shard:97` | `:164`, +`supports_pre_processed_weights`, +`__init__()` that resolves the GEMM backend from `get_current_vllm_config_or_none()`; `adjust_bitsandbytes_4bit_shard` removed | SOFT sig / **HARD** for removed helper | `LinearMethodBase.create_weights/apply/process_weights_after_loading` are unchanged; just note the new `__init__` side effect |
| embedding | `VocabParallelEmbedding.__init__` `vllm/model_executor/layers/vocab_parallel_embedding.py:239`; `get_tensor_model_parallel_rank/size()` | `:249` + keyword-only `disable_tp`, `quant_method`, `parallel_group`; `tp_rank`/`tp_size` are instance attrs; `resolve_quant_method` replaces `quant_config.get_quant_method()`; `update_param_tp_status()` `:360`; `is_embedding_layer = not isinstance(self, ParallelLMHead)` | SOFT additive / **HARD** for `get_quant_method` callers | the feature's `_embed_partial`/`forward_zigzag_local` path only needs `tp_size` + `tensor_model_parallel_all_reduce` — still valid; do not assume `tp_size == global TP world size` (a `parallel_group` may override it) |
| logits | `LogitsProcessor` `vllm/model_executor/layers/logits_processor.py:24`, `forward:64`, `_gather_logits:85`, `_get_logits:138` | `:58`/`:98`/`:122`/`:180`; +`skip_gather: bool = False` (`:103` in `forward`, `:185` in `_get_logits`, early return `:189`); gather now conditional on `lm_head.tp_size > 1` | SOFT additive | Ascend sampler calling `LogitsProcessor.forward` positionally is fine; TP gather is now driven by the LM head's own `tp_size` |
| envs | `VLLM_TRITON_ATTN_USE_TD`, `VLLM_CPU_SGL_KERNEL`, `VLLM_TEST_FORCE_FP8_MARLIN`, `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM` | removed | none | none of them are used by the feature (`vllm_ascend/envs.py`, `ascend_forward_context.py`) |
| runner selection | feature gates on the raw env: `envs_vllm.VLLM_USE_V2_MODEL_RUNNER` (`vllm/envs.py:274`, `bool \| None`, `None` when unset) read in `vllm_ascend/attention/{attention_v1.py:81,dsa_v1.py:199,fa3_v1.py:18,mla_v1.py:79,sfa_v1.py:170}`, `ascend_forward_context.py:223,585,594` | vLLM resolves selection through `VllmConfig.use_v2_model_runner` (`vllm/config/vllm.py:550`→`:667`), which force-enables V2 for dspark / multi-KV-group DFlash / diffusion, and at NEW also for watermarking (`:669-673`) | SOFT, semantics | use the resolved property, as `TGT` does via `vllm_ascend/mrv2_utils.py::use_v2_model_runner` (`ascend_forward_context.py:38`, `models/deepseek_v4/mtp.py:68`); a raw-env gate can disagree with the runner actually instantiated |
| config | `AttentionConfig` `vllm/config/attention.py` (160 lines) | (200) +`__post_init__`, `resolve_indexer_kv_dtype` | SOFT | none for the feature |
| config | `SpeculativeConfig` `vllm/config/speculative.py` (1362 lines) | (1918) +`use_eagle_block_drop`, `use_multi_module_mtp`, `_maybe_override_draft_max_position_embeddings` | SOFT additive | `TGT` patches `SpeculativeConfig.__post_init__` for the main lane — keep the patch aligned |
| spec decode | `SpecDecodeBaseProposer.__init__(vllm_config, device, pass_hidden_states_to_model, runner=None)` `vllm/v1/spec_decode/llm_base_proposer.py` | **identical signature**; `vllm/v1/spec_decode/utils.py` and `metadata.py` also signature-identical | none | bodies changed only; the Ascend MTP proposer still constructs the same way |
| MTP models | `DeepSeekMTP` `vllm/model_executor/models/deepseek_mtp.py:249`, `DeepSeekMultiTokenPredictorLayer:83`, `_rewrite_spec_layer_name:530` | `:233`, `:69`, `:517`; module 560→547 lines, `_restore_full_token_layout_if_needed` removed | SOFT (line shifts) | keep patching by attribute name, not by line |
| eagle3 models | `Eagle3LlamaForCausalLM(LlamaForCausalLM)` `vllm/model_executor/models/llama_eagle3.py:272`; `Eagle3DeepseekV2ForCausalLM` `deepseek_eagle3.py:283` | `Eagle3LlamaForCausalLM(_Eagle3LlamaForCausalLMBase)` `:306`; `LlamaDecoderLayer(_Eagle3LlamaDecoderLayerBase)`; `deepseek_eagle3.py:275` | **HARD** if subclassing by base | Ascend EAGLE3 classes must not assume the old MRO |
| hidden states | `CacheOnlyAttentionBackend.get_kv_cache_shape` `vllm/model_executor/models/extract_hidden_states.py:127` | removed; `CacheOnlyAttentionLayer` itself unchanged | **HARD** for that backend | Ascend `model_runner_v1.py` `isinstance(..., CacheOnlyAttentionLayer)` checks (`23ff2c23c:3692,4626`) still work |
| sampler | `SamplingMetadata` `vllm/v1/sample/metadata.py` (41 lines) | identical (41 lines, no field delta) | none | no action |
| sampler | `build_logitsprocs(vllm_config, device, is_pin_memory, is_pooling_model, custom_logitsprocs=())` `vllm/v1/sample/logits_processor/__init__.py` | identical | none | no action |
| rejection sampler | `RejectionSampler`, `PLACEHOLDER_TOKEN_ID` `vllm/v1/sample/rejection_sampler.py` (955 lines) | identical signatures (953 lines) | none | `_prepare_inputs_padded_kernel`-style Ascend copies are unaffected |

## 2. Area details

### 2.1 Attention backend / metadata interfaces (`vllm/v1/attention/backend.py`, 1110→1109 lines)

Hard removals from `AttentionBackend` (class line 56→59):

* `get_kv_cache_shape` `56afb3a13:90` — deleted at NEW. **There is no caller left in vLLM**
  (`git grep "get_kv_cache_shape(" 84030bbe3d -- vllm` → only the stray definition
  `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py:144`); OLD had callers at
  `vllm/v1/worker/gpu/attn_utils.py:318`, `vllm/v1/worker/gpu_model_runner.py:7359`,
  `vllm/v1/worker/kv_connector_model_runner_mixin.py:209`, `vllm/distributed/kv_transfer/kv_connector/utils.py:428`.
* `get_kv_cache_block_dim` `:100`, `get_kv_cache_stride_order` `:120`, `indexes_kv_by_block_stride` `:206`,
  `get_required_kv_cache_layout` `:396` — all deleted.
* Replaced by: `customize_spec` `:136` ("temporary compatibility API", backend adjusts the
  `AttentionSpec`), `supported_kv_cache_layouts` `:360` (preference-ordered tuple or `None`),
  `supports_dcp` `:230`, `supports_non_causal_dcp` `:237`,
  `supports_device_cpu_query_lens_mismatch` `:208`.
* `validate_configuration` (`:320`→`:270`) gained `use_adaptive_verification`/`use_dcp` (`:288-289`) and now *fails fast*
  (`:326`,`:334`) when DCP is requested from a backend that does not declare `supports_dcp`. With the
  `AttentionImplBase.supports_dcp` default flipped `True`→`False` (`:823`→`:822`), an Ascend backend that
  never overrode it silently becomes "DCP unsupported" at 84030bbe3d.
* New opt-in hooks on `AttentionImplBase`: `supports_dense_mha_prefill` (`:795`).

`AttentionMetadataBuilder` (`:617`→`:593`):

* `__init__(kv_cache_spec, layer_names, vllm_config, device)` (`:630`→`:611`) still has the same 4 params; the
  annotation widened `AttentionSpec`→`KVCacheSpec` and `self.kernel_block_size: int | None = None` was
  added (`:622`), with a setter `set_kernel_block_size()` (`:624`).
* New class attributes `requires_block_table_width` (`:605`) and
  `supports_draft_decode_metadata_update` (`:608`), plus `update_draft_decode_metadata()` (`:737`,
  "must emit capture-safe operations … keep replayed tensor state in persistent storage").
  These exist because the draft loop can now rebuild metadata once and update it in place per draft step
  — relevant to the Ascend MTP draft path (`vllm_ascend/patch/worker/patch_deepseek_mtp.py`).

`AttentionMetadata` (`:404`→`:370`) and `CommonAttentionMetadata` (`:412`→`:378`):

* Added `max_logits_per_req: int | None = None` (`:411`) and
  `replayssm_decode_base_cpu` (`:450`); `as_slice()` forwards both (`:562`,`:569`).
* `_as_token_to_req_indices` now derives `query_lens` from the **device** `query_start_loc`
  (`torch.repeat_interleave`, `:525-533`) instead of the CPU copy (`np.repeat`, OLD `:546-556`).
  This is specifically for adaptive verification, which trims drafts on device. Any Ascend metadata
  builder that slices/reorders using CPU boundaries must be re-checked.

### 2.2 KV-cache layout, specs and shapes

* `KVCacheLayoutType` (a `Literal`) lived in `vllm/v1/attention/backends/utils.py:42`; at NEW the type is
  `class KVCacheLayout(Enum)` in the **new** module `vllm/v1/kv_cache_layout.py:15`, and layout selection
  is an engine-level, cached decision: `CacheConfig.kv_cache_layout` (`vllm/config/cache.py:89`) →
  `CacheConfig.get_resolved_kv_cache_layout()` (`:363`) → `resolve_kv_cache_layout()` (`backends/utils.py:240`),
  recorded by `record_kv_cache_layout()` (`:228`). `WorkerBase` grew
  `get_supported_kv_cache_layouts()`/`set_kv_cache_layout()` (`vllm/v1/worker/worker_base.py:107,112`).
  `TGT` already uses this (`vllm_ascend/worker/worker.py:661-665`, `model_runner_v1.py`).
* `vllm/v1/attention/backends/utils.py` helpers `is_valid_kv_cache_layout:78`, `get_kv_cache_layout:83`,
  `set_kv_cache_layout:112` and `subclass_attention_metadata:774` are gone; `subclass_attention_metadata`
  has **no** replacement (it was the helper that made an algorithm-specific `AttentionMetadata` subclass).
  Ascend metadata subclasses (`AscendCommonAttentionMetadata` in `vllm_ascend/attention/utils.py:226`)
  are defined directly, so they are unaffected — but any helper call site would break.
* `vllm/v1/kv_cache_interface.py` (977→1369 lines): `AttentionSpec.indexes_kv_by_block_stride` (`:182`,
  consumed at `:306`,`:426`) removed; `KVCacheSpec.storage_block_size` (`:119`) replaced by
  `num_states` (`:187`)/`get_num_kernel_states(kernel_block_size)` (`:190`) and
  `state_content_size_bytes` (`:173`); `real_page_size_bytes` moved off `FullAttentionSpec` (`:328`)
  and `MLAAttentionSpec` (`:398`) onto `AttentionSpec` (`:434`);
  `UniformTypeKVCacheSpecs.get_page_sizes`/`get_num_layer_tuples` (`:866`,`:869`) →
  `get_max_layers_per_page_size` (`:1144`); `KVCacheTensor.shared_by` → `layers` (`:1258`) + `layer_stride`
  (`:1259`); new helpers `create_kv_cache_views`, `compute_layout_strides`,
  `compute_layer_kv_cache_shape_bytes`, `group_kernel_blocks`, `iter_layer_specs`, `is_full_attention_spec`.
* `TGT` mirrors the Ascend-specific parts in `vllm_ascend/core/kv_cache_interface.py:114-140`
  (`indexes_kv_by_block_stride` kept as an Ascend contract; `storage_block_size` property kept only for
  the `vllm_version_is("0.28.0")` lane) and `vllm_ascend/utils.py:733` (`get_kv_cache_tensor_layers`,
  `shared_by`→`layers`).

### 2.3 MLA / SFA metadata — the part the zigzag feature actually rewrites

`vllm/model_executor/layers/attention/mla_attention.py` (2603→3240 lines) is the biggest single
blocker. The chunked-context representation was replaced:

* OLD `MLACommonPrefillMetadata.ChunkedContextMetadata` (`:1331`, fields
  `@classmethod chunked_prefill_workspace_size`, `seq_tot`, `max_seq_lens`, `cu_seq_lens_lst`,
  `chunk_size`, `prefill_tokens_with_context`, `padded_local_chunk_seq_lens`, `workspace`, `chunk_total_token`,
  `has_empty_context`).
* NEW (`:1533`) `MLACommonPrefillMetadata.ContextChunk` dataclass + `ChunkedContextMetadata`
  (`context_lens`, `workspace`, `chunks`, `context_lens_list`, `empty_token_slices`,
  `dcp_manager: MLADCPManager | None`).
* New planning/accumulation API used by the CP path: `plan_mla_context_chunks` (`:1841`),
  `align_mla_chunked_context_workspace_size` (`:1917`), `build_mla_chunked_context_metadata` (`:1932`),
  `neutralize_empty_context_partials` (`:2671`), `init_mla_context_partial` (`:2687`),
  `accumulate_mla_context_chunk` (`:2714`).
* `MLACommonMetadata` (`:1377`→`:1599`) gained `query_lens_cpu`, `use_dense_mha`, `topk_mask_workspace`,
  `causal`.
* `MLACommonMetadataBuilder` (`:1666`→`:2158`): `__init__` (`:1761`→`:2283`) and
  `build()` (`:1884`→`:2427`) signatures unchanged (`common_prefix_len, common_attn_metadata, fast_build`); new class flags
  `supports_non_causal_multi_token_decode` (`:2175`) / `supports_non_causal_multi_token_dcp` (`:2178`) and
  a hard validation `_validate_dspark_dcp_support` (`:2187`) that raises for `method == "dspark"` +
  `decode_context_parallel_size > 1` unless the backend opts in.
* `MLACommonBackend` lost `get_kv_cache_shape` (`:1301`) and `get_kv_cache_stride_order` (`:1311`) and
  gained `customize_spec` (`:1502`); `MLAAttention` lost `calc_kv_scales` (`:1048`) and gained
  `bind_kv_cache` (`:693`) and `_use_sparse_mha` (`:1116`).

Porter implication: `AscendSFAMetadataBuilder` / `AscendSFADCPMetadataBuilder` in
`vllm_ascend/attention/sfa_v1.py` + `attention/context_parallel/sfa_cp.py` currently build the OLD chunk
structures; they must be re-expressed on `ContextChunk`/`plan_mla_context_chunks`, and the zigzag plan
(which the feature stores in `DSACPContext` and reads through `_EXTRA_CTX.zigzag_cp_context`,
`vllm_ascend/ascend_forward_context.py`, `vllm_ascend/layers/cp_zigzag.py:41-45`) should hook the new
partial-accumulation helpers rather than reimplementing chunking.

### 2.4 DCP / CP modules

* `vllm/model_executor/layers/attention/pcp.py` (92 lines) **moved** to `vllm/v1/attention/ops/pcp.py`
  (92 lines, functions at identical line numbers: `_gather_prefill_cache_inputs:11`,
  `maybe_gather_mla_latent_cache_inputs:48`, `maybe_gather_indexer_k:69`, `finalize_mla_pcp_decode:83`).
  `TGT` already imports both paths (`vllm_ascend/attention/attention_v1.py:69-71`) and sniffs the module
  with `importlib.util.find_spec` (`vllm_ascend/utils.py:699-700`) to distinguish a tagless v0.28.0 build
  from main. Reuse that pattern.
* `vllm/v1/attention/ops/dcp_alltoall.py` (461 lines at OLD, `dcp_a2a_lse_reduce:392`) is **deleted**;
  `vllm/v1/attention/ops/dcp.py` (1642 lines) is new and also absorbs the CP helpers that used to live in
  `vllm/v1/attention/ops/common.py` (`CPTritonContext`, `correct_attn_out`, `cp_lse_ag_out_rs`,
  `cp_lse_ag_out_ar`, `_correct_attn_cp_out_kernel`; common.py shrank 483→387 lines and now only has the
  `PackSeqTritonKernel`/`UnpackSeqTritonKernel` classes). New in `dcp.py`:
  `mask_dcp_empty_shards_` (`:71`), `get_dcp_workspace_max_num_tokens` (`:996`),
  `reserve_query_head_storage` (`:1018`), `DirectDCPA2AWorkspace` (`:1036`),
  `get_direct_dcp_a2a_workspace` (`:1119`), `DirectDCPQGatherWorkspace` (`:1166`),
  `DirectDCPKVGatherWorkspace` (`:1322`), `DCPCombine` protocol (`:1427`), `MLADCPManager` (`:1438`).
* New `vllm/v1/attention/ops/cp_common.py` (154 lines): `direct_cp_enabled` (`:60`),
  `direct_cp_multicast_enabled` (`:79`), `DirectCPWorkspace` (`:90`) — the direct (pointer/OOB) CP pathway.
* New env switches: `VLLM_USE_DIRECT_DCP_A2A` / `VLLM_USE_DIRECT_DCP_Q_GATHER` /
  `VLLM_USE_DIRECT_DCP_KV_GATHER` (`vllm/envs.py:2181,2184,2187`), `VLLM_DCP_Q_REPLICATE` (`:1580`),
  `VLLM_REPLICATE_EMBED` (`:633`). `vllm/config/parallel.py` keeps `decode_context_parallel_size` (340→351),
  `cp_kv_cache_interleave_size` (360→385) and the deprecated `dcp_kv_cache_interleave_size` (345→356).
* vllm-ascend itself never imports the DCP ops or these env vars (grep at both `23ff2c23c` and `aff1b74b6`),
  so these are *informational*: they show the shape of the new upstream CP contract and the correct place
  to hook `supports_dcp`.

### 2.5 Model runner v1/v2 and worker plumbing

`vllm/v1/worker/gpu_model_runner.py` (7846→7629 lines) — `GPUModelRunner` (`:452`→`:498`):

* **Removed methods** (any Ascend `super()` call or patch target on these is a hard failure):
  `post_kv_cache_wake_up:976`, `init_fp8_kv_scales:980`, `_init_xdrope_positions:1670`,
  `_calc_xdrope_positions:2774`, `_freeze_gc:6469` (→ `freeze_gc_for_cudagraph_capture` import
  `vllm/utils/gc_utils.py`), `_allocate_kv_cache_tensors:7238`, `_reshape_kv_cache_tensors:7290`,
  `_has_mixed_attention_kv_layout:7419`, `_update_hybrid_attention_mamba_layout:7445`,
  `_get_attention_kv_cache_gid:7621`, `_bind_routed_experts_capturer:7677` (→ `get_routed_experts`).
* `__init__` attribute changes: `self.calculate_kv_scales` and `self.uses_xdrope_dim` are gone;
  `self.mrope_num_dims`, `self.jit_warmup_registry`, `self.cp_kv_cache_interleave_size`,
  `self._mamba_state_copy_funcs` are new. `InputBatch(...)` is constructed with the new kwargs.
* New imports the porter should notice: `EncoderCacheManagerMetadata` (`vllm/config/ec_manager_config.py`),
  `PROCESSED_LOGPROBS_MODES` (`vllm/config/model.py`), `MambaStateCopyFuncsByType`
  (`vllm/model_executor/layers/mamba/mamba_utils.py`), `JitWarmupRegistry`
  (`vllm/model_executor/warmup/jit_warmup.py`), `gpu_sync_allowed` (`vllm/utils/gpu_sync_debug.py`),
  `BaseLayerWithLoRA` (`vllm/lora/layers.py`). `MLAAttention` is no longer imported here (only `Attention`).
* `vllm/v1/worker/gpu_input_batch.py`: `CachedRequestState.xdrope_positions` removed (`:49`→gone);
  `InputBatch.__init__` gained `use_replayssm` at `:107` (before `slot_mapping_modes`), `:174`
  `self.use_replayssm`, behavioural sites `:390`,`:613`,`:775`;
  `logits_processing_needs_token_ids` now set unconditionally `False` (`:388`);
  `_make_sampling_metadata` passes `prompt_token_ids=None` (`:889`).
* Runner **v2** (`vllm/v1/worker/gpu/**`, present at both refs): `model_runner.py::GPUModelRunner`
  1658→2253 lines, `initialize_kv_cache` signature changed (`:561`),
  `capture_model(*, profile_only: bool = False)` (`:952`), new `pcp_manager_cls` property (`:2209`),
  new `BatchReqState` NamedTuple; `attn_utils.py` lost `_allocate_kv_cache`, `_reshape_kv_cache`,
  `_align_mixed_attention_kv_cache_views`, `_restride_blocks_first_kv_cache_to_kv_first_storage`,
  `_update_hybrid_attention_layout` and gained `get_attn_cg_support` (`:269`),
  `get_query_lens_mismatch_unsupported_backend`, `FastPrefillHelper` (`:71`), and a rewritten
  `init_kv_cache` (`:520`→`:318`); `model_states/interface.py` renamed
  `get_mm_embeddings`→`prepare_inputs_embeds` and added `get_additional_cg_support`, `execute_mm_encoder`;
  whole new modules `model_states/{encoder_only,prompt_embeds,recoverssm}.py`,
  `spec_decode/{adaptive_verification,dflash2/**,multi_module_mtp/**,extract_hidden_states}.py`,
  `sample/{batch_shard,thinking_budget,watermark,trace_replay}.py`, `ec_connector.py`, `ubatch_utils.py`.

### 2.6 Distributed collectives

* `vllm/distributed/communication_op.py` is **unchanged** (43 lines at both refs) —
  `tensor_model_parallel_all_gather/reduce_scatter/all_reduce`, `get_tp_group`,
  `get_tensor_model_parallel_world_size/rank` and `vllm.distributed.divide` are safe for the feature
  (`vllm_ascend/layers/cp_zigzag.py:29`, `vllm_ascend/distributed/utils.py:5-13`).
* `GroupCoordinator` (`vllm/distributed/parallel_state.py:358`→`:421`): `__init__` grew
  `use_all2all: bool = False` (`:458`); `self.device_index` is now always `local_rank` (the
  `_WORLD.device_index` fallback at OLD `:400-407` is gone); new `isend_object` (`:524`) and a
  `self._pending_isends` deque (`:576`) used by `isend_tensor_dict`, which now also isends the metadata
  list object. `all_reduce/all_gather/reduce_scatter/gather/broadcast/send/recv/broadcast_tensor_dict`
  are signature-identical. `vllm_ascend/distributed/utils.py` uses only
  `group.world_size`, `group.rank_in_group`, `group.device_group`, `group.unique_name` — all stable
  (`unique_name` is still set from `_get_unique_name`, `:141`).
* `get_tp_group` `:1368`→`:1549`, `get_dcp_group` `:1376`→`:1566` (still present — the vllm-ascend
  docstring "v0.21.0 helper removed on vLLM main" refers to the removed
  `get_decode_context_model_parallel_{world_size,rank}` wrappers that vllm-ascend now re-implements in
  `vllm_ascend/distributed/utils.py:8-14`). New: `get_etp_group` (`:1557`),
  `suspend_device_comms`/`resume_device_comms` (`:183`,`:188`),
  `checkpoint_prepare_distributed_state`/`…restore…` (`:2241`,`:2248`).
* Device communicators changed substantially (`custom_all_reduce.py` +366, `pynccl.py` +152,
  `all2all.py` 84, `cuda_communicator.py` 117, new `flashinfer_pcie_ipc_all_reduce.py`,
  `aiter_custom_all_reduce.py`) — CPU/CUDA-only, no NPU impact, but `base_device_communicator.py` and
  `all_reduce_utils.py` interface additions may matter for an NPU communicator subclass.

### 2.7 Custom-op / torch-op registration

* `direct_register_custom_op` (`vllm/utils/torch_utils.py:901`→`:1042`) is **byte-identical**: same
  `(op_name, op_func, mutates_args, fake_impl, target_lib, dispatch_key, tags)` signature and same body.
  All Ascend `torch.ops.vllm.*` registrations keep working unchanged.
* `vllm/model_executor/custom_op.py` keeps `PluggableLayer:32` and `CustomOp:103` with the same public
  API (`enabled`, `default_on`, `register`, `register_oot`, `forward_oot`, `dispatch_forward`,
  `maybe_compile`); only two `assert`s became `ValueError`s (`:287`, `:302`→`:306`). So
  `from vllm.model_executor.custom_op import CustomOp` (`vllm_ascend/ops/...`, used at OLD_FEAT) is safe.
* A **second, parallel** implementation appears at NEW: `vllm/model_executor/hw_agnostic/custom_op.py`
  (318 lines: `maybe_get_oot_by_class:25`, `PluggableLayer:32`, `CustomOp:103`) gated by
  `VLLM_USE_HW_AGNOSTIC` (`vllm/envs.py:1225`). Do not mix the two registries.
* Removed *op* (not a registration helper): `torch.ops.vllm.maybe_calc_kv_scales` — see §2.1/§2.3.

### 2.8 Quantization / linear / embedding / logits

* `QuantizationConfig.is_mxfp4_quant` (`vllm/model_executor/layers/quantization/base_config.py:262`,
  overrides `mxfp4.py:97`, `quark/quark.py:551`, `vllm/models/deepseek_v4/quant_config.py:157`) is
  **deleted**. vllm-ascend never calls it (grep empty at `23ff2c23c`/`aff1b74b6`), so it is a
  no-op for the port; the MXFP4 scheme detection in
  `vllm_ascend/quantization/methods/{w4a4_mxfp4,w4a8_mxfp4,w8a8_mxfp8}.py` stays local.
* `QuantizationConfig` gained `get_checkpoint_weight_mapper`; module-level `resolve_quant_method(
  quant_config, layer, prefix)` is the new dispatch entry (used by `VocabParallelEmbedding`).
* `LinearMethodBase` (`vllm/model_executor/layers/linear.py:141`→`:123`) interface
  (`create_weights/apply/process_weights_after_loading/apply_vllm_mapper`) is unchanged.
  `UnquantizedLinearMethod` (`:182`→`:164`) gained `supports_pre_processed_weights` and an `__init__`
  that resolves the GEMM implementation from `get_current_vllm_config_or_none().kernel_config.linear_backend`.
  `adjust_bitsandbytes_4bit_shard` (`:97`) and `Int8Params.from_layer`
  (`vllm/model_executor/kernels/linear/base.py`) were removed. New class
  `DCPGroupColumnParallelLinear` (`:626`) shows the upstream pattern for group-scoped column sharding.
* `VocabParallelEmbedding` (`vllm/model_executor/layers/vocab_parallel_embedding.py:198`→`:205`,
  `__init__` `:239`→`:249`): new keyword-only `disable_tp`, `quant_method`, `parallel_group`
  (`tp_rank = parallel_group.rank_in_group; tp_size = parallel_group.world_size`); `self.tp_rank` is new;
  quant method resolution switched to `resolve_quant_method`; `is_embedding_layer` is now
  `not isinstance(self, ParallelLMHead)`; new `use_fused_embedding` fast path; new
  `update_param_tp_status()` (`:360`). The feature's `forward_zigzag_local`/`_embed_partial`
  (`vllm_ascend/ops/vocab_parallel_embedding.py` @OLD_FEAT) uses `self.tp_size` +
  `tensor_model_parallel_all_reduce` + the Ascend-local `_mask_input_for_vocab_range` — all still valid,
  but the all-reduce reduction is now *wrong* if the embedding was built with a `parallel_group`
  different from the global TP group (mirror of `vllm_ascend/distributed/parallel_state.py
  get_embed_tp_group/get_lmhead_tp_group`).
* `LogitsProcessor` (`vllm/model_executor/layers/logits_processor.py`, class `:24`→`:58`):
  `forward:64`→`:98` and `_get_logits:138`→`:180` gained additive `skip_gather: bool = False`
  (`:103`,`:185`, early return `:189`); gather (`_gather_logits:85`→`:122`) is now conditional on `lm_head.tp_size > 1`
  and `_get_logits` accepts `UnquantizedLinearMethod` as well as `UnquantizedEmbeddingMethod`. Ascend
  `AscendSampler`/`LogitsProcessor` shims must not assume `get_tensor_model_parallel_world_size()` is the
  LM head's TP size.

### 2.9 envs and config

* All env vars the *feature* reads still exist at NEW: `VLLM_USE_V2_MODEL_RUNNER`
  (`vllm/envs.py:274`→`:302`, `bool | None`, same lambda `:1922`→`:2072`), `VLLM_BATCH_INVARIANT`,
  `VLLM_LOGGING_STREAM`, `VLLM_RPC_BASE_PATH`, `VLLM_USE_MODELSCOPE`, `VLLM_MQ_MAX_CHUNK_BYTES_MB`,
  `VLLM_USE_BREAKABLE_CUDAGRAPH`, `VLLM_LORA_ENABLE_DUAL_STREAM`, `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`,
  `K/Q/V_SCALE_CONSTANT`, `VLLM_ALLOW_INSECURE_SERIALIZATION` (verified name-by-name at both refs).
* **V1/V2 runner gate is a trap:** the feature reads the raw env
  (`import vllm.envs as envs_vllm; envs_vllm.VLLM_USE_V2_MODEL_RUNNER` — `vllm_ascend/ascend_forward_context.py:223,585,594`,
  `vllm_ascend/attention/{attention_v1.py:81, dsa_v1.py:199, fa3_v1.py:18, mla_v1.py:79, sfa_v1.py:170,618}`), which is
  `None` when unset, but vLLM resolves the runner from the *property*
  `VllmConfig.use_v2_model_runner` (`vllm/config/vllm.py:550`→`:667`), which force-enables V2 for
  dspark / multi-KV-group DFlash / diffusion / watermark configs even with the env unset
  (NEW adds the watermark override `:669-673` and the dspark→V2 forcing moved out of the "None" branch
  `:676`). `TGT` already fixed this by switching to the resolved property
  (`vllm_ascend/mrv2_utils.py:use_v2_model_runner`, `vllm_ascend/ascend_forward_context.py:38`,
  `vllm_ascend/models/deepseek_v4/mtp.py:68`). Reuse `TGT`'s helper; do not re-introduce raw-env gating.
* Removed env names (unused by the feature): `VLLM_TRITON_ATTN_USE_TD`, `VLLM_CPU_SGL_KERNEL`,
  `VLLM_TEST_FORCE_FP8_MARLIN`, `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM`.
* New names that must not be collided with: the DCP set (§2.4), `VLLM_RAISE_ON_LOGIT_NANS`,
  `VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS`, `VLLM_USE_HW_AGNOSTIC`, `VLLM_MOE_SKIP_PADDING`-style flags.
* `vllm/config/__init__.py` still exports `CompilationMode`, `CUDAGraphMode`, `get_current_vllm_config`,
  `get_current_vllm_config_or_none`, `get_layers_from_vllm_config` (lines 8/9/58/59/60 at NEW vs
  8/9/55/56/57 at OLD) — the feature's config imports all still resolve.
* `CacheConfig.calculate_kv_scales` **deleted** (`vllm/config/cache.py:111` + validator `:262`), and the
  matching `vllm/model_executor/layers/attention/attention.py:273` branch plus
  `torch.ops.vllm.maybe_calc_kv_scales` (`:696`,`:714`,`:724`) are gone. The feature still reads
  `self.calculate_kv_scales` (`vllm_ascend/worker/model_runner_v1.py:2007,2010` @OLD_FEAT) and
  `vllm_ascend/platform.py:1063-1068`. `TGT` already downgraded these to `getattr`-guarded reads
  (`platform.py:689-693`) and dropped the runner read — do the same.
* `vllm/config/attention.py`: add `AttentionConfig.__post_init__`, `resolve_indexer_kv_dtype`.
* `vllm/config/speculative.py` (1362→1918): additive `use_eagle_block_drop`, `use_multi_module_mtp`,
  `_maybe_override_draft_max_position_embeddings`; `TGT` patches `SpeculativeConfig.__post_init__` on the
  main lane, so keep that patch consistent with the new validation.
* `vllm/config/compilation.py` signatures unchanged (1555→1577 lines) → `CompilationMode`,
  `CUDAGraphMode`, `pass_config.enable_sp`, `use_inductor_graph_partition` used by
  `vllm_ascend/ascend_forward_context.py` and `is_residual_scattered_for_sp` are stable.

### 2.10 Spec-decode / MTP, sampler / logits

* `vllm/v1/spec_decode/llm_base_proposer.py` (1865→1881): `SpecDecodeBaseProposer.__init__(
  vllm_config, device, pass_hidden_states_to_model, runner=None)` **unchanged**; `vllm/v1/spec_decode/utils.py`
  (601 lines) and `metadata.py` (`SpecDecodeMetadata`, 66 lines) have **identical** signatures. The Ascend
  MTP proposer (`vllm_ascend/spec_decode/llm_base_proposer.py`, `patch/worker/patch_deepseek_mtp.py`)
  therefore only needs the *body-level* work.
* `vllm/model_executor/models/deepseek_mtp.py` shrank 560→547: `_restore_full_token_layout_if_needed`
  removed; `DeepSeekMultiTokenPredictorLayer` `:83`→`:69`, `DeepSeekMTP` `:249`→`:233`,
  `_rewrite_spec_layer_name` `:530`→`:517`. Symbols survive (soft), line numbers move.
* `vllm/model_executor/models/llama_eagle3.py`: `Eagle3LlamaForCausalLM` base changed from
  `LlamaForCausalLM` (`:272`) to `_Eagle3LlamaForCausalLMBase` (`:306`); `LlamaDecoderLayer` likewise
  (`Eagle3LlamaDecoderLayerBase`). `Eagle3DeepseekV2ForCausalLM` `deepseek_eagle3.py:283`→`:275`.
  Hard only for Ascend EAGLE3 classes that inherit/subclass these or assume the old MRO.
* `vllm/model_executor/models/extract_hidden_states.py`: `CacheOnlyAttentionBackend.get_kv_cache_shape`
  (`:127`) removed (the class itself and `CacheOnlyAttentionLayer` survive) — see the KV-cache rows.
* `SamplingMetadata` (41 lines both, no field delta), `build_logitsprocs` (identical signature),
  `RejectionSampler`/`PLACEHOLDER_TOKEN_ID` (`vllm/v1/sample/rejection_sampler.py`, 955→953 lines,
  identical signatures) — no API action. `vllm/v1/sample/logits_processor.py` is a *package* at both refs
  (`vllm/v1/sample/logits_processor/__init__.py`), so the `MISSING` result for the flat module is expected.

## 3. Verified no-change (do not spend port effort here)

* `vllm/utils/torch_utils.py::direct_register_custom_op` (`:901`→`:1042`) — identical.
* `vllm/model_executor/custom_op.py` public API — identical (only `assert`→`ValueError`).
* `vllm/distributed/communication_op.py` (43 lines, identical) → `tensor_model_parallel_all_gather`,
  `tensor_model_parallel_reduce_scatter`, `tensor_model_parallel_all_reduce`.
* `GroupCoordinator.all_gather/reduce_scatter/all_reduce/gather/broadcast/send/recv/broadcast_tensor_dict`
  signatures; `world_size`, `rank_in_group`, `device_group`, `unique_name`, `device_index` attributes.
* `get_tp_group`, `get_dcp_group`, `vllm.distributed.divide`, `vllm.distributed.utils.divide`.
* `AttentionMetadataBuilder.__init__` parameter list; `MLACommonMetadataBuilder.build()` /
  `build_for_cudagraph_capture()` parameter lists; `MLACommonMetadataBuilder.__init__` parameter list.
* `AttentionGroup` dataclass fields; `is_residual_scattered_for_sp` (`vllm/v1/worker/utils.py:738`) is
  byte-identical, so the SP padding contract the zigzag alignment relies on is unchanged.
* `SpecDecodeBaseProposer.__init__`, `vllm/v1/spec_decode/{utils,metadata}.py` signatures.
* `SamplingMetadata`, `build_logitsprocs`, `RejectionSampler` signatures.
* All env names the feature reads, and `vllm/config/__init__.py` re-exports.
* `vllm/model_executor/layers/attention/__init__.py` still exports `Attention` and `MLAAttention`.
* `vllm/model_executor/layers/mla.py` (`MLAModules`, `MultiHeadLatentAttentionWrapper`) signatures.
* `vllm/v1/attention/selector.py::get_attn_backend` signature (230→234 lines).
* `vllm/v1/worker/encoder_cudagraph.py` still exists (`EncoderCudaGraphManager:68`, `capture:245`);
  new `is_captured()` hook (`:235`).

## 4. Ordered hard-break checklist for the porter

1. **SFA / DSA-CP metadata rewrite** for `MLACommonPrefillMetadata.ChunkedContextMetadata` →
   `ContextChunk` (`mla_attention.py:1331`→`:1533`) and the new
   `plan_mla_context_chunks`/`init_mla_context_partial`/`accumulate_mla_context_chunk` API
   (`:1841`,`:2687`,`:2714`). This is the functional core of the port.
2. **`supports_dcp` on Ascend impls** (`backend.py:823`→`:822` default flip + new `:230`,`:237`,
   `validate_configuration` `:334-343`) — otherwise DCP startup validation fails.
3. **KV-spec/layout API**: `get_kv_cache_shape`/`get_kv_cache_stride_order`/`indexes_kv_by_block_stride`/
   `get_required_kv_cache_layout` gone from `AttentionBackend`; `KVCacheLayoutType`→`KVCacheLayout`
   (`vllm/v1/kv_cache_layout.py:15`); `AttentionSpec.indexes_kv_by_block_stride` gone
   (`kv_cache_interface.py:182`); `KVCacheTensor.shared_by`→`layers`+`layer_stride` (`:1258-1259`).
4. **`calculate_kv_scales` removal** (`config/cache.py:111`, `attention.py:273`,
   `torch.ops.vllm.maybe_calc_kv_scales`) — delete the reads in `model_runner_v1.py`/`platform.py`.
5. **xdrope removal** in vLLM (`_init_xdrope_positions:1670`, `_calc_xdrope_positions:2774`,
   `CachedRequestState.xdrope_positions:49`) — keep behind `vllm_version_is("0.28.0")` or delete.
6. **`GPUModelRunner` deleted methods** (`post_kv_cache_wake_up:976`, `init_fp8_kv_scales:980`,
   `_allocate_kv_cache_tensors:7238`, `_reshape_kv_cache_tensors:7290`,
   `_update_hybrid_attention_mamba_layout:7445`, `_get_attention_kv_cache_gid:7621`,
   `_has_mixed_attention_kv_layout:7419`, `_bind_routed_experts_capturer:7677`, `_freeze_gc:6469`) —
   re-point to `initialize_kv_cache_tensors:7309` / `vllm/v1/worker/gpu/attn_utils.py:318 init_kv_cache`
   / `vllm/utils/gc_utils.py`.
7. **`InputBatch.__init__` kwarg insertion** (`use_replayssm` at `gpu_input_batch.py:107` before
   `slot_mapping_modes`) — any positional call breaks; verify `NPUInputBatch`
   (`TGT worker/npu_input_batch.py:39-70`) still matches upstream's full parameter order.
8. **`vllm/v1/attention/backends/utils.py` removals** (`is_valid_kv_cache_layout:78`,
   `get_kv_cache_layout:83`, `set_kv_cache_layout:112`, `subclass_attention_metadata:774`).
9. **`pcp.py` module move** (`model_executor/layers/attention/pcp.py` → `v1/attention/ops/pcp.py`).
10. **`Eagle3LlamaForCausalLM` base change** (`llama_eagle3.py:272`→`:306`) and
    **`CacheOnlyAttentionBackend.get_kv_cache_shape` removal** (`extract_hidden_states.py:127`).
11. **WorkerBase new hooks** (`worker_base.py:107,112,128,148`) and v2 `ModelState.get_mm_embeddings`
    → `prepare_inputs_embeds` — needed only if the Ascend worker/state subclasses them.
