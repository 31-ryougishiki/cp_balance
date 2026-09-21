> 历史文档（2026-09 上旬的交接/评审期记录）：结论可能已过时，当前状态以 `docs/scripts_review.md` 为准。
>
> 已变事实（2026-09-21 更正）：
> - 本文写的是 base_layer3（`c7990e5e4` 那棵）。当前参照树 = `vllm-ascend-base`@`base-dp1`（= main aff1b74b6 + dp=1 门修一行）；
>   连续切片的实现已经搬到 `attention/context_parallel/sfa_cp.py`，行号只对 base_layer3 有效。
> - 配置已重构：公共段在 `configs/_base.json`（族差异在 `_common*.json`）；`VLLM_ASCEND_ENABLE_PREFETCH_MLP` 已按定案删除；
>   “无 MTP”只对 A3 的 svc 配置成立，A5 的 `glm52_a5_cur_cp1` 继承 `_common_a5_mtp.json`（deepseek_mtp）。
> - `enable_dsa_cp` 的判定现在在 `vllm_ascend/ascend_config.py`，只判 `has_indexer`（dp=1 放行），且会自动打开 FlashComm；不再有“缺 SP 就抛错”。
> - §7 的“可扩展点”已经实现（zigzag 在 `attention/context_parallel/zigzag_cp.py` + `sfa_cp.py`），该节仅供追溯。

# vllm-ascend-base：GLM-5.2 + DSA-CP + FlashComm1 + TP16 prefill 运行流程

本文只覆盖静态代码阅读结论，所有结论后紧跟 `文件:行号`。代码树为 `/d/code/cp_balance/vllm-ascend-base`（git `c7990e5e4`），上游 vLLM 代码树为 `/d/code/cp_balance/vllm`。

## 1. 版本与入口

模型配置（`GLM-5.2-W4A8C8/config.json`）：`architectures=GlmMoeDsaForCausalLM`、`model_type=glm_moe_dsa`、78 层、`first_k_dense_replace=3`（0-2 层 dense MLP，3-77 层 MoE）、`num_attention_heads=64`、`kv_lora_rank=512`、`qk_nope_head_dim=192`、`qk_rope_head_dim=64`、`v_head_dim=256`、`index_topk=2048`、`index_n_head=32`、`indexer_types` 中含大量 `"shared"`（复用上一层 topk）。

关键开关（`cp_balance/configs/_common.json`）：`additional_config.enable_dsa_cp=true`、`enable_sparse_sfa_c8=true`、`enable_sparse_li_c8=true`、env `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`、TP=16、`--enable-expert-parallel`（EP 组即 16 rank 世界组，dp=pp=1）、`--enforce-eager`（无 ACL graph）、`--no-enable-prefix-caching`、无 speculative config（因此不走 MTP）。`_common.json` 里的 `VLLM_ASCEND_ENABLE_PREFETCH_MLP` 在本仓库中没有任何读取点（见第 6 节末尾）。

入口调用链：

- `vllm serve` 子命令：`vllm/entrypoints/cli/serve.py:44`，转 `uvloop.run(run_server(args))`：`vllm/entrypoints/cli/serve.py:148`。
- `run_server` 构建 engine config 与 `AsyncLLM`：`vllm/entrypoints/openai/api_server.py:163`、`vllm/entrypoints/openai/api_server.py:175`；`AsyncLLM` 通过 `EngineCoreClient.make_async_mp_client` 起引擎核心进程：`vllm/v1/engine/async_llm.py:146`。
- worker 类由 Ascend 平台写回 `parallel_config.worker_cls`：`vllm_ascend/platform.py:616`（`vllm_ascend.worker.worker.NPUWorker`），由 `vllm/v1/worker/worker_base.py:250` 解析实例化。
- `NPUWorker.init_device` 里创建 `NPUModelRunner`：`vllm_ascend/worker/worker.py:500`、`vllm_ascend/worker/worker.py:515`；权重加载 `vllm_ascend/worker/worker.py:663`（`self.model_runner.load_model()` 在 `vllm_ascend/worker/worker.py:674`）。
- worker 每步执行：`vllm_ascend/worker/worker.py:594` → `self.model_runner.execute_model(...)`，`vllm_ascend/worker/worker.py:629`。
- 模型 runner 主循环：`vllm_ascend/worker/model_runner_v1.py:1698`（`execute_model`）→ `_prepare_inputs` 调用点 `vllm_ascend/worker/model_runner_v1.py:1833` → `_determine_batch_execution_and_padding` 调用点 `vllm_ascend/worker/model_runner_v1.py:1855` → `_build_attention_metadata` 调用点 `vllm_ascend/worker/model_runner_v1.py:1965` → `set_ascend_forward_context` `vllm_ascend/worker/model_runner_v1.py:2021` → `_model_forward` `vllm_ascend/worker/model_runner_v1.py:2044`。
- 视觉/文本位置信息更新：`update_cos_sin(positions)` 在 forward 前调用，`vllm_ascend/worker/model_runner_v1.py:1998`。
- 平台侧在配置阶段还做两件与本流程强相关的事：`enable_sp(vllm_config)` 时必须 TP>1 且 MoE 模型必须开 EP，`vllm_ascend/platform.py:744`、`vllm_ascend/platform.py:749`；SFA backend 选择（use_mla=True, use_sparse=True → `AscendSFABackend`），`vllm_ascend/platform.py:806`。

### 1.1 关键开关定义位置

- `enable_sp()`：`vllm_ascend/utils.py:861`；环境变量兜底 `vllm_ascend/utils.py:881`（`VLLM_ASCEND_ENABLE_FLASHCOMM1`），env 定义 `vllm_ascend/envs.py:72`。
- `enable_dsa_cp()`：`vllm_ascend/utils.py:1372`（先要求 hf_text_config 有 `index_topk`，再要求 `additional_config["enable_dsa_cp"]`，最后强制 `enable_sp()`，否则 `vllm_ascend/utils.py:1390` 抛错）。
- `enable_dsa_cp_with_o_proj_tp()`：`vllm_ascend/utils.py:1397`；无 KV transfer 或本 rank 是 KV producer 时为 True，本例（无 PD 分离）恒为 True。
- FlashComm1 是否生效（`forward_context.flash_comm_v1_enabled`）：`vllm_ascend/ascend_forward_context.py:169`（MoE 模型只要 `enable_sp()` 且 `num_tokens` 非空就为真）；`mmrs_fusion` 对 MoE 模型强制 False，`vllm_ascend/ascend_forward_context.py:170`。
- `VLLM_ASCEND_ENABLE_NZ`：`vllm_ascend/envs.py:84`（`maybe_trans_nz` 行为，SFA 里影响 `W_UK_T`，`vllm_ascend/attention/sfa_v1.py:787`）。
- `VLLM_ASCEND_ENABLE_MLAPO`：`vllm_ascend/envs.py:79` → `ascend_config.enable_mlapo`，`vllm_ascend/ascend_config.py:173`。
- `VLLM_ASCEND_ENABLE_FUSED_MC2`：`vllm_ascend/envs.py:92`。
- `VLLM_ASCEND_ENABLE_PREFETCH_MLP`：`vllm_ascend/envs.py` 中不存在；仓库内只有遗留字段 `forward_context.prefetch_mlp_gate_up_proj/prefetch_mlp_down_proj`，恒被置 False，`vllm_ascend/ascend_forward_context.py:195`、`vllm_ascend/ascend_forward_context.py:196`。

## 2. metadata 与连续切片

### 2.1 AscendCommonAttentionMetadata 字段来源

类定义（含字段注释）：`vllm_ascend/attention/utils.py:200`。

在 `_build_attention_metadata` 中构造 `cm_base` 的字段与来源（`vllm_ascend/worker/model_runner_v1.py:2870`）：

| 字段 | 来源 | 位置 |
| --- | --- | --- |
| `query_start_loc` | `self.query_start_loc.gpu[:num_reqs_padded+1]`，由 `_prepare_inputs` 写入累积 token 数，尾部填 -1 | `vllm_ascend/worker/model_runner_v1.py:2871`、写入 `vllm_ascend/worker/model_runner_v1.py:974`、`vllm_ascend/worker/model_runner_v1.py:1000` |
| `seq_lens` | `self.seq_lens[:num_reqs_padded]`，= num_computed_tokens + 本步 scheduled tokens | `vllm_ascend/worker/model_runner_v1.py:2873`、写入 `vllm_ascend/worker/model_runner_v1.py:1163` |
| `_seq_lens_cpu` / `seq_lens_cpu_upper_bound` | 乐观 seq_lens（CPU，避免 D2H 同步） | `vllm_ascend/worker/model_runner_v1.py:2878`、`vllm_ascend/worker/model_runner_v1.py:2879` |
| `num_reqs` / `num_actual_tokens` / `num_input_tokens` | padded 请求数 / 真实 token 数 / padded token 数 | `vllm_ascend/worker/model_runner_v1.py:2885`、`2886`、`2893` |
| `block_table_tensor` | `input_batch.block_table[0].get_device_tensor()[:num_reqs_padded]`，int32 | `vllm_ascend/worker/model_runner_v1.py:2889`、`vllm_ascend/worker/model_runner_v1.py:2838`、buffer 定义 `vllm_ascend/worker/block_table.py:94` |
| `slot_mapping` | `blk_table.slot_mapping.gpu[:num_tokens_padded]`，尾部未用位置填 -1 | `vllm_ascend/worker/model_runner_v1.py:2890`、`vllm_ascend/worker/model_runner_v1.py:2837`、`vllm_ascend/worker/model_runner_v1.py:2841`、dtype/buffer `vllm_ascend/worker/block_table.py:96`、写入 `vllm_ascend/worker/model_runner_v1.py:1182` |
| `actual_seq_lengths_q` | 无投机解码时恒为空 list | `vllm_ascend/worker/model_runner_v1.py:2894`、初始化 `vllm_ascend/worker/model_runner_v1.py:576` |
| `positions` | `np.add(num_computed_tokens_cpu[req_indices], query_pos)` 后搬 NPU | `vllm_ascend/worker/model_runner_v1.py:2895`、写入 `vllm_ascend/worker/model_runner_v1.py:1158` |
| `attn_state` | `_build_attn_state` 判定，本场景为 `PrefillNoCache`/`ChunkedPrefill` | `vllm_ascend/worker/model_runner_v1.py:2897`、`vllm_ascend/worker/model_runner_v1.py:1256` |
| `context_parallel_metadata` | 仅 DCP 路径（`dcp_size>1`）才非空，本例为 None | `vllm_ascend/worker/model_runner_v1.py:2899`、`vllm_ascend/worker/model_runner_v1.py:2799` |
| `group_len/group_key_idx/group_key_cache_idx` | c8 reshape 写 cache 优化用的分组元数据 | `vllm_ascend/worker/model_runner_v1.py:2900`、buffer `vllm_ascend/worker/model_runner_v1.py:293` |
| `is_prefilling` | `num_computed_tokens_cpu < num_prompt_tokens_cpu` | `vllm_ascend/worker/model_runner_v1.py:2862` |

padded token 数由 `_pad_for_sequence_parallelism` 决定（TP 的倍数）：`vllm_ascend/worker/model_runner_v1.py:2605`、调用点 `vllm_ascend/worker/model_runner_v1.py:2678`、断言 `vllm_ascend/worker/model_runner_v1.py:2717`。TP=16 时 `num_tokens_padded = round_up(num_actual_tokens, 16)`。

backend/builder 选择：`AscendSFABackend.get_builder_cls` 默认返回 `AscendSFAMetadataBuilder`，仅当 `enable_sfa_dcp_replicated_indexer()`（要求 `decode_context_parallel_size>1`，本例为 1）才换成 DCP 子类，`vllm_ascend/attention/sfa_v1.py:166`、`vllm_ascend/utils.py:122`。因此本例 100% 走 `AscendSFAMetadataBuilder._build`。

### 2.2 `AscendSFAMetadataBuilder._build` 的连续切片

入口与前半段（cos/sin 生成、block_table/slot_mapping 截断）：`vllm_ascend/attention/sfa_v1.py:368`-`vllm_ascend/attention/sfa_v1.py:390`。

切片逻辑（`if self.enable_dsa_cp`，成员标记 `self.enable_dsa_cp = enable_dsa_cp()`，`vllm_ascend/attention/sfa_v1.py:309`）：

- `global_tp_size = get_tp_group().world_size`（CP 维度借用 TP 组）：`vllm_ascend/attention/sfa_v1.py:400`。
- `num_tokens = num_input_tokens`（已 padded），`num_tokens_pad = _round_up(num_tokens, global_tp_size)`，`num_tokens_per_device = num_tokens_pad // global_tp_size`：`vllm_ascend/attention/sfa_v1.py:401`、`402`；`_round_up` 定义 `vllm_ascend/utils.py:295`。
- `local_start = rank_in_group * num_tokens_per_device`，`local_end_with_pad = local_start + num_tokens_per_device`，`local_end = min(local_end_with_pad, num_actual_tokens)`：`vllm_ascend/attention/sfa_v1.py:403`、`404`、`405`。即按 flattened token 顺序做**连续等分**，rank r 拿 `[r*L, (r+1)*L)`。
- cos/sin 先按全局长度 pad 到 `num_tokens_pad`，再切片到本 rank 区间：`vllm_ascend/attention/sfa_v1.py:410`、`411`、`412`、`421`、`422`。cos/sin 形状 `(num_tokens,1,1,rope_dim)`（`get_cos_and_sin_mla`，`vllm_ascend/ops/rotary_embedding.py:90`），切片后为 `(num_tokens_per_device,1,1,64)`。
- `slot_mapping` 先 pad 到 `num_tokens_pad`（补 -1），再做本 rank 切片得到 `slot_mapping_cp`：`vllm_ascend/attention/sfa_v1.py:414`、`416`、`419`；形状 `(num_tokens_per_device,)`、int32。注意进 metadata 的 `slot_mapping` 仍是**全量 padded** 版本（`vllm_ascend/attention/sfa_v1.py:499`），只有 `slot_mapping_cp` 是 local。
- 断言 local 形状：`vllm_ascend/attention/sfa_v1.py:424`、`428`、`433`。
- 每请求的 local query/key 长度（全程 on-device，不做 D2H）：`global_start = query_start_loc[:num_segs]`、`global_end = cum_query_lens`，`req_local_start = global_start.clamp(min=local_start)`、`req_local_end = global_end.clamp(max=local_end_with_pad)`、`num_local_tokens = req_local_end - req_local_start`：`vllm_ascend/attention/sfa_v1.py:456`、`457`、`461`、`462`、`463`。
- `actual_seq_lengths_query[:num_segs] = cumsum(clamp(num_local_tokens,min=0))`：`vllm_ascend/attention/sfa_v1.py:465`、`467`；即 local 视角的 TND 累积 query 长度，形状 `(num_reqs,)`、int32。
- `offset = global_end - req_local_end`，`local_key_lens = where(num_local_tokens>0, seq_lens-offset, 0)`：`vllm_ascend/attention/sfa_v1.py:466`、`468`、`469`；含义是"本 rank 切片结束时，该请求已写入 cache 的 KV 长度"，落在本 rank 之前的请求为 0（该 rank 不为其算 topk/attention）。
- 写回固定 buffer 并截到 `num_reqs`：`vllm_ascend/attention/sfa_v1.py:470`、`471`；buffer 在 builder 初始化时分配 `max_num_reqs+1`，`vllm_ascend/attention/sfa_v1.py:285`、`vllm_ascend/attention/sfa_v1.py:286`。

`DSACPContext` 字段（`vllm_ascend/attention/sfa_v1.py:197`）：`num_tokens`、`num_tokens_pad`、`local_start`、`local_end`、`local_end_with_pad`、`slot_mapping_cp`、`actual_seq_lengths_query`、`actual_seq_lengths_key`（`vllm_ascend/attention/sfa_v1.py:198`-`205`）；构造点 `vllm_ascend/attention/sfa_v1.py:474`-`481`。其中 `local_end` 在本仓库内除定义外没有被消费（只被 DCP 子类读取，`vllm_ascend/attention/context_parallel/sfa_cp.py:633`）。

被改写/切片的东西汇总：`cos/sin`（切片，`vllm_ascend/attention/sfa_v1.py:421`）、`slot_mapping`（先 pad 到 `num_tokens_pad`，`vllm_ascend/attention/sfa_v1.py:414`）、`slot_mapping_cp`（切片，`vllm_ascend/attention/sfa_v1.py:419`）、`actual_seq_lengths_query/key`（新算，`vllm_ascend/attention/sfa_v1.py:467`、`469`）。`block_table`、`seq_lens`、`cum_query_lens`、`attn_mask` 保持全量（`vllm_ascend/attention/sfa_v1.py:499`-`505`）。

最终 metadata 对象字段见 `vllm_ascend/attention/sfa_v1.py:484`-`510`，其中 `sin/cos` 取本 rank 切片后的值（`vllm_ascend/attention/sfa_v1.py:500`、`501`），`dsa_cp_context` 只有 DSA-CP 时非空（`vllm_ascend/attention/sfa_v1.py:507`）。`AscendSFAMetadata` 定义 `vllm_ascend/attention/sfa_v1.py:209`（`dsa_cp_context` 字段 `vllm_ascend/attention/sfa_v1.py:239`）。

c8 reshape 优化下的额外 cache 元数据下发：`vllm_ascend/attention/sfa_v1.py:485`-`492`（本例 `c8_enable_reshape_optim` 未开，为 no-op）。

## 3. 模型 forward 主链路

### 3.1 patch 后的 DeepseekV2Model.forward

GLM-5.2 走 `GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)`：`vllm/model_executor/models/deepseek_v2.py:1920`；其 `self.model` 为 `DeepseekV2Model`（`vllm/model_executor/models/deepseek_v2.py:1347`）。

Ascend 用 `_patched_forward` 整体替换 `DeepseekV2Model.forward`：定义 `vllm_ascend/patch/worker/patch_deepseek_v2.py:301`，替换赋值 `vllm_ascend/patch/worker/patch_deepseek_v2.py:365`（导入即生效：`vllm_ascend/patch/worker/__init__.py:55`）。

与本流程相关的差异：原版 forward 里"逐层进入前 all_gather hidden+residual"的分支被**删除**（对照原实现 `vllm/model_executor/models/deepseek_v2.py:1455`-`1470`）；patch 后逐层循环体只有 aux_hidden_states 收集和 `layer(...)` 调用：`vllm_ascend/patch/worker/patch_deepseek_v2.py:333`、`342`、`343`。也就是说 FlashComm1 下每层输入/输出都是 rank-local token 段，逐层不做 hidden_states all_gather。

`embed_tokens`：（`VocabParallelEmbedding` 被换成 `AscendVocabParallelEmbedding`，注册表 `vllm_ascend/utils.py:720`、`vllm_ascend/utils.py:779`）。embedding 计算后走 `_forward_origin` 的 `torch.ops.vllm.maybe_pad_and_reduce(output_parallel)`：`vllm_ascend/ops/vocab_parallel_embedding.py:168`、`vllm_ascend/ops/vocab_parallel_embedding.py:248`。该 op 在 `flash_comm_v1_enabled` 为真时执行 **TP reduce_scatter**（否则 all_reduce）：`vllm_ascend/ops/register_custom_ops.py:72`、`vllm_ascend/ops/register_custom_ops.py:78`、`vllm_ascend/ops/register_custom_ops.py:90`。所以 prefill 第 0 层输入就是本 rank 的连续 token 段（行数 `num_tokens_padded/16`），与第 2 节 metadata 的 `local_start/local_end_with_pad` 语义一致。另一条 `embedding_tp_enable()` 分支（显式 all_gather+reduce_scatter）在本配置未开：`vllm_ascend/ops/vocab_parallel_embedding.py:165`、`vllm_ascend/ops/vocab_parallel_embedding.py:170`、`vllm_ascend/utils.py:841`。

`layer.use_sequence_parallel_moe` 为 False（vLLM 侧 SP-MoE 与 Ascend FlashComm1 互斥：`vllm_ascend/platform.py:607`、`vllm/config/parallel.py:654`），因此 `DeepseekV2DecoderLayer.forward` 里那些 `if self.use_sequence_parallel_moe:` 分支不执行（`vllm/model_executor/models/deepseek_v2.py:1272`、`1317`、`1326`、`1329`）。

### 3.2 逐层调用

`DeepseekV2DecoderLayer.forward` 顺序（`vllm/model_executor/models/deepseek_v2.py:1272`）：

1. `hidden_states, residual = self.input_layernorm(hidden_states, residual)`：`vllm/model_executor/models/deepseek_v2.py:1291`；Ascend 实现 `forward_oot` 用 `npu_add_rms_norm`，纯本地无通信：`vllm_ascend/ops/layernorm.py:63`、`vllm_ascend/ops/layernorm.py:73`。
2. `self.self_attn(positions, hidden_states, llama_4_scaling)`：`vllm/model_executor/models/deepseek_v2.py:1300`；调用 `DeepseekV2MLAAttention.forward` → `self.mla_attn(...)`（`vllm/model_executor/models/deepseek_v2.py:1168`），而 `MultiHeadLatentAttentionWrapper` 被 Ascend 实现替换（`vllm_ascend/ops/mla.py:66`）。
3. Ascend MLA 层分配输出并调用自定义 op：`vllm_ascend/ops/mla.py:201`、`vllm_ascend/ops/mla.py:213`、`vllm_ascend/ops/mla.py:216`、`vllm_ascend/ops/mla.py:220`；`mla_forward` 取 `kv_cache` 与 metadata 后调用 impl：`vllm_ascend/ops/mla.py:225`、`vllm_ascend/ops/mla.py:236`。输出行数 = 输入行数（local token 段），`vllm_ascend/ops/mla.py:216`。
4. `hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)`：`vllm/model_executor/models/deepseek_v2.py:1326`。
5. `hidden_states = self.mlp(hidden_states)`：`vllm/model_executor/models/deepseek_v2.py:1333`；MoE 层为 `DeepseekV2MoE`（`vllm/model_executor/models/deepseek_v2.py:1219`），dense 层为 `DeepseekV2MLP`（`vllm/model_executor/models/deepseek_v2.py:230`）。
6. 返回 `(hidden_states, residual)`，两者都是 rank-local token 段。

## 4. 单层内部（SFA / MLP-MoE）

### 4.1 SFA：rank-local 还是全量

impl 类 `AscendSFAImpl`：`vllm_ascend/attention/sfa_v1.py:531`；初始化 `vllm_ascend/attention/sfa_v1.py:544`。

DSA-CP 下的两个关键改写：

- `self.enable_dsa_cp`（`vllm_ascend/attention/sfa_v1.py:674`）为真时 `self.local_num_heads = self.num_heads * self.tp_size`，即**本 rank 持有全部 64 个 head**（`num_heads` 传入的是 `num_local_heads=4`）：`vllm_ascend/attention/sfa_v1.py:684`、`vllm_ascend/attention/sfa_v1.py:685`。
- q/kv 的 up-projection 走"伪单卡"列并行算子，权重**全量复制、不做 TP 切分**：选择逻辑 `vllm_ascend/ops/linear_op.py:458`、`vllm_ascend/ops/linear_op.py:459`；伪通信组 `world_size=1`（`vllm_ascend/ops/linear_op.py:424`、`vllm_ascend/ops/linear_op.py:428`）使 `output_size_per_partition = output_size`、`vllm_ascend/ops/linear.py:401`。kv_b_proj 全量维度断言：`vllm_ascend/attention/sfa_v1.py:745`。

逐项清单（prefill、`attn_state ∉ {DecodeOnly, SpecDecoding}`）：

| 环节 | 位置 | local / 全量 |
| --- | --- | --- |
| hidden_states 入口 all_gather | 仅 `enable_sp and not enable_dsa_cp` 时做，DSA-CP 下**跳过**：`vllm_ascend/attention/sfa_v1.py:2055`、`2056` | rank-local（`num_tokens_padded/16` 行） |
| fused_qkv_a_proj（q_c、kv_no_split） | `vllm_ascend/attention/sfa_v1.py:2060`-`2066`；层是 `DeepSeekV2FusedQkvAProjLinear`（`disable_tp=True`，权重全量）：`vllm/model_executor/models/deepseek_v2.py:905`、`vllm/model_executor/models/deepseek_v2.py:918` | 输入 rank-local，权重全量 |
| q_a_layernorm | `vllm_ascend/attention/sfa_v1.py:2066` | rank-local |
| indexer k 预处理（wk_weights_proj + k_norm + RoPE + hadamard/C8 量化） | `vllm_ascend/attention/sfa_v1.py:2068`、实现 `vllm_ascend/attention/sfa_v1.py:1534`、`vllm_ascend/attention/sfa_v1.py:1553`-`1581`；wq_b/wk_weights_proj 均为全量复制权重：`vllm/model_executor/models/deepseek_v2.py:667`、`vllm/model_executor/models/deepseek_v2.py:676` | rank-local（用本 rank cos/sin） |
| exec_kv（kv_a_layernorm + RoPE + C8 block quant） | `vllm_ascend/attention/sfa_v1.py:2084`；sparse C8 走 `custom_kv_rmsnorm_rope`：`vllm_ascend/attention/sfa_v1.py:2246`、`2259`、`2260`、`2263`；非 C8 的 DSA-CP 才用 `npu_kv_rmsnorm_rope_cache` 直接写 cache：`vllm_ascend/attention/sfa_v1.py:1284`-`1292` | rank-local，本步骤**不写 cache**（C8 路径） |
| KV/indexer KV all_gather | `vllm_ascend/attention/sfa_v1.py:2087`、实现 `vllm_ascend/attention/sfa_v1.py:1742`-`1772`（融合 KV 一次 + k_li 一次 + k_li_scale 一次） | all_gather 后**全量**（每 rank 都拿全量） |
| q_b_proj + k_up_proj（W_UK_T） | `vllm_ascend/attention/sfa_v1.py:2096`；实现 `vllm_ascend/attention/sfa_v1.py:1299`-`1325`（view 到 `local_num_heads=64`，`npu_transpose_batchmatmul`）；W_UK_T 从全量 kv_b_proj 切出：`vllm_ascend/attention/sfa_v1.py:761`、`770` | rank-local token、全 64 head |
| q_pe RoPE | `vllm_ascend/attention/sfa_v1.py:2097`、实现 `vllm_ascend/attention/sfa_v1.py:1029`-`1039`；用 metadata 里**已切片**的 cos/sin | rank-local |
| cache 写入（主 KV） | `vllm_ascend/attention/sfa_v1.py:2110`、实现 `vllm_ascend/attention/sfa_v1.py:1850`-`1857`（用全量 all-gathered KV + 全量 `slot_mapping`） | 写入的是**全量（副本）**，每个 rank 的 KV cache 内容相同 |
| indexer cache 写入 | `vllm_ascend/attention/sfa_v1.py:2130`-`2170`（`slot_mapping` 全量、`k_li` 全量） | 同上，副本 |
| topk 选择 | 长度取 local：`vllm_ascend/attention/sfa_v1.py:2176`、`2177`；skip_topk 时复用 buffer `vllm_ascend/attention/sfa_v1.py:2182`；否则 `vllm_ascend/attention/sfa_v1.py:2187` → `vllm_ascend/attention/sfa_v1.py:1584` → `vllm_ascend/device/device_op.py:476`（LI-C8 走 `npu_lightning_indexer_quant`，`vllm_ascend/device/device_op.py:500`，`sparse_count=2048`） | query 用 local token 段 + local q_c；key/block_table 用**全量副本 cache** |
| 稀疏 attention 主计算 | `vllm_ascend/attention/sfa_v1.py:2200` → `vllm_ascend/device/device_op.py:546`；C8 packed cache 走 `npu_kv_quant_sparse_flash_attention`：`vllm_ascend/device/device_op.py:632`；layout `TND`/`PA_BSND`、`actual_seq_lengths_query/key` 传 local 值：`vllm_ascend/device/device_op.py:640`-`644` | query rank-local，KV 全量副本 |
| `_v_up_proj`（W_UV） | `vllm_ascend/attention/sfa_v1.py:2210`、`vllm_ascend/attention/sfa_v1.py:1326`-`1352` | rank-local token × 全 64 head |
| o_proj | prefill 分支 `vllm_ascend/attention/sfa_v1.py:2212`-`2228` → `vllm_ascend/attention/sfa_v1.py:1135`-`1180`：先 wait o_proj 权重 all_gather（`vllm_ascend/attention/sfa_v1.py:1150`、`1153`），临时把 `o_proj.weight` 换成全量（`vllm_ascend/attention/sfa_v1.py:1157`），跑全量 GEMM（`vllm_ascend/attention/sfa_v1.py:1160`），再换回 TP 分片（`vllm_ascend/attention/sfa_v1.py:1162`-`1164`）；不满足条件时才走 decode 的 all_to_all 分支（`vllm_ascend/attention/sfa_v1.py:1177`）、返回 `vllm_ascend/attention/sfa_v1.py:1179` | 输入 rank-local token，**用全量权重**；prefill 无 reduce_scatter |
| 兜底 o_proj（非 DSA-CP 或 oproj_tp） | `vllm_ascend/attention/sfa_v1.py:2230`-`2238` | 本例不进入 |

补充：`self.preprocess_type` 在 `enable_dsa_cp` 时被强制回 NATIVE（融合预处理不支持 DSA-CP）：`vllm_ascend/attention/sfa_v1.py:860`、`vllm_ascend/attention/sfa_v1.py:861`、`vllm_ascend/attention/sfa_v1.py:784`。因此本例 SFA 永远走 `# native` 分支（`vllm_ascend/attention/sfa_v1.py:2052`）。

o_proj 全量权重的临时 buffer 初始化：`vllm_ascend/attention/sfa_v1.py:780`、`782` → `vllm_ascend/attention/sfa_v1.py:1041`-`1085`（未量化沿 dim1 拼、量化沿 dim0 拼，`vllm_ascend/attention/sfa_v1.py:1058`-`1062`）。

### 4.2 MLP 与 MoE 分支

dense 层（0-2 层，`DeepseekV2MLP`，`vllm/model_executor/models/deepseek_v2.py:230`）：

- `gate_up_proj`：算子选择 `vllm_ascend/ops/linear_op.py:469`、`479`（`SequenceColumnParallelOp`）；执行时先 `torch.ops.vllm.maybe_all_gather_and_maybe_unpad`（TP all_gather）再 GEMM：`vllm_ascend/ops/linear_op.py:301`、`302`；op 实现 `vllm_ascend/ops/register_custom_ops.py:39`、TP all_gather 在 `vllm_ascend/ops/register_custom_ops.py:49`。
- `down_proj`：算子选择 `vllm_ascend/ops/linear_op.py:500`、`506`（`SequenceRowParallelOp`）；`apply_impl` 走 `torch.ops.vllm.matmul_and_reduce`：`vllm_ascend/ops/linear_op.py:332` → `vllm_ascend/ops/linear_op.py:337`；非 `mmrs_fusion` 时是纯 `tensor_model_parallel_reduce_scatter`：`vllm_ascend/ops/linear_op.py:413`。`mmrs_fusion` 对 MoE 模型恒 False，fused `npu_mm_reduce_scatter_base` 分支不生效：`vllm_ascend/ascend_forward_context.py:170`、`vllm_ascend/ops/linear_op.py:371`。
- padding 细节：`pad_size>0` 且不是 DSA-CP 的 attn 输出时补 pad 再 reduce_scatter：`vllm_ascend/ops/linear_op.py:357`、`358`、`359`。o_proj 前缀被显式排除（`dsa_cp_attn_out`）：`vllm_ascend/ops/linear_op.py:357`。

MoE 层（3-77 层，`DeepseekV2MoE`，`vllm/model_executor/models/deepseek_v2.py:277`）：

- 通信方法选择：`select_moe_comm_method`，`vllm_ascend/ascend_forward_context.py:343`；EP>1 时注册 ALLTOALL/ALLGATHER/MC2/FUSED_MC2，`vllm_ascend/ops/fused_moe/moe_comm_method.py:54`。
- `prepare`：调用点 `vllm_ascend/ops/fused_moe/fused_moe.py:639`、参数 `replace_allreduce=_EXTRA_CTX.flash_comm_v1_enabled`：`vllm_ascend/ops/fused_moe/fused_moe.py:643`。
  - AllGather 方法：走 `_prepare_with_ep_group`，对 hidden/router_logits 做 TP all_gather：`vllm_ascend/ops/fused_moe/prepare_finalize.py:386`、`387`（内部 `maybe_all_gather_and_maybe_unpad`，`vllm_ascend/ops/register_custom_ops.py:49`）。
  - All2All / MC2 方法：FlashComm1 下 `replace_allreduce=True`，`prepare` 只做 pad/TP 切分且被跳过：`vllm_ascend/ops/fused_moe/prepare_finalize.py:152`、`283`（MC2 prepare 在 `vllm_ascend/ops/fused_moe/prepare_finalize.py:255`，跳过条件 `vllm_ascend/ops/fused_moe/prepare_finalize.py:283`）。routed token 的实际交换在 dispatch/combine：MC2 用 `npu_moe_distribute_dispatch_v2`/`combine_v2`（`vllm_ascend/ops/fused_moe/token_dispatcher.py:239`、`245`、`343`、`348`），All2All 用 `async_all_to_all`（`vllm_ascend/ops/fused_moe/token_dispatcher.py:488`、`516`、`522`、`564`、`571`）。
- `finalize`：调用点 `vllm_ascend/ops/fused_moe/fused_moe.py:700`。
  - AllGather 方法：`_finalize_with_ep_group` → `torch.ops.vllm.maybe_pad_and_reduce(hidden, True)`（TP reduce_scatter）：`vllm_ascend/ops/fused_moe/prepare_finalize.py:523`、op 实现 `vllm_ascend/ops/register_custom_ops.py:90`。
  - All2All/MC2：同样因 `replace_allreduce=True` 直接返回、不做 TP 通信：`vllm_ascend/ops/fused_moe/prepare_finalize.py:188`、`198`。
- shared experts（每 MoE 层 1 个，`n_shared_experts=1`）：其 `gate_up_proj`/`down_proj` 被显式排除在 SP 自定义算子之外，退回普通 TP 语义（`reduce_results=False`）：`vllm_ascend/ops/linear_op.py:466`、`495`；共享专家输出在 ALLTOALL/MC2/FUSED_MC2 下补一次 TP all_reduce：`vllm_ascend/ops/fused_moe/fused_moe.py:833`。
- 共享+路由输出的合并与最终 all_reduce 抑制：`vllm/model_executor/layers/fused_moe/runner/moe_runner.py:725`、`727`；Ascend 侧 `_maybe_reduce_final_output` 走 `maybe_all_reduce_tensor_model_parallel`，在 MC2/ALLTOALL/FUSED_MC2 或 FlashComm1 下是 no-op：`vllm_ascend/ops/fused_moe/fused_moe.py:609`、`614`、`vllm_ascend/ops/register_custom_ops.py:125`、`131`。

残差、RMSNorm 与通信的先后关系（每层固定顺序，全部 rank-local）：输入 `input_layernorm(hidden,residual)` → self_attn（内部 KV all_gather / o_proj 权重 all_gather）→ `post_attention_layernorm` → MLP/MoE（内部 AG/RS 或 dispatch/combine）→ 返回新 residual。引用见 `vllm/model_executor/models/deepseek_v2.py:1291`、`1300`、`1326`、`1333`、`1336`。注意 o_proj 输出与残差的行数必须同为 local 行数，这也是 DSA-CP prefill 必须用全量 o_proj 权重（而不是 reduce_scatter）的原因：`vllm_ascend/attention/sfa_v1.py:2213`、`2214`。

## 5. 模型出口

- `_model_forward`：`vllm_ascend/worker/model_runner_v1.py:2571`；封装 `self.model(...)` 调用 `vllm_ascend/worker/model_runner_v1.py:2591`、`2596`、`2598`。
- 出口 all_gather：`flash_comm_v1_enabled` 时对 hidden_states（含 aux tuple）做 `_all_gather_hidden_states_and_aux`：`vllm_ascend/worker/model_runner_v1.py:2601`、`2602`、`2603`；实现 `vllm_ascend/worker/model_runner_v1.py:2541`、`2526`（`tensor_model_parallel_all_gather(hidden_states, 0)`，`vllm_ascend/worker/model_runner_v1.py:2527`），并按 `forward_context.pad_size` 去 pad：`vllm_ascend/worker/model_runner_v1.py:2529`、`vllm_ascend/worker/model_runner_v1.py:2530`。本配置 `num_tokens_padded` 已是 16 的倍数，`pad_size=0`（`vllm_ascend/ascend_forward_context.py:182`、`183`）。
- logits：只对 `logits_indices` 取行再算：`vllm_ascend/worker/model_runner_v1.py:2071`、`2072`；`logits_indices` 来自 `_prepare_inputs`（无投机解码时 = `query_start_loc[1:num_reqs+1]-1`）：`vllm_ascend/worker/model_runner_v1.py:1205`。
- `compute_logits`：`vllm/model_executor/models/deepseek_v2.py:1888`、`1893`（`LogitsProcessor`，Ascend 侧替换为 `AscendLogitsProcessor`，`vllm_ascend/utils.py:722`、`vllm_ascend/ops/vocab_parallel_embedding.py:285`；lm_head 为 `AscendParallelLMHead`，`vllm_ascend/utils.py:721`）。
- 之后进入 `sample_tokens`：`vllm_ascend/worker/model_runner_v1.py:2118`。

## 6. 集合通信清单表

参数：TP=16（同时是 DSA-CP 的 CP 维度）；EP 组在 dp=pp=1 时即这 16 个 rank（`--enable-expert-parallel`）；`num_actual_tokens=N`、`num_tokens_padded=N_pad=round_up(N,16)`、`L=N_pad/16`、dense 层 3 层（0-2）、MoE 层 75 层（3-77）。所有 TP 通信都在 `get_tp_group()`；EP 通信在 EP/mc2 组。

| 位置(file:line) | 原语 | group | 张量形状/通信量 | 频次 | prefill 专属 |
| --- | --- | --- | --- | --- | --- |
| `vllm_ascend/ops/vocab_parallel_embedding.py:248` → `vllm_ascend/ops/register_custom_ops.py:90` | `tensor_model_parallel_reduce_scatter`（`maybe_pad_and_reduce`） | TP | in `(N_pad,6144)` bf16 → out `(L,6144)` | 1 次/forward | 否（FlashComm1 通用；DSA-CP 下它就是序列切片） |
| `vllm_ascend/attention/sfa_v1.py:1756`（封装 `vllm_ascend/distributed/utils.py:26`） | `dist.all_gather_into_tensor`（async） | TP | fused KV `(L,656)` int8 → `(N_pad,656)`（k_nope int8 512 + k_pe 128B 视图 + knope_scale 16B） | 每层 1 次（78 层） | 是（`enable_dsa_cp`） |
| `vllm_ascend/attention/sfa_v1.py:1766`、`1768` | `dist.all_gather_into_tensor`（async） | TP | `k_li (L,128)` int8 → `(N_pad,128)` | 每层 1 次（本层有自身 indexer 时；shared 层跳过） | 是 |
| `vllm_ascend/attention/sfa_v1.py:1776`、`1778` | `dist.all_gather_into_tensor`（async） | TP | `k_li_scale (L,1)` fp16 → `(N_pad,1)` | 同上 | 是 |
| `vllm_ascend/attention/sfa_v1.py:1836`、`1838` | `dist.all_gather_into_tensor`（async，写入常驻 buffer） | TP | 每 rank 发 o_proj 分片、收全量权重（bf16 未量化时 `6144×16384×2B≈192MiB/rank`；W4A8 按存储 dtype 折算） | 每层 1 次（DSA-CP prefill/mixed） | 是 |
| `vllm_ascend/attention/sfa_v1.py:1843`、`1845` | `dist.all_gather_into_tensor`（async） | TP | input 维分片的量化参数（deq scale 等，`vllm_ascend/attention/sfa_v1.py:1111`-`1124`） | 每层 1 次（量化时） | 是 |
| `vllm_ascend/attention/sfa_v1.py:1832`、`1833` | `Work.wait()`（等上面的 KV all_gather 完成） | TP | 同步点，非通信量 | 每层 1 次 | 是 |
| `vllm_ascend/ops/linear_op.py:301` → `vllm_ascend/ops/register_custom_ops.py:49` | `tensor_model_parallel_all_gather` | TP | `(L,6144)` → `(N_pad,6144)` bf16 | dense 层 0-2，每层 1 次 | 否（FlashComm1 SP） |
| `vllm_ascend/ops/linear_op.py:332`、`413` | `tensor_model_parallel_reduce_scatter` | TP | `(N_pad,6144)` → `(L,6144)` bf16 | dense 层 0-2，每层 1 次 | 否 |
| `vllm_ascend/ops/fused_moe/prepare_finalize.py:386`、`387` → `vllm_ascend/ops/register_custom_ops.py:49` | `tensor_model_parallel_all_gather` | TP | hidden `(L,6144)`、router_logits `(L,256)` → 全量 | 每 MoE 层 1 次（仅 `MoECommType.ALLGATHER`） | 否 |
| `vllm_ascend/ops/fused_moe/prepare_finalize.py:523` → `vllm_ascend/ops/register_custom_ops.py:90` | `tensor_model_parallel_reduce_scatter` | TP | `(N_pad,6144)` → `(L,6144)` | 每 MoE 层 1 次（仅 ALLGATHER） | 否 |
| `vllm_ascend/ops/fused_moe/token_dispatcher.py:239`、`245` | `torch_npu.npu_moe_distribute_dispatch_v2`（EP+TP 融合 all_to_all） | EP/mc2 | `(L,6144)` + topk_ids → 按 expert 重排 | 每 MoE 层 1 次（MC2/FUSED_MC2） | 否 |
| `vllm_ascend/ops/fused_moe/token_dispatcher.py:343`、`348` | `torch_npu.npu_moe_distribute_combine_v2` | EP/mc2 | 专家输出 → `(L,6144)` | 每 MoE 层 1 次（MC2/FUSED_MC2） | 否 |
| `vllm_ascend/ops/fused_moe/token_dispatcher.py:516`、`522`（`async_all_to_all`） | `all_to_all` 两次（permute1/permute2） | EP | `(L,6144)` ↔ 专家本地 token | 每 MoE 层 2 次（ALLTOALL） | 否 |
| `vllm_ascend/ops/fused_moe/token_dispatcher.py:571` | `all_to_all`（combine） | EP | 专家输出 → `(L,6144)` | 每 MoE 层 1 次（ALLTOALL） | 否 |
| `vllm_ascend/ops/fused_moe/fused_moe.py:833` | `tensor_model_parallel_all_reduce` | TP | shared experts 输出 `(L,6144)` | 每 MoE 层 1 次（ALLTOALL/MC2/FUSED_MC2 且非 shared_expert_dp） | 否 |
| `vllm_ascend/worker/model_runner_v1.py:2601`、`2528` | `tensor_model_parallel_all_gather`（+按 pad_size 去 pad） | TP | `(L,6144)` → `(N_pad,6144)` | 1 次/forward | 否（FlashComm1 通用） |
| `vllm_ascend/attention/sfa_v1.py:1170`、`1177`（decode 路径） | `torch.distributed.all_to_all_single` | TP | head 维交换 | 每层 1 次（**仅 decode**） | 否 |
| `vllm_ascend/ops/linear_op.py:337`、`413`（decode o_proj 路径） | `tensor_model_parallel_reduce_scatter` | TP | `(L,6144)` | 每层 1 次（**仅 decode**） | 否 |

统计：表格共 19 行数据行；其中 `enable_dsa_cp` 专属（prefill 分支才发生）6 行（SFA 的 KV/索引 KV/索引 scale/o_proj 权重/量化参数 all_gather、以及 KV gather 的 wait 同步行）；TP 组内的 all_gather/reduce_scatter 通用 SP 通信 6 行；EP 组 MoE dispatch/combine 4 行；shared experts 补 reduce 1 行；仅 decode 才发生的 2 行。

补充说明：`VLLM_ASCEND_ENABLE_PREFETCH_MLP=1` 在本仓库没有实现，只有恒 False 的占位字段（`vllm_ascend/ascend_forward_context.py:195`、`196`），因此 prefill 没有 MLP 预取带来的额外通信。

## 7. 与 cp_balance 相关的可扩展点

按 zigzag 改造需要触碰的位置排列（base 里全部"能挂载"的点）：

1. `enable_dsa_cp()` 判定与前置约束（要加 zigzag 开关或改切片策略的第一跳）：`vllm_ascend/utils.py:1372`，SP 依赖报错 `vllm_ascend/utils.py:1390`；o_proj 行为开关 `enable_dsa_cp_with_o_proj_tp()`：`vllm_ascend/utils.py:1397`、`vllm_ascend/utils.py:1410`。缓存清理入口（开关热刷新）：`vllm_ascend/utils.py:132`、`136`。
2. metadata 切片点（连续切片的唯一实现处，zigzag 需要在这里改 `local_start/local_end` 与 index 映射）：`vllm_ascend/attention/sfa_v1.py:397`-`481`；重点是 `402`-`405`（区间计算）、`414`/`416`/`419`（slot_mapping pad + 切片）、`421`/`422`（cos/sin 切片）、`456`-`470`（per-request local 长度与 `actual_seq_lengths_*`）。
3. 切片结果的落点（改 zigzag 必须保证这些字段语义一致）：`DSACPContext` `vllm_ascend/attention/sfa_v1.py:197`-`205`（新增字段的落点，例如 per-rank token 索引/置换表）；`AscendSFAMetadata.dsa_cp_context` `vllm_ascend/attention/sfa_v1.py:239`；构造点 `vllm_ascend/attention/sfa_v1.py:474`、`507`。
4. builder 层可覆写钩子（不侵入主流程就能改 metadata 视图）：`AscendSFABackend.get_builder_cls` `vllm_ascend/attention/sfa_v1.py:166`；`_build_with_metadata_view` `vllm_ascend/attention/sfa_v1.py:356`；DCP 子类的实现范式（含 `dsa_cp_context` 二次改写）`vllm_ascend/attention/context_parallel/sfa_cp.py:285`、`vllm_ascend/attention/context_parallel/sfa_cp.py:261`、`vllm_ascend/attention/context_parallel/sfa_cp.py:269`。
5. impl 层所有 `enable_dsa_cp` 判断点（zigzag 下 KV 交换/缓存副本语义与 query 收集需要同步调整）：`vllm_ascend/attention/sfa_v1.py:684`（head 数）、`780`、`781`（o_proj 全量权重初始化）、`858`-`861`（禁融合预处理：DSA-CP 被列为不支持原因）、`1731`-`1785`（KV/indexer all_gather）、`1831`-`1888`（写 cache + o_proj 权重 gather）、`1986`-`1990`（读 context）、`2079`-`2083`（kv_slots）、`2176`-`2179`（topk token 数）、`2212`-`2228`（o_proj 出口）。
6. impl 空实现钩子（base 为 CP 预留、DCP 已示范覆写，zigzag 可挂在这里）：`_record_query_gather_context` `vllm_ascend/attention/sfa_v1.py:1707`（base 直接 return，是 query 侧收集的天然挂点）、`_get_full_kv` `vllm_ascend/attention/sfa_v1.py:1233`、`_get_sfa_kv_slot_mapping` `vllm_ascend/attention/sfa_v1.py:1890`、`_maybe_store_kvcache_for_c8_n_dsacp` `vllm_ascend/attention/sfa_v1.py:1787`、`_execute_sparse_flash_attention_process` `vllm_ascend/attention/sfa_v1.py:1693`、`_maybe_gather_kv_for_dsacp` `vllm_ascend/attention/sfa_v1.py:1716`。
7. 线性层/通信点：q_b/kv_b 的 DSA-CP 分流 `vllm_ascend/ops/linear_op.py:458`；SP 列并行的 all_gather `vllm_ascend/ops/linear_op.py:301`；SP 行并行的 reduce_scatter 与 `dsa_cp_attn_out` 特判 `vllm_ascend/ops/linear_op.py:332`、`337`、`357`、`413`；自定义算子注册 `vllm_ascend/ops/register_custom_ops.py:49`、`90`、`131`。
8. o_proj 通信点（prefill 用权重 all_gather 替代激活 reduce_scatter，是 zigzag 最可能改动的地方）：`vllm_ascend/attention/sfa_v1.py:1835`-`1848`（发起）、`1135`-`1183`（切换全量权重并前向）、`1041`-`1085`（全量权重缓冲池）、`2223`-`2227`（提前返回）。
9. MoE 通信点：`vllm_ascend/ops/fused_moe/fused_moe.py:639`（prepare）、`700`（finalize）；`vllm_ascend/ops/fused_moe/prepare_finalize.py:386`/`387`（TP all_gather）、`vllm_ascend/ops/fused_moe/prepare_finalize.py:523`（TP reduce_scatter）；dispatcher 侧 `vllm_ascend/ops/fused_moe/token_dispatcher.py:239`、`343`、`488`、`564`；通信方法选择 `vllm_ascend/ascend_forward_context.py:343`。
10. 出口 gather 点：`vllm_ascend/worker/model_runner_v1.py:2601`-`2603`、实现 `vllm_ascend/worker/model_runner_v1.py:2526`-`2533`；logits 计算点 `vllm_ascend/worker/model_runner_v1.py:2071`、`2072`。
11. 模型 forward 主循环（逐层入口，是"每层重排/重切片"的挂点）：`vllm_ascend/patch/worker/patch_deepseek_v2.py:333`、`343`、`365`；MLA 层前向挂点 `vllm_ascend/ops/mla.py:201`-`220`、`vllm_ascend/ops/mla.py:225`-`242`。
12. 批次/metadata 构造（若要按请求而非按 token 连续切片，需要改这里的 query_start_loc/seq_lens 语义）：`vllm_ascend/worker/model_runner_v1.py:2870`-`2901`，配套 `_prepare_inputs` 的 `query_start_loc`/`positions`/`slot_mapping` 写入：`vllm_ascend/worker/model_runner_v1.py:974`、`1163`、`1182`；padded token 数 `vllm_ascend/worker/model_runner_v1.py:2605`。
