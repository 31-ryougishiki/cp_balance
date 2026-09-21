> 历史文档（2026-09 上旬的交接/评审期记录）：结论可能已过时，当前状态以 `docs/scripts_review.md` 为准。
>
> 已变事实（2026-09-21 更正）：
> - 本文的工具路径（`cp_balance/*.py`、`cp_balance/run_matrix.sh`）已不存在：现在分在 `accuracy/`、`perf/`，
>   `run.sh` 只吃配置名（旧的 `bash run.sh eth2 8034` 会报错）。
> - 分支判据是 `accuracy/check_branch.py` 的 `RESULT: PASS`，证据串 = `branch=ZIGZAG` 或 `[CP_BALANCE][plan]`。
> - §5 提到的 `_can_zigzag` 在当前线不存在（门是 `zigzag_gate_reason`）；slot==-1 那条已定案：`round2_verify` 锚点不修、a30 永久 SKIP。
> - 被测树/仓的 commit 都变了（现行：harness `main`、被测树 b64c9569b）。

# cp_balance 代码分析与复查

复查方法：静态调用链 + 离线条件模拟；第一遍核对请求调度 -> attention metadata ->
embedding -> SFA -> MLP/MoE -> 模型出口 -> logits 的形状/顺序/长度，第二遍核对
通信 group -> reduce/all-gather -> KV slot -> padding 边界。只记录有结论的问题，
最后更新到 vllm-ascend `78b63b2d9e9474ff30de7b15d0ce7b3498ea8d4d`。

## 1. 版本与入口

- 模型 GLM-5.2（`GlmMoeDsaForCausalLM`，MLA + DSA indexer）。
- 原版 `vllm-ascend-base` 只有 DSA-CP 连续切片路径；被测 `vllm-ascend`
  当前分支新增 model-level zigzag cp_balance。
- 入口 `cp_balance/run.sh`；可覆盖 `VLLM_ASCEND_CP_BALANCE`、
  `VLLM_ASCEND_CP_BALANCE_REDUCE_MODE`、`VLLM_ASCEND_CP_BALANCE_DEBUG`。

## 2. 原版 DSA-CP prefill 路径（B，CP_BALANCE=0）

1. `NPUModelRunner._prepare_inputs` 按 TP 补齐调度总词元数，构建
   `AscendCommonAttentionMetadata`，`slot_mapping` 尾部补 -1。
2. `AscendSFAMetadataBuilder._build` 在 `enable_dsa_cp()` 时按 rank 连续切片
   `local_start=rank*T/TP`，`slot_mapping_cp`、cos/sin、
   `actual_seq_lengths_query/key` 均取该切片，结果存入 `DSACPContext`。
3. `DeepseekV2Model.forward`：FlashComm 下 `embed_tokens` 返回 reduce-scatter
   后的连续局部切片，逐层调用 `DeepseekV2DecoderLayer`。
4. 每层：SFA 只算局部 Q，KV 用 all_gather 汇总；prefill o_proj 用 TP 全量
   权重且输出 rank-local；MLP/MoE 先 all_gather 完整序列，GEMM 后再
   reduce_scatter 回局部行。
5. `_model_forward`：`_all_gather_hidden_states_and_aux` 拼回自然序，logits
   只取 `logits_indices` 的最后一词元生成首 token。

B 的注意力计算量按 rank 递增，最后一个 rank 最重，所以需要 cp_balance。

## 3. cp_balance 路径（C，CP_BALANCE=1）

1. `can_enable_zigzag_for_batch`：纯 prefill、TP=CP>1、DP=1、无 DCP
   replicated、非 V2 runner，每条请求 extend_len >= 2*TP，总实际词元 >=
   MIN_TOKENS，且 `num_tokens_pad % TP == 0`。
2. `build_zigzag_plan`：每条序列切 2*TP 块，rank r 拿第 r 块和第 2*TP-1-r 块；
   remainder 少量分配保证每 rank 局部行数严格为 `T/TP`；生成 `zigzag_index`、
   `zigzag_gather_index`、`inv_gather_index` 和各 half 的 q/kv 长度。
3. 入口：embedding 先 all_gather 回自然序，再按 `zigzag_index` 取
   `[prev_blocks, next_blocks]`；positions 用同一 index；`zigzag_active` 存入
   forward context。
4. SFA：Q/KV 投影、RoPE、KV 写 slot 均按 rank-local zigzag 序；KV 写回自然
   slot 前用 `zigzag_gather_index` 重排；attention/indexer 按合并的 zigzag
   长度在局部序上计算；o_proj 仍用全量权重，输出 rank-local。
5. 层间 MLP/MoE：all_gather 后为 `[r0局部, r1局部, ...]`，reduce_scatter 后
   回到同一 rank 的局部块。
6. 出口：`zigzag_gather_hidden_states_and_aux` 一次 all_gather +
   `inv_gather_index` 还原自然序，再进 logits。

## 4. 根因与已修正问题

历史 A/B 确认：layer 0 的 attention 输出、pre-MLP、量化输入 `gu_q/dn_q` 在
B/C 下逐位相同，第一个不等点在 row-parallel down_proj 之后的跨 rank 归约；
`HCCL_DETERMINISTIC=strict` 可让 L2048 首个请求逐位相同。根因是 ReduceScatter
的归约顺序随 token owner 改变。

1. 跨 rank 归约的 owner 相关舍入（高，已修正）：原
   `vllm_ascend/distributed/utils.py::fixed_order_reduce_scatter` 用
   `all_to_all_single + fixed_order_rank_sum`，假设 HCCL 的 all_to_all 切分
   语义与本地推导一致；不成立时 B/C token 值不同。现默认改为
   `AllReduce + slice`，整张量按 rank 归约后切本地 chunk，归约顺序只由通信
   算法决定，与 owner 无关；原 all_to_all 保留为
   `VLLM_ASCEND_CP_BALANCE_REDUCE_MODE=alltoall` A/B。接入
   `SequenceRowParallelOp`、`MLPRowParallelOp`、
   `_fixed_order_dsa_cp_reduce_scatter`（embedding / MoE finalize）。提交
   `7f4f7f34f`。
2. `query_lens_cpu` 过滤后 block_table 行未同步（中，已修正）：`_build` 中
   `query_lens_cpu/prefix_lens_cpu` 只含 `raw_query_lens > 0` 的请求，原
   `block_table_zigzag = cat([block_table, block_table])` 仍包含全部
   `num_reqs` 行；空行恰好在尾部时才一致，空行在中间时 batch 0 会使用错误
   请求的 block table。现用同一个 `real_req_indices` 先 `index_select`，不足
   `num_reqs` 补零后再复制 prev/next 两份。提交 `e5e86f878`。
3. metadata plan 异常会直接打断服务（中，已修正）：plan 的
   `num_actual_tokens != sum(query_lens)` 等断言说明边界不符预期，不应崩溃。
   原 `_build_zigzag_meta` 的 `ValueError/AssertionError` 会直接抛出使请求
   失败；现捕获 `ValueError/AssertionError/RuntimeError`，warning 后退回本
   batch 连续切片 DSA-CP。提交 `884c79556`。

## 5. 仍开放、需远端日志确认的问题

4. scatter 对 `slot_mapping == -1` 的跳过行为（高假设）：
   `_maybe_store_kvcache_for_c8_n_dsacp`、indexer cache write 中
   `slot_mapping_sfa[zigzag_gather_index]` 含 padding 行的 -1，预期
   `npu_scatter_nd_update_` 跳过；该行为未在 cp_balance 路径显式验证。若 -1
   被当成最后一个 slot，最后一个真实 token 的 KV 会被覆盖，首 token 大幅
   偏移。判据：打开 `VLLM_ASCEND_CP_BALANCE_DEBUG=1` 后对比最后一个真实
   token 对应 slot 的 KV dump，或临时只 scatter `slot >= 0` 的行 A/B。
   普通 DSA-CP 也依赖 -1 跳过，暂按可跳过处理，需远端确认。
5. C 把补齐词元当作有效 query 计算（中）：
   `build_zigzag_plan` / `_build_zigzag_meta` 把 padding 追加到最后一个序列并
   参与块划分，`actual_seq_lengths_query_zigzag` 仍把它当有效 query；B 连续
   切片在最后一个 rank 上会少算这部分 query。逐词元算子不会把 padding 结果
   混给真实 token，但 padding 行可能访问未写入的 KV slot 产生 NaN/Inf，需
   dump 确认其在 SFA/MLP 后仍是孤立行。开放，判断不污染真实 token，但应重点
   检查。
6. MoE/EP group 与 fixed reduce 的 group 假设（中）：
   `register_custom_ops.py::_fixed_order_dsa_cp_reduce_scatter` 在
   `dp_metadata is None` 时用 `get_tp_group()`；MoE prepare 走 EP group 的
   all-gather，finalize 本应在同一 group 做 owner-independent reduce。目标
   DP=1、EP=TP 时 rank 集合相同，但 EP group 的顺序/集合若与 TP 不同就会归约
   错对象。开放；远端启动日志打印 EP/TP world_size 与 rank 顺序即可确认。
7. `_can_zigzag` 入参名与实际含义不一致（低，清理）：
   `sfa_v1.py::_can_zigzag` 的 `num_tokens` 常被理解为未补齐真实词元数，实际
   `_build` 传入的已是 `num_input_tokens`，使
   `num_tokens != num_tokens_pad` 永不触发。当前无功能错误，后续应改名或删除。
8. model runner padding 与资格判断不是同一函数（低）：
   `worker/model_runner_v1.py::_pad_for_sequence_parallelism` 注释声称与
   metadata builder 共用 eligibility check，实际只对齐 `tp_size`，新增的
   `num_scheduled_tokens_np` 未使用。当前 zigzag plan 已支持 TP 对齐，无直接
   错误；注释和参数需清理，避免后续两边不一致。
9. `use_sequence_parallel_moe` 的隐含前提（低）：
   `patch/worker/patch_deepseek_v2.py::_zigzag_layer_forward` 假设
   `self.use_sequence_parallel_moe == False`（zigzag 只在 DP=1 启用，该属性
   要求 DP>1，当前成立）。若上游未来修改该条件，此路径会缺少原来的逐层
   all_gather/reduce_scatter。建议在 `can_enable_zigzag_for_batch` 显式判断或
   在层入口 assert。

## 6. 结论与下一步

- 入口 shard、SFA 本地序、KV/索引写重排、出口 rerange 的置换自洽：
  `local ranks -> zigzag_gather_index -> inv_gather_index -> natural`，离线
  随机长度模拟通过。
- 最大风险是归约实现/通信域假设，以及 padding 行在 scatter 和 attention
  query 长度中的处理。
- 远端第一步按 `cp_balance/README.md` 跑 20 组首词元对比，同时打开
  `VLLM_ASCEND_CP_BALANCE_DEBUG=1`；失败再按第 4/5/6 条 dump。若首 token
  仍偏，检查顺序：1) 启动日志是否有 `[CP_BALANCE][plan]`；2) allreduce 不行
  则 alltoall A/B；3) KV/scatter（第 4 条）；4) 合并 indexer/SFA 单 kernel
  可疑时用历史两 call 版 A/B；5) 多请求/非整除长度，已有 CPU 覆盖/置换检查
  可扩展。

## 7. 远程验证命令（一次只跑一步）

> 这一节的老命令（`bash run.sh eth2 8034`、`cp_balance/compare_first_token.py`）已经不适用：
> `run.sh` 现在只吃 `configs/<名字>.json`，工具都搬到了 `accuracy/`、`perf/`，开关来自配置字段
> 而不是手工 export。现在等价的做法是：

```bash
# 分支自证（一次服务起停）：长 prompt 出现 zigzag 证据、短 prompt 只有 CONTINUOUS
bash tests/run_tests.sh --only smoke/s11_zigzag_request
# 或手工：python3 accuracy/check_branch.py --url http://127.0.0.1:8035 --log <服务日志>

# 精度对比（含 C 验收与 B 等价性）：
bash tests/run_tests.sh --only accuracy/a10_matrix_gate --keep-going

# 产物统一在 tests/_out/<stamp>_<family>/ 下。
```

判据：末行 `[compare] RESULT: PASS` 且 `first-token match: 20/20`；失败回传
两个 JSON 和带 `VLLM_ASCEND_CP_BALANCE_DEBUG=1` 的 server log。

## 8. 本轮版本

- vllm-ascend `78b63b2d9e9474ff30de7b15d0ce7b3498ea8d4d`（基于
  `d22ddb2ef`：`7f4f7f34f` 默认 allreduce 并修 owner 相关归约；`e5e86f878`
  同步过滤 block_table；`884c79556` metadata plan 异常回退；新增 debug plan
  日志）。
- cp_balance 测试目录 `c1fc606`；远端同步后先确认启动日志有
  `[CP_BALANCE][plan]`，再执行第 7 节。
- 新增分支打点：`[CP_BALANCE][branch]`（每 rank 每 batch 一行，两条分支都打，
  含第一个拒绝 zigzag 的门名），由 `VLLM_ASCEND_CP_BALANCE_DEBUG=1` 控制；
  判定脚本 `cp_balance/check_branch.py`（短 prompt 期望 CONTINUOUS，长 prompt
  期望 ZIGZAG）。谓词拆成 `zigzag_ineligible_reason` + 布尔包装
  `can_enable_zigzag_for_batch`，随机 30000 例与旧实现逐例一致。

## 9. 非 cp_balance 路径与 base 的等价性（本轮改造）

判定口径：`VLLM_ASCEND_CP_BALANCE` 只在 `layers/cp_zigzag.py` 的资格判定里读一次；
其余 cp_balance 代码必须落在「每 forward」的新判定 `zigzag_active()`
（`ascend_forward_context.py`）或 `dsa_cp_context.zigzag_index is not None` 之下，
CP=0 才与原版 DSA-CP 逐位一致。

本轮修掉的两类非门控差异：

1. 归约门用错（高，已修正）：`ops/linear_op.py` 的 MLP / sequence 两处与
   `ops/register_custom_ops.py` 的 embedding+MoE finalize 一处，原来按
   `enable_dsa_cp()` 判断；该判定在 CP=0 时同样为真，于是 B 也走了
   owner-independent 归约（默认 allreduce + slice）。现改为 `zigzag_active()`：
   B 默认即走原 `reduce_scatter` / `tensor_model_parallel_reduce_scatter`，
   `VLLM_ASCEND_CP_BALANCE_REDUCE_MODE` 只影响 C。
2. `_q_proj_and_k_up_proj` 缺 base 的融合算子（中，已修正）：`attention/sfa_v1.py`
   补回 base 提交 `0cf905bfb` 的 `npu_transpose_batchmatmul` 分支
   （`perm_x1=(1,0,2), perm_x2=(0,1,2), perm_y=(1,0,2)`），没有该算子时回退
   `transpose+bmm+transpose`。

新增调试日志（`VLLM_ASCEND_CP_BALANCE_DEBUG=1`，每进程一次）：
`[CP_BALANCE][reduce] path=fixed_order mode=...` 与
`path=native site=mlp|sequence|pad_and_reduce`。

校验工具现在的位置：`accuracy/check_b_path.py`（静态：门控、开关外泄、融合算子块一致性），
`accuracy/check_branch.py`（运行期走了哪条分支），`accuracy/compare_first_token.py compare --require-text`（首 token + 前 120 字符文本），
`perf/check_cp_balance_fields.py`（ZigzagPlan / DSACPContext 字段对账）。

跑参照树不用手工 export `VLLM_ASCEND_REPO`：配置里写 `repo_tree: base`，`serve_config.py` 会自己设好
`VLLM_ASCEND_REPO` 与 `PYTHONPATH`（手工 export 会被配置覆盖）。





