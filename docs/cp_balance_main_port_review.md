> 历史文档（2026-09 上旬的交接/评审期记录）：结论可能已过时，当前状态以 `docs/scripts_review.md` 为准。
>
> 已变事实（2026-09-21 更正，正文里“未修/先取证”的写法按此读）：
> - §1 的 A-B1（dp 门自相矛盾）已按“出路 1”落地：被测树 1b638aa2a + 参照树 `patches/dsa_cp_dp1.patch`；
> - §1 的 B3（o_proj 出口）已修：6ca45f53e 在 zigzag 下直接写回 rank-local 行；
> - §3 的“两者不可能同时成立 / 永远走连续切片”已作废（见头部更正），新阻塞是 zigzag 的 MoE 布局（`scripts_review.md` §8）；
> - §4 的 B5（`for_draft` 死参数）已修（99d03e477/b64c9569b）；§6 的“先做 A-B1 取证/再改 dp 前提”已完成，现在第一步是 `bash verify.sh`；
> - 仍未修且仍有效：B4（回退只还原 SFA）、B6（MC2 prepare 按整批切）、H1/H2。
>
> 更正（2026-09-21）：本文 §A-B1 的结论「zigzag 可用 ⟹ dp==1，DSA-CP 生效 ⟹ dp>1，两者不可能同时成立」已被推翻：
> 那道门是写窄了（#15549 把 `enable_dsa_cp` AND 上 `use_sequence_parallel_moe`），本地已改成只判 `has_indexer`，
> dp=1 可以开 DSA-CP；但 zigzag 还多一层 MoE 布局问题待决（见 `scripts_review.md` §4.1c/§8）。

# cp_balance 移植到 main：分阶段 review 结论

评审对象：vllm-ascend 分支 `cp_balance`（移植提交 `612d03a8b`，评审期间修到 `3a954245b`）。
评审方式：三个独立 reviewer（只读、AST/git 取证、行号逐条核对），详细报告在：

- `docs/port/review_a_layout.md`（阶段 A：布局与元数据，371 行）
- `docs/port/review_b_execution.md`（阶段 B：执行路径，322 行，含 6 个必答问题）
- `docs/port/review_c_reductions_parity.md`（阶段 C：归约覆盖 + 与老实现差异，496 行）

## 1. 一览

| 阶段 | 范围 | 未修 BLOCKER | 已修 BLOCKER | 高危 |
| --- | --- | --- | --- | --- |
| A 布局/元数据 | cp_zigzag / zigzag_cp / sfa_cp(builder) / indexer | A-B1 dp 门自相矛盾 | A-B2 dataclass 缺字段、A-B3 indexer slot 用错变体、A-B4 隐藏态二次切片 | H1 两侧 num_tokens_pad 基数不同、H2 gate 不看 PCP、H3 缓冲差 1、H4 -1 slot |
| B 执行 | sfa_cp(impl) / forward context / 模型边界 / draft | B3 o_proj 出口、B4 回退半还原、B6 MC2 prepare | B1 import 名（与 C-B2 同）、B2（=A-B4） | B5 死参数 for_draft、B7 -1 slot |
| C 归约/漏搬 | linear_op / shared_experts / distributed / embedding / device_op / envs | C-B1 embedding 漏 return、C-B2 import（同 B1） | 同左 | 门控覆盖、漏搬清单 |

## 2. 已修（按提交）

- `8912139f9` indexer 的 zigzag 资格门看不到 draft 步：`build_for_drafting` 没把 `draft_index` 传进 `_build`，会出现「SFA 连续切片 + indexer zigzag」的单侧布局。
- `d9dfc4497` planner 热路径改用 runner 发布的 `seq_lens_cpu`（原写法每批一次 device→host 同步）。
- `3572231a0`（A 阶段三条）：
  - `ZigzagCPPlan` 未声明 `slot_mapping_cp_gathered` 却在构造时传入 → 首个可用 prefill 直接 TypeError；已补字段（现 13 字段 ↔ 13 kwargs）。
  - indexer 写的是 TP all-gather 之后的 k_li，slot 必须用 `zigzag_gather_index` 变体；原来用局部序会行数不匹配/错序。
  - `AscendSFADSACPImpl._prepare_native_hidden_states` 在 zigzag 下把已是 rank-local 的 hidden 再 pad 到全量并取连续切片 → 除 rank0 外全零；已早退。
  - plan 构建的 except 放宽到 `Exception`：规划出错降级为连续切片，不允许打死 forward。
- `3a954245b`（B/C 阶段）：
  - `vocab_parallel_embedding._embed_partial` 拆函数时**漏了 return** → 所有模型的 embedding 返回 None（不止 zigzag）；已补。
  - `sfa_cp.py` 从 `vllm_ascend.utils` 导入 `use_v2_model_runner`（实际在 `mrv2_utils`）和两个 builder 导入不存在的 `dsa_cp_with_o_proj_tp_for_config` → **import 即失败**；改为 `mrv2_utils.use_v2_model_runner` + `enable_dsa_cp_full_o_proj()`。
  - zigzag 序列长度缓冲 `2*max_num_reqs+1` 差 1（runner 可能多一个 padded 请求槽）→ 放大到 `2*(max_num_reqs+1)+1`。

## 3. 未修 BLOCKER

### A-B1（设计级，必须先决策）main 上 DSA-CP 与 zigzag 的 dp 前提互斥

- zigzag 资格门拒绝 `dp_size > 1`（`layers/cp_zigzag.py:571-572`，入参 `parallel_config.data_parallel_size`，`sfa_cp.py:483` / `indexer.py:798`），同源第二道闸在 `ascend_forward_context.py:321-326`（DP>1 → 还原连续切片）。
- 但 main 上 `enable_dsa_cp` 被 AND 上 `use_sequence_parallel_moe`（`vllm_ascend/ascend_config.py:660-664`），而 `use_sequence_parallel_moe` 要求 `data_parallel_size > 1`（本地 vllm 84030bbe3d `vllm/config/parallel.py:711-726`）。
- 结论：`zigzag 可用 ⟹ dp==1`，`DSA-CP 生效 ⟹ dp>1`，两者不可能同时成立 → 现在的移植在 main 上永远走连续切片。老线能跑是因为老的 `enable_dsa_cp` 只要求 FlashComm1（`enable_sp`），部署是 tp16+dp1。
- 三条路：
  1. 放宽 `ascend_config` 对 `use_sequence_parallel_moe` 的依赖，回到 tp+dp1+FlashComm1 的部署形态，并验证 main 的 `sfa_cp` 在 dp1 下依旧自洽（改动小，但偏离上游设计）；
  2. 让 zigzag 支持 DP>1（每 DP rank 的 padding、`dp_metadata`、mc2_mask、MoE 通信都要重做，风险最高）；
  3. 放弃在 main 上跑 zigzag，把它作为 main 线之外的一个历史结论（即承认这次移植只在 dp1 语义下成立）。
- 远端最小取证（10 分钟，先做）：`VLLM_ASCEND_CP_BALANCE_DEBUG=1` 起服务发长 prompt，看是否只有 `reason=dp>1`；同时打印 `enable_dsa_cp()` 与 `parallel_config.data_parallel_size`。

### B3（执行级）o_proj 出口仍按「全量自然序 / rank 连续槽」

- `sfa_cp.py:637-645`、`809-845`（输出缓冲 `ops/mla.py:250-255`）：all_gather 后取前 `output.shape[0]` 行、或按 `rank*local` 连续槽写，zigzag 下 rank>0 拿到的都是别人/补齐区。
- 与 `patch/worker/patch_deepseek_v2.py:442-449` 的注释（"SP 分支不会触发"）自相矛盾，需要与 B1 的 dp 决策一起定。

### B4（执行级）回退只还原一半

- `ascend_forward_context.py:118-174` 的 `_disable_zigzag_metadata_for_fallback` 只认带 `dsa_cp_context` 的 SFA metadata；indexer metadata（`indexer.py:58-98`）没有该字段 → draft/V2/DP>1 回退时 SFA 连续切片、indexer 仍 zigzag。

### B6（执行级）MC2 prepare 与 zigzag 局部行错位

- `ops/fused_moe/prepare_finalize.py:252-311` + `ascend_forward_context.py:353-362`：MC2 路径把输入当作全量 batch（pad 到 `padded_num_tokens` 再按 rank 切），zigzag 局部行下 rank>0 取到补齐区；A5 `num_experts_per_tok=8`，2048~4096 token 区间会命中这条。

## 4. 高危（未修，详见各报告）

- H1 SFA 与 indexer 的 `num_tokens_pad` 基数在 dspark adaptive verification 下不同（`sfa_cp.py:344-346` vs `indexer.py:1084-1093`）→ 两套局部布局静默错数。
- H2 zigzag 资格门不看 PCP；indexer 在 PCP 下跳过 zigzag（`indexer.py:1125-1126`）→ SFA 单侧 zigzag。
- H3/H4 见上（H3 已修缓冲大小，H4 是 `slot==-1` 是否被 scatter 跳过，只能 NPU 定向验）。
- B5 `spec_decode/llm_base_proposer.py:2528-2535` 的 `for_draft=True` 在新 builder API 里是死参数（真正入口 `build_for_drafting`，已被 draft_index 修复覆盖）。
- C 门控覆盖：本配置（TP16、fine-grained TP 未开、DP=1）下已门控的两处（`linear_op.py:197`、`shared_experts.py:246-258`）其实是死代码；未门控的 owner 依赖 reduce_scatter 站点：`register_custom_ops.py:98`（A5/A2 的 MoE-ALLGATHER 路径会命中）、`linear_op.py:278`、`models/common/ops/sequence_parallel.py:38`、`attention/dsa_v1.py:1702`、`prepare_finalize.py:544/548`。all_reduce 站点无需改。
- 漏搬：`input_ids` 重排（新 main MoE 不再从 forward context 取 input_ids，只有 hash-routing/tid2eid 模型受影响；`cp_zigzag.py:78-100` docstring 陈旧）、C8 reshape-optim 在 zigzag 下未关闭（老实现有 `and not zigzag_active`）、`register_custom_ops` 的门控与 `site=pad_and_reduce` 打点。

## 5. 中/低（摘要）

M：热路径上 planner 每次 build 新建约 10 个 device 张量、`bool(in_range.any())` 一次同步（indexer：`indexer.py:832-836`）；padding 行计入 2B 累计长度（`zigzag_cp.py:193-214`）；`_update_parallel_slot_mapping` 的 zigzag 分支当前不可达（`sfa_cp.py:571-581`）；`except Exception` 可能造成单侧回退；`_allreduce_slice_reduce_scatter` 就地 all_reduce 后返回视图（`distributed/utils.py:30-52`）。
L：`DSACPContext.num_tokens/local_end/local_tokens`、`ZigzagCPPlan.kv_len_*` 无读点；`resolve_seq_lens_cpu` 的回退仍会同步；`is_prefilling_cpu` 与 `is_prefilling` 同对象（冗余但无害）。

## 6. 远端验证顺序（建议）

1. **先做 A-B1 取证**：起服务 + `VLLM_ASCEND_CP_BALANCE_DEBUG=1`，长 prompt 看 `[CP_BALANCE][branch]` 是否只有 `reason=dp>1`，并打印 `enable_dsa_cp()` / `data_parallel_size`。这一步决定后面所有工作是否有意义。
2. 按决策改 dp 前提（放宽 ascend_config 或支持 DP>1）后，再验 zigzag 端到端（smoke/s11 的 ZIGZAG 判定）。
3. `slot==-1` 语义（H4/B7）用最小用例定向验。
4. 通过后再跑精度矩阵（B==base + C 验收）与性能采集。

## 7. 结论

- 移植的**结构**成立：布局单一来源、出口 gather 反转、V2/draft/DP 三类回退、归约门控；静态阶段能查出的错（import 名、dataclass 字段、漏 return、死参数）已全部修掉（4 个提交）。
- 但**能不能真正跑起来**目前卡在两处：dp 前提互斥（A-B1，需决策）与 o_proj 出口（B3）；其余 BLOCKER 都已修或已定位到具体行。
- 详细证据（每条含文件:行号、现象、原因、建议、本地能否判定）见 `docs/port/review_a_layout.md`、`review_b_execution.md`、`review_c_reductions_parity.md`。
