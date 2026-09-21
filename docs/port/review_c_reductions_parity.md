> 历史文档（2026-09-18 移植期取证/评审）：结论可能已过时；dp>1 那道门与 zigzag 的现状见 `docs/scripts_review.md` §4.1/§4.1c/§8。
>
> 已变事实（2026-09-21 更正）：
> - §C-H2「zigzag 资格门必然返回 dp>1 / zigzag 依旧不可达」已作废：dp=1 已定案，两个树都改了门（§4.1）。
> - 远端取证不要再 grep `…Disabling DSA-CP.` 期望它出现：打过 `patches/dsa_cp_dp1.patch` 的树改成 info `DSA-CP is enabled without sequence-parallel MoE …`。
> - §1 的 C-B1（`_embed_partial` 缺 return）与 C-B2（import 名）两条 BLOCKER 在当前树已修。
> - 仍未修需跟踪：C-H1 站点 1（`_maybe_pad_and_reduce_impl` 无 zigzag 门控）、C-M2（indexer `_use_c8_reshape_optim` 无 zigzag 排除）、C-M1（docstring 仍称会重排 `input_ids`）。

# Stage C 评审：归约与其它挂点 + 与老实现的差异（cp_balance → vllm-ascend main）

## 0. 快照、范围与方法

* 被评审对象：任务指定的移植提交 **612d03a8b**。
* **重要**：评审期间工作树被连续前移，**当前 HEAD = `3572231a0`**（其上另有 `8912139f9 fix(indexer zigzag gate)`、
  `d9dfc4497 perf(seq_lens host copy)`、`3572231a0 fix(three blockers found by the stage-A review)`）。
  本文行号**统一按当前工作树（3572231a0）**给出；相对 612d03a8b 已修的条目一律标注「HEAD 已修」。
* 基线：`aff1b74b6`（注意 `HEAD~1` 现在等于 612d03a8b，**不能**再当基线用；本文所有 diff 都显式写 `aff1b74b6`）。
  老实现：分支 `cp_balance_v0.26.0rc` = `23ff2c23c`（引用写作 `O:...`）。
* 本 stage 范围：`ops/linear_op.py`、`ops/fused_moe/shared_experts.py`、`distributed/utils.py`、
  `ops/vocab_parallel_embedding.py`、`device/device_op.py`、`envs.py`；为回答必答问题交叉读
  `ascend_forward_context.py`、`patch/worker/patch_deepseek_v2.py`、`attention/indexer.py`、
  `attention/context_parallel/sfa_cp.py`、`ascend_config.py`、`ops/register_custom_ops.py`、
  `ops/fused_moe/prepare_finalize.py`、`layers/cp_zigzag.py`，以及本地 vLLM main 检出 `84030bbe3d`
  （`D:/code/cp_balance/vllm`）。
* 方法：只读命令（`git show` / `git diff` / `grep` / `sed` / `python -c` ast 解析）。**未运行任何 torch/NPU 代码**；
  「本地能否判定」列即指"仅靠上述静态手段能否定论"。
* 与 stage A 的关系：stage A（`docs/port/review_a_layout.md`）的 B2/B3/B4 已由 `3572231a0` 修掉（`ZigzagCPPlan`
  补 `slot_mapping_cp_gathered`、indexer 改用 gathered 映射、`_prepare_native_hidden_states` 提前 return），
  本文不重复；只在 §5 交叉引用一次。

**结论摘要：2 条 BLOCKER（都在本次 stage 范围内、且都不是 stage A 报过的）**
① 词表并行 embedding 的 `_embed_partial` 拆函数时漏了 `return`，**任意模型**的 embedding 前向都会返回 `None`；
② `sfa_cp.py` / `indexer.py` 从 `vllm_ascend.utils` 导入两个**全仓库不存在**的名字，SFA/DSA-CP 模块 import 即失败。
另有 2 条高危：门控挂点与 main 的实际归约点不重合（含 4 类完全没门控的 reduce_scatter 站点）＋
shared experts 门控在目标配置下是死代码；以及 4 条中危（input_ids 重排没搬、C8 快速写回没关、宽 except 吞掉配置错误、就地 all_reduce 的视图返回）。

---

## 1. BLOCKER

### C-B1 `_embed_partial` 缺 `return output_parallel` → 所有 embedding 前向返回 `None`

* 级别：**BLOCKER**（非 zigzag 特性问题，是整个 Ascend embedding 链）
* 文件:行号
  * 缺陷：`vllm_ascend/ops/vocab_parallel_embedding.py:287-304`：函数体最后一行是
    `output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)`，**函数没有 return**（ast：`_embed_partial returns=[]`）。
  * 消费点：`:306-307`（`_forward_origin` 第一句 `output_parallel = self._embed_partial(input_)`）、
    `:225`（`forward_zigzag_local` 的 `tensor_model_parallel_all_reduce(self._embed_partial(input_))`）。
  * 对照老实现：`O: vllm_ascend/ops/vocab_parallel_embedding.py:262-281`，同一函数末尾有 `return output_parallel`。
* 现象（静态可推演的三条路径）
  1. `tp_size > 1`：`_forward_origin:313/320` → `torch.ops.vllm.all_reduce(None, tp_group.unique_name)` → TypeError。
  2. `tp_size <= 1`（`disable_tp`、复制组、单卡）：`_forward_origin:309` 直接 `return None` → 上一层拿到 `None`。
  3. `VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL=1`：`forward_zigzag_local:225` 把 `None` 交给 all_reduce → 同上。
* 影响面：**不止 zigzag**。`vllm_ascend/utils.py:867` 把
  `REGISTERED_ASCEND_OPS["VocabParallelEmbedding"] = AscendVocabParallelEmbedding`（`:845-850` 导入），
  即所有走 vLLM 权重加载替换的模型都拿这个类；任何第一次 `embed_tokens(...)`（`forward:190-192` →
  `_forward_origin`）都会崩。只有 fine-grained embedding TP（`forward_type=="embed_tp"` → `_forward_embed_tp`，
  `:228-285`，自带 lookup/reduce_scatter）不受影响。
  仓库自带 UT `tests/ut/ops/test_vocab_parallel_embedding.py:128/149/169/186/207/225`（`forward_with_tp_size_1` /
  `forward_with_tp` / `output_shape` / `forward_zigzag_local`）在逻辑上**必然失败**——这批 UT 正是能立刻抓到它的门。
* 原因：把 `_forward_origin` 拆成 `_embed_partial` + `_forward_origin` 时，原 `else: return output_parallel`
  被改写进新 `_forward_origin`（`:308-309`），但原来的"尾部 return"没有跟着落到 `_embed_partial`。
* 建议
  1. 在 `_embed_partial` 末尾补 `return output_parallel`（与 O 完全一致）。
  2. 回归门里加跑 `tests/ut/ops/test_vocab_parallel_embedding.py`；再补一条 20 行的 ast 静态门：
     "每个名为 `_embed_partial`/`_forward_*` 的 def 至少有一条 `return`"，本次即由此类检查发现。
  3. 顺手核对 `:225` 用的 `tensor_model_parallel_all_reduce`（vllm 的 TP 组）在 `lmhead_tp`/`embed_tp`
     分组下是否合适——本方法已显式拒绝 `embed_tp`，lmhead 不走这个方法，暂无其它问题。
* 本地能否判定：**能**（`grep -n "return output_parallel"` 只有 :309/:313；`python -c` ast 直接给出 `returns=[]`）。

### C-B2 `sfa_cp.py` / `indexer.py` 从 `vllm_ascend.utils` 导入不存在的名字 → ImportError

* 级别：**BLOCKER**（SFA/DSA-CP 这条模块链在 main 上根本 import 不进来）
* 文件:行号
  * `vllm_ascend/attention/context_parallel/sfa_cp.py:47-51`：
    `from vllm_ascend.utils import (_round_up, dsa_cp_with_o_proj_tp_for_config, use_v2_model_runner, ...)`；
    使用点 `:483`（`use_v2_model_runner(self.vllm_config)`）、`:486`（`dsa_cp_with_o_proj_tp_for_config(...)`）。
  * `vllm_ascend/attention/indexer.py:46-52`：同样导入 `dsa_cp_with_o_proj_tp_for_config`；使用点 `:801`。
  * 事实核查：`grep -rn "def dsa_cp_with_o_proj_tp_for_config" vllm_ascend/` → **0 命中**（只有这两处 import 与其调用点）。
    `vllm_ascend/utils.py` 里的对应物是 `:1524 enable_dsa_cp_with_o_proj_tp()`（无参 + `lru_cache`），
    且**没有** `dsa_cp_with_o_proj_tp_for_config(vllm_config)` 这种 config 版；
    `use_v2_model_runner` 定义在 `vllm_ascend/mrv2_utils.py:178`，`vllm_ascend.utils` 里没有
    （`indexer.py:43` 已经从 `mrv2_utils` 正确导入，`sfa_cp.py:50` 走错模块）。
* 现象：`import vllm_ascend.attention.context_parallel.sfa_cp` →
  `ImportError: cannot import name 'dsa_cp_with_o_proj_tp_for_config' from 'vllm_ascend.utils'`；
  `attention/indexer.py` 同因。也就是说 `resolve_sfa_metadata_builder()`（sfa_cp.py:1819-1833 引用这两个模块）
  在任何 SFA 模型上都会先在 import 阶段炸掉，DSA-CP/zigzag 无从谈起。
* 原因：老实现是在 **`vllm_ascend/utils.py` 里新增** `enable_dsa_cp_for_config(vllm_config)` /
  `dsa_cp_with_o_proj_tp_for_config(vllm_config)`（`O: vllm_ascend/utils.py` 共 +35 行的那半，
  见 `git diff ae9c13a42 23ff2c23c -- vllm_ascend/utils.py`），再在主干里用的；移植把调用方搬过来了，
  却没把 utils.py 的这半搬过来（移植 diff 里根本没有 `vllm_ascend/utils.py`）。
* 建议
  1. 在 `vllm_ascend/utils.py` 补回 O 的两个 helper（语义照搬 `23ff2c23c`：`enable_dsa_cp_for_config` 用
     `additional_config["enable_dsa_cp"]` + `enable_sp(vllm_config)`；`dsa_cp_with_o_proj_tp_for_config` 用
     `kv_transfer_config is None or is_kv_producer`），并把 `enable_dsa_cp_with_o_proj_tp()` 改成调用前者，
     避免两份判定长期漂移。
  2. `sfa_cp.py:50` 的 `use_v2_model_runner` 改从 `vllm_ascend.mrv2_utils` 导入（与 `indexer.py:43` 一致）。
  3. **把下面这段 ast 门禁固化到 CI**（本次就是靠它发现的，30 行、无需 torch）：
     `for ImportFrom(module='vllm_ascend.*') → 校验每个 name 在该模块顶层存在（函数/类/赋值/再导出）`。
     （注意 `vllm_ascend.envs` 是 `__getattr__` 动态模块、`DeviceOperator` 是 `AnnAssign`，
     门禁要把这两类误报排除。）
* 本地能否判定：**能**（ast + grep 已定论；无法用 `import` 复验，因为需要 torch/torch_npu）。

---

## 2. 高危

### C-H1 门控覆盖：老实现的两处门控都搬了，但 4 类 row-parallel `reduce_scatter` 站点没有门控；本配置的命中面与门控面不重合

* 级别：**高危**（"B==C 逐位一致"的保护落在达不到的站点上；跨档位/跨模型会静默失去 owner-independent 保证）
* 已门控（对照 O）
  * `vllm_ascend/ops/linear_op.py:196-203`（`MLPRowParallelOp.apply_impl`；O: `linear_op.py:199-215`）。
  * `vllm_ascend/ops/fused_moe/shared_experts.py:246-258`（`_pad_and_reduce_scatter`）。
    O 的同语义门控在 `O: ops/register_custom_ops.py:25-48 + 115-124`（`_fixed_order_zigzag_reduce_scatter`）。
* **没门控**的 token 行 reduce_scatter 站点（`file:line`，全部为 dim-0 token 行）
  | # | 站点 | 说明 | 本配置（TP16、fine-grained TP 未开、DP=1、`--enforce-eager`、无 kv-transfer）命中？ |
  | --- | --- | --- | --- |
  | 1 | `ops/register_custom_ops.py:98`（`_maybe_pad_and_reduce_impl`: `get_ep_group().reduce_scatter(x, 0)`） | 唯一调用点 `ops/fused_moe/prepare_finalize.py:529 _finalize_with_ep_group`，由 `_use_ep_sequence_parallel()` = `moe_config.is_sequence_parallel`（`:373-380`）决定 | **A3 档不命中**：`enable_expert_parallel=True` 时 MoE 大 batch 走 ALLTOALL、小 batch 走 MC2/FUSED_MC2，二者都不经它（`prepare_finalize.py:501-517` / `:316-322`）。**A5 档（`CAPACITY_AND_WORLD_SIZE`，world_size ≤ top_k 时返回 ALLGATHER）与 A2 档（`CAPACITY_AND_EXPERT_DENSITY` 非 MC2 时返回 ALLGATHER）会命中**（`ascend_forward_context.py:427-446`(A2)/`:448-462`(A3)/`:466-492`(A5)）。O 在 `register_custom_ops.py:118-121` 曾对同一站点加固定序归约 |
  | 2 | `ops/linear_op.py:278`（`OProjRowParallelOp`，`get_otp_group()`） | 仅 `oproj_tp_enable()`（fine-grained O-proj TP>0）时构造（`linear_op.py:326-334`） | 不命中（配置未开）。若开，则 zigzag 资格门仍可能通过（`full_o_proj` 只看 kv_transfer 角色，`utils:1524`）→ 属于"开了 fine-grained O-proj TP + zigzag"的组合风险 |
  | 3 | `vllm_ascend/models/common/ops/sequence_parallel.py:38`（`sp_reduce_scatter`） | 调用点：`attention/context_parallel/dsa_cp.py:1693`、`models/glm5next/model.py:457`、`models/deepseek_v4/model.py:780`、`models/deepseek_v41/model.py:897`、`models/kimi_k3.py:162/546` | 不命中（这些是别的模型类/别的 attention 链；目标模型 `GlmMoeDsaForCausalLM` → `models/deepseek_mtp` → vllm `deepseek_v2`，见 `models/__init__.py:73`）。**同一 feature 若要覆盖这些模型，必须先给这里加门控** |
  | 4 | `attention/dsa_v1.py:1702`（`dist.reduce_scatter_tensor(..., group=oproj_group)`） | 非 SFA 的 DSA 注意力链里的 O-proj TP 融合路径 | 不命中（同上：fine-grained O-proj TP 未开，且 DSA-CP 走 SFA） |
  | 5 | `ops/fused_moe/prepare_finalize.py:544/548`（`get_pcp_group()` / `get_dp_group()` 的 reduce_scatter） | `_finalize_with_dp_group`，只在 `_use_ep_sequence_parallel()` 为假时走 | DP 侧被 zigzag 门 `dp>1` 排除；**PCP 侧没有被门检查**（资格门里没有 `pcp_size` 条件）→ 若将来 `prefill_context_parallel_size > 1` 且 `is_sequence_parallel=False`，会命中且无门控 |
* `all_reduce` 站点**不需要**改（这点要写进结论，避免过度修补）：`ops/fused_moe/fused_moe.py:324/325/333/347`、
  `ops/register_custom_ops.py:145/154`、`attention/context_parallel/sfa_cp.py:164/169`、
  以及 vLLM `RowParallelLinear.forward` 的 `tensor_model_parallel_all_reduce`
  （`D:/code/cp_balance/vllm/vllm/model_executor/layers/linear.py`，`AscendRowParallelLinear.forward` 直接
  `super().forward`，`vllm_ascend/ops/linear.py:385-393`）。AllReduce 的累加过程与"这一行归谁"无关，
  本身 owner-independent（移植把 `_forward_origin` 从 O 的 `maybe_pad_and_reduce` 换成 all_reduce 是正确的，
  也是不需要给 embedding 侧加门控的原因）。
* 原因（根因）：O 的 `maybe_pad_and_reduce` 在 O 里同时承担 embedding 与 MoE finalize 两个用途（O
  `vocab_parallel_embedding.py:283-288` + `register_custom_ops.py:101-124`），门控自然挂在
  `register_custom_ops`（TP 组）。T main 把 MoE 侧改成了 **EP 组**、embedding 侧改成了 all_reduce，
  "该门控的站点"从 TP 组的 pad_and_reduce 平移到了 (a) 仍走 `maybe_pad_and_reduce` 的
  ALLGATHER-with-SP finalize、以及 (b) shared experts 的 `_pad_and_reduce_scatter`。
  移植只补了 (b)（见 C-H2 对 (b) 可达性的质疑），(a) 完全没补。
* 建议
  1. 给 `_maybe_pad_and_reduce_impl`（或 `_finalize_with_ep_group`）补上同样的 gating：`zigzag_active()` +
     `x.shape[0] % get_ep_group().world_size == 0` 时走 `fixed_order_reduce_scatter(x, get_ep_group())`，
     其余保持原集合通信；debug 打点沿用 `[CP_BALANCE][reduce] ... site=pad_and_reduce`。
  2. 站点 2/3/4/5 至少加 `assert not zigzag_active()`（或 TODO 注释 + 一条 debug 日志），
     把"这些路径在 zigzag 下没有 owner-independent 归约"这件事显式化，避免以后按模型/档位迁移时静默出问题。
  3. 远端取证顺序：开 `VLLM_ASCEND_CP_BALANCE_DEBUG=1` 跑长 prefill，收集
     `[CP_BALANCE][reduce] path=fixed_order|path=native site=...` 与 `ascend_forward_context.py:546-555`
     的 `MoE comm method selected: policy=..., method=...` 两类日志（各 rank 至少一行），
     即可判定站点 1 是否真在跑；模型/配置换档时同一套日志可复用。
* 本地能否判定：**站点存在性/门控缺失=能**（grep 全文）；
  **"本配置是否命中"=不能完全判定**（需要 MoE comm type 的运行时值，我能给的是分支判定表；
  且远端 vLLM 版本是否含 `data_parallel_size > 1` 条件会改变 `is_sequence_parallel`，见 C-H2）。

### C-H2 shared experts 的门控在目标配置里是死代码；`_pad_and_reduce_scatter` 与真正的归约点不是同一个

* 级别：**高危**（门控看起来覆盖了、实际保护不到；与 stage A B1 同源）
* 文件:行号
  * 门控：`vllm_ascend/ops/fused_moe/shared_experts.py:246-258`。
  * 调用前置：`shared_experts.py:170-193 parallel_mode()`：只有
    `self.moe_config.is_sequence_parallel == True and not weights_replicated` 才返回
    `SEQUENCE_PARALLEL_ONLY`，而 `_pad_and_reduce_scatter` 只在 `SEQUENCE_PARALLEL_ONLY` 下调用
    （`:586-591`、`:601-602`）。
  * `is_sequence_parallel` 的来源：`vllm_ascend` 侧 MoE config 来自 vLLM 的 `FusedMoEConfig`
    （`ops/fused_moe/fused_moe.py:27,86`），其值 = `parallel_config.use_sequence_parallel_moe`
    （`D:/code/cp_balance/vllm/vllm/model_executor/models/deepseek_v2.py:308`；
    构造链 `vllm/model_executor/layers/fused_moe/layer.py:44-66`，`sp_size = tp if is_sequence_parallel else 1`）。
  * 属性定义：`D:/code/cp_balance/vllm/vllm/config/parallel.py:705-724`，要求
    `all2all_backend in (...) and enable_expert_parallel and tp>1 and data_parallel_size > 1`。
  * 配置级前置：`vllm_ascend/ascend_config.py:660-664`
    （`enable_dsa_cp = enable_dsa_cp and has_indexer and use_sequence_parallel_moe`）。
  * harness 配置：`cp_balance/configs/_common.json`（tp16、`devices 0..15`、无 `--data-parallel-size`）、
    `_common_a5.json`（tp8）→ **DP=1**（也与 `data/send/export/*` 的 `dp0_pp0_tp0..tp15` 命名一致）。
* 现象：本地 pin（vLLM 84030bbe3d）下 `use_sequence_parallel_moe == False`（因为 dp=1）→
  ① `enable_dsa_cp` 被 `ascend_config.py:664` 强制置 False → DSA-CP 整体不启用，`resolve_sfa_metadata_builder()`
  返回普通 `AscendSFAMetadataBuilder`（sfa_cp.py:1819-1833）；② 即使 DSA-CP 能启用（若远端 vLLM 没有
  `dp>1` 条件），`moe_config.is_sequence_parallel` 也是 False → shared experts 走 `TENSOR_PARALLEL` 模式，
  `_pad_and_reduce_scatter` **一次都不会被调用** → 移植补的门控是死代码；此时真正做 row-parallel 归约的是
  vLLM `RowParallelLinear` 的 all_reduce（owner-independent，所以"不是错误"，但"B==C 逐位一致"的保证
  落在了一个本来就安全的算子上，而真正会换 owner 的站点没被保护）。
  （`patch/worker/patch_deepseek_v2.py:443-448` 的注释正是基于"dp>1 才 use_sequence_parallel_moe"推出的结论。）
* 与 stage A B1 的关系：若为了让 DSA-CP/SP-MoE 真正生效而把部署改成 `--data-parallel-size 2`，
  则 zigzag 资格门第三道 `dp_size > 1`（`layers/cp_zigzag.py:571-572`）必然返回 `"dp>1"`，
  且 `ascend_forward_context.py:321-326` 还有第二道同源闸门 → zigzag 依旧不可达。
  **所以这条必须先解决"门控挂哪一侧"，再谈 pad 行是否污染归约**（见 §5 必答 2）。
* 建议
  1. 先远端取证：启动日志里抓 `"DSA-CP is enabled, but the current config does not support
     sequence-parallel MoE. Disabling DSA-CP."`（`ascend_config.py:660-663`）是否出现；
     再抓一条 `FusedMoEParallelConfig = ...`（`vllm/model_executor/layers/fused_moe/layer.py:62-64` 的 debug 行）
     与 `enable_dsa_cp()` / `parallel_config.data_parallel_size` / `use_sequence_parallel_moe` 的实际值。
  2. 若确认 `is_sequence_parallel == True`（远端 vLLM 无 `dp>1` 条件）：本门控有效，但要把 C-H1 站点 1
     一并补上；若为 False：shared experts 这一侧应把门控移到实际发生的归约上
     （即 `fused_moe.py:325 maybe_all_reduce_shared_expert` / down_proj 的 all_reduce 之外，
     任何 MC2/ALLTOALL dispatch 里的归约，例如 `npu_dispatch_ffn_combine`/`npu_mm_reduce_scatter_base`
     的 owner 依赖问题），否则 `linear_op` 与 `shared_experts` 两个门控都只是"看起来有覆盖"。
  3. 无论走哪条，建议在 `parallel_mode()` 里加一条 `logger.info_once` 打印解析结果 + `is_sequence_parallel`，
     让"哪个归约点被门控"这件事在运行日志里可见（现在完全不可见）。
* 本地能否判定：**部分**。代码/配置链路可判定（上表）；远端 vLLM 版本与我方 pin 是否一致需确认
  （`patch_deepseek_v2.py:446` 引用的行号 `vllm/config/parallel.py:653-668` 在本地 84030bbe3d 里对应的是
  `stateless_init_dp_group`，真正的属性在 `705-724` —— 说明移植时看的 vLLM 与我方 pin **不是同一份**，
  这一点必须先澄清）。

---

## 3. 中

### C-M1 zigzag 下 MoE 的 `input_ids` 没有重排（老实现有），`zigzag_reorder_moe_aux` 的 docstring 仍在声称有

* 级别：**中**（目标模型不命中；hash-routing 模型会静默选错 expert）
* 文件:行号
  * 老实现有：`O: vllm_ascend/ascend_forward_context.py:333-342`
    （`input_ids = input_ids.to(torch.int64)` → `zigzag_reorder_moe_aux(input_ids, zigzag_cp_context)` →
    `forward_context.input_ids = input_ids`）+ O 的 `set_ascend_forward_context(..., input_ids=None)` 形参
    （O: `ascend_forward_context.py:197`）+ `O: ops/fused_moe/experts_selector.py:247-253` 的注释。
  * 新树现状：`vllm_ascend/ascend_forward_context.py:195-212` 的签名里**没有** `input_ids`；
    `:343` 只挂 `_EXTRA_CTX.zigzag_cp_context/active`；`:353-361` 只对 `mc2_mask` 调
    `zigzag_reorder_moe_aux`；而 `layers/cp_zigzag.py:78-100` 的 docstring 仍写着
    "``set_ascend_forward_context`` applies the reorder to ``input_ids`` once per forward"（与代码不符）。
* 现象/影响：zigzag 下模型边界把 hidden_states 变成 rank-local `[prev,next]` 行，MoE prepare 再 all-gather 成
  rank-concatenating 序；但 `forward_context.input_ids` 仍是**自然序、长度 num_tokens_pad**。
  只有 hash routing（`ops/fused_moe/router/fused_topk_router.py:158-175`：`scoring_func == "sqrtsoftplus"`
  且 `tid2eid is not None`，随后 `pad_and_split_input_ids` / `sequence_parallel_chunk`）会消费它 →
  行错位 → 选中错误 expert。目标模型 `GlmMoeDsaForCausalLM`（sigmoid/noaux_tc，无 `tid2eid`）不命中；
  DSv4 / DSv4.1 vision-hash 一类模型命中。
* 建议：二选一并保证文档一致 ——
  (a) 补 `input_ids` 形参 + `zigzag_reorder_moe_aux`（同时改 `worker/model_runner_v1.py:2497/4085` 等调用点传值），
      顺带在 router 侧加断言"zigzag 下 input_ids 必须是 gathered 序"；
  (b) 明确不做重排：改 docstring、并在 `fused_topk_router.py:158-175` 的 hash 分支加
      `assert not zigzag_active()`（或 debug 一行），让"hash routing + zigzag"变成显式不支持而不是静默错。
* 本地能否判定：**能**（docstring 与代码矛盾、目标模型是否 hash routing 都可静态判定）。

### C-M2 C8「快速写回」（LI C8 reshape-optim）在 zigzag 下不再关闭

* 级别：**中**（本次 harness 配置不命中；PD prefill + LI C8 部署命中）
* 文件:行号
  * 老实现有：`O: vllm_ascend/attention/sfa_v1.py:2567`
    `use_li_c8_reshape_optim = self._use_li_c8_reshape_optim() and not zigzag_active`
    （注释：reshape-optim 的 block writer 假设自然 token 序，zigzag 下退回 scatter）。
  * 新树现状：`vllm_ascend/attention/indexer.py:247` `use_reshape_optim = self._use_c8_reshape_optim()`；
    `:284-286 _use_c8_reshape_optim()` 只判 `enable_sparse_li_c8 and get_ascend_config().c8_reshape_optim_enabled`，
    **没有 zigzag 排除**；`write_cache` 走 `:249-258` 的 `torch.ops._C_ascend.store_kv_block(...)`。
    配套的 group 元数据在 `:1171-1179` 用**当前 metadata 的 slot_mapping**
    （zigzag 下已是 `plan.slot_mapping_cp_gathered`，含 `-1` padding 行）调 `store_kv_block_metadata`。
* 命中条件：`ascend_config.py:765-778` `_c8_reshape_optim_enabled = enable_sparse_li_c8 and is_prefill_node`，
  且 `is_prefill_node = kv_transfer_config is not None and kv_role == "kv_producer"`（`ascend_config.py:771-777`）。
  目标 `cp_balance/configs/_common.json` 无 kv-transfer → **本次跑不到**；但同一 feature 明确支持 kv_producer
  （`dsa_cp_with_o_proj_tp_for_config` 就是为 kv_producer 返回 True）→ PD prefill 部署会命中：
  块写算子按块假设自然序 + `-1` slot 行 → indexer cache 写错位/写坏。
* 建议：把 `_use_c8_reshape_optim()`（或 `:247` 调用点）加上 `and not zigzag_active()`，与 O 一致；
  并给 `store_kv_block_metadata` 的输入加断言（zigzag 下必须整段为 gathered 映射且不得把 `-1` 行交给块写），
  与 stage A H4（-1 假设不统一）一起在远端做一次定向 A/B。
* 本地能否判定：**能**（守卫缺失 + 命中条件都可静态判定）。

### C-M3 宽 `except` 会把「REDUCE_MODE 写错」吞成静默回退，B!=C 的排查会变得非常困难

* 级别：**中**（可维护性/可诊断性；一旦发生会把归约问题伪装成精度问题）
* 文件:行号
  * `ops/linear_op.py:196-203`（`try: output = fixed_order_reduce_scatter(...) except Exception: output = None` → 走 `:203` native）。
  * `ops/fused_moe/shared_experts.py:250-257`（`except Exception as exc` → warning → `:258` native）。
  * 抛点：`distributed/utils.py:144-147`（未知 mode 抛 `ValueError`）、`:125-127`（rows 不整除抛 `ValueError`）。
* 现象：`VLLM_ASCEND_CP_BALANCE_REDUCE_MODE=allreducee`（拼写错误）只产生一条 warning（shared_experts）
  或完全静默（linear_op），然后退回 owner-dependent 的 reduce_scatter → 结果是"配置看起来生效、
  B/C 一致性却对不上"。同理，将来 rows 不整除（前置条件不满足）也会静默走 native，而 C-H1 的站点
  又没人打点，定位成本极高。
* 建议
  1. 启动期 fail fast：在 `reduce_mode()`（`distributed/utils.py:24-27`）里校验取值，非法直接 `raise`；
     `fixed_order_reduce_scatter` 里保留兜底但把 mode 判定挪到函数最前面。
  2. 调用点的 `except` 只兜"shape/通信能力不支持"这一类可预期异常（`RuntimeError/NotImplementedError`），
     并把实际走的分支写进 debug 日志（补 `site=shared_experts`，见 C-L4）。
  3. 顺带把 `linear_op.py:201-202` 的 `logger.info_once(... path=native site=mlp)` 扩成
     "zigzag_active + mode + rows + exception 类型"的信息，让一次跑就能确认归约路径。
* 本地能否判定：**能**。

---

### C-M4 `_allreduce_slice_reduce_scatter` 就地 all_reduce + 视图返回：改写调用方缓冲、返回值延长整段 [T,H] storage 的寿命

* 级别：**中**（当前两个调用点都"安全"，但所有权契约与显存峰值都有隐患）
* 文件:行号
  * 实现：`vllm_ascend/distributed/utils.py:30-52`（`summed = tensor if tensor.is_contiguous() else tensor.contiguous()`
    → `dist.all_reduce(summed, group=group.device_group)` → `return summed[rank*chunk:(rank+1)*chunk].contiguous()`）。
  * 调用点：`ops/linear_op.py:197`、`ops/fused_moe/shared_experts.py:251`。
* 现象
  1. 入参张量被**就地**改写（`dist.all_reduce` in-place）。当前两处调用点之后都不再读入参
     （`output_parallel` 是 `quant_method.apply` 的新张量、`shared_out` 是 `part2`/`F.pad` 的新张量），
     所以现在没有可观测错误，但"改写入参"这件事没有写在函数签名/调用点注释里。
  2. 入参连续时，切出来的子块本身也连续 → `.contiguous()` 不会复制 → 返回值是**入参缓冲的视图**。
     于是返回值的 storage 是整段 `[T, H]`：TP16 下本地 shard 只有 `T/16` 行，但 storage 是 `T` 行
     （T=16384、H=7168、bf16 约 235MB vs 14.7MB），且其生命周期被延长到下游消费完返回值为止
     （原 `reduce_scatter_tensor` 会在返回时释放入参）。
  3. 若将来到 `torch.compile` / custom-op 边界把这个 helper 包成 `mutates_args=[]` 的算子，
     就地改写输入是契约违规（同 `docs/cp_balance_review_round2.md` 的 A1）。
* 建议：二选一 —— (i) 返回前 `.clone()`（代价只有一个 shard 的拷贝，换来干净的所有权契约）；
  (ii) 保留就地，但在 `_allreduce_slice_reduce_scatter` 的 docstring 与两个调用点写明
  "入参会被改写、返回值是入参的视图"，并在任何 custom-op 封装处显式声明 `mutates_args=["x"]`。
* 本地能否判定：**能**（torch 语义 + 代码路径：连续张量的行切片仍是连续视图，`.contiguous()` 返回自身）。

---

## 4. 低

### C-L1 站点 3/4（跨模型 `sp_reduce_scatter` / `dsa_v1` O-proj）没有任何门控或断言

见 C-H1 表。建议在 `models/common/ops/sequence_parallel.py:29-38` 与 `attention/dsa_v1.py:1700-1703`
加 `assert not zigzag_active()`（或 TODO），否则 feature 一旦被复用到 GLM-5-Next / DSv4 / Kimi-K3，
会静默失去 owner-independent 保证。本地可判定。

### C-L2 `forward_zigzag_local` 的 `tp_size == 1` 兜底返回**全量行**，与方法契约/调用方假设不一致

* `vllm_ascend/ops/vocab_parallel_embedding.py:220-223`：`if self.tp_size == 1: return self._forward_origin(input_)`。
  `_forward_origin` 返回的是**全量 padded 自然序行**，而 `forward_zigzag_local` 契约（docstring `:196-219`）
  和调用方 `patch_deepseek_v2.py:320-355`（`hidden_is_zigzag_local = True` 之后**不再校验行数**）都按
  "rank-local `[prev,next]` 行"消费；真被走到就会静默错布局（cp_size>1 ⟹ tp>1，所以当前不可达，
  但这是"cp 与 tp 解耦"时最容易踩的坑）。
* 建议：兜底改成 `return zigzag_shard_tensor(self._forward_origin(input_))`，或 `zigzag_active()` 时直接 raise。
* 本地可判定：能。

### C-L3 envs 五个变量的默认值/解析/使用处一致性

* 与使用处**一致**的部分：`envs.py:95 VLLM_ASCEND_CP_BALANCE`（默认 `1`）只被
  `layers/cp_zigzag.py:563` 读（`check_b_path.py` 的静态门也过）；`:100 VLLM_ASCEND_CP_BALANCE_MIN_TOKENS`
  （默认 `8192`）读两处 `cp_zigzag.py:598-599`；`:106-108 ..._REDUCE_MODE`（`allreduce`，`strip().lower()`
  正确作用在返回值上）经 `distributed/utils.py:24-27 reduce_mode()`（`lru_cache(maxsize=1)`）读一次；
  `:112 ..._DEBUG`（`0`）用在 `distributed/utils.py:131`、`attention/context_parallel/sfa_cp.py:489/524/532`、
  `ops/linear_op.py:199`；`:117-119 ..._EMBED_LOCAL`（`0`）只被 `patch_deepseek_v2.py:320` 读。
* **不一致**：`MIN_TOKENS` 默认 8192 vs harness/场测 2048
  （`cp_balance/configs/_common.json:13`、`_common_a5.json:13`、`serve_config.py:132` 会显式导出 2048）。
  不显式设置时，2048~8191 token 的 batch 会**静默**留在连续路径，`[CP_BALANCE][branch]` 只会给出
  `reason=actual<min(8192)`（`cp_zigzag.py:598-599`）——排查时容易被当成"没进 zigzag 的原因"。
  另外 `_REDUCE_MODE` 的 `lru_cache`（进程内只读一次）与 harness"每个 config 起一个新 server"不冲突，
  但热改环境变量无效，值得在 README/envs 注释里写明。
* 建议：把默认值对齐到 2048（或把 8192 的理由写进 envs 注释 + README），并在启动时 `info_once`
  打印 5 个变量的实际取值（现在只有一个 `[CP_BALANCE][plan]` 打点）。本地可判定。

### C-L4 debug 打点缺 `site=shared_experts`（老实现有 `site=pad_and_reduce`）

* `ops/fused_moe/shared_experts.py:258` 的 native 回退没有 debug 行；新树只有 `ops/linear_op.py:201-202`
  的 `site=mlp`。O 有 `register_custom_ops.py:122-123 path=native site=pad_and_reduce`。
* 建议：补 `logger.info_once("[CP_BALANCE][reduce] path=native site=shared_experts")`，
  并在 C-H1 站点 1 落实后补 `site=pad_and_reduce`。本地可判定。

---

## 5. 必答问题速查

### 5.1 门控覆盖 / 本配置（TP16、fine-grained TP 未开）会不会命中

* 结论有三层，必须分开说：
  1. **门控站点的可达性**：`MLPRowParallelOp`（`linear_op.py:197`）在目标配置里**不会构造**
     （`_get_row_parallel_op` 要求 `"down_proj" in prefix and mlp_tp_enable()`，`linear_op.py:326-334`；
     fine-grained TP 未开 → False），所以该门控是死代码；
     `shared_experts._pad_and_reduce_scatter`（`:246`）只在 `SEQUENCE_PARALLEL_ONLY` 下调用，
     而 `moe_config.is_sequence_parallel` 在 DP=1 时为 False（C-H2）→ 也是死代码。
     即"两个门控都覆盖了不该覆盖的站点"。
  2. **本配置下的其他 reduce 站点**：见 C-H1 表。A3 档（`FUSED_OR_CAPACITY`）大 batch 走 ALLTOALL、
     小 batch 走 MC2，都不会经过 `maybe_pad_and_reduce`；目标模型也不走 `sp_reduce_scatter`；
     `oproj/fine-grained` 未开；DP/PCP 为 1 → **在本配置里看起来"没有任何带 owner 依赖的 row-parallel
     reduce_scatter 会跑"**（这也是为什么现在的 B==C 对比可能看不出问题）。
  3. **换档位/换模型就会命中**：A5/A2 档（MoE comm type = ALLGATHER + `is_sequence_parallel`）命中 C-H1 站点 1；
     开 fine-grained O-proj TP 命中站点 2/4；跑 GLM-5-Next/DSv4/Kimi-K3 命中站点 3；
     PCP>1 命中站点 5。**这些都没有门控**，所以这轮"没命中"不能作为"覆盖完整"的证据。
* 本地无法给出运行时 MoE comm type；请按 C-H1 建议的 debug 日志在远端取一次证。

### 5.2 `fixed_order_reduce_scatter` 在 shared_experts 末尾 pad 过的张量上：前置条件成立吗？pad 行会污染吗？

* **行数整除**：成立。`shared_experts.py:241-245` 先 `pad_size = (tp - T % tp) % tp` 补齐再调归约，
  所以 `rows % world_size == 0`（`distributed/utils.py:125-127` 不会触发）。
* **"按 rank 切块 == 本 rank 本地行"**：成立，但理由与"连续切片"无关。`shared_out` 是
  `_gather_sp_input`（`:227-230`，`tensor_model_parallel_all_gather(hidden_states, dim=0)` 即 rank 序拼接）
  得到的 full 张量，每个 rank 的行序相同、且第 r 块恰好是 rank r 的本地行（连续布局与 zigzag 布局都成立，
  因为 zigzag 的本地行序就是 rank 内部 `[prev,next]` 序）。因此 `all_reduce + 切第 r 块` 与
  `reduce_scatter_tensor` 交付的是**同一批行**，差别只在求和顺序——这正是 owner-independent 的目标。
* **pad 行污染**：不污染，而且实际上不会出现 pad 行：
  (a) zigzag 资格门保证 `num_tokens_pad % cp_size == 0`（`cp_zigzag.py:600-605`），本 feature 里 cp == tp，
  所以到达这里时 `T % tp == 0` → `pad_size == 0`；
  (b) 即便出现，`F.pad` 补的是 0，bf16 加 `+0.0` 精确不改变数值（NaN/-0 边界除外）；
  (c) 需要提示的是**行数**而不是数值：pad 后每个 rank 拿 `ceil(T/tp)` 行（`:232-239` 的 docstring 已说明），
  只有最后一名的 chunk 含 pad 行，依赖调用方截断——这是 T main 原有行为，本次移植未改变。
* **真正需要注意的是返回值语义**（见 C-M4）：`_allreduce_slice_reduce_scatter` 是**就地** all_reduce +
  **视图**返回，会改写调用方缓冲并延长整段 `[T,H]` storage 的寿命。

### 5.3 vocab_parallel_embedding：拆 `_embed_partial` 后各路径是否还正确？`forward_zigzag_local` 的"全量行 all_reduce 再切片"对不对？

* 拆分的**意图**与各路径语义都对，唯一问题是把 return 丢了（C-B1）。逐路径核对：
  * `tp_size == 1`（含 `disable_tp=True` → `ReplicatedGroup`，`:81-89`）：`_forward_origin:308-309`
    `if self.tp_size <= 1: return output_parallel` —— 与老实现 `else: return output_parallel` 等价 ✓
    （`_embed_partial` 里 `tp_size > 1` 才做 mask/`masked_input=input_`，也与老实现一致 ✓）。
  * 常规 TP（`get_tp_group()`）：`:310-320` 保留 `get_tp_group().world_size == 1` 的短路和
    `torch.ops.vllm.all_reduce(...)` ✓。**这里是移植相对 O 的语义升级**：O 的 `_forward_origin` 是
    `maybe_pad_and_reduce`（reduce_scatter，owner-dependent、且返回 rank-local 行），
    T main 的模型边界要求 embedding 返回**全量行**（`:314-319` 的注释），所以 all_reduce 是正确选择，
    也解释了"移植不需要给 `register_custom_ops` 的 embedding 侧加门控"。
  * `embed_tp`（fine-grained embedding TP）：`forward:190-192` 走 `_forward_embed_tp`，
    完全不经过 `_embed_partial`/`_forward_origin` ✓；`forward_zigzag_local:206-210` 显式
    `NotImplementedError` ✓（该路径自带 all_gather + reduce_scatter，语义上确实不能与 zigzag 组合）。
  * lmhead：`AscendParallelLMHead` 继承同一类，但 vLLM `ParallelLMHead.forward` 直接
    `raise RuntimeError("LMHead's weights should be used in the sampler.")`
    （`D:/code/cp_balance/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:627-629`），
    logits 走 `AscendLogitsProcessor`/`lmhead_all_to_all` → 不经过 `_forward_origin` ✓。
* `forward_zigzag_local`（`:195-226`）语义**正确**：词表权重按 TP 切分，只有每个 rank 用**同一批 token ids**
  做 lookup 再 all_reduce，才能得到完整 embedding；然后再 `zigzag_shard_tensor` 取本 rank 的
  `[prev,next]` 行。调用方 `patch_deepseek_v2.py:320-340` 先断言 `input_ids` 是 full padded 流
  （否则 raise），并在 `hidden_is_zigzag_local=True` 时跳过模型边界的 all_gather+shard ✓。
  需要注意的是：(a) C-B1 缺 return；(b) C-L2 的 `tp_size==1` 兜底返回全量行；
  (c) 该路径把"模型边界 all_gather(hidden) + shard"换成"embedding 内 all_reduce(full hidden) + shard"，
  通信量与语义不同（注释已写明，默认 `EMBED_LOCAL=0` 关闭）——本次不判为缺陷，只作记录。
* 另外：`forward_zigzag_local` 里的 `tensor_model_parallel_all_reduce`（vllm 的 TP 组）与
  `self.comm_group` 在常规 TP 下同一组 ✓；`tp_size == 1` 的 `ReplicatedGroup` 路径不会走到它，
  所以没有"用错组"的问题。

### 5.4 envs 五个变量的默认值/解析与使用处

见 C-L3（结论：解析/使用处一致；`MIN_TOKENS` 默认 8192 与 harness 2048 不一致，是唯一的实质差异；
`REDUCE_MODE` 的 `strip().lower()` 正确、被 `lru_cache` 固化在进程内；`CP_BALANCE` 默认 1 = 升级即开，
与老实现相同）。

### 5.5 老实现为「B==base 逐位一致」加的补丁，在新实现里是否完整

| 老实现的补丁 | O 的位置 | 新树现状 | 判定 |
| --- | --- | --- | --- |
| reduce 门控（MLP） | `O: ops/linear_op.py:199-215` | `ops/linear_op.py:196-203` ✓ | 已搬；但目标配置不可达（C-H2/C-H1） |
| reduce 门控（MoE finalize / pad_and_reduce） | `O: ops/register_custom_ops.py:25-48,115-124` | 无（`ops/register_custom_ops.py:98` 未门控） | **没搬** → C-H1 站点 1 |
| reduce 门控（Sequence row-parallel，含 mmrs 融合） | `O: ops/linear_op.py:351-384,459-467` | T main 无 `SequenceRowParallelOp`；T 的 `OProjRowParallelOp`（`:278`）未门控 | 结构性不需要 + 新的未门控站点 |
| q_up 融合算子 byte-identical | `O: attention/sfa_v1.py:1780-1803`（改用 `npu_transpose_batchmatmul`） | T main 自带（`attention/sfa_v1.py:1147-1170`，perm 与 base 相同；另有 `q_nope.shape[0] < 65536` 兜底） | 无需搬；zigzag 下 token 维变小，不会新触发兜底 |
| C8 快速写回关闭 | `O: attention/sfa_v1.py:2567`（`and not zigzag_active`） | `attention/indexer.py:247/284-286` 无此守卫 | **没搬** → C-M2 |
| `input_ids` 重排 | `O: ascend_forward_context.py:197,333-342` | 没有（`:195-212` 无 `input_ids` 形参），docstring 仍声称有 | **没搬** → C-M1 |
| 出口 gather 行数与 base 对齐 | `O: 23ff2c23c` 的 `zigzag_gather_tensor` trim `num_tokens` | 移植改为模型边界 gather + `zigzag_gather_tensor(x, full_num_tokens)`（`patch_deepseek_v2.py:425-437`、`cp_zigzag.py:630-655`），行数由 `positions.shape[0]` 驱动，与同函数里的非 zigzag 分支一致 | 等价，无需搬 |
| V2 runner / draft 排除 | `O: ascend_forward_context.py:306-310`、`spec_decode/llm_base_proposer.py`、`sfa_v1.py` 的 `for_draft` | `ascend_forward_context.py:266-282`（draft/V2）、`spec_decode/llm_base_proposer.py:2525-2536`（`for_draft=True`）、`zigzag_cp.py:127-160`（`speculative`） | 已搬 ✓（HEAD 已含 `8912139f9` 的 indexer draft 对齐） |
| 三种 reduce 模式 + helpers | `O: distributed/utils.py:24-150` | `distributed/utils.py:24-148`（逐字，另 `fixed_order_rank_sum` 在 `layers/cp_zigzag.py:51-75`） | 已搬 ✓（仅把 inline `os.getenv` 换成 `envs` + `lru_cache`） |
| `_sfa_5_3_scope` profiler 打点 | `O: device_op.py` 8 处、`sfa_v1.py`/`dsa_v1.py` 等 | 新树 `grep -c "SFA-5.3"` = 0 | 未搬（判定为 base 侧附赠；但 `docs/profiling_guide.md` 若按这些 scope 名解析会失配，需同步改文档） |
| GLM-5.2 逐层权重过滤 + UT | `O: patch/worker/patch_deepseek_mtp.py:1-74`、`tests/ut/patch/worker/test_patch_deepseek_mtp.py` | 未搬（T 的对应文件是 `models/deepseek_mtp.py`） | 未搬（与 zigzag 无关；若要继续跑"3 层 checkpoint"调试需重新实现） |
| `patch/__init__.py` 文档条目 | `O: patch/__init__.py` +15 | 未改 | 未搬（纯文档；新 patch 模块若不登记会丢索引性） |

### 5.6 顺带回答：`device_op.py` 的 `block_table` 参数（本次范围第 3 项）

* 两处改动（`device/device_op.py:348-351` Base、`:1321-1324` A5）**一致**，`if block_table is None:
  block_table = attn_metadata.block_table` 保持了原行为 ✓；310P 没有 override（`:1480+` 不定义该方法），
  继承 Base 实现 ✓。
* 但**没有任何调用点传这个新形参**：`attention/indexer.py:464-477` 的
  `DeviceOperator.indexer_select_post_process(...)` 没传 `block_table=`。所以它当前是**死参数**；
  zigzag 的 block table 生效靠的是 indexer metadata 自身的 `block_table`（`_build_dsa_cp_zigzag_metadata`
  `:841-844` 返回 `plan.block_table_zigzag`，`attn_metadata.block_table` 即它）。
  SFA 主路径则通过 `sfa_cp.py:669-686` 的 `_execute_sparse_flash_attention_process` override
  把 `block_table=context.block_table_zigzag` 显式传下去 ✓（那条形参在 T main 本来就有）。
* 建议：要么删掉这个死形参，要么在 `indexer.py:464` 显式 `block_table=indexer_metadata.block_table`，
  别让"参数没接上"在下一次改布局时变成默认路径的坑。级别：低。

---

## 6. 本地验证方法与复现命令（全部只读、无 torch）

```bash
cd D:/code/cp_balance/vllm-ascend

# C-B1：_embed_partial 无 return
grep -n "return output_parallel" vllm_ascend/ops/vocab_parallel_embedding.py     # 只有 :309/:313
python -c "
import ast;s=open('vllm_ascend/ops/vocab_parallel_embedding.py',encoding='utf-8').read();t=ast.parse(s)
for c in ast.walk(t):
  if isinstance(c,ast.ClassDef) and c.name=='AscendVocabParallelEmbedding':
    for f in c.body:
      if isinstance(f,ast.FunctionDef): print(f.name,[n.lineno for n in ast.walk(f) if isinstance(n,ast.Return)])
"

# C-B2：跨模块 import 名解析（本次就是靠它发现的；需把 vllm_ascend.envs 与 DeviceOperator(AnnAssign) 排除）
python - <<'EOF'
import ast, pathlib
targets=['vllm_ascend/attention/context_parallel/sfa_cp.py','vllm_ascend/attention/indexer.py']
def defs(mod):
    p=pathlib.Path(mod.replace('.','/')+'.py')
    if not p.is_file(): return None
    t=ast.parse(p.read_text(encoding='utf-8')); out=set()
    for n in t.body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)): out.add(n.name)
        elif isinstance(n,ast.Assign):
            out|={tg.id for tg in n.targets if isinstance(tg,ast.Name)}
        elif isinstance(n,ast.ImportFrom): out|={a.asname or a.name for a in n.names}
        elif isinstance(n,ast.Import): out|={(a.asname or a.name).split('.')[0] for a in n.names}
        elif isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name): out.add(n.target.id)
    return out
for f in targets:
    for n in ast.walk(ast.parse(pathlib.Path(f).read_text(encoding='utf-8'))):
        if isinstance(n,ast.ImportFrom) and (n.module or '').startswith('vllm_ascend'):
            d=defs(n.module)
            if d is None: continue
            for a in n.names:
                if a.name!='*' and a.name not in d: print('MISSING',f,n.lineno,f'{n.module}.{a.name}')
EOF

# C-H1：全部 token 行 reduce/all_reduce 站点
grep -rn "reduce_scatter\|all_reduce" --include=*.py vllm_ascend/ | grep -v "def \|#"

# C-H2：门控可达性
grep -n "mlp_tp_enable\|oproj_tp_enable" vllm_ascend/ops/linear_op.py
grep -n "is_sequence_parallel" vllm_ascend/ops/fused_moe/shared_experts.py vllm_ascend/ops/fused_moe/prepare_finalize.py
grep -n "use_sequence_parallel_moe" D:/code/cp_balance/vllm/vllm/config/parallel.py D:/code/cp_balance/vllm/vllm/model_executor/models/deepseek_v2.py
```

---

## 7. 建议的收敛顺序

1. **先修 C-B2**（补 utils helper + 改 import 源）——否则 SFA/DSA-CP 模块 import 不进来，后面任何验证都跑不动。
2. **再修 C-B1**（补 `return output_parallel`）并在本地跑 `tests/ut/ops/test_vocab_parallel_embedding.py`，
   把这两条加进静态门禁（ast return 检查 + 跨模块 import 名检查）。
3. 用远端一次长 prefill + `VLLM_ASCEND_CP_BALANCE_DEBUG=1` + MoE comm method 日志，判定 C-H1/C-H2
   的"命中面"，再决定门控挂在 (a) `_maybe_pad_and_reduce_impl` 还是 (b) shared experts（或两者都补）。
   同时确认远端 vLLM 是否含 `data_parallel_size > 1` 条件（C-H2 引用的行号与本地 pin 不一致）。
4. 其余中低危按 C-M1 → C-M2 → C-M3 → C-M4 → C-L1..L4 顺序处理，其中 C-M2/C-L3 只需一行判断/一个默认值。
