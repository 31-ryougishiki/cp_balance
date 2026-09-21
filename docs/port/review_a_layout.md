> 历史文档（2026-09-18 移植期取证/评审）：结论可能已过时；dp>1 那道门与 zigzag 的现状见 `docs/scripts_review.md` §4.1/§4.1c/§8。
>
> 已变事实（2026-09-21 更正）：
> - §1 B1「DSA-CP 生效 ⟹ dp>1 ⟹ gate 必然 dp>1」已裁定为上游门写窄：dp=1 + DSA-CP 成立，两个树都改了门（`scripts_review.md` §4.1），zigzag 可达性不再是阻塞。
> - 同段「仓库自带 e2e 印证 DSA-CP 只在 dp>1 跑」不能反推：上游文档 `context_parallel.md` 的 DSA-CP 示例就是 `--tensor-parallel-size <N>`（dp=1），与代码门矛盾。
> - H2（gate 不看 PCP、indexer 在 PCP 下跳过 zigzag）仍未修，只在 indexer 侧加了 `reason=<门> site=indexer` 日志做可见性缓解（跟踪于 §4.3b）。

# Stage A 评审：布局与元数据阶段（cp_balance → vllm-ascend main）

## 0. 快照、范围与方法

* 被评审对象：`cp_balance` 分支移植提交 **612d03a8b**（任务指定）。
* **评审期间工作树在动**，最终快照 = **HEAD = `3572231a0`**（fix(cp_balance): three blockers found by the
  stage-A review）。其前序：`612d03a8b`（移植）→ `8912139f9`（indexer gate 补 draft_index）→
  `d9dfc4497`（planner 用 host seq_lens）→ `3572231a0`（本轮评审反馈的 B2/B3/B4 修复）。
  本文所有行号**按 `3572231a0` 工作树**核对过；B2/B3/B4 保留"发现 + 修复验证 + 残余"三段式记录。
* 覆盖文件（逐函数读完）：`layers/cp_zigzag.py`、`attention/context_parallel/zigzag_cp.py`、
  `attention/context_parallel/sfa_cp.py`（DSACPContext / `__init__` buffer / `_prepare_parallel_metadata` /
  `_prepare_zigzag_layout` / `_update_parallel_slot_mapping`）、`attention/indexer.py`
  （`_build_dsa_cp_zigzag_metadata` + `_build` 调用点）、`attention/utils.py:is_prefilling_cpu` +
  `worker/model_runner_v1.py` 传值点。交叉读了 `ascend_forward_context.py`、`patch/worker/patch_deepseek_v2.py`、
  `attention/sfa_v1.py`、`device/device_op.py`、`distributed/utils.py`、`ascend_config.py`、
  `vllm/config/parallel.py`（本地 main 检出 `D:/code/cp_balance/vllm`）。
* 方法：只读 `git show/git diff/grep/sed/awk` + `python -c`（ast 静态解析），未运行任何 torch/NPU 代码。

**当前状态一览**：`1` 条未修 BLOCKER（B1，特性在 main 上不可达）；`3` 条已在 `3572231a0` 修复
（B2/B3/B4，见 §2 的修复验证）；`4` 条高危（H1-H4）；`5` 条中（M1-M5）；`4` 条低/仅记录（L1-L4）。

---

## 1. 未修 BLOCKER

### B1 zigzag 在主线 DSA-CP 配置下永不可达（资格门与 DSA-CP 的前置条件自相矛盾）

* 级别：**BLOCKER（未修）**——特性完全无法进入，所有下游布局/kernel 问题都被它掩盖
* 文件:行号
  * `vllm_ascend/layers/cp_zigzag.py:571-572`（`if dp_size > 1: return "dp>1"`，gate 第 3 道门），
    同段还有 `:573-574`（dcp_replicated）、`:592`（query_len<2cp）、`:604`（pad%cp_size）等
  * 入参：`vllm_ascend/attention/context_parallel/sfa_cp.py:474-487`（`dp_size=` 在 `:484`）、
    `vllm_ascend/attention/indexer.py:789-802`（`dp_size=` 在 `:799`），
    两处都取 `self.vllm_config.parallel_config.data_parallel_size`
  * 主线前置条件：`vllm_ascend/ascend_config.py:660-664`
    （`self.enable_dsa_cp = self.enable_dsa_cp and has_indexer and vc.parallel_config.use_sequence_parallel_moe`）
  * `D:/code/cp_balance/vllm/vllm/config/parallel.py:711-726`：`use_sequence_parallel_moe` 要求
    `data_parallel_size > 1`（该条件由 commit `c4dd6d78fd Restore data_parallel_size > 1 for
    use_sequence_parallel_moe` 恢复）
  * 第二道同源闸门：`vllm_ascend/ascend_forward_context.py:321-326`
    （`if dp_world_size > 1 and zigzag_cp_active:` → `_disable_zigzag_metadata_for_fallback` + 关掉 active）
* 现象：**DSA-CP 生效 ⟹ `data_parallel_size > 1` ⟹ gate 第 3 道门必然返回 `"dp>1"`**。
  也就是说 `_prepare_zigzag_layout`（`sfa_cp.py:448-550`）与 `_build_dsa_cp_zigzag_metadata`
  （`indexer.py:762-845`）永远不会真正建出 plan；即便只放开 builder 的 gate，
  `ascend_forward_context.py:321-326` 也会在 forward context 里把 zigzag 关掉并把 metadata 退回连续切片。
* 为什么错：这条 gate 是从老实现逐字照搬的（老线 `23ff2c23c:vllm_ascend/layers/cp_zigzag.py:542-618`），
  老线的判定只看 FlashComm1：`23ff2c23c:vllm_ascend/utils.py:1372-1391` 的 `enable_dsa_cp_for_config`
  只要求 `enable_sp()`；部署是 tp=16 + dp=pp=1（`docs/base_dsa_cp_flow.md:9`），所以老线上 dp 门可过。
  主线把 DSA-CP 的前置条件换成 `use_sequence_parallel_moe` 之后，这条门变成"死门"。
  仓库自带 e2e 也印证 main 的 DSA-CP 只在 dp>1 下跑：
  `tests/e2e/pull_request/four_card/context_parallel/test_accuracy.py:130`（`data_parallel_size: 2`）、
  `tests/e2e/nightly/multi_node/internal_dp/config/GLM-5.1-W8A8C8-A3_128k_90_50.yaml`（`--data-parallel-size 8`）。
* 建议改法（先远端取证再动手）：
  1. 远端开 `VLLM_ASCEND_CP_BALANCE_DEBUG=1` 跑一次长 prefill，看 branch 日志
     （`sfa_cp.py:488-495`，`"[CP_BALANCE][branch] ... reason=%s"`）是否只有 `reason=dp>1`；
     同时打印 `enable_dsa_cp()`、`parallel_config.data_parallel_size`、`use_sequence_parallel_moe`、vllm commit。
  2. 若确认：要么按 main 的 DP/SP-MoE 语义重做资格判定 + 布局（builder gate 与 forward-context 那道必须同改），
     要么先把这个矛盾（"老实现要求 dp==1" vs "main 的 DSA-CP 要求 dp>1"）上报/换部署形态，
     不要按当前 gate 去调 kernel 与打点。
* 本地能否判定：**代码层可判定**（同一 `vllm_config` 对象的同一字段，`use_sequence_parallel_moe` 是 pure property）；
  远端 vllm 是否为含 `c4dd6d78fd` 的版本需确认（若远端 vllm 的 `use_sequence_parallel_moe` 不含
  `data_parallel_size > 1`，本条降级为"需远端确认"）。

---

## 2. 已修复的 BLOCKER（commit `3572231a0`，保留验证记录）

### B2 `ZigzagCPPlan` 缺 `slot_mapping_cp_gathered` 字段 → TypeError，且不会被降级回退

* 级别：**BLOCKER（已修复 `3572231a0`）**
* 发现位置（修复前）：构造实参在 `zigzag_cp.py`（当时 `:228`）传 `slot_mapping_cp_gathered=`，
  但 dataclass 字段表没有它；消费端 `sfa_cp.py:440`（当时）读 `plan.slot_mapping_cp_gathered`。
  两处 `except (ValueError, AssertionError, RuntimeError)`（`sfa_cp.py:516`、`indexer.py:821` 当时）
  捕不到 `TypeError` ⇒ 第一次建 plan 就打断 metadata build。
* 证据：`python -c` ast 静态解析（修复前）：`UNKNOWN kwargs: ['slot_mapping_cp_gathered']`。
* 修复验证（当前 `3572231a0`）：
  * `zigzag_cp.py:45-48` 已声明 `slot_mapping_cp_gathered: torch.Tensor`（构造 `:232`、消费 `sfa_cp.py:440`）；
  * 两处 `except Exception as exc  # noqa: BLE001`（`sfa_cp.py:516`、`indexer.py:821`）已放宽，plan 出错会降级连续切片。
* 残余/建议：`except Exception` 见 M5（可能造成两侧 builder 单侧回退）；另外
  `_prepare_native_hidden_states` 的 zigzag 早退（B4 修复）没有加 `shape[0] == local_tokens` 断言，建议补上。

### B3 indexer 的 zigzag slot_mapping 用错变体（局部 [prev,next] 序 + local_tokens 行）

* 级别：**BLOCKER（已修复 `3572231a0`）**
* 发现位置（修复前）：`indexer.py`（当时 `:844`）返回 `slot_mapping[index]`（`index = plan.zigzag_index`，
  局部 [prev,next] 序、行数 `plan.local_tokens`）。
* 为什么错：indexer 的 cache 写在 TP all-gather 之后 —— `indexer.py:362-379`（`all_gather_async(k_li, get_tp_group())`，
  rank 拼接语义见 `distributed/utils.py:150-159`），张量是 `[r0_prev, r0_next, r1_prev, ...]` 序、
  行数 `num_tokens_pad = cp_size * local_tokens`；`indexer.py:404` 把同一个 slot_mapping 交给
  `write_cache`（`:229/259/278` 的 `npu_scatter_nd_update_`；C8 reshape 优化走
  `store_kv_block_metadata`，`indexer.py:1171-1182`）。长度不匹配会报错，即使等长也是错序（静默精度损坏）。
  `indexer.py:343-348` 的既有注释就是需要的语义（"its padded slot mapping already covers the gathered
  layout"）；连续切片路径成立是因为本地行是自然序连续切片、rank 拼接后仍是自然序。
* 修复验证（当前）：`indexer.py:844` 已改为 `plan.slot_mapping_cp_gathered`（= `slot_mapping[zigzag_gather_index]`，
  `zigzag_cp.py:232`），行数与序都与 all-gather 结果一致；SFA 侧一直是同一变体（`sfa_cp.py:779`）。
* 残余/建议：无（indexer 的 C8 reshape 路径用的是 `metadata.slot_mapping.numel()` 生成的 group 元数据，
  修复后与 gathered 行数一致）。建议后续给 planner 返回"两个命名变体"，避免再次混用。

### B4（关联，布局阶段）`_prepare_native_hidden_states` 在 zigzag 下把 rank-local 行二次切片 → rank≥1 全零

* 级别：**BLOCKER（已修复 `3572231a0`）**
* 发现位置（修复前）：`sfa_cp.py` 的 `AscendSFADSACPImpl._prepare_native_hidden_states`
  （现在 `:612-633`，修复前 pad 后直接取 `[local_start, local_end_with_pad)`）。
* 为什么错：zigzag 模型入口已经把 hidden/positions 分片成 rank-local `[prev,next]` 行
  （`patch/worker/patch_deepseek_v2.py:327`、`:365`、`:379`，并断言 `shape[0] == positions.shape[0] // tp_size`），
  进入 attention 的只有 `local_tokens` 行；再 pad 到 `num_tokens_pad = tp*L` 行后取 `[r*L,(r+1)*L)`：
  rank 0 拿到真实行（巧合正确），rank r≥1 完全落在补零区 → 该 rank 的 q/kv 投影输入全零。
  同一份 hidden 也被 indexer k 路径复用（`sfa_v1.py:1707`、`:1711-1713`）。
* 修复验证（当前）：`sfa_cp.py:619-625` 增加 `if context.zigzag_index is not None: return hidden_states`；
  连续切片路径（`:625-633`）语义不变 ✓。建议补 `assert hidden_states.shape[0] == context.local_tokens`。

---

## 3. 高危

### H1 SFA 与 indexer 的 `num_tokens_pad` 基数可能不同（dspark adaptive verification）

* 级别：**高危**（两侧一个 zigzag、一个连续，或两个不同的局部布局）
* 文件:行号
  * SFA：`sfa_cp.py:344-346`（`global_tp_size = get_tp_group().world_size` /
    `num_tokens = common_attn_metadata.num_input_tokens`，即**未被覆盖**的原始值；
    覆盖发生在 `vllm_ascend/attention/sfa_v1.py:517-524` 的局部变量 `num_input_tokens`，**没有写回 metadata**）
  * indexer：`indexer.py:1084-1093`（dspark + `enable_adaptive_verification` 时局部覆盖为 `positions.shape[0]`，
    带 `TODO(lzt)`）+ `indexer.py:782-783`（`num_tokens_pad = _round_up(num_input_tokens, cp_size)` 用覆盖后的值）
* 现象：两侧的 `num_tokens_pad / cp_size / zigzag_index / local_tokens` 全部不同；
  `build_zigzag_cp_plan` 的硬校验（`zigzag_cp.py:193-194`、`cp_zigzag.py:440-449`）只保证
  `prev+next == local_tokens` 与 `num_actual_tokens == sum(query_lens)`，所以**两侧都可能成功建 plan**，
  得到两套不同的行↔token 映射（SFA 的 cos/slot/KV 与 indexer 的 cache 写不再对应）→ 静默错数。
* 建议改法：把 `num_tokens_pad` 的来源收敛成唯一入口（planner 入参由 `common_attn_metadata` 决定，
  两个 builder 调同一个 helper），或在 `_build` 里把 dspark 覆盖值写回 metadata（上游 TODO 的正解）。
* 本地能否判定：代码路径可判定；dspark(adaptive) 与 DSA-CP 是否可同时开启需远端确认（互斥则降为仅记录）。

### H2 资格门不看 PCP，但 indexer 在 PCP 下跳过 zigzag → 两侧布局不一致（PCP+DSA-CP）

* 级别：**高危**（组合可达时 SFA 与 indexer 使用不同 token 布局）
* 文件:行号
  * gate 调用处都没有 PCP 条件：`sfa_cp.py:474-487`、`indexer.py:789-802`（`zigzag_ineligible_reason`
    只检查 `cp_zigzag.py:573-574` 的 `dcp_replicated`）
  * indexer 明确跳过：`indexer.py:1125-1126`（`if self.use_dsa_cp and not self.use_pcp:`），
    PCP 分支用全量 slot_mapping 并重排：`indexer.py:1101-1105`（PCP 路径见 `indexer.py:686-690` 的
    `_build_pcp_ordered_slot_mapping`）
  * builder 选择给 DSA-CP 优先：`sfa_cp.py:1819-1834`（`dsa_cp_enabled` 先于 `pcp_enabled` 返回）
* 现象/为什么错：`resolve_sfa_metadata_builder` 让 "PCP>1 + DSA-CP" 走 DSA-CP builder（含 zigzag 判定），
  而 indexer 在 `use_pcp` 时走 PCP 布局 ⇒ 同一个 forward 里 indexer cache 与 SFA KV cache 的行↔slot 对应不同。
* 建议改法：gate 加 `pcp_size > 1 → "pcp"`（或显式声明 PCP 与 DSA-CP 互斥），两个调用点都传 `pcp_size`；
  同时确认 `get_tp_group().world_size`（SFA 用，`sfa_cp.py:344`）与 `parallel_config.tensor_parallel_size`
  （indexer 用，`indexer.py:534-541`）在 PCP/DCP 嵌套下恒等。
* 本地能否判定：代码层可判定；实际是否部署该组合需远端确认。

### H3 `dsa_cp_zigzag_seq_query/key` 容量上界差 1，且断言在 try 之外

* 级别：**高危**（触发即崩；`python -O` 下会静默截断成长度不足的元数据）
* 文件:行号
  * 容量：`sfa_cp.py:305-310`（`max_num_reqs = scheduler_config.max_num_seqs`；`:310`
    `dsa_cp_zigzag_seq_query = torch.zeros(2 * max_num_reqs + 1, ...)`；对照 `:306-307` 的非 zigzag buffer 是 `+1`）
  * 使用：`sfa_cp.py:415-419`（`num_zigzag_segs = 2 * num_reqs`、`assert capacity >= num_zigzag_segs`
    在 `_prepare_parallel_metadata` 内，**不在** `_prepare_zigzag_layout` 的 try/except 里）
  * `num_reqs` 可被撑到 `max_num_seqs + 1`：`vllm_ascend/worker/model_runner_v1.py:1043-1047`
    （mixed-batch 插 dummy request，`num_reqs_padded += 1`；进入条件是 `model_runner_v1.py:2421-2435`）
* 现象：`num_reqs == max_num_seqs + 1` 时 `2*num_reqs = 2*max_num_seqs + 2 > 2*max_num_seqs + 1` →
  assert 直接抛穿（forward 挂）；若 assert 被 `-O` 剥离，`buffer[:2B]` 静默截断 → kernel 看到少于 2B 个长度项。
* 建议改法：容量改成 `2 * (max_num_reqs + 1)`；把 assert 改为 `if capacity < num_zigzag_segs: 回退连续切片 + warning`。
* 本地能否判定：容量算术可判定；触发概率（prefill 批次能否出现 dummy request 撑到 max_num_seqs+1，
  eager + MLA 下 `_pad_query_start_loc_for_fia` 可能不可达）需远端确认。

### H4 zigzag 的全局 KV 写把"全部 padding 行 + slot=-1"交给写算子（-1 假设不统一）

* 级别：**高危/需远端确认**（若算子不跳过 -1 → 污染真实 slot；连续路径从不这样用）
* 文件:行号
  * zigzag：`sfa_cp.py:770-780`（`zigzag_gathered` 判定 + `scatter_slots = context.slot_mapping_cp_gathered`，
    行数 = `num_tokens_pad`；`kv_to_write = fused_kv_no_split` 全部行）；
    -1 来源：`sfa_cp.py:357-361`（slot pad `value=-1`）与 `zigzag_cp.py:232`（gathered 重排）
  * 连续：`sfa_cp.py:781-783`（`slot_mapping_sfa[:num_actual_tokens]` + `kv_to_write[:num_actual_tokens]`，
    天然不含 padding 行；`_get_sfa_kv_slot_mapping` 返回全 padded 自然序表：`sfa_v1.py:1517-1521`）
  * 写算子：`vllm_ascend/device/device_op.py:63-70`（`npu_scatter_pa_kv_cache`）、
    `device_op.py:52-60`（NHSD 变体）、`sfa_cp.py:785-789`（C8 分支 `npu_scatter_nd_update_`）；
    indexer 侧同族：`indexer.py:259/278`
* 现象：zigzag 的全局写必须一次写"rank 拼接序 + 含所有 rank padding 行"的 `num_tokens_pad` 行，
  其中 padding 行 slot 为 -1；连续路径的全局写永远截断到 `num_actual_tokens`，从不把 -1 交给这些算子。
  （局部写 `exec_kv` → `npu_kv_rmsnorm_rope_cache`，两条路径都可能带 -1，所以 -1 语义本身大概率被支持，
  但"全局 scatter 全量 -1 行"是 zigzag 新增依赖，padding 行数也从"个别 rank 的尾部"变成"所有 rank"。）
* 建议改法：zigzag 全局写前用 `mask = slot >= 0` 压缩两个张量再 scatter，与连续路径假设完全一致；
  或至少在 `VLLM_ASCEND_CP_BALANCE_DEBUG` 下断言 `(scatter_slots < 0).sum() == num_tokens_pad - num_actual_tokens`。
* 本地能否判定：**不能**（需 CANN 算子行为）——远端最小用例：构造含 -1 的 slot_mapping 调
  `reshape_and_cache` / `npu_scatter_nd_update_`，观察是否跳过。

---

## 4. 中

### M1 元数据热路径上的 device 分配/同步

* 级别：**中**
* 文件:行号
  * `indexer.py:832-836`：`in_range = index < cos.shape[0]` + **`if bool(in_range.any()):`** ——
    `.any()` 把 device 张量转 Python bool，**每次 build 触发一次 device→host 同步**；
    `local_cos/local_sin` 每次新建，`local_cos[in_range] = cos[index[in_range]]` 是布尔掩码高级索引。
  * `zigzag_cp.py:204-205`（`torch.tensor(list(values), int64, device)` × 3 次：zigzag/gather/inv_gather）、
    `:229-230`（两个 int32 长度张量）、`:219-220`（`torch.tensor(real_req_indices)` + `index_select`）、
    `:231`（`torch.cat` 复制 block table）⇒ 每个合格 prefill 批次约 10 个新 device 张量 + H2D 拷贝。
  * `sfa_cp.py:417-418`：把 plan 的两个长度张量再 `copy_` 进共享 buffer（另 2 次 D2D）。
  * 其他：`zigzag_cp.py:106/119` 的 `.tolist()` 读的是 host 张量（`query_start_loc_cpu`、
    `is_prefilling_cpu`）✓ 不同步；`sfa_cp.py:532-542` 的 `.tolist()` 有 `VLLM_ASCEND_CP_BALANCE_DEBUG` 保护 ✓。
* 建议改法：indexer 改成 `limit = cos.shape[0]`；`safe = index.clamp(max=limit-1)`、`local = cos[safe]`、
  `local[index >= limit] = 0`（无需 host 同步）；planner 侧缓存索引张量（`torch.from_numpy` + `non_blocking`），
  长度张量直接写进调用方 buffer。
* 本地能否判定：能（纯代码结构；实际耗时需远端 profile）。

### M2 padding 行被纳入 2B 累计长度（SFA/indexer 会为 padding 行真的跑 attention/top-k）

* 级别：**中**（语义确认项；输出被丢弃，但有数值/性能风险）
* 文件:行号：`zigzag_cp.py:193-194`（断言 `prev+next == local_tokens`，即累计长度覆盖 padding 行）、
  `:207-214`（query=前缀和、key=原始长度）、`sfa_cp.py:415-419`、`sfa_cp.py:652-660`、`indexer.py:838-845`
* 现象：连续切片路径的累计 query 长度只覆盖真实 local 行（`common_cp.py:24-31` +
  `indexer.py:888-947` 的 `local_query_lens`），padding 行在尾部"自然跳过"；zigzag 的 padding 行夹在各 seq 的
  tail block 里，必须靠 -1 slot 与因果 mask 兜住，kernel 会真的为这些行算 attention/top-k（KV 从未写过）。
* 为什么可疑：这些行在模型出口被 `zigzag_gather_tensor(..., num_tokens=full_num_tokens)` 截掉
  （`cp_zigzag.py:630-650`），逐行算子（RMSNorm/RoPE）与 MoE 的 `mc2_mask` 重排
  （`ascend_forward_context.py:355-361`）不会把它们混进真实行；但仍需确认
  (a) 不产生 NaN/Inf 并跨行泄漏；(b) top-k 输出不越界；(c) `num_tokens_pad - num_actual_tokens` 行的白算开销。
* 建议改法：在 planner 注释里写明"累计长度必须覆盖 padding 行"；远端若出现 NaN/精度异常优先查这里。
* 本地能否判定：结构可判定，算子数值行为需远端。

### M3 `_update_parallel_slot_mapping` 的 zigzag 分支目前不可达；若将来放开 DCP 需重审语义

* 级别：**中**（死代码 + 潜在双语义）
* 文件:行号：`sfa_cp.py:552-584`（zigzag 分支 `:571-581`，非 zigzag 重算 `:583`）、
  唯一调用点 `sfa_cp.py:1141`（`AscendSFADCPMetadataBuilder._build_with_metadata_view`，定义 `:1075-1103`，
  临时替换 mapping 在 `:1096-1102`）
* 分析：该方法只被 DCP builder 调用，而 DCP builder 只在 `enable_sfa_dcp_replicated_indexer()` 为真时被选中
  （`sfa_cp.py:1819-1834`），同时 zigzag gate 在 `dcp_replicated` 为真时必拒
  （`cp_zigzag.py:573-574` + `sfa_cp.py:485`）⇒ `zigzag_index is not None` 与"该方法被调用"互斥 ⇒
  **`:571-581` 是死代码**（含 `slot_mapping_cp_gathered` 的二次赋值）。
  回答"是否自相矛盾"：不是矛盾，而是不可达；但确实存在语义重叠 —— 连续 fallback 由
  `_prepare_parallel_metadata`（`:441-443`）与 `:579-581` 两处写，后者用的是**原始（未 DCP 复制）** mapping，
  而内层构建看到的却是被临时替换的复制视图。
* 建议改法：删掉 zigzag 分支（连同两个 assert），或改成 `if dsa_cp_context.zigzag_index is None:` 的正向写法
  并注明"zigzag 与 DCP-replicated 互斥"。
* 本地能否判定：能（调用点唯一 + builder 选择条件与 gate 同源）。

### M4 cos/sin 的"全 padded 自然序表"两条路径等价（结论：一致），但实现代价不同

* 级别：**中**（结论正确；记录实现差异与一处隐式耦合）
* 文件:行号
  * SFA：`sfa_cp.py:352-356`（`nn.functional.pad(cos/sin, (0,0,0,0,0,0,0,pad_size))` 整表拷贝）→
    `sfa_cp.py:545-550`（`cos[plan.zigzag_index]` / `sin[...]` / `slot_mapping[...]`）
  * indexer：`indexer.py:832-836`（`cos.new_zeros(local_tokens, *cos.shape[1:])` + 仅 in-range 行赋值）
  * 同源表：`vllm_ascend/ops/rotary_embedding.py:92-122`（`get_cos_and_sin_mla` 返回 `(T,1,1,D)`，按**行号**索引；
    `use_cache=True` 时返回持久 buffer 的 view）；两侧 `input_positions` 都来自
    `common_attn_metadata.positions`（`sfa_v1.py:528` / `indexer.py:1115`，取 cos 分别在 `sfa_v1.py:551`
    / `indexer.py:1118`）
* 结论：形状/dtype 相同，行语义相同；padding 行两路都给 **0**（SFA 先 pad 0 再索引；indexer 掩码填 0），
  与连续切片的 `local_cos.zero_()` + in-range 拷贝（`indexer.py:916-925`）一致 ⇒
  **满足"全 padded 自然序表"一致性要求**（回答必答问题 3）。
  差异仅在代价：SFA 每 batch 复制整张 rope 表（`num_tokens_pad × 1 × 1 × rope_dim` ×2）；indexer 用掩码（M1 同步）。
* 隐式耦合：8 元 `nn.functional.pad` 依赖 cos 为 4D（第 4 对落在 dim 0）；`get_cos_and_sin_mla` 目前返回 4D ⇒ 成立，
  建议改成显式 `torch.cat([cos, cos.new_zeros(pad_size, *cos.shape[1:])])` 以免形状假设被上游改掉。
* 本地能否判定：能。

### M5 `except Exception` 放宽后，"plan 出错 → 回退连续切片"可能变成单侧回退

* 级别：**中**（新引入于 `3572231a0`；可能造成一侧 zigzag、一侧连续）
* 文件:行号：`sfa_cp.py:516`、`indexer.py:821`（均为 `except Exception as exc  # noqa: BLE001`），
  两个 builder 各自独立 try/except、各自独立 `build_zigzag_cp_plan`（`sfa_cp.py:501-515` / `indexer.py:806-820`）
* 现象/为什么可疑：放宽捕获后，任何只在某一侧出现的异常（例如 H1 的 `num_tokens_pad` 差异导致的形状错误、
  或 planner 未来新增的 `KeyError/AttributeError/TypeError`）都会让**那一侧**静默退回连续切片，
  而另一侧照常建 zigzag → 两侧布局不一致（比直接报错更难定位）。修复前 TypeError 会直接崩，
  反而"不会静默不一致"。
* 建议改法：fallback 决策要跨 builder 一致 —— 例如把 plan 缓存到 `common_attn_metadata`（同一批次 SFA/indexer
  共享同一个 plan 对象），或至少把宽捕获限制在 plan 构建本身，并在回退时用
  `VLLM_ASCEND_CP_BALANCE_DEBUG` 打出 rank/原因，配合一个"同批次两侧决定必须一致"的断言。
* 本地能否判定：能（两侧构建路径独立，异常集合不可能完全对称）。

---

## 5. 低 / 仅记录

### L1 `DSACPContext` 写了没人读的字段

* 级别：**低**
* 文件:行号：`sfa_cp.py:203-240`（字段）与 `:425-445`（写入）
  * `num_tokens`（`:426` 写入）：无读取点（`grep '\.num_tokens\b'` 命中均为 `forward_context.num_tokens` 等无关对象）；
  * `local_end`（`:429` 写入）：只有 `local_end_with_pad` 被读（`:580` 死分支、`:583`、`:633`、`:659`、
    `sfa_cp.py:1588-1590` 附近的 DCP 路径不读它）；
  * `local_tokens`（`:444` 写入，注释称 "kept for the debug branch report and for the merged top-k call"）：
    debug 日志用 `local_end_with_pad - local_start`（`:658`、`:665`），merged top-k 用
    `SFAForwardContext.topk_num_tokens`（字段在 `sfa_v1.py:372`，取值在 `sfa_cp.py:658`、`:665`）⇒ **无读取点**。
* 建议：删除，或让 debug 日志/断言真正使用（否则注释与代码不一致）。

### L2 `ZigzagCPPlan` 的 kv 长度字段没有读取点

* 级别：**低**
* 文件:行号：`zigzag_cp.py:54-55`（`kv_len_prev_list/kv_len_next_list` 字段）与 `:236-237`（写入）；
  `:52-53` 的 `q_len_prev_list/q_len_next_list` 只在 `sfa_cp.py:532-542`（debug 日志）被读。
* 建议：只留被消费的字段，或把 kv 长度接进 debug 日志。

### L3 `resolve_seq_lens_cpu` 的取值来源与 async 回退（结论：安全）

* 级别：**低**
* 文件:行号：`zigzag_cp.py:68-79`；`vllm_ascend/worker/model_runner_v1.py:3490-3498`
  （`seq_lens=self.seq_lens[...]`、`_seq_lens_cpu`/`seq_lens_cpu_upper_bound`/`seq_lens_cpu=` 的传值）与
  `:3446-3450`（`if self.use_async_spec_decode: seq_lens_cpu = None`）
* 分析：抢占 host 副本避免 D2H 同步（`d9dfc4497` 意图正确）：(a) async spec decode 下
  `seq_lens_cpu is None` → 回退 `.to("cpu")`（同步仍在但正确）；(b) 非 async 下
  `optimistic_seq_lens_cpu = num_computed_tokens_cpu + num_scheduled_tokens`（`model_runner_v1.py:1393-1401`）
  与 device `seq_lens`（`:1552-1555`）同公式同源 ⇒ 值一致；(c) `_needs_seq_lens_cpu_sync`（`:528-530`）不含 SFA，
  但 async 下字段为 None，所以不会用到未修正的副本。
* 建议：docstring 写清"仅当 `seq_lens_cpu` 非 None 时可信，且此时它与 device seq_lens 同步"。

### L4 `is_prefilling_cpu` 与 `is_prefilling` 是同一对象（新字段冗余，无同步风险）

* 级别：**低**
* 文件:行号：`vllm_ascend/attention/utils.py:280-283`（新字段）、`utils.py:350-351`（`_slice_reqs` 传递）、
  `worker/model_runner_v1.py:3512`（`is_prefilling_cpu=is_prefilling`，同一对象；
  `:3436-3445` 由 CPU 张量比较得到 `is_prefilling`）
* 分析：`is_prefilling` 来自 `num_computed_tokens_cpu_tensor / num_prompt_tokens_cpu_tensor`，是 **CPU** 张量，
  gate 里的 `.tolist()`（`zigzag_cp.py:119`）不会引入 device 同步；但新字段与 `is_prefilling` 语义、长度完全相同，
  没有独立价值。
* 建议：gate 直接读 `common_attn_metadata.is_prefilling`，删掉新字段（少一处"双份真相"）。

---

## 6. 必答问题速查

1. **SFA builder 与 indexer builder 各自独立算计划，资格输入会不会不一致？**
   * 已修：`8912139f9` 之前 indexer 硬写 `draft_index=None` / `v2_model_runner=False`，与 SFA 不一致
     （draft 步一侧 zigzag、一侧连续）；当前两处 gate 调用已对齐（`sfa_cp.py:474-487` vs
     `indexer.py:789-802`，ast 逐 kwarg 比对一致；`dcp_replicated` 同源 ✓）。
   * 仍不一致：`num_tokens_pad` 基数（**H1**）、gate 不看 PCP（**H2**）。
   * 最致命：**B1** —— main 上 gate 恒 `dp>1`，两侧都拿不到 zigzag。
2. **两个 planner 的结果与下游 kernel 期望是否对得上（2B 行 block table、2B 累计长度、缓冲容量）？**
   * 2B 行 block table：`zigzag_cp.py:231` → SFA 覆盖 `sfa_cp.py:680-684`，
     kernel 只用 `block_table + actual_seq_lengths_*`（`device_op.py:419-470`，不读 1B `seq_lens`）；
     indexer top-k 同约定（`device_op.py:338-400`）⇒ **对得上**；顺序 `[all prevs, all nexts]` 与局部行序一致
     （`cp_zigzag.py:471-478` vs `zigzag_cp.py:196-202`、`:216-222`）。
   * 2B 累计长度：query=前缀和、key=原始 per-batch 长度（`zigzag_cp.py:207-214`），与连续路径约定一致
     （`common_cp.py:24-31`）；dtype int32 ✓；padding 行被计入（**M2**）。
   * 容量：SFA `2*max_num_reqs+1`（**H3**，上界可能差 1，断言在 try 外）；indexer 用 plan 新建张量、不查容量
     （`indexer.py:838-845`）—— 无越界风险，但每 build 新分配（**M1**）。
3. **cos/sin 是否都是"全 padded 自然序表"？** 是（**M4**）：同源 `get_cos_and_sin_mla`、形状 `(T,1,1,D)`、
   行语义一致、padding 行两路都为 0，与连续切片路径一致；差异只是 SFA 先 pad 整表再索引、indexer 掩码填 0
   （后者带一次 `.any()` 同步，**M1**）。
4. **slot_mapping：连续截到 num_actual_tokens、zigzag 全 padded + -1，两条路径对 -1 的假设一致吗？**
   不一致（**H4**）：连续路径的全局写从不把 -1 交给 `reshape_and_cache`/`npu_scatter_pa_kv_cache`，
   zigzag 则把所有 rank 的 padding 行（-1）一次性交出去；局部写两条路径都可能带 -1（-1 语义大概率被支持），
   但这是 zigzag 新增且未验证的依赖，需远端最小用例确认。
5. **`_update_parallel_slot_mapping` 的 zigzag 分支是否自相矛盾（zigzag_index 还在时又重算 fallback）？**
   不是矛盾，而是**不可达**（**M3**）：zigzag 与 `dcp_replicated` 互斥，而该方法只被 DCP builder 调用；
   真正需要澄清的是 `:579-581`（原始 mapping）与 `:441-443`（内层构建时看到的复制视图）两处 fallback 的来源差异。
6. **元数据热路径上的 device 分配/同步隐患**：**M1**（indexer `bool(in_range.any())` 同步 + 逐 build 新张量；
   SFA `torch.tensor(list)` ×5 + `cat`/`index_select`/2 次 buffer copy；`resolve_seq_lens_cpu` 在 async 下仍
   `.to("cpu")`，**L3**；debug 日志 `.tolist()` 有 env 保护 ✓；`.tolist()` 的 host 张量 ✓）。
7. **DSACPContext 新字段"写了没人读/读了没人写"**：写了没人读 → `num_tokens`、`local_end`、`local_tokens`（**L1**）、
   `ZigzagCPPlan.kv_len_prev_list/kv_len_next_list`（**L2**）；读了没人写 → 无
   （`fallback_*`/`block_table_zigzag`/`slot_mapping_cp_gathered`/`inv_gather_index` 都有写入点；
   `slot_mapping_cp_gathered` 的第二个写入点 `sfa_cp.py:578` 是死代码）。
   另外 `fallback_*` 与 `_disable_zigzag_metadata_for_fallback`（`ascend_forward_context.py:137-178`）
   只在 draft / V2 / DP>1 触发，而这三条在 builder gate 里已被拦 ⇒ 该回退链路整体不可达；
   但 **B1** 让"DP>1 已被 gate 拦住"这个前提在 main 上反转成"所以特性永不激活"。

---

## 7. 建议的远端验证顺序（收敛"需远端确认"项）

1. `VLLM_ASCEND_CP_BALANCE_DEBUG=1` + 长 prefill：确认 branch 日志是否只有 `reason=dp>1`（B1），
   同时打印 `enable_dsa_cp()` / `data_parallel_size` / `use_sequence_parallel_moe` /
   `python -c "import vllm;print(vllm.__version__)"`。
2. 若 gate 能过：单请求最小 prefill（`query_len >= 2*cp_size`、`num_actual_tokens >= MIN_TOKENS`），
   观察 indexer cache 写（B3 的修复已让 slot mapping 与 gathered 行数一致，若仍有报错/精度问题优先看这里）。
3. 独立最小用例验证 -1 slot 语义（H4），再决定 zigzag 分支是否加 mask 压缩。
4. 观察 H3：打印每批 `num_reqs` 与 `2*max_num_reqs+1`，确认 dummy-request 插入是否会撑到 `max_num_seqs+1`。
