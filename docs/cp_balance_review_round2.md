> 历史文档（2026-09 上旬的交接/评审期记录）：结论可能已过时，当前状态以 `docs/scripts_review.md` 为准。
>
> 已变事实（2026-09-21 更正）：
> - 本文针对 pre-port 树（658b836ba/c7990e5e4）；现被测树 = `cp_balance` b64c9569b（main 之上移植），
>   行号/类名不可直接引用（DSA-CP 段已移到 `attention/context_parallel/sfa_cp.py`）；对照树 = main aff1b74b6 + `patches/dsa_cp_dp1.patch`。
> - §0 A1：移植后 `fixed_order_reduce_scatter` 只被 `ops/linear_op.py` 与 `ops/fused_moe/shared_experts.py` 调用，
>   不再经过 `mutates_args=[]` 的 custom op，那条冲突已不存在；就地 all_reduce + 视图返回仍在且是有意为之。
> - §4 #3 的 `bash accuracy/run_matrix.sh …` 已删，改用 `bash tests/run_tests.sh --only accuracy/a10_matrix_gate`。
> - §1 的 `ZigzagPlan fields=18` 与 §6 的 13 自相矛盾，现以 harness `expect.zigzag_plan_fields=13` 为准；
>   `min_tokens` 现场值 2048 现在 `configs/_base.json`（envs 默认 8192 仍是未决项 D1）。

# cp_balance 代码复查（第二轮）：分模块问题与精简清单

复查对象：`/d/code/cp_balance/vllm-ascend`（分支 `cp_balance`，HEAD `658b836ba`，工作区干净）
对照：`vllm-ascend-base`（`c7990e5e4`）
改动面：13 个已有文件被改（新增行合计约 1150 行，含注释/文档字符串），另有新增文件 `layers/cp_zigzag.py`（669 行）、`layers/__init__.py` 与 1 个 UT（46 行）。

**本轮已落地并提交**（`vllm-ascend` 分支 `cp_balance` 的 `8bec7da45`，见文末 §6）：B1~B9 全部应用，`git diff` = 5 文件 / −94 / +51（净 −43 行），
静态门控与离线复现全绿。A/C/D 组保留为待决策项。

复查方法（本轮实际做了什么，结论都挂在证据上）：

1. 逐文件 diff（`--strip-trailing-cr`，base 是 CRLF）通读，按调用链走一遍 prefill forward；
2. **离线复现 plan 核心**：把 `layers/cp_zigzag.py` 用 stub 掉 `vllm` / `vllm_ascend.envs` 的方式导入，
   对 `build_zigzag_plan` 做 1600 例 × 全 rank（共约 12000 次 build）的不变量 fuzz，另做余数分配 A/B
   与"每 rank 注意力工作量"核算（脚本留在 `.scratch/review/`，可复跑）；
3. **复核远端证据**：`data/send/export/*op_statistic.csv`（四轮 profiling 的 rank0 集合通信条数/耗时）；
4. 静态门控：`pyflakes` + `cp_balance/perf/check_cp_balance_fields.py` + `cp_balance/accuracy/check_b_path.py`
   （后两个本地 PASS）。

一句话结论：**设计是对的、置换自洽、均衡收益成立**（离线复现全绿，远端集合通信条数与代码结论逐条吻合）。
残留问题分成三类：① 归约实现的**所有权契约**（A1：就地 AllReduce + 视图返回，与
`mutates_args=[]` 冲突，当前只在 `--enforce-eager` 下安全）；② **挂死风险**（A2：宽异常包住集合通信）；
③ 三处**重复/防御性代码**与死代码（B/C 组，合计约 370 行可减，其中 40 行零风险）。
另有一项**新依赖需要远端定性**（A4：KV 写现在依赖 `slot=-1` 被 scatter 跳过）。

---

## 0. 结论摘要

| 级别 | 编号 | 位置 | 问题 | 建议 |
| --- | --- | --- | --- | --- |
| P0 契约 | A1 | `distributed/utils.py:48-51` + `register_custom_ops.py:247-253` | 就地 `all_reduce` 改了调用方的缓冲区，返回值又是该缓冲的**视图**；而该路径是注册为 `mutates_args=[]` 的 custom op | 二选一：`clone()` 回来，或显式声明 `mutates_args=["x"]` |
| P0 挂死风险 | A2 | `linear_op.py:203-213`、`register_custom_ops.py:34-48` | 用 `except Exception` 把**集合通信调用**包起来，失败后改发另一种集合通信 → 各 rank 不匹配即挂死，且掩盖真 bug | 可预期失败条件前置判断，`try` 只包能力探测 |
| P1 隐式前提 | A3 | `register_custom_ops.py:38`（对比 `:139`、`prepare_finalize.py:523`） | 固定序归约固定用 TP 域；base 在 `dp_metadata is not None and is_ep_comm` 时用 EP 域。本配置 DP=1 → 两者都是 TP 域，成立但无断言 | 启动时断言 `ep/tp` 域一致，或把 group 传下来 |
| P0 新增依赖 | A4 | `sfa_v1.py:2244-2250`、`:2567-2568` | zigzag 的 KV/索引写改成"全 padded 行 + slot=-1"，**新依赖 -1 被 scatter 跳过**（连续路径只写 `[:num_actual]`） | 远端做一次定向 A/B（过滤 slot≥0） |
| P1 死代码 | B1 | `worker/model_runner_v1.py:37` | `import vllm.envs as envs_vllm` 未被使用（pyflakes；base 无此行） | **已删** |
| P1 死代码 | B2-B4 | `linear_op.py:406-407`、`model_runner_v1.py:2622-2623` | `dsa_cp` / `sp_enabled` 中间变量只用一次；3517 行多余空行 | **已删** |
| P1 死别名 | B5 | `sfa_v1.py:2557-2558` | `k_li_to_write` / `k_li_scale_to_write` 从不被重新赋值 | **已删**，直接用 `k_li` |
| P1 死字段 | B6 | `cp_zigzag.py:113-122` | `ZigzagPlan` 5 个字段全仓库无读者（只写不读） | **已删**（`num_tokens`/`cp_size` 经属性 `local_tokens` 有用，保留） |
| P1 死写 | B7 | `ascend_forward_context.py:327-332` | V2 分支写入的两个值恒为 `None`（`:220-222` 已在该分支置 False） | **已删**（改成单一写入路径） |
| P1 重复 | B8 | `sfa_v1.py:2616-2686` | zigzag/连续两条分支重复同一套 4 条 assert + 两个调用点 | **已改**：前面算一次三元组，后面各留一个调用点（−23 行） |
| P1 陷阱 | B9 | `sfa_v1.py:1952` | `_indexer_qk_proj` 的 `output_dtype` 默认值 `cos.dtype`，唯一调用点总传 `x.dtype` | **已改成必填参数**（唯一调用点核对过） |
| P2 精简 | C1 | `cp_zigzag.py:206-308` | 余数分配两段式（aggregate max-flow + 分解）在 2000 例里退化 22 例，单段 per-row flow 2000/2000 可用；两段结果 37% 的案例不同 | 删 aggregate 段（−73 行）**但需重跑首 token 验收** |
| P2 精简 | C2 | `ascend_forward_context.py:105-150/225-232/306-311` + `sfa_v1.py:236-238/669-672/803-812` | 回退机制：三个条件与 builder gate 重复判定，正常路径不可达（见 §2.4 分析） | 删（−90 行，省一次 RoPE 查表）；风险见文 |
| P2 精简 | C3 | `patch_deepseek_v2.py:317-346`、`vocab_parallel_embedding.py:170-201`、`envs.py:109-112`、UT 46 行 | `EMBED_LOCAL` 实验路径默认 0、远端从未验证、量级可忽略 | 验证一次或删（−110 行） |
| P2 精简 | C4 | `linear_op.py:202-214/351-384/463-470`、`register_custom_ops.py:25-48` | 三处"zigzag 换固定序归约"逻辑重复、异常集合不一致 | 收敛成一个 helper（−50 行） |
| P3 配置 | D1 | `envs.py:85,90` | `CP_BALANCE` 默认 1（升级即默认开）；`MIN_TOKENS` 默认 8192 而所有测试配置用 2048 | 默认值/文档对齐 |
| P3 测量 | D2 | `docs/perf_plan.md` H1 | 实测 16 次归约在窗口内只占 12 ms（0.018%），而轮间漂移 5~13% | 不要据这一轮改 `REDUCE_MODE` 默认值 |

---

## 1. 复查方法与既有验证状态

- 既有静态门控（本地跑通，作为我的基线）：
  `check_cp_balance_fields.py` → `ZigzagPlan fields=18, DSACPContext fields=18 / RESULT: PASS`；
  `check_b_path.py` → 10 项 OK / `RESULT: PASS`。
- 我新增的离线复现（`.scratch/review/plan_fuzz.py`、`flow_ab2.py`、`balance_math.py`）：
  - 1600 例（cp∈{2,4,8,16}，1~6 请求，长度 2cp~40cp，含 pad、含 prefix）全 rank build，
    失败 0：`zigzag_index` 恰为 gather 序第 `cp_rank` 段、`inv_gather_index` 与 gather 互逆、
    三套索引恰好覆盖 `[0, T)` 一次、每 rank 行数恒等于 `T/cp`、`kv_len_*` 与
    `prefix + Σblocks[:r+1]`（prev）/`prefix + Σblocks[:2cp-r]`（next）一致；
  - 均衡核算（同 plan 语义）：连续切片每 rank 工作量 max/mean = 1.83~2.30（16 卡下首末 rank 差 ~31×），
    zigzag = 1.00~1.02 → **cp_balance 的收益来源被复核**；
  - 余数分配 A/B：两段式中 aggregate 分解失败 22/2000（1.1%），单段 per-row flow 失败 0/2000；
    两者结果不同的案例 742/2000（37%）。
- 远端证据复核（`data/send/export/*rank0*_op_statistic.csv`，09-15 四轮 profiling）：
  `cur_cp1` vs `cur_cp0`：`reduce_scatterAicpuKernel` 4920 vs 4936（−16）、`allreduceAicpuKernel` 16 vs 0
  —— 与"3 个 dense 层 down_proj + embedding = 4 处/步 × 4 个 prefill 步"的代码结论**逐条吻合**；
  `cur_cp0` 与 `base_cp0` 的集合通信条数**完全相同**（allgather 8356、alltoall 4680、RS 4936）
  → "CP_BALANCE=0 走 base"在集合通信层面成立。
- 尚未被远端验证的（本轮必须补的）见 §4。

---

## 2. 分模块分析

### 2.1 开关面 `vllm_ascend/envs.py:77-112`

职责：5 个环境变量 = 总开关、最小 token 阈值、归约模式、debug、embedding 实验开关。

- 读到的都是"每进程一次"的量，除了 `MIN_TOKENS`/`DEBUG` 在 metadata 热路径里被反复读（`:90`/`:102`，
  每 batch 每 rank 各 1~2 次）：`bool(int(os.getenv(...)))` 每次都重新解析环境变量。
  与 `distributed/utils.py:25-27` 用 `lru_cache` 缓存的做法不一致。
  → 一致性建议：要么都缓存，要么把 `MIN_TOKENS` 读一次存模块级（预算：每步省 2 次 `getenv`，非关键）。
- `VLLM_ASCEND_CP_BALANCE` **默认 1**（`:85`）：无配置的部署升级上来默认走 zigzag 布局。
  这是一次"静默改变数值路径"的默认值。`docs/cp_balance_walkthrough.md` §14 已把它列为坑，
  但没有行动项。建议默认 0，或在发布说明/README 显式声明。
- `VLLM_ASCEND_CP_BALANCE_MIN_TOKENS` 默认 8192（`:90`）与 `cp_balance/configs/_common.json` 的 2048 不一致：
  按默认值部署时，< 8192 token 的 prefill 永远进不了 zigzag（而 GLM-5.2 的典型长文 prefill 在
  2048~8192 之间）。默认值与实测配置对齐，否则等于"默认关了一半功能"。
- `VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL`（`:109-112`）没有远端验证记录（见 C3）。

### 2.2 plan 核心 `vllm_ascend/layers/cp_zigzag.py`（669 行）

职责：eligibility 谓词、CPU 侧块划分与三套置换、MoE 侧 aux 重排、固定序求和、模型边界的
shard/gather。

**复核通过（不要重复审）：**

- `build_zigzag_plan`（`:407-552`）的数学：`zigzag_index`、`zigzag_gather_index`、`inv_gather_index`
  三者互洽（§1 fuzz）；`_balanced_zigzag_blocks` 的等行断言（`:392-402`）是硬不变量，任一 rank 破坏
  都会在全部 rank 上同时抛（输入只依赖 metadata，rank 间同构）→ **回退也是 rank 对称的**，
  这一点很关键，未来改动不要引入 rank 相关的判定输入。
- `zigzag_ineligible_reason`（`:555-631`）的 gate 顺序合理，且所有输入都来自"各 rank 相同"的 metadata；
  `pad%cp_size` 与 `num_tokens_pad` 同源（model_runner 的 padding）。
- 因果长度：每块是自然流的连续区间，`kv_len = prefix + 块末位置`，配 `sparse_mode=3`（rightDownCausal）
  正好给块内每个 token 正确的因果前缀；next 块（`2cp-1-r`）的 kv_len 取到该块末尾，与 base 的
  `seq_lens - offset` 语义同源。

**问题/精简：**

1. `ZigzagPlan` 有 5 个只写不读的字段：`num_actual_tokens`、`cp_rank`、`query_lens`、
   `effective_query_lens`、`block_sizes`。
   全仓库唯一读者是 `sfa_v1.py` 的 `plan.*`（见 §2.3），都不涉及这 5 个；
   `num_tokens` 与 `cp_size` **要保留**（`local_tokens` 属性读它们）。
   → **已删**（含 `block_sizes` 的两行注释）。
2. 余数分配两段式（C1，`_allocate_remainder_extras:239-308` + `_decompose_remainder_rows:206-236`）：
   先按"余数相同的请求"聚合做一次 max-flow，再把列容量分解回每行——分解这一步 2000 例里
   22 例无解（返回 None）→ 落到 `_individual_remainder_extras`（`:311-344`）。
   单段 per-row flow 单跑 2000/2000 成功，且两段结果在 37% 的案例上**不同**。
   → 删 aggregate 段可省 73 行并去掉一条失败路径；代价是 37% 批次的块分布变化（布局变），
   **必须重跑 `matrix_c_accept` 首 token**。若不删，至少把"aggregate 常退化"这条写进注释，
   免得后人以为它才是主路径（现在真正干活的是 per-row flow）。
3. `_MaxFlow.max_flow`（`:167-203`）内部 `dfs` 是闭包递归，递归深度受节点数限制
   （节点数 = cp + 余数组数 + 2 ≤ 20），无栈风险；`limit` 参数只在两处用，可留。
4. `zigzag_shard_tensor`/`zigzag_gather_tensor`（`:634-660`）在每次调用都走一遍
   `get_zigzag_cp_context()`（内含 try/except + 模块导入）——每 forward 1~2 次、不是每层，
   可以不动；但 embedding/出口两次调用可以显式把 ctx 传进去（`zigzag_reorder_moe_aux` 已有 `ctx` 形参，
   `zigzag_shard_tensor` 没有），风格上统一一下更好。

### 2.3 metadata + 算子路径 `vllm_ascend/attention/sfa_v1.py`（+542 行）

职责：资格判定 → plan → device 张量 → `DSACPContext`；SFA 侧三条 zigzag 分支（KV 写回、indexer 写回、
topk+SFA 合并调用）。

**复核通过：**

- KV 写回：`fused_kv_no_split` 是**全量 gathered 序**（`_maybe_gather_kv_for_dsacp` 的 all_gather 按 rank 拼接），
  `slot_mapping_cp_gathered = slot_mapping[zigzag_gather_index]` 与之同序；连续路径写
  `[:num_actual_tokens]`（自然序前段），zigzag 写全量 padded 行（-1 跳过）——**两者语义各自自洽**（见 A4）。
- indexer 写回：`enable_sparse_li_c8` 时 `k_li` 单独 all_gather（同样 gathered 序），与 `idx_slots` 匹配；
  非 C8 时从 `fused_kv_no_split` 切出全量 `k_li`（`:2276-2281`），也匹配。
- 合并调用：`actual_seq_lengths_query_zigzag` 是 [all prev, all next] 的**累积**前缀和，
  `actual_seq_lengths_key_zigzag` 是**逐 batch raw** kv_len，与 base 的两种语义一致；
  `block_table_zigzag` 行序 = 真实请求序 + 补零，再复制两份，与 Q 行序一致（`:637-655`）。
- `block_table` 只有两个 kernel 消费点（`device_op.py:495` / `:564` 的 lightning_indexer 与 SFA），
  两处都拿到了 zigzag 覆盖值；`attn_metadata` 其余字段在这两条路径上没有被读到（已逐个查过）。

**问题/精简：**

1. 死别名（B5）：`:2556-2558` 的 `k_li_to_write` / `k_li_scale_to_write` 全程未被重新赋值，
   只是 `k_li` / `k_li_scale` 的别名；真正被换掉的只有 `idx_slots`。**已删**（删后这段与 base 逐行一致）。
2. 重复分支（B8）：`:2616-2657`（topk）与 `:2659-2686`（SFA）各写一遍 4 条 assert + 调用，
   差别只在三个 metadata 张量。建议：
   ```python
   ctx = attn_metadata.dsa_cp_context
   if zigzag_active and ctx is not None:
       assert ctx.actual_seq_lengths_query_zigzag is not None  # 只留一处
       aq, ak, bt = ctx.actual_seq_lengths_query_zigzag, ctx.actual_seq_lengths_key_zigzag, ctx.block_table_zigzag
   else:
       aq, ak, bt = actual_seq_lengths_query, actual_seq_lengths_key, None
   ```
   两个调用点各留一次 → 约 −23 行，且消除"改一处忘另一处"的风险。
   **已按此实现**（`:2608-2650`：先算 `query_lens_arg/key_lens_arg/block_table_arg`，topk 与 SFA 各一个调用点）。
3. `_indexer_qk_proj`（`:1936-1940`）：`output_dtype` 缺省回落到 `cos.dtype`，而 `cos` 来自
   `_cos_cache`（fp32：`vllm/model_executor/layers/rotary_embedding/__init__.py:41-42` 在模型不传
   dtype 时取 `torch.get_default_dtype()`），与唯一调用点传的 `x.dtype`（bf16 激活）不同。当前不会触发，但
   一旦有人省略参数就是静默的精度路径变化。→ 改成必填位置参数。
4. `_build` 的连续回退 RoPE（`:669-672`）：zigzag 命中时仍然为**回退用**再算一份 `cos_continuous/sin_continuous`
   （两次 `_cos_cache[positions]` 索引）。与 C2 绑定：若删回退机制，这两行随之删除；若保留，
   建议改成"回退函数需要时再算"（把 `input_positions`/`local_start`/`local_end_with_pad` 存进 ctx）。
5. `_build` 里 `query_lens_cpu` 提取（`:525-556`）在 DSA-CP 关闭 / 非 zigzag 时也照跑一遍
   （`.tolist()` 是 CPU 张量，无 D2H；成本 = 每 metadata 一次 O(num_reqs) 的 Python 循环）。
   可接受；但如果想省，可以放在 `if self.enable_dsa_cp:` 之后再算。
6. `zigzag_gate` 初值 `"dsa_cp_off"`（`:563`）只服务日志，OK；注意 `[CP_BALANCE][plan]` 里的
   `zigzag_index[:8].tolist()` / `slot_mapping_cp[:8].tolist()` 是 D2H 同步（`:705-717`），
   所以 profiling 配置必须 `debug=0`（`configs/_profile_common.json` 已经是）。
7. 新增的 `is_prefilling_cpu` 与 `is_prefilling` **当前是同一个 CPU 张量对象**
   （`model_runner_v1.py:2905-2907` 传的是同一个 `is_prefilling`，而它由两个 `*_cpu_tensor` 比较得到）。
   保留双字段的理由（上游注释把 `is_prefilling` 标成 `torch.Tensor`）可以接受，但要意识到
   "无 D2H 同步"这个保证来自 runner 传的是 CPU 张量，不是来自字段名。

### 2.4 逐 forward 开关 `vllm_ascend/ascend_forward_context.py`（+150 行）

职责：找出本 forward 的 zigzag ctx、**事后否决**（draft / V2 / DP>1）、把 `input_ids` 与 `mc2_mask`
按同一张置换重排、提供 `zigzag_active()`。

**复核通过：**

- `input_ids`（`:334-343`）与 `mc2_mask`（`:352-362`）用**同一个** `zigzag_reorder_moe_aux`，
  且两个消费者（`experts_selector.py:264-270` 的 TP 切分、`prepare_finalize.py:276-280` 的
  `tensor_split(mc2_mask, tp)[tp_rank]`）用的都是"第 tp_rank 段"，与 FlashComm 的 rank 拼接序一致；
  模型本身的 `input_ids`（函数参数）保持自然序，embedding 不受影响。
- `zigzag_active()`（`:605-616`）用 try/except 兜住"没有 forward context"的情况，三处归约点全部挂在它后面；
  `check_b_path.py` 已验证"没有裸读配置开关的归约"。

**问题/精简：**

1. 回退机制（C2）**当前不可达**，但代价和文档口径需要修正：
   - 与 builder gate 重复判定的两个条件：V2 runner（同一环境变量）、DP（`parallel_config.data_parallel_size`
     vs `get_dp_group().world_size`，两者同源）；
   - 只有 `is_draft_model` 在 builder 侧没有对应量，而它**唯一可能触发**的路径是
     `spec_decode/llm_base_proposer.py:653` 用 `builder.build_for_graph_capture(...)` 构建 draft 元数据
     （`draft_index` 未传）；但 `build_for_graph_capture`（`sfa_v1.py:867-881`）只支持
     `DecodeOnly`/`SpecDecoding`，而这两个 state 已在 `_PURE_PREFILL_ATTENTION_STATES` 之外被
     `state=...` 门拒掉 → **正常路径不可达**；
   - 因此 `docs/cp_balance_walkthrough.md` §14-5 "缺 fallback 会直接抛错" 的保护目前是纯防御。
     删掉可省 `_disable_zigzag_metadata_for_fallback`（46 行）+ 调用点（约 14 行）+ 3 个 ctx 字段
     （`sfa_v1.py:236-238`）+ 每次 build 的第二份 cos/sin（`sfa_v1.py:669-672`，4 行），约 90 行。
     保留的理由是"上游将来新增 draft 元数据构建路径"；折中方案：保留但把
     `is_draft_model` 的判定移到 builder（把 forward 的 `is_draft_model` 传进 metadata 构建），
     这样两个判定收敛到一处，回退代码就可以删。
2. V2 分支死写（B7）：`:327-332` 里 `zigzag_cp_active` 在 `:220-222` 已因 V2 置 False，
   两个写值恒为 `None`；`_EXTRA_CTX.__getattr__` 在 V2 下走 `additional_kwargs.get(name)`，
   缺键同样是 `None` → **已删**，改成 `forward_context.zigzag_cp_context/active = ...` 单一写入路径
   （vLLM 的 `ForwardContext` 是普通 dataclass，赋值可用；grep 确认无任何读者依赖这两个
   `additional_kwargs` 键存在）。
3. 缺一条显式不变量：zigzag 语义依赖 FlashComm 的"rank 拼接"集合通信（`flash_comm_v1_enabled`）。
   现在没有任何地方断言 `zigzag_cp_active ⇒ flash_comm_v1_enabled`。DSA-CP 要求 SP，
   而 MoE 模型下 `flash_comm_v1_enabled = enable_sp and num_tokens is not None`，所以当前必然成立，
   但 `_pad_for_sequence_parallelism` 还会因 `enable_sp_by_pass()` 触发 padding（口径不同）。
   → 一行加固：`zigzag_cp_active = zigzag_cp_active and flash_comm_v1_enabled`。
4. `_iter_attn_metadata`（`:70-87`）兼顾 dict / list-of-dict / 单对象，配合 ubatching 是对的；
   若 C2 删掉回退，这个 helper 只剩 `_find_zigzag_cp_context` 一个使用者，可考虑合并成一个函数。

### 2.5 归约实现 `vllm_ascend/distributed/utils.py`（+134 行）

职责：三种与 owner 无关的 reduce-scatter 实现 + 模式解析。

**复核通过：**

- chunk 语义与 vLLM 一致：`_allreduce_slice_reduce_scatter`（`:30-51`）rank r 取 `[r·chunk,(r+1)·chunk)`，
  与 `vllm/distributed/device_communicators/base_device_communicator.py:244-262` 的
  `dist.reduce_scatter_tensor` 相同；
- `alltoall` 模式（`:73-93`）方向正确：`all_to_all_single` 是"输入第 i 段发给 rank i、输出第 i 段来自 rank i"，
  所以 rank r 的 `recv[i]` = rank i 的第 r 段 → 逐段按 source rank 求和正是 RS 的结果（2 卡反例已推演）。
- `_plain_reduce_scatter`（`:54-71`）与 base 完全等价，作为 A/B 基线合理。

**问题/精简：**

1. **A1（P0）就地 AllReduce + 视图返回**：
   - `summed = tensor if tensor.is_contiguous() else tensor.contiguous()` → 对连续输入**改写了调用方的缓冲区**；
   - `return summed[a:b].contiguous()`：dim0 连续切片是连续视图，`.contiguous()` 是 no-op → 返回的
     `[L,H]` 张量仍持有整块 `[T,H]` 存储，且与输入别名（调用方若复用输入会读到"已跨 rank 相加"的值）；
   - 三个调用点目前都不复用输入（已逐个查证：`linear_op.py:199-214`、`:460-470`、
     `register_custom_ops.py:114-121`），但 `maybe_pad_and_reduce` 注册时声明的是
     `mutates_args=[]`（`register_custom_ops.py:247-253`）→ 对 torch.compile / 功能化 / ACL graph
     属于未定义行为；当前只在 `--enforce-eager`（测试配置）下安全。
   - 建议二选一并写进注释：`return summed[...].clone()`（每次多一次 1/cp 张量拷贝），
     或保持就地 + 把注册改成 `mutates_args=["x"]` 并在 docstring 写明"输入将被就地覆盖、归约后不得复用"。
2. `reduce_mode()` 的 `lru_cache`（`:25-27`）在进程内首次读取后固定 —— 与"A/B 需要跑两轮服务"的使用方式一致，
   但要写清：同进程改环境变量无效。
3. 三种模式都保留（`:136-146`）：`reducescatter` 只是 base 基线的等价物，`alltoall` 目前没有实测优势
   （见 D2）→ 若最终不改默认值，`alltoall` + `fixed_order_rank_sum`（`cp_zigzag.py:51-75`）可以只留一份。

### 2.6 三个归约挂载点 `ops/linear_op.py` / `ops/register_custom_ops.py`

**复核通过：** 三处都在 `zigzag_active()` 之后，非 zigzag 时逐字节走 base（`check_b_path.py` 钉住了这一点）；
`_maybe_pad_and_reduce_impl` 的 fake 实现形状与固定序路径一致（`register_custom_ops.py:150-157`）。

**问题/精简：**

1. **A2（P0）异常处理**：`linear_op.py:203-213` 与 `register_custom_ops.py:34-48` 用
   `except Exception` 包住**整个集合通信调用**，失败后回落到另一种集合通信。若某 rank 在通信内部抛错
   （或各 rank 行为不一致），这些 rank 会去发另一个集合通信 → 与其它 rank 不匹配 → **挂死而不是报错**；
   同时它会掩盖真实 bug（例如 dtype/shape 错误被吞掉后静默换实现）。
   → 把"能力判定"和"通信"分开：shape/整除/group 存在性用前置判断（`linear_op.py:365-376`
   已经是这个风格），`try` 只包探测，不包通信。
2. 三处重复（C4）：`linear_op.py:202-214`、`:351-384`、`:463-470`、`register_custom_ops.py:25-48`
   共 4 段同构代码，异常集合却不同（`except Exception` / `except (RuntimeError, NotImplementedError,
   TypeError, ValueError)` / `except Exception`）、日志站点名各异。→ 收敛到一个 helper：
   `zigzag_reduce_scatter(tensor, group, site) -> torch.Tensor`，内部统一 precheck + 日志 + 回退。
3. `dsa_cp = enable_dsa_cp()`（`linear_op.py:406-407`）是纯中间变量，合成一行即可（B2）。
4. **A3（隐式前提，当前成立）**：`register_custom_ops.py:38` 固定用 `get_tp_group()`；同一函数在
   `dp_metadata is not None and is_ep_comm` 时用 `get_ep_group()`（`:139`），而那条路要求 DP>1、
   被 zigzag 资格门拒掉。DP=1 时 base 也是走 TP 域（`prepare_finalize.py:523` 传 `is_ep_comm=True`
   但 `dp_metadata is None` → 走第一支）→ **本轮安全**。
   风险在于：MoE 的 dispatch/all-gather 走的是 EP 域，一旦将来放宽 DP 门，固定序归约必须换成
   EP 域，否则归约的"第几段"与聚合序对不上。→ 启动日志打一行/断言
   `ep_group.world_size, ep_group.rank_in_group == tp_group.world_size, tp_group.rank_in_group`，
   把这条隐式前提变成显式的（成本一行，收益是以后不会踩）。

### 2.7 模型边界 `ops/vocab_parallel_embedding.py` / `patch/worker/patch_deepseek_v2.py`

**复核通过：**

- `_embed_partial` / `_forward_origin` 拆分（`vocab_parallel_embedding.py:262-288`）与 base 逐行等价
  （masking、masked_fill_、`maybe_pad_and_reduce` 的位置都没变）。
- embedding 处"先 all_gather 回全长再 shard"的必要性正确：FlashComm 下 embedding 返回的是
  TP-reduce-scatter 后的 `T/tp` 行本地切片，而 `zigzag_index` 是**全局自然位置**，必须先取回全长。
- `positions` 一起 shard（`patch_deepseek_v2.py:375-379`）不是可选项：出口靠形状判断跳过模型内的
  二次 gather（`:398-419`），切了才成立；三处 `RuntimeError` 校验（`:340/:364/:378`）是好的失败姿态。
- `DeepseekV2DecoderLayer.forward` 不 patch 的前提成立且**同源**：`use_sequence_parallel_moe`
  在 `vllm/config/parallel.py:653-668` 要求 `data_parallel_size > 1`，而 gate 用的正是
  `parallel_config.data_parallel_size`（同一个字段）→ 不是"注释约定"，是真的钉住了。

**问题/精简：**

1. **A4（P0 新增依赖）**：zigzag 路径的 KV/indexer 写改成"全 padded 行 + `slot=-1`"
   （`sfa_v1.py:2244-2250`、`:2567-2568`），连续路径写 `[:num_actual_tokens]`（不含 padding 行）。
   indexer 那条在 base 本来就是全量 slot_mapping（含 -1）→ 有先例；KV 主写这条是**新依赖**。
   若某颗芯片/某个 CANN 版本把 -1 当"最后一个 slot"，被覆盖的就是最后一个真实 token 的 KV，
   表现为首 token 直接偏。→ 远端定向 A/B：把 `slot_mapping_cp_gathered` 里 `<0` 的行过滤后再 scatter，
   比较首 token 与 KV dump（不改代码的等价做法：临时把 `scatter_slots.clamp_min(0)` 与掩码版本对比）。
2. `EMBED_LOCAL`（C3）：`patch_deepseek_v2.py:317-346` + `vocab_parallel_embedding.py:170-201` +
   env + UT，默认 0、远端零验证；`docs/perf_plan.md` H4 已判断"量级可忽略"。
   → 本轮验证一次或直接删；留着会让 embedding 入口有两条互斥路径（且 `hidden_is_zigzag_local`
   这个标志位只为它存在）。
3. `use_embed_local` 的 `hasattr(self.embed_tokens, "forward_zigzag_local")` 探测 + 4 处 shape 校验，
   在删除 C3 后只剩 `hidden_states.shape[0] != positions.shape[0]` 一次 all_gather 判断，可读性会明显变好。

### 2.8 runner 与容器 `worker/model_runner_v1.py` / `attention/utils.py` / `utils.py` / `device/device_op.py` / `ops/fused_moe/experts_selector.py`

- `model_runner_v1.py:37`：**未使用的 `import vllm.envs as envs_vllm`**（B1，pyflakes 唯一的新告警，base 无此行）。
- `model_runner_v1.py:2622-2623`：`sp_enabled` 中间变量只用一次（B2）；`:3517` 多了一个空行。
- 出口 gather（`:2603-2610`）：`flash_comm_v1_enabled or zigzag_cp_active` 两个条件其实后者蕴含前者
  （§2.4-3 的加固若加上，这里可以简化为只看 `zigzag_cp_active`）；
  `getattr(forward_context, "zigzag_cp_active", False)` 读的是 ForwardContext 上的属性，
  与 `_EXTRA_CTX` 读 `additional_kwargs` 的 V2 路径不通用（V2 下这个分支读不到）——V2 走的是
  R1/V2 的 runner，不影响本配置，但值得在注释里点明。
- `attention/utils.py:229-232, 296` + `model_runner_v1.py:2905-2907`：`is_prefilling_cpu` 与
  `is_prefilling` 是同一对象（见 §2.3-7）。
- `utils.py:1372-1425`：`enable_dsa_cp_for_config` / `dsa_cp_with_o_proj_tp_for_config` 的拆分是**纯重构**，
  `enable_dsa_cp()` / `enable_dsa_cp_with_o_proj_tp()` 退化成 `lru_cache` 包装，逻辑没有重复；
  唯一注意点：`dsa_cp_with_o_proj_tp_for_config` 与缓存版必须保持同源（docstring 已写）。
- `device/device_op.py`：`block_table: torch.Tensor | None = None` 的默认回落（`:495`、`:1694`）
  向后兼容，不传就是 base 行为；两处 `indexer_select_post_process`（A5 / 其它 SoC）都改了，
  没有漏改的第三份（已 grep 确认只有两处定义）。
- `experts_selector.py:250-252`：纯注释改动；`input_ids` 的重排已经提前到 forward context，
  这里 `.to(torch.int64)` 在 zigzag 下是 no-op（forward context 已转 int64），非 zigzag 下保持 base 行为。

---

## 3. 精简收益排序

| 序 | 项 | 删除行数（约） | 风险 | 需要重跑验收 |
| --- | --- | --- | --- | --- |
| 1 | B1~B7 死代码/死字段/死别名 | 40 | 极低（纯删无读者/无写入者） | 否（跑 check_b_path + pyflakes 即可） |
| 2 | C1 余数分配单段化 | 73 | 中（37% 批次块分布变） | **是**（matrix_c_accept 首 token） |
| 3 | C2 回退机制删除 | 90 | 中（依赖"gate 与 forward 判定同源"成立） | **是** |
| 4 | C3 EMBED_LOCAL 删除 | 110 | 低（默认关的实验路径） | 否（但需确认没人依赖） |
| 5 | C4 三处归约收敛 + A2 异常收敛 | 50 | 中（碰集合通信，需回归） | **是** |
| 6 | B8 topk/SFA 分支合并 | 20 | 低 | 是（轻量：一次长 prompt 首 token） |

合计理论上可减 ~370 行（当前新增约 1150 行的三分之一），其中"无风险 40 行"可以立刻删。

---

## 4. 远端待验证清单（改完再跑，一次一步）

| # | 目的 | 动作 | 判据 |
| --- | --- | --- | --- |
| 1 | A3：EP 域 == TP 域 | 启动后打印/断言 `get_ep_group()` 与 `get_tp_group()` 的 world_size / rank_in_group | 打印相等；不等则 fixed-order 的 group 必须换成 EP |
| 2 | A4：-1 slot 是否被跳过 | 一轮 `CP_BALANCE=1` + 临时把 `slot_mapping_cp_gathered` 过滤 `>=0` 的 A/B（或 KV dump） | 两版首 token / KV 一致 → 可安全依赖 -1 跳过 |
| 3 | 最新代码的首 token 验收 | ~~`bash accuracy/run_matrix.sh configs/matrix_c_accept.json`~~（该脚本已删）→ `bash tests/run_tests.sh --only accuracy/a10_matrix_gate` | `RESULT: PASS`（矩阵现在还会用 `require_log` 门确认 cp1 真的出现过 zigzag 证据） |
| 4 | 若采纳 C1/C2/C4 | 重跑 #3 | 同上 |
| 5 | D2：归约模式是否值得改默认 | 每模式至少 2 轮重复（`matrix` 已有 repeat 机制），看窗口内 `allreduce` 总时长与整体漂移 | 单轮里 16 次 AR 只有 12 ms（占窗口 0.018%），轮间漂移 5~13% → 不做这个优化 |
| 6 | H0（每层 o_proj 权重 all_gather） | 现有 profiling 数据里 `allgatherAicpuKernel` = 8356 次、10.4 s（rank0 窗口，占 17%） | 若要动，先确认其中多少来自 o_proj 权重（`profile_order.py --devices` 的归因表；上一轮 `trace_view.json` 3.1 GB 解析失败，需要更小的窗口或 6 层配置） |

---

## 5. 给下一次改动的三条纪律（从本次复核里提炼）

1. **改 plan 的输入集合，必须同时想回退的对称性**：所有判定输入都必须"各 rank 同构"，
   否则一个 rank 回退、另一个 rank 走 zigzag 就是挂死。
2. **碰集合通信的代码不要用宽异常兜底**：在 CP 场景下，宽异常 = 挂死风险，且会把真实 bug 藏起来。
3. **文档里的"已验证"要和证据对上**：本轮复核发现两处口径问题——
   `walkthrough §14-5` 说"回退在正常路径不会发生"是对的但没给出不可达的证明（§2.4-1 补上了）；
   `perf_plan` 的 H1 假设（默认归约贵出一倍）在实测里只值 12 ms（D2）。结论性的文档要挂数字。

---

## 6. 本轮已落地的改动（已提交 `8bec7da45`）

`git -C vllm-ascend --no-pager diff --stat`：

```
 vllm_ascend/ascend_forward_context.py |  11 ++--
 vllm_ascend/attention/sfa_v1.py       | 109 +++++++++++++---------------------
 vllm_ascend/layers/cp_zigzag.py       |  17 +-----
 vllm_ascend/ops/linear_op.py          |   3 +-
 vllm_ascend/worker/model_runner_v1.py |   5 +-
 5 files changed, 51 insertions(+), 94 deletions(-)
```

对应的验证（全部本地）：

| 检查 | 结果 |
| --- | --- |
| `python -m py_compile` 5 个改动文件 | OK |
| `python -m pyflakes`（11 个 cp_balance 相关文件） | 仅剩 base 已有的 5 条告警（`utils.py` 4 条 + `SchedulerOutput` 重定义），新增的未使用 import 已消失 |
| `check_cp_balance_fields.py` | `ZigzagPlan fields=13, DSACPContext fields=18` / `RESULT: PASS`（13 = 18 − 5，正是删掉的 5 个） |
| `check_b_path.py` | 10 项 OK / `RESULT: PASS` |
| `.scratch/review/plan_fuzz.py`（1600 例全 rank） | `failures=0`（删字段后 plan 行为不变） |
| `.scratch/review/flow_ab2.py` | 与报告正文数字一致（22/2000、0/2000、742/2000） |
| `.scratch/review/balance_math.py` | 均衡表与正文一致 |

已提交为 `vllm-ascend` 的 `8bec7da45`（分支 `cp_balance`，尚未 push）。远端验收（§4 清单第 1~3 条）
在 push 之后必须跑一次：`matrix_c_accept` 首 token 验收）在 push 后必须跑一次：这几处是
"无读者/无写入者"的确定性删除，但仍要有一轮 C 验收兜底。
