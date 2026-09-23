# cp_balance 远端验证清单（本轮精简之后）

> 2026-09-23：zigzag 改为 attention 内部选行（模型主流恢复 replicated），本文中“模型边界切行/固定序归约”的描述已过时，见 `docs/dp1_zigzag_acc_fix.md`。

配套文档：`docs/cp_balance_review_round2.md`（问题与精简清单）。本清单只回答两件事：
**改完之后怎么验收**、**哪些开放项必须靠远端定性**。

**一条命令跑完**（推荐）：

```bash
cd /opt/its/z30055003/cp_balance
export CP_BALANCE_FAMILY=a3 CP_BALANCE_LOCAL_IP=7.246.78.75 CP_BALANCE_NIC_NAME=eth2
bash verify.sh --family a3        # 前置 -> 静态 -> 冒烟 -> 分支诊断 -> 打包（判据来自 harness.json）

# 只跑某一层（底层入口）：
bash tests/run_tests.sh --tag fast                            # 静态 + 环境/配置/路径（秒级）
bash tests/run_tests.sh --only accuracy/a10_matrix_gate       # C 验收 + B 等价性（可选 A/B 现在恒 SKIP）
```

不需要人肉 `git -C vllm-ascend pull`：`verify.sh` 会按 `harness.json trees` 自动 clone/对齐/打补丁
（本地未 push 的 commit 会被 reset 掉）。

产物在 `tests/_out/<时间戳>/`：`status.tsv` / `results.json` 是总账，每个测试一个子目录
（`accuracy/a10_matrix_gate.<变体>/matrix/` 放该矩阵的 `summary.txt`、`*.log`、`*.json`、
`cmp_*.txt`）。
STEP 3（可选 A/B，`accuracy/a30_slot_filter_ab`）现状恒 SKIP：`round2_verify.py` 的补丁锚点随源码失效，
已定案不修（`scripts_review.md` §7-4）。所以它不会改源码，也不会产生 `03_optional/`。

---

## 0. 这一轮验证什么（5 条断言）

本轮的代码改动是"删无读者字段 / 去死别名 / 合并重复分支"（5 文件，净 −43 行），理论上不碰数值路径；
远端这一轮的目的**不是再证明一遍算法**，而是给这批删改上一个回归网，同时把两个便宜的开放项
在同一个机时里关掉。

| # | 断言 | 由哪一步证明 | 判据 |
| --- | --- | --- | --- |
| 1 | 删掉的 5 个 `ZigzagPlan` 字段确实无读者（没有 `getattr` / 反射 / 外部脚本在用） | §0 静态门控 + §1 起服务不报错 | `fields=13` 且服务能起、能出 token |
| 2 | 合并 topk/SFA 分支后，zigzag 与连续两条路径的 kernel 入参完全一致 | §1 C 验收 | `first-token match: 40/40` |
| 3 | `_indexer_qk_proj` 的 `output_dtype` 改成必填后没有第二处调用点（含 DCP 子类） | §1（zigzag 路径）+ §2（连续路径） | 无 `TypeError`，两侧都 PASS |
| 4 | `_pad_for_sequence_parallelism` 与 forward-context 写入的内联改动对所有 forward 无影响 | §2 B 等价性 + 噪声地板 | `b_vs_base` 与 `noise_floor` 都 PASS |
| 5 | zigzag 的 KV 写依赖 `slot == -1` 被 scatter 跳过 | `accuracy/a30_slot_filter_ab` 已恒 SKIP（锚点失效） | 不再执行 |

不做（别重复花机时）：C1/C2/C4 尚未落地；profiling / H0 已有结论（见 §7）。

前置（1 分钟）：

```bash
cd /opt/its/z30055003/cp_balance
git -C /opt/its/z30055003/vllm-ascend log -1 --format='%h %s'     # 记下 HEAD，回传要带这一行
python3 perf/check_cp_balance_fields.py --repo /opt/its/z30055003/vllm-ascend
python3 accuracy/check_b_path.py --repo /opt/its/z30055003/vllm-ascend \
        --base-repo /opt/its/z30055003/vllm-ascend-base
```

判据：两条都打印 `RESULT: PASS`；`check_cp_balance_fields` 会打印
`ZigzagPlan fields=13, ZigzagCPPlan fields=13, DSACPContext fields=19`（只有 ZigzagPlan=13 受 `harness.json expect.zigzag_plan_fields` 门控；
`check_b_path` 是 8/8）。

---

## 1. C 验收（最紧要：本轮改动后的第一件事）

```bash
bash tests/run_tests.sh --only accuracy/a10_matrix_gate#c_matrix
```

内部依次起停 `glm52_cur_cp1` / `glm52_cur_cp0` 两个服务，各采 40 条 prompt，比较首 token。

判据：末行 `RESULT: PASS`，且 `c_vs_b` 的 `first-token match: 40/40`；矩阵还带 `require_log` 门
（cp1 日志要有 zigzag 证据、cp0 不许出现 `[CP_BALANCE][plan]`），空跑会直接 FAIL。

加强版（把前 120 字符文本也变成硬判据，矩阵里 `require_text` 目前是 `false`）：
把 `configs/matrix_c_accept.json` 复制成 `configs/matrix_c_accept_text.json`，把
`"require_text": false` 改成 `true`，再跑一次同一条命令。判据多一条
`text_head match: 40/40`。

---

## 2. B 等价性 + 噪声地板（改动的兜底）

```bash
bash tests/run_tests.sh --only accuracy/a10_matrix_gate#b_matrix
```

该矩阵自带 `static_check`，会先跑 §0 的两条静态门控；然后跑 `cur_cp0` → `base_cp0` →
`base_cp0_repeat` 三组。

判据：末行 `RESULT: PASS`，其中
`b_vs_base`（当前树 CP_BALANCE=0 vs base）与 `noise_floor`（base 重复两次）都要 PASS。
噪声地板 FAIL 说明测量本身不可复现，先解决测量再谈代码。

---

## 3. 分支自证（zigzag 真的在跑）

```bash
VLLM_ASCEND_CP_BALANCE=1 VLLM_ASCEND_CP_BALANCE_DEBUG=1 PYTHONUNBUFFERED=1 \
    bash run.sh glm52_cur_cp1 > cp_on.log 2>&1 &
python3 accuracy/check_branch.py --url http://127.0.0.1:8035 --log cp_on.log
```

判据：末行 `[check] RESULT: PASS`；短 prompt 只有 `branch=CONTINUOUS`，长 prompt 至少一条 zigzag 证据
（`branch=ZIGZAG` 或 `[CP_BALANCE][plan]`）。长 prompt 的 token 数不到 `min_tokens` 时脚本输出
`RESULT: INCONCLUSIVE`（测试记 SKIP），那是 prompt 阶梯问题。
另：两棵树都带 dp=1 的门修之后，启动日志应是 `DSA-CP is enabled without sequence-parallel MoE`；
看到 `Disabling DSA-CP` 就说明树不对。

注意：`DEBUG=1` 会让 `[CP_BALANCE][plan]` 调 `.tolist()`（D2H 同步），**只用于证明分支**，
不要在 profiling 轮里打开（`configs/_profile_common.json` 已经是 `debug: 0`）。

---

## 4. 开放项 A3：EP 域 == TP 域（已作废）

（2026-09-23 起 `reduce_mode` 及其 A/B 已随 zigzag 改法删除，见 `docs/dp1_zigzag_acc_fix.md`。）

---

## 5. 开放项 A4：`slot == -1` 是否被 scatter 跳过（KV 写的唯一新增依赖）

背景：连续切片路径的 KV 写只写 `[:num_actual_tokens]`（不含 padding 行），zigzag 路径改成写
**全 padded 行 + slot=-1**（`attention/sfa_v1.py` 的 `_maybe_store_kvcache_for_c8_n_dsacp` 与
indexer cache write 两处）。若某版本把 -1 当成"最后一个 slot"，被覆盖的就是最后一个真实 token 的 KV。

### 5a. 代码内 A/B（推荐，10 分钟）

临时在 `attention/sfa_v1.py` 的两处 scatter 前过滤掉 padding 行：

```python
# _maybe_store_kvcache_for_c8_n_dsacp 内
scatter_slots = dsa_cp_context.slot_mapping_cp_gathered
fused_kv_actual = fused_kv_no_split
keep = scatter_slots >= 0                     # 临时 A/B：只写真实行
scatter_slots = scatter_slots[keep]
fused_kv_actual = fused_kv_actual[keep]
```

indexer 那处同理（`idx_slots` / `k_li` / `k_li_scale` 用同一个 `keep` 索引；
注意 `k_li` 可能是从 `fused_kv_no_split` 切出来的全量，切完再过滤）。

判据：过滤版与未过滤版**首 token 相同**（`compare_first_token.py compare` PASS）
→ -1 确实被跳过，可以安全依赖；若不同，说明 -1 被写进了有效 slot，必须改成掩码 scatter。

### 5b. 不改代码（对比 KV）

`VLLM_ASCEND_CP_BALANCE_DEBUG=1` 跑一条长 prompt，确认 `[CP_BALANCE][plan] ... slots=[...]` 有 -1；
再用 `perf/` 那套 dump 工具比较该请求最后几个真实 token 的 KV 与 CP_BALANCE=0 的结果。

判据：最后一个真实 token 的 KV 两侧一致。

---

## 6. 开放项 A1/A2 的补丁（已作废）

（2026-09-23 起 `reduce_mode` 及其 A/B 已随 zigzag 改法删除，见 `docs/dp1_zigzag_acc_fix.md`。）

---

## 7. 若要继续 profiling：先把可判定性补上

上一轮 profiling 的证据（`data/send/export/*op_statistic.csv`，rank0）：

| 轮 | reduce_scatter | allreduce | 备注 |
| --- | --- | --- | --- |
| `cur_cp0` | 4936 次 / 10.13 s | 0 | 与 `base_cp0` 条数完全相同 |
| `cur_cp1` | 4920 次（−16）/ 11.47 s | 16 次 / 12.2 ms | 同一条 RS 自身慢 13% |

结论口径：16 次归约在窗口里只值 12 ms（0.018%），而同一算子跨轮慢 13% → **轮间漂移比效应大两个数量级**，
不要用这一轮数据改默认值。若还要测：

1. 每个配置至少 2 轮，只比较**同一轮内**的差值；
2. 先核对"次数"（zigzag 只改 attention 内部的选行/放回，两侧的集合通信构成应一致），次数不对说明配置没生效；
3. profiling 轮保持 `debug=0`；`trace_view.json` 别整文件加载（上一轮 3.1 GB 直接解析失败），
   用 `perf/profile_order.py --trim` 或 6 层配置。

---

## 8. 回传的最小集合

| 文件 / 行 | 用途 |
| --- | --- |
| `git -C /opt/its/z30055003/vllm-ascend log -1` 输出 | 验收对应的 HEAD |
| `matrix_*/summary.txt` | C 验收 / B 等价性裁定 |
| `cp_on.log` 前 200 行（含 `[cp_balance] REPO=` 与 `additional_config` 指纹） | 确认代码树与开关 |
| `[CP_BALANCE][branch]` 行 | 证明走的是 ZIGZAG 还是 CONTINUOUS 及原因 |
| §5 过滤版/未过滤版的首 token JSON | A4 结论 |
| `prof_*/summary.json` + `export/` | 若继续 H0 |
