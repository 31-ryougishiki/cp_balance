# cp_balance 远端验证清单（本轮精简之后）

配套文档：`docs/cp_balance_review_round2.md`（问题与精简清单）。本清单只回答两件事：
**改完之后怎么验收**、**哪些开放项必须靠远端定性**。

**一条命令跑完**（推荐；它内部就是下面这些步骤）：

```bash
cd /opt/its/z30055003/cp_balance
git -C /opt/its/z30055003/vllm-ascend pull      # 先拉到本轮精简后的提交
bash accuracy/round2_verify.sh                  # 全跑：静态 + C 验收 + B 等价性 + 可选 A/B
bash accuracy/round2_verify.sh --steps 0,1,2    # 跳过可选 A/B（省两次服务起停）
bash accuracy/round2_verify.sh --dry-run        # 只打印将要做什么，不启动任何东西
```

产物在 `round2_<时间戳>/`：`summary.txt` 有每条判定和要回传的文件清单，`01_c_accept/`、
`02_b_equiv/` 是两次矩阵的产物，`03_optional/` 放可选 A/B 的日志与对比。
STEP 3 会临时改源码、结束时按字节还原；若中途被强杀，`03_optional/PATCH_APPLIED.flag` 会提示，
用 `git -C <repo> checkout -- vllm_ascend/ascend_forward_context.py vllm_ascend/attention/sfa_v1.py`
恢复（或从同目录的 `*.round2bak` 拷回）。

---

## 0. 这一轮验证什么（6 条断言）

本轮的代码改动是"删无读者字段 / 去死别名 / 合并重复分支"（5 文件，净 −43 行），理论上不碰数值路径；
远端这一轮的目的**不是再证明一遍算法**，而是给这批删改上一个回归网，同时把两个便宜的开放项
在同一个机时里关掉。

| # | 断言 | 由哪一步证明 | 判据 |
| --- | --- | --- | --- |
| 1 | 删掉的 5 个 `ZigzagPlan` 字段确实无读者（没有 `getattr` / 反射 / 外部脚本在用） | §0 静态门控 + §1 起服务不报错 | `fields=13` 且服务能起、能出 token |
| 2 | 合并 topk/SFA 分支后，zigzag 与连续两条路径的 kernel 入参完全一致 | §1 C 验收 | `first-token match: 40/40` |
| 3 | `_indexer_qk_proj` 的 `output_dtype` 改成必填后没有第二处调用点（含 DCP 子类） | §1（zigzag 路径）+ §2（连续路径） | 无 `TypeError`，两侧都 PASS |
| 4 | `_pad_for_sequence_parallelism` 与 forward-context 写入的内联改动对所有 forward 无影响 | §2 B 等价性 + 噪声地板 | `b_vs_base` 与 `noise_floor` 都 PASS |
| 5 | 固定序归约用的 TP 域与 MoE 的 EP 域同域（隐式前提） | §4（加一行 `info_once`） | `tp=16/0 ep=16/0` |
| 6 | zigzag 的 KV 写依赖 `slot == -1` 被 scatter 跳过（本轮唯一新增假设） | §5a（过滤 `slot>=0` 的 A/B） | 两版首 token 相同 |

不做（别重复花机时）：C1/C2/C4 尚未落地；profiling / H0 / H1 已有结论（见 §7）；
`VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL` 默认关且与本轮改动无关，要决定留删另开一轮。

前置（1 分钟）：

```bash
cd /opt/its/z30055003/cp_balance
git -C /opt/its/z30055003/vllm-ascend log -1 --format='%h %s'     # 记下 HEAD，回传要带这一行
python3 perf/check_cp_balance_fields.py --repo /opt/its/z30055003/vllm-ascend
python3 accuracy/check_b_path.py --repo /opt/its/z30055003/vllm-ascend \
        --base-repo /opt/its/z30055003/vllm-ascend-base
```

判据：两条都打印 `RESULT: PASS`；`check_cp_balance_fields` 这次应该打印
`ZigzagPlan fields=13, DSACPContext fields=18`（删掉 5 个无读者字段后的新值，旧值是 18）。

---

## 1. C 验收（最紧要：本轮改动后的第一件事）

```bash
bash accuracy/run_matrix.sh configs/matrix_c_accept.json
```

内部依次起停 `glm52_cur_cp1` / `glm52_cur_cp0` 两个服务，各采 40 条 prompt，比较首 token。

判据：末行 `RESULT: PASS`，且 `c_vs_b` 的 `first-token match: 40/40`。

加强版（把前 120 字符文本也变成硬判据，矩阵里 `require_text` 目前是 `false`）：
把 `configs/matrix_c_accept.json` 复制成 `configs/matrix_c_accept_text.json`，把
`"require_text": false` 改成 `true`，再跑一次同一条命令。判据多一条
`text_head match: 40/40`。

---

## 2. B 等价性 + 噪声地板（改动的兜底）

```bash
bash accuracy/run_matrix.sh configs/matrix_b_vs_base.json
```

该矩阵自带 `static_check`，会先跑 §0 的两条静态门控；然后跑 `cur_cp0` → `base_cp0` →
`base_cp0_repeat` 三组。

判据：末行 `RESULT: PASS`，其中
`b_vs_base`（当前树 CP_BALANCE=0 vs base）与 `noise_floor`（base 重复两次）都要 PASS。
噪声地板 FAIL 说明测量本身不可复现，先解决测量再谈代码。

日志侧同时核对（证明 CP_BALANCE=0 走的还是原集合通信）：

```bash
grep -c '\[CP_BALANCE\]\[reduce\] path=native' matrix_*/cur_cp0.log   # > 0
grep -c 'path=fixed_order' matrix_*/cur_cp0.log                          # == 0
```

---

## 3. 分支自证（zigzag 真的在跑）

```bash
VLLM_ASCEND_CP_BALANCE=1 VLLM_ASCEND_CP_BALANCE_DEBUG=1 PYTHONUNBUFFERED=1 \
    bash run.sh glm52_cur_cp1 > cp_on.log 2>&1 &
python3 accuracy/check_branch.py --url http://127.0.0.1:8035 --log cp_on.log
```

判据：末行 `[check] RESULT: PASS`；短 prompt 只有 `branch=CONTINUOUS`，长 prompt 至少一条
`branch=ZIGZAG`。

注意：`DEBUG=1` 会让 `[CP_BALANCE][plan]` 调 `.tolist()`（D2H 同步），**只用于证明分支**，
不要在 profiling 轮里打开（`configs/_profile_common.json` 已经是 `debug: 0`）。

---

## 4. 开放项 A3：EP 域 == TP 域（建议顺手落地的加固）

背景：固定序归约（`ops/register_custom_ops.py:38`）用 TP 域，而 MoE 的 dispatch/all-gather 走 EP 域；
DP=1 时两者同域，但代码里没有任何断言（见报告 §2.6-4）。

临时验证补丁：`vllm_ascend/ascend_forward_context.py`，在
`forward_context.zigzag_cp_active = zigzag_cp_active` 之后加：

```python
if zigzag_cp_active:
    from vllm.distributed.parallel_state import get_ep_group

    tp_group, ep_group = get_tp_group(), get_ep_group()
    logger.info_once(
        "[CP_BALANCE][group] tp=%d/%d ep=%d/%d",
        tp_group.world_size, tp_group.rank_in_group,
        ep_group.world_size, ep_group.rank_in_group,
    )
```

判据：日志里出现 `[CP_BALANCE][group] tp=16/0 ep=16/0`（16 个 rank 上 rank 值随 rank 递增）。
若 EP 域的 world_size/rank 与 TP 不同 → 固定序归约必须换成 EP 域（否则"第几段"与聚合序对不上）。
验证过就可以保留这行 `info_once`（成本：每进程一次）。

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

## 6. 开放项 A1/A2 的补丁（若本轮决定修）

### A1 就地 AllReduce 与 `mutates_args=[]` 冲突

二选一（报告 §2.5-1）：

```diff
--- a/vllm_ascend/distributed/utils.py
+++ b/vllm_ascend/distributed/utils.py
@@ -48,7 +48,7 @@ def _allreduce_slice_reduce_scatter(tensor, group):
     summed = tensor if tensor.is_contiguous() else tensor.contiguous()
     dist.all_reduce(summed, group=group.device_group)
     rank = int(group.rank_in_group)
-    return summed[rank * chunk : (rank + 1) * chunk].contiguous()
+    return summed[rank * chunk : (rank + 1) * chunk].clone()
```

或者显式声明契约（`ops/register_custom_ops.py` 的 `maybe_pad_and_reduce` 注册处）：
`mutates_args=[]` → `mutates_args=["x"]`（`matmul_and_reduce` 那处改的是层内新张量，
核对后决定是否一起改）。

验证：`matrix_b_vs_base` + `matrix_c_accept` 双 PASS；另外记一笔：本仓库默认
`--enforce-eager`，A1 的风险只在 compile / ACL graph 场景暴露。

### A2 宽异常包住集合通信

把 `ops/linear_op.py:203-213` 与 `ops/register_custom_ops.py:34-48` 的 `try/except` 收窄：
`rows % world_size != 0`、group 缺失、mode 非法这些**前置条件**先判断并直接回退；
通信调用本身不要再被 `try` 包住（单 rank 抛错后改发另一种集合通信 = 挂死）。

验证：把 `VLLM_ASCEND_CP_BALANCE_REDUCE_MODE` 设成非法值，应当报 `ValueError` 而不是静默回落。

---

> A3 的 profiling 收集由 `perf/profile.sh` 在采集结束时**自动执行** `perf/collect.py`：清点 → 逐长度 clean_s → compare/order 文本 → 指纹 → 打包成 `collect_<时间戳>/*.tgz`（不含原始 trace）；也可单独跑 `bash perf/collect.sh`。

## 7. 若要继续 profiling：先把可判定性补上

上一轮四轮 profiling 的证据（`data/send/export/*op_statistic.csv`，rank0）：

| 轮 | reduce_scatter | allreduce | 备注 |
| --- | --- | --- | --- |
| `cur_cp0` | 4936 次 / 10.13 s | 0 | 与 `base_cp0` 条数完全相同 |
| `cur_cp1` | 4920 次（−16）/ 11.47 s | 16 次 / 12.2 ms | 同一条 RS 自身慢 13% |
| `cur_cp1_a2a` | 4920 次（−16）/ 10.83 s | alltoall +16 | |

结论口径：16 次归约在窗口里只值 12 ms（0.018%），而同一算子跨轮慢 13% → **轮间漂移比效应大两个数量级**，
不要用这一轮数据改 `REDUCE_MODE` 默认值。若还要测：

1. 每个配置至少 2 轮，只比较**同一轮内**的差值；
2. 先核对"次数"（zigzag 相对非 zigzag 应恰好 −16 RS / +16 AR，a2a 模式是 +16 alltoall），次数不对说明配置没生效；
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
| `[CP_BALANCE][group]` 行（§4 加的） | A3 结论 |
| §5 过滤版/未过滤版的首 token JSON | A4 结论 |
| `prof_*/summary.json` + `export/` | 若继续 H0/H1 |
