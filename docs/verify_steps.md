# cp_balance 验证路线（A5 节点：精度 + 性能）

本文只讲"接下来怎么验"。当前状态与未决问题见 `scripts_review.md`；远端通用跑法见 `remote_run.md`。
默认机器族是 a5（`verify.sh` / `tests/run_tests.sh` / `common.sh` / `configs/default.json` 都以 A5 为默认）。

## 0. 前置（一次做完，10 分钟）

```bash
cd <harness>            # /home/z30055003/cp_balance 一类
git pull                # harness、被测树(vllm-ascend@cp_balance)都要先 push 过
export CP_BALANCE_LOCAL_IP=<本机 IP> CP_BALANCE_NIC_NAME=<网卡>   # 不设则自动识别（配置默认 auto）
bash verify.sh --family a5
```

期望：

- 前置阶段打印 `[sync] base 打上补丁 dsa_cp_dp1.patch`（参照树 = main + dp=1 门补丁）；
- 服务日志出现 `DSA-CP is enabled without sequence-parallel MoE (data_parallel_size=1)`，**不是** `Disabling DSA-CP`；
- 诊断 `[diagnose] VERDICT=OK ...`（命中 `branch=ZIGZAG` 或 `[CP_BALANCE][plan]`）；
- 产物：`verify_a5_<stamp>.tar.gz`（报告 + 这一轮 `tests/_out/<stamp>_a5/`）。

诊断是 `VERDICT=KNOWN Disabling DSA-CP` → 说明树没带 dp=1 的门修或补丁没打上，先修树，别往下走。

## 1. 阶段 1：先钉死"DSA-CP 在 dp=1 成立"（cp_balance=0，2 次起停，约 25 分钟）

```bash
bash tests/run_tests.sh --only accuracy/a10_matrix_gate#b_matrix   # cur_cp0 vs base_cp0 + 噪声地板
```

判据：`b_vs_base` 与 `noise_floor` 都 `first-token match: 40/40`（矩阵里 `require_text` 打开时还要 text_head 40/40）。
这一步跑的是连续切片路径，本来就该与参照树一致；失败先查环境（权重/网卡/确定性变量/树版本），不要怀疑 cp_balance。

## 2. 阶段 2：精度 C 验收（cp_balance=1 vs 0，2 次起停，约 25 分钟）

```bash
bash tests/run_tests.sh --only accuracy/a10_matrix_gate#c_matrix
```

两道门：

1. `require_log`：cp1 的服务日志必须出现 zigzag 证据（`branch=ZIGZAG` 或 `[CP_BALANCE][plan]`），cp0 不许出现 `[CP_BALANCE][plan]`；
2. `c_vs_b` 的 `first-token match: 40/40`（要更严就把矩阵的 `require_text` 改成 true，再加 120 字符文本比对）。

三种结果怎么读：

- **PASS**：zigzag 与连续切片数值等价，可以进性能阶段；
- **FAIL 且 cp1 有 zigzag 证据**：当前最可能是已知的 zigzag MoE 布局问题（`scripts_review.md` §8：层内 SP 关闭时 MoE 需要全量行，
  而 zigzag 给的是切片行）。回传：`[CP_BALANCE][plan]` 前后 50 行、诊断报告里的 `moe_comm_type/MC2` 行、`cmp_c_vs_b.txt`，
  再决定三条出路（(a) dp=1 显式不开 zigzag /(b) dp=1 也开 SP /(c) 自己补层内 gather）；
- **FAIL 且 cp1 没有证据**：门/配置没生效（回到阶段 0 的诊断）。

顺带：MTP 的 draft 步会打 `branch=CONTINUOUS reason=draft`，出现它是正常的（显式守卫）。

## 3. 阶段 3：性能（4 组 ≈ 3.5 小时 + 离线解析 ≈ 20 分钟）

```bash
bash tests/run_tests.sh --only perf/p10_capture --skip perf/p10_capture#prof_a2a
bash tests/run_tests.sh --only perf/p20_analyse,perf/p21_window_single_step,perf/p22_report_compare,perf/p23_report_order,perf/p24_collect --keep-going
```

四组：`prof_cp0`（当前树、关）、`prof_cp1`（当前树、开）、`prof_cp0_repeat`（噪声地板）、`prof_base`（参照树）。
每组按 `lengths`（1024~12288 token）逐档采一个只含 prefill 步的窗口，并补跑一次不采样的同请求拿 `clean_s`。

判读顺序（别发明比值）：

1. 先看噪声地板：`cp0 vs cp0_repeat` 的漂移有多大，小于它的差异不下结论；
2. 再核次数：集合通信次数/归约次数是否与配置一致（`p22` 的对比表）；
3. 最后看绝对量：`clean_s` 与逐长度曲线；
4. 归因：`p23_report_order`（顺序 + device kernel），已 prune 原始 trace 时它会 SKIP。

两个要知道的偏差：

- A5 的性能配置 `deterministic=false`（服务配置是 true），结论里要注明"性能数字不是服务同款设置"；
- `p21_window_single_step` 当前恒 SKIP（trace 里 `kernel_details.csv` 没有 Step 列），`p24_collect` 只断言"有 tgz"。

## 4. 回传什么

- 一键验证：`verify_a5_*.tar.gz`（自带上一步的整个 `tests/_out/<stamp>_a5/`）；
- 性能：`collect_<时间戳>/*.tgz`（含 `summary.json`、`windows.json`、`order_rank0.json`、`export/`），原始 trace（GB 级）不拷；
- 失败时另外带：`<id>.log`（合并日志）+ 诊断报告 `<PREFIX>_a5_branch_<stamp>.txt`。

## 5. 判定汇总

| 阶段 | 命令 | 通过判据 |
| --- | --- | --- |
| 0 前置 | `bash verify.sh --family a5` | 无 `Disabling DSA-CP`；诊断 VERDICT=OK |
| 1 B 等价性 | `... --only accuracy/a10_matrix_gate#b_matrix` | `b_vs_base` 与 `noise_floor` 都 40/40 |
| 2 C 验收 | `... --only accuracy/a10_matrix_gate#c_matrix` | `require_log` 通过 + `c_vs_b` 40/40 |
| 3 性能 | `perf/p10_capture` + `p20~p24` | 差异大于噪声地板，且集合通信次数与配置一致 |
