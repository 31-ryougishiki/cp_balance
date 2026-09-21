# cp_balance 验证路线（A5 节点：精度 + 性能）

本文只讲"接下来怎么验"。当前状态与未决问题见 `scripts_review.md`；远端通用跑法见 `remote_run.md`。
默认机器族是 a5（`verify.sh` / `tests/run_tests.sh` / `common.sh` / `configs/default.json` 都以 A5 为默认）。

## 0. 前置（一次做完，10 分钟）

```bash
cd <harness>            # /home/z30055003/cp_balance 一类
git pull                # harness、被测树(vllm-ascend@cp_balance)都要先 push 过
export CP_BALANCE_LOCAL_IP=<本机 IP> CP_BALANCE_NIC_NAME=<网卡>   # 不设则自动识别（配置默认 auto）
bash verify.sh --family a5            # 默认：会拉起模型（smoke 阶段两次起停）
bash verify.sh --family a5 --live-log # 同上，另外把测试与模型服务日志实时打屏
# 只想看日志、不起服务：--skip-smoke（跳过冒烟）或 --diag-only（只对已有证据出诊断）
# 服务整体那一段单独跳过：--skip-service
# 日志链路自检（假服务，20 秒，不占 NPU）：bash tests/run_tests.sh --only smoke/s08_service_log_wiring --live-log
```

期望：

- 前置阶段把参照树对齐到 fork 的 `base-dp1` 分支（= main + dp=1 门修），日志里有 `[sync] vllm-ascend-base 已在 base-dp1@...`；
- `--live-log` 打开后，smoke 阶段的服务日志会实时打屏（同一份内容照旧写文件）；默认不上屏、只写文件，
  等就绪期间每 `limits.ready_note_seconds`（默认 60s）打一行进度（含日志尾行）；起不来时带日志路径、大小和尾部，
  日志是空的话会直接说"日志是空的"（不是被 harness 丢掉）；
- 日志路径：每个 stage 打一行 `[note] stage <id> 日志：...`（run_tests 的测试输出）；服务日志在
  `tests/_out/<stamp>_<family>/<测试 id>/<配置名>.log`（例：`.../smoke/s10_service_ready.svc_cp1/glm52_a5_cur_cp1.log`），
  也可以另开终端 `tail -f`，或直接 `bash verify.sh --live-log` 让它上屏；
- 服务日志出现 `DSA-CP is enabled without sequence-parallel MoE (data_parallel_size=1)`，**不是** `Disabling DSA-CP`；
- 诊断 `[diagnose] VERDICT=OK ...`（命中 `branch=ZIGZAG` 或 `[CP_BALANCE][plan]`）；
- 产物：`verify_a5_<stamp>.tar.gz`（报告 + 这一轮 `tests/_out/<stamp>_a5/`）。

诊断是 `VERDICT=KNOWN Disabling DSA-CP` → 说明树没带 dp=1 的门修或补丁没打上，先修树，别往下走。

若服务日志显示已经起来、测试却一直不往下走：先看环境里有没有 `http_proxy`（远端常见，且没有 `no_proxy`）。
这会把 `curl 127.0.0.1:<port>/v1/models` 的就绪探测发给代理，测试就一直等到超时。
harness 已经在 `common.sh` / `verify.sh` / `serve_config.py` 里给本机地址开了白名单（`127.0.0.1,localhost,::1`），
手工 curl 时记得加 `--noproxy '*'`。

若前置报缺 `vllm_ascend/_build_info.py`：那是 `setup.py` 生成的、只跟芯片型号有关的一行文件（不是编译产物），
**只改过 py 不需要重编译**。`CP_BALANCE_AUTO_BUILD=copy` 可从同芯片的兄弟树拷过来，或者手工 `cp`；
只有新 clone 才需要在该树跑一次 `pip install -e . --no-build-isolation`（顺带生成 C 扩展）。

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

## 3. 阶段 3：服务整体（cp1/cp0 各一次并发突发，约 45 分钟）

```bash
bash tests/run_tests.sh --only service --skip service/sv30_soak,service/sv40_lifecycle --keep-going
```

判据：sv10 全过（契约与 4xx 错误路径）；sv20#cp1 与 #cp0 的 `[check] RESULT: PASS`（零失败 + 同 prompt 回复逐字一致）；
sv21 的 `RESULT: PASS`（每个 batch 的 pad/actual/local 在 cp_size 个 rank 上一致、无 plan_error、至少一个 step 多序列）；
sv22 的 `text match: N/N`（cp1 与 cp0 逐字相同）。sv21/sv22 报 SKIP 时先看 sv20 是不是没跑或没出计划行。

与阶段 2 的关系：阶段 2 一次只发一条请求、只看首 token；这一段是并发突发，看的是整批的 rank 一致性与整段文本。
顺序上先跑阶段 2 可以少一次起停，但两者是独立判据。

回传：`tests/_out/<时间戳>_<family>/service/**`（sv20 的 results.json / check.txt / service.log、sv21 的 plan_*.txt 与 json、sv22 的 compare.txt）。

重活（可选，各一次起停）：`--only service/sv30_soak`（长跑 + RSS/显存采样）、`--only service/sv40_lifecycle`（在飞停服 + 重启）。

## 4. 阶段 4：性能（4 组 ≈ 3.5 小时 + 离线解析 ≈ 20 分钟）

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

## 5. 回传什么

- 一键验证：`verify_a5_*.tar.gz`（自带上一步的整个 `tests/_out/<stamp>_a5/`）；
- 性能：`collect_<时间戳>/*.tgz`（含 `summary.json`、`windows.json`、`order_rank0.json`、`export/`），原始 trace（GB 级）不拷；
- 失败时另外带：`<id>.log`（合并日志）+ 诊断报告 `<PREFIX>_a5_branch_<stamp>.txt`。

## 6. 判定汇总

| 阶段 | 命令 | 通过判据 |
| --- | --- | --- |
| 0 前置 | `bash verify.sh --family a5` | 无 `Disabling DSA-CP`；诊断 VERDICT=OK |
| 1 B 等价性 | `... --only accuracy/a10_matrix_gate#b_matrix` | `b_vs_base` 与 `noise_floor` 都 40/40 |
| 2 C 验收 | `... --only accuracy/a10_matrix_gate#c_matrix` | `require_log` 通过 + `c_vs_b` 40/40 |
| 3 性能 | `perf/p10_capture` + `p20~p24` | 差异大于噪声地板，且集合通信次数与配置一致 |
