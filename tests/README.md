# tests/ —— 全部测试的目录与统一入口

一个测试 = 一个可独立执行的脚本；`tests/run_tests.sh` 只做发现、筛选、按顺序调用、汇总。
一次 run_tests.sh 就是一轮验收，不再保留老的聚合入口（`round2_verify*.sh`、`perf/profile*.sh` 已删）。

## 目录

| 目录 | 语义 | 起服务 |
| --- | --- | --- |
| `tests/smoke/` | 前置检查：环境/配置解析/路径/端口/磁盘 + 静态门控 + 服务能否起来 + 请求是否走 ZIGZAG | s10/s11 各一次 |
| `tests/accuracy/` | 精度验收：矩阵首 token 比对（C/B/nomtp）、复用已采 json 的对比、slot<0 补丁 A/B | 每个矩阵一次 |
| `tests/perf/` | 性能：逐配置采集 → 解析 → 单步窗口审计 → 对比/顺序报告 → 打包 | 采集起一次 |
| `tests/service/` | 服务整体：API 契约与错误路径、并发混合突发的回复一致性、本轮 batch 的 zigzag 计划在所有 rank 上是否一致、长跑稳定性、停服/重启 | sv10/sv20/sv30/sv40 |
| `tests/lib/` | 共用：`common.sh`（断言/服务起停）、`roles.tsv`（机器族角色表）、`hx.py`（配置/指纹/路径/端口/窗口）、`loadgen.py`（并发流量 + /metrics）、`planlog.py`（[CP_BALANCE] 日志审计） | — |

## 用法

```bash
bash tests/run_tests.sh --list                    # 全部测试（id/needs/tags/est/desc）
bash tests/run_tests.sh --tag fast                # 不起服务的那批
bash tests/run_tests.sh --only smoke/s10_service_ready
bash tests/run_tests.sh --only perf               # 目录名也行：perf = perf/ 下全部
bash tests/run_tests.sh --from accuracy           # 从 accuracy 开始（失败后用 --from <id> 续跑）
bash tests/run_tests.sh --keep-going              # 默认第一个 FAIL 就停
bash tests/run_tests.sh --only smoke/s10_service_ready --live-log   # 测试与模型服务日志实时打屏（env：HX_LIVE_LOG=1）
CP_BALANCE_FAMILY=a3 bash tests/run_tests.sh --tag fast
bash tests/run_tests.sh --only service        # 服务整体：契约 + 并发突发 + 计划审计 + cp0/cp1 一致性
```

- 退出码：0 全过；1 有 FAIL（`--strict` 时 SKIP 也算失败）；2 用法错/没发现测试/选择器一个都没命中。
  每个测试自己用 0=PASS、77=SKIP、其它=FAIL 回报。
- `--only` / `--skip` / `--from` 的写法：完整 id、`id#variant`、脚本名、目录名都行。
- 产物：`tests/_out/<时间戳>/` —— 每个测试一个子目录（自己的证据）+ `status.tsv` / `results.json` / `<id>.log`。
- 单独跑：`bash tests/perf/p10_capture.sh prof_cp1`（位置参数就是变体，证据写进 `<名字>.<变体>/`）。

## 配置（harness.json）

harness 级的数据只放在根目录 `harness.json`：

| 段 | 内容 |
| --- | --- |
| `trees` / `model_candidates` | 目标代码树（path/remote/ref/kind/required/patches）与权重候选；`tests/lib/trees.py` 读它 |
| `families` | 机器族：chip 匹配、svc 角色、展示名（`tests/smoke/s01` 用它判断 family 与芯片是否一致） |
| `limits` | 服务就绪/停止重试、磁盘下限、ready_timeout、profiling 默认值（`hx.py limit <名>`） |
| `service` | 服务级测试：流量计划（`configs/plans/*.json`）、长跑时长与采样间隔、阈值（preemption/KV/RSS） |
| `serve` | launcher 默认（vllm 可执行、PYTHONUNBUFFERED、deterministic_env 兜底） |
| `verify` | 一键验证的 stages（id/skip_flag/run）、`{out}` 占位符、诊断判据与已知失败 |
| `expect` | 静态门控的期望值（如 ZigzagPlan 字段数） |

`trees.<角色>.patches` 是树对齐后要打的补丁（`patches/*.patch`，相对 harness 根目录，幂等）：
参照树要跟被测树跑在同一代 DSA-CP 语义上时可以用它（当前没人用：参照树的 dp=1 门修提交在 `base-dp1` 分支上）。

每条服务/矩阵配置仍然是 `configs/*.json`（模型/端口/开关/profiler），`repo` 不写路径而是
`repo_tree: cur|base`，由 `harness.json trees` 解析。

## 机器族与角色表

测试不写死 IP/网卡/路径，配置名由 `tests/lib/roles.tsv` 一行一个角色给出：
`family / group / role / config`，config 为 `-` 表示本族没有这个角色（测试据此 SKIP 而不是 FAIL）。

- `hx_role <role>` 查本族的配置；`hx_group <group>` 列一组配置（group = `svc` / `matrix` / `driver` / `prof`）；
- 机器身份仍走 harness 自己的环境变量 `CP_BALANCE_LOCAL_IP` / `CP_BALANCE_NIC_NAME` / `CP_BALANCE_DEVICES`；
- 静态门控用的代码树可用 `CP_BALANCE_REPO` / `CP_BALANCE_BASE_REPO` 覆盖（默认取 `b_matrix` 的 `static_check`）。

## 环境变量

| 变量 | 作用 |
| --- | --- |
| `CP_BALANCE_FAMILY` | `a5` / `a3`，选角色表（非法值 runner 直接退出 2） |
| `HARNESS_OUT` | 产物目录（runner 默认 `tests/_out/<月日>_<时分秒>`） |
| `HX_READY_TRIES` / `HX_READY_SLEEP` | 服务就绪轮询（默认取 `harness.json limits.ready_tries` / `poll_seconds`） |
| `HX_MIN_FREE_GB` | `smoke/s05_disk` 的磁盘下限（默认取 `limits.min_free_gb`） |
| `CP_BALANCE_LOCAL_IP` / `CP_BALANCE_NIC_NAME` / `CP_BALANCE_DEVICES` | 机器身份，harness 自己读（测试不重复解析） |
| `CP_BALANCE_REPO` / `CP_BALANCE_BASE_REPO` | 静态门控使用的代码树 |

## 加一个新测试

脚本头部注释就是元数据，runner 直接读：

```bash
#!/usr/bin/env bash
# desc:     一句话说明
# needs:    none | service | profiler
# tags:     fast, offline            # fast=不起服务；npu=要用卡；slow=长时间；report=只出报告
# variants: prof_cp1 prof_cp0        # 可选：一个脚本展开成多个实例（runner 传 HX_VARIANT）
# est:      5min
set -uo pipefail
source "$(dirname "$0")/../lib/common.sh"
...
hx_end                              # 0=PASS；hx_skip -> 77=SKIP；其它=FAIL
```

上面 5 个字段都只是给 runner 用：`needs` 目前只打印、不做门禁，`est` 是估算、没有超时。
`tags` 现有的取值：`fast` / `offline` / `static` / `npu` / `slow` / `service` /
`accuracy` / `perf` / `report` / `variant`。

规则：

- 断言用 `hx_ok / hx_fail / hx_warn / hx_skip`，证据写 `$HX_OUT/`；
- 服务用 `hx_service_up <config> <log>` / `hx_service_down <port>`；`common.sh` 装了 EXIT trap，
  测试中途失败或被打断也会收尾，不留后台进程；
- 脚本与配置文件一律 LF：CRLF 的脚本在 Linux 上会在 `set -o pipefail` / `source ...` 那两行直接失败；
- 产物落在仓库根目录的测试（`prof_*`、`matrix_*`）在测试之间是链式的，不是相互独立的：
  `p10_capture → p20_analyse → p21/p22/p23/p24` 依次吃上一步的产物，`a20_compare_collected`
  吃 `a10_matrix_gate` 采到的 json；缺上一步的产物时这些测试 SKIP 而不是 FAIL。

## 覆盖的验收内容

| 验收 | 跑法 |
| --- | --- |
| 静态门控（字段 / B 路径） | `--only smoke/s06_static_fields,smoke/s07_static_b_path` |
| C 验收 + B 等价性 + 噪声地板 | `--only accuracy/a10_matrix_gate`（A3 的 `#nomtp_matrix` 会 SKIP） |
| 复用已采 json 复跑对比 | `--only accuracy/a20_compare_collected` |
| 可选 A/B（slot<0 过滤） | `--only accuracy/a30_slot_filter_ab` |
| profiling 一轮（采集→解析→对比→归因→打包） | `--only perf` |
| 服务整体（契约 + 并发突发 + 计划审计 + cp0/cp1 一致性） | `--only service`（跳过重活：`--skip service/sv30_soak,service/sv40_lifecycle`） |
| 长跑稳定性 / 停服重启 | `--only service/sv30_soak` / `--only service/sv40_lifecycle` |

## 服务级（tests/service/）

精度与性能层都是「一条请求一条响应」的串行世界；服务层问的是另外几个问题：并发时一个 batch 里挤进多条序列还成不成立、
每个 batch 的 zigzag 计划在所有 rank 上是否一致、长跑会不会漂、停服与重启干不干净。

| id | 起服务 | 断言 |
| --- | --- | --- |
| `service/sv10_contract` | 1 次 | /health、/v1/models、/tokenize 正常；坏 JSON、未知名、超长 max_tokens 都返回 4xx；流式中途断开后服务仍健康且还能正常应答 |
| `service/sv20_traffic`（`#svc_load_cp1` / `#svc_load_cp0`） | 各 1 次 | 一次并发混合突发（长短混排 + 同 prompt 重复 + 一条 SSE）：全部成功、同 prompt 回复逐字一致、/metrics 的 corrupted=0 / preemption=0；产物 results.json 与 service.log 留给后面两个测试 |
| `service/sv21_plan_audit` | — | 读 sv20 的 service.log：每个 batch 形状在所有 rank 上的计划行数相等、每行满足 local*cp_size==pad 与 sum(qprev)+sum(qnext)==local、没有 plan_error、cp0 不许出现计划行，且至少一个 step 里有多条序列 |
| `service/sv22_batch_parity` | — | 同一条请求在 sv20#cp1 与 sv20#cp0 下的回复必须逐字相同，并报告两边的 wall/TTFT/ITL |
| `service/sv30_soak` | 1 次 | 持续 service.soak_minutes 分钟：零失败、corrupted/preemption 不涨、KV 使用率写证据、RSS 增长在 max_rss_growth_mb 内 |
| `service/sv40_lifecycle` | 2 次 | 有请求在飞时停服：端口释放、无残留进程，重启后仍能服务 |

- 流量计划在 `configs/plans/*.json`（不是服务配置，`hx.py configs` 不会把它们算成服务配置）；时长与阈值在 `harness.json` 的 `service` 段。
- sv21/sv22 是链式测试：吃 sv20 的产物（`$HARNESS_OUT/service/sv20_traffic.<角色>/`），缺产物就 SKIP。
- 「同 prompt 回复一致」是并发下的新判据：同一个 batch 里多条序列如果布局串了，重复请求会给出不同文本。
- 唯一的 per-rank 证据是 `[CP_BALANCE][plan]`（INFO 级、每 rank 每 step 一行）；`branch=CONTINUOUS` 是 `info_once`，只打一次，所以审计以 plan 行为准。
- cp1 与 cp0 用同一份流量计划，所以 sv22 的逐字比对就是并发场景下的 B/C 等价性；A3 没有 MTP，配置是 svc_a3_load_cp*。

## 分层的原因

服务起停约 10 分钟/次，一轮精度 = 5 次起停、一轮性能 = 4 组采集，失败要能定位到具体一层：

1. 不起服务的检查（配置/路径/端口/磁盘/静态门控）秒级给出结论，先跑它们；
2. 服务能否起来、请求是否走 zigzag 各一个测试，十几分钟定位集成问题；
3. 精度/性能是全量，放在最后，且是**独立可重跑**的：失败后 `--from <id>` 只补跑后面几步。
