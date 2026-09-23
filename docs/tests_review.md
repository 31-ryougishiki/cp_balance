> 历史文档（2026-09 上旬的交接/评审期记录）：结论可能已过时，当前状态以 `docs/scripts_review.md` 为准。
>
> 已变事实（2026-09-21 更正，读本文时先看这几条）：
> - 就绪上限：当时 `360×5s=30min`，现统一到 `harness.json limits.ready_tries=480`（2400s）；`HX_READY_TRIES` 只是覆盖变量。
> - 老入口（`round2_verify*.sh`、`perf/profile*.sh`、`run_matrix.sh`、`collect.sh`、`run_a5.sh`）随后已全部删除，`tests/` 已进版本库。
> - `accuracy/a30_slot_filter_ab` 已定案不修锚点、永久 SKIP（`scripts_review.md` §7-4）。
> - 测试脚本数 17（9 smoke + 3 accuracy + 5 perf）；p10 只判 `windows.json` 有窗口，`rank_count>0` 是 p20 的 `usable` 判据。
> - 2026-09-23：zigzag 改为 attention 内部选行（模型主流恢复 replicated）：`[CP_BALANCE][reduce]`
>   行与归约路径统计已删除（`planlog.py` / `run_matrix.py` 不再跟踪），perf 的归约 A/B 变体与其配置也已删除；
>   `check_b_path.py` 的静态门控改成按新架构断言（项数随之变化，见 `docs/dp1_zigzag_acc_fix.md`）。

# tests/ 分阶段复查（测试树重构 + 统一入口）

复查对象：`cp_balance/tests/`（18 个测试脚本 + `run_tests.sh` + `lib/`）、`.gitignore`、
`README.md` 的 tests 段、`docs/a5_104_runbook.md` 的统一入口说明。
方法：分 6 个阶段，先读契约再逐层核对；本机能跑的全跑（本机无 NPU，"起服务"类只做静态核对），
缺陷都给出可复现的证据，修完复验。

结论：**1 个功能性缺陷（P1）+ 2 个会假 PASS 的缺陷（P1）都已修**，另有 8 个健壮性/一致性问题已修；
1 项留给远端验证，2 项记录不修。复验后本地 fast 组仍是 6 PASS / 1 FAIL / 4 SKIP（唯一 FAIL 是本机缺少
远端路径，属预期）。

## 阶段 0：范围与跑法

| 阶段 | 范围 | 本地能跑到什么程度 |
| --- | --- | --- |
| 1 | `run_tests.sh`：发现/元数据展开/筛选/汇总/退出码 | 全跑（--list/--only/--from/--tag/--dry-run/--keep-going/--strict） |
| 2 | `tests/lib/`：`common.sh` 断言与角色表、`hx.py` 读取与审计 | 全跑 |
| 3 | `tests/smoke/` 9 个测试 | 7 个无服务测试实跑；s10/s11 静态核对 |
| 4 | `tests/accuracy/` 5 个实例 | a20 用合成产物实跑；a10/a30 静态核对 |
| 5 | `tests/perf/` 10 个实例 | p20/p21 用合成产物实跑；p10/p23 静态核对 |
| 6 | 文档与老入口兼容性 | `git status` / README 与 runbook 对照 |

## 阶段 1：runner 契约

跑通的事实：24 个实例（9 smoke + 5 accuracy + 10 perf）由 18 个脚本经 `# variants` 展开；
`--only` 支持完整 id / id#variant / basename；`--from` 按发现顺序续跑；`results.json` 与 `status.tsv` 落盘。

| # | 级别 | 发现 | 证据 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | P1 | `--tag` 只命中第一个 tag：tags 写的是 `npu, slow, service`，匹配串却是 `,slow,` | `--list --tag slow` 只列出 `perf/p20_analyse`、`perf/p23_report_order`（漏掉 11 个 slow 测试） | 已修：匹配前 `tr -d ' '`；复验 `fast/service/npu/slow/report/static/offline` 七组全部正确 |
| 2 | P2 | `--list` 不遵守 `--only/--skip/--tag` | `--list --only smoke/s05_disk` 仍打印 9 行 | 已修：抽出 `sel_ok()`，list 与 run 共用；复验 `--list --only smoke/s05_disk` 只 1 行 |
| 3 | P2 | `--family` 未校验，`A5` 静默按 a3 处理 | `--family A5 --list` 正常退出；`hx.py configs bogus` 返回 a3 列表 | 已修：runner 与 `hx.py` 都要求 `a5|a3`，非法值退出 2 |
| 4 | P3 | 产物目录只到分钟，同分钟两次运行互相覆盖 | runner `%m%d_%H%M` vs lib `%m%d_%H%M%S` | 已修：runner 统一到秒 |
| 5 | P3 | `sel_ok()` 被插到使用之后（本轮修复引入的回归） | `--list --tag slow` 报 `sel_ok: command not found` 24 次 | 已修：定义移到 `tag_match()` 之后，复验通过 |
| 6 | P3 | 未发现任何测试时静默退出 0 | — | 已修：`tests=0` 时退出 2 |

## 阶段 2：lib

| # | 级别 | 发现 | 证据 | 处理 |
| --- | --- | --- | --- | --- |
| 7 | P2 | `kernel_steps` 为 JSON `null` 时打印 `None`，与"键缺失"走两条分支 | 合成 summary.json 后 `p21` 报 `kernel_steps=None` 并 FAIL | 已修：`null` 与缺失统一成 `-`（unknown）；p21 全 unknown 时 SKIP |
| 8 | P3 | 服务就绪上限 240×5s=20min < 项目口径（矩阵 2400s、起停约 10min） | `common.sh` 原值 | 已修：默认 360×5s=30min，`HX_READY_TRIES` 可在 `tests/README.md` 查到 |
| 9 | P3 | `usable` 缺失：判"解析是否真的拿到数据"只能数窗口行 | 见阶段 5 的 #11 | 已修：`hx.py usable <dir>` 输出 `rank_count>0` 的窗口数 |

记录不修：`HX_PY` 走未加引号的展开，harness 路径含空格会失效（本项目路径固定，先记录）。

## 阶段 3：smoke

7 个无服务测试本地实跑：`s01/s02/s03/s04/s05/s06/s07`（其中 s03 因本机没有 A5 路径而 FAIL，属预期）。
两个静态门控对本地两棵树实跑 `RESULT: PASS`（`ZigzagPlan fields=13`、`check_b_path` 10 项 OK）。

| # | 级别 | 发现 | 处理 |
| --- | --- | --- | --- |
| 10 | P2 | `s02` 只校验 CONFIG/TP/IP/NIC，不校验 `CP_BALANCE=` 与 `MODEL=`（权重路径写错是 40 分钟级代价） | 已补两项断言 |
| 11 | P2 | `s03` 只查 `config.json`，权重文件缺失也算过 | 已补：`*.safetensors` 或 `*index.json` 至少一个存在 |

## 阶段 4：accuracy

| # | 级别 | 发现 | 证据 | 处理 |
| --- | --- | --- | --- | --- |
| 12 | P1 | `a20` 的 c_matrix 分支**恒定 SKIP**：compare 的 left/right 来自两个矩阵，但查找目录写死成 b 矩阵的产物 | 合成两套产物后运行：`[warn] c_vs_b: json missing in .../a10_matrix_gate.b_matrix/matrix`，b/noise 两对正常 | 已修：每个 role 查自己的 `a10_matrix_gate.<role>/matrix`；复验三对全部检查（`c_vs_b / b_vs_base / noise_floor`） |

## 阶段 5：perf

| # | 级别 | 发现 | 证据 | 处理 |
| --- | --- | --- | --- | --- |
| 13 | P1 | `p20` 对"有 trace 但解析拿不到任何 rank 数据"是**静默 PASS**：`profile_analyse.py` 仍会写出 0 数据的 `summary.json` | 合成 `prof_a5_cur_cp1/..._ascend_pt` 后运行：analyse 打 `produced no usable rank output`，旧逻辑报 `[ok] summary.json` | 已修：改用 `usable` 计数，0 个可用窗口即 FAIL 并指向 analyse 日志；复验 FAIL |
| 14 | P2 | 同一条测试对"trace 已 prune"也会失败（无法区分 prune 与采集坏） | 合成 prune 场景 | 已修：prune → SKIP 并列出被跳过的目录；采集坏 → FAIL |
| 15 | P2 | 报告类（`p22/p23/p24`）缺数据时一律 SKIP，不会假 PASS | 本地实跑 | 保持；测试语义已写进 `tests/README.md` |

## 阶段 6：文档与兼容性

- 老入口未动：`accuracy/round2_verify*.sh`、`perf/profile*.sh` 与本轮改动无交集（`git status` 只有 `.gitignore`、`README.md` 两个修改 + 新增 `tests/`）。
- 同步过的文档：`README.md` 新增"测试入口（tests/）"+ 6 行文件表；`tests/README.md`（目录/用法/元数据/环境变量/老入口对应表）；`docs/a5_104_runbook.md` 头部加统一入口说明。
- `.gitignore` 增加 `tests/_out/`；本机跑出的产物已清空。

## 复验记录（全部本地）

| 命令 | 结果 |
| --- | --- |
| `--list` / `--list --tag <7 组>` / `--list --only <taget>` | 24 个实例；七组 tag 过滤正确 |
| `--family A5`、`hx.py configs bogus` | 退出 2 + 明确报错 |
| `--tag fast --keep-going` | 6 PASS / 1 FAIL（`s03_paths` 缺远端路径，预期）/ 4 SKIP |
| 合成产物跑 `a20 / p20 / p21` | 三处缺陷按新逻辑给出正确判定（PASS / FAIL / SKIP） |
| `bash -n` 全部脚本 + `py_compile hx.py` | 通过 |

## 远端待验证（要真机才能定）

1. `smoke/s10_service_ready`、`smoke/s11_zigzag_request`：服务就绪上限（30min）与 `branch=ZIGZAG` 计数；
2. `accuracy/a10_matrix_gate#c_matrix|b_matrix`：40 条 prompt 的 `RESULT: PASS`（真判据）；
3. `accuracy/a30_slot_filter_ab`：`slot<0` 补丁 A/B（含补丁锚点是否漂移）；
4. `perf/p10_capture#*` → `perf/p20_analyse` → `perf/p21_window_single_step`：采集是否拿到 `rank_count>0` 的窗口、`kernel_steps==1`；
5. 建议先 `--list`、再 `--tag fast`、最后分层跑；失败用 `--from <id>` 续跑。

## 遗留（记录，不修）

- `HX_PY` 未加引号，harness 路径不能含空格；
- `--from` 不影响 `--list`（列清单时从头列）；
- 报告类测试没有阈值（噪声地板 `5%~13%` 只能人判，写成硬门禁就是假失败）。
