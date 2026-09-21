# 文档索引

（本目录已纳入 harness 仓：`cp_balance/docs/`，远端 `git pull` 就能拿到。）

## 当前有效（先看这三份）

- `verify_steps.md` —— **接下来怎么验**：A5 节点上 cp_balance 的精度（B 等价性 + C 验收）与性能（四组）验证路线、判定与回传。
- `scripts_review.md` —— 脚本/配置的分阶段审查、`use_sequence_parallel_moe` 要求 dp>1 的分阶段考证、
  dp=1 的现状与未决问题（zigzag 的 MoE 布局）、5 项定案。所有"当前状态"以它为准。
- `remote_run.md` —— 远端怎么跑：仓/分支、一键 `verify.sh`、精度/性能命令、回传什么、失败怎么续跑。

## 操作手册

- `a5_104_runbook.md` —— A5 节点（141.61.133.104）现场命令清单：机器族判定、冒烟、精度/性能、打包。
- `a5_test_plan.md` —— A5 与 A3 的差异、判据来源、前置检查（为什么 A5 不能沿用 A3 结论）。
- `cp_balance_remote_checklist.md` —— A3 的验收口径与回传清单（A5 可借用）。

## 性能

- `perf_plan.md` —— 性能轮的实验设计与判读顺序。
- `profiling_guide.md` —— profiling 的机制、字段与失败模式（体量最大，按需查）。

## 历史归档（结论可能已过时，文件头都标了）

- `cp_balance_main_port.md` / `cp_balance_main_port_review.md` —— 移植到 main 线的计划与阶段评审。
- `base_dsa_cp_flow.md` / `cur_cp_balance_flow.md` —— 老线/新线 DSA-CP + cp_balance 的代码流程分析（移植期所写）。
- `cp_balance_analysis.md` / `cp_balance_walkthrough.md` / `cp_balance_review_round2.md` —— 早期分析与 round-2 评审。
- `tests_review.md` —— tests/ 分层与元数据评审（入口后来改成 `harness.json` + `verify.sh`）。
- `port/*.md` —— 移植期取证（特征清单、hook 映射、vllm API delta）与 A/B/C 三份阶段评审。

## 更早的机器/交接记录（原样保留）

- 这三份不在本仓里、留在工作区：`<workspace>/.scratch/HANDOVER_hist.md`、`<workspace>/env/HANDOVER_hist.md`、`<workspace>/env/README_hist.md`（后两份是 UTF-16 导出）。

## 约定

- 判据、入口、超时之类的"当前值"只在三处维护：仓内 `README.md`、`tests/README.md`、`harness.json`；
  历史文档里出现的旧命令（`run_a5.sh`、`round2_verify*.sh`、`perf/profile*.sh`、`tests/lib/trees.json` 等）都已不存在。
- 历史文档里的代码行号只保证写作时的版本（每份头部注了写作日期与背景）。
