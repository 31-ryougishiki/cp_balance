# 远端怎么跑精度测试和性能测试

> 当前状态（dp=1 的 DSA-CP 门、zigzag 的 MoE 布局未决、各次审查结论）见 `docs/scripts_review.md`；
> 本文件只讲远端怎么跑。


对象：A3 机器（16 卡，模型 GLM-5.2-W4A8C8，`/opt/its/z30055003`，7.246.78.75 / eth2）。
入口有两个：`bash verify.sh --family <a5|a3>`（一键：前置 -> 静态 -> 冒烟 -> 分支诊断 -> 打包），
以及底层的 `tests/run_tests.sh`（发现/筛选/执行/汇总）。老的一键脚本（`round2_verify_a5.sh`、`perf/profile_a5.sh`、
`run_matrix.sh`、`collect.sh`、`run_a5.sh`）已删除；`accuracy/run_matrix.py`、`perf/collect.py`、`perf/profile_l6.sh`
这些工具还在（由 tests/ 调用）。
服务起停约 10 分钟/次：先把不起服务的检查跑完，再动机时。

## 0. 前置

代码树同步：harness 仓（`cp_balance`，分支 `main`）与被测树（`vllm-ascend`，分支 `cp_balance`）都要先 push；
远端只做 `git pull`（拿到 `harness.json`、`verify.sh`、`docs/` 等）。
参照树由 harness 按 `harness.json trees` 自动 clone/对齐到 fork 的 `base-dp1` 分支（= main + dp=1 门修一行），
日志里会打印 `[sync] vllm-ascend-base 已在 base-dp1@...`（或 checkout/reset 的动作）。
补丁只改 dp=1 那道门，不会往参照树里带 `VLLM_ASCEND_CP_BALANCE`（verify.sh 前置仍会查这一点）。

每个 shell 先执行（换机器只改这三行）：

```bash
cd /opt/its/z30055003/cp_balance
export CP_BALANCE_FAMILY=a3              # a5 机器写 a5
export CP_BALANCE_LOCAL_IP=7.246.78.75   # ip -o -4 addr show 查本机实际 IP
export CP_BALANCE_NIC_NAME=eth2

git -C /opt/its/z30055003/vllm-ascend      log -1 --oneline   # 待验树
git -C /opt/its/z30055003/vllm-ascend-base log -1 --oneline   # 参照树：base-dp1 分支（main + dp=1 门修，verify.sh 自动对齐）
df -h .                                                       # profiling 会写 GB 级 trace
```

```bash
bash tests/run_tests.sh --list        # 24 个实例：id / needs / tags / est / desc
bash tests/run_tests.sh --tag fast    # 秒级~分钟级，不起服务
```

`--tag fast` 里的 `smoke/s03_paths` 检查配置引用的 repo / 模型 / base 树在不在本机：
它 FAIL 就先修路径（改 `configs/` 或设 `CP_BALANCE_REPO` / `CP_BALANCE_BASE_REPO`），别往下走。

## 1. 精度测试

### 1.1 冒烟（两次起停，约 27 分钟）

```bash
bash tests/run_tests.sh --only smoke/s10_service_ready,smoke/s11_zigzag_request
```

判据：两条都 `[done] ... PASS`。s10 证明服务能起、`/v1/models` 有配置里的模型名；
s11 证明长 prompt（≥ `min_tokens`=2048）出现 zigzag 证据（`branch=ZIGZAG` 或 `[CP_BALANCE][plan]`），
短 prompt 只有 `branch=CONTINUOUS`。日志里还应有 `DSA-CP is enabled without sequence-parallel MoE`
（dp=1 的期望值；若看到 `Disabling DSA-CP`，说明树没带 dp=1 的门修）；draft 步会打 `reason=draft`。
长 prompt token 不够 `min_tokens` 时 s11 记 SKIP（INCONCLUSIVE），那是 prompt 阶梯问题，不是功能失败。

### 1.2 正式：C 验收 + B 等价性（5 次起停，约 1 小时）

```bash
bash tests/run_tests.sh --only accuracy/a10_matrix_gate --keep-going
```

| 实例 | 内容 | 判据 |
| --- | --- | --- |
| `#c_matrix` | `glm52_cur_cp1`（CP_BALANCE=1）vs `glm52_cur_cp0`（0），40 条 prompt | `first-token match: 40/40`，且 `require_log` 门通过（cp1 日志必须有 zigzag 证据、cp0 不许出现 `[CP_BALANCE][plan]`，空跑会 FAIL） |
| `#b_matrix` | `glm52_cur_cp0` vs base 树，加 base 再跑一遍当噪声地板 | 两项都 `first-token match: 40/40` + `text_head match: 40/40` |
| `#nomtp_matrix` | A3 没有这个配置 | SKIP（不是失败） |

离线复跑对比（秒级，不起服务；趁服务还在时先跑完更省事）：

```bash
bash tests/run_tests.sh --only accuracy/a20_compare_collected
```

可选 A/B（临时补丁，跑完自动还原；2 次起停）：

```bash
bash tests/run_tests.sh --only accuracy/a30_slot_filter_ab   # 现状：SKIP（driver 锚点过期，已定案不修）
```

一次跑完上面全部（静态门控 + 矩阵 + 离线复跑 + 可选 A/B）：

```bash
bash tests/run_tests.sh --from accuracy
```

## 2. 性能测试

### 2.1 采集（一次起停 = 一组，约 40 分钟/组）

```bash
bash tests/run_tests.sh --only perf/p10_capture    # 5 个变体，跑完约 3.5 小时
```

标准一轮跑四组（跳过可选的 `#prof_a2a`）：

```bash
bash tests/run_tests.sh --only perf/p10_capture --skip perf/p10_capture#prof_a2a
```

| 变体 | 代码树 | CP_BALANCE | reduce_mode | 作用 |
| --- | --- | --- | --- | --- |
| `#prof_cp0` | 当前 | 0 | allreduce | 与 base 等价的基准 |
| `#prof_cp1` | 当前 | 1 | allreduce | zigzag 现状（主对比） |
| `#prof_cp0_repeat` | 当前 | 0 | allreduce | 噪声地板（同路径重跑一遍） |
| `#prof_base` | base | 0 | allreduce | 参照树（main + dp=1 门补丁）的 DSA-CP 参照 |
| `#prof_a2a` | 当前 | 1 | alltoall | 归约 A/B（可选，另加一组机时） |

每组按 `lengths`（1024/2048/4096/6144/8192/12288 token）逐档采一个只含 prefill 步的窗口，
并补跑一次不采样的同请求拿 `clean_s`。

### 2.2 解析、审计、报告、打包（离线，约 20 分钟）

```bash
bash tests/run_tests.sh --only perf/p20_analyse,perf/p21_window_single_step,perf/p22_report_compare,perf/p23_report_order,perf/p24_collect --keep-going
```

- `p20_analyse`：必须在远端独立进程里跑（torch_npu 解析器不接受 daemon 进程），判据是每个配置
  `summary.json` 里有 `rank_count > 0` 的窗口；
- `p21_window_single_step`：判据是每个窗口 `kernel_steps == 1`，但当前 trace 的 `kernel_details.csv` 没有 Step 列，
`kernel_steps` 恒为 None → **该测试恒 SKIP**（别读成已验证）；
- `p22_report_compare`：`cp1 vs cp0`、`cp0 vs repeat`、`cp0 vs base` 三张对比表（采过 `#prof_a2a` 时多一张 `cp1 vs a2a`）；
- `p23_report_order`：算子顺序 + device kernel 归因（需要原始 trace，已 prune 就 SKIP）；
- `p24_collect`：打成 `collect_<时间戳>/*.tgz`（现状只断言"有 tgz"，空包也算 PASS，别当实质校验）。

老入口一条龙 `perf/profile.sh` 已删；要顺手删原始 trace 用 `python3 perf/collect.py --prune-traces <配置...>`。

判读顺序（先噪声地板，再绝对量，别发明比值）见仓库 README 的 profiling 一节与工程机文档目录的
`docs/perf_plan.md`。

## 3. 回传什么

```bash
cd /opt/its/z30055003/cp_balance
# 一键验证自己会打包（报告 + 这一轮 tests/_out）
ls -lh verify_<family>_*.tar.gz

# 只跑了某些测试时：
tar czf acc_$(date +%m%d_%H%M).tgz \
    tests/_out/*/status.tsv tests/_out/*/results.json tests/_out/*/smoke tests/_out/*/accuracy
tar czf perf_$(date +%m%d_%H%M).tgz \
    tests/_out/*/perf collect_*/ \
    prof_*/summary.json prof_*/windows.json prof_*/order_rank0.json prof_*/export
ls -lh acc_*.tgz perf_*.tgz verify_*.tar.gz
```

`tests/_out/<时间戳>[_<family>]/` 是一次验证的全部证据：每个测试一个子目录（配置指纹、采集到的 json、
驱动日志、服务日志 `<配置名>.log`）+ `status.tsv` + `results.json` + 合并日志 `<id>.log`；
一键验证还会写一份 `<PREFIX>_<family>_branch_<stamp>.txt` 诊断报告。
原始 trace（`prof_*/*_ascend_pt`，GB 级）不用拷；服务日志可能几十 MB，gzip 后会小很多。

## 4. 失败与续跑

- 每个测试的日志在 `tests/_out/<时间戳>[_<family>]/<id 里的 / 换成 _>.log`（stdout + stderr 都在里面），先看它；
- `--keep-going` 让一轮跑到底；失败后用 `--from <id>` 补跑后面几步，例如
  `--from perf/p20_analyse`；
- `--only` / `--skip` / `--from` 都支持目录名（`smoke` / `accuracy` / `perf`）；
- 选择器一个都没命中、或选项缺参数，runner 退出码 2 并打印原因（不会静默跳过）；
- 判定看各测试自己的 `[done] ... PASS/FAIL`，以及 `status.tsv`（PASS/FAIL/SKIP + 耗时）；
- 服务残留（测试被打断）：`pkill -f -- "--port 8035"`；正常路径上 `common.sh` 的 EXIT trap 会自己收尾；
- 就绪探测走 `127.0.0.1`：远端若有 `http_proxy` 而没有 `no_proxy`，探测会被发到代理（表现为"服务明明起来了，测试不往下走"）。
  harness 已内置本机白名单，手工 curl 请加 `--noproxy '*'`。
