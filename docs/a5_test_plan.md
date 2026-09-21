
# A5 环境测试计划（精度 + 性能）

对象：A5 机器（8 卡）。一条测试 = `configs/` 下一条 JSON，harness 级参数（树/超时/验证步骤）在根目录 `harness.json`。
配套：`docs/remote_run.md`（远端怎么跑）、`docs/scripts_review.md`（当前状态与未决问题）。

> 现场命令清单（`141.61.133.104` 的 IP/网卡覆盖、冒烟、回传打包、归约 A/B）见
> `docs/a5_104_runbook.md`；本文件保留 A5/A3 差异、判据来源与回传口径。

## 1. A5 与 A3 的差别（决定"不能沿用 A3 结论"）

| 项 | A3（旧） | A5（本次） | 影响 |
| --- | --- | --- | --- |
| 卡数 / TP | 16 | 8 | cp_size = TP = 8，zigzag 每序列切 16 块（A3 是 32 块） |
| 网卡 / IP | eth2 / 7.246.78.75 | 配置里是 `auto`（现场 `.104`/`eth2` 用 `CP_BALANCE_LOCAL_IP`/`CP_BALANCE_NIC_NAME` 覆盖） | 只影响 HCCL 组网 |
| 模型 | GLM-5.2-W4A8C8 | GLM-5.2-w4a4c8-mxfp4 | 量化不同（4bit 权重 + 4bit 激活），SFA/indexer 的数值路径要重新验收 |
| 投机解码 | 无 | MTP（deepseek_mtp, 1 token） | 多一条 draft 元数据路径；zigzag 资格门只放行主 prefill |
| max-num-seqs | 32 | 500 | 只影响并发 |
| 启动前动作 | 无 | `source /mnt/.../set_env.bash`（vendor 环境） | 已用配置里的 `prelude` 字段承载 |
| 确定性环境变量 | 开 | 精度组开、性能组关 | 性能组不该为确定性付开销 |

量过的输入规模（本仓库 `questions.json`，GLM-5.2 tokenizer 实测）：长 prompt 2768~2781 token（超过阈值 2048，会进 zigzag），短 prompt 72~107 token（走连续切片路径）。

因为模型型号和并行度都变了，**A3 的 C 验收 / 性能结论在 A5 上必须重跑**，尤其是
`enable_sparse_sfa_c8` / `enable_sparse_li_c8` 打开时的 KV 写路径（cp_balance 唯一改动过的写回路径）。

## 2. 前置（3 分钟，先做）

```bash
# 2.1 cp_balance 仓库（本文档所在仓库）与两棵代码树
cd /home/z30055003/cp_balance                     # 或在 A5 上你克隆的位置
git pull                                          # 拿到本轮精简后的 harness

# 2.2 当前代码树：A5 上的 vllm-ascend（harness.json trees.cur 指向的那棵）
git -C /home/z30055003/vllm-ascend log -1 --format='%h %s'
git -C /home/z30055003/vllm-ascend status --short   # 记录 HEAD/脏树状态即可（verify.sh 会按 harness.json trees 自动对齐）

# 2.3 base 代码树：B 等价性要跟它比。默认写成 /home/z30055003/vllm-ascend-base
#     若 A5 上路径不同，改 harness.json trees.*.path（临时覆盖用 CP_BALANCE_REPO / CP_BALANCE_BASE_REPO）；配置里只有 repo_tree
ls -d /home/z30055003/vllm-ascend-base
```

判据：2.2 的 HEAD 是你本轮要验的提交；2.3 的目录存在（B 等价性需要它）。

再花 10 秒确认启动命令本身（不启动服务，只打印指纹与环境）：

```bash
bash run.sh glm52_a5_cur_cp0 --dry-run --print-env
```

要能看到：`REPO=<harness.json trees.cur.path>`、`TP=8`、`NIC=auto`、`IP=auto`（没 export 机器身份时；现场固定成 .104/eth2 时才显示具体值）、
`PRELUDE=True`、`SPEC=deepseek_mtp/1`（这是带 MTP 的配置；`prof_a5_*` 是 `SPEC=off`），
以及 argv 以 `bash -c 'source /mnt/share/.../set_env.bash && mkdir -p /tmp/cp_balance_a5_plog && exec vllm serve ...` 开头。

### 2.4 换到另一台机器（IP / 网卡不同）

同一套配置可以直接在别的机器上跑，只要在那台机器的 shell 里把"机器身份"覆盖掉（不写进配置，
共享检出时两台机器互不干扰）：

```bash
export CP_BALANCE_LOCAL_IP=141.61.133.104
export CP_BALANCE_NIC_NAME=eth2
# 需要时也可以覆盖可见卡：export CP_BALANCE_DEVICES=0,1,2,3,4,5,6,7

bash run.sh glm52_a5_cur_cp0 --dry-run --print-env
#   期望：NIC=eth2 IP=141.61.133.104，且 env 里 HCCL_IF_IP / GLOO_SOCKET_IFNAME /
#   TP_SOCKET_IFNAME / HCCL_SOCKET_IFNAME 都跟着变

bash verify.sh --family a5           # 一键：前置 -> 静态 -> 冒烟 -> 诊断 -> 打包（会沿用这些环境变量）
bash tests/run_tests.sh --only accuracy/a10_matrix_gate --keep-going
```

命令行的 `--set local_ip=141.61.133.104 --set nic_name=eth2` 优先级更高，临时改一个值时用它。

这台机器上还要存在 A5 配置里写死的这几样（不存在就改 `configs/_common_a5.json`）：

| 什么 | 默认值 | 改哪里 |
| --- | --- | --- |
| 当前代码树 | `<workdir>/vllm-ascend` | `harness.json trees.cur.path`（配置里只写 `repo_tree: cur`；临时覆盖用 `CP_BALANCE_REPO`） |
| base 代码树 | `<workdir>/vllm-ascend-base` | `harness.json trees.base.path`（配置里写 `repo_tree: base`；`CP_BALANCE_BASE_REPO` 可覆盖）。树不在时 harness 自动 clone/对齐到 fork 的 `base-dp1` 分支（= main aff1b74b6 + dp=1 门修一行） |
| 模型 | `/mnt/share/weights/GLM-5.2-w4a4c8-mxfp4` | `_common_a5.json` 的 `model` |
| vendor 环境 | `/mnt/share/l00622059/vendors/custom_transformer/bin/set_env.bash` | `_common_a5.json` 的 `prelude` |
| 卡与并行度 | 8 卡（devices 0-7）、TP=8 | `_common_a5.json` 的 `devices` / `tp_size`（base 侧配置继承同一份公共配置） |

## 3. 精度：怎么跑

```bash
bash verify.sh --family a5                        # 一键（含静态门控、冒烟与诊断）
bash tests/run_tests.sh --only accuracy/a10_matrix_gate --keep-going   # C 验收 + B 等价性
bash tests/run_tests.sh --only accuracy/a20_compare_collected        # 复用已采 json 复跑（秒级）
```

`a10_matrix_gate` 展开成三件事（与 A3 同一套逻辑，只是换了 A5 的矩阵与端口；
矩阵里的 `require_log` 门保证 cp1 真的出现过 zigzag 证据，否则整组判 FAIL，不会空跑通过）：

| 步骤 | 内容 | 服务起停 | 判据 |
| --- | --- | --- | --- |
| 0 | 两条静态门控 | 0 | 都 `RESULT: PASS`，且 `ZigzagPlan fields=13` |
| 1 | C 验收：`CP_BALANCE=1` 对 `CP_BALANCE=0`，40 条 prompt 比首 token | 2 | `first-token match: 40/40` |
| 2 | B 等价性：`CP_BALANCE=0` 对 base，base 再跑一遍当噪声地板 | 3 | 两项都 PASS |
| 3 | 可选 A/B（`accuracy/a30_slot_filter_ab`）：基线 vs 临时补丁 + KV 写过滤 padding 行 | 2 | 现状：SKIP（round2_verify 的补丁锚点随源码失效，已定案不修） |

产物统一在 `tests/_out/<时间戳>[_<family>]/`（矩阵在 `accuracy/a10_matrix_gate.<变体>/matrix/`，`summary.txt` 是汇总）；
`round2_<时间戳>/` 只属已 SKIP 的可选 A/B。

**如果 C 验收没过、又怀疑是 MTP 干扰**：换成不带投机解码的同两套配置再跑一次

```bash
bash tests/run_tests.sh --only accuracy/a10_matrix_gate#nomtp_matrix     # 原 run_matrix.sh 已删
```

对应配置 `glm52_a5_nomtp_cur_cp1/cp0`（端口 8038/8039，除 MTP 外与主配置一致）。
若关掉 MTP 后通过、开着不通过，那是一条独立结论，要作为问题记录下来。

## 4. 性能：一条命令

```bash
bash tests/run_tests.sh --only perf --skip perf/p10_capture#prof_a2a   # 四组：prof_cp0 / prof_cp1 / prof_cp0_repeat / prof_base
bash tests/run_tests.sh --only perf/p10_capture#prof_cp1               # 只跑一组（变体名是角色名）
```

四组配置（A5 专用，端口 8084~8088，profiling 组不带 MTP）：

| 配置 | 代码树 | `cp_balance` | `reduce_mode` | 作用 |
| --- | --- | --- | --- | --- |
| `prof_a5_cur_cp0` | 当前 | 0 | — | 与 base 等价的性能基准 |
| `prof_a5_cur_cp1` | 当前 | 1 | `allreduce` | zigzag 现状 |
| `prof_a5_cur_cp0_repeat` | 当前 | 0 | — | **噪声地板**（同代码路径再跑一遍） |
| `prof_a5_base_cp0` | base | 0 | — | 原版 DSA-CP 参照 |
| `prof_a5_cur_cp1_a2a` | 当前 | 1 | `alltoall` | 可选的低通信量归约 A/B（说明见 `docs/a5_104_runbook.md` §4） |

判读口径（A3 那一轮踩过的坑，A5 直接沿用）：

1. **先看噪声地板**：`profile_compare.py prof_a5_cur_cp0 prof_a5_cur_cp0_repeat` 的差值就是本轮噪声；
   两次差异小于噪声时不要下结论。A3 那轮的轮间漂移是 5%~13%。
2. **先核对次数再谈时间**：看 `op_statistic.csv` 的 `OP Type` 计数 —— 非 zigzag 一侧只有
   `reduce_scatterAicpuKernel`；zigzag 一侧应恰好少 N 次 `reduce_scatterAicpuKernel`、多 N 次
   `allreduceAicpuKernel`（`reduce_mode=alltoall` 时是 `alltoallAicpuKernel`），
   N = 4 × 窗口内 prefill 步数（3 个 dense 层 `down_proj` + embedding）；次数不对说明配置没生效。
3. ~~每个长度的窗口 `steps` 必须是 1~~：当前 trace 的 `kernel_details.csv` 没有 Step 列，`kernel_steps` 恒 None，
   `perf/p21_window_single_step` 恒 SKIP —— 这条现在判不了（`scripts_review.md` §3 待定）。
4. `trace_view.json` 很大时用 `--trim`（只回传 `order_rank0.json`）。

采集、解析、对比分别由 `tests/perf/p10_capture`、`p20_analyse`、`p22_report_compare` 承担（原 `perf/profile_a5.sh` 已删）；
解析必须在远端单独起进程（mp 后端 worker 是
daemon，torch_npu 解析器拒绝在 daemon 里跑），脚本已经这么做。

## 5. A5 上要先确认的三件事

| # | 要确认 | 怎么看 | 不对时怎么办 |
| --- | --- | --- | --- |
| 1 | base 代码树路径 | `ls -d /home/z30055003/vllm-ascend-base`（由 harness 自动 clone/对齐到 `base-dp1` 分支） | 改 `harness.json trees.base.path`（临时覆盖 `CP_BALANCE_BASE_REPO`）；矩阵里只用 `static_check.base_tree` |
| 2 | 长 prompt 真的进了 zigzag | 日志里 zigzag 证据计数 > 0：`branch=ZIGZAG` 或 `[CP_BALANCE][plan]`（两个串都受 DEBUG 门控；采集轮 `debug=0` 时不打） | 说明 prompt 不到 `MIN_TOKENS=2048`：用 `--set min_tokens=1536`（或 1024）重跑，并在结论里注明阈值。**已用 GLM-5.2 tokenizer 量过**：20 条长 prompt 是 2768~2781 token（会进 zigzag），20 条短 prompt 是 72~107 token（走连续切片） |
| 3 | 8 卡 HCCL 组网正常 | 服务能起来、`/v1/models` 可访问；`HCCL_IF_IP` / `HCCL_SOCKET_IFNAME` 跟随现场机器身份（配置默认 `auto`） | 用 `CP_BALANCE_LOCAL_IP` / `CP_BALANCE_NIC_NAME` 覆盖（`.104`/`eth2` 的口径见 `docs/a5_104_runbook.md` §0）；想写进配置就把 `_common_a5.json` 的 `local_ip` / `nic_name` 从 `auto` 改成具体值 |

## 6. 判据一览（每步 FAIL 时先看哪）

| 症状 | 先看 |
| --- | --- |
| 步骤 0 静态门控 FAIL | 代码树不对（`--repo`）或本轮改动没同步过去；`fields` 不是 13 说明版本不对 |
| 步骤 1 首 token 不一致 | 顺序：① 指纹行（`[cp_balance] CONFIG=... REPO=... HEAD=...` 两侧是否同树同开关）② `[CP_BALANCE][branch]` 是否真的 ZIGZAG ③ 换 `matrix_a5_c_accept_nomtp.json` 排除 MTP ④ 关掉 `enable_sparse_li_c8` 再试 |
| 步骤 2 噪声地板 FAIL | 测量本身不可复现（不是代码问题）：先重跑一次确认 |
| ~~步骤 3 两版首 token 不同~~ | 该步（`accuracy/a30_slot_filter_ab`）已定案恒 SKIP，不再是排查路径 |
| 性能两次差异小于噪声 | 不要下结论，加大长度档或重复轮次 |

## 7. 回传清单

| 文件 | 用途 |
| --- | --- |
| `tests/_out/<时间戳>_<family>/accuracy/a10_matrix_gate.<变体>/matrix/summary.txt` | 本轮精度验收的全部判定 |
| `tests/_out/<stamp>_<family>/accuracy/a10_matrix_gate.<变体>/matrix/*.log` 的指纹行与 `cmp_*.txt` | C 验收 / B 等价性的证据（含代码树 HEAD、开关、zigzag 证据计数） |
| `verify_<family>_*.tar.gz` | 一键验证打的包：报告 + 这一轮 `tests/_out` |
| 每个 `prof_a5_*/` 的 `summary.json`、`windows.json`、`export/`、`order_rank0.json` | 性能对比与算子归因 |
| `check_cp_balance_fields.py` / `check_b_path.py` 的完整输出 | 静态门控 |

打包示例（只回传必要文件，不要拉原始 trace）：

```bash
# 一键验证已经打好包（含诊断报告与整轮 tests/_out）：
ls -lh verify_a5_*.tar.gz

# 只想带性能产物：
tar czf a5_perf_$(date +%m%d_%H%M).tgz \
    prof_a5_*/summary.json prof_a5_*/windows.json prof_a5_*/order_rank0.json \
    prof_a5_*/export
```
