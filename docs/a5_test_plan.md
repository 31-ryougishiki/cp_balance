
# A5 环境测试计划（精度 + 性能）

对象：A5 机器（8 卡），参考启动脚本 `cp_balance/run_a5.sh` 的配置改写。
配套：`docs/cp_balance_remote_checklist.md`（A3 的验收口径）、`docs/cp_balance_review_round2.md`（本轮改了什么）。

## 1. A5 与 A3 的差别（决定"不能沿用 A3 结论"）

| 项 | A3（旧） | A5（本次） | 影响 |
| --- | --- | --- | --- |
| 卡数 / TP | 16 | 8 | cp_size = TP = 8，zigzag 每序列切 16 块（A3 是 32 块） |
| 网卡 / IP | eth2 / 7.246.78.75 | eth0 / 141.61.133.112 | 只影响 HCCL 组网 |
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

# 2.2 当前代码树：A5 上的 vllm-ascend（run_a5.sh 用的那棵）
git -C /home/z30055003/vllm-ascend log -1 --format='%h %s'
git -C /home/z30055003/vllm-ascend status --short   # 应只看到本轮 5 个文件的改动（或已提交）

# 2.3 base 代码树：B 等价性要跟它比。默认写成 /home/z30055003/vllm-ascend-base
#     若 A5 上路径不同，改 configs/_common_a5.json 与 configs/matrix_a5_b_vs_base.json 里的 repo
ls -d /home/z30055003/vllm-ascend-base
```

判据：2.2 的 HEAD 是你本轮要验的提交；2.3 的目录存在（B 等价性需要它）。

再花 10 秒确认启动命令本身（不启动服务，只打印指纹与环境）：

```bash
bash run.sh glm52_a5_cur_cp0 --dry-run --print-env
```

要能看到：`REPO=/home/z30055003/vllm-ascend`、`TP=8`、`NIC=eth0`、`IP=141.61.133.112`、
`PRELUDE=True`、`SPEC=deepseek_mtp/1`（这是带 MTP 的配置；`prof_a5_*` 是 `SPEC=off`），
以及 argv 以 `bash -c 'source /mnt/share/.../set_env.bash && mkdir -p /tmp/cp_balance_a5_plog && exec vllm serve ...` 开头。

## 3. 精度：一条命令

```bash
bash accuracy/round2_verify_a5.sh                 # 全跑：静态 + C 验收 + B 等价性 + 可选 A/B
bash accuracy/round2_verify_a5.sh --steps 0,1,2   # 跳过可选 A/B（省两次服务起停）
bash accuracy/round2_verify_a5.sh --dry-run       # 只打印计划，不启动任何东西
```

它做四件事（与 A3 完全同一套逻辑，只是换了 A5 的矩阵与端口）：

| 步骤 | 内容 | 服务起停 | 判据 |
| --- | --- | --- | --- |
| 0 | 两条静态门控 | 0 | 都 `RESULT: PASS`，且 `ZigzagPlan fields=13` |
| 1 | C 验收：`CP_BALANCE=1` 对 `CP_BALANCE=0`，40 条 prompt 比首 token | 2 | `first-token match: 40/40` |
| 2 | B 等价性：`CP_BALANCE=0` 对 base，base 再跑一遍当噪声地板 | 3 | 两项都 PASS |
| 3 | 可选 A/B：基线 vs 临时补丁（多一行 TP/EP 分组日志 + KV 写过滤 padding 行） | 2 | 两版首 token 相同；日志里 `tp=8/x ep=8/x` |

产物在 `round2_<时间戳>/`，`summary.txt` 是汇总与回传清单。

**如果 C 验收没过、又怀疑是 MTP 干扰**：换成不带投机解码的同两套配置再跑一次

```bash
bash accuracy/run_matrix.sh configs/matrix_a5_c_accept_nomtp.json
```

对应配置 `glm52_a5_nomtp_cur_cp1/cp0`（端口 8038/8039，除 MTP 外与主配置一致）。
若关掉 MTP 后通过、开着不通过，那是一条独立结论，要作为问题记录下来。

## 4. 性能：一条命令

```bash
bash perf/profile_a5.sh                                   # 默认四组：cp0 / cp1 / cp0_repeat / base_cp0
bash perf/profile_a5.sh prof_a5_cur_cp0 prof_a5_cur_cp1   # 只跑两组（快）
bash perf/profile_a5.sh prof_a5_cur_cp0 prof_a5_cur_cp1 prof_a5_cur_cp1_a2a prof_a5_base_cp0
```

四组配置（A5 专用，端口 8084~8088，profiling 组不带 MTP）：

| 配置 | 代码树 | CP_BALANCE | 归约模式 | 作用 |
| --- | --- | --- | --- | --- |
| `prof_a5_cur_cp0` | 当前 | 0 | — | 与 base 等价的性能基准 |
| `prof_a5_cur_cp1` | 当前 | 1 | allreduce | zigzag 现状 |
| `prof_a5_cur_cp0_repeat` | 当前 | 0 | — | **噪声地板**（同代码路径再跑一遍） |
| `prof_a5_base_cp0` | base | 0 | — | 原版 DSA-CP 参照 |
| `prof_a5_cur_cp1_a2a` | 当前 | 1 | alltoall | 可选的低通信量归约 A/B |

判读口径（A3 那一轮踩过的坑，A5 直接沿用）：

1. **先看噪声地板**：`profile_compare.py prof_a5_cur_cp0 prof_a5_cur_cp0_repeat` 的差值就是本轮噪声；
   两次差异小于噪声时不要下结论。A3 那轮的轮间漂移是 5%~13%。
2. **先核对次数再谈时间**：zigzag 相对非 zigzag 的集合通信次数差应恰好是
   "4 处 × prefill 步数"（3 个 dense 层 down_proj + embedding）；次数不对说明配置没生效。
3. 每个长度的窗口 `steps` 必须是 1（脚本会打 WARNING），不是 1 的话这一档作废。
4. `trace_view.json` 很大时用 `--trim`（只回传 `order_rank0.json`）。

采集 + 解析 + 对比一条龙在 `perf/profile_a5.sh` 里；解析必须在远端单独起进程（mp 后端 worker 是
daemon，torch_npu 解析器拒绝在 daemon 里跑），脚本已经这么做。

## 5. A5 上要先确认的三件事

| # | 要确认 | 怎么看 | 不对时怎么办 |
| --- | --- | --- | --- |
| 1 | base 代码树路径 | `ls -d /home/z30055003/vllm-ascend-base` | 改 `configs/_common_a5.json`（`repo`）与 `configs/matrix_a5_b_vs_base.json`（`static_check.base_repo` + 两条 compare 的右值配置） |
| 2 | 长 prompt 真的进了 zigzag | 步骤 1/3 的日志里 `branch=ZIGZAG` 计数 > 0（driver 会打印） | 说明 prompt 不到 `MIN_TOKENS=2048`：用 `--set min_tokens=1536`（或 1024）重跑，并在结论里注明阈值。**已用 GLM-5.2 tokenizer 量过**：20 条长 prompt 是 2768~2781 token（会进 zigzag），20 条短 prompt 是 72~107 token（走连续切片） |
| 3 | 8 卡 HCCL 组网正常 | 服务能起来、`/v1/models` 可访问；`HCCL_IF_IP=141.61.133.112`、`HCCL_SOCKET_IFNAME=eth0` | 按 A5 现场要求改 `_common_a5.json` 的 `local_ip` / `nic_name` |

## 6. 判据一览（每步 FAIL 时先看哪）

| 症状 | 先看 |
| --- | --- |
| 步骤 0 静态门控 FAIL | 代码树不对（`--repo`）或本轮改动没同步过去；`fields` 不是 13 说明版本不对 |
| 步骤 1 首 token 不一致 | 顺序：① 指纹行（`[cp_balance] CONFIG=... REPO=... HEAD=...` 两侧是否同树同开关）② `[CP_BALANCE][branch]` 是否真的 ZIGZAG ③ 换 `matrix_a5_c_accept_nomtp.json` 排除 MTP ④ 关掉 `enable_sparse_li_c8` 再试 |
| 步骤 2 噪声地板 FAIL | 测量本身不可复现（不是代码问题）：先重跑一次确认 |
| 步骤 3 两版首 token 不同 | padding 行的 `slot=-1` 没有被跳过 → 记录为真实问题，KV 写要改成掩码 scatter |
| 性能两次差异小于噪声 | 不要下结论，加大长度档或重复轮次 |

## 7. 回传清单

| 文件 | 用途 |
| --- | --- |
| `round2_<时间戳>/summary.txt` | 本轮精度验收的全部判定 |
| `round2_*/01_c_accept/summary.txt` 与两个 `*.log` 的指纹行 | C 验收证据（含代码树 HEAD、开关、ZIGZAG 计数） |
| `round2_*/02_b_equiv/summary.txt` | B 等价性 + 噪声地板 |
| `round2_*/03_optional/` 的 `patched.log`（`[CP_BALANCE][group]` 行）与 `cmp_patched_vs_baseline.txt` | TP/EP 分组、padding 行结论 |
| 每个 `prof_a5_*/` 的 `summary.json`、`windows.json`、`export/`、`order_rank0.json` | 性能对比与算子归因 |
| `check_cp_balance_fields.py` / `check_b_path.py` 的完整输出 | 静态门控 |

打包示例（只回传必要文件，不要拉原始 trace）：

```bash
tar czf a5_round2_$(date +%m%d_%H%M).tgz \
    round2_*/summary.txt round2_*/0*_*/summary.txt round2_*/03_optional/*.txt \
    round2_*/03_optional/patched.log \
    prof_a5_*/summary.json prof_a5_*/windows.json prof_a5_*/order_rank0.json \
    prof_a5_*/export
```
