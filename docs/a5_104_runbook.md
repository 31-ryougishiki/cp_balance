# 141.61.133.104（A5 节点）精度 + 性能测试命令

对象：远程 A5 节点 `141.61.133.104`（机器族先用 §0 的探针判定）。
本文件 = 现场命令清单；差异背景与判据来源见 `docs/a5_test_plan.md`、`docs/cp_balance_remote_checklist.md`。
算子/字段一律写原名，见 §7 对照表。

> 统一入口：`bash verify.sh --family <a5|a3>`（前置检查 -> 静态 -> 冒烟 -> 分支诊断 -> 打包，步骤与判据来自 `harness.json`），
> 底层是 `tests/run_tests.sh`（`--list` 看全部、`--tag fast` 只跑不起服务的那批、`--from <id>` 失败后续跑、`--strict` 把 SKIP 也算失败）。
> 本文件剩的命令是（人肉版），与 `tests/smoke/*`、`tests/accuracy/*`、`tests/perf/*` 一一对应；分层与元数据约定见 `tests/README.md`。

约定：

- harness 仓库 `<harness>` 默认 `/home/z30055003/cp_balance`，不同就地替换；
- 产物：测试证据在 `tests/_out/<时间戳>/`，profiling 的 `prof_a5_*/`、`collect_*/` 落在 harness 根目录；
- 机器身份只从环境变量进来，不改配置，三套驱动都会继承：

| 环境变量 | 落到哪 |
| --- | --- |
| `CP_BALANCE_LOCAL_IP` | `HCCL_IF_IP` |
| `CP_BALANCE_NIC_NAME` | `GLOO_SOCKET_IFNAME` / `TP_SOCKET_IFNAME` / `HCCL_SOCKET_IFNAME` |
| `CP_BALANCE_DEVICES` | `ASCEND_RT_VISIBLE_DEVICES` |

---

## 0. 机器族判定（10 秒，决定用哪套配置）

| 族 | 芯片 | 卡数 / TP | 代码树 | 权重 | 入口 |
| --- | --- | --- | --- | --- | --- |
| A5 | Ascend950（`soc_version = 260`） | 8 | `/home/z30055003/vllm-ascend` | `/mnt/share/weights/GLM-5.2-w4a4c8-mxfp4` | `bash verify.sh --family a5`（或 `CP_BALANCE_FAMILY=a5 bash tests/run_tests.sh`） |
| A3 | 910_9391（`250 <= soc_version <= 255`） | 16 | `/opt/its/z30055003/vllm-ascend` | `/opt/its/model/GLM-5.2-W4A8C8` | `bash verify.sh --family a3` |

`env.log` 是 A3 那台的（`SOC_VERSION=ascend910_9391`、`PWD=/opt/its/z30055003`、CANN 9.1.0）；
若 `.104` 就是它，走 A3 那套。探针：

```bash
cd /home/z30055003/cp_balance
python3 -c "import torch_npu; print('soc_version =', torch_npu.npu.get_soc_version())"
npu-smi info | head -12
ip -o -4 addr show | awk '{print $2, $4}'          # 141.61.133.104 落在哪张网卡
ls -d /home/z30055003/vllm-ascend /home/z30055003/vllm-ascend-base 2>&1
ls -d /mnt/share/weights/GLM-5.2-* 2>&1
```

机器身份：配置里默认是 `auto`（自动识别）；要固定成现场这台的具体值就 export 下面两个变量
（不 export 也能跑，`verify.sh` 会打 `[note] 识别到 NIC/IP ...`；裸 `run.sh` 识别不到只打 WARN，且不设
`HCCL_IF_IP` / `*_SOCKET_IFNAME`）：

```bash
export CP_BALANCE_LOCAL_IP=141.61.133.104
export CP_BALANCE_NIC_NAME=eth2        # 换成上面 ip 命令查到的网卡
```

判成 A3 族时，§2 / §3 的入口换成（`export CP_BALANCE_FAMILY=a3` + 本机 IP/网卡）：

```bash
cd /opt/its/z30055003/cp_balance
export CP_BALANCE_FAMILY=a3 CP_BALANCE_LOCAL_IP=7.246.78.75 CP_BALANCE_NIC_NAME=eth2
bash verify.sh --family a3                                                # 一键（含冒烟与诊断）
bash tests/run_tests.sh --from accuracy                                   # §2 精度
bash tests/run_tests.sh --only perf --skip perf/p10_capture#prof_a2a      # §3 性能
```

---

## 1. 前置：同步 harness、对齐两棵代码树

```bash
cd /home/z30055003/cp_balance
git pull
git log -1 --oneline

git -C /home/z30055003/vllm-ascend      log -1 --oneline   # 待验树（cp_balance 分支；verify.sh 会自动对齐到 origin 上的 tip）
git -C /home/z30055003/vllm-ascend-base log -1 --oneline   # 参照树 = main + trees.base.patches（dp=1 门补丁），也由 verify.sh 对齐

df -h .                                                    # profiling 会写 GB 级 trace_view.json
```

指纹自检（不启动服务，只打印 argv 与环境）：

```bash
bash run.sh glm52_a5_cur_cp1 --dry-run --print-env | head -8
```

要看到 `[cp_balance] CONFIG=... REPO=/home/z30055003/vllm-ascend HEAD=... MODEL=... TP=8 NIC=eth2
IP=141.61.133.104 DEVICES=0,...,7 CP_BALANCE=1 MIN_TOKENS=2048 REDUCE_MODE=allreduce DEBUG=1
DET=True LAYERS=all PROFILER=off PRELUDE=True SPEC=deepseek_mtp/1`，且 argv 以
`bash -c 'source /mnt/share/.../set_env.bash && ... exec vllm serve ...'` 开头。

本机权重不是配置里那一个时（例如挂的是 `GLM-5.2-W4A8C8`），只改一处：

```bash
python3 - <<'PY'
import json, pathlib
p = pathlib.Path("configs/_common_a5.json")
cfg = json.loads(p.read_text(encoding="utf-8"))
cfg["model"] = "/mnt/share/weights/GLM-5.2-W4A8C8"     # 换成 probe 到的路径
p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("model ->", cfg["model"])
PY
bash run.sh glm52_a5_cur_cp1 --dry-run | head -1        # 指纹行的 MODEL= 应更新
```

`enable_sparse_sfa_c8` / `enable_sparse_li_c8` 在 `configs/_base.json` 的 `additional_config`（A3/A5 共用），
不是族配置；换量化后若起不来或首 token 不一致，先用 `--set additional_config.enable_sparse_*_c8=false` 剥离定位：

```bash
bash run.sh glm52_a5_cur_cp1 --set additional_config.enable_sparse_sfa_c8=false \
                             --set additional_config.enable_sparse_li_c8=false
```

---

## 2. 精度测试

### 2.1 冒烟（约 10 分钟）

```bash
cd /home/z30055003/cp_balance
setsid bash run.sh glm52_a5_cur_cp1 > smoke_cp1.log 2>&1 &
until curl -sf http://127.0.0.1:8035/v1/models >/dev/null; do sleep 10; done

python3 - <<'PY'
import json, urllib.request
items = json.load(open("questions.json", encoding="utf-8"))["items"]
item = [i for i in items if i["kind"] == "long"][0]
req = urllib.request.Request(
    "http://127.0.0.1:8035/v1/completions",
    data=json.dumps({"model": "glm-52", "prompt": item["prompt"],
                     "max_tokens": 1, "temperature": 0}).encode(),   # pinned vLLM 的 /v1/completions 只认 max_tokens
    headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req, timeout=1800))["choices"][0]["text"][:60])
PY

grep -c "\[CP_BALANCE\]\[branch\] branch=ZIGZAG" smoke_cp1.log     # 或 [CP_BALANCE][plan] 行数
grep -c "\[CP_BALANCE\]\[branch\] branch=CONTINUOUS" smoke_cp1.log
grep -E "Disabling DSA-CP|DSA-CP is enabled without" smoke_cp1.log   # 应只出现后者（dp=1 的门修）
pkill -f -- "--port 8035"          # serve_config 自己转发信号，兜底仍可按端口杀
until ! curl -sf http://127.0.0.1:8035/v1/models >/dev/null 2>&1; do sleep 5; done
```

判据：zigzag 证据计数 >= 1（`branch=ZIGZAG` 或 `[CP_BALANCE][plan]` 行；长 prompt 2768~2781 token > `MIN_TOKENS=2048`）；
否则 `bash run.sh glm52_a5_cur_cp1 --set min_tokens=1536` 重跑，并在结论里记下阈值。
另：expected 的启动日志是 `DSA-CP is enabled without sequence-parallel MoE (data_parallel_size=1)`；
若看到 `Disabling DSA-CP`，说明这棵树没带 dp=1 的门修。draft 步应有 `reason=draft`。

### 2.2 正式（静态门控 + C 验收 + B 等价性 + 可选 A/B）

```bash
cd /home/z30055003/cp_balance
bash tests/run_tests.sh --list                       # 先看清单与机器族
bash tests/run_tests.sh --only smoke/s06_static_fields,smoke/s07_static_b_path,accuracy/a10_matrix_gate --keep-going
bash tests/run_tests.sh --only accuracy/a20_compare_collected        # 离线复跑对比（秒级）
bash tests/run_tests.sh --only accuracy/a30_slot_filter_ab           # 可选：slot < 0 过滤 A/B（现状 SKIP：driver 锚点过期）
```

| 步骤 | 内容 | 服务起停 | 判据 |
| --- | --- | --- | --- |
| 0 | `perf/check_cp_balance_fields.py` + `accuracy/check_b_path.py` | 0 | `[check] RESULT: PASS`（ZigzagPlan fields=13 受门控；DSACPContext 实测 19 但无门控） |
| 1 | C 验收：`glm52_a5_cur_cp1` vs `glm52_a5_cur_cp0`，40 条 prompt | 2 | `[compare] first-token match: 40/40`；`[matrix] RESULT: PASS` |
| 2 | B 等价性：`glm52_a5_cur_cp0` vs `glm52_a5_base_cp0`，另加 `glm52_a5_base_cp0_repeat`（噪声地板） | 3 | 两项都 `RESULT: PASS`（`--require-text`） |
| 3 | ~~可选：临时补丁 A/B~~（`accuracy/a30_slot_filter_ab`） | 0 | 现状 SKIP：`round2_verify.py` 的补丁锚点失效，已定案不修，不排机时 |

只跑最便宜的 C 验收（2 次起停）：

```bash
bash tests/run_tests.sh --only accuracy/a10_matrix_gate#c_matrix
bash tests/run_tests.sh --only accuracy/a10_matrix_gate#nomtp_matrix    # 怀疑 MTP 干扰时
```

产物在 `tests/_out/<时间戳>/accuracy/a10_matrix_gate.<变体>/matrix/`，
`summary.txt` 是全部判定；总账在 `tests/_out/<时间戳>/status.tsv`。

---

## 3. 性能测试

```bash
cd /home/z30055003/cp_balance
df -h .

bash tests/run_tests.sh --list | grep p10_capture                       # 看变体
bash tests/run_tests.sh --only perf/p10_capture#prof_cp0,perf/p10_capture#prof_cp1   # 先快跑两组：证明采得到
bash tests/run_tests.sh --only perf --skip perf/p10_capture#prof_a2a     # 四组：cp0 / cp1 / cp0_repeat / base_cp0
bash tests/run_tests.sh --only perf                                       # 含 §4 归约 A/B（多一组 prof_a2a）
```

采集口径：每个配置一次服务起停（10 分钟级），组内按 `lengths=[1024, 2048, 4096, 6144, 8192, 12288]`
逐档「warmup → `/start_profile` → 单请求 → `/stop_profile` → settle 45s → 不采样的对照」。
解析必须在远端独立进程里跑（mp 后端 worker 是 daemon，torch_npu 解析器拒绝在 daemon 里跑），
`tests/run_tests.sh --only perf` 已按这个顺序串好 采集 → 解析 → 对比/归因 → `perf/collect.py` 打包。

判读（顺序不能反）：

```bash
# 1) 噪声地板：同一代码路径重跑一遍
python3 perf/profile_compare.py prof_a5_cur_cp0 prof_a5_cur_cp0_repeat

# 2) 主对比
python3 perf/profile_compare.py prof_a5_cur_cp0 prof_a5_cur_cp1
python3 perf/profile_compare.py prof_a5_cur_cp0 prof_a5_base_cp0

# 3) 算子顺序 + device kernel 归因
python3 perf/profile_order.py prof_a5_cur_cp1 --rank rank0
python3 perf/profile_order.py prof_a5_cur_cp1 --rank rank0 --devices

# 4) 打包回传（顺带删原始 trace）
python3 perf/collect.py --prune-traces
```

四条判据：

1. **噪声地板**：`prof_a5_cur_cp0` vs `prof_a5_cur_cp0_repeat` 的差值就是本轮噪声；
   要下的结论小于噪声时不下结论（A3 那轮轮间漂移 5%~13%）。
2. **先核次数再谈时间**：看 `op_statistic.csv` 的 `OP Type` 计数。非 zigzag 一侧只有
   `reduce_scatterAicpuKernel`；zigzag 一侧应恰好少 N 次 `reduce_scatterAicpuKernel`、
   多 N 次 `allreduceAicpuKernel`（`reduce_mode=alltoall` 时是 `alltoallAicpuKernel`），
   N = 4 × 窗口内 prefill 步数（3 个 dense 层 `down_proj` + embedding）。次数对不上说明配置没生效。
3. **窗口必须是单步**（判据 `steps=1`）：当前 trace 的 `kernel_details.csv` 没有 Step 列，`kernel_steps` 恒 None，
   所以 `perf/p21_window_single_step` 恒 SKIP —— 这一条现在判不了（`scripts_review.md` §3 待定）。
4. **端到端数字看 `clean_s`**（关 profiler 的同请求），`profiled_s` 只说明 profiling 自身开销。

---

## 4. 归约 A/B

### 4.1 是什么

把 cp_balance 的跨 rank 归约实现换成另一种等价写法（同一正确性目标），只比"对不对 + 快不快"。
开关是 `VLLM_ASCEND_CP_BALANCE_REDUCE_MODE`，配置字段是 `reduce_mode`；
进程内由 `reduce_mode()` 的 `lru_cache` 读一次，**A/B 必须各起一次服务**。

对应配置：`prof_a5_cur_cp1`（端口 8085，`allreduce`，默认）vs
`prof_a5_cur_cp1_a2a`（端口 8086，`alltoall`），两者只差这一个字段。

### 4.2 为什么需要它

zigzag 均衡会把 token 行在 CP=TP 的各 rank 之间重新分配：每个 rank 拿到的行不再等于它自己请求的那段。
row-parallel 层的输出必须跨 rank 求和后切成 1/cp 段交给该段的 owner，而 `dist.reduce_scatter_tensor`
（HCCL 的 `reduce_scatterAicpuKernel`）的累加/舍入顺序依赖 chunk 的接收者。于是同一批 per-token
部分和，在 B（`CP_BALANCE=0`）与 C（`=1`）里可能舍入出不同的 bf16 值。

所以 zigzag 生效时改走 `fixed_order_reduce_scatter`：求和顺序只由 token 身份决定，与 owner 无关。
挂载点（都在 `zigzag_active()` 之后，非 zigzag 时逐字节走 base）：
`ops/linear_op.py` 的 `MLPRowParallelOp`（MLP，日志 `path=native site=mlp`）、`ops/linear_op.py` 的 o_proj/down_proj 分支、
`ops/fused_moe/shared_experts.py` 的共享专家出口（行号会随重构漂，按类名/函数名找）。
（sequence，日志 `path=native site=sequence`）、`ops/register_custom_ops.py:41`。

### 4.3 三种模式

| `reduce_mode` | 实现 | 用途 |
| --- | --- | --- |
| `allreduce`（默认） | `_allreduce_slice_reduce_scatter`：整张量 `dist.all_reduce`，再切本地 chunk | 当前正确性目标；HCCL 上最稳 |
| `alltoall` | `_all_to_all_fixed_order_reduce_scatter`：`dist.all_to_all_single` 交换 chunk，再按 source rank `0..N-1` 顺序 `fixed_order_rank_sum` | 通信量更低，但用的是较不常见的集合通信，每个 SoC 都要另验 —— A/B 的另一臂 |
| `reducescatter` | `_plain_reduce_scatter`：原生 `dist.reduce_scatter_tensor`，与 base 等价 | 调试用：复现"改之前"的基线 |

### 4.4 怎么跑、看什么

```bash
bash tests/run_tests.sh --only perf/p10_capture#prof_cp1,perf/p10_capture#prof_a2a
bash tests/run_tests.sh --only perf/p20_analyse,perf/p21_window_single_step,perf/p22_report_compare,perf/p23_report_order,perf/p24_collect
```

判据同 §3 的四条，其中"次数"的期望值按模式换：`allreduce` 臂应看到 `allreduceAicpuKernel` 增加 N 次，
`alltoall` 臂应看到 `alltoallAicpuKernel` 增加 N 次、`allreduceAicpuKernel` 为 0。

### 4.5 上一轮实测（A3，rank0 窗口）

来自 `data/send/export/*op_statistic.csv`，算子名即 `OP Type` 原名：

| 配置 | `reduce_scatterAicpuKernel` | `allreduceAicpuKernel` | 备注 |
| --- | --- | --- | --- |
| `cur_cp0` | 4936 次 / 10.13 s | 0 | 与 `base_cp0` 条数完全相同 |
| `cur_cp1` | 4920 次（−16）/ 11.47 s | 16 次 / 12.2 ms | 同一条 `reduce_scatterAicpuKernel` 自身慢 13% |
| `cur_cp1_a2a` | 4920 次（−16）/ 10.83 s | `alltoallAicpuKernel` +16 | |

口径：16 次归约在窗口里只值 12 ms（0.018%），而同一算子跨轮波动 13% —— 轮间漂移比效应大两个数量级，
所以 `alltoall` 只作可选 A/B 保留，**不要据此改默认值**。要下结论就每臂至少 2 轮，且只比同一轮内的差值。

### 4.6 别和另外两个 A/B 混

| 名字 | 变量 | 测什么 |
| --- | --- | --- |
| cp0 vs cp1 | `CP_BALANCE=0/1` | zigzag 均衡本身的功能收益（性能主对比） |
| 归约 A/B | `reduce_mode=allreduce/alltoall` | 归约实现的通信代价（都在 `CP_BALANCE=1` 下） |
| 精度步骤 3 的 A/B | 临时补丁 | KV 写过滤 `slot < 0` 的**正确性**，与性能无关 |

---

## 5. 回传打包

```bash
# 远端
cd /home/z30055003/cp_balance
tar czf a5_104_accuracy_$(date +%m%d_%H%M).tgz \
    tests/_out/*/status.tsv tests/_out/*/results.json tests/_out/*/smoke tests/_out/*/accuracy
tar czf a5_104_perf_$(date +%m%d_%H%M).tgz collect_*/ prof_a5_*/summary.json prof_a5_*/windows.json \
    prof_a5_*/order_rank0.json prof_a5_*/export
ls -lh a5_104_*.tgz

# 本地（Git Bash）
scp <user>@141.61.133.104:/home/z30055003/cp_balance/a5_104_*.tgz /d/code/cp_balance/log/
```

这里的"精度"是首 token/文本一致性 + 分支与 B 等价性证明，不是数据集打分；
数据集精度（AIME 等）另走评测工具，本 harness 不覆盖。

---

## 6. 失败时先看哪里

| 症状 | 先看 |
| --- | --- |
| 服务 2400s 没起来 | `tests/_out/<时间戳>_<family>/accuracy/a10_matrix_gate.<变体>/matrix/<配置名>.log` 尾部；先核对指纹行（IP / NIC / PRELUDE / MODEL）与 `df -h`、`npu-smi info` |
| `WARNING: no [cp_balance] fingerprint` | 起服务没走 `bash run.sh`，或配置里的 `repo_tree` 在 `harness.json trees` 里找不到对应当前树 |
| 步骤 1 首 token 不一致 | ① 两侧指纹行（同树同开关）② zigzag 证据（`branch=ZIGZAG` 或 `[CP_BALANCE][plan]`）③ 跑 `bash tests/run_tests.sh --only accuracy/a10_matrix_gate#nomtp_matrix` ④ `bash run.sh glm52_a5_cur_cp1 --set additional_config.enable_sparse_li_c8=false` 剥离定位 |
| 步骤 2 噪声地板 FAIL | 测量不可复现，不是代码问题：重跑一次确认 |
| 步骤 3 两版首 token 不同 | `slot < 0` 没被 scatter 跳过 -> 记真实问题，KV 写要改掩码 scatter |
| 窗口 `steps != 1` | 采样窗口夹带了 decode：该档作废重采，检查 `--max-completion-tokens` 与 settle |
| 两次性能差异 < 噪声 | 不要下结论，加长度档或重复轮次 |

---

## 7. 原名对照表

| 场景 | 原名 |
| --- | --- |
| 集合通信算子（`op_statistic.csv` 的 `OP Type`） | `reduce_scatterAicpuKernel`、`allreduceAicpuKernel`、`alltoallAicpuKernel`、`allgatherAicpuKernel`、`alltoallvAicpuKernel` |
| 代码里的集合通信调用 | `dist.reduce_scatter_tensor`、`dist.all_reduce`、`dist.all_to_all_single`、`all_gather_into_tensor` |
| cp_balance 归约实现 | `fixed_order_reduce_scatter`、`_allreduce_slice_reduce_scatter`、`_all_to_all_fixed_order_reduce_scatter`、`_plain_reduce_scatter`、`fixed_order_rank_sum`、`reduce_mode()` |
| 归约挂载点 | `ops/linear_op.py`（`site=mlp` / `site=sequence`）、`ops/register_custom_ops.py`（`maybe_pad_and_reduce`） |
| 环境变量 / 配置字段 | `VLLM_ASCEND_CP_BALANCE`、`VLLM_ASCEND_CP_BALANCE_MIN_TOKENS`、`VLLM_ASCEND_CP_BALANCE_REDUCE_MODE`、`VLLM_ASCEND_CP_BALANCE_DEBUG`；配置字段 `cp_balance` / `min_tokens` / `reduce_mode` / `debug` |
| 服务日志行 | `[cp_balance] CONFIG=...`（指纹）、`[CP_BALANCE][branch]`、`[CP_BALANCE][plan]`、`[CP_BALANCE][reduce]`（`[CP_BALANCE][group]` 只在当年那个临时补丁里，当前树没有） |
| profiler 产物 | `op_statistic.csv`、`api_statistic.csv`、`kernel_details.csv`、`step_trace_time.csv`（列 `Computing` / `Communication(Not Overlapped)` / `Overlapped` / `Free` / `Bubble`）、`trace_view.json` |
| harness 产物 | `summary.json`、`windows.json`、`order_rank0.json`、`export/`、`clean_s`、`profiled_s`、`collect_<时间戳>/` |
