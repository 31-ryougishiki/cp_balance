# cp_balance 验证与调优

两条独立的线，共用根目录的启动器与配置：

```
cp_balance/
  run.sh              启动入口（共用）
  serve_config.py     配置加载 -> 环境变量 + vllm serve 参数（共用）
  configs/            每条测试一份 JSON（共用）
  questions.json      20 组文章 + 问题 + prompt（共用）
  accuracy/           精度测试：首词元/文本相等、走哪条分支、静态路径证明
    compare_first_token.py    collect / compare 首词元
    check_branch.py           证明请求走 ZIGZAG 还是 CONTINUOUS
    check_b_path.py           静态证明 cp_balance 只作用于 zigzag 路径
    run_matrix.py / .sh       矩阵：串行起停服务、采集、对比、给裁定
  perf/               性能采集：profiling 采集、解析、对比、算子归因
    profile_forward.py        按长度逐档采集（只含 prefill 步的窗口）
    profile_analyse.py        解析 + 按窗口分组 + 审计每个窗口几步
    profile_compare.py        逐长度对比
    profile_order.py          算子调用顺序 + device kernel 归因到 host scope
    check_cp_balance_fields.py  静态自检三个容器的字段是否对得上
    profile.sh / profile_l6.sh  全量 / 6 层的一键入口
```

产物（`matrix_*/`、`prof_*/`、`prof_*.json`、`profile_*.log`）都写在仓库根目录。

## 配置化（两条线共用）：每次测试 = `configs/` 下一条 JSON

下面到「分支证明」之前的几节是精度测试的内容（首词元/文本对比、分支证明、
与 base 的等价性），profiling 那几节属于性能采集。

启动脚本只负责读 JSON。所有产物（服务日志、`*.json`、`matrix_*/`）默认写在**当前目录**，不再用 `/tmp`：

```bash
bash run.sh configs/glm52_cur_cp0.json      # 也可以只写名字
bash run.sh glm52_cur_cp0 --dry-run --print-env   # 只打印命令与环境
```

一条配置覆盖所有会变的参数：

| 字段 | 作用 |
| --- | --- |
| `repo` | 代码树（决定跑哪个分支/版本） |
| `model` / `served_model_name` | 权重路径 / 对外模型名 |
| `host` / `port` / `local_ip` / `nic_name` | 监听地址、端口、HCCL 网卡与 IP |
| `devices` / `tp_size` | 可见卡 / TP 并行度 |
| `cp_balance` / `min_tokens` / `reduce_mode` / `debug` | cp_balance 四个开关 |
| `deterministic` | 打开 4 个确定性环境变量（LCCL/HCCL/ATB 两项） |
| `additional_config` / `server_args` / `env` | 其余 serve 参数与环境变量 |
| `vllm_bin` | 启动命令，默认 `vllm`；可写 `[python3, -m, vllm.entrypoints.cli.main]` |
| `extends` | 继承另一个配置（dict 合并、list 覆盖），公共部分放 `_common.json` |

新加一个测试：复制一份 JSON，改 `name` 与需要变的字段即可（换模型改 `model`，
换机器改 `local_ip`/`nic_name`/`devices`/`port`，换并行度改 `tp_size`）。

矩阵（一次跑多组并给裁定）同样是 JSON：

```bash
bash accuracy/run_matrix.sh configs/matrix_b_vs_base.json      # B==base + 噪声地板
bash accuracy/run_matrix.sh configs/matrix_c_accept.json       # C 验收（首 token）
```

`configs` 字段列出要跑的配置，`compare` 列出对比项：`left`/`right` 用配置的 `name`，
`require_text` 决定是否把前 120 字符文本作为硬判据，`gate=true` 的项失败则整体 FAIL。

当前已有配置：

- `_common.json`：A3 / GLM-5.2-W4A8C8 / TP16 / eth2 的公共参数；
- `glm52_cur_cp0.json`（8034）、`glm52_cur_cp1.json`（8035）：当前分支 B/C；
- `glm52_base_cp0.json`（8036）、`glm52_base_cp0_repeat.json`（8037）：base 两次（噪声地板）；
- `matrix_b_vs_base.json` / `matrix_c_accept.json`：两套矩阵。
## 数据

`questions.json` 是唯一数据源，共 40 条，每条包含 `id`、`kind`、`article`、
`question` 和可直接发送的 `prompt`：

- `items[0:20]`：`kind=short`，prompt 长度全部小于 1000 字符；
- `items[20:40]`：`kind=long`，prompt 约 4400 字符，用于触发 CP_BALANCE。

`collect` 按 JSON 顺序发送，因此会先跑完 20 条短请求，再跑长请求。

请求方式与现场模板一致：

```bash
curl http://<node0_ip>:<port>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "glm-52",
        "prompt": "<questions.json 里的 prompt>",
        "max_completion_tokens": 50,
        "temperature": 0
    }'
```

`compare_first_token.py` 生成请求完全使用上面的模板，只替换 `prompt`；拿到
`choices[0].text` 后再调用 vLLM `/tokenize` 取第一个可见 token，避免比较原始格式 token。

`/tokenize` 的 `return_token_strs=true` 返回的是**词表条目**而不是解码后的文本：GLM-5.2
使用 byte-level BPE，一个中文 token 会显示成 `å¦Ĥæŀľ` 这样的字节视图（`如果` 的 UTF-8
字节 `E5 A6 82 E6 9E 9C`）。这只是 token 字符串的表示形式，不是请求乱码，curl 手发请求
也会有同样的现象。脚本会按 byte-level BPE 的字节表还原成 `如果`；如果某种实现改为
latin-1 形式，也会回退到 latin-1 解码。

## 流程（先短后长）

```bash
#  起 CP_BALANCE=1 的服务（参数见 configs/glm52_cur_cp1.json）
bash run.sh glm52_cur_cp1

#  先只发前 20 条短请求（<1000 字符）
python accuracy/compare_first_token.py collect \
    --url http://127.0.0.1:8034 \
    --kind short \
    --out cp_on_short.json

#  停服务，起 CP_BALANCE=0 的服务（configs/glm52_cur_cp0.json）
bash run.sh glm52_cur_cp0

python accuracy/compare_first_token.py collect \
    --url http://127.0.0.1:8035 \
    --kind short \
    --out cp_off_short.json

python accuracy/compare_first_token.py compare cp_on_short.json cp_off_short.json

#  短请求通过后，再跑长请求
python accuracy/compare_first_token.py collect --url http://127.0.0.1:8035 --kind long --out cp_off_long.json
# 重新起 CP_BALANCE=1 服务，再用 --kind long 采集 cp_on_long.json
# python accuracy/compare_first_token.py compare cp_on_long.json cp_off_long.json
```

短请求不会触发 `MIN_TOKENS=2048`，因此它们主要验证关闭/开启路径的基础行为；
短请求通过后再跑长请求，长请求才会真正进入 CP_BALANCE 和 owner-independent
归约。

判据：

```text
[compare] first-token match: 20/20
[compare] RESULT: PASS
```

失败时把对应 kind 的两个 JSON 和带 `VLLM_ASCEND_CP_BALANCE_DEBUG=1` 的 server log
一起回传。

## 归约模式

`VLLM_ASCEND_CP_BALANCE_REDUCE_MODE`：

- `allreduce`（默认）：AllReduce 后切本 rank chunk，owner-independent；
- `alltoall`：all_to_all_single + 固定 source-rank 求和，通信量更低；
- `reducescatter`：恢复原始 reduce_scatter，仅用于基线对照。

## 分支证明（走 C 还是 B）

`configs/glm52_cur_cp1.json` 里 `cp_balance=1`，但每个 batch 是否真的走 zigzag 由
metadata builder 逐 batch 判定。打开 `VLLM_ASCEND_CP_BALANCE_DEBUG=1` 后，
SFA metadata builder 会为**两条分支**各打一行 `[CP_BALANCE][branch]`（每 rank
每 batch 一次），所以“走哪条分支”由日志行本身回答，而不是靠“没有日志”推断。

```bash
bash run.sh glm52_cur_cp1 > cp_on.log 2>&1
# 另一个终端
python accuracy/check_branch.py --url http://127.0.0.1:8034 --log cp_on.log
```

判据：末行 `[check] RESULT: PASS`。脚本自己选一条短、一条长 prompt，打印各自的
token 数与日志证据：

- 短 prompt（<2048 token）→ 只有 `branch=CONTINUOUS`，`reason=actual<min(2048)`；
- 长 prompt（≥2048 token）→ 至少一行 `branch=ZIGZAG`，`reason=-`。

`[CP_BALANCE][branch]` 字段：
`rank / branch / reason / state / pad / actual / reqs / real / min_qlen / local / min_tokens`。
`reason` 是第一个拒绝 zigzag 的门（`flag_off`、`actual<min(N)`、`query_len<32`、
`state=DecodeOnly`、`dp>1`、`plan_error:...` 等）。

注意：DEBUG 打开后 decode 步也会打 `branch=CONTINUOUS reason=state=DecodeOnly`，
属正常现象；日志没有实时落盘时（重定向未带 `PYTHONUNBUFFERED=1`）脚本会报
"no branch line"。

## 非 cp_balance 路径与 base 一致（验证方案）

目标：`VLLM_ASCEND_CP_BALANCE=0` 时，当前分支与 `vllm-ascend-base`（原版 DSA-CP）
走同一套算子与集合通信。为此 cp_balance 专用逻辑改为按 **forward 级**的
`zigzag_active()` 生效，不再用配置级 `enable_dsa_cp()` 判断。

### 步骤 0：静态门控检查（秒级）

```bash
python accuracy/check_b_path.py --repo /opt/its/z30055003/vllm-ascend \
                       --base-repo /opt/its/z30055003/vllm-ascend-base
# 判据：末行 [check] RESULT: PASS
```

断言内容：两处 row-parallel 归约 + embedding/MoE finalize 归约都由
`zigzag_active()` 门控；裸的 `VLLM_ASCEND_CP_BALANCE` 只被资格判定读取；
`_q_proj_and_k_up_proj` 的融合算子块与 base 逐字节相同。

### 步骤 1：一条命令跑完四组实验（推荐）

```bash
bash accuracy/run_matrix.sh eth2 8034        # 第 3 个参数 det=1（默认）会开确定性变量
```

串行跑四组，每组服务起停一次、采 40 条 prompt：

| 组 | 代码树 | 设置 | 作用 |
| --- | --- | --- | --- |
| `cur_off` | 当前分支 | `CP_BALANCE=0` | 与 base 的**等价性主判据** |
| `cur_on` | 当前分支 | `CP_BALANCE=1` | C 验收（首 token） |
| `base_off` | base 分支 | — | 原版 DSA-CP |
| `base_off2` | base 分支 | — | **噪声地板**（同代码树重复一次） |

输出目录 `matrix_<时间戳>/`：`summary.txt`（每组指纹 + 归约计数 + 对比结论）、
`*.log`、`*.json`、`cmp_*.txt`。裁定：`R1` 与 `R3` 同时 PASS 才打印 `RESULT: PASS`；
`R3` FAIL 说明测量本身不可复现，先别谈代码差异。

可选环境变量：`VLLM_ASCEND_REPO_CUR` / `VLLM_ASCEND_REPO_BASE` / `RUNS`
（如 `RUNS=cur_off,base_off`）/ `OUT_DIR` / `READY_TIMEOUT`。

### 步骤 1-手动：分步等价命令

代码树与开关全部写在配置里（`repo` 字段），不需要命令行覆盖：

```bash
# 当前分支 CP_BALANCE=0（configs/glm52_cur_cp0.json，端口 8034）
bash run.sh glm52_cur_cp0 > cur_off.log 2>&1
python accuracy/compare_first_token.py collect --url http://127.0.0.1:8034 --out cur_off.json
# 停服务，换 base 代码树（configs/glm52_base_cp0.json，端口 8036）
bash run.sh glm52_base_cp0 > base_off.log 2>&1
python accuracy/compare_first_token.py collect --url http://127.0.0.1:8036 --out base_off.json
python accuracy/compare_first_token.py compare --require-text cur_off.json base_off.json
```

判据：`first-token match: 40/40` + `text_head match: 40/40` + 末行
`[compare] RESULT: PASS`（`--require-text` 把前 120 字符生成文本也变成硬判据）。

日志判据（证明 B 走的是原集合通信）：

```bash
grep -c "\[CP_BALANCE\]\[reduce\] path=native" cur_off.log        # > 0
grep -c "\[CP_BALANCE\]\[reduce\] path=fixed_order" cur_off.log   # == 0
grep -c "\[CP_BALANCE\]\[plan\]" cur_off.log                     # == 0
grep -c "branch=CONTINUOUS" cur_off.log                            # > 0
grep -m1 "\[cp_balance\] REPO=" cur_off.log                       # 确认代码树
```

失败时先做（先分清"测量问题"还是"代码差异"）：

```bash
# 1) 两次服务到底加载了哪个代码树 / 开了什么
grep -m1 "\[cp_balance\]" cur_off.log base_off.log
# 2) 运行期用了哪种归约（CP=0 必须是 path=native，不能出现 path=fixed_order）
grep -c "\[CP_BALANCE\]\[reduce\] path=fixed_order" cur_off.log
# 3) 两份 JSON 是不是本轮采集的
python -c "import json;[print(f, json.load(open(f))[\"url\"], json.load(open(f))[\"created_at\"]) for f in (\"cur_off.json\", \"base_off.json\")]"
```

判读：

- 两行的 `REPO=`/`HEAD=` 相同（或 base 那次仍是当前分支 HEAD）→ 代码树没切换，结果无效；
- `path=fixed_order` 出现在 `cur_off.log` → 跑的是改动前的代码（或误设了 `REDUCE_MODE`）；
- 两份 JSON 的 `created_at` 相差很远 → 用的是旧数据；
- 以上都正常而短 prompt（<2048 token，不进入 zigzag）仍不一致 → 说明还有未识别的差异，
  先跑一次"同代码树自比对"（base vs base，或 CP=0 跑两遍）确定测量是否可复现。

### 步骤 2：C 回归（补回融合算子后必须重跑）

当前分支补回了 base 的 `npu_transpose_batchmatmul`（`_q_proj_and_k_up_proj`），
C 的数值随之改变，原验收要重跑：

```bash
python accuracy/compare_first_token.py compare cp_on.json cur_off.json
# C 日志应同时有 [CP_BALANCE][plan] 与 [CP_BALANCE][reduce] path=fixed_order
```

## profiling：对比 cp_balance 开关下的 forward

四个配置串行跑，每个配置一次服务起停。每个配置按 config 里的 lengths
（1024/2048/4096/6144/8192/12288 token）各采一个窗口，每个窗口只含一个 prefill 步：
热身一条（不采）→ /start_profile → 一条 max_completion_tokens=1 的请求 → /stop_profile，
profiler 侧带 delay_iterations=0 / max_iterations=1。采完再按同一串长度跑一遍不采样的对照，
拿到真实端到端耗时。prompt 由 questions.json 的文章按 /tokenize 二分切成指定 token 数，
四个配置用同一段文本、同一组长度，所以可比。

```bash
cd /opt/its/z30055003/cp_balance
python3 perf/profile_forward.py prof_cur_cp0 prof_cur_cp1 prof_cur_cp1_a2a prof_base_cp0
```

| 配置 | 代码树 | CP_BALANCE | REDUCE_MODE | 作用 |
| --- | --- | --- | --- | --- |
| `prof_cur_cp0` | 当前 | 0 | — | 与 base 等价的噪声基准 |
| `prof_cur_cp1` | 当前 | 1 | allreduce | zigzag 现状（默认归约） |
| `prof_cur_cp1_a2a` | 当前 | 1 | alltoall | zigzag + 低通信量归约 |
| `prof_base_cp0` | base | 0 | — | 原版 DSA-CP 参照 |

`profile_forward.py` 起服务 → 2 条 long prompt 热身（不采集）→ `POST /start_profile`
→ 4 条 long prompt（串行，每条一个 prefill batch）→ `POST /stop_profile` → 停服务，
写 `prof_<name>.json`。profiling 由配置里的
`"profiler": {"enabled": true}` 生成 `--profiler-config`；`/start_profile` 只在
设置该参数后才存在。

解析必须在远端另起进程补跑：mp 后端下 worker 是 daemon 进程，torch_npu 的解析器
拒绝在 daemon 里解析，所以 /stop_profile 之后不会自动出现 ASCEND_PROFILER_OUTPUT。
`analyse()` 接收父目录并自己并行处理所有 `*_ascend_pt`，不要在一个进程里按 rank 循环调用。

采集完在远端解析（需要 torch_npu 与 CANN）：

```bash
python3 perf/profile_analyse.py prof_cur_cp0 prof_cur_cp1 prof_cur_cp1_a2a prof_base_cp0
python3 perf/profile_compare.py prof_cur_cp0 prof_cur_cp1
python3 perf/profile_compare.py prof_cur_cp1 prof_cur_cp1_a2a
```

要回传的：每个 profiler 目录下的 `summary.json`、windows.json（窗口与长度的对应关系）、
`export/`（每 rank 的小 CSV），仓库根目录的 `prof_*.json` 与 `profile_*.log` 的指纹行；
`*_ascend_pt` 原始 trace 不用拷。

判读先看 `profile_compare.py` 的第一张表：每个长度一行，`steps` 必须是 1（不是 1 会打
WARNING），然后只看绝对量——attention 时间、集合通信时间、算子总时间、不采样的 clean_s，四个数随
长度怎么走。不要发明比值去消漂移：上一轮同样代码路径的两个配置整体差 12%，这个噪声地板只能靠
重复一轮来量，不能靠除掉一个所谓不受影响的算子——zigzag 会改变各 rank 持有的 token 集合，MoE 的
路由分布跟着变，dispatch 时间本来就可能变。

判读顺序与性能假设见仓库根目录 `docs/perf_plan.md`。

采集、解析、收集三段现在由 `bash perf/profile.sh` 串起来，结束时自动调用 `perf/collect.py` 把清点/clean_s/compare/order/指纹打成一个 `collect_<时间戳>/*.tgz`（不含原始 trace），不用再手工 tar。

### 6 层快跑（只用于快速复看，不用于结论）

结论一律来自 `profile.sh` 的 78 层四组。6 层是给“改完一处后想快速再看一眼顺序和
归因”用的，由启动参数覆盖，不改模型目录：

    bash perf/profile_l6.sh

`_profile_l6_common.json` 里 `--hf-overrides` 设 `num_hidden_layers=6`，并把
`indexer_types` 按真实的 full/shared 周期裁成 6 项
`[full, shared, shared, shared, full, shared]`。不裁的话前 6 层全是 full，而真实模型
里多数层是 shared（`patch_deepseek_v2.py:57` 会跳过它们的 indexer），算子构成就失真了。
这样仍然保留 3 个 dense + 3 个 MoE、2 个 full indexer。

注意：6 层只能用来比“一层里各算子占多少”和“顺序”，不能拿来报绝对值——
每步固定开销（metadata、embedding、出口 gather、logits）的占比会被放大。
报绝对耗时仍用 `profile.sh` 的 78 层四组。

### 算子调用顺序 + 对应代码

    python3 perf/profile_order.py prof_l6_cur_cp1 --rank rank0
    python3 perf/profile_order.py prof_l6_cur_cp1 --rank rank0 --devices
    python3 perf/profile_order.py prof_l6_cur_cp1 --rank rank0 --trim

它流式读 `ASCEND_PROFILER_OUTPUT/trace_view.json`（不整文件加载），把主线程的 host 事件按时间
排开，自动找出重复的层周期，打印三张表：

- `head`：第一个完整层之前的非重复部分；
- `one decoder-layer cycle`：一层里按顺序调了什么、各花多少、累计占比；
- `cycle: share by name`：一层内按名字汇总的占比。

每个名字后面跟 `<-` 加来源：`record_function` 标签直接给 `文件:行号`，
`torch.ops.vllm.*` 给注册点，`Hccl*` 给对应集合通信原语的候选调用点。

`--devices` 多给两张表：`device time by enclosing host scope` 把每个 device kernel 归到包住
它的 host scope 上（同一份 trace，共用时钟），这是“耗时到底出在哪段代码”最直接的答案；
再给指定 step 的 device kernel 顺序。

前提是服务带了 `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1`——没有它 `record_function` 会退化成
nullcontext（`vllm/v1/utils.py:747`），trace 里就没有任何命名区间。`_profile_common.json` 已经设好了。
`--trim` 会写一份 `order_<rank>.json`（几十 KB），回传这个就够，不用拉原始 trace。

注意：profiling 配置里 `debug=0`（`[CP_BALANCE][plan]` 日志里有 `.tolist()`，会
触发 device→host 同步）。"是否真的走 zigzag"请用 `check_branch.py` 在非采集的
一轮里证明，不要靠 profiling 这一轮。

## A3 环境（16 卡，GLM-5.2-W4A8C8，7.246.78.75 / eth2）

一条命令：

```bash
# 0. 预演（10 秒，不启动服务）：确认解析出的就是 A3 配置
bash run.sh glm52_cur_cp0 --dry-run --print-env      # 期望 TP=16 NIC=eth2 IP=7.246.78.75

# 1. 精度：静态门控 + C 验收 + B 等价性 + 可选 A/B（约 1 小时）
bash accuracy/round2_verify_a3.sh

# 2. 性能：78 层四组（cp0 / cp1 / cp0_repeat / base_cp0）+ 解析 + 归因
bash perf/profile.sh
```

前置（两棵代码树都在同一台机器上）：

```bash
git -C /opt/its/z30055003/vllm-ascend pull           # 当前树，应含最新的 cp_balance 提交
git -C /opt/its/z30055003/vllm-ascend-base log -1    # base 树，应为 c7990e5e4
cd /opt/its/z30055003/cp_balance && git pull         # harness（本轮 driver + A5 配置）
```

与 A5 的差别：TP=16（cp_size=16，zigzag 每序列切 32 块）、模型是 W4A8C8（不是 mxfp4）、
**没有 MTP**、profiling 沿用确定性环境变量；配置是 `configs/_common.json` +
`configs/prof_*.json`，端口 8034~8037（prof_cur_cp0 与 repeat 同为 8034，串行不冲突）。
判据、失败排查与回传清单见 `docs/cp_balance_remote_checklist.md`。

## A5 环境（8 卡，GLM-5.2-w4a4c8-mxfp4）

A5 用独立的一套配置（`configs/*_a5*.json`），差别、判据、回传清单见 `docs/a5_test_plan.md`。
一条命令：

```bash
bash accuracy/round2_verify_a5.sh     # 精度：静态门控 + C 验收 + B 等价性 + 可选 A/B
bash perf/profile_a5.sh               # 性能：cp0 / cp1 / cp0_repeat / base_cp0 四组
```

A5 与 A3 的差异集中在 `configs/_common_a5.json`（TP=8、eth0、141.61.133.112、
`/mnt/share/weights/GLM-5.2-w4a4c8-mxfp4`、vendor 环境用 `prelude` 字段承载），
`configs/_common_a5_mtp.json` 在其之上打开 MTP（deepseek_mtp, 1 token）：精度组用它，性能组不用。
换机器（IP / 网卡不同）不用改配置：在那台机器的 shell 里导出 `CP_BALANCE_LOCAL_IP` / `CP_BALANCE_NIC_NAME`（需要时还有 `CP_BALANCE_DEVICES`）即可，例如
`export CP_BALANCE_LOCAL_IP=141.61.133.104 CP_BALANCE_NIC_NAME=eth2`；命令行 `--set` 优先于环境变量。

两处需要按现场确认：base 代码树路径（默认 `/home/z30055003/vllm-ascend-base`）与
`cp_balance` 仓库位置（脚本假定在当前目录运行）。

## 文件

| 文件 | 作用 |
| --- | --- |
| `run.sh` | 启动入口：读 `configs/*.json`，交给 `serve_config.py` |
| `serve_config.py` | 配置加载/继承/覆盖 → 环境变量 + `vllm serve` 参数（含 `--profiler-config`） |
| `configs/` | 每条测试一份 JSON（模型/ip/port/nic/tp/开关/profiler），含矩阵配置 |
| `accuracy/run_matrix.py` | 按矩阵 JSON 串行起停服务、采集、对比、给裁定 |
| `questions.json` | 20 组 article + question + prompt |
| `accuracy/compare_first_token.py` | collect / compare 首词元 |
| `accuracy/check_branch.py` | 证明请求走的是 ZIGZAG 还是 CONTINUOUS 分支 |
| `accuracy/check_b_path.py` | 静态证明 cp_balance 只作用于 zigzag 路径（B == 原版 DSA-CP） |
| `accuracy/run_matrix.sh` | `run_matrix.py` 的入口包装 |
| `accuracy/round2_verify_a5.sh` | 本轮验收，换 A5 的矩阵/端口/代码树 |
| `perf/profile_a5.sh` | A5 全量性能采集（78 层四组 + 噪声地板 + 解析 + 算子归因） |
| `configs/_common_a5.json` / `_common_a5_mtp.json` | A5 公共参数（后者额外打开 MTP） |
| `configs/matrix_a5_*.json` | A5 的两套矩阵（C 验收、B 等价性） |
| `accuracy/round2_verify.py` / `.sh` | 本轮验收一键跑：静态门控 + C 验收 + B 等价性 + 可选 A/B（临时补丁自动还原，产物在 `round2_<时间戳>/`） |
| `perf/profile_forward.py` | 每个配置一段 profiling 采集（起停服务 + start/stop_profile） |
| `perf/profile_analyse.py` | 远端跑 `torch_npu analyse` 并把 CSV 压成 `summary.json`（`export/` 只留三个小 CSV，`communication*.json` 不回传） |
| `perf/profile_compare.py` | 对比两份 `summary.json`：rank 间失衡、HCCL、算子差 |
| `perf/profile.sh` | 78 层全量一轮：自检 + 采集 + 解析 + **自动收集打包**（对比/顺序文本进包），结论来源 |
| `perf/collect.py` / `.sh` | 收集一轮 profiling 的结果：清点产物 → 逐长度 clean_s 表 → compare/order 文本 → 指纹 → 打包（不含原始 trace） |
| `perf/check_cp_balance_fields.py` | 静态自检 ZigzagPlan / meta dict / DSACPContext 三方字段是否对得上 |
| `perf/profile_order.py` | 算子调用顺序 + device kernel 归因到 host scope + 映射回 文件:行号 |
| `perf/profile_l6.sh` | 6 层快跑：只用于快速复看顺序与归因，不用于结论 |
| `_profile_l6_common.json` | 6 层覆盖（`--hf-overrides`），其余继承 `_profile_common.json` |
