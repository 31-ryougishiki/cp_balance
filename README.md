# cp_balance 首词元对比

用 20 组中文文章 + 问题长 prompt，分别请求 `VLLM_ASCEND_CP_BALANCE=1` 和
`=0` 的服务，比较第一个生成词元。

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
#  起 CP_BALANCE=1 的服务
VLLM_ASCEND_CP_BALANCE=1 \
VLLM_ASCEND_CP_BALANCE_REDUCE_MODE=allreduce \
VLLM_ASCEND_CP_BALANCE_DEBUG=1 \
bash run.sh eth2 8034

#  先只发前 20 条短请求（<1000 字符）
python compare_first_token.py collect \
    --url http://127.0.0.1:8034 \
    --kind short \
    --out /tmp/cp_on_short.json

#  停服务，起 CP_BALANCE=0 的服务
VLLM_ASCEND_CP_BALANCE=0 \
VLLM_ASCEND_CP_BALANCE_REDUCE_MODE=reducescatter \
bash run.sh eth2 8035

python compare_first_token.py collect \
    --url http://127.0.0.1:8035 \
    --kind short \
    --out /tmp/cp_off_short.json

python compare_first_token.py compare /tmp/cp_on_short.json /tmp/cp_off_short.json

#  短请求通过后，再跑长请求
python compare_first_token.py collect --url http://127.0.0.1:8035 --kind long --out /tmp/cp_off_long.json
# 重新起 CP_BALANCE=1 服务，再用 --kind long 采集 /tmp/cp_on_long.json
# python compare_first_token.py compare /tmp/cp_on_long.json /tmp/cp_off_long.json
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

`run.sh` 默认 `VLLM_ASCEND_CP_BALANCE=1`，但每个 batch 是否真的走 zigzag 由
metadata builder 逐 batch 判定。打开 `VLLM_ASCEND_CP_BALANCE_DEBUG=1` 后，
SFA metadata builder 会为**两条分支**各打一行 `[CP_BALANCE][branch]`（每 rank
每 batch 一次），所以“走哪条分支”由日志行本身回答，而不是靠“没有日志”推断。

```bash
VLLM_ASCEND_CP_BALANCE=1 VLLM_ASCEND_CP_BALANCE_DEBUG=1 PYTHONUNBUFFERED=1 \
    bash run.sh eth2 8034 > /tmp/cp_on.log 2>&1
# 另一个终端
python check_branch.py --url http://127.0.0.1:8034 --log /tmp/cp_on.log
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
python check_b_path.py --repo /opt/its/z30055003/vllm-ascend \
                       --base-repo /opt/its/z30055003/vllm-ascend-base
# 判据：末行 [check] RESULT: PASS
```

断言内容：两处 row-parallel 归约 + embedding/MoE finalize 归约都由
`zigzag_active()` 门控；裸的 `VLLM_ASCEND_CP_BALANCE` 只被资格判定读取；
`_q_proj_and_k_up_proj` 的融合算子块与 base 逐字节相同。

### 步骤 1：CP_BALANCE=0 vs base（主判据）

`run.sh` 的代码树由 `VLLM_ASCEND_REPO` 决定（默认当前分支目录）。起 base
分支时必须覆盖它，否则加载的仍是当前分支代码：

```bash
# 当前分支：CP_BALANCE=0，且不设 REDUCE_MODE（验证"默认即一致"）
VLLM_ASCEND_CP_BALANCE=0 VLLM_ASCEND_CP_BALANCE_DEBUG=1 PYTHONUNBUFFERED=1 \
    VLLM_ASCEND_REPO=/opt/its/z30055003/vllm-ascend \
    bash run.sh eth2 8034 > /tmp/cur_off.log 2>&1
python compare_first_token.py collect --url http://127.0.0.1:8034 --out /tmp/cur_off.json
# 停服务，换 base 代码树（同一个 run.sh）
VLLM_ASCEND_CP_BALANCE=0 PYTHONUNBUFFERED=1 \
    VLLM_ASCEND_REPO=/opt/its/z30055003/vllm-ascend-base \
    bash run.sh eth2 8035 > /tmp/base_off.log 2>&1
python compare_first_token.py collect --url http://127.0.0.1:8035 --out /tmp/base_off.json
python compare_first_token.py compare --require-text /tmp/cur_off.json /tmp/base_off.json
```

判据：`first-token match: 40/40` + `text_head match: 40/40` + 末行
`[compare] RESULT: PASS`（`--require-text` 把前 120 字符生成文本也变成硬判据）。

日志判据（证明 B 走的是原集合通信）：

```bash
grep -c "\[CP_BALANCE\]\[reduce\] path=native" /tmp/cur_off.log        # > 0
grep -c "\[CP_BALANCE\]\[reduce\] path=fixed_order" /tmp/cur_off.log   # == 0
grep -c "\[CP_BALANCE\]\[plan\]" /tmp/cur_off.log                     # == 0
grep -c "branch=CONTINUOUS" /tmp/cur_off.log                            # > 0
grep -m1 "\[cp_balance\] REPO=" /tmp/cur_off.log                       # 确认代码树
```

### 步骤 2：C 回归（补回融合算子后必须重跑）

当前分支补回了 base 的 `npu_transpose_batchmatmul`（`_q_proj_and_k_up_proj`），
C 的数值随之改变，原验收要重跑：

```bash
python compare_first_token.py compare /tmp/cp_on.json /tmp/cur_off.json
# C 日志应同时有 [CP_BALANCE][plan] 与 [CP_BALANCE][reduce] path=fixed_order
```

## plan 自测

在远端 vLLM 环境执行，不需要 NPU：

```bash
python selftest_plan.py --cp-size 16 --cases 2000
```

判据：末行 `SELFTEST PLAN OK`。

## 文件

| 文件 | 作用 |
| --- | --- |
| `run.sh` | 启动脚本，支持 `VLLM_ASCEND_CP_BALANCE`、`..._REDUCE_MODE`、`..._DEBUG` 覆盖 |
| `questions.json` | 20 组 article + question + prompt |
| `compare_first_token.py` | collect / compare 首词元 |
| `check_branch.py` | 证明请求走的是 ZIGZAG 还是 CONTINUOUS 分支 |
| `check_b_path.py` | 静态证明 cp_balance 只作用于 zigzag 路径（B == 原版 DSA-CP） |
| `selftest_plan.py` | CPU 自测 zigzag plan 的覆盖、置换、equal-shape |
