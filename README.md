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
部分 vLLM 版本返回 latin-1 形式的 token 字符串（如 `åĤæŀľ`），脚本会自动重新解码为 UTF-8。

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
| `selftest_plan.py` | CPU 自测 zigzag plan 的覆盖、置换、equal-shape |
