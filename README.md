# cp_balance 首词元对比

用 20 组中文文章 + 问题长 prompt，分别请求 `VLLM_ASCEND_CP_BALANCE=1` 和
`=0` 的服务，比较第一个生成词元。

## 数据

`questions.json` 是唯一数据源，共 20 条，每条包含 `id`、`article`、
`question` 和可直接发送的 `prompt`。修改或新增问题时只改这个 JSON。

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

## 流程

```bash
#  起 CP_BALANCE=1 的服务，并打开 debug 确认进入 zigzag
VLLM_ASCEND_CP_BALANCE=1 \
VLLM_ASCEND_CP_BALANCE_REDUCE_MODE=allreduce \
VLLM_ASCEND_CP_BALANCE_DEBUG=1 \
bash run.sh eth2 8034

#  采集 20 个首词元
python compare_first_token.py collect \
    --url http://127.0.0.1:8034 \
    --out /tmp/cp_on.json

#  停服务，起 CP_BALANCE=0 的服务
VLLM_ASCEND_CP_BALANCE=0 \
VLLM_ASCEND_CP_BALANCE_REDUCE_MODE=allreduce \
bash run.sh eth2 8035

python compare_first_token.py collect \
    --url http://127.0.0.1:8035 \
    --out /tmp/cp_off.json

#  比较
python compare_first_token.py compare /tmp/cp_on.json /tmp/cp_off.json
```

判据：

```text
[compare] first-token match: 20/20
[compare] RESULT: PASS
```

失败时把两个 JSON 和 `VLLM_ASCEND_CP_BALANCE_DEBUG=1` 的 server 日志一起回传。

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
