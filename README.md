# cp_balance 首词元对比

这个目录只做一件事：用同一组足够长的中文问题，分别请求
`VLLM_ASCEND_CP_BALANCE=1` 和 `=0` 的 vLLM 服务，比较第一个生成词元是否相同。

## 一次性流程

在远端机器上按下面的顺序执行。每次只拉起一个服务，避免两张服务争抢同一批
NPU；`run.sh` 已经支持环境变量覆盖。

```bash
cd <本目录>

#  拉起 cp_balance 开启的服务
VLLM_ASCEND_CP_BALANCE=1 VLLM_ASCEND_CP_BALANCE_REDUCE_MODE=allreduce \
    bash run.sh eth2 8034
# 另一个终端执行采集
python compare_first_token.py collect \
    --url http://127.0.0.1:8034 \
    --out /tmp/cp_on.json

#  停掉 ，拉起 cp_balance 关闭的服务
VLLM_ASCEND_CP_BALANCE=0 VLLM_ASCEND_CP_BALANCE_REDUCE_MODE=allreduce \
    bash run.sh eth2 8035
python compare_first_token.py collect \
    --url http://127.0.0.1:8035 \
    --out /tmp/cp_off.json

#  比较首词元
python compare_first_token.py compare /tmp/cp_on.json /tmp/cp_off.json
```

判据：

- 每个案例打印 `token=...`；
- 末行 `[compare] RESULT: PASS`，并且 `first-token match: 20/20`；
- 若出现 `DIFF`，把屏幕输出和两个 JSON 一起贴回来。

## 模式说明

| env | 取值 | 说明 |
| --- | --- | --- |
| `VLLM_ASCEND_CP_BALANCE` | 0 / 1 | 是否启用 zigzag CP balance |
| `VLLM_ASCEND_CP_BALANCE_MIN_TOKENS` | 默认 2048 | 低于该预填充词元数不走 zigzag |
| `VLLM_ASCEND_CP_BALANCE_REDUCE_MODE` | `allreduce`(默认) / `alltoall` | 行并行归约的 owner 无关实现；默认 allreduce 更稳，alltoall 用于性能 A/B |
| `VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL` | 0(默认) / 1 | 实验性的嵌入入口，默认走模型边界 shard |

`questions.txt` 有 20 个有意义的中文问题；脚本会给每个问题补上同样的中文长
上下文，默认补到 8000 字符，确保超过 `MIN_TOKENS=2048`。采样固定为
`temperature=0, max_tokens=1`，所以首词元是实现的确定性函数。

## 文件

| 文件 | 作用 |
| --- | --- |
| `run.sh` | 原有启动脚本，现支持上述环境变量覆盖 |
| `compare_first_token.py` | 采集 / 比较首词元的唯一入口，纯标准库 |
| `questions.txt` | 20 个中文问题，每行一个 |

如果服务已经由别的脚本拉起，可以跳过 `run.sh`，直接执行 `collect` 和
`compare` 两条命令。
