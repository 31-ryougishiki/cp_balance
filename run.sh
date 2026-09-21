#!/usr/bin/env bash
# 启动服务：所有参数来自 configs/*.json（serve_config.py 的默认 config 也是 default）
#
#   bash run.sh configs/glm52_cur_cp0.json
#   bash run.sh glm52_cur_cp0                       # 只写名字也行
#   bash run.sh glm52_cur_cp0 --set port=8035       # 临时覆盖字段
#   bash run.sh glm52_cur_cp0 --dry-run --print-env # 只打印将执行的命令与环境
#   bash run.sh --dry-run glm52_cur_cp0             # 参数顺序随意
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
exec python3 "$HERE/serve_config.py" "$@"
