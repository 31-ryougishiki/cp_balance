#!/usr/bin/env bash
# 启动服务：所有参数来自 configs/*.json
#
#   bash run.sh configs/glm52_cur_cp0.json
#   bash run.sh glm52_cur_cp0                       # 只写名字也行
#   bash run.sh glm52_cur_cp0 --set port=8035       # 临时覆盖字段
#   bash run.sh glm52_cur_cp0 --dry-run --print-env # 只打印将执行的命令与环境
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CFG=$1
[ -n "$CFG" ] || CFG=default
shift || true
exec python3 "$HERE/serve_config.py" "$CFG" "$@"
