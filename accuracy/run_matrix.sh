#!/usr/bin/env bash
# 精度测试的矩阵入口。用法：
#   bash accuracy/run_matrix.sh configs/matrix_b_vs_base.json
# 产物（matrix_* 目录）落在仓库根目录。
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
cd "$ROOT"
exec python3 "$HERE/run_matrix.py" "$@"
