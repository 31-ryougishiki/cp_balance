#!/usr/bin/env bash
# 用法：bash run_matrix.sh configs/matrix_b_vs_base.json
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
exec python3 "$HERE/run_matrix.py" "$@"
