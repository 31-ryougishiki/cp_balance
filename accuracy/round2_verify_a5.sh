#!/usr/bin/env bash
# A5 环境的一轮验收：同一套 driver，换 A5 的矩阵与配置。
#   bash accuracy/round2_verify_a5.sh                 # 全跑（静态 + C 验收 + B 等价性 + 可选 A/B）
#   bash accuracy/round2_verify_a5.sh --steps 0,1,2   # 跳过可选 A/B
#   bash accuracy/round2_verify_a5.sh --dry-run       # 只打印计划
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
exec python3 "$HERE/round2_verify.py"     --c-matrix configs/matrix_a5_c_accept.json     --b-matrix configs/matrix_a5_b_vs_base.json     --optional-config glm52_a5_cur_cp1     "$@"
