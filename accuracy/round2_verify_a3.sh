#!/usr/bin/env bash
# A3 环境（16 卡，7.246.78.75 / eth2，GLM-5.2-W4A8C8）的一轮验收：
# 与 A5 共用同一套 driver，只是固定 A3 的矩阵与配置。
#   bash accuracy/round2_verify_a3.sh                 # 全跑
#   bash accuracy/round2_verify_a3.sh --steps 0,1,2   # 跳过可选 A/B
#   bash accuracy/round2_verify_a3.sh --dry-run       # 只打印计划
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
exec python3 "$HERE/round2_verify.py" \
    --c-matrix configs/matrix_c_accept.json \
    --b-matrix configs/matrix_b_vs_base.json \
    --optional-config glm52_cur_cp1 \
    "$@"
