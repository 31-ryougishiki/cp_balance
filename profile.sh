#!/usr/bin/env bash
# 全量（78 层）一轮 profiling：采集 -> 解析 -> 算子顺序归因 -> 对比。
#
#   bash profile.sh                                  # 默认四组
#   bash profile.sh prof_cur_cp0 prof_cur_cp1         # 只跑两组
#
# 服务启动约 10 分钟/组，四组约 45 分钟。产物：
#   prof_<name>/summary.json      要回传的对比数据（几 KB）
#   prof_<name>/export/           每 rank 的小 CSV 与 communication.json
#   prof_<name>/order_rank0.json  算子顺序与 device 归因（几十 KB）
#   prof_<name>.json              客户端时延与指纹
#   profile_<name>.log            服务日志
#
# 跑之前先看两次磁盘：4 组 x 16 rank x (原始 + 解析) 会占不少空间。
#   df -h /opt/its/z30055003
#
# 这是给出结论的那一轮（绝对耗时、rank 间失衡、净收益）。
# 6 层的 profile_l6.sh 只用来在改动后快速复看顺序与归因，不能拿它报结论。
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

if [ "$#" -gt 0 ]; then
  CONFIGS="$*"
else
  CONFIGS="prof_cur_cp0 prof_cur_cp1 prof_cur_cp1_a2a prof_base_cp0"
fi

set -x
python3 profile_forward.py $CONFIGS
python3 profile_analyse.py $CONFIGS

# 算子调用顺序 + device kernel 归因到 host scope（只用 rank0 的 trace）
python3 profile_order.py prof_cur_cp1 --rank rank0 --trim
python3 profile_order.py prof_cur_cp0 --rank rank0 --trim
python3 profile_order.py prof_cur_cp1 --rank rank0 --devices
python3 profile_order.py prof_cur_cp0 --rank rank0 --devices

python3 profile_compare.py prof_cur_cp0 prof_cur_cp1
python3 profile_compare.py prof_cur_cp1 prof_cur_cp1_a2a
python3 profile_compare.py prof_cur_cp0 prof_base_cp0
