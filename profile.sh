#!/usr/bin/env bash
# 一轮 profiling：采集 -> 解析 -> 对比。
#
#   bash profile.sh                                  # 默认四组
#   bash profile.sh prof_cur_cp0 prof_cur_cp1         # 只跑两组
#
# 服务启动约 10 分钟/组，四组约 45 分钟。产物：
#   prof_<name>/summary.json   要回传的对比数据（几 KB）
#   prof_<name>.json           客户端时延与指纹
#   profile_<name>.log         服务日志
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
python3 profile_compare.py prof_cur_cp0 prof_cur_cp1
python3 profile_compare.py prof_cur_cp1 prof_cur_cp1_a2a
python3 profile_compare.py prof_cur_cp0 prof_base_cp0
