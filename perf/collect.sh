#!/usr/bin/env bash
# 一次性收集一轮 profiling 的结果并打包（不拷贝原始 trace）。
#   bash perf/collect.sh                                   # 自动：所有已有 summary.json 的配置
#   bash perf/collect.sh prof_cur_cp0 prof_cur_cp1         # 指定配置
#   bash perf/collect.sh --prune-traces                    # 打包后删掉原始 trace（腾空间）
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
exec python3 "$HERE/collect.py" "$@"
