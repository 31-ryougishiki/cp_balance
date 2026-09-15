#!/usr/bin/env bash
# 全量（78 层）一轮 profiling：采集 -> 解析 -> 对比 -> 算子顺序归因。
#
#   bash profile.sh                                  # 默认四组
#   bash profile.sh prof_cur_cp0 prof_cur_cp1         # 只跑两组
#
# 服务启动约 10 分钟/组，四组约 50 分钟（含 trace flush）。
# 产物：
#   prof_<name>/summary.json      要回传的对比数据（几 KB）
#   prof_<name>/export/           每 rank 的小 CSV 与 communication.json
#   prof_<name>/order_rank0.json  算子顺序与 device 归因（几十 KB）
#   prof_<name>.json              客户端时延与指纹
#   profile_<name>.log            服务日志
#
# 跑之前先看磁盘：4 组 x 16 rank 的原始 + 解析产物。
#   df -h /opt/its/z30055003
#
# 这是给出结论的那一轮（绝对耗时、rank 间失衡、净收益）。
# 6 层的 profile_l6.sh 只用来在改动后快速复看顺序与归因。
#
# 阶段之间刻意不互相阻塞：某一个配置挂了不会中断后面；对比缺数据只跳过那一条；
# 顺序归因依赖 trace_view.json，失败也不影响已经算出来的对比结论。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

if [ "$#" -gt 0 ]; then
  CONFIGS="$*"
else
  CONFIGS="prof_cur_cp0 prof_cur_cp1 prof_cur_cp1_a2a prof_base_cp0"
fi

# ---- 阶段 0：静态自检（秒级，先拦掉 plumbing 级别的低级错误）----
set -x
python3 check_cp_balance_fields.py --repo /opt/its/z30055003/vllm-ascend || true
set +x

# ---- 阶段 1：采集（每个配置的失败在 profile_forward.py 内部记录并继续）----
set -x
python3 profile_forward.py $CONFIGS
set +x

# ---- 阶段 2：解析 ----
set -x
python3 profile_analyse.py $CONFIGS
set +x

# ---- 阶段 3：对比（主判据，先跑）----
set -x
python3 profile_compare.py prof_cur_cp0 prof_cur_cp1 || true
python3 profile_compare.py prof_cur_cp1 prof_cur_cp1_a2a || true
python3 profile_compare.py prof_cur_cp0 prof_base_cp0 || true
set +x

# ---- 阶段 4：算子调用顺序 + device 归因到 host scope（只用 rank0 的 trace）----
set -x
python3 profile_order.py prof_cur_cp1 --rank rank0 --trim || true
python3 profile_order.py prof_cur_cp0 --rank rank0 --trim || true
python3 profile_order.py prof_cur_cp1 --rank rank0 --devices || true
python3 profile_order.py prof_cur_cp0 --rank rank0 --devices || true
set +x

echo "[profile] ---- artifacts ----"
for name in $CONFIGS; do
  if [ -f "$name/summary.json" ]; then
    echo "[profile] OK   $name/summary.json"
  else
    echo "[profile] MISS $name/summary.json"
  fi
  [ -f "$name/order_rank0.json" ] && echo "[profile] OK   $name/order_rank0.json"
done
