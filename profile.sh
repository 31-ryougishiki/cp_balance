#!/usr/bin/env bash
# 全量（78 层）一轮 profiling：自检 -> 采集 -> 解析 -> 对比 -> 算子顺序归因。
#
#   bash profile.sh                                   # 默认四组
#   bash profile.sh prof_cur_cp0 prof_cur_cp1         # 只跑两组
#   bash profile.sh prof_cur_cp0 prof_cur_cp1 prof_cur_cp0_repeat prof_base_cp0 prof_cur_cp1_a2a
#
# 每组一次服务起停（约 10 分钟），组内按 config 的 lengths 逐档采集：
# 每档一个只含 prefill 步的窗口，再加一遍不采样的对照。单组约 15 分钟。
#
# 顺序有意排成 cp0 -> cp1 -> cp0_repeat：
#   上一轮同样代码路径的两个配置整体差 12%，比要测的效应还大。既然不再用比值消漂移，
#   噪声地板就只能靠"同一配置再跑一遍"来量，所以 repeat 必须在默认流程里。
#   prof_base_cp0 是跨代码树的等价性对照，留在最后。
#   prof_cur_cp1_a2a（归约模式 A/B）默认不跑：上一轮已经证明它只替换 16 次集合通信。
#
# 跑之前先看磁盘：上一轮 16 rank 的原始 trace 是 3 GB/rank 量级，这一轮
# max_iterations=1 会把窗口压到 1~2 步，应该小一两个数量级，但仍要留空间。
#   df -h /opt/its/z30055003
#
# 阶段之间刻意不互相阻塞：某个配置挂了不中断后面；对比缺数据只跳过那一条；
# 顺序归因失败也不影响已经算出来的对比结论。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

if [ "$#" -gt 0 ]; then
  CONFIGS="$*"
else
  CONFIGS="prof_cur_cp0 prof_cur_cp1 prof_cur_cp0_repeat prof_base_cp0"
fi

# ---- 阶段 0：静态自检（秒级，先拦掉 plumbing 级别的低级错误）----
set -x
python3 check_cp_balance_fields.py --repo /opt/its/z30055003/vllm-ascend || true
set +x

# ---- 阶段 1：采集 ----
set -x
python3 profile_forward.py $CONFIGS
set +x

# ---- 阶段 2：解析（按窗口分组，并审计每个窗口几步）----
set -x
python3 profile_analyse.py $CONFIGS
set +x

# ---- 阶段 3：对比 ----
set -x
python3 profile_compare.py prof_cur_cp0 prof_cur_cp1        || true   # 主判据
python3 profile_compare.py prof_cur_cp0 prof_cur_cp0_repeat || true   # 噪声地板
python3 profile_compare.py prof_cur_cp0 prof_base_cp0       || true   # 跨代码树等价性
python3 profile_compare.py prof_cur_cp1 prof_cur_cp1_a2a    || true   # 只在跑了 a2a 时有意义
set +x

# ---- 阶段 4：算子调用顺序 + device 归因（默认取最长的那一档窗口）----
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
  for extra in windows.json order_rank0.json; do
    [ -f "$name/$extra" ] && echo "[profile] OK   $name/$extra"
  done
done
echo "[profile] 回传：每个目录的 summary.json / windows.json / export/，加上 log 里的指纹行"
