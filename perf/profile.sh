#!/usr/bin/env bash
# 全量（78 层）一轮性能采集：自检 -> 采集 -> 解析 -> 收集打包。
#
#   bash perf/profile.sh                                   # 默认四组
#   bash perf/profile.sh prof_cur_cp0 prof_cur_cp1         # 只跑两组
#
# 每组一次服务起停（约 10 分钟），组内按 config 的 lengths 逐档采集：
# 每档一个只含 prefill 步的窗口，再加一遍不采样的对照。
#
# 顺序有意排成 cp0 -> cp1 -> cp0_repeat -> base_cp0：
#   上一轮同样代码路径的两个配置整体差 12%，比要测的效应还大。噪声地板只能靠
#   "同一配置再跑一遍"来量，所以 repeat 必须在默认流程里。
#   prof_base_cp0 是跨代码树的等价性对照，留在最后。
#   prof_cur_cp1_a2a（归约模式 A/B）默认不跑：上一轮已证明它只替换 16 次集合通信。
#
# 采集完自动执行 perf/collect.py：清点产物 -> 逐长度 clean_s 表 -> compare/order
# 文本 -> 指纹 -> 打包成 collect_<时间戳>/<目录名>.tgz（永不包含原始 trace）。
# 想顺手删掉原始 trace（每 rank 上 GB）就单独跑：
#   python3 perf/collect.py --prune-traces prof_cur_cp0 prof_cur_cp1 ...
#
# 跑之前先看磁盘：
#   df -h /opt/its/z30055003
set -uo pipefail
PERF=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$PERF/.." && pwd)
cd "$ROOT"

if [ "$#" -gt 0 ]; then
  CONFIGS="$*"
else
  CONFIGS="prof_cur_cp0 prof_cur_cp1 prof_cur_cp0_repeat prof_base_cp0"
fi

set -x
python3 "$PERF/check_cp_balance_fields.py" --repo /opt/its/z30055003/vllm-ascend || true
python3 "$PERF/profile_forward.py" $CONFIGS
python3 "$PERF/profile_analyse.py" $CONFIGS
python3 "$PERF/collect.py" $CONFIGS || true
set +x

echo "[profile] ---- artifacts ----"
for name in $CONFIGS; do
  if [ -f "$name/summary.json" ]; then
    echo "[profile] OK   $name/summary.json"
  else
    echo "[profile] MISS $name/summary.json"
  fi
done
echo "[profile] 回传：collect_<时间戳>/ 下的 .tgz（内含 inventory/clean_s/cmp_*/order_*/fingerprints）"
