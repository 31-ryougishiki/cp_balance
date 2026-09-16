#!/usr/bin/env bash
# A5 全量（78 层）一轮性能采集：自检 -> 采集 -> 解析 -> 收集打包。
#
#   bash perf/profile_a5.sh                                   # 默认四组
#   bash perf/profile_a5.sh prof_a5_cur_cp0 prof_a5_cur_cp1   # 只跑两组
#   bash perf/profile_a5.sh prof_a5_cur_cp0 prof_a5_cur_cp1 prof_a5_cur_cp1_a2a prof_a5_base_cp0
#
# 与 A3 的 perf/profile.sh 的差别（结论不能沿用 A3 的 profile 数据）：
#   * 代码树 /home/z30055003/vllm-ascend，TP=8，eth0，141.61.133.112，8 张卡；
#   * 模型是 GLM-5.2-w4a4c8-mxfp4（A3 是 W4A8C8）：数值路径与算子构成都不同；
#   * profiling 组关掉了 MTP（不带 --speculative-config），只测目标模型的 prefill 步；
#   * 关掉了确定性环境变量（A3 的 profiling 继承了 deterministic=true）。
#   * 换机器（IP/网卡不同）不用改配置，先 export CP_BALANCE_LOCAL_IP / CP_BALANCE_NIC_NAME。
#
# 顺序 cp0 -> cp1 -> cp0_repeat -> base_cp0：repeat 是噪声地板，不能省。
# 采集完自动执行 perf/collect.py 打包（清点/clean_s/compare/order/指纹，不含原始 trace）。
# 跑之前先看磁盘：
#   df -h .
set -uo pipefail
PERF=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$PERF/.." && pwd)
cd "$ROOT"

if [ "$#" -gt 0 ]; then
  CONFIGS="$*"
else
  CONFIGS="prof_a5_cur_cp0 prof_a5_cur_cp1 prof_a5_cur_cp0_repeat prof_a5_base_cp0"
fi

set -x
python3 "$PERF/check_cp_balance_fields.py" --repo /home/z30055003/vllm-ascend || true
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
