#!/usr/bin/env bash
# 6 层快跑：只用于快速复看算子顺序与归因，结论不取这一轮。
# 结论（绝对耗时、rank 间失衡、净收益）一律来自 profile.sh 的 78 层四组。
#
#   bash profile_l6.sh                              # 默认四组
#   bash profile_l6.sh prof_l6_cur_cp0 prof_l6_cur_cp1
#
# 6 层由 --hf-overrides 在启动时覆盖，不改模型目录：
#   num_hidden_layers = 6, indexer_types 按真实 full/shared 周期裁成 6 项。
# 用途是看一层里按什么顺序调了什么、各占多少，不是绝对值：
# 每步固定开销（metadata、embedding、出口 gather、logits）占比会被放大。
#
# 阶段之间刻意不互相阻塞，理由同 profile.sh。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

if [ "$#" -gt 0 ]; then
  CONFIGS="$*"
else
  CONFIGS="prof_l6_cur_cp0 prof_l6_cur_cp1 prof_l6_cur_cp1_a2a prof_l6_base_cp0"
fi

set -x
python3 check_cp_balance_fields.py --repo /opt/its/z30055003/vllm-ascend || true
python3 profile_forward.py $CONFIGS
set +x
set -x
python3 profile_analyse.py $CONFIGS
set +x

set -x
python3 profile_compare.py prof_l6_cur_cp0 prof_l6_cur_cp1 || true
python3 profile_compare.py prof_l6_cur_cp1 prof_l6_cur_cp1_a2a || true
python3 profile_compare.py prof_l6_cur_cp0 prof_l6_base_cp0 || true
python3 profile_order.py prof_l6_cur_cp1 --rank rank0 --trim || true
python3 profile_order.py prof_l6_cur_cp1 --rank rank0 --devices || true
set +x

echo "[profile] ---- artifacts ----"
for name in $CONFIGS; do
  if [ -f "$name/summary.json" ]; then
    echo "[profile] OK   $name/summary.json"
  else
    echo "[profile] MISS $name/summary.json"
  fi
done
