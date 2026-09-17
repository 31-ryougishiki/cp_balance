#!/usr/bin/env bash
# desc:     报告：算子调用顺序 + device kernel 归因（需要原始 trace）
# needs:    none
# tags:     slow, perf, report
# est:      10min
set -uo pipefail
source "$(dirname "$0")/../lib/common.sh"

any=0
for role in prof_cp1 prof_cp0; do
  cfg=$(hx_role "$role") || continue
  [ -n "$cfg" ] || continue
  if ls -d "$cfg"/*_ascend_pt >/dev/null 2>&1; then
    out=$HX_OUT/order_${cfg}.txt
    python3 perf/profile_order.py "$cfg" --rank rank0 --trim > "$out" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
      hx_ok "order $cfg -> $out"
    else
      hx_fail "order $cfg rc=$rc -> $out"
    fi
    python3 perf/profile_order.py "$cfg" --rank rank0 --devices \
      > "$HX_OUT/order_${cfg}_devices.txt" 2>&1 || true
    any=$((any + 1))
  else
    hx_note "skip $cfg (raw trace already pruned)"
  fi
done
[ "$any" -gt 0 ] || hx_skip "no raw trace left (collect.py --prune-traces already ran?)"
hx_end
