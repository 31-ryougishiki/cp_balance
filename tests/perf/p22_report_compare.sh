#!/usr/bin/env bash
# desc:     报告：逐长度/算子/集合通信对比表（cp1 vs cp0、cp0 vs repeat、cp0 vs base）
# needs:    none
# tags:     fast, perf, report
# est:      20s
set -uo pipefail
source "$(dirname "$0")/../lib/common.sh"

pairs="prof_cp0:prof_cp1 prof_cp0:prof_cp0_repeat prof_cp0:prof_base"
any=0
for pair in $pairs; do
  a=$(hx_role "${pair%%:*}"); b=$(hx_role "${pair##*:}")
  [ -n "$a" ] && [ -n "$b" ] || continue
  out=$HX_OUT/cmp_${a}_vs_${b}.txt
  if [ -f "$a/summary.json" ] && [ -f "$b/summary.json" ]; then
    python3 perf/profile_compare.py "$a" "$b" > "$out" 2>&1
    rc=$?
    any=$((any + 1))
    if [ "$rc" -eq 0 ]; then
      hx_ok "compare $a vs $b -> $out"
    else
      hx_fail "compare $a vs $b rc=$rc -> $out"
    fi
  else
    hx_note "skip $a vs $b (summary.json missing)"
  fi
done
[ "$any" -gt 0 ] || hx_skip "nothing to compare yet"
hx_note "判读：差值要大于噪声地板（cp0 vs cp0_repeat）"
hx_end
