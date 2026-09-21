#!/usr/bin/env bash
# desc:     计划审计（不起服务）：每个 zigzag step 的 pad/actual/local 在所有 rank 上一致、无 plan_error、cp0 不出计划行
# needs:    none
# tags:     fast, service
# est:      1min
set -uo pipefail
source "$(dirname "$0")/../lib/common.sh"

min_reqs=$(hx_service_kv min_reqs_per_step); min_reqs=${min_reqs:-2}
checked=0
for role in svc_load_cp1 svc_load_cp0; do
  cfg=$(hx_resolve "$role") || continue
  [ -n "$cfg" ] || continue
  dir=$(hx_prev_out service/sv20_traffic "$role")
  if [ -z "$dir" ] || [ ! -f "$dir/service.log" ]; then
    hx_note "$role: no service.log (run tests/run_tests.sh --only service/sv20_traffic first)"
    continue
  fi
  checked=$((checked + 1))
  zig=$(grep -c -F "[CP_BALANCE][plan]" "$dir/service.log" || true)
  if [ "${zig:-0}" -eq 0 ]; then
    if [ "$role" = "svc_load_cp1" ]; then
      hx_fail "$role: cp_balance=1 but the log has no zigzag plan line -> $dir/service.log"
    else
      hx_ok "$role: cp_balance=0, no zigzag plan line (as expected)"
    fi
    continue
  fi
  if [ "$role" = "svc_load_cp0" ]; then
    hx_fail "$role: cp_balance=0 but the log has $zig zigzag plan line(s)"
    continue
  fi
  cp_size=$(hx_cfg_field "$cfg" tp_size)
  python3 tests/lib/planlog.py audit "$dir/service.log" --cp-size "$cp_size" --min-reqs-per-step "$min_reqs" --json "$HX_OUT/plan_$role.json" > "$HX_OUT/plan_$role.txt" 2>&1
  rc=$?
  sed "s/^/   /" "$HX_OUT/plan_$role.txt"
  case "$rc" in
    0)  hx_ok "$role: plan audit PASS (cp_size=$cp_size, min reqs/step=$min_reqs)" ;;
    77) hx_skip "$role: plan audit has no evidence (see $HX_OUT/plan_$role.txt)" ;;
    *)  hx_fail "$role: plan audit FAIL -> $HX_OUT/plan_$role.txt" ;;
  esac
done
[ "$checked" -gt 0 ] || hx_skip "no service/sv20_traffic evidence in $HARNESS_OUT"
hx_end
