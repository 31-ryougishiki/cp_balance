#!/usr/bin/env bash
# desc:     并发混合突发：长短请求同时发出，全部成功、同 prompt 回复一致、引擎计数干净（产物给 sv21/sv22）
# needs:    service
# tags:     npu, slow, service
# variants: svc_load_cp1 svc_load_cp0
# est:      20min
set -uo pipefail
source "$(dirname "$0")/../lib/common.sh"

role=${1:-${HX_VARIANT:-svc_load_cp1}}
plan=$(hx_service_kv plan)
[ -n "$plan" ] || plan=configs/plans/load_mixed.json
[ -f "$plan" ] || { hx_fail "traffic plan missing: $plan"; hx_end; exit 1; }
max_pre=$(hx_service_kv max_preemptions); max_pre=${max_pre:-0}
max_kv=$(hx_service_kv max_kv_usage_perc); max_kv=${max_kv:-0.99}

if ! hx_svc_up "$role"; then hx_fail "service $role not ready"; hx_end; exit 1; fi
hx_ok "service $HX_CFG ready on $HX_BASE (plan=$plan)"

python3 tests/lib/loadgen.py run --url "$HX_BASE" --plan "$plan" --out "$HX_OUT/results.json" --timeout 900 > "$HX_OUT/load.log" 2>&1
run_rc=$?
sed "s/^/   /" "$HX_OUT/load.log"
if [ -f "$HX_OUT/results.json" ]; then
  python3 tests/lib/loadgen.py check --in "$HX_OUT/results.json" --require-identical --max-preemptions "$max_pre" --max-kv-usage "$max_kv" --out "$HX_OUT/check.txt" > "$HX_OUT/check.log" 2>&1
  check_rc=$?
  sed "s/^/   /" "$HX_OUT/check.txt"
  case "$check_rc" in
    0)  hx_ok "burst check PASS" ;;
    77) hx_warn "burst check INCONCLUSIVE (see $HX_OUT/check.txt)" ;;
    *)  hx_fail "burst check FAIL -> $HX_OUT/check.txt" ;;
  esac
else
  hx_fail "loadgen produced no results.json (rc=$run_rc) -> $HX_OUT/load.log"
fi
[ "$run_rc" -eq 0 ] || hx_warn "loadgen run rc=$run_rc (request failures are reported by the check)"

printf "%s" "$HX_CFG" > "$HX_OUT/config.txt"
if hx_service_down "$HX_PORT"; then hx_ok "service stopped"; else hx_fail "port $HX_PORT still answering after stop"; fi
cp "$HX_LOG" "$HX_OUT/service.log" 2>/dev/null || hx_warn "no service log at $HX_LOG"
hx_end
