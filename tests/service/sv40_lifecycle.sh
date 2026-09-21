#!/usr/bin/env bash
# desc:     生命周期：请求在飞时停服 -> 端口与进程清干净 -> 重启后仍能服务
# needs:    service
# tags:     npu, slow, service
# variants: svc_load_cp1
# est:      30min
set -uo pipefail
source "$(dirname "$0")/../lib/common.sh"

role=${1:-${HX_VARIANT:-svc_load_cp1}}
plan=$(hx_service_kv inflight_plan); [ -n "$plan" ] || plan=configs/plans/load_inflight.json
[ -f "$plan" ] || { hx_fail "in-flight plan missing: $plan"; hx_end; exit 1; }

if ! hx_svc_up "$role"; then hx_fail "service $role not ready"; hx_end; exit 1; fi
port=$HX_PORT
hx_ok "service $HX_CFG ready on $HX_BASE"

python3 tests/lib/loadgen.py run --url "$HX_BASE" --plan "$plan" --out "$HX_OUT/inflight.json" --timeout 900 > "$HX_OUT/inflight.log" 2>&1 &
client=$!
sleep 20
if kill -0 "$client" 2>/dev/null; then
  hx_ok "a request is still in flight when the stop begins"
else
  hx_warn "the in-flight request finished before the stop; the stop was not exercised under load"
fi
procs_before=$(pgrep -f -- "--port $port" | wc -l)
hx_note "processes matching --port $port while running: $procs_before"

if hx_service_down "$port"; then hx_ok "service stopped while a request was in flight"; else hx_fail "port $port still answering after stop"; fi
wait "$client" 2>/dev/null
client_rc=$?
hx_note "in-flight client rc=$client_rc (0 = completed, 1 = request failed; both are acceptable)"

procs_after=$(pgrep -f -- "--port $port" | wc -l)
if [ "${procs_after:-0}" -eq 0 ]; then
  hx_ok "no leftover process for port $port"
else
  hx_fail "$procs_after process(es) still match --port $port"
  pgrep -af -- "--port $port" > "$HX_OUT/leftovers.txt" 2>&1
fi
listen=$($HX_PY listen "$port")
if [ "$listen" = "0" ]; then hx_ok "port $port is free after the stop"; else hx_fail "port $port still listening"; fi

if ! hx_svc_up "$role"; then hx_fail "restart of $role failed"; hx_end; exit 1; fi
hx_ok "restart OK on $HX_BASE"
plan2=$(hx_service_kv plan); [ -n "$plan2" ] || plan2=configs/plans/load_mixed.json
python3 tests/lib/loadgen.py run --url "$HX_BASE" --plan "$plan2" --out "$HX_OUT/after_restart.json" --timeout 900 > "$HX_OUT/after_restart.log" 2>&1
rc=$?
python3 tests/lib/loadgen.py check --in "$HX_OUT/after_restart.json" --require-identical --out "$HX_OUT/after_restart_check.txt" > /dev/null 2>&1
check_rc=$?
if [ "$rc" -eq 0 ] && [ "$check_rc" -eq 0 ]; then
  hx_ok "the restarted service serves the burst plan"
else
  hx_fail "after restart: loadgen rc=$rc check rc=$check_rc -> $HX_OUT/after_restart.log"
fi
if hx_service_down "$HX_PORT"; then hx_ok "service stopped"; else hx_fail "port $HX_PORT still answering after stop"; fi
hx_end
