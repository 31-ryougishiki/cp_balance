#!/usr/bin/env bash
# desc:     长跑：持续突发 N 分钟，请求零失败、无 corrupted/preemption、无 plan_error、RSS 增长有界
# needs:    service
# tags:     npu, slow, service
# variants: svc_load_cp1
# est:      50min
set -uo pipefail
source "$(dirname "$0")/../lib/common.sh"

role=${1:-${HX_VARIANT:-svc_load_cp1}}
minutes=$(hx_service_kv soak_minutes); minutes=${minutes:-20}
sample_s=$(hx_service_kv soak_sample_s); sample_s=${sample_s:-60}
plan=$(hx_service_kv soak_plan); [ -n "$plan" ] || plan=configs/plans/load_soak.json
max_rss_mb=$(hx_service_kv max_rss_growth_mb); max_rss_mb=${max_rss_mb:-8192}
max_pre=$(hx_service_kv max_preemptions); max_pre=${max_pre:-0}
max_kv=$(hx_service_kv max_kv_usage_perc); max_kv=${max_kv:-0.99}
[ -f "$plan" ] || { hx_fail "soak plan missing: $plan"; hx_end; exit 1; }

svc_rss_kb() {
  python3 - <<PY
import subprocess
rows = subprocess.run(["ps", "-eo", "rss=,args="], capture_output=True, text=True).stdout.splitlines()
total = 0
for row in rows:
    if "vllm" in row or "run.sh" in row:
        total += int(row.split(None, 1)[0])
print(total)
PY
}

if ! hx_svc_up "$role"; then hx_fail "service $role not ready"; hx_end; exit 1; fi
hx_ok "soak start: $HX_CFG on $HX_BASE, ${minutes}min, plan=$plan"
printf "seconds\trss_kb\n" > "$HX_OUT/samples.tsv"
( while [ ! -f "$HX_OUT/stop_sampler" ]; do
    printf "%s\t%s\n" "$(date +%s)" "$(svc_rss_kb)" >> "$HX_OUT/samples.tsv"
    npu-smi info >> "$HX_OUT/npu_smi.txt" 2>&1 || true
    sleep "$sample_s"
  done ) &
sampler=$!

python3 tests/lib/loadgen.py run --url "$HX_BASE" --plan "$plan" --out "$HX_OUT/results.json" --duration $((minutes * 60)) --timeout 900 > "$HX_OUT/load.log" 2>&1
run_rc=$?
touch "$HX_OUT/stop_sampler"
kill "$sampler" 2>/dev/null
wait "$sampler" 2>/dev/null

code=$(hx_http_code "$HX_BASE/health")
if [ "$code" = "200" ]; then hx_ok "service still healthy at the end of the soak"; else hx_fail "health after soak -> $code"; fi

if [ -f "$HX_OUT/results.json" ]; then
  python3 tests/lib/loadgen.py check --in "$HX_OUT/results.json" --require-identical --max-preemptions "$max_pre" --max-kv-usage "$max_kv" --out "$HX_OUT/check.txt" > "$HX_OUT/check.log" 2>&1
  check_rc=$?
  sed "s/^/   /" "$HX_OUT/check.txt"
  case "$check_rc" in
    0)  hx_ok "soak traffic check PASS" ;;
    77) hx_warn "soak traffic check INCONCLUSIVE (metrics unreadable)" ;;
    *)  hx_fail "soak traffic check FAIL -> $HX_OUT/check.txt" ;;
  esac
  python3 - <<PY > "$HX_OUT/soak_summary.txt"
import json
data = json.load(open("$HX_OUT/results.json"))
sum = data["summary"]
print("requests=%d failed=%d rounds=%d wall_s=%s" % (sum["requests"], sum["failed"], sum["rounds"], sum["wall_s"]))
print("latency_s=%s" % sum["latency_s"])
print("counters=%s" % sum["metrics"]["counters"])
print("gauges_max=%s" % sum["metrics"]["gauges_max"])
print("averages=%s" % sum["metrics"]["averages"])
PY
  sed "s/^/   /" "$HX_OUT/soak_summary.txt"
else
  hx_fail "soak produced no results.json (rc=$run_rc) -> $HX_OUT/load.log"
fi

python3 - <<PY > "$HX_OUT/rss.txt"
rows = [line.split() for line in open("$HX_OUT/samples.tsv") if line.strip()]
values = [int(row[1]) for row in rows[1:]]
if values:
    print("samples=%d first_mb=%.0f last_mb=%.0f peak_mb=%.0f growth_mb=%.0f" % (
        len(values), values[0] / 1024.0, values[-1] / 1024.0, max(values) / 1024.0,
        (max(values) - min(values)) / 1024.0))
else:
    print("no RSS sample")
PY
cat "$HX_OUT/rss.txt"
growth=$(python3 - <<PY
rows = [line.split() for line in open("$HX_OUT/samples.tsv") if line.strip()]
values = [int(row[1]) for row in rows[1:]]
print(int((max(values) - min(values)) / 1024) if values else 0)
PY
)
if [ "${growth:-0}" -le "$max_rss_mb" ]; then
  hx_ok "RSS growth ${growth}MB within ${max_rss_mb}MB"
else
  hx_fail "RSS growth ${growth}MB above ${max_rss_mb}MB (see $HX_OUT/rss.txt)"
fi

if hx_service_down "$HX_PORT"; then hx_ok "service stopped"; else hx_fail "port $HX_PORT still answering after stop"; fi
cp "$HX_LOG" "$HX_OUT/service.log" 2>/dev/null || hx_warn "no service log at $HX_LOG"

cp_size=$(hx_cfg_field "$HX_CFG" tp_size)
python3 tests/lib/planlog.py audit "$HX_OUT/service.log" --cp-size "$cp_size" > "$HX_OUT/plan_audit.txt" 2>&1
audit_rc=$?
grep -F "planlog]" "$HX_OUT/plan_audit.txt" | sed "s/^/   /"
case "$audit_rc" in
  0)  hx_ok "soak plan audit PASS" ;;
  77) hx_warn "soak plan audit INCONCLUSIVE (no zigzag batch in the soak traffic)" ;;
  *)  hx_fail "soak plan audit FAIL -> $HX_OUT/plan_audit.txt" ;;
esac
hx_end
