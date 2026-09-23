#!/usr/bin/env bash
# desc:     单组 profiling 采集
# needs:    profiler
# tags:     npu, slow, perf
# variants: prof_cp0 prof_cp1 prof_cp0_repeat prof_base
# est:      40min
set -uo pipefail
source "$(dirname "$0")/../lib/common.sh"

role=${1:-${HX_VARIANT:-prof_cp1}}
cfg=$(hx_resolve "$role") || { hx_fail "unknown variant: $role"; hx_end; exit 1; }
profdir=$(hx_profdir "$cfg")
[ -n "$profdir" ] || { hx_fail "$cfg: no profiler dir"; hx_end; exit 1; }

python3 perf/profile_forward.py "$cfg" > "$HX_OUT/forward.log" 2>&1
rc=$?
grep -E "^\[profile\]" "$HX_OUT/forward.log" | tail -n 12 | sed 's/^/   /'
if [ ! -f "$profdir/windows.json" ]; then
  hx_fail "$cfg: no windows.json rc=$rc -> $HX_OUT/forward.log"
  hx_end
  exit 1
fi
n=$($HX_PY windows "$profdir" | wc -l)
if [ "$n" -ge 1 ]; then
  hx_ok "$cfg: $n capture window(s)"
else
  hx_fail "$cfg: windows.json has no window"
fi
if grep -q "WARNING expected about" "$HX_OUT/forward.log"; then
  hint=$(grep -m1 "WARNING expected about" "$HX_OUT/forward.log")
  hx_warn "$cfg: rank dir count below expectation ($hint)"
fi
[ "$rc" -eq 0 ] || hx_fail "$cfg: profile_forward rc=$rc"
hx_end
