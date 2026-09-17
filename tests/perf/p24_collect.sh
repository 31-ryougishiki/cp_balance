#!/usr/bin/env bash
# desc:     打包回传：collect.py 生成 collect_<时间戳>/<名字>.tgz（不含原始 trace）
# needs:    none
# tags:     fast, perf
# est:      2min
set -uo pipefail
source "$(dirname "$0")/../lib/common.sh"

cfgs=""
for cfg in $(hx_group prof); do
  [ -f "$cfg/summary.json" ] || continue
  cfgs="$cfgs $cfg"
done
[ -n "$cfgs" ] || hx_skip "no prof summary.json yet (run perf/p10_capture + perf/p20_analyse first)"

python3 perf/collect.py $cfgs --out "$HX_OUT/collect" > "$HX_OUT/collect.log" 2>&1
rc=$?
grep -E "^\[collect\]" "$HX_OUT/collect.log" | tail -n 12 | sed 's/^/   /'
tgz=$(ls "$HX_OUT"/collect/*.tgz 2>/dev/null | head -1)
if [ -n "$tgz" ]; then
  hx_ok "bundle $(du -h "$tgz" | cut -f1) -> $tgz"
else
  hx_fail "no bundle produced rc=$rc -> $HX_OUT/collect.log"
fi
hx_end
