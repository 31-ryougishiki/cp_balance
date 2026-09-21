#!/usr/bin/env bash
# desc:     cp0/cp1 同计划一致性（不起服务）：并发突发下同一条请求的两条路径回复必须逐字相同
# needs:    none
# tags:     fast, service
# est:      1min
set -uo pipefail
source "$(dirname "$0")/../lib/common.sh"

dir1=$(hx_prev_out service/sv20_traffic svc_load_cp1)
dir0=$(hx_prev_out service/sv20_traffic svc_load_cp0)
if [ -z "$dir1" ] || [ -z "$dir0" ] || [ ! -f "$dir1/results.json" ] || [ ! -f "$dir0/results.json" ]; then
  hx_skip "need results.json from service/sv20_traffic#svc_load_cp1 and #svc_load_cp0"
fi
python3 tests/lib/loadgen.py compare --left "$dir1/results.json" --right "$dir0/results.json" --out "$HX_OUT/compare.txt" > "$HX_OUT/compare.log" 2>&1
rc=$?
if [ -f "$HX_OUT/compare.txt" ]; then sed "s/^/   /" "$HX_OUT/compare.txt"; else sed "s/^/   /" "$HX_OUT/compare.log"; fi
case "$rc" in
  0)  hx_ok "cp1 vs cp0 text parity PASS" ;;
  77) hx_warn "compare INCONCLUSIVE (no shared request)" ;;
  *)  hx_fail "cp1 vs cp0 differ -> $HX_OUT/compare.txt" ;;
esac
hx_end
