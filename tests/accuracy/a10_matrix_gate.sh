#!/usr/bin/env bash
# desc:     矩阵门禁：起停服务、采 40 条 prompt、给 RESULT
# needs:    service
# tags:     npu, slow, accuracy
# variants: c_matrix b_matrix nomtp_matrix
# est:      60min
set -uo pipefail
source "$(dirname "$0")/../lib/common.sh"

role=${1:-${HX_VARIANT:-c_matrix}}
matrix=$(hx_role "$role") || { hx_fail "unknown role: $role"; hx_end; exit 1; }
[ -n "$matrix" ] || hx_skip "family=$CP_BALANCE_FAMILY has no $role"
[ -f "$matrix" ] || hx_skip "$matrix not found"

out=$HX_OUT/matrix
python3 accuracy/run_matrix.py "$matrix" --out "$out" > "$HX_OUT/driver.log" 2>&1
rc=$?
grep -E "^\[matrix\] (RESULT|count |compare |WARNING|static)" "$HX_OUT/driver.log" | sed 's/^/   /'
if [ -f "$out/summary.txt" ] && grep -q "RESULT: PASS" "$out/summary.txt"; then
  hx_ok "$matrix -> RESULT: PASS"
else
  hx_fail "$matrix -> RESULT: FAIL rc=$rc (see $out/summary.txt)"
fi
zig=$(grep -c "branch=ZIGZAG" "$HX_OUT/driver.log" || true)
hx_note "branch=ZIGZAG lines in driver log: $zig (0 means the request never took zigzag)"
hx_end
