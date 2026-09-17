#!/usr/bin/env bash
# desc:     可选 A/B：KV 写过滤 slot<0 + TP/EP 分组打点（临时补丁，自动还原）
# needs:    service
# tags:     npu, slow, accuracy, variant
# est:      30min
set -uo pipefail
source "$(dirname "$0")/../lib/common.sh"

driver=accuracy/round2_verify.py
[ -f "$driver" ] || hx_skip "$driver not found"
c_matrix=$(hx_role c_matrix) || hx_skip "family=$CP_BALANCE_FAMILY has no c_matrix"
b_matrix=$(hx_role b_matrix) || hx_skip "family=$CP_BALANCE_FAMILY has no b_matrix"
opt=$(hx_resolve svc_cp1) || hx_skip "family=$CP_BALANCE_FAMILY has no svc_cp1"

python3 "$driver" --steps 3 --c-matrix "$c_matrix" --b-matrix "$b_matrix" \
  --optional-config "$opt" > "$HX_OUT/step3.log" 2>&1
rc=$?
grep -E "^\[round2\] (PASS|FAIL|SKIP|WARNING)|the TP and EP halves" "$HX_OUT/step3.log" | sed 's/^/   /'
if grep -q "RESULT: FAIL" "$HX_OUT/step3.log"; then
  hx_fail "step 3 FAIL rc=$rc -> $HX_OUT/step3.log"
elif grep -q "RESULT: PASS" "$HX_OUT/step3.log"; then
  hx_ok "slot<0 filter A/B PASS (patch reverted)"
else
  hx_fail "step 3 produced no verdict rc=$rc -> $HX_OUT/step3.log"
fi
if grep -q "\[CP_BALANCE\]\[group\]" "$HX_OUT/step3.log"; then
  hx_ok "TP/EP group line present: $(grep -m1 '\[CP_BALANCE\]\[group\]' "$HX_OUT/step3.log" | cut -c1-100)"
else
  hx_warn "no [CP_BALANCE][group] line (batch may not have been zigzag-eligible)"
fi
hx_end
