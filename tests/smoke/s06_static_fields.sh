#!/usr/bin/env bash
# desc:     静态门控：ZigzagPlan / DSACPContext 字段对得上
# needs:    none
# tags:     fast, offline, static
# est:      10s
set -uo pipefail
source "$(dirname "$0")/../lib/common.sh"

repo=$(hx_cur_repo)
[ -n "$repo" ] || hx_skip "no repo (set CP_BALANCE_REPO, or check static_check in $(hx_role b_matrix))"
[ -d "$repo" ] || hx_skip "repo not on this machine: $repo"

python3 perf/check_cp_balance_fields.py --repo "$repo" > "$HX_OUT/fields.txt" 2>&1
rc=$?
grep -E "^\[check\]" "$HX_OUT/fields.txt" | sed 's/^/   /'
if grep -q "RESULT: PASS" "$HX_OUT/fields.txt"; then
  hx_ok "check_cp_balance_fields PASS ($repo)"
else
  hx_fail "check_cp_balance_fields FAIL rc=$rc ($repo) -> $HX_OUT/fields.txt"
fi
want=$($HX_PY harness expect.zigzag_plan_fields 2>/dev/null | tr -d '\r')
: "${want:=13}"
if grep -q "ZigzagPlan fields=$want" "$HX_OUT/fields.txt"; then
  hx_ok "ZigzagPlan fields=$want (harness.json expect.zigzag_plan_fields)"
else
  hx_fail "unexpected field count: $(grep -o 'ZigzagPlan fields=[0-9]*' "$HX_OUT/fields.txt" | head -1) (want $want)"
fi
hx_end
