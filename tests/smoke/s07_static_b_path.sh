#!/usr/bin/env bash
# desc:     静态门控：cp_balance 只作用于 zigzag 路径（B == 原版）
# needs:    none
# tags:     fast, offline, static
# est:      10s
set -uo pipefail
source "$(dirname "$0")/../lib/common.sh"

cur=$(hx_cur_repo); base=$(hx_base_repo)
[ -n "$cur" ] && [ -n "$base" ] || hx_skip "need cur+base repo (set CP_BALANCE_REPO / CP_BALANCE_BASE_REPO)"
[ -d "$cur" ] && [ -d "$base" ] || hx_skip "trees not on this machine: $cur / $base"

python3 accuracy/check_b_path.py --repo "$cur" --base-repo "$base" > "$HX_OUT/b_path.txt" 2>&1
rc=$?
grep -E "^\[check\]" "$HX_OUT/b_path.txt" | sed 's/^/   /'
if grep -q "RESULT: PASS" "$HX_OUT/b_path.txt"; then
  hx_ok "check_b_path PASS"
else
  hx_fail "check_b_path FAIL rc=$rc -> $HX_OUT/b_path.txt"
fi
hx_end
