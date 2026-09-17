#!/usr/bin/env bash
# desc:     复用 a10 采到的 json 重跑对比：C 验收 + B 等价性 + 噪声地板（不起服务）
# needs:    none
# tags:     fast, accuracy
# est:      20s
set -uo pipefail
source "$(dirname "$0")/../lib/common.sh"

checked=0
for role in c_matrix b_matrix; do
  matrix=$(hx_role "$role") || continue
  [ -n "$matrix" ] && [ -f "$matrix" ] || continue
  dir="$HARNESS_OUT/accuracy/a10_matrix_gate.$role/matrix"
  if [ ! -d "$dir" ]; then
    dir=$(ls -dt "$HARNESS_OUT"/accuracy/a10_matrix_gate."$role"*/matrix 2>/dev/null | head -1)
  fi
  if [ -z "$dir" ] || [ ! -d "$dir" ]; then
    hx_note "$role: no collected dir (run accuracy/a10_matrix_gate#$role first)"
    continue
  fi
  while IFS=$'\t' read -r label left right require; do
    if [ ! -f "$dir/$left.json" ] || [ ! -f "$dir/$right.json" ]; then
      hx_warn "$label: json missing in $dir"
      continue
    fi
    extra=""
    [ "$require" = "True" ] && extra=--require-text
    python3 accuracy/compare_first_token.py compare $extra \
      "$dir/$left.json" "$dir/$right.json" > "$HX_OUT/$label.txt" 2>&1
    rc=$?
    checked=$((checked + 1))
    if [ "$rc" -eq 0 ] && grep -q "RESULT: PASS" "$HX_OUT/$label.txt"; then
      hx_ok "$label: $(grep -m1 'first-token match' "$HX_OUT/$label.txt")"
    else
      hx_fail "$label: compare FAIL -> $HX_OUT/$label.txt"
    fi
  done < <($HX_PY compares "$matrix")
done
[ "$checked" -gt 0 ] || hx_skip "nothing comparable (jsons missing)"
hx_end
