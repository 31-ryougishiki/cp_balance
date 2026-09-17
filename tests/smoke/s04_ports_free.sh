#!/usr/bin/env bash
# desc:     family 用到的所有端口当前都空闲
# needs:    none
# tags:     fast, offline
# est:      10s
set -uo pipefail
source "$(dirname "$0")/../lib/common.sh"

used=0; checked=0
while read -r name port; do
  [ -n "$port" ] || continue
  checked=$((checked + 1))
  answer=$($HX_PY listen "$port"); rc=$?
  case "$answer" in
    0) hx_ok "port $port free ($name)" ;;
    1) hx_fail "port $port ($name) is in use"; used=$((used + 1)) ;;
    *) hx_fail "port $port ($name): probe failed (rc=$rc, answer='$answer')" ;;
  esac
done < <($HX_PY ports $(hx_family_configs))

[ "$checked" -gt 0 ] || hx_skip "no config port to check"
[ "$used" -eq 0 ] && hx_note "$checked ports checked, no conflicts"
hx_end
