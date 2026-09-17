#!/usr/bin/env bash
# desc:     请求真的走 ZIGZAG：长 prompt 进 zigzag、短 prompt 走连续切片
# needs:    service
# tags:     npu, slow, service
# variants: svc_cp1
# est:      15min
set -uo pipefail
source "$(dirname "$0")/../lib/common.sh"

arg=${1:-${HX_VARIANT:-svc_cp1}}
cfg=$(hx_resolve "$arg") || { hx_fail "unknown variant: $arg"; hx_end; exit 1; }
log=$HX_OUT/$cfg.log
port=$(hx_service_up "$cfg" "$log") || { hx_fail "service $cfg not ready"; hx_end; exit 1; }

min_tokens=$(hx_cfg_field "$cfg" min_tokens)
python3 accuracy/check_branch.py --url "http://127.0.0.1:$port" --log "$log" \
  --min-tokens "$min_tokens" > "$HX_OUT/branch.txt" 2>&1
rc=$?
sed 's/^/   /' "$HX_OUT/branch.txt"
if [ "$rc" -eq 0 ] && grep -q "RESULT: PASS" "$HX_OUT/branch.txt"; then
  hx_ok "long prompt -> zigzag, short prompt -> continuous (min_tokens=$min_tokens)"
else
  hx_fail "check_branch FAIL rc=$rc -> $HX_OUT/branch.txt"
fi

if hx_service_down "$port"; then
  hx_ok "service stopped"
else
  hx_fail "port $port still answering after stop"
fi
hx_end
