#!/usr/bin/env bash
# desc:     服务能被拉起：起停一次 svc_cp1，/v1/models 列出配置里的模型名
# needs:    service
# tags:     npu, slow, service
# variants: svc_cp1
# est:      12min
set -uo pipefail
source "$(dirname "$0")/../lib/common.sh"

arg=${1:-${HX_VARIANT:-svc_cp1}}
cfg=$(hx_resolve "$arg") || { hx_fail "unknown variant: $arg"; hx_end; exit 1; }
log=$HX_OUT/$cfg.log
port=$(hx_service_up "$cfg" "$log") || { hx_fail "service $cfg not ready"; hx_end; exit 1; }
hx_ok "service $cfg ready on port $port"

if curl -sf "http://127.0.0.1:$port/v1/models" > "$HX_OUT/models.json"; then
  hx_ok "/v1/models reachable"
  for want in $(hx_cfg_field "$cfg" served_model_name); do
    if grep -q "$want" "$HX_OUT/models.json"; then
      hx_ok "served name $want listed"
    else
      hx_fail "served name $want not in /v1/models"
    fi
  done
else
  hx_fail "/v1/models request failed"
fi

fp=$(grep -m1 "^\[cp_balance\] CONFIG=" "$log" || true)
if [ -n "$fp" ]; then
  echo "$fp" > "$HX_OUT/fingerprint.txt"
  hx_ok "fingerprint: $(echo "$fp" | cut -c1-140)"
else
  hx_fail "no [cp_balance] fingerprint in $log"
fi

if hx_service_down "$port"; then
  hx_ok "service stopped"
else
  hx_fail "port $port still answering after stop"
fi
hx_end
