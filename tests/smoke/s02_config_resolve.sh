#!/usr/bin/env bash
# desc:     family 里每个 config 都能解析出预期指纹（不起服务）
# needs:    none
# tags:     fast, offline
# est:      30s
set -uo pipefail
source "$(dirname "$0")/../lib/common.sh"

has_field() {  # <指纹行> <KEY=value>
  case "$1" in
    *" $2 "*) return 0 ;;
    *)        return 1 ;;
  esac
}

cfgs=$(hx_family_configs)
[ -n "$cfgs" ] || hx_skip "family $CP_BALANCE_FAMILY has no configs"

: > "$HX_OUT/fingerprints.txt"
for cfg in $cfgs; do
  line=$(hx_fingerprint "$cfg")
  if [ -z "$line" ]; then
    hx_fail "$cfg: no fingerprint ([cp_balance] CONFIG=... missing)"
    continue
  fi
  echo "$line" >> "$HX_OUT/fingerprints.txt"
  case "$line" in
    "[cp_balance] CONFIG=$cfg "*) hx_ok "$cfg resolves" ;;
    *)                            hx_fail "$cfg: fingerprint does not start with CONFIG=$cfg" ;;
  esac
  for want in "IP=$(hx_cfg_field "$cfg" local_ip)" \
              "TP=$(hx_cfg_field "$cfg" tp_size)" \
              "NIC=$(hx_cfg_field "$cfg" nic_name)" \
              "CP_BALANCE=$(hx_cfg_field "$cfg" cp_balance)" \
              "MODEL=$(hx_cfg_field "$cfg" model)"; do
    has_field "$line" "$want" || hx_fail "$cfg: fingerprint lacks $want"
  done
  case "$cfg" in
    prof_*)
      case "$line" in
        *" PROFILER=off"*) hx_fail "$cfg: profiler config but PROFILER=off" ;;
        *)                 hx_ok "$cfg profiler dir resolved" ;;
      esac ;;
  esac
done
hx_note "fingerprints -> $HX_OUT/fingerprints.txt"
hx_end
