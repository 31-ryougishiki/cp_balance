#!/usr/bin/env bash
# desc:     机器身份：IP/网卡/NPU/芯片 + 生效的 CP_BALANCE_* 覆盖
# needs:    none
# tags:     fast, offline
# est:      5s
set -uo pipefail
source "$(dirname "$0")/../lib/common.sh"

echo "host=$(hostname) family=$CP_BALANCE_FAMILY out=$HX_OUT"
for v in CP_BALANCE_LOCAL_IP CP_BALANCE_NIC_NAME CP_BALANCE_DEVICES CP_BALANCE_REPO CP_BALANCE_BASE_REPO; do
  echo "  $v=${!v:-unset}"
done
ip -o -4 addr show 2>/dev/null | awk '{print "  nic", $2, $4}'

if [ -n "${CP_BALANCE_LOCAL_IP:-}" ]; then
  nic=$(ip -o -4 addr show 2>/dev/null | awk -v ip="$CP_BALANCE_LOCAL_IP" '$4 ~ "^"ip"/" {print $2; exit}')
  if [ -n "$nic" ]; then
    hx_ok "CP_BALANCE_LOCAL_IP=$CP_BALANCE_LOCAL_IP is on $nic"
  else
    hx_fail "CP_BALANCE_LOCAL_IP=$CP_BALANCE_LOCAL_IP is on no NIC of this host"
  fi
  if [ -n "${CP_BALANCE_NIC_NAME:-}" ] && ! ip link show "$CP_BALANCE_NIC_NAME" >/dev/null 2>&1; then
    hx_fail "CP_BALANCE_NIC_NAME=$CP_BALANCE_NIC_NAME does not exist"
  fi
fi

soc=$(python3 -c "import torch_npu; print(torch_npu.npu.get_soc_version())" 2>/dev/null)
if [ -z "$soc" ]; then
  hx_warn "torch_npu not importable in this shell (config prelude sets it up for the service)"
else
  case "$soc" in
    260)     dev=A5 ;;
    25[0-5]) dev=A3 ;;
    *)       dev=unknown ;;
  esac
  hx_ok "soc_version=$soc ($dev)"
  case "$CP_BALANCE_FAMILY:$dev" in
    a5:A5|a3:A3) hx_ok "family=$CP_BALANCE_FAMILY matches chip" ;;
    *)           hx_fail "family=$CP_BALANCE_FAMILY does not match chip=$dev" ;;
  esac
  npu=$(python3 -c "import torch_npu; print(torch_npu.npu.device_count())" 2>/dev/null)
  if [ -n "$npu" ]; then
    hx_ok "npu device_count=$npu"
  else
    hx_warn "device_count unavailable"
  fi
fi
hx_end
