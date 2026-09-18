#!/usr/bin/env bash
# desc:     机器身份：IP/网卡/NPU/芯片 + 生效的 CP_BALANCE_* 覆盖（容器内没有 hostname/ip 也能跑）
# needs:    none
# tags:     fast, offline
# est:      5s
set -uo pipefail
source "$(dirname "$0")/../lib/common.sh"

NETIF="python3 $HARNESS_ROOT/tests/lib/netif.py"

host=$(cat /proc/sys/kernel/hostname 2>/dev/null || python3 -c 'import socket;print(socket.gethostname())' 2>/dev/null || echo unknown)
echo "host=$host family=$CP_BALANCE_FAMILY out=$HX_OUT"
for v in CP_BALANCE_LOCAL_IP CP_BALANCE_NIC_NAME CP_BALANCE_DEVICES CP_BALANCE_REPO CP_BALANCE_BASE_REPO; do
  echo "  $v=${!v:-unset}"
done

nics=$($NETIF 2>/dev/null || true)
if [ -n "$nics" ]; then
  printf '%s\n' "$nics" | awk -F'\t' '{printf "  nic %s %s %s\n", $1, $2, $3}'
else
  hx_warn "没有识别到 IPv4 网卡（容器里可显式 export CP_BALANCE_LOCAL_IP / CP_BALANCE_NIC_NAME）"
fi

if [ -n "${CP_BALANCE_LOCAL_IP:-}" ]; then
  if [ -z "$nics" ]; then
    hx_warn "本机枚举不到网卡，跳过 CP_BALANCE_LOCAL_IP 归属校验"
  elif printf '%s\n' "$nics" | cut -f2 | grep -qx "$CP_BALANCE_LOCAL_IP"; then
    hx_ok "CP_BALANCE_LOCAL_IP=$CP_BALANCE_LOCAL_IP 在本机网卡上"
  else
    hx_fail "CP_BALANCE_LOCAL_IP=$CP_BALANCE_LOCAL_IP 不在本机任何网卡上（候选见上）"
  fi
  if [ -n "${CP_BALANCE_NIC_NAME:-}" ]; then
    ifaces=$($NETIF --ifaces 2>/dev/null || true)
    if [ -z "$ifaces" ]; then
      hx_warn "本机枚举不到网卡名，跳过 CP_BALANCE_NIC_NAME 校验"
    elif printf '%s\n' "$ifaces" | grep -qx "$CP_BALANCE_NIC_NAME"; then
      hx_ok "CP_BALANCE_NIC_NAME=$CP_BALANCE_NIC_NAME 存在"
    else
      hx_fail "CP_BALANCE_NIC_NAME=$CP_BALANCE_NIC_NAME 不存在（候选：$(printf '%s' "$ifaces" | tr '\n' ' ')）"
    fi
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
