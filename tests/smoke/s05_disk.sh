#!/usr/bin/env bash
# desc:     磁盘余量够放 profiling 的原始 trace
# needs:    none
# tags:     fast, offline
# est:      5s
set -uo pipefail
source "$(dirname "$0")/../lib/common.sh"

min=${HX_MIN_FREE_GB:-20}
avail_kb=$(df -Pk "$HARNESS_ROOT" | awk 'NR==2 {print $4}')
if [ -z "$avail_kb" ]; then
  hx_skip "df gave no answer"
else
  avail_gb=$((avail_kb / 1024 / 1024))
  if [ "$avail_gb" -ge "$min" ]; then
    hx_ok "free ${avail_gb}GB >= ${min}GB on $HARNESS_ROOT"
  else
    hx_fail "free ${avail_gb}GB < ${min}GB on $HARNESS_ROOT (profiling writes GBs per rank)"
  fi
fi
hx_end
