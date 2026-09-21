#!/usr/bin/env bash
# desc:     每个采集窗口必须只含 1 个 prefill step（kernel_steps == 1）
# needs:    none
# tags:     fast, perf
# est:      10s
set -uo pipefail
source "$(dirname "$0")/../lib/common.sh"

total=0; bad=0; unknown=0
for cfg in $(hx_group prof); do
  [ -f "$cfg/summary.json" ] || continue
  while IFS=$'\t' read -r window steps _; do
    total=$((total + 1))
    if [ "$steps" = "-" ]; then
      unknown=$((unknown + 1))
      hx_warn "$cfg/$window: no kernel_steps (kernel_details.csv missing?)"
    elif [ "$steps" != "1" ]; then
      bad=$((bad + 1))
      hx_fail "$cfg/$window: kernel_steps=$steps (window not prefill-only, discard this length)"
    fi
  done < <($HX_PY windows "$(hx_profdir "$cfg")")
done
[ "$total" -gt 0 ] || hx_skip "no summary.json yet (run perf/p20_analyse first)"
[ "$unknown" -eq "$total" ] && hx_skip "$total windows but kernel_steps is unavailable for all of them"
[ "$bad" -eq 0 ] && hx_ok "$total windows, all single-step (unknown=$unknown)"
hx_end
