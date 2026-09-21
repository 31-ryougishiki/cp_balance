#!/usr/bin/env bash
# desc:     torch_npu 解析 + 逐窗口聚合（对已采集的配置；不起服务）
# needs:    none
# tags:     slow, perf
# est:      10min
set -uo pipefail
source "$(dirname "$0")/../lib/common.sh"

dirs=""; pruned=""
for cfg in $(hx_group prof); do
  profdir=$(hx_profdir "$cfg")
  [ -n "$profdir" ] && [ -f "$profdir/windows.json" ] || continue
  if ls -d "$profdir"/*_ascend_pt >/dev/null 2>&1; then
    dirs="$dirs $profdir"
  else
    pruned="$pruned $cfg"
  fi
done
[ -n "$pruned" ] && hx_note "no raw trace left (pruned?):$pruned"
[ -n "$dirs" ] || hx_skip "no config with raw trace to analyse (pruned:$pruned)"

python3 perf/profile_analyse.py $dirs > "$HX_OUT/analyse.log" 2>&1
rc=$?
grep -E "^\[analyse\]" "$HX_OUT/analyse.log" | tail -n 30 | sed 's/^/   /'
for profdir in $dirs; do
  if [ ! -f "$profdir/summary.json" ]; then
    hx_fail "$profdir: summary.json missing"
    continue
  fi
  n=$($HX_PY usable "$profdir")
  if [ "$n" -ge 1 ]; then
    hx_ok "$profdir/summary.json ($n usable windows)"
  else
    hx_fail "$profdir: 0 usable windows (analyse produced no usable rank output; see $HX_OUT/analyse.log)"
  fi
done
[ "$rc" -eq 0 ] || hx_fail "profile_analyse rc=$rc -> $HX_OUT/analyse.log"
if grep -q "WARNING window" "$HX_OUT/analyse.log"; then
  hx_warn "some windows hold more than one step; perf/p21_window_single_step judges them"
fi
hx_end
