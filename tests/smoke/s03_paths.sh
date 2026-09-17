#!/usr/bin/env bash
# desc:     配置引用到的 repo / model / prelude / base 树路径都存在
# needs:    none
# tags:     fast, offline
# est:      10s
set -uo pipefail
source "$(dirname "$0")/../lib/common.sh"

cfgs=$(hx_family_configs)
[ -n "$cfgs" ] || hx_skip "family $CP_BALANCE_FAMILY has no configs"

for cfg in $cfgs; do
  hx_need_dir "$cfg repo" "$(hx_path "$cfg" repo)"
  model=$(hx_path "$cfg" model)
  if [ -z "$model" ]; then
    hx_fail "$cfg has no model"
  elif [ ! -f "$model/config.json" ]; then
    hx_fail "$cfg model incomplete (no config.json): $model"
  elif ls "$model"/*.safetensors >/dev/null 2>&1 || ls "$model"/*index.json >/dev/null 2>&1; then
    hx_ok "$cfg model $model"
  else
    hx_fail "$cfg model has config.json but no weight files: $model"
  fi
  prelude=$(hx_path "$cfg" prelude)
  [ -z "$prelude" ] || hx_need_file "$cfg prelude" "$prelude"
done

base=$(hx_base_repo)
if [ -z "$base" ]; then
  hx_note "no base_repo in $(hx_role b_matrix) (set CP_BALANCE_BASE_REPO to check)"
elif [ -f "$base/vllm_ascend/attention/sfa_v1.py" ]; then
  hx_ok "base tree $base"
else
  hx_fail "base tree unusable (no vllm_ascend/attention/sfa_v1.py): $base"
fi
cur=$(hx_cur_repo)
if [ -n "$cur" ]; then
  hx_need_dir "cur tree" "$cur"
fi
hx_end
