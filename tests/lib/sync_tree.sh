#!/usr/bin/env bash
# 代码树自动对齐（供 verify.sh / 其它入口 source）
#
#   hx_tree_field <角色> <字段>            # harness.json trees（trees.py get）
#   hx_tree_patches <角色>                 # harness.json trees.<角色>.patches（每行一个绝对路径）
#   hx_apply_tree_patches <角色> <路径>    # 对齐后按清单打补丁（幂等：已经打上就跳过）
#   hx_ensure_tree <角色> [<路径覆盖>]     # 缺了就 clone，版本不对就切到目标（分支 tip 或固定 commit）
#   hx_sync_tree <路径> <分支> [<远端>]    # 只在指定路径上做分支对齐（向后兼容的小工具）
#
# 目标版本全部来自 harness.json；CP_BALANCE_AUTO_CHECKOUT=0 时只报告不切换。
# 返回 0 = 已就绪（或已修好/可跳过）；1 = 失败。

hx_tree_field() {
  python3 "$HX_LIB/trees.py" get "$1" "$2" 2>/dev/null
}

hx_tree_roles() {
  python3 "$HX_LIB/trees.py" list 2>/dev/null
}

hx_tree_patches() {
  python3 "$HX_LIB/trees.py" patches "$1" 2>/dev/null
}

hx_apply_tree_patches() {   # <角色> <路径>：补丁是"harness 让参照树也能跑起来"的显式改动
  local role=$1 repo=$2 patch
  while IFS= read -r patch; do
    [ -n "$patch" ] || continue
    case "$patch" in /*) ;; *) patch="$HARNESS_ROOT/$patch" ;; esac   # git -C 的补丁路径不是相对 cwd
    [ -f "$patch" ] || { echo "[sync] $role 的补丁不存在：$patch" >&2; return 1; }
    if git -C "$repo" apply --check --reverse "$patch" >/dev/null 2>&1; then
      echo "[sync] $role 已含补丁 $(basename "$patch")"
      continue
    fi
    if git -C "$repo" apply --check "$patch" >/dev/null 2>&1; then
      git -C "$repo" apply "$patch" && echo "[sync] $role 打上补丁 $(basename "$patch")"
      continue
    fi
    echo "[sync] $role 打不上补丁 $(basename "$patch")（源码变了？重新生成 patches/ 下的补丁）" >&2
    return 1
  done < <(hx_tree_patches "$role")
  return 0
}

hx_git_align() {   # <路径> <远端> <ref> <kind>
  local repo=$1 remote=$2 ref=$3 kind=$4
  if [ -n "$remote" ]; then
    local cur_url
    cur_url=$(git -C "$repo" remote get-url origin 2>/dev/null || true)
    if [ -n "$cur_url" ] && [ "$cur_url" != "$remote" ]; then
      echo "[sync] $repo origin 从 $cur_url 改为 $remote"
      git -C "$repo" remote set-url origin "$remote"
    fi
  fi
  if [ "$kind" = "commit" ]; then
    git -C "$repo" fetch --quiet origin 2>/dev/null || echo "[sync] $repo fetch 失败（离线？）" >&2
    local head
    head=$(git -C "$repo" rev-parse HEAD 2>/dev/null || true)
    if [ "$head" = "$ref" ]; then
      echo "[sync] $(basename "$repo") 已在固定 commit $(echo "$ref" | cut -c1-10)"
      return 0
    fi
    if ! hx_tree_clean "$repo"; then
      return 1
    fi
    if [ "${CP_BALANCE_AUTO_CHECKOUT:-1}" != "1" ]; then
      echo "[sync] $(basename "$repo") 需要切到 commit $(echo "$ref" | cut -c1-10)（未自动切换）"
      return 1
    fi
    echo "[sync] $(basename "$repo"): $(git -C "$repo" rev-parse --short HEAD 2>/dev/null) -> commit $(echo "$ref" | cut -c1-10)"
    git -C "$repo" checkout --quiet --detach "$ref" || { echo "[sync] checkout $ref 失败" >&2; return 1; }
    echo "[sync] 对齐完成：$(git -C "$repo" log -1 --oneline)"
    return 0
  fi
  # 先 fetch 再比：已经对齐的树不再因为"打过补丁"被判脏，落后就照旧切
  git -C "$repo" fetch --quiet origin "$ref" 2>/dev/null     || echo "[sync] fetch origin/$ref 失败（离线？）——只跟本地 ref 比对" >&2
  if hx_tree_at_branch "$repo" "$ref" origin; then
    return 0
  fi
  if ! hx_tree_clean "$repo"; then
    return 1
  fi
  hx_sync_tree "$repo" "$ref" origin
}

hx_tree_clean() {   # <路径>：有未提交改动就报出来（只在真的要切换时才拦）
  local repo=$1 dirty
  dirty=$(git -C "$repo" status --porcelain 2>/dev/null | head -5)
  if [ -n "$dirty" ]; then
    echo "[sync] $repo 有未提交改动，不自动切换（先 add/commit/stash）：" >&2
    printf '%s
' "$dirty" | sed 's/^/        /' >&2
    return 1
  fi
  return 0
}

hx_tree_at_branch() {   # <路径> <分支> <远端>：HEAD 与本地分支都已在目标上
  local repo=$1 branch=$2 remote=${3:-origin} head cur target
  head=$(git -C "$repo" rev-parse HEAD 2>/dev/null || true)
  cur=$(git -C "$repo" rev-parse --abbrev-ref HEAD 2>/dev/null || true)
  target=$(git -C "$repo" rev-parse --verify --quiet "$remote/$branch" 2>/dev/null || true)
  [ -n "$target" ] || target=$(git -C "$repo" rev-parse --verify --quiet "$branch" 2>/dev/null || true)
  if [ -n "$head" ] && [ "$head" = "$target" ] && [ "$cur" = "$branch" ]; then
    echo "[sync] $(basename "$repo") 已在 $branch@$(git -C "$repo" rev-parse --short HEAD)"
    return 0
  fi
  return 1
}

hx_sync_tree() {   # <路径> <分支> [<远端>]
  local repo=$1 branch=$2 remote=${3:-origin}
  if [ ! -d "$repo/.git" ]; then
    echo "[sync] $repo 不是 git 仓库，无法对齐版本" >&2
    return 1
  fi
  git -C "$repo" fetch --quiet "$remote" "$branch" 2>/dev/null || echo "[sync] fetch $remote/$branch 失败（离线？）——只跟本地 ref 比对" >&2
  local target
  target=$(git -C "$repo" rev-parse --verify --quiet "$remote/$branch" 2>/dev/null || true)
  [ -n "$target" ] || target=$(git -C "$repo" rev-parse --verify --quiet "$branch" 2>/dev/null || true)
  if [ -z "$target" ]; then
    echo "[sync] 找不到目标版本 $remote/$branch" >&2
    return 1
  fi
  local head cur_branch
  head=$(git -C "$repo" rev-parse HEAD)
  cur_branch=$(git -C "$repo" rev-parse --abbrev-ref HEAD)
  if [ "$head" = "$target" ] && [ "$cur_branch" = "$branch" ]; then
    echo "[sync] $(basename "$repo") 已在 $branch@$(git -C "$repo" rev-parse --short HEAD)"
    return 0
  fi
  local target_short
  target_short=$(git -C "$repo" rev-parse --short "$target")
  if [ "${CP_BALANCE_AUTO_CHECKOUT:-1}" != "1" ]; then
    echo "[sync] $(basename "$repo") 版本不符（$cur_branch@$(git -C "$repo" rev-parse --short HEAD) != $branch@$target_short），未自动切换"
    return 1
  fi
  # 直接调用本工具时也要拦脏树（hx_git_align 会先查一次）
  if ! hx_tree_clean "$repo"; then
    return 1
  fi
  local dropped
  dropped=$(git -C "$repo" log --oneline "$target..HEAD" 2>/dev/null | head -10)
  if [ -n "$dropped" ]; then
    echo "[sync] 警告：$repo 上有不在 $remote/$branch 的提交，reset 会丢弃它们（reflog 里还能找回）："
    printf '%s\n' "$dropped" | sed 's/^/        /'
  fi
  echo "[sync] $(basename "$repo"): $cur_branch@$(git -C "$repo" rev-parse --short HEAD) -> $branch@$target_short，自动切换"
  git -C "$repo" checkout --quiet "$branch" 2>/dev/null || git -C "$repo" checkout --quiet -B "$branch" "$target" || {
    echo "[sync] checkout $branch 失败" >&2
    return 1
  }
  git -C "$repo" reset --hard --quiet "$target" || { echo "[sync] reset --hard $target 失败" >&2; return 1; }
  echo "[sync] 对齐完成：$(git -C "$repo" log -1 --oneline)"
  return 0
}

hx_ensure_tree() {   # <角色> [<路径覆盖>]
  local role=$1 override=$2
  local path remote ref kind required
  path=$(hx_tree_field "$role" path) || { echo "[sync] harness.json trees 里没有角色 $role" >&2; return 1; }
  [ -n "$override" ] && path=$override
  remote=$(hx_tree_field "$role" remote)
  ref=$(hx_tree_field "$role" ref)
  kind=$(hx_tree_field "$role" kind)
  required=$(hx_tree_field "$role" required)

  if [ ! -d "$path" ]; then
    if [ "$required" != "true" ]; then
      echo "[note] $role 树不存在（$path，required=false），跳过；期望版本 $ref"
      return 0
    fi
    if [ -z "$remote" ]; then
      echo "[sync] $role 树不存在（$path）且 harness.json 没给 remote，无法自动拉取" >&2
      return 1
    fi
    if [ "${CP_BALANCE_AUTO_CHECKOUT:-1}" != "1" ]; then
      echo "[sync] $role 树不存在（$path），需要：git clone $remote $path（未自动执行）" >&2
      return 1
    fi
    echo "[sync] $role 树不存在，自动拉取：git clone $remote $path"
    if ! git clone --quiet "$remote" "$path" 2>/dev/null; then
      echo "[sync] clone $remote 失败（网络/权限？）" >&2
      return 1
    fi
    echo "[sync] clone 完成：$(git -C "$path" log -1 --oneline 2>/dev/null)"
  fi
  hx_git_align "$path" "$remote" "$ref" "$kind" || return 1
  hx_apply_tree_patches "$role" "$path"
}
