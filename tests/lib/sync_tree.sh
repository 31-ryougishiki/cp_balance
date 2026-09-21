#!/usr/bin/env bash
# 代码树版本对齐（供 verify_a5.sh / 其它入口 source）
#
#   hx_tree_branch cur|base                  # 读 tests/lib/targets.tsv
#   hx_sync_tree <路径> <分支> [<远端>]      # 不在目标版本上就自动切过去
#
# 行为：脏树（有未提交改动）不自动切换，直接失败并列出改动；干净树会
# checkout <分支> + reset --hard <远端>/<分支>，切换前把将被丢弃的本地提交打出来。
# CP_BALANCE_AUTO_CHECKOUT=0 时只检查不切换（打印该执行什么命令）。
# 返回 0 = 已在目标版本或已修好；1 = 失败。

hx_tree_branch() {
  local role=$1
  [ -f "$HX_LIB/targets.tsv" ] || return 1
  awk -v role="$role" '!/^#/ && NF >= 2 && $1 == role {print $2; exit}' "$HX_LIB/targets.tsv"
}

hx_sync_tree() {
  local repo=$1 branch=$2 remote=${3:-origin}
  if [ ! -d "$repo/.git" ]; then
    echo "[sync] $repo 不是 git 仓库，无法对齐版本" >&2
    return 1
  fi

  local dirty
  dirty=$(git -C "$repo" status --porcelain 2>/dev/null | head -5)
  if [ -n "$dirty" ]; then
    echo "[sync] $repo 有未提交改动，不自动切换（先 add/commit/stash）：" >&2
    printf '%s\n' "$dirty" | sed 's/^/        /' >&2
    return 1
  fi

  if ! git -C "$repo" fetch --quiet "$remote" "$branch" 2>/dev/null; then
    echo "[sync] fetch $remote/$branch 失败（离线？）——只跟本地 ref 比对" >&2
  fi

  local target
  target=$(git -C "$repo" rev-parse --verify --quiet "$remote/$branch" 2>/dev/null || true)
  [ -n "$target" ] || target=$(git -C "$repo" rev-parse --verify --quiet "$branch" 2>/dev/null || true)
  if [ -z "$target" ]; then
    echo "[sync] 找不到目标版本 $remote/$branch（先 git fetch $remote）" >&2
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
    echo "[sync] $(basename "$repo") 版本不符（$cur_branch@$(git -C "$repo" rev-parse --short HEAD) != $branch@$target_short）"
    echo "[sync] CP_BALANCE_AUTO_CHECKOUT=0，未自动切换。手动执行："
    echo "        git -C $repo checkout $branch && git -C $repo reset --hard $remote/$branch"
    return 1
  fi

  local dropped
  dropped=$(git -C "$repo" log --oneline "$target..HEAD" 2>/dev/null | head -10)
  if [ -n "$dropped" ]; then
    echo "[sync] 警告：$repo 上有不在 $remote/$branch 的提交，reset 会丢弃它们（reflog 里还能找回）："
    printf '%s\n' "$dropped" | sed 's/^/        /'
  fi

  echo "[sync] $(basename "$repo"): $cur_branch@$(git -C "$repo" rev-parse --short HEAD) -> $branch@$target_short，自动切换"
  if ! git -C "$repo" checkout --quiet "$branch" 2>/dev/null; then
    git -C "$repo" checkout --quiet -B "$branch" "$target" || {
      echo "[sync] checkout $branch 失败" >&2
      return 1
    }
  fi
  if ! git -C "$repo" reset --hard --quiet "$target"; then
    echo "[sync] reset --hard $target 失败" >&2
    return 1
  fi
  echo "[sync] 对齐完成：$(git -C "$repo" log -1 --oneline)"
  return 0
}
