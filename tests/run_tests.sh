#!/usr/bin/env bash
# tests/run_tests.sh —— 统一入口：发现 tests/{smoke,accuracy,service,perf}/*.sh，按脚本头部元数据
# （# desc/needs/tags/variants/est，见 tests/README.md）筛选、逐个执行、汇总。
# 每个测试也能单独跑：bash tests/smoke/s02_config_resolve.sh
# 退出码：0 全过；1 有 FAIL（--strict 时 SKIP 也算失败）；2 用法错、没发现测试、选择器一个都没命中。
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
cd "$ROOT"

usage() {
  cat <<'USAGE'
tests/run_tests.sh [options]
  --list           只列出会跑的测试（含 needs/tags/est/desc）
  --dry-run        只打印将要执行的命令
  --only a,b       只跑这些（id、id#variant、basename、目录名如 accuracy 均可）
  --skip a,b       跳过这些（写法同 --only）
  --tag fast       只跑带该 tag 的（fast/offline/static/npu/slow/service/accuracy/perf/report/variant）
  --from <id>      从该测试开始（写法同 --only；一个都没命中按用法错误退出 2）
  --family a5|a3   机器族（等价 CP_BALANCE_FAMILY，默认 a5）
  --out DIR        产物目录（默认 tests/_out/<时间戳>）
  --keep-going     有 FAIL 也继续
  --live-log       测试与模型服务日志实时打屏（同时写入 $OUT）
  --strict         SKIP 也算失败
USAGE
}

FAMILY=${CP_BALANCE_FAMILY:-a5}
ONLY=""; SKIP=""; TAGS=""; FROM=""; OUT=""
LIST=0; DRY=0; KEEP=0; STRICT=0; LIVE=${HX_LIVE_LOG:-0}
while [ $# -gt 0 ]; do
  opt=$1
  case "$opt" in
    --only|--skip|--tag|--from|--family|--out)
      [ $# -ge 2 ] || { echo "run_tests: $opt needs a value" >&2; usage >&2; exit 2; }
      case "$opt" in
        --only)   ONLY=$2 ;;
        --skip)   SKIP=$2 ;;
        --tag)    TAGS=$2 ;;
        --from)   FROM=$2 ;;
        --family) FAMILY=$2 ;;
        --out)    OUT=$2 ;;
      esac
      shift 2 ;;
    --keep-going) KEEP=1; shift ;;
    --live-log)   LIVE=1; shift ;;
    --strict)     STRICT=1; shift ;;
    --list)       LIST=1; shift ;;
    --dry-run)    DRY=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "run_tests: unknown option $opt" >&2; usage >&2; exit 2 ;;
  esac
done

case "$FAMILY" in
  a5|a3) ;;
  *) echo "run_tests: --family must be a5 or a3 (got '$FAMILY')" >&2; exit 2 ;;
esac
export CP_BALANCE_FAMILY=$FAMILY
export HX_STREAM_SERVICE_LOG=$LIVE
: "${OUT:=$ROOT/tests/_out/$(date +%m%d_%H%M%S)}"
export HARNESS_OUT=$OUT
mkdir -p "$OUT"

meta() { sed -n "s/^#[[:space:]]*$2:[[:space:]]*//p" "$1" | head -1; }

sel_match() {  # <选择器列表> <id>：完整 id / id#variant / basename / 目录前缀（accuracy）都能命中
  local list=$1 id=$2 base=${2%%#*} item
  local -a parts
  IFS=',' read -r -a parts <<< "$list"
  for item in "${parts[@]}"; do
    [ -z "$item" ] && continue
    [ "$item" = "$id" ] && return 0
    [ "$item" = "$base" ] && return 0
    [ "$item" = "${base##*/}" ] && return 0
    [ "$item" = "${id##*/}" ] && return 0
    [ "$item" = "${base##*/}#${id##*#}" ] && return 0
    [ "${base#"$item"/}" != "$base" ] && return 0
  done
  return 1
}

tag_match() {  # 匹配前去掉空格：tags 写的是 "npu, slow, service"
  local want have item
  want=$(printf '%s' "$1" | tr -d ' ')
  have=$(printf '%s' "$2" | tr -d ' ')
  local -a parts
  IFS=',' read -r -a parts <<< "$want"
  for item in "${parts[@]}"; do
    [ -z "$item" ] && continue
    case ",$have," in *",$item,"*) return 0 ;; esac
  done
  return 1
}

sel_ok() {  # <index>：ONLY / SKIP / TAGS 三个筛选（list 与 run 共用）
  local idx=$1 id=${ids[$1]}
  [ -n "$ONLY" ] && ! sel_match "$ONLY" "$id" && return 1
  [ -n "$SKIP" ] && sel_match "$SKIP" "$id" && return 1
  [ -n "$TAGS" ] && ! tag_match "$TAGS" "${tags_list[$idx]}" && return 1
  return 0
}

ids=(); files=(); variants=(); needs_list=(); tags_list=(); descs=(); ests=(); cmds=()
n=0
for f in "$HERE"/smoke/*.sh "$HERE"/accuracy/*.sh "$HERE"/service/*.sh "$HERE"/perf/*.sh; do
  [ -f "$f" ] || continue
  rel=${f#"$HERE"/}; rel=${rel%.sh}
  needs=$(meta "$f" needs); tags=$(meta "$f" tags); desc=$(meta "$f" desc)
  est=$(meta "$f" est); vars=$(meta "$f" variants)
  if [ -n "$vars" ]; then
    for v in $vars; do
      ids[$n]="$rel#$v"; files[$n]="$f"; variants[$n]="$v"
      needs_list[$n]="$needs"; tags_list[$n]="$tags"; descs[$n]="$desc"; ests[$n]="$est"
      cmds[$n]="bash tests/$rel.sh $v"
      n=$((n + 1))
    done
  else
    ids[$n]="$rel"; files[$n]="$f"; variants[$n]=""
    needs_list[$n]="$needs"; tags_list[$n]="$tags"; descs[$n]="$desc"; ests[$n]="$est"
    cmds[$n]="bash tests/$rel.sh"
    n=$((n + 1))
  fi
done

[ "$n" -gt 0 ] || { echo "run_tests: no tests found under $HERE" >&2; exit 2; }
echo "[run_tests] family=$FAMILY tests=$n out=$OUT live_log=$LIVE"
for v in CP_BALANCE_LOCAL_IP CP_BALANCE_NIC_NAME CP_BALANCE_DEVICES CP_BALANCE_REPO CP_BALANCE_BASE_REPO; do
  [ -n "${!v:-}" ] && echo "[run_tests] env $v=${!v}"
done

FMT='%-46s %-10s %-30s %-6s %s\n'
if [ "$LIST" -eq 1 ]; then
  printf "$FMT" id needs tags est desc
  rows=0
  i=0
  while [ "$i" -lt "$n" ]; do
    if sel_ok "$i"; then
      printf "$FMT" "${ids[$i]}" "${needs_list[$i]}" "${tags_list[$i]}" "${ests[$i]}" "${descs[$i]}"
      rows=$((rows + 1))
    fi
    i=$((i + 1))
  done
  [ "$rows" -gt 0 ] || { echo "run_tests: no test matched (only=$ONLY skip=$SKIP tag=$TAGS)" >&2; exit 2; }
  exit 0
fi

status_file=$OUT/status.tsv
: > "$status_file"
pass=0; fail=0; skipped=0; started=0; executed=0; first_fail=""; first_cmd=""

i=0
while [ "$i" -lt "$n" ]; do
  id=${ids[$i]}; file=${files[$i]}; var=${variants[$i]}
  i=$((i + 1))
  sel_ok "$((i - 1))" || continue
  if [ -n "$FROM" ] && [ "$started" -eq 0 ]; then
    sel_match "$FROM" "$id" || continue
    started=1
  fi
  executed=$((executed + 1))
  if [ "$DRY" -eq 1 ]; then
    echo "[dry-run] ${cmds[$((i - 1))]}"
    continue
  fi
  log=$OUT/${id//\//_}.log
  printf '\n== %s ==\n' "$id"
  t0=$(date +%s)
  if [ "$LIVE" = "1" ]; then
    HX_TEST_PATH=${id%%#*} HX_VARIANT=$var bash "$file" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
  else
    HX_TEST_PATH=${id%%#*} HX_VARIANT=$var bash "$file" > "$log" 2>&1
    rc=$?
  fi
  t1=$(date +%s); secs=$((t1 - t0))
  [ "$LIVE" = "1" ] || sed 's/^/   /' "$log"
  case "$rc" in
    0)  st=PASS; pass=$((pass + 1)) ;;
    77) st=SKIP; skipped=$((skipped + 1)) ;;
    *)  st=FAIL; fail=$((fail + 1))
        if [ -z "$first_fail" ]; then first_fail=$id; first_cmd=${cmds[$((i - 1))]}; fi ;;
  esac
  printf '%s\t%s\t%s\t%s\n' "$st" "$id" "$secs" "$log" >> "$status_file"
  if [ "$st" = FAIL ] && [ "$KEEP" -eq 0 ]; then
    echo "[run_tests] first FAIL -> stop（--keep-going 可继续）"
    break
  fi
done

[ "$executed" -gt 0 ] || { echo "run_tests: no test matched (only=$ONLY tag=$TAGS from=$FROM)" >&2; exit 2; }

echo
echo "[run_tests] ---- summary ----"
awk -F'\t' '{printf "  %-4s %-46s %ss\n", $1, $2, $3}' "$status_file"
echo "[run_tests] pass=$pass skip=$skipped fail=$fail out=$OUT"
if [ -n "$first_fail" ]; then
  echo "[run_tests] next: $first_cmd"
fi

python3 - "$status_file" "$OUT/results.json" <<'PY'
import json, pathlib, sys
lines = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
rows = [line.split("\t") for line in lines if line.strip()]
data = [{"status": r[0], "id": r[1], "seconds": int(r[2]), "log": r[3]} for r in rows]
pathlib.Path(sys.argv[2]).write_text(json.dumps(data, indent=2), encoding="utf-8")
print("[run_tests] results -> %s" % sys.argv[2])
PY

rc=0
[ "$fail" -gt 0 ] && rc=1
[ "$STRICT" -eq 1 ] && [ "$skipped" -gt 0 ] && rc=1
echo "[run_tests] exit=$rc"
exit $rc
