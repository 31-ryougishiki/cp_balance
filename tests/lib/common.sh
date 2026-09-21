#!/usr/bin/env bash
# tests/lib/common.sh —— 所有测试共用：角色表查询、配置读取、断言、服务起停（用法见 tests/README.md）。
HX_LIB=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HARNESS_ROOT=$(cd "$HX_LIB/../.." && pwd)
cd "$HARNESS_ROOT"
HX_PY="python3 $HARNESS_ROOT/tests/lib/hx.py"
HX_ROLES=$HX_LIB/roles.tsv

: "${CP_BALANCE_FAMILY:=a5}"
# 远端常设 http_proxy/https_proxy 又没有 no_proxy：那会把 127.0.0.1 的就绪探测也发给代理，
# 服务明明起来了却一直等不到 ready。这里给本机地址开白名单（外网流量仍走代理）。
no_proxy="127.0.0.1,localhost,::1${no_proxy:+,$no_proxy}"; export no_proxy
NO_PROXY="$no_proxy"; export NO_PROXY
: "${HARNESS_OUT:=$HARNESS_ROOT/tests/_out/$(date +%m%d_%H%M%S)}"
# 超时/阈值来自 harness.json limits（环境变量仍可覆盖）
hx_limit() { $HX_PY limit "$1" 2>/dev/null | tr -d '\r'; }
HX_READY_TRIES=${HX_READY_TRIES:-$(hx_limit ready_tries)};  : "${HX_READY_TRIES:=360}"
HX_READY_SLEEP=${HX_READY_SLEEP:-$(hx_limit poll_seconds)}; : "${HX_READY_SLEEP:=5}"
HX_STOP_TRIES=${HX_STOP_TRIES:-$(hx_limit stop_tries)};     : "${HX_STOP_TRIES:=24}"
HX_STOP_SLEEP=${HX_STOP_SLEEP:-$(hx_limit stop_seconds)};   : "${HX_STOP_SLEEP:=5}"
HX_TEST_PATH=${HX_TEST_PATH:-$(basename "$0" .sh)}   # 例：accuracy/a10_matrix_gate
HX_VARIANT=${HX_VARIANT:-${1:-}}                     # 单独跑时位置参数就是变体
HX_TEST_ID=$HX_TEST_PATH
[ -n "$HX_VARIANT" ] && HX_TEST_ID=$HX_TEST_ID#$HX_VARIANT
HX_OUT=$HARNESS_OUT/$HX_TEST_PATH
[ -n "$HX_VARIANT" ] && HX_OUT=$HX_OUT.$HX_VARIANT
mkdir -p "$HX_OUT"

# ---- 角色表 tests/lib/roles.tsv：family / group / role / config ----
hx_role() {  # <role>：打印配置；本族有该角色但未配配置则打印空；表里没有该角色返回 1
  local out
  out=$(awk -F'\t' -v f="$CP_BALANCE_FAMILY" -v r="$1" '$1 == f && $3 == r {print $4; exit}' "$HX_ROLES")
  [ -n "$out" ] || return 1
  [ "$out" = "-" ] || printf '%s\n' "$out"
  return 0
}

hx_group() {  # <group>：本族该组的配置，每行一个
  awk -F'\t' -v f="$CP_BALANCE_FAMILY" -v g="$1" '$1 == f && $2 == g && $4 != "-" {print $4}' "$HX_ROLES"
}

hx_resolve() {  # 角色名 -> 配置名；也可以直接写 configs/<名字>.json 里的名字
  local out
  out=$(hx_role "$1") || { [ -f "$HARNESS_ROOT/configs/$1.json" ] || return 1; echo "$1"; return 0; }
  [ -n "$out" ] && echo "$out"
}

# ---- 配置读取 ----
hx_cfg_field()      { $HX_PY field "$1" "$2" 2>/dev/null | tr -d '\r'; }
hx_fingerprint()    { $HX_PY fingerprint "$1" 2>/dev/null | tr -d '\r'; }
hx_path()           { $HX_PY path "$1" "$2" 2>/dev/null | tr -d '\r'; }
hx_family_configs() { $HX_PY configs "$CP_BALANCE_FAMILY" 2>/dev/null | tr -d '\r'; }
hx_cur_repo() {
  [ -n "${CP_BALANCE_REPO:-}" ] && { echo "$CP_BALANCE_REPO"; return 0; }
  $HX_PY static_check "$(hx_role b_matrix)" 2>/dev/null | tr -d '\r' | sed -n 's/^repo=//p'
}
hx_base_repo() {
  [ -n "${CP_BALANCE_BASE_REPO:-}" ] && { echo "$CP_BALANCE_BASE_REPO"; return 0; }
  $HX_PY static_check "$(hx_role b_matrix)" 2>/dev/null | tr -d '\r' | sed -n 's/^base_repo=//p'
}

# ---- 输出与断言：0=PASS，77=SKIP，其它=FAIL ----
HX_FAILED=0
hx_ok()   { echo "[ok]   $*"; }
hx_note() { echo "[note] $*"; }
hx_warn() { echo "[warn] $*"; }
hx_fail() { echo "[FAIL] $*"; HX_FAILED=$((HX_FAILED + 1)); }
hx_skip() { echo "[skip] $*"; exit 77; }
hx_end() {
  if [ "$HX_FAILED" -eq 0 ]; then
    echo "[done] $HX_TEST_ID PASS"
    return 0
  fi
  echo "[done] $HX_TEST_ID FAIL ($HX_FAILED)"
  return 1
}

# ---- 服务级测试共用（tests/service/）：起服务 + HTTP 断言 + 阈值读取 + 上一轮产物 ----
hx_svc_up() {  # <role|config>：成功导出 HX_CFG/HX_LOG/HX_PORT/HX_BASE，失败返回 1
  HX_CFG=$(hx_resolve "$1") || return 1
  [ -n "$HX_CFG" ] || return 1
  HX_LOG=$HX_OUT/$HX_CFG.log
  HX_PORT=$(hx_service_up "$HX_CFG" "$HX_LOG") || { HX_PORT=""; return 1; }
  HX_BASE=http://127.0.0.1:$HX_PORT
  export HX_CFG HX_LOG HX_PORT HX_BASE
  return 0
}

hx_http_code() {  # 与 curl 同参数，只打印状态码（本机地址绕开代理）
  curl -s --noproxy "*" -o /dev/null -w "%{http_code}" "$@"
}

hx_service_kv() {  # <harness.json service.<键>>：服务级阈值/计划文件，缺省打印空
  $HX_PY harness "service.$1" 2>/dev/null
}

hx_prev_out() {  # <测试路径> <变体>：同一次 run 里前一个测试的产物目录，找不到就打印空
  local dir=$HARNESS_OUT/$1.$2
  if [ ! -d "$dir" ]; then dir=$(ls -dt "$HARNESS_OUT/$1.$2"* 2>/dev/null | head -1); fi
  if [ -n "$dir" ] && [ -d "$dir" ]; then echo "$dir"; else echo ""; fi
}

hx_need_dir() {  # <标签> <路径>
  [ -n "$2" ] || { hx_fail "$1: not set in config"; return 0; }
  [ -d "$2" ] && hx_ok "$1 $2" || hx_fail "$1 missing: $2"
}

hx_need_file() {  # <标签> <路径>
  [ -n "$2" ] || { hx_fail "$1: not set in config"; return 0; }
  [ -f "$2" ] && hx_ok "$1 $2" || hx_fail "$1 missing: $2"
}

# ---- 服务生命周期：登记端口后立刻挂 EXIT trap，起服务中途失败/被打断也能收尾 ----
# 进程与日志由 tests/lib/service.sh 负责；这里只做配置解析并保持原来的调用方式：
#   port=$(hx_service_up <config> <logfile>)   # stdout 里只能有端口，别的都走 stderr
. "$HX_LIB/service.sh"
HX_PORT=""
hx_cleanup() { hx_svc_stop_all >/dev/null 2>&1; return 0; }
trap hx_cleanup EXIT

hx_service_up() {  # <config> <logfile>；成功只把端口打到 stdout（stderr 上给日志路径与进度）
  local cfg=$1 log=$2 port
  port=$(hx_cfg_field "$cfg" port)
  [ -n "$port" ] || { echo "[FAIL] $cfg has no port" >&2; return 1; }
  if ! hx_service_down "$port"; then
    echo "[FAIL] port $port still answers before starting $cfg (stop it first)" >&2
    return 1
  fi
  HX_PORT=$port
  hx_svc_start "$port" "$cfg" "$log" || return 1
  # 服务起停约 10 分钟，重试上限与报进度间隔见 harness.json limits
  if ! hx_svc_wait_ready "$cfg" "$log" "$port"; then
    hx_svc_stop "$port" >/dev/null 2>&1
    HX_PORT=""
    return 1
  fi
  hx_svc_note "$cfg 就绪：port=$port 日志=$log"
  echo "$port"
}

hx_service_down() {  # <port>：端口已停返回 0，HX_STOP_TRIES*HX_STOP_SLEEP 秒后仍在应答返回 1
  hx_svc_stop "$1" || { hx_warn "port $1 still answering after stop"; return 1; }
  HX_PORT=""
  return 0
}
