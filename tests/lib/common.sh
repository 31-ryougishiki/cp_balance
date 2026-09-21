#!/usr/bin/env bash
# tests/lib/common.sh —— 所有测试共用：角色表查询、配置读取、断言、服务起停（用法见 tests/README.md）。
HX_LIB=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HARNESS_ROOT=$(cd "$HX_LIB/../.." && pwd)
cd "$HARNESS_ROOT"
HX_PY="python3 $HARNESS_ROOT/tests/lib/hx.py"
HX_ROLES=$HX_LIB/roles.tsv

: "${CP_BALANCE_FAMILY:=a5}"
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

hx_need_dir() {  # <标签> <路径>
  [ -n "$2" ] || { hx_fail "$1: not set in config"; return 0; }
  [ -d "$2" ] && hx_ok "$1 $2" || hx_fail "$1 missing: $2"
}

hx_need_file() {  # <标签> <路径>
  [ -n "$2" ] || { hx_fail "$1: not set in config"; return 0; }
  [ -f "$2" ] && hx_ok "$1 $2" || hx_fail "$1 missing: $2"
}

# ---- 服务生命周期：登记端口后立刻挂 EXIT trap，起服务中途失败/被打断也能收尾 ----
HX_PORT=""
hx_cleanup() { [ -n "$HX_PORT" ] && hx_service_down "$HX_PORT" >/dev/null 2>&1; return 0; }
trap hx_cleanup EXIT

hx_service_up() {  # <config> <logfile>；成功只把端口打到 stdout
  local cfg=$1 log=$2 port i
  port=$(hx_cfg_field "$cfg" port)
  [ -n "$port" ] || { echo "[FAIL] $cfg has no port" >&2; return 1; }
  if ! hx_service_down "$port"; then
    echo "[FAIL] port $port still answers before starting $cfg (stop it first)" >&2
    return 1
  fi
  HX_PORT=$port
  # 独立会话，停服时能连整组 mp worker 一起收：setsid 没有就退回普通后台进程
  # HX_STREAM_SERVICE_LOG=1：服务日志同时 tee 到测试 stdout（run_tests.sh --live-log 会打开），文件照旧用于解析
  if command -v setsid >/dev/null 2>&1; then
    HX_SETSID=1
    if [ "${HX_STREAM_SERVICE_LOG:-0}" = "1" ]; then
      setsid bash run.sh "$cfg" > >(tee "$log") 2>&1 &
    else
      setsid bash run.sh "$cfg" > "$log" 2>&1 &
    fi
  else
    HX_SETSID=0
    if [ "${HX_STREAM_SERVICE_LOG:-0}" = "1" ]; then
      nohup bash run.sh "$cfg" > >(tee "$log") 2>&1 &
    else
      nohup bash run.sh "$cfg" > "$log" 2>&1 &
    fi
  fi
  HX_PID=$!
  # 服务起停约 10 分钟，重试上限见 harness.json limits.ready_tries（HX_READY_TRIES 可覆盖）
  for i in $(seq 1 "$HX_READY_TRIES"); do
    curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null && break
    if [ -n "$HX_PID" ] && ! kill -0 "$HX_PID" 2>/dev/null; then
      echo "[FAIL] $cfg exited before it became ready (pid $HX_PID); tail of $log:" >&2
      tail -n 20 "$log" >&2
      HX_PORT=""
      return 1
    fi
    sleep "$HX_READY_SLEEP"
  done
  if ! curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null; then
    echo "[FAIL] $cfg not ready on port $port after $((HX_READY_TRIES * HX_READY_SLEEP))s; tail of $log:" >&2
    tail -n 20 "$log" >&2
    return 1
  fi
  echo "$port"
}

hx_service_down() {  # <port>：端口已停返回 0，HX_STOP_TRIES*HX_STOP_SLEEP 秒后仍在应答返回 1
  local port=$1 i
  if [ -n "${HX_PID:-}" ]; then
    if [ "${HX_SETSID:-0}" = "1" ]; then
      kill -TERM -"$HX_PID" 2>/dev/null    # 整个会话/进程组：vllm 的 mp worker 也在里面
    else
      kill -TERM "$HX_PID" 2>/dev/null
    fi
    HX_PID=""
  fi
  pkill -f -- "--port $port" >/dev/null 2>&1
  for i in $(seq 1 "$HX_STOP_TRIES"); do
    if ! curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null; then
      HX_PORT=""
      return 0
    fi
    sleep "$HX_STOP_SLEEP"
  done
  hx_warn "port $port still answering $((HX_STOP_TRIES * HX_STOP_SLEEP))s after stop"
  HX_PORT=""
  return 1
}
