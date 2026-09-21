#!/usr/bin/env bash
# tests/lib/service.sh —— 模型服务的生命周期与日志（单一职责，common.sh 只负责配置/断言）。
#
# 约定（踩过的坑，别改回去）：
#   * 服务进程的 stdout/stderr 永远直接重定向到日志文件，中间不放 tee/管道：日志文件是
#     唯一真相，跟随进程死掉也不影响服务与落盘。
#   * 实时上屏用另起的 tail -F 跟同一份文件，输出走 stderr。这样调用方
#     `port=$(hx_service_up ...)` 的 stdout 里只有端口，也不会因为跟随进程占着管道而挂住。
#   * 服务的 pid / 跟随进程的 pid 写在 $HX_SVC_STATE_DIR/port-<port>.state：hx_service_up 一定
#     在 $( ) 子 shell 里跑，普通变量带不出来，停服务只能靠这个文件（旧实现因此只能 pkill）。
HX_SVC_STATE_DIR=${HX_SVC_STATE_DIR:-${HARNESS_OUT:-/tmp}/.svc}
# 起服务的命令；自检（tests/smoke/s08_service_log_wiring.sh）把它换成假服务，参数仍是 <config>
HX_SVC_CMD=${HX_SVC_CMD:-bash run.sh}
# 包装层：服务进程把自己的 pid 追加进 state 文件再 exec（exec 后 pid 不变）。
# 必须由服务自己报 pid：`setsid` 在调用方已是进程组组长时会 fork 后退出，`$!` 就成了死 pid，
# 用它判断"进程是否已退出"会把还在加载的服务误判成崩溃，也停不掉它。
HX_SVC_RECORD_PID='printf "pid=%s\n" "$$" >> "$1"; shift; exec "$@"'

hx_svc_note() { printf '[note] %s\n' "$*" >&2; }

hx_svc_state_file() { printf '%s/port-%s.state\n' "$HX_SVC_STATE_DIR" "$1"; }

hx_svc_state_get() {  # <state文件> <键>：后写的覆盖先写的
  [ -f "$1" ] || return 1
  sed -n "s/^$2=//p" "$1" | tail -n 1
}

hx_svc_follow_stop() {  # <pid>：收掉日志跟随进程，否则 run_tests --live-log 的管道不会关
  local pid=$1
  [ -n "$pid" ] || return 0
  kill "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  return 0
}

hx_svc_start() {  # <port> <config> <log>：起服务（stdout/stderr -> $log），可选的跟随进程
  local port=$1 cfg=$2 log=$3 state pid use_setsid=0 tail_pid=""
  local -a cmd
  read -r -a cmd <<< "$HX_SVC_CMD"
  [ ${#cmd[@]} -gt 0 ] || { printf '[FAIL] HX_SVC_CMD 是空的\n' >&2; return 1; }
  mkdir -p "$(dirname "$log")" "$HX_SVC_STATE_DIR"
  state=$(hx_svc_state_file "$port")
  command -v setsid >/dev/null 2>&1 && use_setsid=1
  # state 先写好（服务会往同一个文件追加自己的 pid，所以这里只能用 >，不能晚于启动）
  printf 'setsid=%s\nport=%s\ncfg=%s\nlog=%s\n' "$use_setsid" "$port" "$cfg" "$log" > "$state"
  : > "$log"
  # 独立会话，停服时能连整组 mp worker 一起收：setsid 没有就退回普通后台进程
  if [ "$use_setsid" = "1" ]; then
    setsid bash -c "$HX_SVC_RECORD_PID" _ "$state" "${cmd[@]}" "$cfg" > "$log" 2>&1 &
  else
    nohup bash -c "$HX_SVC_RECORD_PID" _ "$state" "${cmd[@]}" "$cfg" > "$log" 2>&1 &
  fi
  pid=$!
  printf 'launch_pid=%s\n' "$pid" >> "$state"
  if [ "${HX_STREAM_SERVICE_LOG:-0}" = "1" ]; then
    if command -v tail >/dev/null 2>&1; then
      tail -F -n +1 "$log" >&2 &
      tail_pid=$!
    else
      printf '[warn] 没有 tail：--live-log 打不出服务日志（文件照旧写 %s）\n' "$log" >&2
    fi
  fi
  printf 'tail=%s\n' "$tail_pid" >> "$state"
  hx_svc_note "$cfg 启动中（日志 $log；实时看：tail -f $log）"
  return 0
}

hx_svc_report_failure() {  # <config> <log> <原因>：失败现场必须带日志路径、大小和内容
  local cfg=$1 log=$2 why=$3 size
  size=$(wc -c < "$log" 2>/dev/null || echo 0)
  printf '[FAIL] %s %s\n' "$cfg" "$why" >&2
  printf '[FAIL]   服务日志：%s（%s 字节）\n' "$log" "$size" >&2
  if [ "${size:-0}" -eq 0 ]; then
    printf '[warn]   日志是空的：进程一行输出都没有，不是被 harness 丢掉；看 NPU/权重/依赖，' >&2
    printf 'vllm-ascend 自己的文件日志在 ASCEND_PROCESS_LOG_PATH 或 ~/ascend/log/vllm_ascend/\n' >&2
  else
    printf '[warn]   日志尾部：\n' >&2
    tail -n 20 "$log" | sed 's/^/   /' >&2
  fi
}

hx_svc_wait_ready() {  # <config> <log> <port>：等到 /v1/models 就绪；过程中定期报进度
  local cfg=$1 log=$2 port=$3 i tries=${HX_READY_TRIES:-360} sleep_s=${HX_READY_SLEEP:-5}
  local note_s=${HX_READY_NOTE_S:-60} every last pid
  local state; state=$(hx_svc_state_file "$port")
  [ "$sleep_s" -gt 0 ] || sleep_s=1
  every=$(( note_s / sleep_s )); [ "$every" -lt 1 ] && every=1
  pid=$(hx_svc_state_get "$state" pid || true)
  for i in $(seq 1 "$tries"); do
    curl -sf --noproxy '*' "http://127.0.0.1:$port/v1/models" >/dev/null && return 0
    if [ -n "${pid:-}" ] && ! kill -0 "$pid" 2>/dev/null; then
      hx_svc_report_failure "$cfg" "$log" "启动进程已退出（pid $pid）"
      return 1
    fi
    if [ $(( i % every )) -eq 0 ]; then
      last=$(tail -n 1 "$log" 2>/dev/null | cut -c1-160)
      if [ -n "$last" ]; then
        hx_svc_note "$cfg 等服务就绪 $(( i * sleep_s ))s，日志尾行：$last"
      else
        hx_svc_note "$cfg 等服务就绪 $(( i * sleep_s ))s，服务日志还是空的（$log）"
      fi
    fi
    sleep "$sleep_s"
  done
  hx_svc_report_failure "$cfg" "$log" "等了 $(( tries * sleep_s ))s 仍未就绪"
  return 1
}

hx_svc_stop() {  # <port>：先收跟随进程，再收服务（会话/进程组），最后按端口兜底；端口已停返回 0
  local port=$1 state pid use_setsid tail_pid i tries=${HX_STOP_TRIES:-24} sleep_s=${HX_STOP_SLEEP:-5}
  state=$(hx_svc_state_file "$port")
  pid=$(hx_svc_state_get "$state" pid || true)
  [ -n "${pid:-}" ] || pid=$(hx_svc_state_get "$state" launch_pid || true)   # 服务没报 pid 时退回包装层
  use_setsid=$(hx_svc_state_get "$state" setsid || true)
  tail_pid=$(hx_svc_state_get "$state" tail || true)
  hx_svc_follow_stop "$tail_pid"
  if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
    if [ "${use_setsid:-0}" = "1" ]; then
      kill -TERM -"$pid" 2>/dev/null    # 整个会话/进程组：vllm 的 mp worker 也在里面
    else
      kill -TERM "$pid" 2>/dev/null
    fi
  fi
  pkill -f -- "--port $port" >/dev/null 2>&1
  for i in $(seq 1 "$tries"); do
    if ! curl -sf --noproxy '*' "http://127.0.0.1:$port/v1/models" >/dev/null; then
      rm -f "$state"
      return 0
    fi
    sleep "$sleep_s"
  done
  printf '[warn] port %s 停了 %ss 还在应答（state 留在 %s 供排查）\n' \
    "$port" "$(( tries * sleep_s ))" "$state" >&2
  return 1
}

hx_svc_stop_all() {  # 收尾用：本轮的 state 文件全收掉（测试异常退出时服务不会漏在卡上）
  local state port rc=0
  [ -d "$HX_SVC_STATE_DIR" ] || return 0
  for state in "$HX_SVC_STATE_DIR"/port-*.state; do
    [ -f "$state" ] || continue
    port=${state##*/port-}; port=${port%.state}
    hx_svc_stop "$port" || rc=1
  done
  return "$rc"
}
