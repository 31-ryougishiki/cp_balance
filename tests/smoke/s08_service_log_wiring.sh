#!/usr/bin/env bash
# desc:     服务日志链路（假服务，秒级）：落盘 / --live-log 实时上屏 / 端口捕获不被日志污染 / 停干净
# needs:    none
# tags:     fast, offline
# est:      40s
#
# 回归对象：hx_service_up 的日志与端口契约（tests/lib/service.sh）。
# 曾经的做法是在 hx_service_up 里 `> >(tee "$log")`，而它总在 `port=$(...)` 里被调用：
# tee 拿到的是命令替换的管道 —— 日志被 $() 吞掉（屏幕上看不到），
# 而且 tee 一直占着写端，$() 要等服务结束才返回（整条 verify 卡死）。
# 这里用假服务把三条契约钉住：文件必须有内容、live 时同一份内容要上屏、$() 只拿到端口。
set -uo pipefail
source "$(dirname "$0")/../lib/common.sh"

port=18099                       # 自检专用端口，不占配置里的服务端口
cfg=${HX_OUT#"$HARNESS_ROOT"/}/_fake_service.json   # 相对 harness 根目录，两侧 python 都能解析
svc=$HX_OUT/_fake_service.sh
fake_log=$HX_OUT/fake_service.log
note_log=$HX_OUT/notes.log
stream_log=$HX_OUT/fake_service.stream.log

cat > "$cfg" <<JSON
{"name": "_fake_service", "port": $port}
JSON
cat > "$svc" <<'SH'
#!/usr/bin/env bash
# 假服务：先打启动日志，隔几秒才让 /v1/models 就绪（覆盖"边启动边看日志"）
echo "fake: boot pid=$$"
for i in 1 2 3; do echo "fake: boot line $i"; sleep 1; done
python3 - "$1" <<'PY' &
import http.server, json, sys
port = json.load(open(sys.argv[1]))["port"]
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "fake"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, *args):
        pass
http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()
PY
child=$!
trap 'kill "$child" 2>/dev/null' TERM INT   # 真启动器（serve_config.py）也是这样收子进程的
wait
SH
export HX_SVC_CMD="bash $svc"
# 心跳间隔压到 1s：假服务要 3s 才就绪，保证能看到进度提示
export HX_READY_SLEEP=1 HX_READY_TRIES=30 HX_READY_NOTE_S=1 HX_STOP_TRIES=6 HX_STOP_SLEEP=1

hx_ck() {  # <判据描述> <0/1>
  if [ "$2" = "0" ]; then hx_ok "$1"; else hx_fail "$1"; fi
}
hx_proc_count() { ps -ef 2>/dev/null | grep -F "$1" | grep -v grep | wc -l; }

# 1) 默认（不上屏）：日志落盘、进度提示里给出日志路径、$() 只拿到端口、服务在跑也立刻返回
t0=$(date +%s)
got=$(hx_service_up "$cfg" "$fake_log" 2> "$note_log"); rc=$?
elapsed=$(( $(date +%s) - t0 ))
hx_ck "hx_service_up 返回 0" "$rc"
hx_ck "端口捕获是纯端口（实际 '$got'）" "$([ "$got" = "$port" ] && echo 0 || echo 1)"
hx_ck "服务在跑也立刻返回（${elapsed}s，不等服务退出）" "$([ "$elapsed" -le 10 ] && echo 0 || echo 1)"
hx_ck "服务日志落盘非空（$(wc -c < "$fake_log" 2>/dev/null || echo 0) 字节）" "$([ -s "$fake_log" ] && echo 0 || echo 1)"
grep -q "fake: boot line 3" "$fake_log"; hx_ck "日志内容完整（含启动日志）" "$?"
grep -q "服务日志：$fake_log\|日志 $fake_log" "$note_log"; hx_ck "提示里给出了日志路径" "$?"
grep -q "等服务就绪" "$note_log"; hx_ck "等待就绪时有进度提示（心跳）" "$?"

# 2) live-log：同一份日志要实时上屏（另起子 shell，把它的屏幕输出收下来看）
( HX_STREAM_SERVICE_LOG=1; port=$(hx_service_up "$cfg" "$fake_log"); echo "live-got=$port" ) \
  > "$stream_log" 2>&1
sleep 0.5
grep -q "^live-got=$port$" "$stream_log"; hx_ck "live 模式下端口捕获仍未被日志污染" "$?"
grep -q "fake: boot line 3" "$stream_log"; hx_ck "live 模式下服务日志实时上屏" "$?"
hx_note "live 捕获到的屏幕输出（前 8 行，就是 --live-log 时你会看到的东西）："
sed -n '1,8p' "$stream_log" | sed 's/^/   | /'

# 3) 停干净：端口释放、跟随进程与假服务都不残留、state 文件清掉
hx_service_down "$port" >/dev/null 2>&1
hx_ck "停服后端口不再应答" "$([ "$($HX_PY listen "$port")" = "0" ] && echo 0 || echo 1)"
sleep 1
hx_ck "假服务进程收干净了（剩 $(hx_proc_count "$svc") 个）" \
  "$([ "$(hx_proc_count "$svc")" = "0" ] && echo 0 || echo 1)"
hx_ck "没有残留的日志跟随进程（剩 $(hx_proc_count "tail -F -n +1 $fake_log") 个）" \
  "$([ "$(hx_proc_count "tail -F -n +1 $fake_log")" = "0" ] && echo 0 || echo 1)"
hx_ck "state 文件已清理" "$([ -f "$HX_SVC_STATE_DIR/port-$port.state" ] && echo 1 || echo 0)"

hx_end
