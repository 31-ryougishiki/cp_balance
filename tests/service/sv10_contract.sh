#!/usr/bin/env bash
# desc:     服务契约与错误路径：/health、/v1/models、/tokenize、坏请求 4xx、流式中途断开后服务仍可用
# needs:    service
# tags:     npu, slow, service
# variants: svc_cp1
# est:      15min
set -uo pipefail
source "$(dirname "$0")/../lib/common.sh"

arg=${1:-${HX_VARIANT:-svc_cp1}}
if ! hx_svc_up "$arg"; then
  hx_fail "service $arg not ready (see $HX_OUT/$arg.log)"
  hx_end
  exit 1
fi
set -- $(hx_cfg_field "$HX_CFG" served_model_name)
model=$1
hx_ok "service $HX_CFG ready on $HX_BASE (model=$model)"

code=$(hx_http_code "$HX_BASE/health")
if [ "$code" = "200" ]; then hx_ok "/health -> 200"; else hx_fail "/health -> $code"; fi

code=$(curl -s --noproxy "*" -o "$HX_OUT/models.json" -w "%{http_code}" "$HX_BASE/v1/models")
if [ "$code" = "200" ] && grep -q "$model" "$HX_OUT/models.json"; then
  hx_ok "/v1/models -> 200 and lists $model"
else
  hx_fail "/v1/models -> $code without $model"
fi

expect_4xx() {  # <标签> <状态码>
  case "$2" in
    4*) hx_ok "$1 -> HTTP $2 (rejected, service kept running)" ;;
    *) hx_fail "$1 -> HTTP $2 (expected a 4xx)" ;;
  esac
}

post_json() {  # <请求体文件> <路径>：打印状态码
  hx_http_code -X POST -H "Content-Type: application/json" --data-binary @"$1" "$HX_BASE$2"
}

printf "%s" "{\"model\": \"$model\", \"prompt\": \"你好\"}" > "$HX_OUT/req_tokenize.json"
code=$(post_json "$HX_OUT/req_tokenize.json" /tokenize)
if [ "$code" = "200" ]; then hx_ok "/tokenize -> 200"; else hx_fail "/tokenize -> $code"; fi

printf "%s" "{\"model\": \"$model\"," > "$HX_OUT/req_malformed.json"
expect_4xx "malformed JSON body" "$(post_json "$HX_OUT/req_malformed.json" /v1/completions)"

python3 - <<PY > "$HX_OUT/req_unknown_model.json"
import json
print(json.dumps({"model": "not-a-served-model", "prompt": "hello", "max_tokens": 4}))
PY
expect_4xx "unknown model name" "$(post_json "$HX_OUT/req_unknown_model.json" /v1/completions)"

python3 - <<PY > "$HX_OUT/req_over_long.json"
import json
items = json.load(open("questions.json"))["items"]
prompt = [item for item in items if item["kind"] == "long"][0]["prompt"]
print(json.dumps({"model": "$model", "prompt": prompt, "max_tokens": 200000}, ensure_ascii=False))
PY
expect_4xx "max_tokens beyond max_model_len" "$(post_json "$HX_OUT/req_over_long.json" /v1/completions)"

python3 - <<PY > "$HX_OUT/req_stream.json"
import json
items = json.load(open("questions.json"))["items"]
prompt = [item for item in items if item["kind"] == "long"][0]["prompt"]
print(json.dumps({"model": "$model", "prompt": prompt, "max_tokens": 256, "stream": True}, ensure_ascii=False))
PY
curl -s --noproxy "*" -N -m 3 -H "Content-Type: application/json" --data-binary @"$HX_OUT/req_stream.json" -o "$HX_OUT/stream.txt" "$HX_BASE/v1/completions"
client_rc=$?
hx_note "client aborted mid-stream: curl rc=$client_rc (28 = cut by its own timeout, as intended)"
code=$(hx_http_code "$HX_BASE/health")
if [ "$code" = "200" ]; then hx_ok "service healthy after the aborted stream"; else hx_fail "health after abort -> $code"; fi

python3 - <<PY > "$HX_OUT/req_ok.json"
import json
print(json.dumps({"model": "$model", "prompt": "你好", "max_tokens": 4}))
PY
code=$(hx_http_code -X POST -H "Content-Type: application/json" --data-binary @"$HX_OUT/req_ok.json" -o "$HX_OUT/ok.json" "$HX_BASE/v1/completions")
size=$(wc -c < "$HX_OUT/ok.json")
if [ "$code" = "200" ] && [ "$size" -gt 20 ]; then hx_ok "recovery request -> 200 ($size bytes)"; else hx_fail "recovery request -> $code ($size bytes)"; fi

tracebacks=$(grep -c -F "Traceback (most recent call last)" "$HX_LOG" || true)
if [ "${tracebacks:-0}" -eq 0 ]; then hx_ok "service log has no traceback"; else hx_warn "service log has $tracebacks traceback(s): $HX_LOG"; fi

if hx_service_down "$HX_PORT"; then hx_ok "service stopped"; else hx_fail "port $HX_PORT still answering after stop"; fi
hx_end
