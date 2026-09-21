#!/usr/bin/env bash
# 兼容入口：等价于 bash verify.sh --family a5 "$@"（逻辑全在 verify.sh，配置全在 harness.json）
exec bash "$(cd "$(dirname "$0")" && pwd)/verify.sh" --family a5 "$@"
