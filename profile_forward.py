#!/usr/bin/env python3
"""Collect one profiling trace per config for the prefill forward.

    python3 profile_forward.py glm52_cur_cp0_prof glm52_cur_cp1_prof

For every config the script starts the service, sends ``--warmup`` long
requests to pay the one-time costs (HCCL setup, metadata caches, MoE
workspace), then ``POST /start_profile``, ``--requests`` long requests,
``POST /stop_profile``, stops the service and writes ``prof_<name>.json``.

The config must set ``"profiler": {"enabled": true}``: ``/start_profile`` only
exists when the service was launched with ``--profiler-config``.

Requests are sent one at a time, so each one is its own prefill batch and the
two runs have identical batch shapes.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import serve_config

HERE = Path(__file__).resolve().parent


def log(message: str) -> None:
    print(message, flush=True)


def load_cases(kind: str) -> tuple[list[dict], str]:
    payload = json.loads(io.open(HERE / "questions.json", encoding="utf-8").read())
    items = list(payload.get("items") or payload)
    cases = [item for item in items if str(item.get("kind") or "") == kind]
    if not cases:
        raise SystemExit("no cases of kind " + kind)
    return cases, str(payload.get("model") or "")


def post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_empty(url: str, timeout: float) -> int:
    request = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def wait_ready(port: int, proc, timeout: int) -> None:
    url = "http://127.0.0.1:%s/v1/models" % port
    waited = 0
    while waited < timeout:
        if proc.poll() is not None:
            raise SystemExit("service exited rc=%s" % proc.returncode)
        try:
            with urllib.request.urlopen(url, timeout=5):
                log("[profile] ready after %s seconds" % waited)
                return
        except Exception:
            time.sleep(10)
            waited += 10
    raise SystemExit("service not ready after %s seconds" % timeout)


def wait_port_free(port: int, timeout: int = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return
        time.sleep(2)
    raise SystemExit("port %s still busy" % port)


def stop_service(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if hasattr(os, "killpg"):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    else:
        proc.terminate()
    deadline = time.time() + 120
    while time.time() < deadline and proc.poll() is None:
        time.sleep(2)
    if proc.poll() is None and hasattr(os, "killpg"):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    time.sleep(5)


def send_one(endpoint: str, model: str, prompt: str, max_tokens: int, timeout: float) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_completion_tokens": max_tokens,
        "temperature": 0,
    }
    started = time.time()
    try:
        response = post_json(endpoint, payload, timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit("completion failed %s: %s" % (exc.code, detail[:400]))
    elapsed = time.time() - started
    usage = response.get("usage") or {}
    return {
        "elapsed_s": round(elapsed, 4),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def run_config(name: str, args: argparse.Namespace) -> dict:
    cfg = serve_config.load_config(name)
    if not (cfg.get("profiler") or {}).get("enabled"):
        raise SystemExit(
            "config %s must set \"profiler\": {\"enabled\": true}" % name
        )
    prof_dir = serve_config.profiler_dir(cfg)
    env = serve_config.build_env(cfg)
    argv = serve_config.build_argv(cfg)
    port = int(cfg["port"])
    log(serve_config.fingerprint(cfg, env))
    log("[profile] dir=%s" % prof_dir)
    cases, model_hint = load_cases(args.kind)
    model = args.model or model_hint or "glm-52"
    base = "http://127.0.0.1:%s" % port
    endpoint = base + args.endpoint
    log_path = HERE / ("profile_%s.log" % cfg["name"])
    log("[profile] service log -> %s" % log_path)

    wait_port_free(port)
    with open(log_path, "w", encoding="utf-8") as handle:
        proc = subprocess.Popen(argv, env=env, stdout=handle, stderr=handle)
        try:
            wait_ready(port, proc, args.ready_timeout)
            for index in range(args.warmup):
                case = cases[index % len(cases)]
                result = send_one(
                    endpoint, model, case["prompt"], args.max_completion_tokens, args.timeout
                )
                log("[profile] warmup %d/%d %ss" % (index + 1, args.warmup, result["elapsed_s"]))

            code = post_empty(base + "/start_profile", args.timeout)
            if code != 200:
                raise SystemExit("/start_profile returned %s" % code)
            log("[profile] profiler started")

            samples = []
            for index in range(args.requests):
                case = cases[index % len(cases)]
                result = send_one(
                    endpoint, model, case["prompt"], args.max_completion_tokens, args.timeout
                )
                result["id"] = case.get("id")
                samples.append(result)
                log(
                    "[profile] req %d/%d id=%s %ss prompt_tokens=%s"
                    % (
                        index + 1,
                        args.requests,
                        result["id"],
                        result["elapsed_s"],
                        result["prompt_tokens"],
                    )
                )

            code = post_empty(base + "/stop_profile", args.timeout)
            log("[profile] profiler stopped rc=%s" % code)
        finally:
            stop_service(proc)

    summary = {
        "config": cfg["name"],
        "repo": cfg.get("repo"),
        "head": serve_config.git_head(cfg.get("repo")),
        "cp_balance": cfg.get("cp_balance"),
        "reduce_mode": cfg.get("reduce_mode"),
        "min_tokens": cfg.get("min_tokens"),
        "profiler_dir": prof_dir,
        "kind": args.kind,
        "warmup": args.warmup,
        "requests": args.requests,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples": samples,
        "mean_elapsed_s": round(
            sum(item["elapsed_s"] for item in samples) / max(len(samples), 1), 4
        ),
    }
    out = HERE / ("prof_%s.json" % cfg["name"])
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log("[profile] mean request time %ss -> %s" % (summary["mean_elapsed_s"], out))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+", help="config name under configs/ (profiler enabled)")
    parser.add_argument("--kind", default="long", help="questions.json kind (default: long, >MIN_TOKENS)")
    parser.add_argument("--warmup", type=int, default=2, help="unprofiled requests before start_profile")
    parser.add_argument("--requests", type=int, default=4, help="profiled requests")
    parser.add_argument("--max-completion-tokens", type=int, default=1)
    parser.add_argument("--endpoint", default="/v1/completions")
    parser.add_argument("--model", default="", help="served model name (default: questions.json hint)")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--ready-timeout", type=int, default=1800)
    args = parser.parse_args()

    results = [run_config(name, args) for name in args.configs]
    log("[profile] ---- summary ----")
    for item in results:
        log(
            "[profile] %-24s cp_balance=%s mean_request=%ss dir=%s"
            % (item["config"], item["cp_balance"], item["mean_elapsed_s"], item["profiler_dir"])
        )
    log("[profile] next: python3 profile_analyse.py " + " ".join(item["profiler_dir"] for item in results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
