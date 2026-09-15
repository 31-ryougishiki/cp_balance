#!/usr/bin/env python3
"""Collect one profiling window per prompt length, prefill only.

    python3 profile_forward.py prof_cur_cp1

Per config the script starts the service once and then, for every length in the
config's "lengths" list, runs one round:

    warmup request (unprofiled) -> POST /start_profile -> one request with
    max_completion_tokens=1 -> POST /stop_profile -> settle

The profiler config carries delay_iterations=0 / max_iterations=1, so the window
holds one prefill step and at most one decode step instead of everything that
happens until /stop_profile arrives.  That matters: with an unbounded window the
previous round captured roughly 64 steps for 4 requests, and the composition was
dominated by decode, which cp_balance does not touch at all.

Prompts are built deterministically from questions.json and measured with
/tokenize, so the same target length produces the same prompt in every config.

Everything the later stages need lands in <profiler_dir>/windows.json: which
length each window belongs to, the real prompt token count, the window id (the
timestamp inside the rank directory name) and the client wall time, both with
profiling on and during the unprofiled second pass.

Configs without "lengths" keep the old single-window behaviour.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import serve_config

HERE = Path(__file__).resolve().parent
WINDOW_RE = re.compile(r"_(\d{17})_ascend_pt$")


def log(message: str) -> None:
    print(message, flush=True)


def prepare_dir(path: str) -> None:
    """torch_npu refuses to parse a directory the current user cannot write."""
    Path(path).mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o755)


def dir_size_mb(path: str) -> float:
    total = 0
    for item in Path(path).rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            pass
    return total / 1e6


def load_cases(kind: str) -> tuple:
    payload = json.loads(io.open(HERE / "questions.json", encoding="utf-8").read())
    items = list(payload.get("items") or payload)
    cases = [item for item in items if str(item.get("kind") or "") == kind]
    if not cases:
        raise SystemExit("no cases of kind " + kind)
    return cases, str(payload.get("model") or "")


def source_text(cases: list) -> str:
    """One long deterministic body to slice prompts out of."""
    parts = []
    for case in cases:
        text = str(case.get("article") or case.get("prompt") or "")
        if text:
            parts.append(text)
    if not parts:
        raise SystemExit("questions.json has no article text to build prompts from")
    return "\n".join(parts)


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


def wait_port_free(port: int, timeout: int = 300) -> None:
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


def count_tokens(base: str, model: str, text: str, timeout: float) -> int:
    try:
        payload = post_json(base + "/tokenize", {"model": model, "prompt": text}, timeout)
    except Exception as exc:  # noqa: BLE001 - fall back to the character estimate
        log("[profile] /tokenize unavailable (%s), using a character estimate" % exc)
        return 0
    return int(payload.get("count") or 0)


def make_prompt(base: str, model: str, target: int, source: str, timeout: float) -> tuple:
    """Smallest prefix of the repeated source whose token count reaches target."""
    body = source
    while len(body) < max(target * 4, 8192):
        body += "\n" + source
    if count_tokens(base, model, body[:64], timeout) == 0:
        chars = min(len(body), int(target))
        return body[:chars], target
    low, high, best = 1, len(body), None
    while low <= high:
        mid = (low + high) // 2
        found = count_tokens(base, model, body[:mid], timeout)
        if found >= target:
            best = (mid, found)
            high = mid - 1
        else:
            low = mid + 1
    if best is None:
        return body, count_tokens(base, model, body, timeout)
    return body[: best[0]], best[1]


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


def window_ids(prof_dir: str, before: set) -> list:
    found = []
    for item in sorted(Path(prof_dir).rglob("*_ascend_pt")):
        if not item.is_dir() or item.name in before:
            continue
        match = WINDOW_RE.search(item.name)
        if match:
            found.append(match.group(1))
    return sorted(set(found))


def snapshot(prof_dir: str) -> set:
    return {item.name for item in Path(prof_dir).rglob("*_ascend_pt") if item.is_dir()}


def run_config(name: str, args: argparse.Namespace) -> dict:
    cfg = serve_config.load_config(name)
    if not (cfg.get("profiler") or {}).get("enabled"):
        raise SystemExit('config %s must set "profiler": {"enabled": true}' % name)
    prof_dir = serve_config.profiler_dir(cfg)
    env = serve_config.build_env(cfg)
    argv = serve_config.build_argv(cfg)
    port = int(cfg["port"])
    fingerprint = serve_config.fingerprint(cfg, env)
    log(fingerprint)
    log("[profile] dir=%s" % prof_dir)
    cases, model_hint = load_cases(args.kind)
    model = args.model or model_hint or "glm-52"
    base = "http://127.0.0.1:%s" % port
    endpoint = base + args.endpoint
    lengths = [int(x) for x in (cfg.get("lengths") or [])]
    log_path = HERE / ("profile_%s.log" % cfg["name"])
    log("[profile] service log -> %s" % log_path)

    prepare_dir(prof_dir)
    before = snapshot(prof_dir)
    wait_port_free(port)

    entries = []
    with open(log_path, "w", encoding="utf-8") as handle:
        proc = subprocess.Popen(argv, env=env, stdout=handle, stderr=handle, start_new_session=True)
        try:
            wait_ready(port, proc, args.ready_timeout)

            if lengths:
                body = source_text(cases)
                for target in lengths:
                    prompt, tokens = make_prompt(base, model, target, body, args.timeout)
                    warm = send_one(endpoint, model, prompt, 1, args.timeout)
                    log(
                        "[profile] len=%-6d warmup tokens=%s wall=%.2fs"
                        % (target, warm["prompt_tokens"], warm["elapsed_s"])
                    )
                    code = post_empty(base + "/start_profile", args.timeout)
                    if code != 200:
                        raise SystemExit("/start_profile returned %s" % code)
                    result = send_one(endpoint, model, prompt, 1, args.timeout)
                    stop_rc = post_empty(base + "/stop_profile", args.timeout)
                    time.sleep(args.settle)
                    ids = window_ids(prof_dir, before)
                    before = snapshot(prof_dir)
                    entries.append(
                        {
                            "label": "len%d" % target,
                            "target_tokens": target,
                            "prompt_tokens": result["prompt_tokens"],
                            "tokenized": tokens,
                            "wall_s": result["elapsed_s"],
                            "stop_rc": stop_rc,
                            "window_ids": ids,
                            "completion_tokens": result["completion_tokens"],
                        }
                    )
                    log(
                        "[profile] len=%-6d tokens=%s stop=%s window=%s wall=%.2fs"
                        % (target, result["prompt_tokens"], stop_rc, ids, result["elapsed_s"])
                    )

                if args.clean_pass:
                    # Same ladder again with the profiler off: the only honest
                    # end-to-end numbers, and the reference the profiled pass is
                    # inflated against.
                    for entry in entries:
                        prompt, _ = make_prompt(base, model, entry["target_tokens"], body, args.timeout)
                        clean = send_one(endpoint, model, prompt, 1, args.timeout)
                        entry["clean_wall_s"] = clean["elapsed_s"]
                        entry["clean_prompt_tokens"] = clean["prompt_tokens"]
                        log(
                            "[profile] len=%-6d clean wall=%.3fs (profiled %.3fs)"
                            % (entry["target_tokens"], clean["elapsed_s"], entry["wall_s"])
                        )
            else:
                for index in range(args.warmup):
                    case = cases[index % len(cases)]
                    warm = send_one(endpoint, model, case["prompt"], args.max_completion_tokens, args.timeout)
                    log("[profile] warmup %d/%d %.2fs" % (index + 1, args.warmup, warm["elapsed_s"]))
                code = post_empty(base + "/start_profile", args.timeout)
                if code != 200:
                    raise SystemExit("/start_profile returned %s" % code)
                samples = []
                for index in range(args.requests):
                    case = cases[index % len(cases)]
                    result = send_one(endpoint, model, case["prompt"], args.max_completion_tokens, args.timeout)
                    result["id"] = case.get("id")
                    samples.append(result)
                    log("[profile] req %d/%d %.2fs" % (index + 1, args.requests, result["elapsed_s"]))
                stop_rc = post_empty(base + "/stop_profile", args.timeout)
                time.sleep(args.settle)
                entries.append(
                    {
                        "label": "all",
                        "target_tokens": None,
                        "prompt_tokens": samples[0]["prompt_tokens"] if samples else None,
                        "wall_s": round(sum(s["elapsed_s"] for s in samples) / max(len(samples), 1), 4),
                        "stop_rc": stop_rc,
                        "window_ids": window_ids(prof_dir, before),
                        "samples": samples,
                    }
                )

            total = len(snapshot(prof_dir))
            log("[profile] trace dirs=%d, total %.1f MB" % (total, dir_size_mb(prof_dir)))
            expected = int(cfg.get("tp_size") or 1) * max(len(entries), 1)
            if total < expected:
                log(
                    "[profile] WARNING expected about %d rank dirs, saw %d; check the service log"
                    % (expected, total)
                )
        finally:
            stop_service(proc)

    windows = {
        "config": cfg["name"],
        "fingerprint": fingerprint,
        "repo": cfg.get("repo"),
        "head": serve_config.git_head(cfg.get("repo")),
        "cp_balance": cfg.get("cp_balance"),
        "reduce_mode": cfg.get("reduce_mode"),
        "min_tokens": cfg.get("min_tokens"),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "lengths" if lengths else "single",
        "entries": entries,
    }
    (Path(prof_dir) / "windows.json").write_text(
        json.dumps(windows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = dict(windows)
    summary["profiler_dir"] = prof_dir
    summary["mean_elapsed_s"] = round(
        sum(entry["wall_s"] for entry in entries) / max(len(entries), 1), 4
    )
    out = HERE / ("prof_%s.json" % cfg["name"])
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log("[profile] %d windows -> %s" % (len(entries), out))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+", help="config name under configs/ (profiler enabled)")
    parser.add_argument("--kind", default="long", help="questions.json kind used to build prompts")
    parser.add_argument("--warmup", type=int, default=2, help="unprofiled requests (single-window mode only)")
    parser.add_argument("--requests", type=int, default=4, help="profiled requests (single-window mode only)")
    parser.add_argument("--max-completion-tokens", type=int, default=1)
    parser.add_argument("--endpoint", default="/v1/completions")
    parser.add_argument("--model", default="", help="served model name (default: questions.json hint)")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--ready-timeout", type=int, default=1800)
    parser.add_argument("--settle", type=int, default=45, help="seconds to let the trace flush after /stop_profile")
    parser.add_argument("--no-clean-pass", dest="clean_pass", action="store_false", help="skip the unprofiled second pass")
    args = parser.parse_args()

    results = []
    failed = []
    for name in args.configs:
        try:
            results.append(run_config(name, args))
        except Exception as exc:  # noqa: BLE001 - keep the other configs running
            log("[profile] FAILED %s: %s" % (name, exc))
            failed.append(str(name))
    log("[profile] ---- summary ----")
    for item in results:
        log(
            "[profile] %-24s windows=%s cp_balance=%s dir=%s"
            % (item["config"], len(item["entries"]), item["cp_balance"], item["profiler_dir"])
        )
    if failed:
        log("[profile] FAILED configs: %s" % ", ".join(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
