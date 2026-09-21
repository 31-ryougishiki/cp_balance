#!/usr/bin/env python3
"""Service-level traffic driver: burst load, /metrics snapshots, checks.

    loadgen.py run --url URL --plan configs/svc_load_mixed.json --out results.json
                   [--model M] [--concurrency N] [--rounds N] [--duration S] [--timeout S]
    loadgen.py check --in results.json [--require-identical] [--max-failed N]
                     [--max-preemptions N] [--max-kv-usage F] [--out report.txt]
    loadgen.py compare --left cp1.json --right cp0.json [--out report.txt]
    loadgen.py metrics --url URL --out metrics.json

One plan file describes one burst: every request names a prompt from the shared
questions.json as "<kind>:<index>" (kind = short|long) plus its own sampling
parameters, so the traffic mix lives in configs/ instead of in the scripts.

run walks the plan round by round; inside a round all requests are issued at the
same time (that is what puts several sequences into one prefill batch) and the
round is a barrier, so a soak run keeps burst semantics.  Metrics are sampled
before and after the traffic, and the engine counters are part of the evidence.

check gates one run: every request must succeed, requests sharing prompt and
sampling parameters must return the same text, and the engine counters must stay
clean (corrupted requests, preemptions, KV pressure).
compare gates cp0 against cp1 on the same plan (text equality) and reports the
latency difference without gating it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accuracy.compare_first_token import _load_cases, _prompt_of  # noqa: E402

NEWLINE = chr(10)

for _key in ("no_proxy", "NO_PROXY"):
    _parts = [part for part in os.environ.get(_key, "").split(",") if part]
    for _host in ("127.0.0.1", "localhost", "::1"):
        if _host not in _parts:
            _parts.append(_host)
    os.environ[_key] = ",".join(_parts)

COUNTERS = (
    "vllm:request_success",
    "vllm:num_preemptions",
    "vllm:corrupted_requests",
    "vllm:prompt_tokens",
    "vllm:generation_tokens",
)
GAUGES = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
)
SUMS = (
    "vllm:time_to_first_token_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:e2e_request_latency_seconds",
    "vllm:request_queue_time_seconds",
)


def _prompts(questions: Path) -> tuple[dict[str, list[str]], str]:
    cases, hint = _load_cases(questions)
    by_kind: dict[str, list[str]] = {}
    for item in cases:
        by_kind.setdefault(str(item.get("kind") or "any"), []).append(_prompt_of(item))
    return by_kind, hint


def _post(base: str, path: str, payload: dict, timeout: float):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(base.rstrip("/") + path, data=data,
                                     headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(request, timeout=timeout)


def _stream_text(response) -> tuple[str, float | None]:
    chunks: list[str] = []
    started = time.time()
    ttft = None
    for raw in response:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if body == "[DONE]":
            break
        piece = json.loads(body)
        delta = ((piece.get("choices") or [{}])[0].get("text")) or ""
        if delta and ttft is None:
            ttft = time.time() - started
        chunks.append(delta)
    return "".join(chunks), ttft


def _one_request(base: str, model: str, prompt: str, params: dict, timeout: float) -> dict:
    payload = {"model": model, "prompt": prompt, "temperature": 0.0}
    payload.update(params)
    started = time.time()
    try:
        with _post(base, "/v1/completions", payload, timeout) as response:
            if params.get("stream"):
                text, ttft = _stream_text(response)
                return {"ok": True, "http_status": response.status, "text": text,
                        "generation_tokens": None, "ttft_s": ttft,
                        "latency_s": time.time() - started}
            body = json.loads(response.read().decode("utf-8"))
            text = (body.get("choices") or [{}])[0].get("text") or ""
            usage = body.get("usage") or {}
            return {"ok": True, "http_status": response.status, "text": text,
                    "generation_tokens": usage.get("completion_tokens"), "ttft_s": None,
                    "latency_s": time.time() - started}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        return {"ok": False, "http_status": exc.code, "text": "", "generation_tokens": None,
                "ttft_s": None, "latency_s": time.time() - started,
                "error": "HTTP %d %s" % (exc.code, detail)}
    except Exception as exc:  # noqa: BLE001 - a client-side failure is a result too
        return {"ok": False, "http_status": None, "text": "", "generation_tokens": None,
                "ttft_s": None, "latency_s": time.time() - started,
                "error": "%s: %s" % (type(exc).__name__, exc)}


def _metrics(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=timeout) as response:
        body = response.read().decode("utf-8", "replace")
    out: dict[str, dict] = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        name = name.split("{")[0].strip()
        try:
            number = float(value)
        except ValueError:
            continue
        slot = out.setdefault(name, {"sum": 0.0, "max": float("-inf"), "series": 0})
        slot["sum"] += number
        slot["max"] = max(slot["max"], number)
        slot["series"] += 1
    return out


def _metrics_safe(url: str) -> dict:
    try:
        return _metrics(url)
    except Exception as exc:  # noqa: BLE001 - carry on, but say so
        print("[loadgen] /metrics unavailable: %s: %s" % (type(exc).__name__, exc))
        return {}


def _lookup(node: dict, name: str) -> dict:
    # prometheus_client appends _total to counters; accept both spellings.
    return node.get(name) or node.get(name + "_total") or {}


def _delta(before: dict, after: dict, name: str) -> float:
    return round(_lookup(after, name).get("sum", 0.0) - _lookup(before, name).get("sum", 0.0), 4)


def _gauge(after: dict, name: str) -> float:
    return round(_lookup(after, name).get("max", 0.0), 4)


def _average(after: dict, before: dict, name: str) -> float | None:
    calls = _lookup(after, name + "_count").get("sum", 0.0) - _lookup(before, name + "_count").get("sum", 0.0)
    if calls <= 0:
        return None
    total = _lookup(after, name + "_sum").get("sum", 0.0) - _lookup(before, name + "_sum").get("sum", 0.0)
    return round(total / calls, 4)


def _group_key(spec: dict) -> str:
    return "%s|%s" % (spec["prompt_ref"], json.dumps(spec["params"], sort_keys=True))


def _specs(plan: dict, prompts: dict, questions: str) -> list[dict]:
    specs = []
    for spec in plan.get("requests") or []:
        kind, _, index = str(spec["prompt"]).partition(":")
        pool = prompts.get(kind) or []
        if not pool:
            raise SystemExit("plan prompt %r: %s has no kind %r" % (spec["prompt"], questions, kind))
        params = {key: value for key, value in spec.items() if key not in ("id", "prompt")}
        params.setdefault("max_tokens", 16)
        specs.append({"id": spec["id"], "prompt_ref": spec["prompt"],
                      "prompt": pool[int(index or 0) % len(pool)], "params": params})
    if not specs:
        raise SystemExit("plan has no requests")
    return specs


def _run(args: argparse.Namespace) -> int:
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    prompts, hint = _prompts(Path(args.questions))
    model = args.model or hint
    specs = _specs(plan, prompts, args.questions)
    rounds = args.rounds or int(plan.get("rounds") or 1)
    workers = args.concurrency or int(plan.get("concurrency") or len(specs))
    deadline = time.time() + args.duration if args.duration else None

    before = _metrics_safe(args.url)
    started = time.time()
    results: list[dict] = []
    round_index = 0
    while True:
        if deadline is not None:
            if round_index >= 1 and time.time() >= deadline:
                break
        elif round_index >= rounds:
            break
        round_started = time.time()
        with ThreadPoolExecutor(max_workers=min(workers, len(specs))) as pool:
            futures = [pool.submit(_one_request, args.url, model, spec["prompt"],
                                   spec["params"], args.timeout) for spec in specs]
            for spec, future in zip(specs, futures):
                row = future.result()
                row.update({"id": spec["id"], "round": round_index, "prompt_ref": spec["prompt_ref"],
                            "params": spec["params"], "group": _group_key(spec)})
                row["text_sha1"] = hashlib.sha1(row["text"].encode("utf-8")).hexdigest()[:16]
                results.append(row)
        done = [row for row in results if row["round"] == round_index]
        print("[loadgen] round %d in %.1fs (ok=%d failed=%d)"
              % (round_index, time.time() - round_started,
                 sum(1 for row in done if row["ok"]), sum(1 for row in done if not row["ok"])))
        round_index += 1
    after = _metrics_safe(args.url)

    failures = [row for row in results if not row["ok"]]
    latencies = [row["latency_s"] for row in results if row["ok"]]
    summary = {
        "requests": len(results),
        "failed": len(failures),
        "rounds": round_index,
        "wall_s": round(time.time() - started, 2),
        "latency_s": {"min": round(min(latencies), 3), "p50": round(statistics.median(latencies), 3),
                      "max": round(max(latencies), 3)} if latencies else None,
        "metrics": {
            "counters": {name: _delta(before, after, name) for name in COUNTERS},
            "gauges_max": {name: _gauge(after, name) for name in GAUGES},
            "averages": {name: _average(after, before, name) for name in SUMS},
        },
        "first_error": failures[0].get("error") if failures else None,
        "metrics_available": bool(before and after),
    }
    payload = {"plan": str(plan_path), "url": args.url, "model": model, "started_at": time.time(),
               "summary": summary, "requests": results,
               "metrics_before": before, "metrics_after": after}
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[loadgen] %d requests, %d failed, wall=%.1fs -> %s"
          % (summary["requests"], summary["failed"], summary["wall_s"], args.out))
    print("[loadgen] counters %s" % json.dumps(summary["metrics"]["counters"]))
    print("[loadgen] gauges_max %s" % json.dumps(summary["metrics"]["gauges_max"]))
    print("[loadgen] averages %s" % json.dumps(summary["metrics"]["averages"]))
    if summary["first_error"]:
        print("[loadgen] first error: %s" % summary["first_error"])
    return 0 if not failures else 1


def _check(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.infile).read_text(encoding="utf-8"))
    summary = data["summary"]
    rows = data["requests"]
    failed = int(summary["failed"])
    prints: list[str] = []
    ok = True

    good = failed <= args.max_failed
    prints.append("[check] failed requests: %d (allowed %d) %s"
                  % (failed, args.max_failed, "ok" if good else "FAIL"))
    ok = ok and good
    if failed:
        prints.append("[check] first error: %s" % summary.get("first_error"))

    if args.require_identical:
        groups: dict[str, set] = {}
        for row in rows:
            if row["ok"]:
                groups.setdefault(row["group"], set()).add(row["text_sha1"])
        bad = {key: sorted(value) for key, value in groups.items() if len(value) > 1}
        prints.append("[check] identical replies within a prompt group: %s%s"
                      % ("ok" if not bad else "FAIL", "" if not bad else " %s" % list(bad)[:3]))
        ok = ok and not bad

    metrics_ok = bool(summary.get("metrics_available"))
    if not metrics_ok:
        prints.append("[check] /metrics was not readable: counter gates not evaluated")
    counters = summary["metrics"]["counters"]
    corrupted = counters.get("vllm:corrupted_requests", 0)
    preemptions = counters.get("vllm:num_preemptions", 0)
    prints.append("[check] corrupted requests: %g, preemptions: %g (allowed %d)"
                  % (corrupted, preemptions, args.max_preemptions))
    ok = ok and (not metrics_ok or (corrupted == 0 and preemptions <= args.max_preemptions))

    kv = summary["metrics"]["gauges_max"].get("vllm:kv_cache_usage_perc", 0)
    waiting = summary["metrics"]["gauges_max"].get("vllm:num_requests_waiting", 0)
    prints.append("[check] kv_cache_usage_perc max: %g (allowed %g), waiting max: %g"
                  % (kv, args.max_kv_usage, waiting))
    ok = ok and (not metrics_ok or kv <= args.max_kv_usage)

    success = counters.get("vllm:request_success", 0)
    prints.append("[check] engine request_success delta: %g for %d client requests"
                  % (success, summary["requests"]))
    if metrics_ok and success and success < summary["requests"] - failed:
        prints.append("[check] engine counted fewer successes than the client sent: FAIL")
        ok = False

    if not ok:
        verdict = "FAIL"
    elif not metrics_ok:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PASS"
    text = NEWLINE.join(prints) + NEWLINE + "[check] RESULT: %s" % verdict
    sys.stdout.write(text + NEWLINE)
    if args.out:
        Path(args.out).write_text(text + NEWLINE, encoding="utf-8")
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 77}[verdict]


def _compare(args: argparse.Namespace) -> int:
    left = json.loads(Path(args.left).read_text(encoding="utf-8"))
    right = json.loads(Path(args.right).read_text(encoding="utf-8"))
    left_rows = {(row["id"], row["round"]): row for row in left["requests"]}
    right_rows = {(row["id"], row["round"]): row for row in right["requests"]}
    keys = sorted(set(left_rows) & set(right_rows))
    missing = sorted(set(left_rows) ^ set(right_rows))
    if not keys:
        print("[compare] RESULT: INCONCLUSIVE (no shared request id+round)")
        return 77
    mismatch = [{"key": key, "left": left_rows[key]["text"][:60], "right": right_rows[key]["text"][:60]}
                for key in keys if left_rows[key]["text_sha1"] != right_rows[key]["text_sha1"]]
    prints = [
        "[compare] left=%s (%d req, %d failed) right=%s (%d req, %d failed)"
        % (left["url"], len(left["requests"]), left["summary"]["failed"],
           right["url"], len(right["requests"]), right["summary"]["failed"]),
        "[compare] shared requests: %d, one-sided: %s" % (len(keys), missing[:6] or "none"),
        "[compare] text match: %d/%d%s"
        % (len(keys) - len(mismatch), len(keys), "" if not mismatch else " %s" % mismatch[:2]),
        "[compare] wall: left=%.1fs right=%.1fs -> left/right=%.3f"
        % (left["summary"]["wall_s"], right["summary"]["wall_s"],
           (left["summary"]["wall_s"] / right["summary"]["wall_s"]) if right["summary"]["wall_s"] else 0.0),
        "[compare] project drift is 5-13 percent: a difference below that is not a result",
    ]
    for side, data in (("left", left), ("right", right)):
        averages = data["summary"]["metrics"]["averages"]
        prints.append("[compare] %s averages (s): ttft=%s itl=%s e2e=%s"
                      % (side, averages.get("vllm:time_to_first_token_seconds"),
                         averages.get("vllm:inter_token_latency_seconds"),
                         averages.get("vllm:e2e_request_latency_seconds")))
    ok = not mismatch and not missing
    prints.append("[compare] RESULT: %s" % ("PASS" if ok else "FAIL"))
    text = NEWLINE.join(prints)
    sys.stdout.write(text + NEWLINE)
    if args.out:
        Path(args.out).write_text(text + NEWLINE, encoding="utf-8")
    return 0 if ok else 1


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run")
    run.add_argument("--url", required=True)
    run.add_argument("--plan", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--model", default="")
    run.add_argument("--questions", default=str(ROOT / "questions.json"))
    run.add_argument("--concurrency", type=int, default=0)
    run.add_argument("--rounds", type=int, default=0)
    run.add_argument("--duration", type=float, default=0.0)
    run.add_argument("--timeout", type=float, default=600.0)

    check = sub.add_parser("check")
    check.add_argument("--in", dest="infile", required=True)
    check.add_argument("--out", default="")
    check.add_argument("--max-failed", type=int, default=0)
    check.add_argument("--require-identical", action="store_true")
    check.add_argument("--max-preemptions", type=int, default=0)
    check.add_argument("--max-kv-usage", type=float, default=0.99)

    compare = sub.add_parser("compare")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--out", default="")

    metrics = sub.add_parser("metrics")
    metrics.add_argument("--url", required=True)
    metrics.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "run":
        return _run(args)
    if args.cmd == "check":
        return _check(args)
    if args.cmd == "compare":
        return _compare(args)
    Path(args.out).write_text(json.dumps(_metrics(args.url), indent=2), encoding="utf-8")
    print("[loadgen] metrics -> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
