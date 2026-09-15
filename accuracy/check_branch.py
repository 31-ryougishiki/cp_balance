#!/usr/bin/env python3
"""Prove which CP branch a request takes: ZIGZAG (cp_balance) or CONTINUOUS.

Start the server with VLLM_ASCEND_CP_BALANCE_DEBUG=1.  The SFA metadata
builder then logs one "[CP_BALANCE][branch]" line per rank and batch for BOTH
branches, including the gate that refused zigzag, so a branch is proven by a
log line instead of by the absence of one.

    python check_branch.py --url http://127.0.0.1:8034 --log cp_on.log

It sends one short prompt (must stay CONTINUOUS) and one long prompt (must
become ZIGZAG) and prints the log evidence of both windows.

PASS means: the short window has no ZIGZAG line but at least one CONTINUOUS
line, and the long window has at least one ZIGZAG line.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_QUESTIONS = Path(__file__).resolve().parent.parent / "questions.json"
BRANCH_TAG = "[CP_BALANCE][branch]"


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _load_cases(path: Path) -> tuple[list[dict[str, Any]], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "items" in payload:
        return list(payload["items"]), str(payload.get("model") or "")
    if isinstance(payload, list):
        return list(payload), ""
    raise SystemExit(f"unsupported questions file structure: {path}")


def _pick(cases: list[dict[str, Any]], kind: str) -> str:
    for item in cases:
        if str(item.get("kind") or "") == kind and str(item.get("prompt") or "").strip():
            return str(item["prompt"])
    raise SystemExit(f"questions.json has no {kind} case with a prompt")


def _count_tokens(url: str, model: str, prompt: str, timeout: float) -> int:
    response = _post_json(
        url.rstrip("/") + "/tokenize",
        {"model": model, "prompt": prompt, "add_special_tokens": True},
        timeout,
    )
    tokens = response.get("tokens")
    if isinstance(tokens, list):
        return len(tokens)
    return int(response.get("count") or 0)


def _generate(url: str, model: str, prompt: str, timeout: float) -> None:
    """Send the exact service template; the answer itself is irrelevant."""
    _post_json(
        url.rstrip("/") + "/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "max_completion_tokens": 1,
            "temperature": 0,
        },
        timeout,
    )


def _branch_lines(log: Path, offset: int) -> list[str]:
    """Branch log lines appended after byte offset."""
    with log.open("rb") as handle:
        handle.seek(offset)
        data = handle.read()
    text = data.decode("utf-8", errors="replace")
    return [line for line in text.splitlines() if BRANCH_TAG in line]


def _window(args: argparse.Namespace, kind: str, prompt: str, model: str) -> dict[str, Any]:
    log = Path(args.log)
    tokens = _count_tokens(args.url, model, prompt, args.timeout)
    offset = log.stat().st_size
    _generate(args.url, model, prompt, args.timeout)
    lines = _branch_lines(log, offset)
    zigzag = [line for line in lines if "branch=ZIGZAG" in line]
    continuous = [line for line in lines if "branch=CONTINUOUS" in line]
    print(f"[check] {kind}: tokens={tokens} branch_lines={len(lines)} "
          f"zigzag={len(zigzag)} continuous={len(continuous)}")
    for line in (zigzag or continuous)[:1]:
        print(f"[check] {kind} sample: {line.strip()}")
    return {
        "tokens": tokens,
        "zigzag": len(zigzag),
        "continuous": len(continuous),
        "total": len(lines),
    }


def _run(args: argparse.Namespace) -> int:
    log = Path(args.log)
    if not log.is_file():
        raise SystemExit(f"server log not found: {log}")
    cases, model_hint = _load_cases(Path(args.questions))
    model = args.model or model_hint or "glm-52"
    print(f"[check] url={args.url} log={log} model={model} "
          f"min_tokens={args.min_tokens} tag={BRANCH_TAG}")

    results = {kind: _window(args, kind, _pick(cases, kind), model) for kind in ("short", "long")}
    short, long = results["short"], results["long"]
    short_tokens, long_tokens = short["tokens"], long["tokens"]
    problems: list[str] = []

    if short["total"] == 0:
        problems.append("short window has no branch line (DEBUG off, or the log lags: try PYTHONUNBUFFERED=1)")
    if short["zigzag"]:
        problems.append("short prompt entered ZIGZAG")
    if short["total"] > 0 and short["continuous"] == 0:
        problems.append("short window has no CONTINUOUS line")
    if short_tokens >= args.min_tokens:
        print(f"[check] warning: short prompt has {short_tokens} tokens >= min_tokens, ZIGZAG is expected")

    if long["zigzag"] == 0:
        problems.append("long prompt never entered ZIGZAG")
    if long_tokens < args.min_tokens:
        print(f"[check] warning: long prompt has {long_tokens} tokens < min_tokens, zigzag is refused by design")

    print(f"[check] short: tokens={short_tokens} (min_tokens={args.min_tokens}) -> expect CONTINUOUS")
    print(f"[check] long:  tokens={long_tokens} (min_tokens={args.min_tokens}) -> expect ZIGZAG")
    for problem in problems:
        print(f"[check] FAIL: {problem}")
    print(f"[check] RESULT: {'FAIL' if problems else 'PASS'}")
    return 1 if problems else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="base URL, e.g. http://127.0.0.1:8034")
    parser.add_argument("--log", required=True, help="server stdout/stderr log file")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--model", default="", help="served model name (default: questions.json hint)")
    parser.add_argument("--min-tokens", type=int, default=2048, help="VLLM_ASCEND_CP_BALANCE_MIN_TOKENS of the run")
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        return _run(args)
    except urllib.error.URLError as exc:
        raise SystemExit(f"request failed: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
