#!/usr/bin/env python3
"""Prove which CP branch a request takes: ZIGZAG (cp_balance) or CONTINUOUS.

Start the server with VLLM_ASCEND_CP_BALANCE_DEBUG=1.  The SFA metadata
builder then logs one "[CP_BALANCE][branch]" line per rank and batch for BOTH
branches, including the gate that refused zigzag, so a branch is proven by a
log line instead of by the absence of one.

    python check_branch.py --url http://127.0.0.1:8034 --log cp_on.log

It sends one short prompt (must stay CONTINUOUS) and one long prompt (must
become ZIGZAG) and prints the log evidence of both windows.

PASS means: the short window has no zigzag evidence but at least one
CONTINUOUS line, and the long window has at least one zigzag evidence line
(`branch=ZIGZAG`, or the `[CP_BALANCE][plan]` line the zigzag branch prints;
the list is `harness.json` verify.diagnose.zigzag_evidence).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_QUESTIONS = Path(__file__).resolve().parent.parent / "questions.json"
BRANCH_TAG = "[CP_BALANCE][branch]"
PLAN_TAG = "[CP_BALANCE][plan]"
# 证明“这批请求走了 zigzag”的日志证据；可在 harness.json verify.diagnose.zigzag_evidence 里改。
# 资格门拒绝时打 [branch] 行，成功时打 [plan] 行，所以两种都算证据。
DEFAULT_EVIDENCE = ("branch=ZIGZAG", PLAN_TAG)
DISABLED_HINTS = (
    "Disabling DSA-CP",
    "does not support sequence-parallel MoE",
    "FlashComm1 is enabled",
    "FlashComm1 is disabled",
)


def zigzag_evidence() -> tuple:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import serve_config  # noqa: PLC0415 - harness helper next to the drivers

        found = (serve_config.harness().get("verify") or {}).get("diagnose", {}).get("zigzag_evidence") or []
        return tuple(str(item) for item in found) or DEFAULT_EVIDENCE
    except Exception:  # noqa: BLE001 - a bare checkout must still run this check
        return DEFAULT_EVIDENCE


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


def _window(args: argparse.Namespace, kind: str, prompt: str, model: str, evidence: tuple) -> dict[str, Any]:
    log = Path(args.log)
    tokens = _count_tokens(args.url, model, prompt, args.timeout)
    offset = log.stat().st_size
    _generate(args.url, model, prompt, args.timeout)
    lines = _branch_lines(log, offset)
    plan_lines = _tail_lines(log, offset, PLAN_TAG)
    hits = [line for line in lines + plan_lines if any(item in line for item in evidence)]
    continuous = [line for line in lines if "branch=CONTINUOUS" in line]
    print(f"[check] {kind}: tokens={tokens} branch_lines={len(lines)} plan_lines={len(plan_lines)} "
          f"zigzag={len(hits)} continuous={len(continuous)}")
    for line in (hits or continuous)[:1]:
        print(f"[check] {kind} sample: {line.strip()}")
    return {
        "tokens": tokens,
        "zigzag": len(hits),
        "continuous": len(continuous),
        "total": len(lines) + len(plan_lines),
    }


def _whole_log_lines(log: Path, needle: str) -> list[str]:
    text = log.read_text(encoding="utf-8", errors="replace")
    return [line for line in text.splitlines() if needle in line]


def _tail_lines(log: Path, offset: int, needle: str) -> list[str]:
    with log.open("rb") as handle:
        handle.seek(offset)
        data = handle.read()
    return [line for line in data.decode("utf-8", errors="replace").splitlines() if needle in line]


def _diagnose_disabled(log: Path, evidence: tuple) -> None:
    """No zigzag evidence anywhere: say why, instead of blaming DEBUG only."""
    text = log.read_text(encoding="utf-8", errors="replace")
    print(f"[check] diagnose: evidence={list(evidence)} 在本窗口没有出现，全日志线索：")
    for line in text.splitlines():
        if any(hint in line for hint in DISABLED_HINTS):
            print(f"[check] diagnose: {line.strip()[:200]}")
    reasons: list[str] = []
    for line in text.splitlines():
        if BRANCH_TAG in line and "reason=" in line and line not in reasons:
            reasons.append(line)
    for line in reasons[-3:]:
        print(f"[check] diagnose: {line.strip()[:200]}")
    if not reasons:
        print("[check] diagnose: 全日志没有 [CP_BALANCE][branch] 行 -> "
              "要么 VLLM_ASCEND_CP_BALANCE_DEBUG=0，要么 DSA-CP 整条路径没启用（看上面的 Disabling DSA-CP/FlashComm1 行）")


def _run(args: argparse.Namespace) -> int:
    log = Path(args.log)
    if not log.is_file():
        raise SystemExit(f"server log not found: {log}")
    cases, model_hint = _load_cases(Path(args.questions))
    model = args.model or model_hint or "glm-52"
    evidence = zigzag_evidence()
    print(f"[check] url={args.url} log={log} model={model} "
          f"min_tokens={args.min_tokens} tag={BRANCH_TAG} evidence={list(evidence)}")

    results = {kind: _window(args, kind, _pick(cases, kind), model, evidence) for kind in ("short", "long")}
    short, long = results["short"], results["long"]
    short_tokens, long_tokens = short["tokens"], long["tokens"]
    problems: list[str] = []

    # 源码里 CONTINUOUS 行是 logger.info_once（每进程一次），窗口内可能没有：
    # 短 prompt 的判据是「窗口内没有 zigzag 证据」，CONTINUOUS 只要求全日志出现过。
    if short["zigzag"]:
        problems.append("short prompt entered ZIGZAG")
    continuous_any = _whole_log_lines(log, "branch=CONTINUOUS")
    if short["continuous"] == 0 and not continuous_any:
        problems.append("short window has no branch line (DEBUG off, or the log lags: try PYTHONUNBUFFERED=1)")
    elif short["continuous"] == 0:
        print(f"[check] note: short window has no new CONTINUOUS line, whole log has {len(continuous_any)} "
              "(info_once logs it once per process)")
    if short_tokens >= args.min_tokens:
        print(f"[check] warning: short prompt has {short_tokens} tokens >= min_tokens, ZIGZAG is expected")

    if long["zigzag"] == 0:
        _diagnose_disabled(log, evidence)
        if long_tokens < args.min_tokens:
            # prompt 阶梯坏了（长 prompt 不够长），不是功能坏了：判为不可结论
            print(f"[check] INCONCLUSIVE: long prompt has {long_tokens} tokens < min_tokens={args.min_tokens}; "
                  "zigzag 被设计性拒绝（换更长的 prompt 或调小 min_tokens 再跑）")
            print("[check] RESULT: INCONCLUSIVE")
            return 77
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
