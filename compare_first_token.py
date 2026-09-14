#!/usr/bin/env python3
"""Collect and compare the first generated token of long Chinese prompts.

The questions live in ``questions.json`` (article + question + full prompt).
Requests follow the service template used for GLM-5.2::

    POST /v1/completions
    {"model": "glm-52", "prompt": "...", "max_completion_tokens": 50,
     "temperature": 0}

The script additionally asks for one logprob so the first sampled token can be
extracted exactly instead of comparing characters.  It only uses the standard
library.

Usage::

    python compare_first_token.py collect --url http://127.0.0.1:8034 --out /tmp/cp_on.json
    # restart the service with the other VLLM_ASCEND_CP_BALANCE value
    python compare_first_token.py collect --url http://127.0.0.1:8035 --out /tmp/cp_off.json
    python compare_first_token.py compare /tmp/cp_on.json /tmp/cp_off.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_QUESTIONS = Path(__file__).with_name("questions.json")


def _load_cases(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Return (cases, model_hint) from questions.json."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "items" in payload:
        return list(payload["items"]), str(payload.get("model") or "")
    if isinstance(payload, list):
        return list(payload), ""
    raise SystemExit(f"unsupported questions file structure: {path}")


def _prompt_of(item: dict[str, Any]) -> str:
    prompt = item.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    raise SystemExit(f"questions.json item is missing a non-empty prompt: {item!r}")


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def _first_token(choice: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]]]:
    """Extract the first sampled token from a completions/chat response."""
    logprobs = choice.get("logprobs")
    if isinstance(logprobs, dict):
        content = logprobs.get("content")
        if isinstance(content, list) and content:
            first = content[0] or {}
            return first.get("token"), list(first.get("top_logprobs") or [])
        tokens = logprobs.get("tokens")
        if isinstance(tokens, list) and tokens:
            top_raw = logprobs.get("top_logprobs") or []
            top: list[dict[str, Any]] = []
            if isinstance(top_raw, list) and top_raw and isinstance(top_raw[0], dict):
                top = [{"token": key, "logprob": value} for key, value in top_raw[0].items()]
            return tokens[0], top

    text = choice.get("text")
    if isinstance(text, str) and text:
        return text[:1], []
    message = choice.get("message") or {}
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content:
            return content[:1], []
    return None, []


def _collect(args: argparse.Namespace) -> int:
    cases, model_hint = _load_cases(Path(args.questions))
    model = args.model or model_hint or "glm-52"
    endpoint = args.url.rstrip("/") + args.endpoint
    results: list[dict[str, Any]] = []
    print(
        f"[collect] url={args.url} endpoint={args.endpoint} model={model} "
        f"cases={len(cases)} max_completion_tokens={args.max_completion_tokens} "
        f"temperature={args.temperature}"
    )

    for index, item in enumerate(cases):
        question = str(item.get("question") or item.get("q") or "")
        prompt = _prompt_of(item)
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "max_completion_tokens": args.max_completion_tokens,
            "temperature": args.temperature,
            "logprobs": args.top_logprobs,
        }
        started = time.time()
        try:
            response = _post_json(endpoint, payload, args.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 400 and "max_completion_tokens" in payload:
                # Older/vLLM-compatible servers may still expose only max_tokens.
                payload.pop("max_completion_tokens")
                payload["max_tokens"] = args.max_completion_tokens
                try:
                    response = _post_json(endpoint, payload, args.timeout)
                except Exception as retry_exc:  # noqa: BLE001
                    print(f"[{index:02d}] retry with max_tokens failed: {retry_exc!r}", file=sys.stderr)
                    print(f"[{index:02d}] first failure body: {detail[:300]}", file=sys.stderr)
                    return 2
            else:
                print(f"[{index:02d}] HTTP {exc.code}: {detail[:300]}", file=sys.stderr)
                return 2
        except Exception as exc:  # noqa: BLE001 - report the first failure clearly
            print(f"[{index:02d}] request failed: {exc!r}", file=sys.stderr)
            return 2

        choices = response.get("choices") or []
        if not choices:
            print(f"[{index:02d}] response has no choices", file=sys.stderr)
            return 2
        choice = choices[0]
        token, top = _first_token(choice)
        text = choice.get("text") or ""
        elapsed = time.time() - started
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        results.append(
            {
                "index": index,
                "id": item.get("id", index + 1),
                "question": question,
                "prompt_sha256": prompt_sha,
                "prompt_chars": len(prompt),
                "first_token": token,
                "text_head": text[:80],
                "top_logprobs": top,
                "elapsed_s": round(elapsed, 3),
            }
        )
        print(f"[{index:02d}] chars={len(prompt):5d} first_token={token!r} text={text[:24]!r} ({elapsed:.2f}s)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "url": args.url,
                "endpoint": args.endpoint,
                "model": model,
                "max_completion_tokens": args.max_completion_tokens,
                "temperature": args.temperature,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[collect] saved {len(results)} results to {out}")
    return 0


def _compare(args: argparse.Namespace) -> int:
    left = json.loads(Path(args.left).read_text(encoding="utf-8"))
    right = json.loads(Path(args.right).read_text(encoding="utf-8"))
    a = left.get("results") or []
    b = right.get("results") or []
    if len(a) != len(b):
        print(f"[compare] case count differs: {len(a)} vs {len(b)}")
        return 1

    passed = 0
    mismatches: list[str] = []
    for item_a, item_b in zip(a, b):
        same_prompt = item_a.get("prompt_sha256") == item_b.get("prompt_sha256")
        same_token = item_a.get("first_token") == item_b.get("first_token")
        if same_prompt and same_token:
            passed += 1
            print(f"[{item_a.get('index', 0):02d}] OK  token={item_a.get('first_token')!r}")
        else:
            mismatches.append(
                f"case {item_a.get('index')}: prompt_same={same_prompt} "
                f"A={item_a.get('first_token')!r} B={item_b.get('first_token')!r}"
            )
            print(
                f"[{item_a.get('index', 0):02d}] DIFF token {item_a.get('first_token')!r} != "
                f"{item_b.get('first_token')!r}, prompt_same={same_prompt}"
            )
    print(f"[compare] first-token match: {passed}/{len(a)}")
    if mismatches:
        print("[compare] RESULT: FAIL")
        for line in mismatches:
            print("  " + line)
        return 1
    print("[compare] RESULT: PASS")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="send prompts and save the first token")
    collect.add_argument("--url", required=True, help="base URL, e.g. http://127.0.0.1:8034")
    collect.add_argument("--out", required=True, help="output JSON path")
    collect.add_argument("--questions", default=str(DEFAULT_QUESTIONS), help="questions.json")
    collect.add_argument("--model", default="", help="served model name (default: questions.json hint)")
    collect.add_argument("--endpoint", default="/v1/completions")
    collect.add_argument("--max-completion-tokens", type=int, default=50)
    collect.add_argument("--temperature", type=float, default=0.0)
    collect.add_argument("--top-logprobs", type=int, default=1)
    collect.add_argument("--timeout", type=float, default=600.0)
    collect.set_defaults(func=_collect)

    compare = sub.add_parser("compare", help="compare two collect JSON files")
    compare.add_argument("left")
    compare.add_argument("right")
    compare.set_defaults(func=_compare)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
