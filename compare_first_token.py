#!/usr/bin/env python3
"""Collect and compare the first output token of long Chinese prompts.

Generation requests follow the GLM-5.2 service template exactly::

    POST /v1/completions
    {"model": "glm-52", "prompt": "...", "max_completion_tokens": 50,
     "temperature": 0}

Only the ``prompt`` content changes between cases.  After generation we call the
vLLM ``/tokenize`` endpoint on ``choices[0].text`` to recover the first visible
token string; if that endpoint is unavailable the script falls back to the first
visible character.  This keeps the generation request clean while avoiding raw
special/format tokens in the comparison.

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
import re
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


def _visible_first_unit(text: str) -> str | None:
    """First visible word/character, used when /tokenize is unavailable."""
    text = text.lstrip()
    if not text:
        return None
    match = re.match(r"[A-Za-z0-9]+", text)
    if match:
        return match.group(0)
    return text[0]


def _byte_decoder() -> dict[str, int]:
    """Return the ``unicode char -> byte`` table of byte-level BPE tokenizers.

    GPT-2 style tokenizers (GLM-5.2 included) store every token as a string of
    printable unicode characters standing for the token's raw bytes, so byte
    ``0x82`` is written ``Ĥ`` (U+0124) and ``0x9E`` is written ``ŀ`` (U+0140).
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    table = {chr(byte): byte for byte in printable}
    for offset, byte in enumerate(b for b in range(256) if b not in printable):
        table[chr(256 + offset)] = byte
    return table


_BYTE_DECODER = _byte_decoder()


def _normalize_token_str(token: Any) -> Any:
    """Repair byte-level BPE token strings returned by /tokenize.

    ``/tokenize`` with ``return_token_strs=True`` returns raw vocabulary
    entries, not decoded text.  For byte-level BPE a Chinese token therefore
    comes back like ``å¦Ĥæŀľ`` (the byte view of ``E5 A6 82 E6 9E 9C``) instead
    of ``如果``.  Rebuild the underlying bytes and decode them as UTF-8; keep
    the original token when no round-trip yields valid UTF-8.
    """
    if not isinstance(token, str) or not token:
        return token
    candidates = []
    try:
        candidates.append(bytes(_BYTE_DECODER[char] for char in token))
    except KeyError:
        pass
    try:
        candidates.append(token.encode("latin-1"))
    except UnicodeEncodeError:
        pass
    for raw in candidates:
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if "\ufffd" not in decoded:
            return decoded
    return token


def _resolve_first_token(
    base_url: str,
    model: str,
    text: str,
    timeout: float,
) -> tuple[str | int | None, int | None, str]:
    """Return (first_token, first_token_id, source).

    ``choices[0].text`` may omit raw special/format tokens.  Tokenizing it again
    gives the first visible token of the generated answer, which is the signal
    we want to compare.
    """
    text = text or ""
    if not text.strip():
        return None, None, "empty"
    try:
        response = _post_json(
            base_url.rstrip("/") + "/tokenize",
            {
                "model": model,
                "prompt": text,
                "add_special_tokens": False,
                "return_token_strs": True,
            },
            timeout,
        )
        token_ids = response.get("tokens")
        token_strs = response.get("token_strs")
        token = None
        token_id = None
        if isinstance(token_strs, list) and token_strs:
            token = _normalize_token_str(token_strs[0])
        if isinstance(token_ids, list) and token_ids:
            token_id = int(token_ids[0])
            if token is None:
                token = token_id
        if token is not None:
            return token, token_id, "tokenize"
    except Exception:  # noqa: BLE001 - fall back to text prefix
        pass
    return _visible_first_unit(text), None, "text"


def _collect(args: argparse.Namespace) -> int:
    cases, model_hint = _load_cases(Path(args.questions))
    if args.kind:
        cases = [
            item for item in cases
            if str(item.get("kind") or "") == args.kind
        ]
    if args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("no cases selected")
    model = args.model or model_hint or "glm-52"
    endpoint = args.url.rstrip("/") + args.endpoint
    results: list[dict[str, Any]] = []
    print(
        f"[collect] url={args.url} endpoint={args.endpoint} model={model} "
        f"cases={len(cases)} kind={args.kind or 'all'} "
        f"max_completion_tokens={args.max_completion_tokens} "
        f"temperature={args.temperature}"
    )

    for index, item in enumerate(cases):
        question = str(item.get("question") or item.get("q") or "")
        kind = str(item.get("kind") or "?")
        prompt = _prompt_of(item)
        # Keep this payload exactly equal to the service template; only prompt
        # content changes per case.
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "max_completion_tokens": args.max_completion_tokens,
            "temperature": args.temperature,
        }
        started = time.time()
        try:
            response = _post_json(endpoint, payload, args.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 400 and "max_completion_tokens" in payload:
                # vLLM-compatible servers that only understand max_tokens.
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
        text = choice.get("text") or ""
        token, token_id, token_source = _resolve_first_token(
            args.url, model, text, args.timeout
        )
        elapsed = time.time() - started
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        results.append(
            {
                "index": index,
                "id": item.get("id", index + 1),
                "kind": kind,
                "question": question,
                "prompt_sha256": prompt_sha,
                "prompt_chars": len(prompt),
                "first_token": token,
                "first_token_id": token_id,
                "first_token_source": token_source,
                "text_head": text[:120],
                "elapsed_s": round(elapsed, 3),
            }
        )
        print(
            f"[{index:02d}][{kind}] chars={len(prompt):5d} token={token!r} "
            f"source={token_source} text={text[:24]!r} ({elapsed:.2f}s)"
        )

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
    by_kind: dict[str, list[int]] = {}
    mismatches: list[str] = []
    for item_a, item_b in zip(a, b):
        kind = str(item_a.get("kind") or "?")
        same_prompt = item_a.get("prompt_sha256") == item_b.get("prompt_sha256")
        same_token = item_a.get("first_token") == item_b.get("first_token")
        if same_token and item_a.get("first_token_id") is not None:
            same_token = item_a.get("first_token_id") == item_b.get("first_token_id")
        stats = by_kind.setdefault(kind, [0, 0])
        stats[1] += 1
        if same_prompt and same_token:
            passed += 1
            stats[0] += 1
            print(
                f"[{item_a.get('index', 0):02d}][{kind}] OK  token={item_a.get('first_token')!r}"
            )
        else:
            mismatches.append(
                f"case {item_a.get('index')}[{kind}]: prompt_same={same_prompt} "
                f"A={item_a.get('first_token')!r} B={item_b.get('first_token')!r}"
            )
            print(
                f"[{item_a.get('index', 0):02d}][{kind}] DIFF token {item_a.get('first_token')!r} != "
                f"{item_b.get('first_token')!r}, prompt_same={same_prompt}"
            )
    print(f"[compare] first-token match: {passed}/{len(a)}")
    for kind, (ok, total) in sorted(by_kind.items()):
        print(f"[compare] {kind}: {ok}/{total}")
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
    collect.add_argument(
        "--kind",
        default="",
        help="only run this kind from questions.json, e.g. short or long",
    )
    collect.add_argument(
        "--limit",
        type=int,
        default=0,
        help="only run the first N selected cases; 20 runs the short set first",
    )
    collect.add_argument("--temperature", type=float, default=0.0)
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
