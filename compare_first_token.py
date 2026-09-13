#!/usr/bin/env python3
"""Send the same long Chinese prompts to one vLLM endpoint and compare first tokens.

Usage
-----
Collect the first generated token from the service under test::

    python compare_first_token.py collect --url http://127.0.0.1:8034 --out /tmp/cp_on.json

Restart the service with the other VLLM_ASCEND_CP_BALANCE value, then::

    python compare_first_token.py collect --url http://127.0.0.1:8035 --out /tmp/cp_off.json

Compare the two result files::

    python compare_first_token.py compare /tmp/cp_on.json /tmp/cp_off.json

The script only uses the Python standard library.  It pads every question with
the same Chinese context so that cp_balance's MIN_TOKENS threshold is reached.
Sampling is greedy (temperature=0, max_tokens=1), so the first generated token
is a deterministic function of the service implementation; comparing it is the
cheapest end-to-end signal for cp_balance correctness.
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

CONTEXT = (
    "在大规模语言模型推理系统中，服务端需要在有限的加速卡和显存条件下"
    "同时处理许多用户的请求。调度器会把请求拆分为预填充和解码两个阶段："
    "预填充阶段一次处理整个提示词，计算量随提示长度增长；解码阶段每次只生成"
    "一个或少数几个词元，计算量小但通信频繁。为了降低显存占用，键值缓存通常"
    "按块管理，并可能分布在多张加速卡上。张量并行把线性层的参数切分到多张卡，"
    "数据并行则在不同副本之间复制参数并切分请求，流水线并行把网络层切成多个"
    "阶段。上下文并行会把一条很长的序列切成若干片段，每个片段由不同设备计算，"
    "注意力操作仍需要访问完整的键值缓存。连续切片会让靠前的设备看到较短的键值"
    "范围，靠后的设备看到较长的范围，因此计算负载并不均衡。折线式切分把序列的"
    "头部和尾部配对分给同一张卡，使每张卡处理的总词元数接近相等，同时让注意力"
    "计算量更均匀。无论采用哪种切分方式，都必须保证量化、矩阵乘、归一化和跨卡"
    "归约的计算结果与连续切片版本一致，否则第一个生成词元就可能出现明显偏差。"
)


def _repeat_context(min_chars: int) -> str:
    if min_chars <= 0:
        return ""
    repeats = (min_chars + len(CONTEXT) - 1) // len(CONTEXT)
    return (CONTEXT * repeats)[:min_chars]


def _load_questions(path: Path) -> list[str]:
    questions: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        questions.append(line)
    if not questions:
        raise SystemExit(f"no questions found in {path}")
    return questions


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
    """Extract the first sampled token string from chat/completions responses."""
    logprobs = choice.get("logprobs")
    if isinstance(logprobs, dict):
        content = logprobs.get("content")
        if isinstance(content, list) and content:
            first = content[0] or {}
            token = first.get("token")
            top = first.get("top_logprobs") or []
            return token, top
        tokens = logprobs.get("tokens")
        if isinstance(tokens, list) and tokens:
            token = tokens[0]
            top_raw = logprobs.get("top_logprobs") or []
            top: list[dict[str, Any]] = []
            if isinstance(top_raw, list) and top_raw:
                first_top = top_raw[0]
                if isinstance(first_top, dict):
                    top = [
                        {"token": key, "logprob": value}
                        for key, value in first_top.items()
                    ]
            return token, top

    message = choice.get("message") or {}
    if isinstance(message, dict) and message.get("content"):
        text = str(message["content"])
        return (text[:1] if text else None), []
    text = choice.get("text")
    if isinstance(text, str) and text:
        return (text[:1] if text else None), []
    return None, []


def _collect(args: argparse.Namespace) -> int:
    questions = _load_questions(Path(args.questions))
    context = _repeat_context(args.min_chars)
    endpoint = args.url.rstrip("/") + "/v1/chat/completions"
    results: list[dict[str, Any]] = []
    print(
        f"[collect] url={args.url} model={args.model} questions={len(questions)} "
        f"context_chars={len(context)} temperature={args.temperature}"
    )
    for index, question in enumerate(questions):
        prompt = f"{context}\n\n问题：{question}\n请直接给出答案。"
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": 1.0,
            "logprobs": True,
            "top_logprobs": args.top_logprobs,
            "stream": False,
        }
        started = time.time()
        try:
            response = _post_json(endpoint, payload, args.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"[{index:02d}] HTTP {exc.code}: {detail[:300]}", file=sys.stderr)
            return 2
        except Exception as exc:  # noqa: BLE001 - report the first failure clearly
            print(f"[{index:02d}] request failed: {exc!r}", file=sys.stderr)
            return 2
        choices = response.get("choices") or []
        if not choices:
            print(f"[{index:02d}] response has no choices: {response}", file=sys.stderr)
            return 2
        token, top = _first_token(choices[0])
        elapsed = time.time() - started
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        results.append(
            {
                "index": index,
                "question": question,
                "prompt_sha256": prompt_sha,
                "prompt_chars": len(prompt),
                "first_token": token,
                "top_logprobs": top,
                "elapsed_s": round(elapsed, 3),
            }
        )
        shown = repr(token)
        print(f"[{index:02d}] chars={len(prompt):5d} first_token={shown} ({elapsed:.2f}s)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "url": args.url,
                "model": args.model,
                "min_chars": args.min_chars,
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


def _load_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compare(args: argparse.Namespace) -> int:
    left = _load_results(Path(args.left))
    right = _load_results(Path(args.right))
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
            print(
                f"[{item_a['index']:02d}] OK  token={item_a.get('first_token')!r} "
                f"chars={item_a.get('prompt_chars')}"
            )
        else:
            mismatches.append(
                f"case {item_a['index']}: prompt_same={same_prompt} "
                f"A={item_a.get('first_token')!r} B={item_b.get('first_token')!r}"
            )
            print(
                f"[{item_a['index']:02d}] DIFF token {item_a.get('first_token')!r} != "
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

    collect = sub.add_parser("collect", help="send questions and save first tokens")
    collect.add_argument("--url", required=True, help="base URL, e.g. http://127.0.0.1:8034")
    collect.add_argument("--out", required=True, help="output JSON path")
    collect.add_argument("--model", default="glm")
    collect.add_argument(
        "--questions",
        default=str(Path(__file__).with_name("questions.txt")),
        help="one question per line",
    )
    collect.add_argument("--min-chars", type=int, default=8000, help="context padding length")
    collect.add_argument("--max-tokens", type=int, default=1)
    collect.add_argument("--temperature", type=float, default=0.0)
    collect.add_argument("--top-logprobs", type=int, default=5)
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
