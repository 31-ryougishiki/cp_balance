#!/usr/bin/env python3
"""CPU self-test for the cp_balance token plan.

Run it in the remote vLLM environment (PYTHONPATH includes vllm-ascend).  It
does not touch the NPU; it only checks, for many random query-length sets, that

* every natural token position is owned by exactly one rank;
* the rank-concatenating all-gather order equals zigzag_gather_index;
* inv_gather_index really restores natural order;
* every rank owns the same number of local tokens.

Usage::

    python selftest_plan.py --cp-size 16 --cases 2000
"""

from __future__ import annotations

import argparse
import random
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cp-size", type=int, default=16)
    parser.add_argument("--cases", type=int, default=2000)
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument("--max-reqs", type=int, default=4)
    args = parser.parse_args()

    from vllm_ascend.layers.cp_zigzag import build_zigzag_plan

    rng = random.Random(0)
    p = args.cp_size
    checked = 0
    for case in range(args.cases):
        num_reqs = rng.randint(1, args.max_reqs)
        query_lens = [rng.randint(args.min_len, args.max_len) for _ in range(num_reqs)]
        prefix_lens = [rng.randint(0, 128) for _ in range(num_reqs)]
        total = sum(query_lens)
        num_tokens_pad = (total + p - 1) // p * p

        per_rank = []
        for rank in range(p):
            plan = build_zigzag_plan(
                query_lens,
                prefix_lens,
                p,
                rank,
                num_tokens_pad,
                total,
            )
            index = list(plan.zigzag_index)
            if len(index) != num_tokens_pad // p:
                raise AssertionError(
                    f"case {case}: rank {rank} has {len(index)} local tokens, "
                    f"expected {num_tokens_pad // p}"
                )
            per_rank.append(index)

            gather = list(plan.zigzag_gather_index)
            inv = list(plan.inv_gather_index)
            if len(gather) != num_tokens_pad or len(inv) != num_tokens_pad:
                raise AssertionError(f"case {case}: gather/inv length mismatch")
            if sorted(gather) != list(range(num_tokens_pad)):
                raise AssertionError(f"case {case}: gather_index is not a permutation")
            if not all(gather[inv[pos]] == pos for pos in range(num_tokens_pad)):
                raise AssertionError(f"case {case}: inv_gather_index mismatch")

        together = [pos for index in per_rank for pos in index]
        if sorted(together) != list(range(num_tokens_pad)):
            raise AssertionError(f"case {case}: rank local indices overlap or miss tokens")
        checked += 1

    print(f"SELFTEST PLAN OK: {checked} random batches, cp_size={p}, padded={num_tokens_pad}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - keep the failure easy to grep
        print(f"SELFTEST PLAN FAILED: {exc!r}", file=sys.stderr)
        raise
