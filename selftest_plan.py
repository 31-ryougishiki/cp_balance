#!/usr/bin/env python3
"""CPU self-test for the cp_balance token plan.

Does not touch the NPU.  For many random query-length sets it checks that

* every natural token position is owned by exactly one rank;
* the rank-concatenating all-gather order equals zigzag_gather_index;
* inv_gather_index really restores natural order;
* every rank owns the same number of local tokens.

Usage::

    python selftest_plan.py --cp-size 16 --cases 2000
    python selftest_plan.py --repo /opt/its/z30055003/vllm-ascend --cases 2000

--repo is prepended to sys.path, so it decides which vllm_ascend is imported.
That matters: vllm_ascend/layers/ only exists on the cp_balance branches, not on
main, and an installed vllm-ascend in site-packages can shadow the repo.  The
script always reports which tree it loaded and, on failure, the path it expected
plus the commands to fix it.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

DEFAULT_REPO = "/opt/its/z30055003/vllm-ascend"


def diagnose(repo: str, exc: BaseException) -> None:
    print("[selftest] import failed: %s: %s" % (type(exc).__name__, exc))
    print("[selftest] --repo=%s" % repo)
    target = Path(repo) / "vllm_ascend/layers/cp_zigzag.py"
    print("[selftest] expected file %s exists=%s" % (target, target.is_file()))
    try:
        import vllm_ascend  # noqa: PLC0415 - diagnosis only

        print("[selftest] vllm_ascend resolved to %s" % vllm_ascend.__file__)
        print("[selftest]   -> that tree is not the one carrying layers/cp_zigzag.py")
    except Exception as inner:  # noqa: BLE001 - report and continue
        print("[selftest] vllm_ascend is not importable either: %s" % inner)
    print("[selftest] how to fix:")
    print("[selftest]   git -C %s branch --show-current   # must be cp_balance" % repo)
    print("[selftest]   git -C %s log --oneline -1         # must be a cp_balance commit" % repo)
    print("[selftest]   python3 selftest_plan.py --repo %s --cases 2000" % repo)
    print("[selftest]   or: export PYTHONPATH=%s:$PYTHONPATH" % repo)
    print("[selftest] note: vllm_ascend/layers exists on cp_balance and")
    print("[selftest]       glm52_cp_balance_v3, but NOT on main.")


def load_plan(repo: str):
    if repo:
        sys.path.insert(0, repo)
    try:
        from vllm_ascend.layers.cp_zigzag import build_zigzag_plan  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        diagnose(repo, exc)
        raise SystemExit(1)
    import vllm_ascend.layers.cp_zigzag as module  # noqa: PLC0415

    print("[selftest] cp_zigzag loaded from %s" % module.__file__)
    return build_zigzag_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="code tree to import vllm_ascend from")
    parser.add_argument("--cp-size", type=int, default=16)
    parser.add_argument("--cases", type=int, default=2000)
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument("--max-reqs", type=int, default=4)
    args = parser.parse_args()

    build_zigzag_plan = load_plan(args.repo)

    rng = random.Random(0)
    p = args.cp_size
    checked = 0
    num_tokens_pad = 0
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
                    f"case {case}: rank {rank} has {len(index)} local tokens, expected {num_tokens_pad // p}"
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
