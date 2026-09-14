#!/usr/bin/env python3
"""Static proof that cp_balance is confined to the zigzag path.

Checks, without NPU and without importing the runtime:

1. every cp_balance-specific collective is gated on the per-forward zigzag flag;
2. the bare VLLM_ASCEND_CP_BALANCE switch is only read by the eligibility
   predicate inside layers/cp_zigzag.py, never by a collective;
3. the fused q_up projection of the base branch is present with the same
   permutations.

    python check_b_path.py --repo /opt/its/z30055003/vllm-ascend \
        --base-repo /opt/its/z30055003/vllm-ascend-base

PASS means the non-cp_balance path cannot statically reach any cp_balance-only
code; the numeric part is verified by compare_first_token.py on the remote node.
"""

from __future__ import annotations

import argparse
from pathlib import Path

SWITCH = "VLLM_ASCEND_CP_BALANCE"
SWITCH_ALLOWED = {"envs.py", "layers/cp_zigzag.py"}
FUSED = 'if hasattr(torch_npu, "npu_transpose_batchmatmul"):'


def _function_body(text: str, marker: str) -> str:
    start = text.index(marker) + len(marker)
    lines = text[start:].splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith("    def ") or line.startswith("class "):
            return "".join(lines[:index])
    return "".join(lines)


def _fused_block(text: str) -> str:
    start = text.index(FUSED)
    end = text.index("else:", start)
    return " ".join(text[start:end].split())


def _find_all(text: str, needle: str) -> list[int]:
    hits: list[int] = []
    start = text.find(needle)
    while start != -1:
        hits.append(start)
        start = text.find(needle, start + 1)
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="/opt/its/z30055003/vllm-ascend")
    parser.add_argument("--base-repo", default="", help="optional vllm-ascend-base checkout")
    args = parser.parse_args()

    pkg = Path(args.repo) / "vllm_ascend"
    if not pkg.is_dir():
        raise SystemExit(f"vllm_ascend package not found under {args.repo}")
    failures: list[str] = []

    def src(rel: str) -> str:
        return (pkg / rel).read_text(encoding="utf-8")

    def check(ok: bool, what: str, detail: list[str] | None = None) -> None:
        print(f"[check] {'OK  ' if ok else 'FAIL'} {what}")
        if not ok:
            failures.append(what)
            for line in (detail or [])[:6]:
                print(f"        {line}")

    ctx = src("ascend_forward_context.py")
    linear = src("ops/linear_op.py")
    custom = src("ops/register_custom_ops.py")

    check(ctx.count("def zigzag_active() -> bool:") == 1, "per-forward zigzag_active() helper exists")
    check(
        linear.count("if zigzag_active():") == 2,
        "both row-parallel reductions are gated on zigzag_active()",
    )
    check("dsa_cp_enabled" not in linear, "config-level DSA-CP gate removed from the MLP reduction")
    check(
        "def _fixed_order_zigzag_reduce_scatter" in custom and "if not zigzag_active():" in custom,
        "pad_and_reduce (embedding + MoE finalize) is gated on zigzag_active()",
    )
    check(
        "_fixed_order_dsa_cp_reduce_scatter" not in custom,
        "old config-level reduction helper is gone",
    )

    offenders: list[str] = []
    for path in sorted(pkg.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(pkg).as_posix()
        for hit in _find_all(text, SWITCH):
            after = text[hit + len(SWITCH) : hit + len(SWITCH) + 1]
            if after == "_":
                continue
            if rel not in SWITCH_ALLOWED:
                line = text[:hit].count(chr(10)) + 1
                offenders.append(f"{rel}:{line}")
    check(not offenders, "bare VLLM_ASCEND_CP_BALANCE is only read by the eligibility predicate", offenders)

    sfa = src("attention/sfa_v1.py")
    body = _function_body(sfa, "def _q_proj_and_k_up_proj")
    check(FUSED in body, "q_up_proj keeps the fused transpose-batchmatmul branch")
    check(
        "perm_x1=(1, 0, 2)," in body and "perm_x2=(0, 1, 2)," in body and "perm_y=(1, 0, 2)," in body,
        "q_up_proj fused op uses the base-branch permutations",
    )
    check("torch.bmm(q_nope, self.W_UK_T)" in body, "q_up_proj keeps the bmm fallback")

    if args.base_repo:
        base_file = Path(args.base_repo) / "vllm_ascend/attention/sfa_v1.py"
        if not base_file.is_file():
            check(False, f"base checkout not readable: {base_file}")
        else:
            base_body = _function_body(base_file.read_text(encoding="utf-8"), "def _q_proj_and_k_up_proj")
            check(
                FUSED in base_body and _fused_block(body) == _fused_block(base_body),
                "q_up_proj fused block is byte-identical to the base branch",
            )

    print(f"[check] RESULT: {'FAIL' if failures else 'PASS'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
