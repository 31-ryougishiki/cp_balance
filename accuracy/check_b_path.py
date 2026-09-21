#!/usr/bin/env python3
"""[新 main 线] 静态证明 cp_balance 只作用在 zigzag 路径上。

    python3 check_b_path.py --repo /home/z30055003/vllm-ascend \
        --base-repo /home/z30055003/vllm-ascend-base

只读源码，不需要 NPU、不 import 运行时。PASS 表示：非 zigzag 的 forward 静态上碰不到
cp_balance 专有代码（逐位等价仍由远端 compare_first_token.py 验证）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SWITCH = "VLLM_ASCEND_CP_BALANCE"
SWITCH_ALLOWED = {"envs.py", "layers/cp_zigzag.py"}
ZIGZAG_FUNCTION = "def zigzag_active() -> bool:"
FIXED_ORDER_CALL = "fixed_order_reduce_scatter("
FIXED_ORDER_GATED_FILES = ("ops/linear_op.py", "ops/fused_moe/shared_experts.py")
GUARD = "if zigzag_active():"


def gated_call_sites(text: str, call: str, guard: str) -> list:
    """Call sites of the call that are not inside a block opened by the guard."""
    lines = text.splitlines()
    guard_lines = [i for i, line in enumerate(lines) if line.strip() == guard]
    problems: list = []
    for index, line in enumerate(lines):
        if call not in line or line.lstrip().startswith("#"):
            continue
        ok = False
        for guard_index in reversed(guard_lines):
            if guard_index >= index:
                continue
            between = lines[guard_index + 1 : index]
            if any(bl.startswith(("def ", "class ")) for bl in between):
                break
            ok = True
            break
        if not ok:
            problems.append("%d: %s is not guarded by %s" % (index + 1, call, guard))
    return problems


def function_body(text: str, marker: str) -> str:
    start = text.index(marker) + len(marker)
    lines = text[start:].splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith("    def ") or line.startswith("class "):
            return "".join(lines[:index])
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="", help="default: harness.json trees.cur")
    parser.add_argument("--base-repo", default="", help="optional vllm-ascend-base checkout")
    args = parser.parse_args()
    if not args.repo:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import serve_config
        args.repo = serve_config.tree_path("cur")

    pkg = Path(args.repo) / "vllm_ascend"
    if not pkg.is_dir():
        raise SystemExit(f"vllm_ascend package not found under {args.repo}")
    failures: list = []

    def src(rel: str) -> str:
        return (pkg / rel).read_text(encoding="utf-8")

    def check(ok: bool, what: str, detail: list | None = None) -> None:
        print(f"[check] {'OK  ' if ok else 'FAIL'} {what}")
        if not ok:
            failures.append(what)
            for line in (detail or [])[:6]:
                print(f"        {line}")

    ctx = src("ascend_forward_context.py")

    # 1. 逐 forward 的门控开关只此一处
    check(ctx.count(ZIGZAG_FUNCTION) == 1, "per-forward zigzag_active() helper exists exactly once")

    # 2. 所有 owner 无关归约都在 zigzag 门控内（连续切片路径逐位走原集合通信）
    for rel in FIXED_ORDER_GATED_FILES:
        text = src(rel)
        problems = gated_call_sites(text, FIXED_ORDER_CALL, GUARD)
        check(not problems, f"{rel}: every fixed-order reduce site is gated on zigzag_active()", problems)

    # 3. zigzag 只在 DSA-CP 元数据里落地：别处不得写 zigzag_index
    writers: list = []
    allowed_writers = (
        "attention/context_parallel/sfa_cp.py",
        "attention/context_parallel/zigzag_cp.py",
        "layers/cp_zigzag.py",
        "ascend_forward_context.py",
    )
    for path in sorted(pkg.rglob("*.py")):
        rel = path.relative_to(pkg).as_posix()
        if rel in allowed_writers:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for index, line in enumerate(text.splitlines(), start=1):
            if "zigzag_index =" in line and "None" not in line:
                writers.append(f"{rel}:{index}")
    check(not writers, "zigzag_index is only published by the DSA-CP metadata path", writers)

    # 4. 出口 gather 只有一处（模型边界），runner 不再自己拼
    runner = src("worker/model_runner_v1.py")
    check(
        "zigzag_gather_hidden_states_and_aux" not in runner,
        "model runner does not gather zigzag hidden states itself (model boundary owns it)",
    )
    boundary = src("patch/worker/patch_deepseek_v2.py")
    check(
        boundary.count("zigzag_gather_hidden_states_and_aux") >= 1 and "zigzag_active" in boundary,
        "model boundary performs the zigzag exit gather behind the per-forward flag",
    )

    # 5. 裸开关只允许出现在 envs.py 与资格判定里
    offenders: list = []
    for path in sorted(pkg.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(pkg).as_posix()
        start = 0
        while True:
            hit = text.find(SWITCH, start)
            if hit < 0:
                break
            start = hit + len(SWITCH)
            if text[hit + len(SWITCH) : hit + len(SWITCH) + 1] == "_":
                continue
            if rel not in SWITCH_ALLOWED:
                offenders.append("%s:%d" % (rel, text[:hit].count(chr(10)) + 1))
    check(not offenders, "bare VLLM_ASCEND_CP_BALANCE is only read by envs.py / the eligibility predicate", offenders)

    # 6. 与 base 对照：移植没有动过融合 q_up 分支
    if args.base_repo:
        base_file = Path(args.base_repo) / "vllm_ascend/attention/sfa_v1.py"
        if not base_file.is_file():
            check(False, f"base checkout not readable: {base_file}")
        else:
            marker = "def _q_proj_and_k_up_proj"
            check(
                function_body(src("attention/sfa_v1.py"), marker)
                == function_body(base_file.read_text(encoding="utf-8"), marker),
                "q_up_proj body is byte-identical to the base checkout",
            )

    print(f"[check] RESULT: {'FAIL' if failures else 'PASS'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
