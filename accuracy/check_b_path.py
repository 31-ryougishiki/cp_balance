#!/usr/bin/env python3
"""[新 main 线] 静态证明 cp_balance 只作用在 zigzag 路径上。

    python3 check_b_path.py --repo /home/z30055003/vllm-ascend \
        --base-repo /home/z30055003/vllm-ascend-base

只读源码，不需要 NPU、不 import 运行时。PASS 表示：zigzag 的行布局只存在于 DSA-CP
attention 内部，模型主流（embedding / 模型边界 / dense MLP / MoE / runner）与无
cp_balance 时逐位相同（逐位等价仍由远端 compare_first_token.py 验证）。
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

SWITCH = "VLLM_ASCEND_CP_BALANCE"
SWITCH_ALLOWED = {"envs.py", "layers/cp_zigzag.py"}
# 布局的唯一来源：每层的 DSA-CP 元数据（DSACPContext），不允许再有全局开关
LAYOUT_SOURCES = ("dsa_cp_context", "zigzag_index", "inv_gather_index")
GLOBAL_FLAGS = ("def zigzag_active()", "_EXTRA_CTX.zigzag_cp_active", "_EXTRA_CTX.zigzag_cp_context")
# 模型主流里必须保持与布局无关的文件（它们是 dp=1 下 element-wise 集合通信的宿主）；
# 模型边界是有意例外：它按计划只做"进出各一次行序重排"。
MODEL_STREAM_FILES = (
    "ops/vocab_parallel_embedding.py",
    "ops/linear_op.py",
    "ops/fused_moe/shared_experts.py",
    "worker/model_runner_v1.py",
)
BOUNDARY_FILE = "patch/worker/patch_deepseek_v2.py"
STREAM_HANDLES = ("zigzag_stream_gather_index", "zigzag_stream_inv_gather_index")
# 模型级布局用的原语：随"主流按计划序存储"的改法一起删除，残留即代表回到了错布局
DEAD_PRIMITIVES = ("fixed_order_reduce_scatter", "zigzag_shard_tensor", "zigzag_reorder_moe_aux", "zigzag_gather_tensor")
ALLOWED_WRITERS = (
    "attention/context_parallel/sfa_cp.py",
    "attention/context_parallel/zigzag_cp.py",
    "layers/cp_zigzag.py",
    "ascend_forward_context.py",
)


def zigzag_identifiers(text: str) -> list:
    """Lines of zigzag-looking code identifiers (comments/docstrings excluded)."""
    hits: list = []
    for node in ast.walk(ast.parse(text)):
        names: list = []
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.arg):
            names.append(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            names.append(node.arg)
        elif isinstance(node, ast.alias):
            names.extend([node.name, node.asname or ""])
        if any("zigzag" in name.lower() for name in names):
            hits.append(node.lineno)
    return sorted(set(hits))


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

    # 1. 布局只能由元数据决定：旧的全局开关/上下文句柄不得复活
    leftovers = [name for name in GLOBAL_FLAGS if name in ctx]
    check(
        not leftovers,
        "no global zigzag flag survives (the plan plus the boundary handles are the layout)",
        leftovers,
    )

    # 2. 模型主流（embedding / dense MLP / MoE / runner）保持布局无关
    leaks: list = []
    for rel in MODEL_STREAM_FILES:
        for line in zigzag_identifiers(src(rel)):
            leaks.append("%s:%d" % (rel, line))
    check(not leaks, "the replicated model stream (embedding/MLP/MoE/runner) is zigzag-free", leaks)

    # 3. 边界是唯一的例外：只做两次行序重排，且句柄来自 forward context
    boundary = src(BOUNDARY_FILE)
    missing = [name for name in STREAM_HANDLES if name not in boundary]
    check(not missing, "the model boundary permutes the stream through the published handles", missing)

    # 4. 句柄在 veto 之后发布（veto 掉计划时必须得到 None）
    veto = ctx.index('_disable_zigzag_metadata_for_fallback(attn_metadata)')
    publish = ctx.index("_EXTRA_CTX.zigzag_stream_gather_index = getattr(")
    check(publish > veto, "the stream handles are published after the forward-level veto")
    check(
        all(ctx.count(name) >= 2 for name in STREAM_HANDLES),
        "the stream handles are declared as extra forward-context attributes and published",
    )

    # 5. attention 层不再做任何布局搬运：窗口是连续切片，出口是普通 all_gather
    sfa = src("attention/context_parallel/sfa_cp.py")
    moved = [token for token in ("hidden_states[context.zigzag_index]", "zigzag_gather_tensor", "zigzag_inv_gather_index") if token in sfa]
    check(not moved, "the attention layer adds no row-movement of its own", moved)
    check(
        "hidden_states[context.local_start : context.local_end_with_pad]" in sfa,
        "the attention row window is the plain contiguous slice",
    )

    # 5. zigzag 只在 DSA-CP 元数据里落地：别处不得写 zigzag_index
    writers: list = []
    for path in sorted(pkg.rglob("*.py")):
        rel = path.relative_to(pkg).as_posix()
        if rel in ALLOWED_WRITERS:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for index, line in enumerate(text.splitlines(), start=1):
            if "zigzag_index =" in line and "None" not in line:
                writers.append(f"{rel}:{index}")
    check(not writers, "zigzag_index is only published by the DSA-CP metadata path", writers)

    # 6. 裸开关只允许出现在 envs.py 与资格判定里
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
