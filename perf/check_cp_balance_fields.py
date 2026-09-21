#!/usr/bin/env python3
"""[新 main 线] cp_balance 元数据字段静态审计。

    python3 check_cp_balance_fields.py --repo /home/z30055003/vllm-ascend

zigzag 布局在三个手写容器之间传递，字段名写错 = 首个 prefill 直接崩：

    ZigzagPlan     (vllm_ascend/layers/cp_zigzag.py)                     CPU 侧计划
    ZigzagCPPlan   (vllm_ascend/attention/context_parallel/zigzag_cp.py) 设备张量（共享 planner）
    DSACPContext   (vllm_ascend/attention/context_parallel/sfa_cp.py)    随注意力元数据下发

纯 ast，不需要 torch。判据：末行 [check] RESULT: PASS，退出码 0。
"""

from __future__ import annotations

import argparse
import sys
import ast
from pathlib import Path

PLAN_FILE = "vllm_ascend/layers/cp_zigzag.py"
ZIGZAG_FILE = "vllm_ascend/attention/context_parallel/zigzag_cp.py"
SFA_FILE = "vllm_ascend/attention/context_parallel/sfa_cp.py"
INDEXER_FILE = "vllm_ascend/attention/indexer.py"
CTX_FILE = "vllm_ascend/ascend_forward_context.py"

TARGET_FILES = (PLAN_FILE, ZIGZAG_FILE, SFA_FILE, INDEXER_FILE, CTX_FILE)


def read_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def body_fields(cls: ast.ClassDef) -> dict:
    """name -> has_default, for annotated attributes and properties."""
    found: dict = {}
    for stmt in cls.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            found[stmt.target.id] = stmt.value is not None
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for deco in stmt.decorator_list:
                name = deco.id if isinstance(deco, ast.Name) else getattr(deco, "attr", "")
                if name == "property":
                    found[stmt.name] = True
    return found


def find_class(tree: ast.Module, name: str) -> ast.ClassDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def attributes_of(tree: ast.Module) -> list:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            out.append((node.value.id, node.attr, node.lineno))
    return out


def binds_dsacp(value) -> bool:
    """True only when the whole right side IS a DSACPContext reference."""
    if isinstance(value, ast.Attribute) and value.attr == "dsa_cp_context":
        return True
    if isinstance(value, ast.Call):
        func = value.func
        if isinstance(func, ast.Name) and func.id == "get_zigzag_cp_context":
            return True
        if isinstance(func, ast.Name) and func.id == "getattr":
            for arg in value.args[1:2]:
                if isinstance(arg, ast.Constant) and arg.value == "dsa_cp_context":
                    return True
    return False


def function_dsacp_reads(tree: ast.Module, extra: tuple) -> list:
    """(function, base, attr, line) for attributes read on a DSACPContext."""
    out = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names = set(extra)
        for node in ast.walk(func):
            targets = []
            if isinstance(node, ast.Assign):
                if binds_dsacp(node.value):
                    targets = node.targets
            elif isinstance(node, ast.AnnAssign) and node.value is not None and binds_dsacp(node.value):
                targets = [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        if not names:
            continue
        for node in ast.walk(func):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id in names:
                    out.append((func.name, node.value.id, node.attr, node.lineno))
    return out


def constructions(tree: ast.Module, cls_name: str) -> list:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == cls_name
    ]


def check_construction(cls_name: str, cls_fields: dict, calls: list, where: str, problems: list) -> int:
    for node in calls:
        passed = {kw.arg for kw in node.keywords if kw.arg}
        unknown = sorted(passed - set(cls_fields))
        missing = sorted(n for n, has_default in cls_fields.items() if not has_default and n not in passed)
        if unknown:
            problems.append("%s:%d %s got unknown fields %s" % (where, node.lineno, cls_name, unknown))
        if missing:
            problems.append("%s:%d %s missing required fields %s" % (where, node.lineno, cls_name, missing))
    return len(calls)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="", help="default: harness.json trees.cur")
    args = parser.parse_args()
    if not args.repo:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import serve_config
        args.repo = serve_config.tree_path("cur")
    repo = Path(args.repo)

    trees = {}
    for relative in TARGET_FILES:
        path = repo / relative
        if not path.is_file():
            print("[check] cannot read %s" % path)
            return 1
        trees[relative] = read_tree(path)

    plan = find_class(trees[PLAN_FILE], "ZigzagPlan")
    zigzag_plan = find_class(trees[ZIGZAG_FILE], "ZigzagCPPlan")
    ctx_cls = find_class(trees[SFA_FILE], "DSACPContext")
    if plan is None or zigzag_plan is None or ctx_cls is None:
        print("[check] FAIL ZigzagPlan / ZigzagCPPlan / DSACPContext class not found")
        return 1

    plan_fields = body_fields(plan)
    zigzag_fields = body_fields(zigzag_plan)
    ctx_fields = body_fields(ctx_cls)
    problems: list = []

    # 1. 构造点：字段名与必填项
    plan_built = check_construction(
        "ZigzagCPPlan", zigzag_fields, constructions(trees[ZIGZAG_FILE], "ZigzagCPPlan"), "zigzag_cp.py", problems
    )
    ctx_built = check_construction(
        "DSACPContext", ctx_fields, constructions(trees[SFA_FILE], "DSACPContext"), "sfa_cp.py", problems
    )
    if not plan_built:
        problems.append("no ZigzagCPPlan construction found")
    if not ctx_built:
        problems.append("no DSACPContext construction found")

    # 2. plan.<attr> 读点：zigzag_cp.py 里的 plan 是 ZigzagPlan，两个 builder 里的是 ZigzagCPPlan
    for where, fields in ((ZIGZAG_FILE, plan_fields), (SFA_FILE, zigzag_fields), (INDEXER_FILE, zigzag_fields)):
        for base, attr, line in attributes_of(trees[where]):
            if base == "plan" and attr not in fields:
                problems.append("%s:%d reads plan.%s, not on the plan dataclass" % (where, line, attr))

    # 3. DSACPContext 读点（跨文件；按函数作用域绑定，避免把 forward context 的 ctx 误判）
    for relative, tree in trees.items():
        for func, base, attr, line in function_dsacp_reads(tree, ("dsa_cp_context", "dsa_cp_ctx")):
            if attr not in ctx_fields:
                problems.append("%s:%d %s() reads %s.%s, not on DSACPContext" % (relative, line, func, base, attr))

    # 4. 回退三件套必须同时构造（缺一个 _disable_zigzag_metadata_for_fallback 会抛错）
    sfa_text = (repo / SFA_FILE).read_text(encoding="utf-8")
    ctx_text = (repo / CTX_FILE).read_text(encoding="utf-8")
    for name in ("fallback_slot_mapping_cp=", "fallback_cos=", "fallback_sin="):
        if name not in sfa_text:
            problems.append("sfa_cp.py never sets %s" % name.rstrip("="))
    if "cannot be safely disabled" not in ctx_text:
        problems.append("ascend_forward_context.py lost the hard failure for an incomplete fallback")

    print(
        "[check] ZigzagPlan fields=%d, ZigzagCPPlan fields=%d, DSACPContext fields=%d"
        % (len(plan_fields), len(zigzag_fields), len(ctx_fields))
    )
    print("[check] ZigzagCPPlan constructions=%d, DSACPContext constructions=%d" % (plan_built, ctx_built))
    for item in problems:
        print("[check] FAIL %s" % item)
    print("[check] RESULT: %s" % ("PASS" if not problems else "FAIL"))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
