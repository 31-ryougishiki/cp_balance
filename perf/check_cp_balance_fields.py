#!/usr/bin/env python3
"""Static audit of the cp_balance metadata plumbing.

    python3 check_cp_balance_fields.py --repo /opt/its/z30055003/vllm-ascend

The zigzag path passes data through three hand-written containers:

    ZigzagPlan   (vllm_ascend/layers/cp_zigzag.py)
    the meta dict returned by _build_zigzag_meta (attention/sfa_v1.py)
    DSACPContext (attention/sfa_v1.py)

A single renamed or missing field is a TypeError on the first prefill, that is a
service which dies ten minutes into a run, or a silent fallback to the
continuous DSA-CP path.  This checks with ast only, no torch required:

* every keyword passed to DSACPContext exists on the dataclass and every
  required field is passed;
* every plan.<name> read in sfa_v1.py exists on ZigzagPlan (properties included);
* every zigzag["<key>"] read exists among the keys _build_zigzag_meta returns;
* every <dsacp>.<name> read exists on DSACPContext.

Exit code 0 and a final [check] RESULT: PASS means the plumbing is consistent.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

TARGET_FILES = (
    "vllm_ascend/attention/sfa_v1.py",
    "vllm_ascend/layers/cp_zigzag.py",
    "vllm_ascend/ascend_forward_context.py",
)


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


def find_function(tree: ast.Module, name: str):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def attributes_of(tree: ast.Module) -> list:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            out.append((node.value.id, node.attr, node.lineno))
    return out


def subscript_keys(tree: ast.Module) -> list:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                out.append((node.value.id, key.value, node.lineno))
    return out


def returned_keys(func) -> set:
    keys: set = set()
    if func is None:
        return keys
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            for key in node.value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(key.value)
    return keys


def binds_dsacp(value) -> bool:
    """True only when the whole right side IS a DSACPContext reference.

    Matching on the unparsed text would also catch `x = ctx.slot_mapping_cp`,
    where x is a plain tensor and its attributes have nothing to do with the
    dataclass.
    """
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
    """(function, base, attr, line) for attributes read on a DSACPContext.

    The scan is per function on purpose: the same file uses ``ctx`` for the
    vLLM forward context elsewhere, and a file-wide name list would flag
    those reads as missing DSACPContext fields.
    """
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
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                if node.value is not None and binds_dsacp(node.value):
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="/opt/its/z30055003/vllm-ascend")
    args = parser.parse_args()
    repo = Path(args.repo)

    trees = {}
    for relative in TARGET_FILES:
        path = repo / relative
        if not path.is_file():
            print("[check] cannot read %s" % path)
            return 1
        trees[relative] = (path, read_tree(path))

    sfa_path, sfa = trees["vllm_ascend/attention/sfa_v1.py"]
    zigzag_path, zigzag = trees["vllm_ascend/layers/cp_zigzag.py"]

    plan_cls = find_class(zigzag, "ZigzagPlan")
    ctx_cls = find_class(sfa, "DSACPContext")
    if plan_cls is None or ctx_cls is None:
        print("[check] FAIL ZigzagPlan / DSACPContext class not found")
        return 1
    plan_fields = body_fields(plan_cls)
    ctx_fields = body_fields(ctx_cls)

    problems: list = []

    built = 0
    for node in ast.walk(sfa):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "DSACPContext":
            built += 1
            passed = {kw.arg for kw in node.keywords if kw.arg}
            unknown = sorted(passed - set(ctx_fields))
            missing = sorted(n for n, has_default in ctx_fields.items() if not has_default and n not in passed)
            if unknown:
                problems.append("sfa_v1.py:%d DSACPContext got unknown fields %s" % (node.lineno, unknown))
            if missing:
                problems.append("sfa_v1.py:%d DSACPContext missing required fields %s" % (node.lineno, missing))
    if not built:
        problems.append("no DSACPContext construction found")

    for base, attr, line in attributes_of(sfa):
        if base == "plan" and attr not in plan_fields:
            problems.append("sfa_v1.py:%d reads plan.%s, not on ZigzagPlan" % (line, attr))

    meta_keys = returned_keys(find_function(sfa, "_build_zigzag_meta"))
    if not meta_keys:
        problems.append("could not read the key set of _build_zigzag_meta")
    for base, key, line in subscript_keys(sfa):
        if base == "zigzag" and key not in meta_keys:
            problems.append('sfa_v1.py:%d reads zigzag[%r], never returned by _build_zigzag_meta' % (line, key))

    for relative, (path, tree) in trees.items():
        for func, base, attr, line in function_dsacp_reads(tree, ("dsa_cp_context", "dsa_cp_ctx")):
            if attr not in ctx_fields:
                problems.append(
                    "%s:%d %s() reads %s.%s, not on DSACPContext" % (path.name, line, func, base, attr)
                )

    print("[check] ZigzagPlan fields=%d, DSACPContext fields=%d" % (len(plan_fields), len(ctx_fields)))
    print("[check] _build_zigzag_meta returns %d keys, DSACPContext constructions=%d" % (len(meta_keys), built))
    for item in problems:
        print("[check] FAIL %s" % item)
    print("[check] RESULT: %s" % ("PASS" if not problems else "FAIL"))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
