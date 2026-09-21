#!/usr/bin/env python3
"""(新 main 线) 目标代码树清单：tests/lib/trees.json 的读取器。

    trees.py list                 # role<TAB>abs_path<TAB>remote<TAB>ref<TAB>kind<TAB>required<TAB>exists
    trees.py get <role> <field>   # path|remote|ref|kind|required  （path 已绝对化）

换分支/换远端/固定 commit 只改 tests/lib/trees.json，脚本不用动。
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

LIB = Path(__file__).resolve().parent
ROOT = LIB.parents[1]          # harness 根目录
CONFIG = LIB / "trees.json"


def load() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8")).get("trees", {})


def resolve(role: str) -> dict:
    entry = dict(load().get(role) or {})
    if not entry:
        raise SystemExit("unknown tree role: " + role)
    path = Path(str(entry.get("path", "")))
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    entry["path"] = str(path)
    entry.setdefault("remote", "")
    entry.setdefault("ref", "main")
    entry.setdefault("kind", "tip")
    entry.setdefault("required", True)
    return entry


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] == "list":
        for role in load():
            entry = resolve(role)
            fields = [role, entry["path"], entry["remote"], str(entry["ref"]), str(entry["kind"]),
                      str(entry["required"]).lower(), "yes" if Path(entry["path"]).is_dir() else "no"]
            print("\t".join(fields))
        return 0
    if args[0] == "model-candidates":
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
        for pattern in cfg.get("model_candidates", []):
            for hit in sorted(glob.glob(pattern)):
                print(hit)
        return 0
    if args[0] == "get" and len(args) >= 3:
        entry = resolve(args[1])
        value = entry.get(args[2])
        if value is None:
            return 1
        print(str(value).lower() if isinstance(value, bool) else value)
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
