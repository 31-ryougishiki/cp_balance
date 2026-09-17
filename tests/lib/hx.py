#!/usr/bin/env python3
"""tests/ 的公共小工具（bash 侧经 tests/lib/common.sh 调用）。

    hx.py configs <family>                 family 用到的配置名（每行一个）
    hx.py field <cfg> <field>              生效字段值（含 CP_BALANCE_* 覆盖），JSON
    hx.py fingerprint <cfg>                等价 run.sh --dry-run 的第一行
    hx.py path <cfg> <repo|model|prelude>  路径（prelude 取 source 的文件）
    hx.py ports <cfg...>                   "cfg port"（每行一个）
    hx.py static_check <matrix>            "repo=<v> base_repo=<v>"
    hx.py compares <matrix>                "label<TAB>left<TAB>right<TAB>require_text"
    hx.py windows <dir>                    "window<TAB>kernel_steps<TAB>step_trace_rows<TAB>label"
    hx.py usable <dir>                     rank_count > 0 的窗口数（0=解析没拿到数据）
    hx.py listen <port>                    0=空闲 1=有人在听
"""

from __future__ import annotations

import json
import re
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import serve_config as sc  # noqa: E402


def effective(name: str) -> dict:
    return sc.apply_env_overrides(sc.load_config(name))


def cmd_configs(family: str) -> int:
    if family not in ("a5", "a3"):
        print("family must be a5 or a3, got %r" % family, file=sys.stderr)
        return 2
    out = []
    for path in sorted((ROOT / "configs").glob("*.json")):
        name = path.stem
        if name.startswith("_") or name.startswith("matrix_") or name == "default":
            continue
        if (family == "a5") != ("_a5" in name):
            continue
        out.append(name)
    print("\n".join(out))
    return 0


def cmd_field(name: str, key: str) -> int:
    node = effective(name)
    for part in key.split("."):
        node = node[part]
    print(node if isinstance(node, str) else json.dumps(node, ensure_ascii=False))
    return 0


def cmd_fingerprint(name: str) -> int:
    cfg = effective(name)
    print(sc.fingerprint(cfg, sc.build_env(cfg)))
    return 0


def cmd_path(name: str, key: str) -> int:
    cfg = effective(name)
    if key == "prelude":
        match = re.search(r"source\s+(\S+)", str(cfg.get("prelude") or ""))
        print(match.group(1) if match else "")
        return 0
    print(cfg.get(key) or "")
    return 0


def cmd_ports(names: list) -> int:
    for name in names:
        print("%s %s" % (name, effective(name).get("port")))
    return 0


def cmd_static_check(matrix: str) -> int:
    check = sc.load_config(matrix).get("static_check") or {}
    print("repo=%s" % (check.get("repo") or ""))
    print("base_repo=%s" % (check.get("base_repo") or ""))
    return 0


def cmd_compares(matrix: str) -> int:
    for item in sc.load_config(matrix).get("compare") or []:
        fields = (item.get("label"), item.get("left"), item.get("right"), bool(item.get("require_text")))
        print("%s\t%s\t%s\t%s" % fields)
    return 0


def cmd_windows(prof_dir: str) -> int:
    root = Path(prof_dir)
    summary = root / "summary.json"
    if summary.is_file():
        data = json.loads(summary.read_text(encoding="utf-8"))
        for name, info in sorted((data.get("windows") or {}).items()):
            audit = info.get("audit") or {}
            raw_steps = audit.get("kernel_steps")
            raw_rows = audit.get("step_trace_rows")
            steps = "-" if raw_steps is None else raw_steps
            rows = "-" if raw_rows is None else raw_rows
            print("%s\t%s\t%s\t%s" % (name, steps, rows, info.get("label", "-")))
        return 0
    windows = root / "windows.json"
    if windows.is_file():
        data = json.loads(windows.read_text(encoding="utf-8"))
        for entry in data.get("entries") or []:
            for name in entry.get("window_ids") or []:
                print("%s\t-\t-\t%s" % (name, entry.get("label", "-")))
        return 0
    print("no summary.json / windows.json under %s" % root, file=sys.stderr)
    return 3


def cmd_usable(prof_dir: str) -> int:
    """rank_count > 0 的窗口数（0 表示解析没拿到任何 rank 数据）。"""
    summary = Path(prof_dir) / "summary.json"
    if not summary.is_file():
        print(0)
        return 0
    data = json.loads(summary.read_text(encoding="utf-8"))
    count = 0
    for info in (data.get("windows") or {}).values():
        if int(info.get("rank_count") or 0) > 0:
            count += 1
    print(count)
    return 0


def cmd_listen(port: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2)
        print(1 if sock.connect_ex(("127.0.0.1", int(port))) == 0 else 0)
    return 0


def main(argv: list) -> int:
    # shell 侧按行取值：不要把 \n 翻成 \r\n
    try:
        sys.stdout.reconfigure(newline="")
    except Exception:
        pass
    if not argv:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    cmd, rest = argv[0], argv[1:]
    table = {
        "configs": lambda: cmd_configs(rest[0]),
        "field": lambda: cmd_field(rest[0], rest[1]),
        "fingerprint": lambda: cmd_fingerprint(rest[0]),
        "path": lambda: cmd_path(rest[0], rest[1]),
        "ports": lambda: cmd_ports(rest),
        "static_check": lambda: cmd_static_check(rest[0]),
        "compares": lambda: cmd_compares(rest[0]),
        "windows": lambda: cmd_windows(rest[0]),
        "usable": lambda: cmd_usable(rest[0]),
        "listen": lambda: cmd_listen(rest[0]),
    }
    if cmd not in table:
        print("unknown command: %s" % cmd, file=sys.stderr)
        return 2
    return table[cmd]()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
