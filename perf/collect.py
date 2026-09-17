#!/usr/bin/env python3
"""Collect one profiling round into a single tarball (never the raw traces).

Driven by tests/perf/p24_collect (the standard entry) or by hand:

    python3 perf/collect.py                          # every config with a summary.json
    python3 perf/collect.py prof_cur_cp0 prof_cur_cp1
    python3 perf/collect.py --prune-traces           # delete *_ascend_pt after bundling

Output goes to collect_<timestamp>/: inventory.txt, clean_s.txt, cmp_*.txt,
order_*.txt, fingerprints.txt and <dir>.tgz (with size and md5 printed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (HERE, ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import serve_config  # noqa: E402

SMALL = ("summary.json", "windows.json", "order_rank0.json")
# Files that must never enter the bundle (older export/ directories may still
# hold them): they are huge and no analysis here reads them.
SKIP_EXPORT = ("communication", "communication_matrix")


def export_keep(name: str) -> bool:
    """export/ files are named <rank>__<original>; drop the huge communication dumps."""
    return not name.split("__", 1)[-1].startswith(SKIP_EXPORT)


def log(msg: str = "") -> None:
    print("[collect] " + msg if msg else "", flush=True)


def disk_mb(path: Path) -> float:
    if not path.is_dir():
        return 0.0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / 1048576.0


def discover() -> list:
    names = []
    for path in sorted((ROOT / "configs").glob("*.json")):
        try:
            cfg = serve_config.load_config(path.name)
        except SystemExit:
            continue
        if not (cfg.get("profiler") or {}).get("enabled"):
            continue
        if (Path(serve_config.profiler_dir(cfg)) / "summary.json").is_file():
            names.append(str(cfg.get("name") or path.stem))
    return names


def scan(names: list) -> tuple:
    info = {}
    lines = []
    for name in names:
        cfg = serve_config.load_config(name)
        prof_dir = Path(serve_config.profiler_dir(cfg))
        root_json = ROOT / ("prof_%s.json" % cfg.get("name"))
        service_log = ROOT / ("profile_%s.log" % cfg.get("name"))
        traces = [item for item in prof_dir.rglob("*_ascend_pt") if item.is_dir()] if prof_dir.is_dir() else []
        checks = [("summary.json", prof_dir / "summary.json"), ("windows.json", prof_dir / "windows.json"),
                  ("export/", prof_dir / "export"), ("order_rank0.json", prof_dir / "order_rank0.json"),
                  (root_json.name, root_json), (service_log.name, service_log)]
        missing = [label for label, path in checks if not path.exists()]
        info[name] = {"cfg": cfg, "dir": prof_dir, "root_json": root_json, "log": service_log,
                      "traces": traces, "ok": "summary.json" not in missing, "missing": missing}
        lines.append("%-22s %s" % (name, prof_dir))
        lines += ["    %-4s %s" % ("OK" if label not in missing else "MISS", label) for label, _path in checks]
        lines.append("    %-4s raw traces: %d dirs, %.1f MB (never bundled)"
                     % ("info", len(traces), sum(disk_mb(item) for item in traces)))
    return info, chr(10).join(lines) + chr(10)



def clean_table(info: dict) -> str:
    rows = []
    for name, meta in info.items():
        path = meta["root_json"]
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            log("WARNING %s is not valid JSON, skipped" % path.name)
            continue
        per_len = {int(e["target_tokens"]): e for e in (data.get("entries") or []) if e.get("target_tokens")}
        if per_len:
            rows.append((name, per_len, data.get("mean_elapsed_s")))
    if not rows:
        log("no prof_<name>.json found, skip the clean_s table")
        return ""
    lengths = sorted({item for _name, per_len, _mean in rows for item in per_len})
    text = ["%-22s %10s %12s %12s %10s" % ("config", "length", "profiled_s", "clean_s", "tokens")]
    for name, per_len, mean in rows:
        for length in lengths:
            entry = per_len.get(length)
            if entry is None:
                continue
            text.append("%-22s %10d %12s %12s %10s" % (
                name, length, entry.get("wall_s", "-"), entry.get("clean_wall_s", "-"),
                entry.get("clean_prompt_tokens", entry.get("prompt_tokens", "-"))))
        text.append("%-22s %10s %12s" % (name, "mean", mean if mean is not None else "-"))
    return chr(10).join(text) + chr(10)


def pairs(names: list) -> list:
    reference = next((name for name in names if "cur_cp0" in name and "repeat" not in name), names[0] if names else "")
    pairs_out = [(reference, name) for name in names if name != reference and reference]
    pairs_out += [(name, name + "_a2a") for name in names if name.endswith("_cp1") and name + "_a2a" in names]
    log("reference=%s, %d compare pair(s)" % (reference or "-", len(pairs_out)))
    return pairs_out


def write(out: Path, name: str, text: str, echo: bool = False) -> Path:
    path = out / name
    path.write_text(text, encoding="utf-8")
    log("wrote %s (%d lines)" % (name, text.count(chr(10))))
    if echo:
        sys.stdout.write(text if text.endswith(chr(10)) else text + chr(10))
        sys.stdout.flush()
    return path


def reports(out: Path, info: dict, names: list) -> None:
    def run(*cmd):
        proc = subprocess.run([sys.executable] + [str(item) for item in cmd], capture_output=True, text=True)
        return (proc.stdout or "") + (proc.stderr or ""), proc.returncode

    for left, right in pairs(names):
        if not (info.get(left, {}).get("ok") and info.get(right, {}).get("ok")):
            log("skip compare %s vs %s (missing summary.json)" % (left, right))
            continue
        text, rc = run(HERE / "profile_compare.py", left, right)
        write(out, "cmp_%s_vs_%s.txt" % (left, right), text, echo=True)
        if rc:
            log("WARNING profile_compare %s vs %s rc=%s" % (left, right, rc))
    targets = [n for n in names if "cur_cp1" in n and "a2a" not in n] + [n for n in names if "cur_cp0" in n and "repeat" not in n]
    for name in targets:
        if not info.get(name, {}).get("traces"):
            log("skip order for %s (raw traces already removed)" % name)
            continue
        for extra, suffix in (("--trim", ""), ("--devices", "_devices")):
            text, rc = run(HERE / "profile_order.py", name, "--rank", "rank0", extra)
            write(out, "order_%s%s.txt" % (name, suffix), text, echo=True)
            if rc:
                log("WARNING profile_order %s %s rc=%s" % (name, extra, rc))



def fingerprints(info: dict) -> str:
    lines = []
    for name, meta in info.items():
        line = ""
        if meta["log"].is_file():
            for item in meta["log"].read_text(encoding="utf-8", errors="replace").splitlines():
                if item.startswith("[cp_balance] CONFIG="):
                    line = item.strip()
                    break
        if not line:
            line = "[cp_balance] CONFIG=%s fingerprint: MISSING (%s)" % (name, meta["log"].name)
        repo = meta["cfg"].get("repo")
        head = ""
        if repo and Path(str(repo)).is_dir():
            proc = subprocess.run(["git", "-C", str(repo), "log", "-1", "--format=%h %s"], capture_output=True, text=True)
            head = (proc.stdout or "").strip()
        lines.append(line)
        lines.append("    repo=%s HEAD=%s" % (repo or "-", head or "-"))
    return chr(10).join(lines) + chr(10)


def bundle(out: Path, info: dict, name: str) -> None:
    entries = []
    for meta in info.values():
        prof_dir = meta["dir"]
        try:
            rel = prof_dir.relative_to(ROOT)
        except ValueError:
            log("note %s is outside the repo, tar it manually: %s" % (meta["cfg"].get("name"), prof_dir))
            continue
        entries += [(ROOT, str(rel / item)) for item in SMALL if (prof_dir / item).exists()]
        export = prof_dir / "export"
        if export.is_dir():
            entries += [
                (ROOT, str(rel / "export" / item.name))
                for item in sorted(export.iterdir())
                if item.is_file() and export_keep(item.name)
            ]
        if meta["root_json"].is_file():
            entries.append((ROOT, meta["root_json"].name))
    entries += [(out, item.name) for item in sorted(out.iterdir()) if item.is_file() and item.name != name]
    if not entries:
        log("nothing to bundle")
        return
    target = out / name
    with tarfile.open(target, "w:gz") as handle:
        for base, rel in entries:
            handle.add(str(base / rel), arcname=rel)
    size_mb = target.stat().st_size / 1048576.0
    log("bundle -> %s (%.1f MB, %d entries)" % (target, size_mb, len(entries)))
    log("md5    -> %s" % hashlib.md5(target.read_bytes()).hexdigest())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="*", help="profiling config names (default: every config with a summary.json)")
    parser.add_argument("--out", default="", help="output directory (default collect_<timestamp>)")
    parser.add_argument("--prune-traces", action="store_true", help="delete *_ascend_pt once the bundle is written")
    args = parser.parse_args()

    os.chdir(ROOT)
    names = args.configs or discover()
    if not names:
        log("no profiling config with data found; pass config names explicitly")
        return 1
    out = Path(args.out) if args.out else ROOT / ("collect_" + time.strftime("%m%d_%H%M"))
    out.mkdir(parents=True, exist_ok=True)
    log("out=%s" % out)
    info, inventory = scan(names)
    write(out, "inventory.txt", inventory)
    table = clean_table(info)
    if table:
        write(out, "clean_s.txt", table)
    reports(out, info, names)
    write(out, "fingerprints.txt", fingerprints(info))
    bundle(out, info, out.name + ".tgz")
    if args.prune_traces:
        for name, meta in info.items():
            if meta["ok"] and (meta["dir"] / "export").is_dir():
                freed = sum(disk_mb(item) for item in meta["traces"])
                for item in meta["traces"]:
                    shutil.rmtree(item, ignore_errors=True)
                log("pruned %s: %d trace dir(s), %.1f MB freed" % (name, len(meta["traces"]), freed))
    missing = [item for meta in info.values() for item in meta["missing"]]
    log("---- result ----")
    log("configs: %s" % ", ".join(names))
    log("missing: %s" % (", ".join(missing) if missing else "none"))
    log("send back: %s" % (out / (out.name + ".tgz")))
    return 0 if any(meta["ok"] for meta in info.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

