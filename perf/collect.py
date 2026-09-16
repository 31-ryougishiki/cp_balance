#!/usr/bin/env python3
"""One-shot collector for a profiling round: inventory -> text reports ->
fingerprints -> bundle.

    bash perf/collect.sh                                  # auto: every config with a summary.json
    bash perf/collect.sh prof_cur_cp0 prof_cur_cp1        # explicit subset
    bash perf/collect.sh --out /tmp/collect_a3 --no-tar   # keep the files unpacked
    bash perf/collect.sh --prune-traces                    # delete the raw traces afterwards

What it does, in order:

  1. inventory   - which of summary.json / windows.json / export/ / prof_<name>.json /
                   profile_<name>.log exist per config (and how big the raw traces are)
  2. clean_s     - the unprofiled wall time per prompt length, one table for all configs
  3. reports     - profile_compare.py for every meaningful pair and profile_order.py
                   (--trim + --devices) for the zigzag and the reference config
  4. fingerprints- the [cp_balance] CONFIG= line of every service log plus the git HEAD
                   of the code tree each config used
  5. bundle      - a .tgz with the small files only (never the *_ascend_pt traces),
                   plus its size and md5

Exit code is 1 only when the reference config has no summary.json (nothing to analyse).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

CONFIG_DIR = ROOT / "configs"
EXPORT_DIR = "export"
TEXT_FILES = ("inventory.txt", "clean_s.txt", "fingerprints.txt")


class Report:
    def __init__(self, out: Path | None) -> None:
        self.out = out
        self.summary = (out / "collect_summary.txt").open("w", encoding="utf-8") if out is not None else None
        self.missing: list[str] = []

    def log(self, msg: str = "") -> None:
        print("[collect] " + msg if msg else "", flush=True)
        if self.summary is not None:
            self.summary.write(("[collect] " + msg if msg else "") + chr(10))
            self.summary.flush()

    def save(self, name: str, text: str) -> Path:
        if self.out is None:
            return ROOT / name
        path = self.out / name
        path.write_text(text, encoding="utf-8")
        self.log("wrote %s (%d lines)" % (path.name, text.count(chr(10))))
        return path

    def close(self) -> None:
        if self.summary is not None:
            self.summary.close()



def discover(explicit: list, rep: Report) -> list:
    """Config names to collect: the explicit list, or every profiler config with data."""
    if explicit:
        return explicit
    found = []
    for path in sorted(CONFIG_DIR.glob("*.json")):
        try:
            cfg = serve_config.load_config(path.name)
        except (SystemExit, ValueError):
            continue
        if not (cfg.get("profiler") or {}).get("enabled"):
            continue
        name = str(cfg.get("name") or path.stem)
        if (Path(serve_config.profiler_dir(cfg)) / "summary.json").is_file():
            found.append(name)
    rep.log("auto-discovered %d config(s) with a summary.json: %s" % (len(found), ", ".join(found) or "-"))
    return found


def dir_size_mb(path: Path) -> float:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total / (1024.0 * 1024.0)


def inventory(names: list, rep: Report) -> dict:
    info = {}
    lines = []
    for name in names:
        cfg = serve_config.load_config(name)
        prof_dir = Path(serve_config.profiler_dir(cfg))
        root_json = ROOT / ("prof_%s.json" % cfg.get("name"))
        service_log = ROOT / ("profile_%s.log" % cfg.get("name"))
        checks = [
            ("summary.json", (prof_dir / "summary.json").is_file()),
            ("windows.json", (prof_dir / "windows.json").is_file()),
            ("export/", (prof_dir / EXPORT_DIR).is_dir()),
            ("order_rank0.json", (prof_dir / "order_rank0.json").is_file()),
            (root_json.name, root_json.is_file()),
            (service_log.name, service_log.is_file()),
        ]
        traces = [item for item in prof_dir.rglob("*_ascend_pt") if item.is_dir()] if prof_dir.is_dir() else []
        trace_mb = sum(dir_size_mb(item) for item in traces)
        info[name] = {
            "cfg": cfg,
            "dir": prof_dir,
            "root_json": root_json,
            "service_log": service_log,
            "traces": traces,
            "trace_mb": trace_mb,
            "ok": {label: flag for label, flag in checks},
            "summary": (prof_dir / "summary.json").is_file(),
        }
        lines.append("%-22s dir=%s" % (name, prof_dir))
        for label, flag in checks:
            if not flag:
                rep.missing.append("%s: %s" % (name, label))
            lines.append("    %-4s %s" % ("OK" if flag else "MISS", label))
        lines.append("    %-4s raw traces: %d dirs, %.1f MB (never bundled)" % ("info", len(traces), trace_mb))
    rep.save("inventory.txt", chr(10).join(lines) + chr(10))
    return info


def clean_s_table(info: dict, rep: Report) -> None:
    rows = []
    header = None
    for name, meta in info.items():
        path = meta["root_json"]
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            rep.log("WARNING %s is not valid JSON (%s)" % (path.name, exc))
            continue
        entries = data.get("entries") or []
        line = {}
        for entry in entries:
            length = entry.get("target_tokens")
            if length is None:
                continue
            line[int(length)] = entry
        if not line:
            continue
        header = header or sorted(line)
        rows.append((name, line, data.get("mean_elapsed_s")))
    if not rows:
        rep.log("no prof_<name>.json found, skip the clean_s table")
        return
    lengths = sorted({item for _name, line, _mean in rows for item in line})
    text = ["%-22s %10s %12s %12s %10s" % ("config", "length", "profiled_s", "clean_s", "tokens")]
    for name, line, mean in rows:
        for length in lengths:
            entry = line.get(length)
            if entry is None:
                continue
            text.append(
                "%-22s %10d %12s %12s %10s"
                % (
                    name,
                    length,
                    entry.get("wall_s", "-"),
                    entry.get("clean_wall_s", "-"),
                    entry.get("clean_prompt_tokens", entry.get("prompt_tokens", "-")),
                )
            )
        text.append("%-22s %10s %12s" % (name, "mean", mean if mean is not None else "-"))
    rep.save("clean_s.txt", chr(10).join(text) + chr(10))



def pairs_for(names: list, rep: Report) -> list:
    """Compare the CP_BALANCE=0 reference against everything else, plus cp1 vs a2a."""
    reference = ""
    for name in names:
        if "cur_cp0" in name and "repeat" not in name:
            reference = name
            break
    if not reference and names:
        reference = names[0]
    pairs = []
    if reference:
        for name in names:
            if name != reference:
                pairs.append((reference, name))
    for name in names:
        if name.endswith("_cp1"):
            sibling = name + "_a2a"
            if sibling in names:
                pairs.append((name, sibling))
    rep.log("reference=%s, %d compare pair(s)" % (reference or "-", len(pairs)))
    return pairs


def reports(info: dict, names: list, rep: Report) -> None:
    pairs = pairs_for(names, rep)
    for left, right in pairs:
        if not (info.get(left, {}).get("summary") and info.get(right, {}).get("summary")):
            rep.log("skip compare %s vs %s (missing summary.json)" % (left, right))
            continue
        label = "cmp_%s_vs_%s.txt" % (left, right)
        proc = subprocess.run(
            [sys.executable, str(HERE / "profile_compare.py"), left, right],
            capture_output=True,
            text=True,
        )
        text = (proc.stdout or "") + (proc.stderr or "")
        rep.save(label, text)
        if proc.returncode != 0:
            rep.log("WARNING profile_compare %s vs %s rc=%s" % (left, right, proc.returncode))
    order_targets = [name for name in names if "cur_cp1" in name and "a2a" not in name]
    order_targets += [name for name in names if "cur_cp0" in name and "repeat" not in name]
    for name in order_targets:
        meta = info.get(name) or {}
        if not meta.get("traces"):
            rep.log("skip order for %s (raw traces already removed)" % name)
            continue
        for extra, suffix in ((["--trim"], ""), (["--devices"], "_devices")):
            proc = subprocess.run(
                [sys.executable, str(HERE / "profile_order.py"), name, "--rank", "rank0"] + extra,
                capture_output=True,
                text=True,
            )
            text = (proc.stdout or "") + (proc.stderr or "")
            rep.save("order_%s%s.txt" % (name, suffix), text)
            if proc.returncode != 0:
                rep.log("WARNING profile_order %s %s rc=%s" % (name, extra, proc.returncode))


def fingerprints(info: dict, rep: Report) -> None:
    lines = []
    for name, meta in info.items():
        line = ""
        path = meta["service_log"]
        if path.is_file():
            for item in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if item.startswith("[cp_balance] CONFIG="):
                    line = item.strip()
                    break
        if not line:
            line = "[cp_balance] CONFIG=%s fingerprint: MISSING (service log %s)" % (name, path.name)
            rep.missing.append("%s: fingerprint line" % name)
        head = ""
        repo = meta["cfg"].get("repo")
        if repo and Path(str(repo)).is_dir():
            proc = subprocess.run(
                ["git", "-C", str(repo), "log", "-1", "--format=%h %s"],
                capture_output=True,
                text=True,
            )
            head = (proc.stdout or "").strip()
        lines.append(line)
        lines.append("    repo=%s HEAD=%s" % (repo or "-", head or "-"))
    rep.save("fingerprints.txt", chr(10).join(lines) + chr(10))


def bundle(info: dict, rep: Report, out: Path, name: str) -> None:
    entries = []
    for cfg_name, meta in info.items():
        prof_dir = meta["dir"]
        rel = prof_dir.relative_to(ROOT) if str(prof_dir).startswith(str(ROOT)) else None
        if rel is None:
            rep.log("note %s lives outside the repo, add it to the tarball manually: %s" % (cfg_name, prof_dir))
            continue
        for item in ("summary.json", "windows.json", EXPORT_DIR, "order_rank0.json"):
            if (prof_dir / item).exists():
                entries.append((ROOT, str(rel / item)))
        if meta["root_json"].is_file():
            entries.append((ROOT, meta["root_json"].name))
    text_files = sorted(item.name for item in out.iterdir() if item.is_file() and item.name != name)
    for item in text_files:
        entries.append((out, item))
    if not entries:
        rep.log("nothing to bundle")
        return
    target = out / name
    with tarfile.open(target, "w:gz") as handle:
        for base, rel in entries:
            handle.add(str(base / rel), arcname=rel)
    digest = hashlib.md5(target.read_bytes()).hexdigest()
    size_mb = target.stat().st_size / (1024.0 * 1024.0)
    rep.log("bundle -> %s (%.1f MB, %d entries)" % (target, size_mb, len(entries)))
    rep.log("md5    -> %s" % digest)
    rep.log("entries: " + ", ".join(rel for _base, rel in entries))



def prune_traces(info: dict, rep: Report) -> None:
    import shutil

    for name, meta in info.items():
        if not (meta["summary"] and (meta["dir"] / EXPORT_DIR).is_dir()):
            rep.log("keep raw traces for %s (export/ or summary.json missing)" % name)
            continue
        freed = 0.0
        for item in meta["traces"]:
            freed += dir_size_mb(item)
            shutil.rmtree(item, ignore_errors=True)
        rep.log("pruned %s: %d trace dir(s), %.1f MB freed" % (name, len(meta["traces"]), freed))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="*", help="profiling config names (default: all with a summary.json)")
    parser.add_argument("--out", default="", help="output directory (default collect_<timestamp>)")
    parser.add_argument("--bundle-name", default="", help="tarball name inside --out")
    parser.add_argument("--no-reports", action="store_true", help="skip profile_compare / profile_order")
    parser.add_argument("--no-tar", action="store_true", help="collect the files but do not bundle them")
    parser.add_argument("--prune-traces", action="store_true", help="delete *_ascend_pt after bundling")
    parser.add_argument("--list", action="store_true", help="print the configs that would be collected and exit")
    args = parser.parse_args()

    os.chdir(ROOT)
    out = Path(args.out) if args.out else ROOT / ("collect_" + time.strftime("%m%d_%H%M"))
    name = args.bundle_name or (out.name + ".tgz")

    if args.list:
        names = args.configs or discover([], Report(None))
        for item in names:
            print(item)
        return 0

    out.mkdir(parents=True, exist_ok=True)
    rep = Report(out)
    rep.log("out=%s" % out)
    names = args.configs or discover([], rep)
    if not names:
        rep.log("no profiling config with data found; pass config names explicitly")
        rep.close()
        return 1
    info = inventory(names, rep)
    ref_ok = any(meta["summary"] for meta in info.values())
    clean_s_table(info, rep)
    if not args.no_reports:
        reports(info, names, rep)
    fingerprints(info, rep)
    if not args.no_tar:
        bundle(info, rep, out, name)
    if args.prune_traces:
        prune_traces(info, rep)
    rep.log("---- result ----")
    rep.log("configs: %s" % ", ".join(names))
    if rep.missing:
        rep.log("missing (%d):" % len(rep.missing))
        for item in rep.missing:
            rep.log("  " + item)
    else:
        rep.log("missing: none")
    rep.log("send back: %s" % (out / name if not args.no_tar else out))
    rep.close()
    return 0 if ref_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

