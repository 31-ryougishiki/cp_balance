#!/usr/bin/env python3
"""Parse the rank directories written by profile_forward.py.

    python3 profile_analyse.py prof_cur_cp0 prof_cur_cp1
    python3 profile_analyse.py --force prof_cur_cp0

Step 1 runs torch_npu.profiler.profiler.analyse once on the whole profiler
directory: the official entry point walks every *_ascend_pt below it in a
parallel pool.  Do not loop it per rank in one process (ProfilerConfig is a
singleton and gets reloaded for each call).  Parsing needs CANN and cannot run
inside a daemon process, which is why the automatic parse on /stop_profile fails
under --distributed-executor-backend mp and has to be redone here.

Step 2 groups the rank directories by capture window (the timestamp inside the
directory name), because one config now produces one window per prompt length.
For every window it reads ASCEND_PROFILER_OUTPUT/*.csv, writes summary.json
next to the traces and copies the small per-rank outputs into export/.

Step 3 audits each window: how many steps it actually holds.  A window is
supposed to be one prefill step (plus at most one decode), so "steps=1" is the
good case; anything larger means the composition is diluted by decode and the
window cannot be compared against a different length.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
from pathlib import Path

OUTPUT_DIR = "ASCEND_PROFILER_OUTPUT"
DONE_MARK = "analyse.done"
EXPORT_DIR = "export"
# Only the small per-rank CSVs are copied out.  communication.json /
# communication_matrix.json stay in the raw trace directory: they are orders of
# magnitude bigger, nothing in this repo parses them (HCCL time and counts come
# from the hcom_* rows of op_statistic.csv, the per-step comm/compute split from
# step_trace_time.csv), and they are never bundled.
EXPORT_FILES = ("op_statistic.csv", "step_trace_time.csv", "api_statistic.csv")
COMM_HINTS = ("hccl", "hcom", "allgather", "allreduce", "reducescatter", "alltoall", "barrier", "aicpukernel")
ATTENTION_HINTS = ("FlashAttention", "LightningIndexer")
TIME_COLUMNS = ("total time(us)", "total time", "duration(us)", "duration", "total_time")
NAME_COLUMNS = ("op type", "optype", "name", "type", "kernel name")
COUNT_COLUMNS = ("count", "calls", "num")
WINDOW_RE = re.compile(r"_(\d{17})_ascend_pt$")
def norm(text: str) -> str:
    return "".join(ch for ch in str(text).lower() if ch not in " _-()%")


def pick(header: list, candidates: tuple):
    lookup = {norm(name): name for name in header}
    for candidate in candidates:
        if norm(candidate) in lookup:
            return lookup[norm(candidate)]
    return None


def read_rows(path: Path) -> tuple:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    return header, rows


def to_float(value) -> float:
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0.0


def analyse_root(root: Path, force: bool) -> None:
    ranks = [path for path in sorted(root.rglob("*_ascend_pt")) if path.is_dir()]
    if not ranks:
        print("[analyse] no *_ascend_pt directory under %s" % root)
        return
    pending = [path for path in ranks if force or not (path / OUTPUT_DIR / DONE_MARK).is_file()]
    print("[analyse] %d rank dirs, %d need parsing" % (len(ranks), len(pending)))
    if not pending:
        return
    try:
        from torch_npu.profiler.profiler import analyse
    except Exception as exc:  # noqa: BLE001 - already parsed data stays usable
        print("[analyse] torch_npu unavailable (%s); using existing CSVs" % exc)
        return
    try:
        analyse(str(root) + "/")
    except Exception as exc:  # noqa: BLE001 - other windows and configs still matter
        print("[analyse] parse failed for %s: %s" % (root, exc))


def summarize_ops(out_dir: Path, limit: int = 60) -> dict:
    """Aggregate op_statistic.csv (or kernel_details.csv) by operator name."""
    result = {"ops": [], "comm": [], "attention_us": 0.0, "op_total_us": 0.0, "source": None}
    stats = out_dir / "op_statistic.csv"
    if stats.is_file():
        header, rows = read_rows(stats)
        name_col = pick(header, NAME_COLUMNS)
        time_col = pick(header, TIME_COLUMNS)
        count_col = pick(header, COUNT_COLUMNS)
        if name_col and time_col:
            result["source"] = "op_statistic.csv"
            for row in rows:
                name = str(row.get(name_col))
                total = to_float(row.get(time_col))
                item = {
                    "name": name,
                    "count": int(to_float(row.get(count_col))) if count_col else None,
                    "total_us": round(total, 3),
                }
                result["ops"].append(item)
                if any(hint in norm(name) for hint in COMM_HINTS):
                    result["comm"].append(item)
                if any(hint in name for hint in ATTENTION_HINTS):
                    result["attention_us"] += total
    if not result["ops"]:
        details = out_dir / "kernel_details.csv"
        if details.is_file():
            header, rows = read_rows(details)
            name_col = pick(header, ("type", "name", "kernel name"))
            time_col = pick(header, TIME_COLUMNS)
            if name_col and time_col:
                result["source"] = "kernel_details.csv"
                bucket = {}
                for row in rows:
                    name = str(row.get(name_col))
                    item = bucket.setdefault(name, {"name": name, "count": 0, "total_us": 0.0})
                    item["count"] += 1
                    item["total_us"] += to_float(row.get(time_col))
                result["ops"] = [{**item, "total_us": round(item["total_us"], 3)} for item in bucket.values()]
                result["comm"] = [x for x in result["ops"] if any(h in norm(x["name"]) for h in COMM_HINTS)]
    result["ops"].sort(key=lambda item: -item["total_us"])
    result["comm"].sort(key=lambda item: -item["total_us"])
    result["op_total_us"] = round(sum(item["total_us"] for item in result["ops"]), 3)
    result["attention_us"] = round(result["attention_us"], 3)
    result["ops"] = result["ops"][:limit]
    return result


def summarize_steps(out_dir: Path) -> list:
    path = out_dir / "step_trace_time.csv"
    if not path.is_file():
        return []
    _header, rows = read_rows(path)
    return [{str(k): str(v) for k, v in row.items() if k is not None} for row in rows]


def step_metric(steps: list, *hints, exclude=()) -> float:
    best = None
    for row in steps:
        value = None
        for key, raw in row.items():
            low = key.lower()
            if any(hint in low for hint in hints) and not any(bad in low for bad in exclude):
                value = to_float(raw)
                break
        if value is None:
            continue
        best = value if best is None else max(best, value)
    return best


def audit_window(rank_dir: Path) -> dict:
    """How many steps the window holds, from the kernel steps if available."""
    steps = summarize_steps(rank_dir / OUTPUT_DIR)
    audit = {"step_trace_rows": len(steps), "kernel_steps": None, "busiest_step": None}
    path = rank_dir / OUTPUT_DIR / "kernel_details.csv"
    if not path.is_file():
        return audit
    counts = {}
    total = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = {key.lower(): key for key in (reader.fieldnames or [])}
        step_col = header.get("step")
        dur_col = header.get("duration(us)") or header.get("duration")
        if not step_col or not dur_col:
            return audit
        for row in reader:
            key = str(row.get(step_col))
            counts[key] = counts.get(key, 0) + 1
            total[key] = total.get(key, 0.0) + to_float(row.get(dur_col))
    audit["kernel_steps"] = len(counts)
    if counts:
        busiest = max(counts, key=lambda k: counts[k])
        audit["busiest_step"] = busiest
        audit["busiest_kernels"] = counts[busiest]
        audit["busiest_duration_us"] = round(total[busiest], 3)
        audit["per_step_duration_us"] = {k: round(v, 3) for k, v in sorted(total.items())}
    return audit


def copy_exports(root: Path, ranks: list) -> int:
    export = root / EXPORT_DIR
    copied = 0
    for rank_dir in ranks:
        out_dir = rank_dir / OUTPUT_DIR
        for name in EXPORT_FILES:
            source = out_dir / name
            if not source.is_file():
                continue
            export.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, export / ("%s__%s" % (rank_dir.name, name)))
            copied += 1
    return copied


def digest(per_rank: dict, key) -> dict:
    values = {rank: key(payload) for rank, payload in per_rank.items() if key(payload)}
    if not values:
        return {}
    ordered = sorted(values.values())
    return {
        "max": round(ordered[-1], 3),
        "mean": round(sum(ordered) / len(ordered), 3),
        "min": round(ordered[0], 3),
        "per_rank": values,
    }


def load_windows_meta(root: Path) -> dict:
    path = root / "windows.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    meta = {}
    for entry in data.get("entries") or []:
        for window in entry.get("window_ids") or []:
            meta[str(window)] = {
                "label": entry.get("label"),
                "target_tokens": entry.get("target_tokens"),
                "prompt_tokens": entry.get("prompt_tokens"),
                "wall_s": entry.get("wall_s"),
                "clean_wall_s": entry.get("clean_wall_s"),
            }
    return meta


def summarize_window(window: str, ranks: list, meta: dict) -> dict:
    per_rank = {}
    for rank_dir in ranks:
        out_dir = rank_dir / OUTPUT_DIR
        if not out_dir.is_dir():
            continue
        payload = summarize_ops(out_dir)
        payload["steps"] = summarize_steps(out_dir)
        payload["step_computing_us"] = step_metric(payload["steps"], "computing")
        payload["step_comm_us"] = step_metric(payload["steps"], "communication")
        per_rank[rank_dir.name] = payload
    info = dict(meta.get(window) or {})
    info["window"] = window
    info["rank_count"] = len(per_rank)
    info["digest"] = {
        "op_total_us": digest(per_rank, lambda p: p.get("op_total_us")),
        "comm_total_us": digest(per_rank, lambda p: sum(i["total_us"] for i in p.get("comm", []))),
        "comm_calls": digest(per_rank, lambda p: sum(i["count"] or 0 for i in p.get("comm", []))),
        "attention_us": digest(per_rank, lambda p: p.get("attention_us")),
        "step_computing_us": digest(per_rank, lambda p: p.get("step_computing_us")),
        "step_comm_us": digest(per_rank, lambda p: p.get("step_comm_us")),
    }
    first = sorted(per_rank)[0] if per_rank else None
    if first:
        info["audit"] = audit_window(Path(str(ranks[0].parent)) / first)
    info["ranks"] = per_rank
    return info


def summarize_dir(root: Path, force: bool) -> dict:
    analyse_root(root, force)
    ranks = [path for path in sorted(root.rglob("*_ascend_pt")) if path.is_dir()]
    groups = {}
    for rank_dir in ranks:
        match = WINDOW_RE.search(rank_dir.name)
        key = match.group(1) if match else "unknown"
        groups.setdefault(key, []).append(rank_dir)
    meta = load_windows_meta(root)
    windows = {}
    for window, group in groups.items():
        info = summarize_window(window, group, meta)
        windows[window] = info
        label = info.get("label") or "?"
        audit = info.get("audit") or {}
        print(
            "[analyse] %-18s %-10s ranks=%-3d steps=%-4s op_total=%9.1fus attention=%8.1fus comm=%9.1fus"
            % (
                window,
                label,
                info["rank_count"],
                audit.get("kernel_steps"),
                info["digest"]["op_total_us"].get("mean", 0.0),
                info["digest"]["attention_us"].get("mean", 0.0),
                info["digest"]["comm_total_us"].get("mean", 0.0),
            )
        )
        if audit.get("kernel_steps") not in (None, 1):
            print(
                "[analyse]   WARNING window %s holds %s steps; only a one-step window is prefill-only"
                % (window, audit.get("kernel_steps"))
            )
    summary = {
        "profiler_dir": str(root),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "lengths" if meta else "single",
        "windows": windows,
    }
    all_ranks = {rank: info for info in windows.values() for rank, info in info["ranks"].items()}
    summary["rank_count"] = len(all_ranks)
    summary["digest"] = {
        "op_total_us": digest(all_ranks, lambda p: p.get("op_total_us")),
        "comm_total_us": digest(all_ranks, lambda p: sum(i["total_us"] for i in p.get("comm", []))),
    }
    out = root / "summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    copied = copy_exports(root, ranks)
    print("[analyse] %d windows, %d rank dirs -> %s" % (len(windows), len(all_ranks), out))
    print("[analyse] export -> %s (%d files)" % (root / EXPORT_DIR, copied))
    if not all_ranks:
        print("[analyse] %s produced no usable rank output" % root)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", help="profiler dir printed by profile_forward.py")
    parser.add_argument("--force", action="store_true", help="re-run torch_npu analyse")
    args = parser.parse_args()
    for name in args.dirs:
        summarize_dir(Path(name).resolve(), args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
