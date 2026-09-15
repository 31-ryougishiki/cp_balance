#!/usr/bin/env python3
"""Parse the ``*_ascend_pt`` directories written by profile_forward.py.

    python3 profile_analyse.py prof_cur_cp0 prof_cur_cp1
    python3 profile_analyse.py --force prof_cur_cp0

Step 1 runs ``torch_npu.profiler.profiler.analyse`` once on the whole profiler
directory: the official entry point walks every ``*_ascend_pt`` below it in a
parallel pool.  Do not loop it per rank in one process (ProfilerConfig is a
singleton and gets reloaded for each call).  Parsing needs CANN and cannot run
inside a daemon process, which is why the automatic parse on ``/stop_profile``
fails under ``--distributed-executor-backend mp`` and has to be redone here.

Step 2 reads ``ASCEND_PROFILER_OUTPUT/*.csv``, writes a compact ``summary.json``
next to the traces and copies the small per-rank outputs into ``export/`` so a
single directory can be downloaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

OUTPUT_DIR = "ASCEND_PROFILER_OUTPUT"
DONE_MARK = "analyse.done"
EXPORT_DIR = "export"
EXPORT_FILES = ("op_statistic.csv", "step_trace_time.csv", "api_statistic.csv", "communication.json")
COMM_HINTS = ("hccl", "hcom", "allgather", "allreduce", "reducescatter", "alltoall", "barrier")
TIME_COLUMNS = ("total time(us)", "total time", "duration(us)", "duration", "total_time")
NAME_COLUMNS = ("op type", "optype", "name", "type", "kernel name")
COUNT_COLUMNS = ("count", "calls", "num")


def norm(text: str) -> str:
    return "".join(ch for ch in str(text).lower() if ch not in " _-()%")


def pick(header: list[str], candidates: tuple[str, ...]) -> str | None:
    lookup = {norm(name): name for name in header}
    for candidate in candidates:
        if norm(candidate) in lookup:
            return lookup[norm(candidate)]
    return None


def read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = [row for row in reader]
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
    except Exception as exc:  # noqa: BLE001 - other configs and any already\n        # parsed CSVs still matter, so report and keep going.
        print("[analyse] parse failed for %s: %s" % (root, exc))


def summarize_ops(out_dir: Path) -> dict:
    """Aggregate op_statistic.csv (or kernel_details.csv) by operator name."""
    result: dict = {"ops": [], "comm": [], "op_total_us": 0.0, "source": None}
    stats = out_dir / "op_statistic.csv"
    if stats.is_file():
        header, rows = read_rows(stats)
        name_col = pick(header, NAME_COLUMNS)
        time_col = pick(header, TIME_COLUMNS)
        count_col = pick(header, COUNT_COLUMNS)
        if name_col and time_col:
            result["source"] = "op_statistic.csv"
            for row in rows:
                item = {
                    "name": str(row.get(name_col)),
                    "count": int(to_float(row.get(count_col))) if count_col else None,
                    "total_us": round(to_float(row.get(time_col)), 3),
                }
                result["ops"].append(item)
                if any(hint in norm(item["name"]) for hint in COMM_HINTS):
                    result["comm"].append(item)
    if not result["ops"]:
        details = out_dir / "kernel_details.csv"
        if details.is_file():
            header, rows = read_rows(details)
            name_col = pick(header, ("type", "name", "kernel name"))
            time_col = pick(header, TIME_COLUMNS)
            if name_col and time_col:
                result["source"] = "kernel_details.csv"
                bucket: dict[str, dict] = {}
                for row in rows:
                    name = str(row.get(name_col))
                    item = bucket.setdefault(name, {"name": name, "count": 0, "total_us": 0.0})
                    item["count"] += 1
                    item["total_us"] += to_float(row.get(time_col))
                result["ops"] = [
                    {**item, "total_us": round(item["total_us"], 3)} for item in bucket.values()
                ]
                result["comm"] = [
                    item for item in result["ops"] if any(hint in norm(item["name"]) for hint in COMM_HINTS)
                ]
    result["ops"].sort(key=lambda item: -item["total_us"])
    result["comm"].sort(key=lambda item: -item["total_us"])
    result["op_total_us"] = round(sum(item["total_us"] for item in result["ops"]), 3)
    result["ops"] = result["ops"][:60]
    return result


def summarize_steps(out_dir: Path) -> list[dict]:
    path = out_dir / "step_trace_time.csv"
    if not path.is_file():
        return []
    _header, rows = read_rows(path)
    return [{str(key): str(value) for key, value in row.items() if key is not None} for row in rows]


def step_metric(steps: list[dict], *hints, exclude=()) -> float | None:
    """Max over steps of the first column whose name matches ``hints``."""
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


def copy_exports(root: Path, ranks: list[Path]) -> list[str]:
    export = root / EXPORT_DIR
    copied = []
    for rank_dir in ranks:
        out_dir = rank_dir / OUTPUT_DIR
        for name in EXPORT_FILES:
            source = out_dir / name
            if not source.is_file():
                continue
            export.mkdir(parents=True, exist_ok=True)
            target = export / ("%s__%s" % (rank_dir.name, name))
            shutil.copyfile(source, target)
            copied.append(str(target.relative_to(root)))
    return copied


def digest(per_rank: dict, key) -> dict:
    values = {rank: key(payload) for rank, payload in per_rank.items() if key(payload) is not None}
    if not values:
        return {}
    ordered = sorted(values.values())
    return {
        "max": round(ordered[-1], 3),
        "mean": round(sum(ordered) / len(ordered), 3),
        "min": round(ordered[0], 3),
        "per_rank": values,
    }


def summarize_dir(root: Path, force: bool) -> dict:
    analyse_root(root, force)
    ranks = [path for path in sorted(root.rglob("*_ascend_pt")) if path.is_dir()]
    per_rank: dict[str, dict] = {}
    for rank_dir in ranks:
        out_dir = rank_dir / OUTPUT_DIR
        if not out_dir.is_dir():
            print("[analyse] %s: no %s" % (rank_dir.name, OUTPUT_DIR))
            continue
        payload = summarize_ops(out_dir)
        payload["steps"] = summarize_steps(out_dir)
        payload["step_computing_us"] = step_metric(payload["steps"], "computing")
        payload["step_comm_us"] = step_metric(payload["steps"], "communication")
        per_rank[rank_dir.name] = payload
        print(
            "[analyse] %-46s ops=%-3d total=%9.1fus comm=%9.1fus step_comp=%s step_comm=%s"
            % (
                rank_dir.name[-46:],
                len(payload["ops"]),
                payload["op_total_us"],
                sum(item["total_us"] for item in payload["comm"]),
                payload["step_computing_us"],
                payload["step_comm_us"],
            )
        )
    summary = {
        "profiler_dir": str(root),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rank_count": len(per_rank),
        "digest": {
            "op_total_us": digest(per_rank, lambda payload: payload.get("op_total_us")),
            "comm_total_us": digest(
                per_rank, lambda payload: sum(item["total_us"] for item in payload.get("comm", []))
            ),
            "comm_calls": digest(
                per_rank, lambda payload: sum(item["count"] or 0 for item in payload.get("comm", []))
            ),
            "step_computing_us": digest(per_rank, lambda payload: payload.get("step_computing_us")),
            "step_comm_us": digest(per_rank, lambda payload: payload.get("step_comm_us")),
        },
        "ranks": per_rank,
    }
    out = root / "summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    copied = copy_exports(root, ranks)
    print("[analyse] %d ranks -> %s" % (len(per_rank), out))
    for label, values in summary["digest"].items():
        if not values:
            print("[analyse] %-14s no data" % label)
            continue
        print(
            "[analyse] %-14s max=%.1f mean=%.1f min=%.1f"
            % (label, values["max"], values["mean"], values["min"])
        )
    if not per_rank:
        print("[analyse] %s produced no usable rank output" % root)
    print("[analyse] export -> %s (%d files; download summary.json + %s/)" % (root / EXPORT_DIR, len(copied), EXPORT_DIR))
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
