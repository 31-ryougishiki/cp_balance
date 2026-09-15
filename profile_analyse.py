#!/usr/bin/env python3
"""Parse the ``*_ascend_pt`` directories written by profile_forward.py.

    python3 profile_analyse.py prof_cur_cp0 prof_cur_cp1
    python3 profile_analyse.py --force prof_cur_cp0

Step 1 runs ``torch_npu.profiler.profiler.analyse`` on every rank directory
(needs the NPU environment).  Step 2 reads the resulting
``ASCEND_PROFILER_OUTPUT/*.csv`` and writes a compact ``summary.json`` next to
the traces, so only a few KB have to be copied off the machine.

Every rank directory gets its own digest and the ranks are also compared
against each other: cp_balance is a load-balancing change, so the spread of
per-rank step time matters as much as the absolute number.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

OUTPUT_DIR = "ASCEND_PROFILER_OUTPUT"
COMM_HINTS = ("hccl", "hcom", "allgather", "allreduce", "reducescatter", "alltoall", "barrier")
TIME_COLUMNS = (
    "total time(us)",
    "total time",
    "duration(us)",
    "duration",
    "total_time",
)
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


def analyse_rank(rank_dir: Path, force: bool) -> Path | None:
    """Run torch_npu analyse; returns the ASCEND_PROFILER_OUTPUT directory."""
    out = rank_dir / OUTPUT_DIR
    if out.is_dir() and not force:
        return out
    try:
        from torch_npu.profiler.profiler import analyse
    except Exception as exc:  # noqa: BLE001 - offline inspection is still useful
        print("[analyse] torch_npu unavailable (%s); using existing CSVs" % exc)
        return out if out.is_dir() else None
    print("[analyse] %s" % rank_dir)
    analyse(str(rank_dir) + "/")
    return out if out.is_dir() else None


def summarize_ops(out_dir: Path) -> dict:
    """Aggregate op_statistic.csv / kernel_details.csv by operator name."""
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
                total = to_float(row.get(time_col))
                item = {
                    "name": str(row.get(name_col)),
                    "count": int(to_float(row.get(count_col))) if count_col else None,
                    "total_us": round(total, 3),
                }
                result["ops"].append(item)
                if any(hint in norm(item["name"]) for hint in COMM_HINTS):
                    result["comm"].append(item)
    if not result["ops"]:
        details = out_dir / "kernel_details.csv"
        if details.is_file():
            header, rows = read_rows(details)
            name_col = pick(header, ("type", "name", "kernel name"))
            time_col = pick(header, ("duration(us)", "duration", "total time(us)", "total time"))
            if name_col and time_col:
                result["source"] = "kernel_details.csv"
                bucket: dict[str, dict] = {}
                for row in rows:
                    name = str(row.get(name_col))
                    item = bucket.setdefault(name, {"name": name, "count": 0, "total_us": 0.0})
                    item["count"] += 1
                    item["total_us"] += to_float(row.get(time_col))
                result["ops"] = [
                    {**item, "total_us": round(item["total_us"], 3)}
                    for item in bucket.values()
                ]
                result["comm"] = [
                    item
                    for item in result["ops"]
                    if any(hint in norm(item["name"]) for hint in COMM_HINTS)
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
    header, rows = read_rows(path)
    steps = []
    for row in rows:
        steps.append({str(key): str(value) for key, value in row.items() if key is not None})
    return steps


def summarize_kernels(out_dir: Path, limit: int) -> list[dict]:
    path = out_dir / "kernel_details.csv"
    if not path.is_file():
        return []
    header, rows = read_rows(path)
    name_col = pick(header, ("type", "name", "kernel name"))
    time_col = pick(header, ("duration(us)", "duration", "total time(us)", "total time"))
    step_col = pick(header, ("step",))
    if not name_col or not time_col:
        return []
    bucket: dict[tuple, dict] = {}
    for row in rows:
        key = (str(row.get(step_col)) if step_col else "?", str(row.get(name_col)))
        item = bucket.setdefault(
            key, {"step": key[0], "name": key[1], "count": 0, "total_us": 0.0}
        )
        item["count"] += 1
        item["total_us"] += to_float(row.get(time_col))
    ordered = sorted(bucket.values(), key=lambda item: -item["total_us"])
    for item in ordered:
        item["total_us"] = round(item["total_us"], 3)
    return ordered[:limit]


def digest(per_rank: dict, key) -> dict:
    values = {
        rank: key(payload) for rank, payload in per_rank.items() if key(payload) is not None
    }
    if not values:
        return {}
    ordered = sorted(values.values())
    return {
        "max": round(ordered[-1], 3),
        "mean": round(sum(ordered) / len(ordered), 3),
        "min": round(ordered[0], 3),
        "per_rank": values,
    }


def summarize_dir(root: Path, force: bool, with_kernels: int) -> dict:
    ranks = sorted(
        path for path in root.rglob("*_ascend_pt") if path.is_dir()
    )
    if not ranks:
        print("[analyse] no *_ascend_pt directory under %s" % root)
        return {"profiler_dir": str(root), "ranks": {}}
    per_rank: dict[str, dict] = {}
    for rank_dir in ranks:
        out_dir = analyse_rank(rank_dir, force)
        if out_dir is None:
            print("[analyse] %s: no %s" % (rank_dir, OUTPUT_DIR))
            continue
        payload = summarize_ops(out_dir)
        payload["steps"] = summarize_steps(out_dir)
        if with_kernels:
            payload["top_kernels"] = summarize_kernels(out_dir, with_kernels)
        per_rank[rank_dir.name] = payload
        print(
            "[analyse] %-40s ops=%d total=%sus comm=%sus"
            % (
                rank_dir.name[-40:],
                len(payload["ops"]),
                payload["op_total_us"],
                round(sum(item["total_us"] for item in payload["comm"]), 3),
            )
        )
    summary = {
        "profiler_dir": str(root),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rank_count": len(per_rank),
        "digest": {
            "op_total_us": digest(per_rank, lambda payload: payload.get("op_total_us")),
            "comm_total_us": digest(
                per_rank,
                lambda payload: sum(item["total_us"] for item in payload.get("comm", [])),
            ),
        },
        "ranks": per_rank,
    }
    out = root / "summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[analyse] %d ranks -> %s" % (len(per_rank), out))
    for label, values in summary["digest"].items():
        print("[analyse] %-14s max=%.1f mean=%.1f min=%.1f" % (label, values["max"], values["mean"], values["min"]))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", help="profiler dir printed by profile_forward.py")
    parser.add_argument("--force", action="store_true", help="re-run torch_npu analyse")
    parser.add_argument("--with-kernels", type=int, default=0, help="top N kernels per rank (slow)")
    args = parser.parse_args()
    for name in args.dirs:
        summarize_dir(Path(name).resolve(), args.force, args.with_kernels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
