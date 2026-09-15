#!/usr/bin/env python3
"""Compare the summary.json files written by profile_analyse.py.

    python3 profile_compare.py prof_cur_cp0 prof_cur_cp1

Prints, side by side:

* per-rank operator time spread (cp_balance is a balancing change, so the
  max/mean gap across ranks is the first thing to look at);
* the per-operator composition difference (which kernel got cheaper / more
  expensive when zigzag is on);
* the collective (HCCL) time and call count, which is what the owner
  independent reduction adds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TOP = 15


def load(root: Path) -> dict:
    path = root / "summary.json"
    if not path.is_file():
        raise SystemExit("missing %s; run profile_analyse.py first" % path)
    return json.loads(path.read_text(encoding="utf-8"))


def rank_union(summary: dict, field: str) -> dict:
    totals: dict[str, float] = {}
    for payload in summary["ranks"].values():
        for item in payload.get(field, []):
            totals[item["name"]] = totals.get(item["name"], 0.0) + item["total_us"]
    return totals


def rank_union_count(summary: dict, field: str) -> dict:
    totals: dict[str, int] = {}
    for payload in summary["ranks"].values():
        for item in payload.get(field, []):
            totals[item["name"]] = totals.get(item["name"], 0) + (item["count"] or 0)
    return totals


def print_spread(labels: list[str], summaries: list[dict]) -> None:
    metrics = (
        ("step_computing_us", "per-rank step Computing (max over steps)"),
        ("step_comm_us", "per-rank step Communication (max over steps)"),
        ("op_total_us", "per-rank operator time over the whole window"),
        ("comm_total_us", "per-rank collective time"),
        ("comm_calls", "per-rank collective call count"),
    )
    for key, title in metrics:
        if not all(summary["digest"].get(key) for summary in summaries):
            continue
        print("[compare] %s" % title)
        print("  %-18s %12s %12s %12s %10s" % ("dir", "max", "mean", "min", "max/mean"))
        for label, summary in zip(labels, summaries):
            values = summary["digest"][key]
            print(
                "  %-18s %12.1f %12.1f %12.1f %10.3f"
                % (label, values["max"], values["mean"], values["min"], values["max"] / max(values["mean"], 1e-9))
            )


def print_ops(labels: list[str], summaries: list[dict]) -> None:
    tables = [rank_union(summary, "ops") for summary in summaries]
    names = sorted(set().union(*[set(table) for table in tables]), key=lambda name: -max(t.get(name, 0.0) for t in tables))
    print("[compare] operator time over all ranks, top %d by max" % TOP)
    print("  %-52s %s" % ("op", " ".join("%12s" % label for label in labels)))
    for name in names[:TOP]:
        print(
            "  %-52s %s"
            % (name[:52], " ".join("%12.1f" % table.get(name, 0.0) for table in tables))
        )
    print("[compare] operator call count over all ranks")
    counts = [rank_union_count(summary, "ops") for summary in summaries]
    print("  %-52s %s" % ("op", " ".join("%12s" % label for label in labels)))
    for name in names[:TOP]:
        print(
            "  %-52s %s"
            % (name[:52], " ".join("%12d" % table.get(name, 0) for table in counts))
        )


def print_comm(labels: list[str], summaries: list[dict]) -> None:
    tables = [rank_union(summary, "comm") for summary in summaries]
    counts = [rank_union_count(summary, "comm") for summary in summaries]
    names = sorted(set().union(*[set(table) for table in tables]), key=lambda name: -max(t.get(name, 0.0) for t in tables))
    print("[compare] collective time / count over all ranks")
    print("  %-44s %s" % ("op", " ".join("%19s" % label for label in labels)))
    for name in names:
        cells = " ".join("%9.1f/%-8d" % (table.get(name, 0.0), count.get(name, 0)) for table, count in zip(tables, counts))
        print("  %-44s %s" % (name[:44], cells))


def print_delta(labels: list[str], summaries: list[dict]) -> None:
    if len(summaries) != 2:
        return
    base = rank_union(summaries[0], "ops")
    new = rank_union(summaries[1], "ops")
    rows = []
    for name in sorted(set(base) | set(new), key=lambda key: -abs(new.get(key, 0.0) - base.get(key, 0.0))):
        before, after = base.get(name, 0.0), new.get(name, 0.0)
        rows.append((name, before, after, after - before, (after / before - 1) * 100 if before else float("inf")))
    print("[compare] delta %s -> %s (all ranks, us)" % (labels[0], labels[1]))
    print("  %-44s %12s %12s %12s %9s" % ("op", labels[0], labels[1], "delta", "delta%"))
    for name, before, after, diff, pct in rows[:TOP]:
        print("  %-44s %12.1f %12.1f %+12.1f %9.1f" % (name[:44], before, after, diff, pct))


def step_table(summary: dict) -> list[tuple]:
    """Normalize step_trace_time rows to (step, computing, comm, overlapped, free)."""
    if not summary.get("ranks"):
        return []
    rank = sorted(summary["ranks"])[0]
    rows = summary["ranks"][rank].get("steps") or []
    table = []
    for row in rows:
        def find(*hints, exclude=()):
            for key, value in row.items():
                low = key.lower()
                if any(hint in low for hint in hints) and not any(bad in low for bad in exclude):
                    return value
            return ""
        step = find("step")
        if step == "":
            continue
        table.append(
            (
                step,
                find("computing"),
                find("communication"),
                find("overlap", exclude=("not",)),
                find("free"),
            )
        )
    return table


def print_steps(labels: list[str], summaries: list[dict]) -> None:
    tables = [step_table(summary) for summary in summaries]
    if not any(tables):
        return
    print("[compare] step_trace_time of the lowest-rank worker (us)")
    for label, summary, table in zip(labels, summaries, tables):
        rank = sorted(summary["ranks"])[0] if summary.get("ranks") else "?"
        print("  %s rank=%s" % (label, rank))
        print("    %6s %14s %14s %14s %14s" % ("step", "computing", "comm", "overlapped", "free"))
        for row in table[:16]:
            print("    %6s %14s %14s %14s %14s" % row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", help="profiler dirs to compare, in display order")
    args = parser.parse_args()
    labels = []
    summaries = []
    for name in args.dirs:
        root = Path(name).resolve()
        label = root.name
        labels.append(label)
        summaries.append(load(root))
        print("[compare] %s -> %s ranks=%d" % (label, root, summaries[-1].get("rank_count", 0)))
    print_spread(labels, summaries)
    print_ops(labels, summaries)
    print_comm(labels, summaries)
    print_delta(labels, summaries)
    print_steps(labels, summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
