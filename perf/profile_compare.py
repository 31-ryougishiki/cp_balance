#!/usr/bin/env python3
"""Compare the summary.json files written by profile_analyse.py.

    python3 profile_compare.py prof_cur_cp0 prof_cur_cp1

One config now produces one capture window per prompt length, so the primary
table is per length, side by side:

* clean_s is the same request measured with the profiler off, i.e. the only
  honest end-to-end number; profiled_s shows what profiling itself costs.

Then the per-operator composition, aggregated over all windows, and the
collective time and call counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TOP = 15


def load(root: Path) -> dict:
    path = root / "summary.json"
    if not path.is_file():
        raise FileNotFoundError("missing %s; run profile_analyse.py first" % path)
    return json.loads(path.read_text(encoding="utf-8"))


def windows_of(summary: dict) -> dict:
    """label -> window info, falling back to the window id when unlabelled."""
    out = {}
    for window, info in (summary.get("windows") or {}).items():
        label = info.get("label") or ("window_%s" % window[-6:])
        out[label] = info
    if not out and summary.get("ranks"):
        out["all"] = {"rank_count": summary.get("rank_count"), "digest": summary.get("digest", {}), "ranks": summary["ranks"]}
    return out


def window_metric(info: dict, key: str) -> float:
    values = info.get("digest", {}).get(key) or {}
    return float(values.get("mean") or 0.0)


def order_labels(a: dict, b: dict) -> list:
    def size(label: str) -> int:
        for source in (a, b):
            target = source.get(label)
            if target and target.get("target_tokens"):
                return int(target["target_tokens"])
        return 10 ** 9
    return sorted(set(a) | set(b), key=size)


def print_lengths(labels: list, summaries: list) -> None:
    windows = [windows_of(summary) for summary in summaries]
    print("[compare] per-length windows (values are the mean over ranks)")
    header = "  %-10s %9s %6s" % ("length", "tokens", "steps")
    for label in labels:
        header += " | %11s %11s %12s %9s" % (label + " attn_us", "comm_us", "op_total_us", "clean_s")
    print(header)
    for length in order_labels(windows[0], windows[1] if len(windows) > 1 else windows[0]):
        info0 = next((w.get(length) for w in windows if w.get(length)), None)
        if info0 is None:
            continue
        audit = info0.get("audit") or {}
        row = "  %-10s %9s %6s" % (length, info0.get("prompt_tokens"), audit.get("kernel_steps"))
        for source in windows:
            info = source.get(length)
            if info is None:
                row += " | %11s %11s %12s %9s" % ("-", "-", "-", "-")
                continue
            clean = info.get("clean_wall_s")
            row += " | %11.1f %11.1f %12.1f %9s" % (
                window_metric(info, "attention_us"),
                window_metric(info, "comm_total_us"),
                window_metric(info, "op_total_us"),
                ("%.3f" % clean) if clean else "-",
            )
        print(row)


def print_length_delta(labels: list, summaries: list) -> None:
    if len(summaries) != 2:
        return
    left, right = (windows_of(summary) for summary in summaries)
    print("[compare] per-length delta %s -> %s (absolute times and the unprofiled wall clock)" % tuple(labels[:2]))
    print(
        "  %-10s %12s %12s %10s | %12s %12s %10s"
        % ("length", "A attn_us", "B attn_us", "delta%", "A clean_s", "B clean_s", "delta%")
    )
    for length in order_labels(left, right):
        a, b = left.get(length), right.get(length)
        if not a or not b:
            continue
        aa, ab = window_metric(a, "attention_us"), window_metric(b, "attention_us")
        ca, cb = a.get("clean_wall_s"), b.get("clean_wall_s")
        attn = ("%+.1f%%" % (100.0 * (ab / aa - 1))) if aa > 0 else "-"
        clean = ("%+.1f%%" % (100.0 * (cb / ca - 1))) if ca and cb else "-"
        print(
            "  %-10s %12.1f %12.1f %10s | %12s %12s %10s"
            % (length, aa, ab, attn, "%.3f" % ca if ca else "-", "%.3f" % cb if cb else "-", clean)
        )


def rank_union(summary: dict, field: str) -> dict:
    totals = {}
    for info in (summary.get("windows") or {}).values():
        for payload in info.get("ranks", {}).values():
            for item in payload.get(field, []):
                totals[item["name"]] = totals.get(item["name"], 0.0) + item["total_us"]
    if not totals:
        for payload in (summary.get("ranks") or {}).values():
            for item in payload.get(field, []):
                totals[item["name"]] = totals.get(item["name"], 0.0) + item["total_us"]
    return totals


def rank_union_count(summary: dict, field: str) -> dict:
    totals = {}
    for info in (summary.get("windows") or {}).values():
        for payload in info.get("ranks", {}).values():
            for item in payload.get(field, []):
                totals[item["name"]] = totals.get(item["name"], 0) + (item["count"] or 0)
    if not totals:
        for payload in (summary.get("ranks") or {}).values():
            for item in payload.get(field, []):
                totals[item["name"]] = totals.get(item["name"], 0) + (item["count"] or 0)
    return totals


def print_ops(labels: list, summaries: list, top: int = TOP) -> None:
    tables = [rank_union(summary, "ops") for summary in summaries]
    names = sorted(set().union(*[set(t) for t in tables]), key=lambda n: -max(t.get(n, 0.0) for t in tables))
    print("[compare] operator time over all ranks and windows, top %d by max" % top)
    print("  %-42s %s" % ("op", " ".join("%14s" % label for label in labels)))
    for name in names[:top]:
        print("  %-42s %s" % (name[:42], " ".join("%14.0f" % table.get(name, 0.0) for table in tables)))
    counts = [rank_union_count(summary, "ops") for summary in summaries]
    print("[compare] operator call count over all ranks and windows")
    print("  %-42s %s" % ("op", " ".join("%14s" % label for label in labels)))
    for name in names[:top]:
        print("  %-42s %s" % (name[:42], " ".join("%14d" % table.get(name, 0) for table in counts)))


def print_comm(labels: list, summaries: list) -> None:
    tables = [rank_union(summary, "comm") for summary in summaries]
    counts = [rank_union_count(summary, "comm") for summary in summaries]
    names = sorted(set().union(*[set(t) for t in tables]), key=lambda n: -max(t.get(n, 0.0) for t in tables))
    if not names:
        return
    print("[compare] collective time / count over all ranks and windows")
    print("  %-40s %s" % ("op", " ".join("%19s" % label for label in labels)))
    for name in names:
        cells = " ".join("%9.0f/%-8d" % (table.get(name, 0.0), count.get(name, 0)) for table, count in zip(tables, counts))
        print("  %-40s %s" % (name[:40], cells))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", help="profiler dirs to compare, in display order")
    args = parser.parse_args()
    labels = []
    summaries = []
    for name in args.dirs:
        root = Path(name).resolve()
        label = root.name
        try:
            summaries.append(load(root))
        except FileNotFoundError as exc:
            print("[compare] SKIP %s: %s" % (label, exc))
            return 1
        labels.append(label)
        print("[compare] %s -> %s windows=%d" % (label, root, len(summaries[-1].get("windows") or {})))
    print_lengths(labels, summaries)
    print_length_delta(labels, summaries)
    print_ops(labels, summaries)
    print_comm(labels, summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
