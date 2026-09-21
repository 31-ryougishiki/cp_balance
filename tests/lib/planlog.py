#!/usr/bin/env python3
"""Audit the [CP_BALANCE] lines of one service log (one traffic round).

    planlog.py audit <log> [--cp-size N] [--json OUT] [--min-zigzag-steps N]
                           [--min-reqs-per-step N] [--quiet]

Exit codes: 0 = gates pass, 1 = a gate failed, 77 = no usable evidence.

Why a log audit: every zigzag batch is planned independently on every rank, so
agreement between the ranks is a property only a real service run can show.  A
rank that silently falls back (gate / plan_error) while its peers shard the
batch is the failure mode these gates look for.

Ordering: worker ranks write to one shared log, so a rank that lags can
interleave with the next step.  The gates are therefore based on ordering-free
invariants (per-line arithmetic, per-shape per-rank line counts); the ordered
grouping is only used for the step count and for a possible truncated tail.

Caveats recorded on purpose:
  * [CP_BALANCE][plan] is logged with logger.info -> one line per rank per step,
    from every rank: the only per-rank evidence in the log.
  * [CP_BALANCE][branch] ... branch=CONTINUOUS uses logger.info_once with the
    default scope=local -> at most one line per (rank, reason), local first rank
    only.  Reason counts are a sample, not a histogram.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

NUM = "([0-9]+)"
IDX_HEAD = re.escape("idx=[")
Q_PREV = re.escape("] qprev=[")
Q_NEXT = re.escape("] qnext=[")
PLAN_RE = re.compile(
    re.escape("[CP_BALANCE][plan] rank=") + NUM +
    " pad=" + NUM + " actual=" + NUM + " local=" + NUM + " " + IDX_HEAD +
    "([^]]*)" + Q_PREV + "([^]]*)" + Q_NEXT + "([^]]*)" + re.escape("]")
)
BRANCH_RE = re.compile(
    re.escape("[CP_BALANCE][branch] rank=") + NUM + " branch=([A-Za-z_]+)" +
    " reason=([^ ]+?)(?: site=([A-Za-z_]+))?" + "$"
)
REDUCE_RE = re.compile(re.escape("[CP_BALANCE][reduce] path=") + "([A-Za-z_]+)")


def _ints(raw: str) -> list:
    return [int(part) for part in raw.split(",") if part.strip()]


def _scan(log: Path) -> dict:
    branch = {}
    reduce_paths = {}
    lines = []
    with log.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            hit = PLAN_RE.search(line)
            if hit:
                lines.append({
                    "rank": int(hit.group(1)),
                    "pad": int(hit.group(2)),
                    "actual": int(hit.group(3)),
                    "local": int(hit.group(4)),
                    "idx_rows": len(_ints(hit.group(5))),
                    "qprev": _ints(hit.group(6)),
                    "qnext": _ints(hit.group(7)),
                })
                continue
            hit = BRANCH_RE.search(line)
            if hit:
                key = (hit.group(2), hit.group(3), hit.group(4) or "-")
                branch[key] = branch.get(key, 0) + 1
                continue
            hit = REDUCE_RE.search(line)
            if hit:
                reduce_paths[hit.group(1)] = reduce_paths.get(hit.group(1), 0) + 1
    return {"lines": lines, "branch": branch, "reduce": reduce_paths}


def _groups(lines: list) -> list:
    """Consecutive lines with distinct ranks = one batch (best effort)."""
    out = []
    current = []
    seen = set()
    for row in lines:
        if row["rank"] in seen:
            out.append(current)
            current = []
            seen = set()
        seen.add(row["rank"])
        current.append(row)
    if current:
        out.append(current)
    return out


def _shape(row: dict) -> tuple:
    return (row["pad"], row["actual"], row["local"], len(row["qprev"]))


def _analyse(data: dict, cp_size: int) -> dict:
    lines = data["lines"]
    shapes = {}
    broken_scale = []
    broken_sum = []
    broken_len = []
    bad_idx = []
    for index, row in enumerate(lines):
        if row["local"] * cp_size != row["pad"]:
            broken_scale.append(index)
        if sum(row["qprev"]) + sum(row["qnext"]) != row["local"]:
            broken_sum.append(index)
        if len(row["qprev"]) != len(row["qnext"]) or not row["qprev"]:
            broken_len.append(index)
        if not 1 <= row["idx_rows"] <= 8:
            bad_idx.append(index)
        shape = shapes.setdefault(_shape(row), {"counts": {}, "last_index": index})
        shape["counts"][row["rank"]] = shape["counts"].get(row["rank"], 0) + 1
        shape["last_index"] = index

    groups = _groups(lines)
    last_group = groups[-1] if groups else []
    summary = {
        "plan_lines": len(lines),
        "steps": len(groups),
        "ranks_per_step": sorted({len(group) for group in groups}),
        "max_reqs_per_step": max((len(row["qprev"]) for row in lines), default=0),
        "reqs_per_step": sorted({len(row["qprev"]) for row in lines}),
        "shapes": len(shapes),
        "ranks_seen": sorted({row["rank"] for row in lines}),
        "zigzag_tokens": sum(row["actual"] for row in lines),
        "padding_tokens": sum(row["pad"] - row["actual"] for row in lines),
        "tail_group_ranks": len(last_group),
    }

    checks = []
    coverage = []

    def add(target: list, name: str, ok: bool, detail: str) -> None:
        target.append({"name": name, "ok": bool(ok), "detail": detail})

    arithmetic_ok = not (broken_scale or broken_sum or broken_len)
    add(checks, "line-arithmetic", arithmetic_ok,
        "local*cp_size==pad, sum(qprev)+sum(qnext)==local and len(qprev)==len(qnext) on every line"
        if arithmetic_ok else
        "broken lines: scale=%s block-sum=%s qprev/qnext=%s" % (broken_scale[:4], broken_sum[:4], broken_len[:4]))
    add(checks, "idx-list-shape", not bad_idx,
        "idx list is the truncated index head (1..8 entries) on every line" if not bad_idx else
        "lines with an unexpected idx list: %s" % bad_idx[:4])

    newest = max((info["last_index"] for info in shapes.values()), default=-1)
    ragged = []
    ordered_shapes = sorted(shapes.items(), key=lambda item: item[1]["last_index"])
    for shape, info in ordered_shapes:
        counts = info["counts"]
        high = max(counts.values())
        if len(counts) != cp_size or any(count < high for count in counts.values()):
            ragged.append({
                "shape": {"pad": shape[0], "actual": shape[1], "local": shape[2], "reqs": shape[3]},
                "ranks_reporting": len(counts),
                "min_lines": min(counts.values()),
                "max_lines": high,
                "tail": info["last_index"] == newest,
            })
    detail = "every batch shape was planned by all %d ranks, same number of times" % cp_size \
        if not ragged else "shapes with unequal per-rank line counts: %s" % json.dumps(ragged[:4])
    tail_only = bool(ragged) and all(item["tail"] for item in ragged) and summary["tail_group_ranks"] < cp_size
    add(coverage if tail_only else checks, "per-rank-line-counts", not ragged, detail)

    plan_errors = {"%s|%s|%s" % key: count for key, count in data["branch"].items() if "plan_error" in key[1]}
    add(checks, "no-plan_error", not plan_errors,
        "the planner rejected no batch" if not plan_errors else
        "plan_error fallbacks: %s (a rank falling back alone is the hazard)" % plan_errors)

    zigzag_lines = sum(count for key, count in data["branch"].items() if key[0] == "ZIGZAG")
    add(coverage, "zigzag-steps>=1", len(lines) > 0,
        "zigzag plan lines: %d in %d step(s), %d per-batch ZIGZAG line(s)" % (len(lines), len(groups), zigzag_lines))

    add(coverage, "reqs-per-step", True, "per-step request counts seen: %s" % summary["reqs_per_step"])
    return {"summary": summary, "checks": checks, "coverage": coverage}


def _print_report(log: Path, data: dict, analysis: dict, result: str) -> None:
    summary = analysis["summary"]
    print("[planlog] log %s" % log)
    print("[planlog] plan lines=%d steps=%d ranks/step=%s ranks seen=%s"
          % (summary["plan_lines"], summary["steps"], summary["ranks_per_step"], summary["ranks_seen"]))
    print("[planlog] requests per step=%s max=%d distinct batch shapes=%d"
          % (summary["reqs_per_step"], summary["max_reqs_per_step"], summary["shapes"]))
    print("[planlog] rank-local zigzag rows=%d padding rows=%d tail step ranks=%d"
          % (summary["zigzag_tokens"], summary["padding_tokens"], summary["tail_group_ranks"]))
    for key in sorted(data["branch"]):
        print("[planlog] branch=%s reason=%s site=%s lines=%d" % (key[0], key[1], key[2], data["branch"][key]))
    for name in sorted(data["reduce"]):
        print("[planlog] reduce path=%s lines=%d" % (name, data["reduce"][name]))
    for check in analysis["checks"]:
        print("[planlog] %-24s %s (%s)" % (check["name"], "ok" if check["ok"] else "FAIL", check["detail"]))
    for check in analysis["coverage"]:
        print("[planlog] %-24s %s (%s)" % (check["name"], "ok" if check["ok"] else "n/a", check["detail"]))
    print("[planlog] RESULT: %s" % result)


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("log")
    audit.add_argument("--cp-size", type=int, default=8)
    audit.add_argument("--json", default="")
    audit.add_argument("--min-zigzag-steps", type=int, default=1)
    audit.add_argument("--min-reqs-per-step", type=int, default=1)
    audit.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    log = Path(args.log)
    if not log.is_file():
        print("[planlog] RESULT: INCONCLUSIVE (no log at %s)" % log)
        return 77
    data = _scan(log)
    analysis = _analyse(data, args.cp_size)
    summary = analysis["summary"]
    gate = {"name": "reqs-per-step>=%d" % args.min_reqs_per_step,
            "ok": summary["max_reqs_per_step"] >= args.min_reqs_per_step,
            "detail": "max requests sharing one zigzag step: %d" % summary["max_reqs_per_step"]}
    analysis["coverage"] = [gate if item["name"] == "reqs-per-step" else item for item in analysis["coverage"]]

    if summary["plan_lines"] < args.min_zigzag_steps:
        result = "INCONCLUSIVE"
    elif not all(check["ok"] for check in analysis["checks"]):
        result = "FAIL"
    elif not all(check["ok"] for check in analysis["coverage"]):
        result = "INCONCLUSIVE"
    else:
        result = "PASS"
    if not args.quiet:
        _print_report(log, data, analysis, result)
    if args.json:
        payload = {"log": str(log), "cp_size": args.cp_size, "result": result,
                   "summary": summary, "checks": analysis["checks"], "coverage": analysis["coverage"],
                   "branch": {"%s|%s|%s" % key: count for key, count in data["branch"].items()},
                   "reduce": data["reduce"]}
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("[planlog] json -> %s" % args.json)
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 77}[result]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
