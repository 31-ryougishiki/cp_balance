#!/usr/bin/env python3
"""Read a profile trace as an ordered call list and map it back to the code.

    python3 profile_order.py prof_l6_cur_cp1 --rank rank0
    python3 profile_order.py prof_l6_cur_cp1 --rank rank0 --devices
    python3 profile_order.py prof_l6_cur_cp1 --rank rank0 --trim

op_statistic / step_trace_time tell you *what* is expensive, not *where in the
code* it comes from, and not in which order it runs.  This script reads the
host-side timeline of ASCEND_PROFILER_OUTPUT/trace_view.json, which carries the
torch.profiler.record_function scopes, and reduces it to

* the non-repeating head of the forward,
* one full decoder-layer cycle (what one layer does, in order, on one screen),
* the share of each name inside that cycle,

then resolves every name to the file:line that emits it.

The scopes only exist when the service runs with
VLLM_CUSTOM_SCOPES_FOR_PROFILING=1 (vllm/v1/utils.py:747); the profiling configs
set it.  Without it there is nothing to map and the script says so.

--trim writes order_<rank>.json next to the trace with just the ordered events,
so that small file can be pulled back instead of the whole trace.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

OUTPUT_DIR = "ASCEND_PROFILER_OUTPUT"
HOST_CATS = ("user_annotation", "cpu_op", "python_function")
DEVICE_CATS = ("kernel", "npu_kernel", "hcc", "aicore")

LABEL_PATTERNS = (
    re.compile(r'record_function_or_nullcontext\(\s*"([^"]+)"'),
    re.compile(r'record_function\(\s*"([^"]+)"'),
    re.compile(r'_sfa_5_3_scope\(\s*"([^"]+)"'),
)
CUSTOM_OP_PATTERN = re.compile(r'op_name="([^"]+)"')
WINDOW_RE = re.compile(r"_(\d{17})_ascend_pt$")
COLLECTIVES = (
    "all_gather_async",
    "all_gather_into_tensor",
    "tensor_model_parallel_all_gather",
    "tensor_model_parallel_all_reduce",
    "tensor_model_parallel_reduce_scatter",
    "reduce_scatter_tensor",
    "all_to_all_single",
    "all_reduce",
    "reduce_scatter",
    "all_gather",
    "all_to_all",
)
HCCL_KINDS = (
    ("allreduce", "all_reduce"),
    ("reducescatter", "reduce_scatter"),
    ("allgather", "all_gather"),
    ("alltoall", "all_to_all"),
)


def log(message: str) -> None:
    print(message, flush=True)


def to_float(value) -> float:
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0.0


def iter_trace_events(path: Path):
    """Stream the traceEvents array without loading the whole file."""
    decoder = json.JSONDecoder()
    window = 1 << 22
    with open(path, "r", encoding="utf-8") as handle:
        buf = handle.read(window)
        anchor = buf.find('"traceEvents"')
        start = buf.find("[", anchor if anchor >= 0 else 0)
        if start < 0:
            raise SystemExit("no traceEvents array in " + str(path))
        pos = start + 1
        while True:
            if pos >= len(buf):
                more = handle.read(window)
                if not more:
                    return
                buf = buf[pos:] + more
                pos = 0
            while pos < len(buf) and buf[pos] in " \t\r\n,":
                pos += 1
            if pos >= len(buf):
                continue
            if buf[pos] == "]":
                return
            try:
                event, end = decoder.raw_decode(buf, pos)
            except ValueError:
                more = handle.read(window)
                if not more:
                    return
                buf = buf[pos:] + more
                pos = 0
                continue
            pos = end
            if isinstance(event, dict):
                yield event


def longest_window(root: Path) -> str:
    """Window label with the largest target prompt, from windows.json."""
    path = root / "windows.json"
    if not path.is_file():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return ""
    best, best_target = "", -1
    for entry in data.get("entries") or []:
        target = int(entry.get("target_tokens") or 0)
        if target > best_target:
            for window in entry.get("window_ids") or []:
                best, best_target = str(window), target
    return best


def find_rank_dir(root: Path, wanted: str | None, window: str | None = None) -> Path:
    ranks = [path for path in sorted(root.rglob("*_ascend_pt")) if path.is_dir()]
    stamps = {m.group(1) for m in (WINDOW_RE.search(path.name) for path in ranks) if m}
    if not window and not wanted and len(stamps) > 1:
        # Several capture windows: attribute the longest prompt by default, it is
        # the one where attention matters most relative to the fixed costs.
        window = longest_window(root) or ""
    if window:
        ranks = [path for path in ranks if window in path.name]
        if not ranks:
            raise SystemExit("window %r not under %s" % (window, root))
    if not ranks:
        raise SystemExit("no *_ascend_pt under " + str(root))
    if not wanted and window:
        wanted = "rank0_"
    if wanted:
        # rank1 is a substring of rank10..rank15, so prefer the delimited form
        for needle in (wanted + "_", wanted):
            for path in ranks:
                if needle in path.name:
                    return path
        raise SystemExit("rank %r not in %s" % (wanted, [p.name for p in ranks]))
    return ranks[0]


def scan_repo(repo: Path) -> dict:
    """label / custom-op / collective -> emitting file:line."""
    labels: dict = {}
    ops: dict = {}
    collectives: dict = {}
    for path in sorted(repo.glob("vllm_ascend/**/*.py")):
        relative = path.relative_to(repo).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), 1):
            where = "%s:%d" % (relative, number)
            for pattern in LABEL_PATTERNS:
                for label in pattern.findall(line):
                    labels.setdefault(label, []).append(where)
            for name in CUSTOM_OP_PATTERN.findall(line):
                ops.setdefault(name, []).append(where)
            for primitive in COLLECTIVES:
                if re.search(r"\b%s\s*\(" % re.escape(primitive), line):
                    collectives.setdefault(primitive, []).append(where)
    return {"labels": labels, "ops": ops, "collectives": collectives}


def lookup(name: str, code_map: dict, limit: int = 3) -> str:
    labels = code_map["labels"]
    if name in labels:
        return "label " + ", ".join(labels[name][:limit])
    for label, sites in labels.items():
        if label and (label in name or name in label):
            return "label~%s %s" % (label, ", ".join(sites[:limit]))
    for op_name, sites in code_map["ops"].items():
        if op_name in name or name.endswith(op_name):
            return "op %s %s" % (op_name, ", ".join(sites[:limit]))
    lowered = name.lower()
    for needle, primitive in HCCL_KINDS:
        if needle in lowered:
            sites = code_map["collectives"].get(primitive, [])
            if sites:
                return "collective %s -> %d call sites, e.g. %s" % (
                    primitive,
                    len(sites),
                    ", ".join(sites[:limit]),
                )
    for primitive, sites in code_map["collectives"].items():
        if primitive in lowered:
            return "collective %s %s" % (primitive, ", ".join(sites[:limit]))
    return ""


def collect_timeline(rank_dir: Path, max_events: int) -> tuple:
    """One streaming pass: host events, device events, and a cat histogram.

    The histogram is the important part.  The previous round produced a 3.1 GB
    trace_view.json and not a single event matched the host categories this
    script guessed, which told us nothing about why.  Now every category that
    actually exists is printed with a count and an example, so an unmatched
    trace is self-explaining instead of empty.
    """
    trace = rank_dir / OUTPUT_DIR / "trace_view.json"
    if not trace.is_file():
        raise SystemExit("missing %s (run profile_analyse.py first)" % trace)
    log("[order] reading %s (%.1f MB)" % (trace, trace.stat().st_size / 1e6))
    cats = {}
    host = []
    device = []
    for event in iter_trace_events(trace):
        if event.get("ph") != "X":
            continue
        duration = event.get("dur")
        stamp = event.get("ts")
        # Some CANN/torch_npu trace versions serialize ts (and occasionally dur)
        # as JSON strings ("1790092908380061.150") rather than numbers; coerce
        # instead of rejecting, so the event stream isn't silently emptied
        # ("trace has no X events at all").
        try:
            duration = float(duration)
            stamp = float(stamp)
        except (TypeError, ValueError):
            continue
        raw_cat = str(event.get("cat") or "")
        item = cats.setdefault(raw_cat, {"count": 0, "example": "", "names": 0})
        item["count"] += 1
        if not item["example"]:
            item["example"] = str(event.get("name") or "?")
        category = raw_cat.lower()
        entry = {
            "name": str(event.get("name") or "?"),
            "cat": category,
            "pid": event.get("pid"),
            "tid": event.get("tid"),
            "ts": float(stamp),
            "dur": float(duration),
        }
        if any(key in category for key in DEVICE_CATS):
            if len(device) < max_events:
                device.append(entry)
        elif any(key in category for key in HOST_CATS):
            if len(host) < max_events:
                host.append(entry)
    return host, device, cats


def report_cats(cats: dict, top: int = 25) -> None:
    if not cats:
        log("[order] trace has no X events at all")
        return
    total = sum(item["count"] for item in cats.values())
    log("[order] event categories in this trace (%d X events, %d categories)" % (total, len(cats)))
    log("  %-34s %10s %8s %s" % ("cat", "count", "share", "example name"))
    for cat, item in sorted(cats.items(), key=lambda kv: -kv[1]["count"])[:top]:
        log(
            "  %-34s %10d %7.1f%% %s"
            % (cat[:34], item["count"], 100.0 * item["count"] / total, str(item["example"])[:40])
        )


def main_stream(events: list) -> list:
    counter: dict = {}
    for event in events:
        if "cpu_op" in event["cat"] or "user_annotation" in event["cat"]:
            key = (event["pid"], event["tid"])
            counter[key] = counter.get(key, 0) + 1
    if not counter:
        return events
    pid, tid = max(counter.items(), key=lambda item: item[1])[0]
    picked = [event for event in events if event["pid"] == pid and event["tid"] == tid]
    return picked or events


def cycle_slice(events: list) -> tuple:
    """Split into (head, one cycle, marker name) using the repeating scope."""
    counts: dict = {}
    for event in events:
        if "user_annotation" in event["cat"]:
            counts[event["name"]] = counts.get(event["name"], 0) + 1
    repeated = [(count, name) for name, count in counts.items() if count >= 3]
    if not repeated:
        return events, [], ""
    repeated.sort(key=lambda item: -item[0])
    marker = repeated[0][1]
    hits = [index for index, event in enumerate(events) if event["name"] == marker]
    if len(hits) < 3:
        return events, [], ""
    return events[: hits[1]], events[hits[1] : hits[2]], marker


def report(title: str, rows: list, code_map: dict, top: int) -> None:
    if not rows:
        return
    total = sum(row["dur"] for row in rows) or 1.0
    running = 0.0
    log("[order] %s" % title)
    log("  %-5s %-10s %9s %8s  %s" % ("#", "dur_us", "cum_us", "cum_%", "op / code"))
    for index, row in enumerate(rows[:top], 1):
        running += row["dur"]
        code = lookup(row["name"], code_map)
        log(
            "  %-5d %-10.1f %9.1f %7.1f%%  %s%s"
            % (index, row["dur"], running, 100.0 * running / total, row["name"], ("  <- " + code) if code else "")
        )
    if len(rows) > top:
        log("  ... %d more events" % (len(rows) - top))
    log("[order] %s: events=%d total=%.1fus" % (title, len(rows), sum(row["dur"] for row in rows)))


def report_share(title: str, rows: list, code_map: dict, top: int) -> None:
    total = sum(row["dur"] for row in rows) or 1.0
    bucket: dict = {}
    for row in rows:
        item = bucket.setdefault(row["name"], [0, 0.0])
        item[0] += 1
        item[1] += row["dur"]
    ordered = sorted(bucket.items(), key=lambda kv: -kv[1][1])
    log("[order] %s: share by name" % title)
    log("  %-46s %6s %12s %8s  %s" % ("name", "count", "total_us", "share", "code"))
    for op_name, (count, total_us) in ordered[:top]:
        log(
            "  %-46s %6d %12.1f %7.1f%%  %s"
            % (op_name[:46], count, total_us, 100.0 * total_us / total, lookup(op_name, code_map))
        )


def enclosing_scope(kernel: dict, annotations: list) -> str:
    """Innermost host scope covering the kernel, preferring a named scope.

    record_function scopes are the ones we can map back to a source line, so
    they win over a plain cpu_op; among equal kinds the innermost wins.
    """
    stamp = float(kernel.get("ts", kernel.get("start", 0.0)))
    best_named = None
    best_any = None
    for scope in annotations:
        if not (scope["start"] <= stamp < scope["start"] + scope["dur"]):
            continue
        if best_any is None or scope["dur"] < best_any["dur"]:
            best_any = scope
        if scope.get("named") and (best_named is None or scope["dur"] < best_named["dur"]):
            best_named = scope
    chosen = best_named or best_any
    return chosen["name"] if chosen else ""


def report_scopes(device_events: list, annotations: list, code_map: dict, top: int) -> None:
    """Attribute device kernel time to the host record_function scope it runs in.

    Both sides come from trace_view.json, so they share one clock.  A low
    coverage means the trace does not carry enclosing host scopes (see
    VLLM_CUSTOM_SCOPES_FOR_PROFILING) or host/device clocks differ.
    """
    if not device_events:
        return
    total = sum(event["dur"] for event in device_events) or 1.0
    covered = 0.0
    scope_us: dict = {}
    scope_count: dict = {}
    for event in device_events:
        scope = enclosing_scope(event, annotations)
        if not scope:
            continue
        covered += event["dur"]
        scope_us[scope] = scope_us.get(scope, 0.0) + event["dur"]
        scope_count[scope] = scope_count.get(scope, 0) + 1
    log("[order] device time by enclosing host scope: %d kernels, %.1fus, %.1f%% covered" % (len(device_events), total, 100.0 * covered / total))
    if not scope_us:
        log("[order] nothing covered; check VLLM_CUSTOM_SCOPES_FOR_PROFILING=1 and clock alignment")
        return
    log("  %-46s %6s %12s %8s  %s" % ("scope", "kers", "total_us", "share", "code"))
    for scope, used in sorted(scope_us.items(), key=lambda kv: -kv[1])[:top]:
        log(
            "  %-46s %6d %12.1f %7.1f%%  %s"
            % (scope[:46], scope_count[scope], used, 100.0 * used / total, lookup(scope, code_map))
        )


def csv_columns(header: list) -> dict:
    lookup = {str(key).lower(): key for key in header}
    return {
        "name": lookup.get("name") or lookup.get("type"),
        "start": lookup.get("start time(us)") or lookup.get("start time"),
        "dur": lookup.get("duration(us)") or lookup.get("duration"),
        "core": lookup.get("accelerator core"),
    }


def read_kernels(path: Path) -> list:
    """Ordered device kernels.  This profiler writes no Step column, so the
    order comes from Start Time(us) instead."""
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = csv_columns(reader.fieldnames or [])
        if not columns["name"] or not columns["dur"]:
            log("[order] kernel_details.csv has no usable name/duration column")
            return []
        for row in reader:
            rows.append(
                {
                    "name": str(row.get(columns["name"])),
                    "start": to_float(row.get(columns["start"])) if columns["start"] else 0.0,
                    "dur": to_float(row.get(columns["dur"])),
                    "core": str(row.get(columns["core"])) if columns["core"] else "",
                }
            )
    rows.sort(key=lambda item: item["start"])
    return rows


def report_devices(rank_dir: Path, top: int) -> None:
    """Ordered device kernels plus one repeating layer cycle."""
    path = rank_dir / OUTPUT_DIR / "kernel_details.csv"
    if not path.is_file():
        log("[order] no kernel_details.csv, skipping device order")
        return
    log("[order] streaming %s" % path)
    rows = read_kernels(path)
    if not rows:
        return
    busy = sum(row["dur"] for row in rows)
    span = rows[-1]["start"] + rows[-1]["dur"] - rows[0]["start"]
    log(
        "[order] kernels=%d busy=%.1fus over a %.1fus span (%.1f%% occupied)"
        % (len(rows), busy, span, 100.0 * busy / span if span else 0.0)
    )
    bucket = {}
    for row in rows:
        item = bucket.setdefault(row["name"], [0, 0.0])
        item[0] += 1
        item[1] += row["dur"]
    log("[order] device time by kernel, top %d" % top)
    log("  %-46s %8s %13s %8s" % ("kernel", "count", "total_us", "share"))
    for name, (count, total) in sorted(bucket.items(), key=lambda kv: -kv[1][1])[:top]:
        log("  %-46s %8d %13.1f %7.1f%%" % (name[:46], count, total, 100.0 * total / busy))
    counts = {name: count for name, (count, _total) in bucket.items()}
    markers = [name for name, count in counts.items() if count >= 3]
    if not markers:
        return
    marker = max(markers, key=lambda name: counts[name])
    hits = [index for index, row in enumerate(rows) if row["name"] == marker]
    if len(hits) < 3:
        log("[order] no repeating device cycle found (marker %s x%d)" % (marker, len(hits)))
        return
    cycle = rows[hits[1]:hits[2]]
    log("[order] one device cycle, split on %s (occurrence %d..%d)" % (marker, 2, 3))
    log("  %-5s %-46s %12s %11s" % ("#", "kernel", "start_us", "dur_us"))
    for index, row in enumerate(cycle[:top], 1):
        log("  %-5d %-46s %12.1f %11.1f" % (index, row["name"][:46], row["start"], row["dur"]))
    if len(cycle) > top:
        log("  ... %d more kernels" % (len(cycle) - top))
    log("[order] cycle kernels=%d busy=%.1fus" % (len(cycle), sum(row["dur"] for row in cycle)))


def infer_repo(prof_dir: Path, explicit: str) -> Path | None:
    if explicit:
        return Path(explicit)
    here = Path(__file__).resolve().parent.parent
    for config in sorted((here / "configs").glob("*.json")):
        try:
            data = json.loads(config.read_text(encoding="utf-8"))
        except ValueError:
            continue
        configured = (data.get("profiler") or {}).get("dir")
        if configured and Path(str(configured)).resolve() == prof_dir:
            return Path(data["repo"]) if data.get("repo") else None
    # 没在 configs 里找到：退回 harness.json trees.cur（不再写死某台机器的路径）
    try:
        import serve_config

        return Path(serve_config.tree_path("cur"))
    except Exception:  # noqa: BLE001 - mapping is optional
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiler_dir", help="directory printed by profile_forward.py")
    parser.add_argument("--rank", default="", help="substring of the rank directory (default: first)")
    parser.add_argument(
        "--window",
        default="",
        help="capture window id (the 17-digit stamp in the rank dir name), one per prompt length",
    )
    parser.add_argument("--repo", default="", help="code tree to map names to (default: from configs/)")
    parser.add_argument("--top", type=int, default=60, help="rows printed per table")
    parser.add_argument("--devices", action="store_true", help="print the device kernel order instead")
    parser.add_argument("--max-events", type=int, default=2000000, help="cap on stored events per class")
    parser.add_argument("--trim", action="store_true", help="also write order_<rank>.json")
    args = parser.parse_args()

    root = Path(args.profiler_dir).resolve()
    rank_dir = find_rank_dir(root, args.rank or None, args.window or None)
    log("[order] rank_dir=%s" % rank_dir)
    repo = infer_repo(root, args.repo)
    code_map = scan_repo(repo) if repo and repo.is_dir() else {"labels": {}, "ops": {}, "collectives": {}}
    if repo:
        log(
            "[order] code map from %s: labels=%d ops=%d collectives=%d"
            % (repo, len(code_map["labels"]), len(code_map["ops"]), len(code_map["collectives"]))
        )
    if not code_map["labels"]:
        log("[order] no record_function labels; run the service with VLLM_CUSTOM_SCOPES_FOR_PROFILING=1")

    host, device, cats = collect_timeline(rank_dir, args.max_events)

    if args.devices:
        annotations = [
            {
                "name": event["name"],
                "start": event["ts"],
                "dur": event["dur"],
                "named": "user_annotation" in event["cat"],
            }
            for event in host
            if "user_annotation" in event["cat"] or "cpu_op" in event["cat"]
        ]
        report_scopes(device, annotations, code_map, args.top)
        report_devices(rank_dir, args.top)
        return 0

    events = host
    if not events:
        report_cats(cats)
        log("[order] no event matched the host categories %s" % (HOST_CATS,))
        log("[order] add the right category to HOST_CATS, or use --devices which reads the CSV")
        return 1
    events.sort(key=lambda item: item["ts"])
    main_events = main_stream(events)
    head, cycle, marker = cycle_slice(main_events)
    log("[order] host events=%d main-thread=%d device events=%d marker=%s" % (len(events), len(main_events), len(device), marker or "none"))
    report("head (before the first full layer cycle)", head, code_map, args.top)
    if cycle:
        report("one decoder-layer cycle", cycle, code_map, args.top)
        report_share("cycle", cycle, code_map, 30)
    else:
        report_share("whole forward", main_events, code_map, 30)
    if args.trim:
        out = root / ("order_%s.json" % (args.rank or "first"))
        out.write_text(
            json.dumps(
                {"rank_dir": str(rank_dir), "marker": marker, "head": head, "cycle": cycle or main_events},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        log("[order] trimmed -> %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
