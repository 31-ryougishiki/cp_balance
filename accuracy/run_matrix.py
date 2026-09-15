#!/usr/bin/env python3
"""Run a matrix config: launch each referenced config, collect 40 prompts, compare.

    python run_matrix.py configs/matrix_b_vs_base.json
    bash run_matrix.sh configs/matrix_b_vs_base.json

Each entry of "configs" is one test config under configs/; the ports come from
those configs, so runs are fully described by JSON.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import serve_config  # noqa: E402 - needs ROOT on sys.path first

HERE = Path(__file__).resolve().parent
COUNTS = ("path=fixed_order", "path=native", "branch=ZIGZAG", "[CP_BALANCE][plan]", "branch=CONTINUOUS")


def wait_port_free(port: int, timeout: int = 60) -> bool:
    """Wait until nothing listens on the port (previous service released it)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return True
        time.sleep(2)
    return False


def wait_ready(port: int, proc: subprocess.Popen, timeout: int) -> tuple:
    url = "http://127.0.0.1:%s/v1/models" % port
    waited = 0
    while waited < timeout:
        if proc.poll() is not None:
            return False, "service exited rc=%s" % proc.returncode
        try:
            with urllib.request.urlopen(url, timeout=5):
                return True, "ready after %s seconds" % waited
        except Exception:
            time.sleep(10)
            waited += 10
    return False, "not ready after %s seconds" % timeout


def terminate_tree(proc: subprocess.Popen, force: bool) -> None:
    """Stop the wrapper and every process it spawned (POSIX group, Windows tree)."""
    if hasattr(os, "killpg"):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL if force else signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        return
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)


def stop(proc: subprocess.Popen, log) -> None:
    if proc.poll() is not None:
        return
    terminate_tree(proc, force=False)
    deadline = time.time() + 120
    while time.time() < deadline and proc.poll() is None:
        time.sleep(2)
    if proc.poll() is None:
        terminate_tree(proc, force=True)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            pass
    time.sleep(5)
    log("  service stopped")


def fingerprint_from_log(path: Path) -> str:
    if not path.is_file():
        return ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("[cp_balance] CONFIG="):
            return line.strip()
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("matrix", help="matrix config path or name under configs/")
    parser.add_argument("--out", default="", help="output directory")
    parser.add_argument("--only", default="", help="comma separated subset of config names")
    parser.add_argument("--dry-run", action="store_true", help="print the resolved plan and exit")
    args = parser.parse_args()

    matrix = serve_config.load_config(args.matrix)
    entries = [str(item) for item in (matrix.get("configs") or [])]
    resolved = {entry: serve_config.load_config(entry) for entry in entries}
    keys = {entry: str(resolved[entry].get("name") or entry) for entry in entries}
    if args.only:
        keep = {item.strip() for item in args.only.split(",") if item.strip()}
        entries = [entry for entry in entries if entry in keep or keys[entry] in keep]
        resolved = {entry: resolved[entry] for entry in entries}
        keys = {entry: keys[entry] for entry in entries}
    compares = matrix.get("compare") or []
    plan = ["matrix=%s configs=%s" % (matrix.get("name"), ",".join(entries))]
    for entry in entries:
        cfg = resolved[entry]
        plan.append("plan %-24s port=%s repo=%s CP_BALANCE=%s det=%s file=%s" % (keys[entry], cfg.get("port"), cfg.get("repo"), cfg.get("cp_balance"), bool(cfg.get("deterministic")), entry))
    for item in compares:
        plan.append("compare %-16s %s vs %s require_text=%s gate=%s" % (item.get("label"), item.get("left"), item.get("right"), item.get("require_text", False), item.get("gate", False)))

    if args.dry_run:
        for line in plan:
            print("[matrix] " + line, flush=True)
        print("[matrix] dry-run: nothing launched, no output directory created", flush=True)
        return 0

    out = Path(args.out) if args.out else Path.cwd() / ("matrix_%s_%s" % (matrix.get("name", "run"), time.strftime("%m%d_%H%M")))
    out.mkdir(parents=True, exist_ok=True)
    summary = (out / "summary.txt").open("w", encoding="utf-8")

    def log(msg: str) -> None:
        print("[matrix] " + msg, flush=True)
        summary.write("[matrix] " + msg + chr(10))
        summary.flush()

    log("out=%s" % out)
    for line in plan:
        log(line)

    check = matrix.get("static_check")
    if check:
        log("step 0: static gate check")
        proc = subprocess.run(
            [sys.executable, str(HERE / "check_b_path.py"), "--repo", str(check.get("repo")), "--base-repo", str(check.get("base_repo"))],
            capture_output=True,
            text=True,
        )
        lines = [line for line in proc.stdout.strip().splitlines() if line.strip()]
        for line in lines[-2:]:
            log("  " + line)

    timeout = int(matrix.get("ready_timeout", 1800))
    for entry in entries:
        name = keys[entry]
        port = int(resolved[entry]["port"])
        log_path = out / (name + ".log")
        log("run %s (port %s)" % (name, port))
        with log_path.open("w", encoding="utf-8") as handle:
            proc = subprocess.Popen(
                ["bash", str(ROOT / "run.sh"), entry],
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            ok, msg = wait_ready(port, proc, timeout)
            log("  " + msg)
            if ok:
                collect_log = out / (name + ".collect.txt")
                with collect_log.open("w", encoding="utf-8") as clog:
                    collect = subprocess.run(
                        [sys.executable, str(HERE / "compare_first_token.py"), "collect", "--url", "http://127.0.0.1:%s" % port, "--out", str(out / (name + ".json"))],
                        stdout=clog,
                        stderr=subprocess.STDOUT,
                    )
                lines = [line for line in collect_log.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
                suffix = "" if collect.returncode == 0 else " (collect rc=%s)" % collect.returncode
                log("  " + (lines[-1] if lines else "collect produced no output") + suffix)
            log("  " + (fingerprint_from_log(log_path) or "WARNING: no [cp_balance] fingerprint in " + str(log_path)))
            text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
            for key in COUNTS:
                log("  count %-22s %s" % (key, text.count(key)))
            stop(proc, log)
            if not wait_port_free(port):
                log("  WARNING: port %s still busy after stopping the service" % port)

    failed = []
    for item in compares:
        label = str(item.get("label"))
        left = out / (str(item.get("left")) + ".json")
        right = out / (str(item.get("right")) + ".json")
        if not left.is_file() or not right.is_file():
            log("compare %s: SKIP (missing json)" % label)
            if item.get("gate"):
                failed.append(label)
            continue
        cmd = [sys.executable, str(HERE / "compare_first_token.py"), "compare"]
        if item.get("require_text"):
            cmd.append("--require-text")
        cmd += [str(left), str(right)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        output = proc.stdout + proc.stderr
        (out / ("cmp_%s.txt" % label)).write_text(output, encoding="utf-8")
        log("compare %s -> exit=%s" % (label, proc.returncode))
        for line in output.splitlines():
            if line.startswith("[compare]") or " DIFF " in line:
                log("  " + line)
        if proc.returncode != 0 and item.get("gate"):
            failed.append(label)

    log("---- verdict ----")
    if failed:
        log("RESULT: FAIL (gated compare failed: %s)" % ", ".join(failed))
    else:
        log("RESULT: PASS")
    log("artifacts: %s" % out)
    summary.close()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
