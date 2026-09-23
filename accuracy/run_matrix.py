#!/usr/bin/env python3
"""Run a matrix config: launch each referenced config, collect 40 prompts, compare.

    python3 accuracy/run_matrix.py configs/matrix_b_vs_base.json
    bash tests/run_tests.sh --only accuracy/a10_matrix_gate     # 走测试入口（推荐）

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
COUNTS = ("branch=ZIGZAG", "[CP_BALANCE][plan]", "branch=CONTINUOUS")


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

    failed = []
    check = matrix.get("static_check")
    if check:
        # repo/base_repo may name harness.json tree roles instead of hardcoding paths
        repo = check.get("repo") or (serve_config.tree_path(check["repo_tree"]) if check.get("repo_tree") else "")
        base = check.get("base_repo") or (serve_config.tree_path(check["base_tree"]) if check.get("base_tree") else "")
        log("step 0: static gate check (repo=%s base=%s)" % (repo, base))
        proc = subprocess.run(
            [sys.executable, str(HERE / "check_b_path.py"), "--repo", str(repo), "--base-repo", str(base)],
            capture_output=True,
            text=True,
        )
        lines = [line for line in proc.stdout.strip().splitlines() if line.strip()]
        for line in lines[-2:]:
            log("  " + line)
        if proc.returncode != 0:
            log("  static gate FAILED (rc=%s)" % proc.returncode)
            failed.append("static_check")

    timeout = int(matrix.get("ready_timeout", serve_config.limit("ready_timeout_s", 2400)))
    log_text: dict = {}
    for entry in entries:
        name = keys[entry]
        port = int(resolved[entry]["port"])
        log_path = out / (name + ".log")
        log("run %s (port %s)" % (name, port))
        if not wait_port_free(port):
            log("  FAIL port %s is busy before launch: 先停掉残留服务（pkill -f -- \"--port %s\"）" % (port, port))
            failed.append("run:%s" % name)
            continue
        handle = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            ["bash", str(ROOT / "run.sh"), entry],
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            ok, msg = wait_ready(port, proc, timeout)
            log("  " + msg)
            if not ok:
                failed.append("run:%s" % name)
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
                if collect.returncode != 0:
                    failed.append("collect:%s" % name)
            fingerprint = fingerprint_from_log(log_path)
            log("  " + (fingerprint or "WARNING: no [cp_balance] fingerprint in " + str(log_path)))
            # 端口上必须是这条配置的服务：残留进程/串台会让整轮对比失去意义
            if fingerprint and ("CONFIG=%s " % name) not in fingerprint:
                log("  FAIL fingerprint is not CONFIG=%s: %s" % (name, fingerprint[:140]))
                failed.append("fingerprint:%s" % name)
            elif not fingerprint:
                failed.append("fingerprint:%s" % name)
            text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
            log_text[name] = text
            for key in COUNTS:
                log("  count %-22s %s" % (key, text.count(key)))
        finally:
            # 任何异常/中断都要把 8 卡服务收掉，否则后面的测试全部撞端口
            stop(proc, log)
            handle.close()
        if not wait_port_free(port):
            log("  WARNING: port %s still busy after stopping the service" % port)

    # require_log: 证明这批服务真的走在预期分支上，防止“两边都没进 zigzag”也 PASS
    for item in matrix.get("require_log") or []:
        name = str(item.get("config"))
        text = log_text.get(name)
        if text is None:
            log("require %s: SKIP (config not run)" % name)
            continue
        any_pats = [str(pat) for pat in item.get("any") or []]
        all_pats = [str(pat) for pat in item.get("all") or []]
        none_pats = [str(pat) for pat in item.get("none") or []]
        bad = []
        if any_pats and not any(pat in text for pat in any_pats):
            bad.append("none of %s" % any_pats)
        bad += ["missing %s" % pat for pat in all_pats if pat not in text]
        bad += ["unexpected %s" % pat for pat in none_pats if pat in text]
        if bad:
            log("require %s: FAIL %s -- %s" % (name, "; ".join(bad), item.get("why", "")))
            failed.append("require_log:%s" % name)
        else:
            log("require %s: ok (%s)" % (name, item.get("why", "")))

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
        log("RESULT: FAIL (failed: %s)" % ", ".join(failed))
    else:
        log("RESULT: PASS")
    log("artifacts: %s" % out)
    summary.close()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
