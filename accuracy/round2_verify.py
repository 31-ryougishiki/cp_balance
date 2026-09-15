#!/usr/bin/env python3
"""Round-2 verification driver: one command, one verdict per item.

    bash accuracy/round2_verify.sh                  # run every step
    bash accuracy/round2_verify.sh --steps 0,1,2    # skip the optional A/B
    bash accuracy/round2_verify.sh --dry-run        # show the resolved plan only

Steps
  0  static gates (no service): check_cp_balance_fields + check_b_path
  1  C acceptance: matrix_c_accept -- CP_BALANCE=1 vs 0, 40 prompts, first token
  2  B equivalence: matrix_b_vs_base -- CP_BALANCE=0 vs base, plus the noise floor
  3  optional A/B that needs a temporary source patch (restored afterwards):
     3a baseline run of glm52_cur_cp1 with the long prompts -> first tokens + log
     3b same run again with the patch: reports the TP/EP domain line and writes
        the MLA / indexer KV cache with the padding rows (slot < 0) filtered out
     3c compare baseline vs patched; a match means relying on slot == -1 being
        skipped by the scatter op is safe

Everything lands in round2_<timestamp>/; summary.txt carries every verdict and
lists the files that have to be sent back.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (HERE, ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import serve_config  # noqa: E402  needs ROOT on sys.path

import run_matrix as rm  # noqa: E402  reuse the service start/stop helpers

C_MATRIX = "configs/matrix_c_accept.json"
B_MATRIX = "configs/matrix_b_vs_base.json"
OPTIONAL_CONFIG = "glm52_cur_cp1"
GROUP_MARK = "[CP_BALANCE][group]"
PATCH_TAG = "ROUND2-PATCH"
FLAT = chr(10)
CRLF = chr(13) + chr(10)


class Report:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.summary = (out / "summary.txt").open("w", encoding="utf-8")
        self.results = []

    def log(self, msg: str) -> None:
        print("[round2] " + msg, flush=True)
        self.summary.write("[round2] " + msg + chr(10))
        self.summary.flush()

    def verdict(self, step: str, label: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        self.results.append((step, label, status, detail))
        self.log("%-4s %-4s %s" % (status, step, (label + "  " + detail).rstrip()))

    def skip(self, step: str, label: str, detail: str = "") -> None:
        self.results.append((step, label, "SKIP", detail))
        self.log("SKIP %-4s %s" % (step, (label + "  " + detail).rstrip()))

    def close(self) -> None:
        self.summary.close()


def run(cmd, **kwargs):
    return subprocess.run([str(item) for item in cmd], capture_output=True, text=True, **kwargs)


def git_head(repo: Path) -> str:
    proc = run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"])
    return (proc.stdout or proc.stderr).strip() or "(unknown)"



def base_tree_ok(base: Path) -> bool:
    """The base comparison needs <base>/vllm_ascend/attention/sfa_v1.py."""
    return (base / "vllm_ascend/attention/sfa_v1.py").is_file()


def base_missing_note(base: Path) -> str:
    return (
        "base checkout not readable at %s/vllm_ascend/attention/sfa_v1.py -- "
        "create it with: git clone <same origin as the current tree> %s && "
        "git -C %s checkout c7990e5e4   (or rerun with --steps 0,1,3)"
        % (base, base, base)
    )


def step0(rep: Report, cur: Path, base: Path) -> bool:
    rep.log("step 0: static gates (no service)")
    ok_all = True
    checks = [("fields", [sys.executable, str(ROOT / "perf/check_cp_balance_fields.py"), "--repo", cur])]
    if base_tree_ok(base):
        checks.append(
            ("b_path", [sys.executable, str(HERE / "check_b_path.py"), "--repo", cur, "--base-repo", base])
        )
    else:
        rep.skip("0", "static/b_path", base_missing_note(base))
    for label, cmd in checks:
        proc = run(cmd)
        text = proc.stdout + proc.stderr
        (rep.out / ("00_static_" + label + ".txt")).write_text(text, encoding="utf-8")
        ok = proc.returncode == 0 and "RESULT: PASS" in text
        for line in text.splitlines():
            if "FAIL" in line:
                rep.log("  " + line.strip())
        tail = [line for line in text.splitlines() if line.strip()]
        rep.verdict("0", "static/" + label, ok, (tail[-1].strip() if tail else "(no output)"))
        if label == "fields":
            for line in text.splitlines():
                if "ZigzagPlan fields=" in line:
                    rep.log("  note: " + line.strip() + "   (expect fields=13 after the cleanup)")
        ok_all = ok_all and ok
    return ok_all


def step_matrix(rep: Report, step: str, matrix: str, tag: str) -> tuple:
    out = rep.out / ("0%s_%s" % (step, tag))
    out.mkdir(parents=True, exist_ok=True)
    rep.log("step %s: %s  (out=%s)" % (step, matrix, out))
    proc = subprocess.run(
        ["bash", str(HERE / "run_matrix.sh"), matrix, "--out", str(out)],
        capture_output=True,
        text=True,
    )
    text = proc.stdout + proc.stderr
    (out / "driver.txt").write_text(text, encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("[matrix]"):
            continue
        if any(key in stripped for key in ("RESULT", "compare ", "WARNING", "count branch=", "count path=")):
            rep.log("  " + stripped.replace("[matrix] ", ""))
    summary = out / "summary.txt"
    body = summary.read_text(encoding="utf-8", errors="replace") if summary.is_file() else ""
    ok = proc.returncode == 0 and "RESULT: PASS" in body
    rep.verdict(step, matrix, ok, "artifacts=%s" % out)
    if not ok:
        rep.log("  next: read %s (the per-config logs sit next to it)" % (out / "summary.txt"))
    return ok, out


def patch_edits(cur: Path) -> list:
    """Temporary edits for step 3. Every anchor must occur exactly once."""
    fwd = cur / "vllm_ascend/ascend_forward_context.py"
    sfa = cur / "vllm_ascend/attention/sfa_v1.py"
    fwd_old = (
        "        forward_context.zigzag_cp_context = zigzag_cp_context\n"
        "        forward_context.zigzag_cp_active = zigzag_cp_active\n"
    )
    fwd_new = fwd_old + (
        "\n"
        "        if zigzag_cp_active:\n"
        "            # " + PATCH_TAG + ": report the TP/EP domains once per process.\n"
        "            from vllm.distributed.parallel_state import get_ep_group, get_tp_group\n"
        "\n"
        "            _tp_group, _ep_group = get_tp_group(), get_ep_group()\n"
        "            logger.info_once(\n"
        '                "[CP_BALANCE][group] tp=%d/%d ep=%d/%d",\n'
        "                _tp_group.world_size, _tp_group.rank_in_group,\n"
        "                _ep_group.world_size, _ep_group.rank_in_group,\n"
        "            )\n"
    )
    sfa_old = (
        "                    assert dsa_cp_context.slot_mapping_cp_gathered is not None\n"
        "                    scatter_slots = dsa_cp_context.slot_mapping_cp_gathered\n"
        "                    fused_kv_actual = fused_kv_no_split\n"
    )
    sfa_new = sfa_old + (
        "                    # " + PATCH_TAG + ": write only the real rows.\n"
        "                    _keep = scatter_slots >= 0\n"
        "                    scatter_slots = scatter_slots[_keep]\n"
        "                    fused_kv_actual = fused_kv_actual[_keep]\n"
    )
    idx_old = (
        "                    assert attn_metadata.dsa_cp_context.slot_mapping_cp_gathered is not None\n"
        "                    idx_slots = attn_metadata.dsa_cp_context.slot_mapping_cp_gathered\n"
    )
    idx_new = idx_old + (
        "                    # " + PATCH_TAG + ": write only the real rows.\n"
        "                    _keep = idx_slots >= 0\n"
        "                    idx_slots = idx_slots[_keep]\n"
        "                    k_li = k_li[_keep]\n"
        "                    if k_li_scale is not None:\n"
        "                        k_li_scale = k_li_scale[_keep]\n"
    )
    edits = [(fwd, fwd_old, fwd_new), (sfa, sfa_old, sfa_new), (sfa, idx_old, idx_new)]
    for path, old, _new in edits:
        flat = path.read_bytes().decode("utf-8").replace(CRLF, FLAT)
        if flat.count(old) != 1:
            raise RuntimeError(
                "patch anchor not found exactly once in %s (found %d); the source "
                "changed since this script was written" % (path, flat.count(old))
            )
    return edits


def apply_patch(data: bytes, old: str, new: str) -> bytes:
    """Replace old with new once, tolerating CRLF (keeping the file style)."""
    text = data.decode("utf-8")
    crlf = CRLF in text
    flat = text.replace(CRLF, FLAT)
    if flat.count(old) != 1:
        raise RuntimeError("patch anchor found %d times (expected 1)" % flat.count(old))
    patched = flat.replace(old, new, 1)
    if crlf:
        patched = patched.replace(FLAT, CRLF)
    return patched.encode("utf-8")



def single_collect(rep: Report, out: Path, port: int, timeout: int, stem: str, log_tail: bool = False) -> bool:
    log_path = out / (stem + ".log")
    json_path = out / (stem + ".json")
    rep.log("  %s: starting %s on port %s" % (stem, OPTIONAL_CONFIG, port))
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            ["bash", str(ROOT / "run.sh"), OPTIONAL_CONFIG],
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        ok, msg = rm.wait_ready(port, proc, timeout)
        rep.log("  %s: %s" % (stem, msg))
        if ok:
            collect = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "compare_first_token.py"),
                    "collect",
                    "--url",
                    "http://127.0.0.1:%s" % port,
                    "--kind",
                    "long",
                    "--out",
                    str(json_path),
                ],
                capture_output=True,
                text=True,
            )
            text = collect.stdout + collect.stderr
            (out / (stem + ".collect.txt")).write_text(text, encoding="utf-8")
            lines = [line for line in text.splitlines() if line.strip()]
            if lines:
                rep.log("  " + lines[-1].strip())
            rep.log("  " + (rm.fingerprint_from_log(log_path) or "WARNING: no [cp_balance] fingerprint"))
            body = log_path.read_text(encoding="utf-8", errors="replace")
            rep.log("  branch lines: ZIGZAG=%d CONTINUOUS=%d" % (body.count("branch=ZIGZAG"), body.count("branch=CONTINUOUS")))
            if log_tail:
                for key in ("[CP_BALANCE][plan]", GROUP_MARK):
                    for line in body.splitlines():
                        if key in line:
                            rep.log("  " + line.strip()[:200])
                            break
        rm.stop(proc, rep.log)
        if not rm.wait_port_free(port):
            rep.log("  WARNING: port %s still busy after stopping the service" % port)
    return ok and json_path.is_file()


def step3(rep: Report, cur: Path, timeout: int) -> bool:
    rep.log("step 3: optional A/B with a temporary source patch")
    out = rep.out / "03_optional"
    out.mkdir(parents=True, exist_ok=True)
    flag = out / "PATCH_APPLIED.flag"
    if flag.is_file():
        rep.log("  ABORT: %s exists -- a previous run left the tree patched." % flag)
        rep.log("  restore with: git -C %s checkout -- vllm_ascend/ascend_forward_context.py vllm_ascend/attention/sfa_v1.py" % cur)
        rep.verdict("3", "optional A/B", False, "leftover patch from a previous run")
        return False
    try:
        edits = patch_edits(cur)
    except RuntimeError as exc:
        rep.verdict("3", "optional A/B", False, str(exc))
        return False

    cfg = serve_config.load_config(OPTIONAL_CONFIG)
    cfg_repo = os.path.realpath(str(cfg.get("repo") or ""))
    if cfg_repo and cfg_repo != os.path.realpath(str(cur)):
        rep.log("  ABORT: config repo %s != --repo %s (the patch would hit the wrong tree)" % (cfg_repo, cur))
        rep.verdict("3", "optional A/B", False, "repo mismatch between the config and --repo")
        return False
    port = int(cfg["port"])
    backups = {path: path.read_bytes() for path, _old, _new in edits}
    try:
        ok_baseline = single_collect(rep, out, port, timeout, "baseline")
        rep.verdict("3", "baseline collect", ok_baseline, "json=%s" % (out / "baseline.json"))
        for path, old, new in edits:
            path.write_bytes(apply_patch(path.read_bytes(), old, new))
            (out / (path.name + ".round2bak")).write_bytes(backups[path])
        flag.write_text("patched: " + ", ".join(str(path) for path in backups) + chr(10), encoding="utf-8")
        rep.log("  temporary patch applied to: %s" % ", ".join(sorted({path.name for path in backups})))
        compile_proc = run([sys.executable, "-m", "py_compile"] + sorted({str(path) for path in backups}))
        if compile_proc.returncode != 0:
            rep.log("  ABORT: patched sources do not compile: " + (compile_proc.stdout + compile_proc.stderr))
            rep.verdict("3", "optional A/B", False, "temporary patch failed to compile")
            return False
        ok_patched = single_collect(rep, out, port, timeout, "patched", log_tail=True)
        rep.verdict("3", "patched collect", ok_patched, "json=%s" % (out / "patched.json"))
    finally:
        for path, data in backups.items():
            path.write_bytes(data)
        if flag.is_file():
            flag.unlink()
        rep.log("  temporary patch reverted (byte-identical restore)")

    patched_log = out / "patched.log"
    body = patched_log.read_text(encoding="utf-8", errors="replace") if patched_log.is_file() else ""
    group_lines = [line.strip() for line in body.splitlines() if GROUP_MARK in line]
    if group_lines:
        rep.log("  " + group_lines[0])
        rep.log("  A3 judgement: the TP and EP halves must print the same world_size/rank pair")
    else:
        rep.log("  WARNING: no " + GROUP_MARK + " line in the patched log (was the batch eligible?)")

    left = out / "baseline.json"
    right = out / "patched.json"
    if not left.is_file() or not right.is_file():
        rep.verdict("3", "slot<0 filter A/B", False, "missing collect json")
        return False
    proc = run([sys.executable, str(HERE / "compare_first_token.py"), "compare", left, right])
    text = proc.stdout + proc.stderr
    (out / "cmp_patched_vs_baseline.txt").write_text(text, encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("[compare]") or " DIFF " in line:
            rep.log("  " + line.strip())
    ok = proc.returncode == 0 and "RESULT: PASS" in text
    rep.verdict("3", "slot<0 filter A/B", ok, "match => slot == -1 is skipped by the scatter")
    return ok


def main() -> int:
    global C_MATRIX, B_MATRIX, OPTIONAL_CONFIG

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", default="0,1,2,3", help="comma separated subset of 0,1,2,3")
    parser.add_argument("--c-matrix", default=C_MATRIX, help="C acceptance matrix (default for this repo)")
    parser.add_argument("--b-matrix", default=B_MATRIX, help="B equivalence matrix (also holds the repo paths)")
    parser.add_argument("--optional-config", default=OPTIONAL_CONFIG, help="config used by the optional A/B")
    parser.add_argument("--repo", default="", help="current vllm-ascend tree (default: seen in matrix configs)")
    parser.add_argument("--base-repo", default="", help="vllm-ascend-base tree (default: seen in matrix configs)")
    parser.add_argument("--out", default="", help="output directory (default round2_<timestamp>)")
    parser.add_argument("--ready-timeout", type=int, default=1800, help="service ready timeout in seconds")
    parser.add_argument("--dry-run", action="store_true", help="print the resolved plan and exit")
    args = parser.parse_args()
    os.chdir(ROOT)  # products and relative config paths follow the repo root, like run_matrix.sh

    C_MATRIX = args.c_matrix
    B_MATRIX = args.b_matrix
    OPTIONAL_CONFIG = args.optional_config

    steps = {item.strip() for item in args.steps.split(",") if item.strip()}
    matrix_b = serve_config.load_config(B_MATRIX)
    static_check = matrix_b.get("static_check") or {}
    cur_raw = args.repo or static_check.get("repo") or "/opt/its/z30055003/vllm-ascend"
    base_raw = args.base_repo or static_check.get("base_repo") or "/opt/its/z30055003/vllm-ascend-base"
    cur = Path(cur_raw)
    base = Path(base_raw)
    out = Path(args.out) if args.out else Path.cwd() / ("round2_" + time.strftime("%m%d_%H%M"))

    plan = [
        "steps            %s" % ",".join(sorted(steps)),
        "repo(cur)        %s  HEAD=%s" % (cur_raw, git_head(cur)),
        "repo(base)       %s  HEAD=%s%s"
        % (base_raw, git_head(base), "" if base_tree_ok(base) else "   <-- MISSING, steps 0/b_path and 2 will be skipped"),
        "out              %s" % out,
        "step 0           static gates: check_cp_balance_fields, check_b_path",
        "step 1           %s" % C_MATRIX,
        "step 2           %s" % B_MATRIX,
        "step 3           %s long prompts, baseline then patched (patch is reverted)" % OPTIONAL_CONFIG,
    ]
    if args.dry_run:
        for line in plan:
            print("[round2] " + line)
        print("[round2] dry-run: nothing launched, no output directory created")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    rep = Report(out)
    for line in plan:
        rep.log(line)
    rep.log("==== round-2 verification ====")
    verdicts = []
    if "0" in steps:
        verdicts.append(step0(rep, cur, base))
    if "1" in steps:
        ok, _ = step_matrix(rep, "1", C_MATRIX, "c_accept")
        verdicts.append(ok)
    if "2" in steps:
        if base_tree_ok(base):
            ok, _ = step_matrix(rep, "2", B_MATRIX, "b_equiv")
            verdicts.append(ok)
        else:
            rep.skip("2", B_MATRIX, base_missing_note(base))
    if "3" in steps:
        verdicts.append(step3(rep, cur, args.ready_timeout))

    rep.log("---- verdict ----")
    for step, label, status, detail in rep.results:
        rep.log("%-4s step %s  %-28s %s" % (status, step, label, detail))
    fails = [item for item in rep.results if item[2] == "FAIL"]
    skips = [item for item in rep.results if item[2] == "SKIP"]
    if fails:
        overall = False
        rep.log("RESULT: FAIL")
    elif skips:
        overall = True
        rep.log("RESULT: PASS (partial: %d item(s) skipped, see SKIP lines)" % len(skips))
    else:
        overall = True
        rep.log("RESULT: PASS")
    rep.log("send back: %s/summary.txt" % out)
    for name in ("01_c_accept/summary.txt", "02_b_equiv/summary.txt", "03_optional/cmp_patched_vs_baseline.txt"):
        path = out / name
        if path.is_file():
            rep.log("send back: %s" % path)
    rep.close()
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
