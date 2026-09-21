#!/usr/bin/env python3
"""分支诊断：从一轮 tests/_out 证据里判断 zigzag 到底有没有跑起来。

    diagnose.py report <out_dir> <report_file>   # 写报告，同时把结论打到 stdout
    diagnose.py verdict <out_dir>                # 只打结论

判据、grep 段、已知失败都来自 harness.json verify.diagnose，脚本里不再写死日志格式。

退出码：0 = 有 zigzag 证据；1 = 没有任何证据；3 = 命中已知失败（信息更具体）。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import serve_config as sc  # noqa: E402


def diagnose_cfg() -> dict:
    return (sc.harness().get("verify") or {}).get("diagnose") or {}


def all_text(out_dir: Path) -> str:
    """把一轮证据目录里的文本读成一份（不依赖外部 grep）。"""
    chunks = []
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "\n".join(chunks).replace("\r", "")


def section_lines(text: str, pattern: str) -> list:
    if not pattern:
        return []
    try:
        return [match.group(0) for match in re.finditer(pattern, text)]
    except re.error as exc:
        return ["(bad pattern %r: %s)" % (pattern, exc)]


def section_body(text: str, pattern: str, post: str) -> str:
    lines = [line for line in section_lines(text, pattern) if line.strip()]
    if post == "sort_count":
        counts: dict = {}
        for line in lines:
            counts[line] = counts.get(line, 0) + 1
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return "\n".join("%6d %s" % (count, line) for line, count in ordered)
    if post == "head5":
        return "\n".join(lines[:5])
    if post == "sort_uniq_head20":
        return "\n".join(sorted(set(lines))[:20])
    return "\n".join(lines)


def write_report(out_dir: Path, report: Path, text: str) -> None:
    cfg = diagnose_cfg()
    chunks = ["# cp_balance 分支诊断", "# 证据目录: %s" % out_dir]
    for section in cfg.get("sections") or []:
        body = section_body(text, str(section.get("grep", "")), str(section.get("post", "")))
        chunks += ["", "## " + str(section.get("title", "")), body or "(无)"]
    report.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def verdict(text: str) -> tuple:
    """(code, detail) — code: 0 evidence, 1 nothing, 3 known failure."""
    cfg = diagnose_cfg()
    evidence = [str(item) for item in cfg.get("zigzag_evidence") or ["branch=ZIGZAG"]]
    hits = {pattern: text.count(pattern) for pattern in evidence}
    found = [pattern for pattern, count in hits.items() if count > 0]
    if found:
        return 0, "zigzag 证据命中: " + ", ".join("%s x%d" % (pat, hits[pat]) for pat in found)
    for known in cfg.get("known") or []:
        match = str(known.get("match", ""))
        if match and match in text:
            return 3, "%s -> %s" % (match, known.get("message", ""))
    branch_lines = len(re.findall(r"\[CP_BALANCE\]\[branch\]", text))
    return 1, "没有任何 zigzag 证据（evidence=%s，日志里 [CP_BALANCE][branch] 行=%d）" % (evidence, branch_lines)


def main() -> int:
    if len(sys.argv) < 3 or sys.argv[1] not in ("report", "verdict"):
        print(__doc__, file=sys.stderr)
        return 2
    mode, out_dir = sys.argv[1], Path(sys.argv[2])
    if not out_dir.is_dir():
        print("[diagnose] FAIL 证据目录不存在: %s" % out_dir, file=sys.stderr)
        return 1
    text = all_text(out_dir)
    if mode == "report":
        report = Path(sys.argv[3])
        write_report(out_dir, report, text)
        print(report.read_text(encoding="utf-8"))
    code, detail = verdict(text)
    label = {0: "OK", 3: "KNOWN", 1: "MISSING"}[code]
    print("[diagnose] VERDICT=%s %s" % (label, detail))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
