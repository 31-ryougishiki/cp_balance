#!/usr/bin/env python3
"""Launch vllm serve from a JSON config under configs/.

Every test run is one config file.  A config may extend another one
("extends": "_common.json"): dict entries are merged, lists are replaced.

    python serve_config.py configs/glm52_cur_cp0.json --dry-run
    bash run.sh configs/glm52_cur_cp0.json

The "[cp_balance] CONFIG=... REPO=... HEAD=..." line printed before the launch
is the fingerprint used by run_matrix.py to prove which code tree and which
switch values a service actually used.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "configs"
DETERMINISTIC_ENV = {
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
}
REPORT_ENV = (
    "VLLM_ASCEND_REPO",
    "PYTHONPATH",
    "ASCEND_RT_VISIBLE_DEVICES",
    "HCCL_IF_IP",
    "HCCL_SOCKET_IFNAME",
    "HCCL_ALGO",
    "HCCL_BUFFSIZE",
    "VLLM_ASCEND_CP_BALANCE",
    "VLLM_ASCEND_CP_BALANCE_MIN_TOKENS",
    "VLLM_ASCEND_CP_BALANCE_REDUCE_MODE",
    "VLLM_ASCEND_CP_BALANCE_DEBUG",
    "LCCL_DETERMINISTIC",
    "HCCL_DETERMINISTIC",
    "ATB_MATMUL_SHUFFLE_K_ENABLE",
    "ATB_LLM_LCOC_ENABLE",
)


def resolve(ref: str) -> Path:
    path = Path(ref)
    if path.is_file():
        return path
    for candidate in (CONFIG_DIR / ref, CONFIG_DIR / (ref + ".json")):
        if candidate.is_file():
            return candidate
    raise SystemExit("config not found: " + str(ref))


def merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    return out


def load_config(ref: str, seen: tuple = ()) -> dict:
    path = resolve(ref)
    if str(path) in seen:
        raise SystemExit("config extends cycle at " + str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    parent = data.pop("extends", None)
    if parent:
        data = merge(load_config(parent, seen + (str(path),)), data)
    data.setdefault("name", path.stem)
    return data


def apply_set(cfg: dict, expr: str) -> None:
    key, sep, raw = expr.partition("=")
    if not sep:
        raise SystemExit("--set expects KEY=VALUE, got " + expr)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    node = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def build_env(cfg: dict) -> dict:
    env = dict(os.environ)
    for key, value in (cfg.get("env") or {}).items():
        env[str(key)] = str(value)
    if cfg.get("local_ip"):
        env["HCCL_IF_IP"] = str(cfg["local_ip"])
    if cfg.get("nic_name"):
        for key in ("GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME", "HCCL_SOCKET_IFNAME"):
            env[key] = str(cfg["nic_name"])
    if cfg.get("devices"):
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(cfg["devices"])
    repo = cfg.get("repo")
    if repo:
        env["VLLM_ASCEND_REPO"] = str(repo)
        env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    if cfg.get("cp_balance") is not None:
        env["VLLM_ASCEND_CP_BALANCE"] = str(int(cfg["cp_balance"]))
    if cfg.get("min_tokens") is not None:
        env["VLLM_ASCEND_CP_BALANCE_MIN_TOKENS"] = str(int(cfg["min_tokens"]))
    if cfg.get("reduce_mode"):
        env["VLLM_ASCEND_CP_BALANCE_REDUCE_MODE"] = str(cfg["reduce_mode"])
    if cfg.get("debug") is not None:
        env["VLLM_ASCEND_CP_BALANCE_DEBUG"] = str(int(cfg["debug"]))
    if cfg.get("deterministic"):
        env.update(DETERMINISTIC_ENV)
    if cfg.get("pythonunbuffered", True):
        env["PYTHONUNBUFFERED"] = "1"
    return env


def build_argv(cfg: dict) -> list:
    if not cfg.get("model"):
        raise SystemExit("config is missing 'model'")
    if cfg.get("port") is None:
        raise SystemExit("config is missing 'port'")
    bin_cfg = cfg.get("vllm_bin", "vllm")
    argv = bin_cfg.split() if isinstance(bin_cfg, str) else [str(item) for item in bin_cfg]
    argv += ["serve", str(cfg["model"])]
    argv += ["--host", str(cfg.get("host", "0.0.0.0"))]  # noqa: E501
    argv += ["--port", str(int(cfg["port"]))]
    if cfg.get("tp_size") is not None:
        argv += ["--tensor-parallel-size", str(int(cfg["tp_size"]))]
    if cfg.get("served_model_name"):
        argv += ["--served-model-name", *str(cfg["served_model_name"]).split()]
    if cfg.get("additional_config") is not None:
        argv += ["--additional_config", json.dumps(cfg["additional_config"], ensure_ascii=False)]
    argv += [str(item) for item in (cfg.get("server_args") or [])]
    return argv


def git_head(repo) -> str:
    if not repo:
        return "unknown"
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unknown"
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def fingerprint(cfg: dict, env: dict) -> str:
    return (
        "[cp_balance] CONFIG=%s REPO=%s HEAD=%s MODEL=%s PORT=%s TP=%s NIC=%s IP=%s DEVICES=%s "
        "CP_BALANCE=%s MIN_TOKENS=%s REDUCE_MODE=%s DEBUG=%s DET=%s"
        % (
            cfg.get("name"),
            cfg.get("repo"),
            git_head(cfg.get("repo")),
            cfg.get("model"),
            cfg.get("port"),
            cfg.get("tp_size"),
            cfg.get("nic_name"),
            cfg.get("local_ip"),
            cfg.get("devices"),
            env.get("VLLM_ASCEND_CP_BALANCE"),
            env.get("VLLM_ASCEND_CP_BALANCE_MIN_TOKENS"),
            env.get("VLLM_ASCEND_CP_BALANCE_REDUCE_MODE"),
            env.get("VLLM_ASCEND_CP_BALANCE_DEBUG"),
            bool(cfg.get("deterministic")),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", default="default", help="config path or name under configs/")
    parser.add_argument("--dry-run", action="store_true", help="print fingerprint/env/argv without launching")
    parser.add_argument("--print-env", action="store_true", help="with --dry-run, print the effective env")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config field")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for expr in args.set:
        apply_set(cfg, expr)
    env = build_env(cfg)
    argv = build_argv(cfg)
    print(fingerprint(cfg, env), flush=True)
    if args.dry_run:
        if args.print_env:
            for key in REPORT_ENV:
                if key in env:
                    print("  env %s=%s" % (key, env[key]))
        print("  argv " + " ".join(shlex.quote(item) for item in argv))
        return 0
    proc = subprocess.Popen(argv, env=env)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
