#!/usr/bin/env python3
"""Launch vllm serve from a JSON config under configs/.

Every test run is one config file.  A config may extend one or more others
("extends": "_common.json" or ["_common.json", "_profile_base.json"]): dict
entries are merged, lists are replaced, later parents win.

`server_args` is either a list (used as is) or a dict of option -> value
(true = bare flag, false = dropped), so a shared base can list the common
options once and a machine profile only overrides what differs.

    python serve_config.py configs/glm52_cur_cp0.json --dry-run
    bash run.sh configs/glm52_cur_cp0.json

Harness-wide defaults (service argv defaults, deterministic env, limits) live
in harness.json; machine identity and model stay in configs/*.json.

The "[cp_balance] CONFIG=... REPO=... HEAD=..." line printed before the launch
is the fingerprint used by run_matrix.py to prove which code tree and which
switch values a service actually used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# 远端常设 http_proxy：本机地址要绕开它，否则就绪探测/接口调用会被发到代理上
for _proxy_key in ("no_proxy", "NO_PROXY"):
    _parts = [part for part in os.environ.get(_proxy_key, "").split(",") if part]
    for _host in ("127.0.0.1", "localhost", "::1"):
        if _host not in _parts:
            _parts.append(_host)
    os.environ[_proxy_key] = ",".join(_parts)
CONFIG_DIR = HERE / "configs"
HARNESS_CONFIG = HERE / "harness.json"
# Defaults only; a config's "deterministic_env" (or harness.json serve.deterministic_env)
# replaces them, because the knobs differ per vendor/CANN on each machine.
DETERMINISTIC_ENV = {
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
}
SERVE_DEFAULTS = {
    "host": "0.0.0.0",
    "vllm_bin": "vllm",
    "pythonunbuffered": True,
}
KNOWN_KEYS = {
    "name", "extends", "repo", "repo_tree", "model", "served_model_name", "host", "port", "devices",
    "local_ip", "nic_name", "tp_size", "cp_balance", "min_tokens", "debug",
    "deterministic", "deterministic_env", "additional_config", "hf_overrides",
    "speculative_config", "server_args", "env", "prelude", "profiler", "lengths",
    "pythonunbuffered", "vllm_bin",
    # matrix configs
    "configs", "compare", "static_check", "require_log", "ready_timeout",
}
# Machine fields that may differ per host (same configs, several servers, or a
# shared checkout).  Set them in the shell before launching the harness; the
# command line --set still wins over them.
ENV_OVERRIDES = (
    ("model", "CP_BALANCE_MODEL"),
    ("local_ip", "CP_BALANCE_LOCAL_IP"),
    ("nic_name", "CP_BALANCE_NIC_NAME"),
    ("devices", "CP_BALANCE_DEVICES"),
)
def harness() -> dict:
    """harness.json: defaults and limits shared by every launcher/test."""
    if not HARNESS_CONFIG.is_file():
        return {}
    return json.loads(HARNESS_CONFIG.read_text(encoding="utf-8"))


def limit(name: str, default=None):
    """harness.json limits.<name>（脚本/驱动的默认超时与阈值都从这里取）。"""
    return (harness().get("limits") or {}).get(name, default)


def serve_defaults() -> dict:
    out = dict(SERVE_DEFAULTS)
    out.update(harness().get("serve") or {})
    return out


def tree_path(role: str) -> str:
    """harness.json trees.<role>.path, absolutized against the harness dir."""
    entry = (harness().get("trees") or {}).get(str(role)) or {}
    path = entry.get("path")
    if not path:
        raise SystemExit("harness.json has no tree role %r (configs must not hardcode tree paths)" % role)
    return str(path) if _is_abs(path) else str((HERE / str(path)).resolve())


def resolve(ref: str) -> Path:
    # A bare name means "under configs/", so a same-named file in the cwd (or in
    # tests/_out) cannot shadow it; paths with a separator are used as given.
    if "/" in ref or "\\" in ref or _is_abs(ref):
        path = Path(ref)
        if path.is_file():
            return path
    else:
        for candidate in (CONFIG_DIR / ref, CONFIG_DIR / (ref + ".json")):
            if candidate.is_file():
                return candidate
    path = Path(ref)
    if path.is_file():
        return path
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
    data.setdefault("name", path.stem)   # before the merge: the child owns its name
    parents = data.pop("extends", None)
    if parents:
        if isinstance(parents, str):
            parents = [parents]
        base: dict = {}
        for parent in parents:
            base = merge(base, load_config(parent, seen + (str(path),)))
        data = merge(base, data)
    return data


def unknown_keys(cfg: dict) -> list:
    return sorted(key for key in cfg if key not in KNOWN_KEYS)


INT_KEYS = {"port", "tp_size", "cp_balance", "min_tokens", "debug"}


def apply_set(cfg: dict, expr: str) -> None:
    key, sep, raw = expr.partition("=")
    if not sep:
        raise SystemExit("--set expects KEY=VALUE, got " + expr)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    if key in INT_KEYS:
        try:
            value = int(str(value).strip())
        except ValueError:
            raise SystemExit("--set %s=%s: expected an integer" % (key, raw)) from None
    node = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def detect_netif() -> tuple[str | None, str | None]:
    """(iface, ip) of the best local IPv4 link; (None, None) when unknown."""
    sys.path.insert(0, str(HERE / "tests" / "lib"))
    try:
        import netif  # noqa: PLC0415 - harness helper next to this file
    except Exception:
        return None, None
    item = netif.best()
    if not item:
        return None, None
    return item["iface"], item["ip"]


def _is_abs(value) -> bool:
    text = str(value)
    return os.path.isabs(text) or text.startswith("/")


def absolutize(cfg: dict) -> dict:
    """Resolve tree / model / profiler paths relative to the harness dir.

    The checkouts normally sit next to the harness (../vllm-ascend), which makes
    the configs machine independent; PYTHONPATH and the service argv still need
    absolute paths.  `repo_tree` names a role in harness.json trees instead of
    repeating the path in every config.
    """
    if cfg.get("repo_tree"):
        cfg["repo"] = tree_path(cfg["repo_tree"])
    for key in ("repo", "model"):
        value = cfg.get(key)
        if value and not _is_abs(value):
            cfg[key] = str((HERE / str(value)).resolve())
    profiler = cfg.get("profiler")
    if isinstance(profiler, dict) and profiler.get("dir") and not _is_abs(profiler["dir"]):
        profiler["dir"] = str((HERE / str(profiler["dir"])).resolve())
    return cfg


def _is_auto(value) -> bool:
    return value is None or str(value).strip() in ("", "auto")


def apply_env_overrides(cfg: dict) -> dict:
    for key, env_name in ENV_OVERRIDES:
        value = os.environ.get(env_name)
        if value:
            cfg[key] = value
    # Machine fields may be left as "auto" (or unset) on hosts/containers where
    # the IP and NIC cannot be written down in advance.
    if _is_auto(cfg.get("local_ip")) or _is_auto(cfg.get("nic_name")):
        iface, ip = detect_netif()
        if _is_auto(cfg.get("local_ip")) and ip:
            cfg["local_ip"] = ip
        if _is_auto(cfg.get("nic_name")) and iface:
            cfg["nic_name"] = iface
    if _is_auto(cfg.get("devices")):
        env_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        if env_devices:
            cfg["devices"] = env_devices
    return cfg


def build_env(cfg: dict, managed: dict | None = None) -> dict:
    """Effective env for the service; `managed` collects what this launcher set.

    The collected names are what --print-env reports, so a new config env key
    shows up without touching a hardcoded list here.
    """
    defaults = serve_defaults()
    env = dict(os.environ)

    def put(key: str, value: str) -> None:
        env[key] = value
        if managed is not None:
            managed[key] = value

    for key, value in (cfg.get("env") or {}).items():
        put(str(key), str(value))
    if cfg.get("local_ip") and not _is_auto(cfg.get("local_ip")):
        put("HCCL_IF_IP", str(cfg["local_ip"]))
    if cfg.get("nic_name") and not _is_auto(cfg.get("nic_name")):
        for key in ("GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME", "HCCL_SOCKET_IFNAME"):
            put(key, str(cfg["nic_name"]))
    if cfg.get("devices"):
        put("ASCEND_RT_VISIBLE_DEVICES", str(cfg["devices"]))
    repo = cfg.get("repo")
    if repo:
        put("VLLM_ASCEND_REPO", str(repo))
        # drop empty entries: a trailing ':' would add "" == cwd to sys.path
        put("PYTHONPATH", os.pathsep.join(part for part in (str(repo), env.get("PYTHONPATH", "")) if part))
    if cfg.get("cp_balance") is not None:
        put("VLLM_ASCEND_CP_BALANCE", str(int(cfg["cp_balance"])))
    if cfg.get("min_tokens") is not None:
        put("VLLM_ASCEND_CP_BALANCE_MIN_TOKENS", str(int(cfg["min_tokens"])))
    if cfg.get("debug") is not None:
        put("VLLM_ASCEND_CP_BALANCE_DEBUG", str(int(cfg["debug"])))
    if cfg.get("deterministic"):
        deterministic = dict(DETERMINISTIC_ENV)
        deterministic.update(harness().get("serve", {}).get("deterministic_env") or {})
        deterministic.update(cfg.get("deterministic_env") or {})
        for key, value in deterministic.items():
            put(str(key), str(value))
    if cfg.get("pythonunbuffered", defaults["pythonunbuffered"]):
        put("PYTHONUNBUFFERED", "1")
    return env


def expand_server_args(value) -> list:
    """list form (used as is) or dict form (true = bare flag, false = dropped)."""
    if not isinstance(value, dict):
        return [str(item) for item in (value or [])]
    argv: list = []
    for option, val in value.items():
        if val is False or val is None:
            continue
        if val is True:
            argv.append(str(option))
        elif isinstance(val, list):
            argv += [str(option), *[str(item) for item in val]]
        else:
            argv += [str(option), str(val)]
    return argv


def build_argv(cfg: dict) -> list:
    name = cfg.get("name")
    if cfg.get("configs") or cfg.get("compare"):
        raise SystemExit("%s is a matrix definition, not a service config" % name)
    if not cfg.get("model"):
        raise SystemExit("%s is missing 'model' (a service config needs model/port/tp_size)" % name)
    if cfg.get("port") is None:
        raise SystemExit("%s is missing 'port'" % name)
    defaults = serve_defaults()
    bin_cfg = cfg.get("vllm_bin", defaults["vllm_bin"])
    argv = bin_cfg.split() if isinstance(bin_cfg, str) else [str(item) for item in bin_cfg]
    argv += ["serve", str(cfg["model"])]
    argv += ["--host", str(cfg.get("host") or defaults["host"])]
    argv += ["--port", str(int(cfg["port"]))]
    if cfg.get("tp_size") is not None:
        argv += ["--tensor-parallel-size", str(int(cfg["tp_size"]))]
    if cfg.get("served_model_name"):
        argv += ["--served-model-name", *str(cfg["served_model_name"]).split()]
    if cfg.get("additional_config") is not None:
        argv += ["--additional_config", json.dumps(cfg["additional_config"], ensure_ascii=False)]
    if cfg.get("hf_overrides"):
        argv += ["--hf-overrides", json.dumps(cfg["hf_overrides"], ensure_ascii=False)]
    if cfg.get("speculative_config") is not None:
        argv += ["--speculative-config", json.dumps(cfg["speculative_config"], ensure_ascii=False)]
    if (cfg.get("profiler") or {}).get("enabled"):
        argv += ["--profiler-config", json.dumps(profiler_payload(cfg), ensure_ascii=False)]
    argv += expand_server_args(cfg.get("server_args"))
    return argv


def layer_override(cfg: dict) -> str:
    """``--hf-overrides`` layer count, or "all" for the real model."""
    value = (cfg.get("hf_overrides") or {}).get("num_hidden_layers")
    return str(int(value)) if value is not None else "all"


def profiler_dir(cfg: dict) -> str:
    profiler = cfg.get("profiler") or {}
    path = profiler.get("dir") or (HERE / str(cfg.get("name")))
    return str(Path(path).resolve())


def profiler_payload(cfg: dict) -> dict:
    """`--profiler-config` value; only the worker-side torch_npu fields."""
    profiler = cfg.get("profiler") or {}
    payload = {
        "profiler": "torch",
        "torch_profiler_dir": profiler_dir(cfg),
        "torch_profiler_with_stack": bool(profiler.get("with_stack", False)),
        "ignore_frontend": bool(profiler.get("ignore_frontend", True)),
    }
    # delay_iterations / max_iterations bound the capture window on the worker
    # side.  max_iterations=1 keeps one request's window to its prefill step
    # plus at most one decode, instead of everything that happens before
    # /stop_profile arrives.
    for key in ("delay_iterations", "max_iterations"):
        if profiler.get(key) is not None:
            payload[key] = int(profiler[key])
    return payload


def git_head(repo) -> str:
    if not repo:
        return "unknown"
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def spec_summary(cfg: dict) -> str:
    spec = cfg.get("speculative_config")
    if not spec:
        return "off"
    return "%s/%s" % (spec.get("method", "?"), spec.get("num_speculative_tokens", "?"))


def config_digest(cfg: dict) -> str:
    """Short digest of everything except model/port/tree, so two configs that
    differ only in env/args cannot look identical in the fingerprint."""
    payload = json.dumps(
        {
            "server_args": expand_server_args(cfg.get("server_args")),
            "env": cfg.get("env") or {},
            "additional_config": cfg.get("additional_config"),
            "speculative_config": cfg.get("speculative_config"),
            "hf_overrides": cfg.get("hf_overrides"),
            "prelude": cfg.get("prelude") or "",
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


ENV_FROM_FIELD = (
    ("VLLM_ASCEND_CP_BALANCE", "cp_balance"),
    ("VLLM_ASCEND_CP_BALANCE_MIN_TOKENS", "min_tokens"),
    ("VLLM_ASCEND_CP_BALANCE_DEBUG", "debug"),
    ("ASCEND_RT_VISIBLE_DEVICES", "devices"),
    ("HCCL_IF_IP", "local_ip"),
)


def env_field_conflicts(cfg: dict) -> list:
    """Keys a config sets both in `env` and as a top-level field (field wins)."""
    env = cfg.get("env") or {}
    return [
        "%s=%r in env vs %s=%r (the field wins)" % (key, env[key], field, cfg.get(field))
        for key, field in ENV_FROM_FIELD
        if key in env and cfg.get(field) is not None
    ]


def fingerprint(cfg: dict, env: dict) -> str:
    return (
        "[cp_balance] CONFIG=%s REPO=%s HEAD=%s MODEL=%s PORT=%s TP=%s NIC=%s IP=%s DEVICES=%s "
        "CP_BALANCE=%s MIN_TOKENS=%s DEBUG=%s DET=%s LAYERS=%s PROFILER=%s "
        "PRELUDE=%s SPEC=%s ARGS=%s"
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
            env.get("VLLM_ASCEND_CP_BALANCE_DEBUG"),
            bool(cfg.get("deterministic")),
            layer_override(cfg),
            profiler_dir(cfg) if (cfg.get("profiler") or {}).get("enabled") else "off",
            bool(cfg.get("prelude")),
            spec_summary(cfg),
            config_digest(cfg),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", default="default", help="config path or name under configs/")
    parser.add_argument("--dry-run", action="store_true", help="print fingerprint/env/argv without launching")
    parser.add_argument("--print-env", action="store_true", help="with --dry-run, print the effective env")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config field")
    args = parser.parse_args()

    if args.print_env and not args.dry_run:
        # --print-env is a diagnostic flag: never launch a 16-NPU service by accident
        args.dry_run = True
        print("[cp_balance] WARN --print-env implies --dry-run (nothing launched)", file=sys.stderr)
    cfg = apply_env_overrides(load_config(args.config))
    for expr in args.set:
        apply_set(cfg, expr)
    cfg = absolutize(cfg)   # after --set, so a relative --set path is resolved too
    for key in unknown_keys(cfg):
        print(
            "[cp_balance] WARN config %s has unknown key %r (typo? ignored by the launcher)"
            % (cfg.get("name"), key),
            file=sys.stderr,
        )
    for conflict in env_field_conflicts(cfg):
        print("[cp_balance] WARN config %s sets %s" % (cfg.get("name"), conflict), file=sys.stderr)
    for key in ("local_ip", "nic_name"):
        if _is_auto(cfg.get(key)):
            print(
                "[cp_balance] WARN config %s left %s unresolved (%r): HCCL_IF_IP / *_SOCKET_IFNAME "
                "will not be set -- export CP_BALANCE_%s or pin it in the config"
                % (cfg.get("name"), key, cfg.get(key), key.upper()),
                file=sys.stderr,
            )
    managed: dict = {}
    env = build_env(cfg, managed)
    argv = build_argv(cfg)
    prelude = cfg.get("prelude")
    if prelude:
        # prelude is a shell snippet that must run before the server, e.g.
        # "source /path/set_env.bash".  exec keeps the wrapper pid so the
        # harness can still stop the whole process group.
        argv = ["bash", "-c", str(prelude) + " && exec " + shlex.join(argv)]
    print(fingerprint(cfg, env), flush=True)
    if args.dry_run:
        if args.print_env:
            for key in sorted(managed):
                print("  env %s=%s" % (key, managed[key]))
        print("  argv " + " ".join(shlex.quote(item) for item in argv))
        return 0
    proc = subprocess.Popen(argv, env=env)

    def forward(_signum, _frame):
        # the service owns the NPUs: on SIGTERM/SIGINT stop it before exiting
        try:
            proc.terminate()
        except OSError:
            pass

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, forward)
        except (ValueError, OSError):   # not the main thread / unsupported
            pass
    try:
        status = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        status = proc.wait()
    return os.waitstatus_to_exitcode(status)


if __name__ == "__main__":
    raise SystemExit(main())
