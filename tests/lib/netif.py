#!/usr/bin/env python3
"""网卡 / IP 自动识别（容器内也可用，不依赖 iproute2 / net-tools）。

    python3 tests/lib/netif.py            # 候选列表：iface<TAB>ip<TAB>default?
    python3 tests/lib/netif.py --best     # 最佳候选：iface<TAB>ip（没有则 exit 1）
    python3 tests/lib/netif.py --list-json

识别顺序（合并去重）：psutil.net_if_addrs() -> ip -o -4 addr show -> ifconfig -a；
默认路由接口从 /proc/net/route 读（容器里通常只有一块 eth0，取到的就是它）。
排序：默认路由接口 -> eth/en/ens/bond/ib 类 -> 其它；过滤 lo/docker*/veth*/br-*/virbr*/cni* 与
127.*/169.254.*。
"""

from __future__ import annotations

import ipaddress
import json
import re
import subprocess
import sys
from pathlib import Path

SKIP_IFACE = re.compile(r"^(lo|docker|veth|br-|virbr|cni|kube|flannel|tunl|cali)")
PREFER_IFACE = re.compile(r"^(eth|en|ens|enp|bond|ib|roce)")
VIRTUAL_IFACE = re.compile(r"^(docker|veth|br-|virbr|cni|kube|flannel|tunl|cali)")


def default_route_iface() -> str | None:
    """Default-route interface name from /proc/net/route (no shell needed)."""
    try:
        lines = Path("/proc/net/route").read_text().splitlines()[1:]
    except OSError:
        return None
    for line in lines:
        fields = line.split()
        if len(fields) >= 3 and fields[1] == "00000000":  # Destination == default
            return fields[0]
    return None


def _from_ioctl() -> list[tuple[str, str]]:
    """Pure-python SIOCGIFADDR scan: works in containers without ip/ifconfig."""
    try:
        import fcntl  # noqa: PLC0415 - Linux only
        import socket  # noqa: PLC0415
        import struct  # noqa: PLC0415
    except Exception:
        return []
    out: list[tuple[str, str]] = []
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        names = socket.if_nameindex()
    except OSError:
        return []
    try:
        for _idx, name in names:
            try:
                packed = struct.pack("256s", name[:15].encode())
                addr = fcntl.ioctl(sock.fileno(), 0x8915, packed)  # SIOCGIFADDR
            except OSError:
                continue
            out.append((name, socket.inet_ntoa(addr[20:24])))
    finally:
        sock.close()
    return out


def _from_psutil() -> list[tuple[str, str]]:
    try:
        import psutil  # noqa: PLC0415 - optional dependency
    except Exception:
        return []
    out: list[tuple[str, str]] = []
    try:
        addrs = psutil.net_if_addrs()
    except Exception:
        return []
    for iface, entries in addrs.items():
        for entry in entries:
            family = getattr(entry, "family", None)
            if family is not None and family.name != "AF_INET":
                continue
            ip = entry.address or ""
            if ip and ":" not in ip:
                out.append((iface, ip))
    return out


def _from_ip_command() -> list[tuple[str, str]]:
    try:
        text = subprocess.run(
            ["ip", "-o", "-4", "addr", "show"], capture_output=True, text=True, timeout=5, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    out: list[tuple[str, str]] = []
    for line in text.splitlines():
        # "3: eth0    inet 10.0.0.5/24 brd ... scope global eth0"
        parts = line.split()
        if len(parts) >= 4 and parts[2] == "inet":
            out.append((parts[1], parts[3].split("/")[0]))
    return out


def _from_ifconfig() -> list[tuple[str, str]]:
    try:
        text = subprocess.run(["ifconfig", "-a"], capture_output=True, text=True, timeout=5, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    out: list[tuple[str, str]] = []
    current = None
    for line in text.splitlines():
        if line and not line[0].isspace():
            current = line.split(":")[0].split()[0]
        match = re.search(r"inet (?:addr:)?(\d+\.\d+\.\d+\.\d+)", line)
        if current and match:
            out.append((current, match.group(1)))
    return out


def _from_hostname() -> list[tuple[str, str]]:
    """Last resort: 'hostname -I' has IPs but no interface names."""
    try:
        text = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=5, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [("", ip) for ip in text.split()]


def candidates() -> list[dict]:
    default = default_route_iface()
    seen: dict[tuple[str, str], dict] = {}
    for source in (
        _from_ioctl(),
        _from_psutil(),
        _from_ip_command(),
        _from_ifconfig(),
        _from_hostname(),
    ):
        for iface, ip in source:
            if not iface or SKIP_IFACE.match(iface) or VIRTUAL_IFACE.match(iface):
                continue
            try:
                addr = ipaddress.IPv4Address(ip)
            except ValueError:
                continue
            if addr.is_loopback or addr.is_link_local:
                continue
            key = (iface, ip)
            seen[key] = {"iface": iface, "ip": ip, "default": iface == default}

    def sort_key(item: dict) -> tuple:
        return (
            0 if item["default"] else 1,
            0 if PREFER_IFACE.match(item["iface"]) else 1,
            item["iface"],
        )

    return sorted(seen.values(), key=sort_key)


def best() -> dict | None:
    found = candidates()
    return found[0] if found else None


def main() -> int:
    if "--best" in sys.argv:
        item = best()
        if item is None:
            print("no usable IPv4 interface found", file=sys.stderr)
            return 1
        print(f"{item['iface']}\t{item['ip']}")
        return 0
    if "--list-json" in sys.argv:
        print(json.dumps(candidates(), ensure_ascii=False))
        return 0
    found = candidates()
    if not found:
        print("no usable IPv4 interface found", file=sys.stderr)
        return 1
    for item in found:
        print(f"{item['iface']}\t{item['ip']}\t{'default' if item['default'] else ''}".rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
