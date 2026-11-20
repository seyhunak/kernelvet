"""Operating system, kernel and architecture discovery."""

from __future__ import annotations

import hashlib
import platform
import socket

from kernelvet.models.system import HostInfo
from kernelvet.utils.proc import HostContext


def parse_os_release(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key.strip()] = value
    return out


def _in_container(ctx: HostContext) -> bool:
    if ctx.exists("/.dockerenv") or ctx.exists("/run/.containerenv"):
        return True
    cgroup = ctx.read_text("/proc/1/cgroup") or ""
    return any(tag in cgroup for tag in ("docker", "containerd", "kubepods", "libpod"))


def _hostname_hash() -> str:
    try:
        name = socket.gethostname()
    except OSError:
        return "unknown"
    return hashlib.sha256(name.encode()).hexdigest()[:12]


def collect_host(ctx: HostContext) -> HostInfo:
    os_name = ctx.system
    os_id = os_version = os_pretty = None
    if ctx.system == "Linux":
        rel = ctx.read_text("/etc/os-release") or ctx.read_text("/usr/lib/os-release")
        if rel:
            info = parse_os_release(rel)
            os_id = info.get("ID")
            os_version = info.get("VERSION_ID")
            os_pretty = info.get("PRETTY_NAME")
            os_name = info.get("NAME", "Linux")
    elif ctx.system == "Darwin":
        os_name, os_pretty = "macOS", f"macOS {platform.mac_ver()[0] or ''}".strip()
    elif ctx.system == "Windows":
        os_name, os_pretty = "Windows", f"Windows {platform.release()}"
    return HostInfo(
        os_name=os_name,
        os_id=os_id,
        os_version=os_version,
        os_pretty=os_pretty,
        kernel=ctx.release or None,
        architecture=ctx.machine or "unknown",
        hostname_hash=_hostname_hash(),
        python_version=platform.python_version(),
        in_container=ctx.system == "Linux" and _in_container(ctx),
    )
