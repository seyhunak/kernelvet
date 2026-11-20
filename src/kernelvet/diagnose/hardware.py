"""GPU discovery from ``amd-smi``, ``rocminfo`` and ``/sys``.

Sources are merged by PCI bus address when available, otherwise by order. A GPU
is only reported when some source actually lists it; nothing is invented.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from kernelvet.models.common import Evidence
from kernelvet.models.system import GpuDevice
from kernelvet.utils.proc import CommandResult, HostContext

AMD_PCI_VENDOR = "0x1002"


# --------------------------------------------------------------------------- rocminfo


@dataclass
class RocminfoAgent:
    name: str | None = None
    marketing_name: str | None = None
    vendor: str | None = None
    device_type: str | None = None
    compute_units: int | None = None
    node: int | None = None
    uuid: str | None = None
    chip_id: str | None = None
    bdf: str | None = None
    pool_bytes: list[int] = field(default_factory=list)


_KV = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 ()/_.-]*?):\s+(.*\S)\s*$")


def parse_rocminfo(text: str) -> list[RocminfoAgent]:
    """Parse ``rocminfo`` output into agents (CPU and GPU).

    Agent blocks start with ``Agent N`` and are separated by asterisk rules.
    Memory pools are nested in each agent; we collect the GLOBAL pool sizes
    so VRAM can be inferred when amd-smi is unavailable.
    """
    agents: list[RocminfoAgent] = []
    current: RocminfoAgent | None = None
    in_pool = False
    pool_segment: str | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if re.match(r"^\s*Agent \d+\s*$", line):
            current = RocminfoAgent()
            agents.append(current)
            in_pool = False
            continue
        if current is None:
            continue
        if re.match(r"^\s*Pool \d+\s*$", line):
            in_pool = True
            pool_segment = None
            continue
        if re.match(r"^\s*ISA Info:", line):
            in_pool = False
            continue
        m = _KV.match(line)
        if not m:
            continue
        key, value = m.group(1).strip(), m.group(2).strip()
        if in_pool:
            if key == "Segment":
                pool_segment = value
            elif key == "Size" and pool_segment and pool_segment.startswith("GLOBAL"):
                size = _parse_size(value)
                if size:
                    current.pool_bytes.append(size)
            continue
        if key == "Name" and current.name is None:
            current.name = value
        elif key == "Marketing Name":
            current.marketing_name = value
        elif key == "Vendor Name":
            current.vendor = value
        elif key == "Device Type":
            current.device_type = value
        elif key == "Compute Unit":
            current.compute_units = _int(value)
        elif key == "Node":
            current.node = _int(value)
        elif key == "Uuid":
            current.uuid = value
        elif key == "Chip ID":
            current.chip_id = value.split("(")[-1].rstrip(")") if "(" in value else value
        elif key == "BDFID":
            current.bdf = _bdf_from_id(value)
    return agents


def _int(value: str) -> int | None:
    m = re.match(r"-?\d+", value)
    return int(m.group(0)) if m else None


def _parse_size(value: str) -> int | None:
    m = re.match(r"(\d+)\s*\(?\s*0x[0-9a-fA-F]+\s*\)?\s*(KB|MB|GB)?", value)
    if not m:
        m = re.match(r"(\d+)\s*(KB|MB|GB)?", value)
    if not m:
        return None
    n = int(m.group(1))
    unit = (m.group(2) or "KB").upper()
    return n * {"KB": 1024, "MB": 1024**2, "GB": 1024**3}[unit]


def _bdf_from_id(value: str) -> str | None:
    """rocminfo prints BDFID as a decimal int: bus<<8 | device<<3 | function."""
    n = _int(value)
    if n is None:
        return None
    bus, dev, fn = (n >> 8) & 0xFF, (n >> 3) & 0x1F, n & 0x7
    return f"{bus:02x}:{dev:02x}.{fn}"


def gpu_agents(agents: list[RocminfoAgent]) -> list[RocminfoAgent]:
    return [a for a in agents if (a.device_type or "").upper() == "GPU"]


# --------------------------------------------------------------------------- amd-smi


def _unwrap(value: Any) -> Any:
    """amd-smi wraps numbers as {"value": 12, "unit": "MB"} in newer releases."""
    if isinstance(value, dict) and "value" in value:
        return value["value"], value.get("unit")
    return value, None


def _mem_bytes(value: Any) -> int | None:
    v, unit = _unwrap(value)
    if v in (None, "N/A"):
        return None
    try:
        n = float(str(v).split()[0])
    except ValueError:
        return None
    unit = (unit or (str(v).split()[1] if len(str(v).split()) > 1 else "MB")).upper()
    mult = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "KIB": 1024, "MIB": 1024**2, "GIB": 1024**3}
    return int(n * mult.get(unit, 1024**2))


def parse_amd_smi_static(text: str) -> list[dict[str, Any]]:
    """Parse ``amd-smi static --json`` (list or ``{"gpu_data": [...]}``)."""
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("gpu_data") or data.get("gpus") or [data]
    out: list[dict[str, Any]] = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            continue
        asic = entry.get("asic") or {}
        bus = entry.get("bus") or {}
        vram = entry.get("vram") or {}
        board = entry.get("board") or {}
        out.append(
            {
                "index": entry.get("gpu", i),
                "marketing_name": asic.get("market_name") or board.get("product_name"),
                "vendor": asic.get("vendor_name"),
                "vendor_id": str(asic.get("vendor_id") or "").lower() or None,
                "device_id": asic.get("device_id"),
                "gfx_target": asic.get("target_graphics_version"),
                "compute_units": _unwrap(asic.get("num_compute_units"))[0],
                "uuid": asic.get("asic_serial") or entry.get("uuid"),
                "pci_bus": _norm_bdf(bus.get("bdf")),
                "vram_bytes": _mem_bytes(vram.get("size")),
            }
        )
    return out


def _norm_bdf(bdf: str | None) -> str | None:
    if not bdf:
        return None
    bdf = bdf.lower()
    # "0000:03:00.0" -> "03:00.0"
    return bdf.split(":", 1)[1] if bdf.count(":") == 2 else bdf


# --------------------------------------------------------------------------- /sys fallback


def sysfs_amd_gpus(ctx: HostContext) -> list[dict[str, Any]]:
    """AMD display/compute devices visible in /sys/class/drm (no ROCm required)."""
    found: list[dict[str, Any]] = []
    for card in ctx.listdir("/sys/class/drm"):
        if not re.fullmatch(r"card\d+", card):
            continue
        base = f"/sys/class/drm/{card}/device"
        vendor = (ctx.read_text(f"{base}/vendor") or "").strip().lower()
        if vendor != AMD_PCI_VENDOR:
            continue
        vram = (ctx.read_text(f"{base}/mem_info_vram_total") or "").strip()
        uevent = ctx.read_text(f"{base}/uevent") or ""
        slot = re.search(r"PCI_SLOT_NAME=(\S+)", uevent)
        found.append(
            {
                "card": card,
                "device_id": (ctx.read_text(f"{base}/device") or "").strip() or None,
                "vram_bytes": int(vram) if vram.isdigit() else None,
                "pci_bus": _norm_bdf(slot.group(1)) if slot else None,
            }
        )
    return found


# --------------------------------------------------------------------------- merge


@dataclass
class HardwareProbe:
    gpus: list[GpuDevice]
    rocminfo: CommandResult | None = None
    amd_smi: CommandResult | None = None
    sysfs_count: int = 0
    kfd_accessible: bool | None = None
    notes: list[str] = field(default_factory=list)


def _device_access(ctx: HostContext) -> bool | None:
    """Can this user open /dev/kfd and a render node? None when not on Linux."""
    if ctx.system != "Linux":
        return None
    kfd = ctx.path("/dev/kfd")
    if not kfd.exists():
        return False
    return os.access(kfd, os.R_OK | os.W_OK)


def discover_gpus(ctx: HostContext) -> HardwareProbe:
    probe = HardwareProbe(gpus=[])
    smi_entries: list[dict[str, Any]] = []
    if ctx.which("amd-smi"):
        res = ctx.run(["amd-smi", "static", "--json"])
        probe.amd_smi = res
        if res.ok and res.stdout.strip():
            try:
                smi_entries = parse_amd_smi_static(res.stdout)
            except (json.JSONDecodeError, TypeError) as exc:
                probe.notes.append(f"amd-smi output could not be parsed: {exc}")
        elif not res.ok:
            probe.notes.append(res.describe_failure())

    agents: list[RocminfoAgent] = []
    if ctx.which("rocminfo"):
        res = ctx.run(["rocminfo"])
        probe.rocminfo = res
        if res.ok:
            agents = gpu_agents(parse_rocminfo(res.stdout))
        else:
            probe.notes.append(res.describe_failure())

    sysfs = sysfs_amd_gpus(ctx) if ctx.system == "Linux" else []
    probe.sysfs_count = len(sysfs)
    probe.kfd_accessible = _device_access(ctx)

    by_bus: dict[str, GpuDevice] = {}
    ordered: list[GpuDevice] = []

    def upsert(bus: str | None, idx: int) -> GpuDevice:
        if bus and bus in by_bus:
            return by_bus[bus]
        if not bus and idx < len(ordered):
            return ordered[idx]
        dev = GpuDevice(index=len(ordered), pci_bus=bus, evidence=Evidence.DETECTED)
        ordered.append(dev)
        if bus:
            by_bus[bus] = dev
        return dev

    for i, e in enumerate(smi_entries):
        d = upsert(e.get("pci_bus"), i)
        d.marketing_name = d.marketing_name or e.get("marketing_name")
        d.vendor = d.vendor or e.get("vendor")
        d.gfx_target = d.gfx_target or e.get("gfx_target")
        d.compute_units = d.compute_units or _safe_int(e.get("compute_units"))
        d.vram_bytes = d.vram_bytes or e.get("vram_bytes")
        d.device_id = d.device_id or e.get("device_id")
        d.uuid = d.uuid or e.get("uuid")
        d.sources.append("amd-smi")
        vid = e.get("vendor_id") or ""
        if (
            vid in ("0x1002", "1002")
            or "amd" in (d.vendor or "").lower()
            or "advanced micro" in (d.vendor or "").lower()
        ):
            d.vendor_is_amd = True

    for i, a in enumerate(agents):
        d = upsert(a.bdf, i)
        d.marketing_name = d.marketing_name or a.marketing_name
        d.vendor = d.vendor or a.vendor
        d.gfx_target = d.gfx_target or a.name
        d.compute_units = d.compute_units or a.compute_units
        d.node_id = a.node
        d.uuid = d.uuid or a.uuid
        if d.vram_bytes is None and a.pool_bytes:
            d.vram_bytes = max(a.pool_bytes)
        d.sources.append("rocminfo")
        if (a.vendor or "").upper() == "AMD":
            d.vendor_is_amd = True

    for i, s in enumerate(sysfs):
        sd = upsert(s.get("pci_bus"), i) if s.get("pci_bus") else None
        if sd is None:
            continue
        d = sd
        d.vendor_is_amd = True
        d.vendor = d.vendor or "AMD"
        d.device_id = d.device_id or s.get("device_id")
        d.vram_bytes = d.vram_bytes or s.get("vram_bytes")
        d.sources.append("sysfs")

    for d in ordered:
        d.accessible = probe.kfd_accessible
    probe.gpus = ordered
    return probe


def _safe_int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
