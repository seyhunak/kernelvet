"""Individual compatibility validators.

Each validator returns one :class:`CheckResult`. The rules:

* PASS only when the detected value is explicitly listed in the matrix;
* FAIL only for a known incompatibility (explicitly unsupported GPU, wrong OS
  family, missing kernel driver on a GPU host);
* WARNING when the value is not listed or the data is ambiguous;
* SKIPPED when the input could not be detected.
"""

from __future__ import annotations

from dataclasses import dataclass

from kernelvet.compatibility.matrix import CompatibilityMatrix, GpuClass, OsEntry
from kernelvet.models.common import CheckResult, CheckStatus, Evidence
from kernelvet.models.system import GpuDevice, HostInfo, SoftwareInfo
from kernelvet.utils.versions import major_minor, parse_version, same_release

_RANK = {CheckStatus.PASS: 0, CheckStatus.SKIPPED: 1, CheckStatus.WARNING: 2, CheckStatus.FAIL: 3}


@dataclass
class CompatibilityInput:
    host: HostInfo
    gpus: list[GpuDevice]
    software: SoftwareInfo

    @property
    def amd_gpus(self) -> list[GpuDevice]:
        return [g for g in self.gpus if g.vendor_is_amd]


def _check(
    cid: str,
    name: str,
    status: CheckStatus,
    summary: str,
    *,
    detail: str = "",
    remediation: str | None = None,
    source: str | None = None,
) -> CheckResult:
    return CheckResult(
        id=cid,
        name=name,
        category="compatibility",
        status=status,
        evidence=Evidence.CONFIGURED,
        summary=summary,
        detail=detail,
        remediation=remediation,
        source=source,
    )


def gpu_classes(inp: CompatibilityInput, m: CompatibilityMatrix) -> set[GpuClass]:
    return {c for g in inp.amd_gpus if (c := m.class_for_gfx(g.gfx_target)) is not None}


# --------------------------------------------------------------------------- GPU


def validate_gpu(inp: CompatibilityInput, m: CompatibilityMatrix) -> CheckResult:
    cid, name = "compat.gpu", "GPU Architecture Support"
    gpus = inp.amd_gpus
    if not gpus:
        return _check(cid, name, CheckStatus.SKIPPED, "No AMD GPU detected, nothing to validate")
    lines: list[str] = []
    refs: list[str] = []
    worst = CheckStatus.PASS
    remediation = None
    for g in gpus:
        label = g.marketing_name or f"GPU {g.index}"
        gfx = (g.gfx_target or "").split(":")[0]
        if not gfx:
            status, why = CheckStatus.WARNING, "architecture (gfx target) could not be detected"
        elif entries := m.entries_for_gfx(gfx):
            refs += [r for e in entries for r in e.ref]
            statuses = {e.status for e in entries}
            names = ", ".join(e.name for e in entries)
            if statuses == {"unsupported"}:
                status, why = CheckStatus.FAIL, f"{gfx} is listed as unsupported ({names})"
                remediation = "This GPU architecture is not supported by current ROCm releases"
            elif "deprecated" in statuses and "supported" not in statuses:
                status, why = CheckStatus.WARNING, f"{gfx} is listed as deprecated ({names})"
            else:
                status, why = CheckStatus.PASS, f"{gfx} is listed as supported ({entries[0].arch}: {names})"
        elif gfx in m.targets_uncertain:
            refs += m.targets_uncertain_ref
            status, why = CheckStatus.WARNING, f"{gfx}: {m.targets_uncertain[gfx]}"
        else:
            refs += [r for e in m.gpus for r in e.ref]
            status = CheckStatus.WARNING
            why = f"{gfx} is not listed; AMD states unlisted GPUs are not officially supported"
        lines.append(f"{label}: {why}")
        if _RANK[status] > _RANK[worst]:
            worst = status
    return _check(
        cid,
        name,
        worst,
        lines[0] if len(lines) == 1 else f"{len(gpus)} GPUs checked",
        detail="\n".join(lines),
        remediation=remediation,
        source=m.cite(refs, "sysreq+matrix"),
    )


# --------------------------------------------------------------------------- OS


def _os_key(host: HostInfo) -> tuple[str | None, str | None]:
    os_id = (host.os_id or "").lower() or None
    ver = host.os_version
    if os_id in ("rocky", "ol", "debian") and ver:
        ver = ver.split(".")[0]
    elif os_id in ("sles", "sled", "opensuse-leap") and ver:
        os_id = "sles"
    return os_id, ver


def match_os(
    host: HostInfo, m: CompatibilityMatrix, classes: set[GpuClass]
) -> list[tuple[GpuClass, OsEntry]]:
    os_id, ver = _os_key(host)
    if not os_id or not ver:
        return []
    out = []
    for cls in classes or {"instinct", "radeon"}:
        for e in m.os.get(cls, []):
            if e.id == os_id and (e.version == ver or e.version == major_minor(ver)):
                out.append((cls, e))
    return out


def validate_os(inp: CompatibilityInput, m: CompatibilityMatrix) -> CheckResult:
    cid, name = "compat.os", "Operating System Support"
    host = inp.host
    src = m.cite([r for entries in m.os.values() for e in entries for r in e.ref], "sysreq+matrix")
    if host.os_name == "macOS" or (host.os_id is None and host.os_name == "Darwin"):
        return _check(
            cid,
            name,
            CheckStatus.FAIL,
            "ROCm is not available on macOS",
            remediation="Run ROCm workloads on a supported Linux distribution",
            source=src,
        )
    if host.os_name == "Windows":
        return _check(
            cid,
            name,
            CheckStatus.WARNING,
            "Windows detected; this matrix snapshot covers Linux ROCm only",
            detail="The 10.1.0 matrix lists Windows 11 25H2 for Radeon and Ryzen APUs via the HIP SDK; "
            "KernelVet does not validate Windows configurations.",
            source=src,
        )
    classes = gpu_classes(inp, m)
    matches = match_os(host, m, classes)
    pretty = host.os_pretty or f"{host.os_id} {host.os_version}"
    if not matches:
        considered = [e for c in (classes or {"instinct", "radeon"}) for e in m.os.get(c, [])]
        listed = sorted({e.label for e in considered})
        return _check(
            cid,
            name,
            CheckStatus.WARNING,
            f"{pretty} is not in the supported OS list",
            detail="Listed: " + "; ".join(listed),
            remediation="Use a distribution listed in AMD's ROCm system requirements",
            source=m.cite([r for e in considered for r in e.ref], "sysreq+matrix"),
        )
    # GPU-specific OS restrictions
    os_id, ver = _os_key(host)
    for r in m.os_gpu_restrictions:
        r_id, _, r_ver = r.os.partition(":")
        if r_id == os_id and (r_ver == ver or r_ver == major_minor(ver)):
            bad = [g for g in inp.amd_gpus if (g.gfx_target or "").split(":")[0] not in r.gfx]
            if bad:
                return _check(
                    cid,
                    name,
                    CheckStatus.WARNING,
                    f"{pretty} is restricted to specific GPUs",
                    detail=r.note,
                    remediation="Use an OS listed for your GPU",
                    source=m.cite(r.ref, r.source),
                )
    labels = ", ".join(sorted({f"{e.label} ({c})" for c, e in matches}))
    note = "" if classes else " (no AMD GPU detected, so the GPU-specific OS list was not applied)"
    return _check(
        cid,
        name,
        CheckStatus.PASS,
        f"{pretty} is listed{note}",
        detail=labels,
        source=m.cite([r for _, e in matches for r in e.ref], "sysreq+matrix"),
    )


def validate_kernel(inp: CompatibilityInput, m: CompatibilityMatrix) -> CheckResult:
    cid, name = "compat.kernel", "Kernel Compatibility"
    host = inp.host
    if host.os_name in ("macOS", "Windows", "Darwin"):
        return _check(cid, name, CheckStatus.SKIPPED, "Not a Linux host")
    kernel_mm = major_minor(host.kernel)
    if not kernel_mm:
        return _check(cid, name, CheckStatus.SKIPPED, "Kernel version could not be detected")
    matches = match_os(host, m, gpu_classes(inp, m))
    if not matches:
        return _check(cid, name, CheckStatus.SKIPPED, "OS not in the matrix, so no kernel list applies")
    kernels = sorted({k for _, e in matches for k in e.kernels})
    kernel_src = m.cite([r for _, e in matches for r in e.ref], "sysreq+matrix")
    if not kernels:
        return _check(
            cid,
            name,
            CheckStatus.WARNING,
            f"Kernel {host.kernel}: the matrix lists no explicit "
            "kernel version for this OS (e.g. Oracle UEK)",
            source=kernel_src,
        )
    if kernel_mm in kernels:
        return _check(
            cid,
            name,
            CheckStatus.PASS,
            f"Kernel {host.kernel} matches listed {kernel_mm}",
            source=kernel_src,
        )
    return _check(
        cid,
        name,
        CheckStatus.WARNING,
        f"Kernel {host.kernel} is not one of the listed kernels",
        detail="Listed: " + ", ".join(kernels),
        remediation="Use the GA or HWE kernel listed for your distribution",
        source=kernel_src,
    )


# --------------------------------------------------------------------------- ROCm / driver / HIP


def validate_rocm(inp: CompatibilityInput, m: CompatibilityMatrix) -> CheckResult:
    cid, name = "compat.rocm", "ROCm Version"
    ver = inp.software.rocm_version
    if not ver:
        return _check(
            cid,
            name,
            CheckStatus.SKIPPED,
            "ROCm installation not detected",
            remediation="Install ROCm: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/",
        )
    if same_release(ver, m.rocm_version, depth=3) or same_release(ver, m.rocm_version, depth=2):
        return _check(
            cid,
            name,
            CheckStatus.PASS,
            f"ROCm {ver} matches the matrix snapshot ({m.rocm_version})",
            source=m.source_url("matrix"),
        )
    pv, pm = parse_version(ver), parse_version(m.rocm_version)
    newer = pv is not None and pm is not None and pv > pm
    listed = any(same_release(ver, r, depth=2) for r in m.release_versions)
    if newer:
        why = f"ROCm {ver} is newer than this matrix snapshot ({m.rocm_version}); update KernelVet's matrix"
    elif listed:
        why = (
            f"ROCm {ver} is a known release, but this snapshot's tables describe {m.rocm_version}; "
            "GPU/OS results may not apply"
        )
    else:
        why = f"ROCm {ver} is not covered by this matrix snapshot ({m.rocm_version})"
    return _check(
        cid,
        name,
        CheckStatus.WARNING,
        why,
        remediation=f"Check the compatibility matrix for ROCm {major_minor(ver)} in AMD's docs",
        source=m.cite(m.release_versions_ref, "releases"),
    )


def validate_driver(inp: CompatibilityInput, m: CompatibilityMatrix) -> CheckResult:
    cid, name = "compat.driver", "AMDGPU Driver"
    sw = inp.software
    if inp.host.os_name in ("macOS", "Windows", "Darwin"):
        return _check(cid, name, CheckStatus.SKIPPED, "Not a Linux host")
    if not inp.amd_gpus:
        return _check(cid, name, CheckStatus.SKIPPED, "No AMD GPU detected")
    if sw.amdgpu_module_loaded is False:
        return _check(
            cid,
            name,
            CheckStatus.FAIL,
            "amdgpu kernel module is not loaded",
            remediation="Install the amdgpu driver (amdgpu-dkms) and reboot, or `sudo modprobe amdgpu`",
        )
    ver = sw.amdgpu_driver_version
    classes = gpu_classes(inp, m)
    if not ver:
        return _check(
            cid,
            name,
            CheckStatus.WARNING,
            "amdgpu is loaded but reports no version (likely the distribution's inbox driver)",
            detail="The matrix lists specific amdgpu versions for Instinct and Radeon; "
            "Ryzen APUs use the inbox driver.",
            source=m.cite(m.amdgpu_driver.ref, m.amdgpu_driver.source),
        )
    listed: set[str] = set()
    for c in classes or {"instinct", "radeon"}:
        listed.update(getattr(m.amdgpu_driver, c))
    if any(same_release(ver, v, depth=3) for v in listed):
        return _check(
            cid,
            name,
            CheckStatus.PASS,
            f"amdgpu {ver} is listed",
            source=m.cite(m.amdgpu_driver.ref, m.amdgpu_driver.source),
        )
    return _check(
        cid,
        name,
        CheckStatus.WARNING,
        f"amdgpu {ver} is not in the listed driver versions",
        detail="Listed: "
        + ", ".join(sorted(listed))
        + "\nThe kernel module may report a different version scheme than the matrix.",
        source=m.cite(m.amdgpu_driver.ref, m.amdgpu_driver.source),
    )


def validate_hip(inp: CompatibilityInput, m: CompatibilityMatrix) -> CheckResult:
    cid, name = "compat.hip", "HIP Compatibility"
    sw = inp.software
    if not sw.has("hipcc") and not sw.hip_runtime_version:
        return _check(cid, name, CheckStatus.SKIPPED, "HIP toolchain not detected")
    if not sw.hip_runtime_version:
        return _check(cid, name, CheckStatus.WARNING, "hipcc found but the HIP version could not be read")
    if sw.rocm_version and not same_release(sw.hip_runtime_version, sw.rocm_version):
        return _check(
            cid,
            name,
            CheckStatus.WARNING,
            f"HIP {sw.hip_runtime_version} does not match ROCm {sw.rocm_version}",
            remediation="Mixed ROCm installs detected; check PATH and ROCM_PATH point at one release",
        )
    return _check(
        cid, name, CheckStatus.PASS, f"HIP {sw.hip_runtime_version} is consistent with the ROCm install"
    )


# --------------------------------------------------------------------------- frameworks


def validate_pytorch(inp: CompatibilityInput, m: CompatibilityMatrix) -> CheckResult:
    cid, name = "compat.pytorch", "PyTorch Compatibility"
    t = inp.software.torch
    fw = m.frameworks.get("pytorch")
    if t is None:
        return _check(cid, name, CheckStatus.SKIPPED, "PyTorch check disabled")
    if not t.installed:
        return _check(cid, name, CheckStatus.SKIPPED, "PyTorch not installed in the checked interpreter")
    if not t.is_rocm_build:
        build = f"CUDA {t.cuda_version}" if t.cuda_version else "CPU-only"
        return _check(
            cid,
            name,
            CheckStatus.WARNING,
            f"PyTorch {t.version} is a {build} build, not ROCm",
            remediation="Install a ROCm build of PyTorch (https://pytorch.org/get-started/locally/)",
        )
    if fw is None:
        return _check(cid, name, CheckStatus.SKIPPED, "Matrix has no PyTorch data")
    tmm = major_minor(t.version)
    src = m.cite(fw.ref, fw.source)
    if tmm not in fw.versions:
        return _check(
            cid,
            name,
            CheckStatus.WARNING,
            f"PyTorch {t.version} is not listed for ROCm {m.rocm_version}",
            detail="Listed: " + ", ".join(fw.versions),
            source=src,
        )
    sys_rocm = inp.software.rocm_version
    if sys_rocm and not same_release(t.hip_version, sys_rocm):
        return _check(
            cid,
            name,
            CheckStatus.WARNING,
            f"PyTorch {t.version} was built for HIP {t.hip_version}; system ROCm is {sys_rocm}",
            detail="Wheels bundle their own ROCm runtime libraries, but the kernel driver must "
            "still support them.",
            source=src,
        )
    if sys_rocm and not same_release(sys_rocm, m.rocm_version):
        return _check(
            cid,
            name,
            CheckStatus.WARNING,
            f"PyTorch {tmm} is listed for ROCm {m.rocm_version}, but this host runs ROCm {sys_rocm}",
            source=src,
        )
    return _check(
        cid, name, CheckStatus.PASS, f"PyTorch {t.version} (HIP {t.hip_version}) is listed", source=src
    )


VALIDATORS = [
    validate_gpu,
    validate_os,
    validate_kernel,
    validate_rocm,
    validate_driver,
    validate_hip,
    validate_pytorch,
]
