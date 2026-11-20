"""Diagnostic orchestration shared by the CLI and the TUI."""

from __future__ import annotations

from dataclasses import dataclass, field

from kernelvet.compatibility.engine import evaluate
from kernelvet.compatibility.matrix import load_matrix
from kernelvet.diagnose.hardware import HardwareProbe, discover_gpus
from kernelvet.diagnose.runtime import run_hip_probe, run_torch_probe
from kernelvet.diagnose.software import collect_software
from kernelvet.diagnose.system import collect_host
from kernelvet.models.common import CheckResult, CheckStatus, Evidence
from kernelvet.models.report import DiagnosticReport
from kernelvet.models.system import GpuDevice, HostInfo, SoftwareInfo
from kernelvet.utils.proc import HostContext, ProgressCallback, noop_progress


@dataclass
class DiagnoseOptions:
    runtime: bool = True  # compile + run the HIP probe
    torch: bool = True  # detect PyTorch and run tensor ops on the GPU
    compatibility: bool = True
    python: str | None = None
    matrix_path: str | None = None
    extra_notes: list[str] = field(default_factory=list)


def _c(
    cid: str,
    name: str,
    category: str,
    status: CheckStatus,
    summary: str,
    evidence: Evidence = Evidence.DETECTED,
    remediation: str | None = None,
    detail: str = "",
) -> CheckResult:
    return CheckResult(
        id=cid,
        name=name,
        category=category,
        status=status,
        evidence=evidence,
        summary=summary,
        remediation=remediation,
        detail=detail,
    )


def _rocm_where(sw: SoftwareInfo) -> str:
    if sw.rocm_source == "rocm-sdk":
        where = f"at {sw.rocm_path} " if sw.rocm_path else ""
        return f"{where}(TheRock Python packages, found with rocm-sdk)".strip()
    return f"at {sw.rocm_path}" if sw.rocm_path else ""


def static_checks(host: HostInfo, hw: HardwareProbe, sw: SoftwareInfo) -> list[CheckResult]:
    checks: list[CheckResult] = []
    linux = host.os_name not in ("macOS", "Windows", "Darwin")
    checks.append(
        _c(
            "system.platform",
            "Supported Platform",
            "system",
            CheckStatus.PASS if linux else CheckStatus.FAIL,
            f"{host.os_pretty or host.os_name} ({host.architecture})",
            remediation=None if linux else "ROCm runs on Linux; use a supported distribution",
        )
    )

    amd = [g for g in hw.gpus if g.vendor_is_amd]
    if amd:
        sources = sorted({s for g in amd for s in g.sources})
        checks.append(
            _c(
                "hardware.gpu_detected",
                "GPU Detection",
                "hardware",
                CheckStatus.PASS,
                f"{len(amd)} AMD GPU(s) detected via {', '.join(sources)}",
            )
        )
    else:
        others = len(hw.gpus) - len(amd)
        why = "No AMD GPU detected"
        if not linux:
            why += f" ({host.os_name} host; ROCm GPU discovery needs Linux)"
        elif hw.rocminfo is None and hw.amd_smi is None:
            why += " (rocminfo and amd-smi not installed, and no AMD device in /sys/class/drm)"
        if others:
            why += f"; {others} non-AMD GPU(s) listed"
        checks.append(
            _c(
                "hardware.gpu_detected",
                "GPU Detection",
                "hardware",
                CheckStatus.FAIL,
                why,
                remediation="Check the GPU is installed, the amdgpu driver is loaded, and ROCm is installed",
                detail="\n".join(hw.notes),
            )
        )

    if hw.kfd_accessible is None:
        checks.append(
            _c(
                "hardware.device_access",
                "GPU Device Access",
                "hardware",
                CheckStatus.SKIPPED,
                "Not a Linux host",
            )
        )
    elif hw.kfd_accessible:
        checks.append(
            _c(
                "hardware.device_access",
                "GPU Device Access",
                "hardware",
                CheckStatus.PASS,
                "/dev/kfd is readable and writable by this user",
            )
        )
    else:
        checks.append(
            _c(
                "hardware.device_access",
                "GPU Device Access",
                "hardware",
                CheckStatus.FAIL,
                "/dev/kfd is missing or not accessible",
                remediation="Load the amdgpu driver and add your user to the `render` and `video` groups: "
                "`sudo usermod -aG render,video $USER` then log in again",
            )
        )

    if sw.rocm_version:
        checks.append(
            _c(
                "software.rocm",
                "ROCm Installation",
                "software",
                CheckStatus.PASS,
                f"ROCm {sw.rocm_version} {_rocm_where(sw)}",
            )
        )
    elif sw.rocm_path or sw.rocm_source:
        checks.append(
            _c(
                "software.rocm",
                "ROCm Installation",
                "software",
                CheckStatus.WARNING,
                f"ROCm found {_rocm_where(sw)}, but its version could not be read",
            )
        )
    else:
        checks.append(
            _c(
                "software.rocm",
                "ROCm Installation",
                "software",
                CheckStatus.FAIL,
                "ROCm not found (checked ROCM_PATH, HIP_PATH, rocm-sdk on PATH and /opt/rocm)",
                remediation="Install ROCm: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/ . "
                "For ROCm installed with pip (TheRock Python packages), activate that virtual environment "
                "so `rocm-sdk` is on PATH",
            )
        )

    rocminfo = sw.tool("rocminfo")
    if rocminfo and rocminfo.available:
        if hw.rocminfo is not None and not hw.rocminfo.ok:
            checks.append(
                _c(
                    "software.rocminfo",
                    "rocminfo",
                    "software",
                    CheckStatus.FAIL,
                    hw.rocminfo.describe_failure(),
                    remediation="rocminfo failed; this usually means /dev/kfd is not accessible "
                    "or the amdgpu driver is not loaded",
                )
            )
        else:
            checks.append(
                _c(
                    "software.rocminfo",
                    "rocminfo",
                    "software",
                    CheckStatus.PASS,
                    f"available at {rocminfo.path}",
                )
            )
    else:
        checks.append(
            _c(
                "software.rocminfo",
                "rocminfo",
                "software",
                CheckStatus.WARNING,
                "not installed",
                remediation="rocminfo ships with ROCm (package `rocminfo`)",
            )
        )

    for tool, label, fix in (
        ("amd-smi", "AMD SMI", "Install AMD SMI (package `amd-smi-lib`)"),
        ("hipcc", "HIP Compiler", "Install the HIP SDK (package `hip-dev` / `rocm-hip-sdk`)"),
    ):
        t = sw.tool(tool)
        if t and t.available:
            checks.append(
                _c(
                    f"software.{tool}",
                    label,
                    "software",
                    CheckStatus.PASS,
                    f"{t.version or 'version unknown'} at {t.path}",
                )
            )
        else:
            checks.append(
                _c(
                    f"software.{tool}",
                    label,
                    "software",
                    CheckStatus.WARNING,
                    "not installed",
                    remediation=fix,
                )
            )

    if linux:
        if sw.amdgpu_module_loaded:
            checks.append(
                _c(
                    "software.amdgpu",
                    "amdgpu Kernel Module",
                    "software",
                    CheckStatus.PASS,
                    f"loaded{f' (version {sw.amdgpu_driver_version})' if sw.amdgpu_driver_version else ''}",
                )
            )
        else:
            checks.append(
                _c(
                    "software.amdgpu",
                    "amdgpu Kernel Module",
                    "software",
                    CheckStatus.FAIL,
                    "not loaded",
                    remediation="Install amdgpu-dkms or load the module: `sudo modprobe amdgpu`",
                )
            )
    return checks


def run_diagnosis(
    ctx: HostContext | None = None,
    options: DiagnoseOptions | None = None,
    progress: ProgressCallback = noop_progress,
) -> DiagnosticReport:
    ctx = ctx or HostContext()
    opts = options or DiagnoseOptions()

    # load (and validate) the matrix first, so a bad matrix file fails before the slow probes run
    matrix = load_matrix(opts.matrix_path) if opts.compatibility else None

    progress("Collecting host information", 0.05)
    host = collect_host(ctx)
    progress("Discovering GPUs (amd-smi, rocminfo, sysfs)", 0.15)
    hw = discover_gpus(ctx)
    progress("Inspecting ROCm, HIP, drivers and tools", 0.35)
    sw = collect_software(ctx, python=opts.python, torch=opts.torch)
    checks = static_checks(host, hw, sw)

    if opts.runtime:
        progress("Compiling and running the HIP probe", 0.55)
        targets = [g.gfx_target for g in hw.gpus if g.gfx_target]
        checks.extend(run_hip_probe(ctx, gfx_targets=[t for t in targets if t]))
    if opts.torch:
        progress("Running PyTorch GPU checks", 0.75)
        checks.extend(run_torch_probe(ctx, sw.torch, opts.python))

    compat = None
    if matrix is not None:
        progress("Evaluating compatibility matrix", 0.9)
        compat = evaluate(host, hw.gpus, sw, matrix)

    report = DiagnosticReport(host=host, gpus=hw.gpus, software=sw, checks=checks, compatibility=compat)
    progress("Done", 1.0)
    return report.finalize()


def collect_inventory(
    ctx: HostContext | None = None, *, torch: bool = False, python: str | None = None
) -> tuple[HostInfo, list[GpuDevice], SoftwareInfo]:
    """Static inventory only (used by `gpu list`, `system info`, `compatibility`)."""
    ctx = ctx or HostContext()
    host = collect_host(ctx)
    hw = discover_gpus(ctx)
    sw = collect_software(ctx, python=python, torch=torch)
    return host, hw.gpus, sw
