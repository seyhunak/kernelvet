"""Formatting helpers shared by every exporter and the TUI."""

from __future__ import annotations

from kernelvet.diagnose.software import torch_summary
from kernelvet.models.common import Evidence
from kernelvet.models.migration import (
    CompileDbSummary,
    ReadinessScore,
    StageStatus,
    TestCaseResult,
    VerificationStages,
)
from kernelvet.models.system import GpuDevice, SoftwareInfo


def fmt_bytes(n: int | None) -> str:
    if n is None:
        return "unknown"
    for unit, div in (("TiB", 1024**4), ("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{n} B"


def gpu_label(g: GpuDevice) -> str:
    return g.marketing_name or g.vendor or f"GPU {g.index}"


def software_rows(sw: SoftwareInfo) -> list[tuple[str, str]]:
    def v(x: str | None, missing: str = "not detected") -> str:
        return x or missing

    return [
        ("ROCm", f"{v(sw.rocm_version)}{f' ({sw.rocm_path})' if sw.rocm_path else ''}"),
        ("HIP runtime", v(sw.hip_runtime_version)),
        ("HIP compiler", v(sw.hip_compiler_version)),
        (
            "amdgpu driver",
            "not loaded"
            if sw.amdgpu_module_loaded is False
            else v(
                sw.amdgpu_driver_version,
                "loaded (version not reported)" if sw.amdgpu_module_loaded else "n/a",
            ),
        ),
        ("AMD SMI", "available" if sw.has("amd-smi") else "not installed"),
        ("HIPIFY", ", ".join(t for t in ("hipify-clang", "hipify-perl") if sw.has(t)) or "not installed"),
        ("Profiler", ", ".join(t for t in ("rocprofv3", "rocprof") if sw.has(t)) or "not installed"),
        ("ROCm libraries", ", ".join(sw.rocm_libraries) or "none found"),
        ("PyTorch", torch_summary(sw.torch)),
    ]


STAGE_LABELS = {
    "source_conversion": "1. Source conversion",
    "static_audit": "2. Static audit",
    "compilation": "3. Compilation",
    "tests": "4. Tests",
    "gpu_runtime_validation": "5. AMD GPU runtime validation",
}


def stage_rows(stages: VerificationStages) -> list[tuple[str, str]]:
    return [(STAGE_LABELS[k], str(getattr(stages, k))) for k in STAGE_LABELS]


def verification_note(stages: VerificationStages) -> str:
    if stages.gpu_runtime_validation == StageStatus.PASSED:
        return "Migrated code compiled and its tests passed on an AMD GPU."
    reached: str | None = None
    for k, label in STAGE_LABELS.items():
        if getattr(stages, k) != StageStatus.PASSED:
            break
        reached = label
    if reached is None:
        return "No verification stage has passed yet."
    return (
        f"Verified up to: {reached}. Later stages are not verified; "
        "do not treat earlier stages as proof of runtime correctness."
    )


def gpu_cell(case: TestCaseResult) -> str:
    """Per-test GPU evidence, worded so detection is never mistaken for execution."""
    if case.gpu_evidence is Evidence.EXECUTED:
        n = case.gpu_kernel_launches
        return f"ran on GPU ({n} kernel launch{'es' if n != 1 else ''})"
    if case.gpu_evidence is Evidence.DETECTED:
        return "GPU detected, no kernel launch seen"
    return "no AMD GPU"


def compile_db_line(db: CompileDbSummary | None) -> str:
    if db is None:
        return "none (flags are guessed from the directory layout)"
    if db.error:
        return f"{db.path}: unusable ({db.error})"
    total = len(db.covered) + len(db.uncovered)
    return f"{db.path} ({db.entries} entries; covers {len(db.covered)} of {total} CUDA translation unit(s))"


def readiness_line(r: ReadinessScore) -> str:
    """One-line summary: score, level, and what changed it."""
    if r.score is None:
        return "n/a (no CUDA constructs found)"
    text = f"{r.score}/100 · {r.level.value}"
    if r.cap is not None:
        text += f" (coverage {r.coverage}%, capped: {r.cap_reason})"
    return text


def readiness_rows(r: ReadinessScore) -> list[tuple[str, str, str, str, str]]:
    """Breakdown rows: category, weight, constructs, occurrences, files."""
    return [
        (c.category.value, f"{c.weight:g}", str(c.constructs), str(c.occurrences), str(c.files))
        for c in r.components
    ]
