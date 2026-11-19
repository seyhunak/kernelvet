"""Migration planning: an actionable plan produced before anything is modified."""

from __future__ import annotations

from pathlib import Path

from kernelvet.migration.hipify import TRANSLATE_KINDS, choose_translator
from kernelvet.models.migration import (
    AnalysisResult,
    FindingCategory,
    MigrationPlan,
    PlanStep,
    Severity,
)
from kernelvet.utils.proc import HostContext


def build_plan(
    analysis: AnalysisResult,
    ctx: HostContext | None = None,
    *,
    preference: str = "auto",
    offload_arch: list[str] | None = None,
) -> MigrationPlan:
    ctx = ctx or HostContext()
    inv = analysis.inventory
    root = Path(inv.root)
    choice = choose_translator(ctx, preference, has_compile_db=(root / "compile_commands.json").exists())
    to_migrate = [f.path for f in inv.files if f.kind in TRANSLATE_KINDS]
    to_copy = [f.path for f in inv.files if f.kind not in TRANSLATE_KINDS]

    conversions: dict[str, str] = {}
    for f in analysis.findings:
        if f.category is FindingCategory.AUTO_CONVERTIBLE and f.symbol and f.hip_equivalent:
            conversions.setdefault(f.symbol, f.hip_equivalent)
    manual = [f for f in analysis.findings if f.category is FindingCategory.MANUAL_INTERVENTION]
    blockers = [
        f
        for f in analysis.findings
        if f.category is FindingCategory.UNSUPPORTED
        or (f.category is FindingCategory.MANUAL_INTERVENTION and f.severity is Severity.HIGH)
    ]
    verify = [f for f in analysis.findings if f.category is FindingCategory.REQUIRES_VERIFICATION]
    deps = [d for d in inv.dependencies if d.kind == "nvidia-library"]

    arch = offload_arch or []
    arch_txt = ";".join(arch) if arch else "<your gfx targets, e.g. gfx942;gfx1100>"
    if any(b.kind == "cmake" for b in inv.build_systems):
        build_rec = (
            f'CMake: enable_language(HIP), set CMAKE_HIP_ARCHITECTURES="{arch_txt}", '
            "find_package(hip REQUIRED) plus per-library packages (hipblas, hipfft, rccl, ...), "
            "set LANGUAGE HIP on .cu sources, and link hip::device. Build with "
            "`cmake -S . -B build -DCMAKE_HIP_COMPILER=$ROCM_PATH/llvm/bin/clang++`."
        )
    elif any(b.kind == "make" for b in inv.build_systems):
        build_rec = (
            f"Make: replace nvcc with hipcc, replace -arch=sm_XX with --offload-arch={arch_txt.split(';')[0]}, "
            "and replace -lcudart/-lcublas/... with -lamdhip64/-lhipblas/...."
        )
    else:
        build_rec = f"No build system detected: compile each translation unit with `hipcc --offload-arch={arch_txt.split(';')[0]}`."

    steps: list[PlanStep] = []

    def add(title: str, detail: str, files: list[str] | None = None, blocking: bool = False) -> None:
        steps.append(
            PlanStep(order=len(steps) + 1, title=title, detail=detail, files=files or [], blocking=blocking)
        )

    if choice.name is None:
        add("Install HIPIFY", choice.reason, blocking=True)
    add(
        "Convert sources with " + (choice.name or "HIPIFY"),
        f"{len(to_migrate)} CUDA-related file(s) are translated into a separate output tree; "
        f"{len(to_copy)} other file(s) are copied unchanged. The original project is not modified.",
        to_migrate,
    )
    for d in deps:
        add(
            f"Replace {d.name}",
            f"Use {d.hip_alternative or 'no direct equivalent'}" + (f" ({d.note})" if d.note else ""),
            [e.split(":")[0] for e in d.evidence],
        )
    if blockers:
        add(
            "Resolve blockers",
            f"{len(blockers)} construct(s) have no mechanical HIP translation "
            "(inline PTX, CUTLASS/TensorRT, legacy textures, cuDNN ...). Port these by hand.",
            sorted({f.file for f in blockers}),
            blocking=True,
        )
    if manual:
        add(
            "Apply manual changes",
            f"{len(manual)} finding(s) need manual edits (build files, NVTX/NVML, cuDNN ...).",
            sorted({f.file for f in manual}),
        )
    if verify:
        add(
            "Review wavefront/architecture assumptions",
            f"{len(verify)} finding(s) depend on warp size 32, 32-bit lane masks, compute capability or "
            "managed memory. HIPIFY translates the calls but not the assumptions.",
            sorted({f.file for f in verify}),
        )
    add("Update the build", build_rec)
    add(
        "Audit converted sources",
        "Run `kernelvet migrate audit <output>` to find leftover CUDA and risky patterns.",
    )
    add("Compile with hipcc", "Run `kernelvet migrate build <output>`.")
    add(
        "Run tests on an AMD GPU",
        "Run `kernelvet migrate test <output>`; only this stage validates runtime behaviour.",
    )

    return MigrationPlan(
        project=inv.root,
        translator=choice.name,
        translator_reason=choice.reason,
        files_to_migrate=to_migrate,
        files_to_copy=to_copy,
        api_conversions=dict(sorted(conversions.items())),
        dependency_replacements=deps,
        manual_changes=manual,
        blockers=blockers,
        build_recommendation=build_rec,
        steps=steps,
    )
