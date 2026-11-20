"""Migration workflow state: persisted per migrated tree under ``.kernelvet/``.

Each stage saves its result in the *output* tree, never in the CUDA source tree:

    <hip-project>/.kernelvet/source.json       written by convert
    <hip-project>/.kernelvet/conversion.json   written by convert
    <hip-project>/.kernelvet/audit.json        written by audit
    <hip-project>/.kernelvet/build.json        written by build
    <hip-project>/.kernelvet/tests.json        written by test

``migrate report`` assembles them, re-running the read-only analysis of the
original project when it is still available.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from kernelvet.migration.analyzer import analyze_project
from kernelvet.migration.auditor import migrated_source_of
from kernelvet.models.migration import (
    AnalysisResult,
    AuditResult,
    BuildResult,
    ConversionResult,
    FindingCategory,
    StageStatus,
    TestRunResult,
    VerificationStages,
)
from kernelvet.models.report import MigrationReport

STATE = ".kernelvet"
M = TypeVar("M", bound=BaseModel)


def save_state(project: str | Path, name: str, model: BaseModel) -> Path:
    d = Path(project) / STATE
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.json"
    p.write_text(model.model_dump_json(indent=2))
    return p


def load_state(project: str | Path, name: str, model: type[M]) -> M | None:
    p = Path(project) / STATE / f"{name}.json"
    if not p.is_file():
        return None
    try:
        return model.model_validate_json(p.read_text())
    except (OSError, ValidationError):
        return None


def compute_stages(
    conversion: ConversionResult | None,
    audit: AuditResult | None,
    build: BuildResult | None,
    tests: TestRunResult | None,
) -> VerificationStages:
    s = VerificationStages()
    if conversion and not conversion.dry_run:
        s.source_conversion = StageStatus.FAILED if conversion.failed else StageStatus.PASSED
    if audit:
        blocking = [
            f
            for f in audit.findings
            if f.category in (FindingCategory.UNSUPPORTED, FindingCategory.MANUAL_INTERVENTION)
        ]
        s.static_audit = StageStatus.FAILED if blocking else StageStatus.PASSED
    if build:
        s.compilation = build.status
    if tests:
        s.tests = tests.status
        if (
            tests.status is StageStatus.PASSED
            and tests.gpu_present
            and any(c.ran_on_gpu for c in tests.cases)
        ):
            s.gpu_runtime_validation = StageStatus.PASSED
        elif tests.status is StageStatus.FAILED and tests.gpu_present:
            s.gpu_runtime_validation = StageStatus.FAILED
        elif not tests.gpu_present and tests.status is not StageStatus.NOT_RUN:
            s.gpu_runtime_validation = StageStatus.UNAVAILABLE
    return s


def recommendations(r: MigrationReport) -> list[str]:
    out: list[str] = []
    st = r.stages
    if r.plan and r.plan.blockers:
        out.append(f"Resolve {len(r.plan.blockers)} blocker(s) that HIPIFY cannot translate (see plan).")
    if r.conversion and r.conversion.failed:
        out.append(f"{len(r.conversion.failed)} file(s) failed to convert; see {r.conversion.log_path}.")
    if st.static_audit is StageStatus.FAILED:
        out.append("Fix MANUAL_INTERVENTION/UNSUPPORTED audit findings before relying on the build.")
    if r.audit and any(f.category is FindingCategory.REQUIRES_VERIFICATION for f in r.audit.findings):
        out.append(
            "Review REQUIRES_VERIFICATION findings: warp size, lane masks and compute-capability logic."
        )
    if st.compilation is StageStatus.FAILED:
        out.append("Compilation failed; read the compiler errors in the build section.")
    if st.compilation is StageStatus.UNAVAILABLE:
        out.append('Install the ROCm HIP SDK (hipcc) to compile, or set isolation = "docker".')
    if st.gpu_runtime_validation is StageStatus.UNAVAILABLE:
        out.append("Run `kernelvet migrate test` on a host with an AMD GPU to validate runtime behaviour.")
    return out


def assemble_report(project: str | Path, *, analysis: AnalysisResult | None = None) -> MigrationReport:
    p = Path(project).resolve()
    conversion = load_state(p, "conversion", ConversionResult)
    audit = load_state(p, "audit", AuditResult)
    build = load_state(p, "build", BuildResult)
    tests = load_state(p, "tests", TestRunResult)
    if analysis is None:
        source = migrated_source_of(p)
        if source and Path(source).is_dir():
            analysis = analyze_project(source)
        elif conversion is None:
            analysis = analyze_project(p)  # a CUDA project that hasn't been converted yet
    report = MigrationReport(
        project=str(p),
        analysis=analysis,
        conversion=conversion,
        audit=audit,
        build=build,
        tests=tests,
        stages=compute_stages(conversion, audit, build, tests),
    )
    report.recommendations = recommendations(report)
    return report
