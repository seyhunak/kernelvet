"""Top-level report models shared by CLI, TUI and every exporter."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field, computed_field

from kernelvet import __version__
from kernelvet.models.common import CheckResult, OverallStatus, overall_status
from kernelvet.models.migration import (
    AnalysisResult,
    AuditResult,
    BuildResult,
    ConversionResult,
    MigrationPlan,
    TestRunResult,
    VerificationStages,
)
from kernelvet.models.system import GpuDevice, HostInfo, SoftwareInfo

DISCLAIMER = (
    "KernelVet is an independent open-source community project. It is not affiliated with, "
    "sponsored by or endorsed by Advanced Micro Devices, Inc. (AMD), is not an AMD product "
    "and is not a certification authority."
)
TRADEMARKS = (
    "AMD, AMD ROCm, AMD Instinct, AMD CDNA, AMD RDNA and Radeon are trademarks of Advanced "
    "Micro Devices, Inc. CUDA is a trademark of NVIDIA Corporation. Other names are used for "
    "identification purposes only and may be trademarks of their respective owners."
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ReportMeta(BaseModel):
    tool: str = "kernelvet"
    version: str = __version__
    generated_at: str = Field(default_factory=_now)
    disclaimer: str = DISCLAIMER


class CompatibilityReport(BaseModel):
    kind: str = "compatibility"
    meta: ReportMeta = Field(default_factory=ReportMeta)
    matrix_version: str
    matrix_sources: list[str]
    matrix_retrieved: str
    checks: list[CheckResult]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def status(self) -> OverallStatus:
        return overall_status(self.checks)


class DiagnosticReport(BaseModel):
    kind: str = "diagnosis"
    meta: ReportMeta = Field(default_factory=ReportMeta)
    host: HostInfo
    gpus: list[GpuDevice] = Field(default_factory=list)
    software: SoftwareInfo
    checks: list[CheckResult] = Field(default_factory=list)  # hardware/software/runtime
    compatibility: CompatibilityReport | None = None
    recommendations: list[str] = Field(default_factory=list)
    status: OverallStatus = OverallStatus.REVIEW_REQUIRED

    def all_checks(self) -> list[CheckResult]:
        compat = self.compatibility.checks if self.compatibility else []
        return [*self.checks, *compat]

    def finalize(self) -> DiagnosticReport:
        self.status = overall_status(self.all_checks())
        recs: list[str] = []
        for c in self.all_checks():
            if c.remediation and c.status in ("FAIL", "WARNING") and c.remediation not in recs:
                recs.append(c.remediation)
        self.recommendations = recs
        return self


class BenchmarkMeasurement(BaseModel):
    name: str
    unit: str
    value: float
    iterations: int
    detail: str | None = None


class BenchmarkResult(BaseModel):
    id: str
    name: str
    backend: str  # "hip" | "pytorch"
    status: str  # "PASS" | "FAIL" | "SKIPPED"
    reason: str | None = None
    measurements: list[BenchmarkMeasurement] = Field(default_factory=list)
    profiler: str | None = None


class BenchmarkReport(BaseModel):
    kind: str = "benchmark"
    meta: ReportMeta = Field(default_factory=ReportMeta)
    host: HostInfo
    gpus: list[GpuDevice] = Field(default_factory=list)
    software: SoftwareInfo
    results: list[BenchmarkResult] = Field(default_factory=list)
    note: str = (
        "All numbers are measured on this host. Stages that could not run are SKIPPED "
        "and carry no numbers. No cross-vendor comparison is made."
    )


class MigrationReport(BaseModel):
    kind: str = "migration"
    meta: ReportMeta = Field(default_factory=ReportMeta)
    project: str
    analysis: AnalysisResult | None = None
    plan: MigrationPlan | None = None
    conversion: ConversionResult | None = None
    audit: AuditResult | None = None
    build: BuildResult | None = None
    tests: TestRunResult | None = None
    stages: VerificationStages = Field(default_factory=VerificationStages)
    recommendations: list[str] = Field(default_factory=list)
