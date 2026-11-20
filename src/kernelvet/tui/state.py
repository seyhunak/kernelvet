"""Session state shared by every TUI pane."""

from __future__ import annotations

from dataclasses import dataclass, field

from kernelvet.config import Settings
from kernelvet.history import HistoryStore
from kernelvet.models.migration import (
    AnalysisResult,
    AuditResult,
    BuildResult,
    ConversionResult,
    MigrationPlan,
    TestRunResult,
)
from kernelvet.models.report import BenchmarkReport, CompatibilityReport, DiagnosticReport
from kernelvet.models.system import GpuDevice, HostInfo, SoftwareInfo
from kernelvet.utils.proc import HostContext


@dataclass
class MigrationSession:
    project: str = ""
    output: str = ""
    analysis: AnalysisResult | None = None
    plan: MigrationPlan | None = None
    conversion: ConversionResult | None = None
    audit: AuditResult | None = None
    build: BuildResult | None = None
    tests: TestRunResult | None = None


@dataclass
class SessionState:
    ctx: HostContext
    settings: Settings
    history: HistoryStore
    host: HostInfo | None = None
    gpus: list[GpuDevice] = field(default_factory=list)
    software: SoftwareInfo | None = None
    diagnosis: DiagnosticReport | None = None
    compatibility: CompatibilityReport | None = None
    benchmarks: BenchmarkReport | None = None
    migration: MigrationSession = field(default_factory=MigrationSession)

    @property
    def amd_gpus(self) -> list[GpuDevice]:
        return [g for g in self.gpus if g.vendor_is_amd]

    @property
    def gfx_targets(self) -> list[str]:
        if self.settings.offload_arch:
            return [a.strip() for a in self.settings.offload_arch.split(",") if a.strip()]
        return sorted({g.gfx_target.split(":")[0] for g in self.amd_gpus if g.gfx_target})
