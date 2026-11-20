"""CUDA-to-HIP migration models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class FindingCategory(StrEnum):
    AUTO_CONVERTIBLE = "AUTO_CONVERTIBLE"
    MANUAL_INTERVENTION = "MANUAL_INTERVENTION"
    UNSUPPORTED = "UNSUPPORTED"
    REQUIRES_VERIFICATION = "REQUIRES_VERIFICATION"


class Severity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def sarif_level(self) -> str:
        return {"info": "note", "low": "note", "medium": "warning", "high": "error"}[self.value]


class FileKind(StrEnum):
    CUDA_SOURCE = "cuda_source"  # .cu
    CUDA_HEADER = "cuda_header"  # .cuh
    CUDA_RELATED_CPP = "cuda_cpp"  # .cpp/.cc/.cxx that uses CUDA APIs
    CUDA_RELATED_HEADER = "cuda_h"  # .h/.hpp that uses CUDA APIs
    CPP = "cpp"  # C/C++ without CUDA usage
    HIP_SOURCE = "hip_source"  # .hip or HIP-using .cpp
    CMAKE = "cmake"
    MAKEFILE = "makefile"
    BUILD_SCRIPT = "build_script"
    TEST = "test"
    OTHER = "other"


class Finding(BaseModel):
    rule_id: str
    category: FindingCategory
    severity: Severity
    message: str
    file: str  # path relative to the project root
    line: int | None = None
    column: int | None = None
    snippet: str | None = None
    symbol: str | None = None
    hip_equivalent: str | None = None
    suggestion: str | None = None


class SourceFile(BaseModel):
    path: str  # relative to project root
    kind: FileKind
    lines: int
    cuda_api_calls: int = 0
    kernels: int = 0
    is_test: bool = False


class Dependency(BaseModel):
    name: str  # e.g. "cuBLAS"
    kind: str  # "nvidia-library" | "cuda-toolkit" | "external"
    evidence: list[str] = Field(default_factory=list)  # "file:line" references
    hip_alternative: str | None = None
    note: str | None = None


class BuildSystem(BaseModel):
    kind: str  # "cmake" | "make" | "script" | "none"
    files: list[str] = Field(default_factory=list)
    uses_cuda_language: bool = False
    cuda_arch_flags: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ProjectInventory(BaseModel):
    root: str
    files: list[SourceFile] = Field(default_factory=list)
    build_systems: list[BuildSystem] = Field(default_factory=list)
    dependencies: list[Dependency] = Field(default_factory=list)
    test_files: list[str] = Field(default_factory=list)
    total_lines: int = 0
    skipped: list[str] = Field(default_factory=list, description="Files skipped and why")

    def files_of(self, *kinds: FileKind) -> list[SourceFile]:
        return [f for f in self.files if f.kind in kinds]

    @property
    def cuda_files(self) -> list[SourceFile]:
        return self.files_of(
            FileKind.CUDA_SOURCE,
            FileKind.CUDA_HEADER,
            FileKind.CUDA_RELATED_CPP,
            FileKind.CUDA_RELATED_HEADER,
        )


class ReadinessScore(BaseModel):
    """How much of the detected CUDA usage HIPIFY can translate mechanically.

    This is *source conversion coverage*. It says nothing about whether the
    converted program is correct or runs on AMD hardware.
    """

    score: int  # 0..100
    total_findings: int
    by_category: dict[str, int]
    explanation: str = (
        "Share of detected CUDA constructs that have a mechanical HIP translation. "
        "It measures source conversion coverage only, not correctness or runtime compatibility."
    )


class AnalysisResult(BaseModel):
    inventory: ProjectInventory
    findings: list[Finding] = Field(default_factory=list)
    readiness: ReadinessScore


class PlanStep(BaseModel):
    order: int
    title: str
    detail: str
    files: list[str] = Field(default_factory=list)
    blocking: bool = False


class MigrationPlan(BaseModel):
    project: str
    translator: str | None  # "hipify-clang" | "hipify-perl" | None if none available
    translator_reason: str
    files_to_migrate: list[str]
    files_to_copy: list[str]
    api_conversions: dict[str, str]  # CUDA symbol -> HIP symbol
    dependency_replacements: list[Dependency]
    manual_changes: list[Finding]
    blockers: list[Finding]
    build_recommendation: str
    steps: list[PlanStep]


class FileConversion(BaseModel):
    source: str
    output: str
    translator: str
    status: str  # "converted" | "copied" | "failed"
    returncode: int | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    replacements: int | None = None
    unconverted: int | None = None


class ConversionResult(BaseModel):
    project: str
    output: str
    translator: str
    translator_version: str | None = None
    dry_run: bool = False
    files: list[FileConversion] = Field(default_factory=list)
    log_path: str | None = None

    @property
    def failed(self) -> list[FileConversion]:
        return [f for f in self.files if f.status == "failed"]


class AuditResult(BaseModel):
    project: str
    files_scanned: int
    findings: list[Finding] = Field(default_factory=list)
    note: str = (
        "Static audit only: it finds leftover CUDA constructs and risky patterns. "
        "It cannot establish that the migrated program is semantically correct."
    )


class StageStatus(StrEnum):
    NOT_RUN = "NOT_RUN"
    PASSED = "PASSED"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"  # tool or hardware missing


class BuildResult(BaseModel):
    project: str
    status: StageStatus
    method: str  # "cmake" | "hipcc" | "none"
    command: list[str] = Field(default_factory=list)
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    artifacts: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    duration_s: float | None = None
    isolated: bool = False
    reason: str | None = None


class TestCaseResult(BaseModel):
    __test__ = False  # not a pytest test class

    name: str
    command: list[str]
    status: StageStatus
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_s: float | None = None
    expected_output_match: bool | None = None  # compared against a reference file
    ran_on_gpu: bool = False


class TestRunResult(BaseModel):
    __test__ = False  # not a pytest test class

    project: str
    status: StageStatus
    cases: list[TestCaseResult] = Field(default_factory=list)
    gpu_present: bool = False
    reason: str | None = None


class VerificationStages(BaseModel):
    """The five verification stages, kept separate on purpose."""

    source_conversion: StageStatus = StageStatus.NOT_RUN
    static_audit: StageStatus = StageStatus.NOT_RUN
    compilation: StageStatus = StageStatus.NOT_RUN
    tests: StageStatus = StageStatus.NOT_RUN
    gpu_runtime_validation: StageStatus = StageStatus.NOT_RUN
