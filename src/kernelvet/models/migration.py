"""CUDA-to-HIP migration models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from kernelvet.models.common import Evidence


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


class CompileDbSummary(BaseModel):
    """The project's compile_commands.json, if one was found or given."""

    path: str
    entries: int = 0
    # Translation units (CUDA sources and C++ files using CUDA) with / without an entry.
    covered: list[str] = Field(default_factory=list)
    uncovered: list[str] = Field(default_factory=list)
    error: str | None = None  # set when the file could not be used


class ProjectInventory(BaseModel):
    root: str
    files: list[SourceFile] = Field(default_factory=list)
    build_systems: list[BuildSystem] = Field(default_factory=list)
    dependencies: list[Dependency] = Field(default_factory=list)
    test_files: list[str] = Field(default_factory=list)
    total_lines: int = 0
    skipped: list[str] = Field(default_factory=list, description="Files skipped and why")
    compile_db: CompileDbSummary | None = None

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


class ReadinessLevel(StrEnum):
    """What kind of work the source code needs, from least to most."""

    NO_CUDA = "no-cuda"  # nothing to convert
    MECHANICAL = "mechanical"  # everything found has a mechanical HIPIFY translation
    REVIEW = "review"  # converts, but some constructs behave differently on AMD GPUs
    MANUAL_WORK = "manual-work"  # someone has to rewrite some code
    BLOCKED = "blocked"  # some code has no HIP equivalent and needs a redesign


class ScoreComponent(BaseModel):
    """One finding category's share of the score."""

    category: FindingCategory
    weight: float  # how much one construct of this category counts as converted
    constructs: int  # distinct (rule, file) pairs: what the score counts
    occurrences: int  # every match, for effort estimates
    files: int  # files with at least one


READINESS_LIMITATIONS = [
    "Static pattern matching: only constructs with a rule in rules.py are counted. CUDA usage "
    "without a rule is invisible, which makes coverage look better than it is.",
    "It measures what HIPIFY can translate, not whether the result compiles, is correct or is fast.",
    "The weights (1 converted, 0.5 needs review, 0 manual or unsupported) are a fixed convention, "
    "not calibrated against real migrations.",
    "Each construct counts once per file, however often it appears and however much code it "
    "involves: one inline PTX block weighs the same as one cudaMalloc.",
    "Code behind macros, disabled #ifdef branches or generated at build time is matched as text, "
    "not evaluated.",
    "Build-system changes (CUDA language, nvcc, sm_XX, NVIDIA libraries in CMake/Make) are listed "
    "separately and not scored; they are always needed.",
]


class ReadinessScore(BaseModel):
    """How much of the detected CUDA usage HIPIFY can translate mechanically.

    This is *source conversion coverage*. It says nothing about whether the
    converted program is correct or runs on AMD hardware. ``limitations`` lists
    what the number cannot tell you.
    """

    score: int | None  # 0..100 after caps; None when there is no CUDA code
    level: ReadinessLevel = ReadinessLevel.NO_CUDA
    coverage: int | None = None  # 0..100 before caps, over distinct source constructs
    occurrence_coverage: int | None = None  # same weights, counting every occurrence
    cap: int | None = None  # upper limit applied to ``score``, if any
    cap_reason: str | None = None
    components: list[ScoreComponent] = Field(default_factory=list)
    build_items: int = 0  # distinct build-system findings (not scored)
    total_findings: int
    by_category: dict[str, int]  # every finding, source and build, by category
    explanation: str = ""
    limitations: list[str] = Field(default_factory=lambda: list(READINESS_LIMITATIONS))


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
    # hipify-clang only: "compile_commands.json" when the file's own flags were used, else "guessed".
    flags_from: str | None = None


class ConversionResult(BaseModel):
    project: str
    output: str
    translator: str
    translator_version: str | None = None
    dry_run: bool = False
    files: list[FileConversion] = Field(default_factory=list)
    log_path: str | None = None
    compile_db: str | None = None  # compile_commands.json used for per-file flags

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
    # How we know whether the test ran code on the GPU:
    #   executed = the HIP runtime log shows at least one kernel launch that returned hipSuccess
    #   detected = an AMD GPU is present, but no successful kernel launch was observed
    #   none     = no AMD GPU on the host
    gpu_evidence: Evidence = Evidence.NONE
    gpu_kernel_launches: int = 0
    # True only for a PASSED test with gpu_evidence == executed; this is what GPU validation counts.
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
