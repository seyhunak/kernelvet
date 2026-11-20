"""Shared enums and small value types used across every module."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, computed_field, model_validator


class CheckStatus(StrEnum):
    """Outcome of a single check.

    PASS     -- the condition was verified.
    WARNING  -- verification was incomplete or there is a potential issue.
    FAIL     -- a known failure or incompatibility.
    SKIPPED  -- the check could not be performed (missing tool, hardware, ...).
    """

    PASS = "PASS"  # noqa: S105 - a status name, not a credential
    WARNING = "WARNING"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"


class Evidence(StrEnum):
    """How a value or a check result was obtained.

    The distinction between static detection and actual execution is a core
    product requirement: a GPU that ``rocminfo`` lists is DETECTED, a kernel that
    ran and returned correct results is EXECUTED.
    """

    DETECTED = "detected"  # read from a tool's output, a file or a system interface
    EXECUTED = "executed"  # produced by actually running code on the device
    CONFIGURED = "configured"  # derived from configuration or reference data
    NONE = "none"  # nothing could be collected

    @property
    def label(self) -> str:
        """Plain-language wording shown wherever evidence is displayed (CLI, TUI, reports)."""
        return EVIDENCE_LABELS[self]


# The JSON value ("executed", "detected", ...) is a stable API; these labels are what people read.
EVIDENCE_LABELS: dict[Evidence, str] = {
    Evidence.EXECUTED: "ran on GPU",
    Evidence.DETECTED: "detected, not run",
    Evidence.CONFIGURED: "inferred from AMD docs",
    Evidence.NONE: "none",
}

EVIDENCE_LEGEND = (
    "Evidence: 'ran on GPU' (executed) = KernelVet compiled and ran code on the GPU during this run; "
    "only a PASS with this evidence is verified on AMD hardware. "
    "'detected, not run' (detected) = read from tools, files or /sys; nothing was executed. "
    "'inferred from AMD docs' (configured) = compared with AMD's published compatibility data or "
    "configuration; not tested on this hardware. "
    "'none' = not performed."
)


class OverallStatus(StrEnum):
    HEALTHY = "HEALTHY"
    REVIEW_REQUIRED = "REVIEW REQUIRED"
    UNHEALTHY = "UNHEALTHY"


class CheckResult(BaseModel):
    """One diagnostic or compatibility check."""

    id: str
    name: str
    category: str
    status: CheckStatus
    evidence: Evidence = Evidence.NONE
    summary: str = ""
    detail: str = ""
    remediation: str | None = None
    source: str | None = Field(default=None, description="Where the reference data for this check came from")
    duration_ms: float | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def hardware_verified(self) -> bool:
        """True only for a PASS produced by running code on the GPU in this run."""
        return self.status == CheckStatus.PASS and self.evidence == Evidence.EXECUTED

    @computed_field  # type: ignore[prop-decorator]
    @property
    def evidence_label(self) -> str:
        return self.evidence.label

    @model_validator(mode="after")
    def _skipped_has_no_evidence(self) -> Self:
        # A skipped check observed nothing. It must never look as if something
        # was detected, configured or executed.
        if self.status == CheckStatus.SKIPPED:
            self.evidence = Evidence.NONE
        return self


def overall_status(checks: list[CheckResult]) -> OverallStatus:
    """Collapse check results into one status.

    Any FAIL makes the environment UNHEALTHY. A WARNING, or a SKIPPED check,
    means something could not be verified, so it needs human review. Only an
    all-PASS run is HEALTHY: skipping is never treated as success.
    """
    statuses = {c.status for c in checks}
    if not checks:
        return OverallStatus.REVIEW_REQUIRED
    if CheckStatus.FAIL in statuses:
        return OverallStatus.UNHEALTHY
    if CheckStatus.WARNING in statuses or CheckStatus.SKIPPED in statuses:
        return OverallStatus.REVIEW_REQUIRED
    return OverallStatus.HEALTHY


class ExitCode:
    """Process exit codes, documented in README and ``kernelvet --help``."""

    OK = 0
    FAILURE = 1  # at least one FAIL / a stage failed
    USAGE = 2  # bad arguments (Typer/Click default)
    REVIEW = 3  # no failures, but warnings or skipped checks need review
    UNAVAILABLE = 4  # a required tool (hipify, hipcc, ...) is not installed
    INTERNAL = 70  # unexpected internal error (EX_SOFTWARE)

    @staticmethod
    def for_status(status: OverallStatus) -> int:
        return {
            OverallStatus.HEALTHY: ExitCode.OK,
            OverallStatus.REVIEW_REQUIRED: ExitCode.REVIEW,
            OverallStatus.UNHEALTHY: ExitCode.FAILURE,
        }[status]
