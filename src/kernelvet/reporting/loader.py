"""Load a saved JSON report back into its model (used by `kernelvet export` and the TUI Reports pane)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from kernelvet.models.report import BenchmarkReport, CompatibilityReport, DiagnosticReport, MigrationReport

SavedReport = DiagnosticReport | CompatibilityReport | MigrationReport | BenchmarkReport
REPORT_KINDS: dict[str, type[SavedReport]] = {
    "diagnosis": DiagnosticReport,
    "compatibility": CompatibilityReport,
    "migration": MigrationReport,
    "benchmark": BenchmarkReport,
}


class ReportFileError(ValueError):
    """A file that is not a readable KernelVet JSON report. The message names the file and the problem."""


def load_report_file(path: Path) -> SavedReport:
    try:
        data = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError) as exc:
        raise ReportFileError(f"could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ReportFileError(
            f"{path} is not valid JSON (line {exc.lineno}, column {exc.colno}: {exc.msg})"
        ) from exc
    kind = data.get("kind") if isinstance(data, dict) else None
    model = REPORT_KINDS.get(kind) if isinstance(kind, str) else None
    if model is None:
        raise ReportFileError(
            f'{path} is not a KernelVet report: expected a JSON object whose "kind" is one of '
            f"{', '.join(REPORT_KINDS)}"
        )
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(x) for x in e['loc']) or '(top level)'}: {e['msg']}" for e in exc.errors()[:5]
        )
        more = f" (and {exc.error_count() - 5} more)" if exc.error_count() > 5 else ""
        raise ReportFileError(f"{path} is not a valid {kind} report: {problems}{more}") from exc
