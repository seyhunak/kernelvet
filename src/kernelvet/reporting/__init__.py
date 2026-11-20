"""Shared reporting engine used by the CLI and the TUI."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from kernelvet.models.report import (
    BenchmarkReport,
    CompatibilityReport,
    DiagnosticReport,
    MigrationReport,
)
from kernelvet.reporting.html import to_html
from kernelvet.reporting.json import to_json
from kernelvet.reporting.markdown import (
    benchmark_markdown,
    compatibility_markdown,
    diagnostic_markdown,
    migration_markdown,
)
from kernelvet.reporting.sarif import migration_to_sarif


class OutputFormat(StrEnum):
    TERMINAL = "terminal"
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    SARIF = "sarif"


EXTENSIONS = {
    OutputFormat.JSON: ".json",
    OutputFormat.MARKDOWN: ".md",
    OutputFormat.HTML: ".html",
    OutputFormat.SARIF: ".sarif",
}


def format_from_path(path: str) -> OutputFormat | None:
    lower = path.lower()
    for fmt, ext in EXTENSIONS.items():
        if lower.endswith(ext) or (fmt is OutputFormat.SARIF and lower.endswith(".sarif.json")):
            return fmt
    if lower.endswith(".markdown"):
        return OutputFormat.MARKDOWN
    if lower.endswith(".htm"):
        return OutputFormat.HTML
    return None


class UnsupportedFormat(ValueError):
    pass


def render_text(report: BaseModel, fmt: OutputFormat) -> str:
    """Render a report to a machine/text format (everything except TERMINAL)."""
    if fmt is OutputFormat.JSON:
        return to_json(report)
    if fmt is OutputFormat.HTML:
        return to_html(report)
    if fmt is OutputFormat.MARKDOWN:
        if isinstance(report, DiagnosticReport):
            return diagnostic_markdown(report)
        if isinstance(report, MigrationReport):
            return migration_markdown(report)
        if isinstance(report, BenchmarkReport):
            return benchmark_markdown(report)
        if isinstance(report, CompatibilityReport):
            return compatibility_markdown(report)
    if fmt is OutputFormat.SARIF:
        if isinstance(report, MigrationReport):
            return migration_to_sarif(report)
        raise UnsupportedFormat("SARIF is only available for migration analysis/audit findings")
    raise UnsupportedFormat(f"{fmt} is not supported for {type(report).__name__}")


__all__ = ["EXTENSIONS", "OutputFormat", "UnsupportedFormat", "format_from_path", "render_text"]
