from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from kernelvet.diagnose.engine import DiagnoseOptions, run_diagnosis
from kernelvet.migration import analyze_project, audit_project
from kernelvet.models.report import DiagnosticReport, MigrationReport
from kernelvet.reporting import OutputFormat, UnsupportedFormat, format_from_path, render_text
from kernelvet.reporting.html import markdown_to_html
from kernelvet.utils.proc import HostContext


@pytest.fixture
def diag(mi300x_ctx: HostContext) -> DiagnosticReport:
    return run_diagnosis(mi300x_ctx, DiagnoseOptions())


def test_json_roundtrip_and_schema(diag: DiagnosticReport) -> None:
    text = render_text(diag, OutputFormat.JSON)
    data = json.loads(text)
    jsonschema.validate(data, DiagnosticReport.model_json_schema())
    assert DiagnosticReport.model_validate_json(text) == diag
    assert data["meta"]["tool"] == "kernelvet" and data["meta"]["generated_at"]
    assert "not an AMD product" in data["meta"]["disclaimer"]
    assert data["host"]["hostname_hash"] and "hostname" not in data["host"]


def test_markdown_sections(diag: DiagnosticReport) -> None:
    md = render_text(diag, OutputFormat.MARKDOWN)
    for heading in (
        "# KernelVet System Diagnosis",
        "## System",
        "## GPUs",
        "## Software",
        "## Runtime Checks (code run on the GPU, when one is available)",
        "## Evidence",
        "## Compatibility",
    ):
        assert heading in md
    assert "**Overall status: HEALTHY**" in md
    assert "AMD Instinct MI300X" in md


def test_html_is_self_contained_and_escaped(diag: DiagnosticReport) -> None:
    diag.checks[0].summary = "<script>alert(1)</script> | pipe"
    html = render_text(diag, OutputFormat.HTML)
    assert html.startswith("<!doctype html>")
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "<link" not in html and "src=" not in html
    assert 'class="PASS"' in html


def test_markdown_to_html_tables_and_code() -> None:
    html = markdown_to_html("# T\n\n| a | b |\n|---|---|\n| `x` | y\\|z |\n\n```\n<raw>\n```\n- item", "t")
    assert "<table>" in html and "<code>x</code>" in html and "y|z" in html
    assert "&lt;raw&gt;" in html and "<li>item</li>" in html


def test_sarif_from_audit(hip_project: Path) -> None:
    report = MigrationReport(
        project=str(hip_project), audit=audit_project(hip_project, installed_libraries=[])
    )
    sarif = json.loads(render_text(report, OutputFormat.SARIF))
    assert sarif["version"] == "2.1.0"
    run = sarif["runs"][0]
    assert run["tool"]["driver"]["name"] == "kernelvet"
    rule_ids = {r["id"] for r in run["tool"]["driver"]["rules"]}
    assert all(res["ruleId"] in rule_ids for res in run["results"])
    first = run["results"][0]
    assert first["level"] in ("note", "warning", "error")
    loc = first["locations"][0]["physicalLocation"]
    assert loc["artifactLocation"]["uriBaseId"] == "SRCROOT" and loc["region"]["startLine"] >= 1
    assert run["originalUriBaseIds"]["SRCROOT"]["uri"].startswith("file://")


def test_sarif_from_analysis(cuda_project: Path) -> None:
    report = MigrationReport(project=str(cuda_project), analysis=analyze_project(cuda_project))
    sarif = json.loads(render_text(report, OutputFormat.SARIF))
    assert len(sarif["runs"][0]["results"]) == len(report.analysis.findings)  # type: ignore[union-attr]


def test_sarif_is_only_for_findings(diag: DiagnosticReport) -> None:
    with pytest.raises(UnsupportedFormat):
        render_text(diag, OutputFormat.SARIF)


def test_migration_markdown_states_verification_limits(cuda_project: Path) -> None:
    report = MigrationReport(project=str(cuda_project), analysis=analyze_project(cuda_project))
    md = render_text(report, OutputFormat.MARKDOWN)
    assert "## Verification stages" in md and "No verification stage has passed yet." in md
    assert "not correctness" in md


def test_every_migration_format_says_conversion_is_not_guaranteed(cuda_project: Path) -> None:
    from kernelvet.models.report import HIPIFY_DISCLAIMER

    report = MigrationReport(project=str(cuda_project), analysis=analyze_project(cuda_project))
    for fmt in (OutputFormat.MARKDOWN, OutputFormat.HTML, OutputFormat.JSON):
        assert "not guaranteed to produce working HIP code" in render_text(report, fmt), fmt
    sarif = json.loads(render_text(report, OutputFormat.SARIF))
    assert sarif["runs"][0]["properties"]["conversionDisclaimer"] == HIPIFY_DISCLAIMER
    assert "manual" in HIPIFY_DISCLAIMER.lower() or "by hand" in HIPIFY_DISCLAIMER
    assert "compile" in HIPIFY_DISCLAIMER and "tests" in HIPIFY_DISCLAIMER


def test_sarif_results_are_marked_static_not_hardware_verified(cuda_project: Path) -> None:
    report = MigrationReport(project=str(cuda_project), analysis=analyze_project(cuda_project))
    run = json.loads(render_text(report, OutputFormat.SARIF))["runs"][0]
    assert "Nothing was compiled or run on an AMD GPU" in run["properties"]["evidence"]
    assert run["results"]
    assert all(r["properties"]["evidence"] == "static-analysis" for r in run["results"])
    assert not any(r["properties"]["hardwareVerified"] for r in run["results"])


def test_only_executed_passes_count_as_hardware_verified(diag: DiagnosticReport) -> None:
    checks = diag.all_checks()
    verified = [c for c in checks if c.hardware_verified]
    assert verified, "the fake MI300X host runs the HIP probe"
    assert all(c.evidence == "executed" and c.status == "PASS" for c in verified)
    assert diag.hardware_verified_checks == [c.id for c in verified]
    assert diag.compatibility is not None
    assert not any(c.hardware_verified for c in diag.compatibility.checks)
    data = json.loads(render_text(diag, OutputFormat.JSON))
    compat = data["compatibility"]["checks"]
    assert {c["evidence"] for c in compat} == {"configured"}  # stable JSON value
    assert {c["evidence_label"] for c in compat} == {"inferred from AMD docs"}
    assert not any(c["hardware_verified"] for c in compat)
    assert "Nothing is run on the GPU" in data["compatibility"]["basis"]
    assert "only a PASS with this evidence is verified on AMD hardware" in data["evidence_legend"]


def test_reports_label_inferred_results_as_not_hardware_verified(diag: DiagnosticReport) -> None:
    from rich.console import Console

    from kernelvet.reporting.terminal import render_diagnosis

    md = render_text(diag, OutputFormat.MARKDOWN)
    html_text = render_text(diag, OutputFormat.HTML)
    console = Console(width=200, record=True)
    console.print(render_diagnosis(diag))
    term = " ".join(console.export_text().split())
    for text in (md, html_text, term):
        assert "inferred from AMD docs" in text
        assert "Verified on AMD hardware" in text
        assert "| configured |" not in text  # raw enum values are not the displayed wording
    assert "Compatibility, inferred from AMD docs" in md and "Compatibility, inferred from AMD" in term
    assert "Exact references per check:" in md and "Supported GPUs" in md


@pytest.mark.parametrize(
    "name,fmt",
    [
        ("r.json", OutputFormat.JSON),
        ("r.md", OutputFormat.MARKDOWN),
        ("r.HTML", OutputFormat.HTML),
        ("r.sarif", OutputFormat.SARIF),
        ("r.sarif.json", OutputFormat.JSON),
        ("r.txt", None),
    ],
)
def test_format_from_path(name: str, fmt: OutputFormat | None) -> None:
    assert format_from_path(name) == fmt


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("[]", 'expected a JSON object whose "kind"'),
        ('"text"', 'expected a JSON object whose "kind"'),
        ('{"kind": "nope"}', 'expected a JSON object whose "kind"'),
        ("{bad", "is not valid JSON (line 1, column 2"),
        ("", "is not valid JSON"),
        ('{"kind": "diagnosis"}', "is not a valid diagnosis report: host: Field required"),
    ],
)
def test_load_report_file_rejects_bad_files_cleanly(tmp_path: Path, content: str, message: str) -> None:
    from kernelvet.reporting.loader import ReportFileError, load_report_file

    path = tmp_path / "r.json"
    path.write_text(content)
    with pytest.raises(ReportFileError) as info:
        load_report_file(path)
    assert message in str(info.value) and str(path) in str(info.value)
    path.write_bytes(b"\xca\xfe\xba\xbe")
    with pytest.raises(ReportFileError, match="could not read"):
        load_report_file(path)


def test_shared_reports_say_not_affiliated_with_amd(diag: DiagnosticReport, mi300x_ctx: HostContext) -> None:
    from kernelvet.benchmark.engine import BenchmarkOptions, run_benchmarks
    from kernelvet.models.report import TRADEMARKS

    assert "not affiliated with" in diag.meta.disclaimer
    reports = [diag, run_benchmarks(mi300x_ctx, BenchmarkOptions(torch=False))]
    assert diag.compatibility is not None
    reports.append(diag.compatibility)
    for report in reports:
        for fmt in (OutputFormat.MARKDOWN, OutputFormat.HTML):
            text = render_text(report, fmt)
            assert "not affiliated with" in text, (type(report).__name__, fmt)
            assert text.count("trademarks of Advanced") == 1, (type(report).__name__, fmt)
    # HIP is not on AMD's trademark list, so it must not be claimed as one
    assert "HIP" not in TRADEMARKS


def test_cli_help_says_not_affiliated_with_amd() -> None:
    from typer.testing import CliRunner

    from kernelvet.cli.app import app

    out = " ".join(CliRunner().invoke(app, ["--help"]).stdout.split())
    assert "not affiliated with, sponsored by or endorsed by" in out and "trademarks of Advanced" in out
