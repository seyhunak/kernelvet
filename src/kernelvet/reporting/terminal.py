"""Rich renderables for terminal output (CLI). The TUI reuses the status styles."""

from __future__ import annotations

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from kernelvet.models.common import EVIDENCE_LEGEND, CheckResult, CheckStatus, Evidence, OverallStatus
from kernelvet.models.migration import Finding, FindingCategory, StageStatus
from kernelvet.models.report import (
    BenchmarkReport,
    CompatibilityReport,
    DiagnosticReport,
    MigrationReport,
)
from kernelvet.models.system import GpuDevice, HostInfo, SoftwareInfo
from kernelvet.reporting.common import (
    compile_db_line,
    fmt_bytes,
    gpu_cell,
    gpu_label,
    software_rows,
    stage_rows,
    verification_note,
)

STATUS_STYLE = {
    CheckStatus.PASS: "bold green",
    CheckStatus.WARNING: "bold yellow",
    CheckStatus.FAIL: "bold red",
    CheckStatus.SKIPPED: "dim",
}
OVERALL_STYLE = {
    OverallStatus.HEALTHY: "bold green",
    OverallStatus.REVIEW_REQUIRED: "bold yellow",
    OverallStatus.UNHEALTHY: "bold red",
}
STAGE_STYLE = {
    StageStatus.PASSED: "bold green",
    StageStatus.FAILED: "bold red",
    StageStatus.UNAVAILABLE: "yellow",
    StageStatus.NOT_RUN: "dim",
}
CATEGORY_STYLE = {
    FindingCategory.AUTO_CONVERTIBLE: "green",
    FindingCategory.MANUAL_INTERVENTION: "yellow",
    FindingCategory.UNSUPPORTED: "red",
    FindingCategory.REQUIRES_VERIFICATION: "cyan",
}


def status_text(status: CheckStatus | str) -> Text:
    s = CheckStatus(status)
    return Text(s.value, style=STATUS_STYLE[s])


def kv_table(title: str, rows: list[tuple[str, str]]) -> Table:
    t = Table(
        title=title, title_justify="left", title_style="bold", show_header=False, box=None, padding=(0, 2)
    )
    t.add_column(style="cyan", no_wrap=True)
    t.add_column()
    for k, v in rows:
        t.add_row(k, v)
    return t


def host_table(host: HostInfo) -> Table:
    return kv_table(
        "System",
        [
            ("Operating System", host.os_pretty or host.os_name),
            ("Kernel", host.kernel or "unknown"),
            ("Architecture", host.architecture),
            ("Container", "yes" if host.in_container else "no"),
        ],
    )


def gpu_table(gpus: list[GpuDevice]) -> Table:
    t = Table(title="GPU", title_justify="left", title_style="bold", expand=False)
    for col in ("#", "Device", "Target", "CUs", "VRAM", "PCI", "Detected by"):
        t.add_column(col)
    if not gpus:
        t.add_row("-", Text("No GPU detected", style="yellow"), "", "", "", "", "")
    for g in gpus:
        t.add_row(
            str(g.index),
            gpu_label(g) + ("" if g.vendor_is_amd else " (non-AMD)"),
            g.gfx_target or "?",
            str(g.compute_units or "?"),
            fmt_bytes(g.vram_bytes),
            g.pci_bus or "",
            ", ".join(g.sources),
        )
    return t


def software_table(sw: SoftwareInfo) -> Table:
    return kv_table("Software", software_rows(sw))


def checks_table(title: str, checks: list[CheckResult], *, verbose: bool = False) -> Table:
    t = Table(title=title, title_justify="left", title_style="bold")
    t.add_column("Check", no_wrap=True)
    t.add_column("Status", no_wrap=True)
    t.add_column("Evidence", style="dim", no_wrap=True)
    t.add_column("Result")
    for c in checks:
        result = Text(c.summary)
        if verbose and c.detail:
            result.append("\n" + c.detail, style="dim")
        if c.remediation and c.status in (CheckStatus.FAIL, CheckStatus.WARNING):
            result.append("\n→ " + c.remediation, style="italic")
        t.add_row(c.name, status_text(c.status), c.evidence.label, result)
    return t


def evidence_summary(checks: list[CheckResult]) -> Text:
    """One line saying how many results were verified on AMD hardware versus detected or inferred."""
    verified = [c for c in checks if c.hardware_verified]
    detected = sum(1 for c in checks if c.evidence == Evidence.DETECTED)
    inferred = sum(1 for c in checks if c.evidence == Evidence.CONFIGURED)
    if verified:
        head = Text(
            f"Verified on AMD hardware (ran on GPU and passed): {len(verified)} check(s)", style="green"
        )
    else:
        head = Text("Verified on AMD hardware: none. No check ran on the GPU and passed", style="yellow")
    head.append(
        f"\nNot hardware-verified: {detected} detected, not run · {inferred} inferred from AMD docs",
        style="dim",
    )
    return head


def evidence_legend() -> Text:
    return Text(EVIDENCE_LEGEND, style="dim")


def overall_panel(status: OverallStatus) -> Panel:
    return Panel(Text(f"Overall Status: {status.value}", style=OVERALL_STYLE[status]), expand=False)


def render_diagnosis(r: DiagnosticReport, *, verbose: bool = False) -> RenderableType:
    parts: list[RenderableType] = [
        Text("KernelVet System Diagnosis", style="bold"),
        host_table(r.host),
        gpu_table(r.gpus),
        software_table(r.software),
    ]
    cats = {
        "system": "System",
        "hardware": "Hardware",
        "software": "Software",
        "runtime": "Runtime (code run on the GPU, when one is available)",
    }
    for cat, title in cats.items():
        checks = [c for c in r.checks if c.category == cat]
        if checks:
            parts.append(checks_table(title, checks, verbose=verbose))
    if r.compatibility:
        parts.append(render_compatibility(r.compatibility, verbose=verbose, legend=False))
    parts += [evidence_summary(r.all_checks()), evidence_legend(), overall_panel(r.status)]
    return Group(*parts)


def render_compatibility(
    c: CompatibilityReport, *, verbose: bool = False, legend: bool = True
) -> RenderableType:
    title = f"Compatibility, inferred from AMD ROCm {c.matrix_version} docs (retrieved {c.matrix_retrieved})"
    parts: list[RenderableType] = [
        checks_table(title, c.checks, verbose=verbose),
        Text(c.basis, style="yellow"),
        Text("Sources: " + " | ".join(c.matrix_sources), style="dim"),
    ]
    if legend:
        parts.append(evidence_legend())
    return Group(*parts)


def findings_table(title: str, findings: list[Finding], limit: int = 50) -> Table:
    t = Table(title=title, title_justify="left", title_style="bold")
    for col in ("Category", "Sev", "Location", "Message", "HIP"):
        t.add_column(col)
    for f in findings[:limit]:
        t.add_row(
            Text(str(f.category), style=CATEGORY_STYLE[f.category]),
            str(f.severity),
            f"{f.file}:{f.line}" if f.line else f.file,
            f.message,
            f.hip_equivalent or "",
        )
    if len(findings) > limit:
        t.caption = f"{len(findings) - limit} more — use --format json for all findings"
    return t


def render_migration(r: MigrationReport) -> RenderableType:
    parts: list[RenderableType] = [Text(f"KernelVet Migration: {r.project}", style="bold")]
    if r.analysis:
        a = r.analysis
        inv = a.inventory
        parts.append(
            kv_table(
                "Inventory",
                [
                    ("Files", f"{len(inv.files)} ({inv.total_lines} lines)"),
                    ("CUDA-related", str(len(inv.cuda_files))),
                    ("Build systems", ", ".join(b.kind for b in inv.build_systems) or "none"),
                    ("Compile database", compile_db_line(inv.compile_db)),
                    ("Dependencies", ", ".join(d.name for d in inv.dependencies) or "none"),
                    ("Tests", str(len(inv.test_files))),
                    ("Readiness", f"{a.readiness.score}/100 (source conversion coverage only)"),
                ],
            )
        )
        parts.append(findings_table("Findings", a.findings))
    if r.plan:
        p = r.plan
        t = Table(
            title=f"Plan — translator: {p.translator or 'none available'}",
            title_justify="left",
            title_style="bold",
        )
        t.add_column("#")
        t.add_column("Step")
        t.add_column("Detail")
        for s in p.steps:
            t.add_row(str(s.order), Text(s.title, style="red" if s.blocking else ""), s.detail)
        parts.append(t)
    if r.conversion:
        cv = r.conversion
        t = Table(
            title=f"Conversion ({cv.translator}{', dry run' if cv.dry_run else ''})",
            title_justify="left",
            title_style="bold",
        )
        for col in ("Source", "Status", "Warnings", "Errors"):
            t.add_column(col)
        for f in cv.files:
            style = {"failed": "red", "converted": "green"}.get(f.status, "")
            t.add_row(f.source, Text(f.status, style=style), str(len(f.warnings)), str(len(f.errors)))
        parts.append(t)
    if r.audit:
        parts.append(findings_table(f"Audit ({r.audit.files_scanned} files)", r.audit.findings))
        parts.append(Text(r.audit.note, style="dim italic"))
    if r.build:
        b = r.build
        parts.append(
            kv_table(
                "Build",
                [
                    ("Status", b.status),
                    ("Method", b.method),
                    ("Command", " ".join(b.command)),
                    ("Note", b.reason or ""),
                ],
            )
        )
        if b.errors:
            parts.append(Panel("\n".join(b.errors[:20]), title="Compiler errors", border_style="red"))
    if r.tests:
        tr = r.tests
        t = Table(title=f"Tests — {tr.status}", title_justify="left", title_style="bold")
        for col in ("Test", "Status", "GPU", "Reference"):
            t.add_column(col)
        for c in tr.cases:
            t.add_row(
                c.name,
                Text(c.status, style=STAGE_STYLE[c.status]),
                gpu_cell(c),
                "" if c.expected_output_match is None else str(c.expected_output_match),
            )
        if tr.reason:
            t.caption = tr.reason
        parts.append(t)
    st = Table(title="Verification stages", title_justify="left", title_style="bold")
    st.add_column("Stage")
    st.add_column("Status")
    for label, value in stage_rows(r.stages):
        st.add_row(label, Text(value, style=STAGE_STYLE[StageStatus(value)]))
    parts += [
        st,
        Text(verification_note(r.stages), style="italic"),
        Text(r.conversion_disclaimer, style="yellow"),
    ]
    return Group(*parts)


def render_benchmarks(r: BenchmarkReport) -> RenderableType:
    t = Table(title="Benchmarks", title_justify="left", title_style="bold")
    for col in ("Benchmark", "Backend", "Status", "Measurement", "Value", "Unit"):
        t.add_column(col)
    for b in r.results:
        style = STATUS_STYLE.get(CheckStatus(b.status), "")
        if not b.measurements:
            t.add_row(b.name, b.backend, Text(b.status, style=style), b.reason or "", "", "")
        for i, m in enumerate(b.measurements):
            t.add_row(
                b.name if i == 0 else "",
                b.backend if i == 0 else "",
                Text(b.status, style=style) if i == 0 else "",
                m.name,
                f"{m.value:.4g}",
                m.unit,
            )
    profilers = [Text(f"Profiler ({b.name}): {b.profiler}", style="dim") for b in r.results if b.profiler]
    return Group(t, *profilers, Text(r.note, style="dim italic"))
