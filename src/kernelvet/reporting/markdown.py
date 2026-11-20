"""Markdown export."""

from __future__ import annotations

from kernelvet.models.common import CheckResult, Evidence
from kernelvet.models.migration import Finding
from kernelvet.models.report import (
    TRADEMARKS,
    BenchmarkReport,
    CompatibilityReport,
    DiagnosticReport,
    MigrationReport,
    ReportMeta,
)
from kernelvet.models.system import GpuDevice, HostInfo, SoftwareInfo
from kernelvet.reporting.common import (
    fmt_bytes,
    gpu_cell,
    gpu_label,
    software_rows,
    stage_rows,
    verification_note,
)

ICON = {"PASS": "✅ PASS", "WARNING": "⚠️ WARNING", "FAIL": "❌ FAIL", "SKIPPED": "⏭️ SKIPPED"}


def _esc(text: str | None) -> str:
    return (text or "").replace("|", "\\|").replace("\n", "<br>")


def _footer() -> list[str]:
    """Trademark footnote at the end of every Markdown/HTML report (reports get shared)."""
    return ["", f"_{TRADEMARKS}_", ""]


def _meta(meta: ReportMeta) -> list[str]:
    return [
        f"_Generated {meta.generated_at} by KernelVet {meta.version}._",
        "",
        f"> {meta.disclaimer}",
        "",
    ]


def _host(host: HostInfo) -> list[str]:
    return [
        "## System",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Operating system | {_esc(host.os_pretty or host.os_name)} |",
        f"| Kernel | {_esc(host.kernel)} |",
        f"| Architecture | {_esc(host.architecture)} |",
        f"| Container | {'yes' if host.in_container else 'no'} |",
        "",
    ]


def _gpus(gpus: list[GpuDevice]) -> list[str]:
    out = ["## GPUs", ""]
    if not gpus:
        return [*out, "No GPU detected.", ""]
    out += ["| # | Device | Target | CUs | VRAM | PCI | Sources |", "|---|---|---|---|---|---|---|"]
    for g in gpus:
        out.append(
            f"| {g.index} | {_esc(gpu_label(g))} | {_esc(g.gfx_target)} | {g.compute_units or ''} | "
            f"{fmt_bytes(g.vram_bytes)} | {_esc(g.pci_bus)} | {', '.join(g.sources)} |"
        )
    return [*out, ""]


def _software(sw: SoftwareInfo) -> list[str]:
    out = ["## Software", "", "| Component | Value |", "|---|---|"]
    out += [f"| {k} | {_esc(v)} |" for k, v in software_rows(sw)]
    tools = ", ".join(f"`{t.name}`" for t in sw.tools if t.available) or "none"
    missing = ", ".join(f"`{t.name}`" for t in sw.tools if not t.available) or "none"
    return [*out, "", f"Tools found: {tools}", "", f"Tools missing: {missing}", ""]


def _checks(title: str, checks: list[CheckResult]) -> list[str]:
    if not checks:
        return []
    out = [f"## {title}", "", "| Check | Status | Evidence | Result |", "|---|---|---|---|"]
    for c in checks:
        out.append(f"| {_esc(c.name)} | {ICON[c.status]} | {c.evidence.label} | {_esc(c.summary)} |")
    return [*out, ""]


def _evidence(checks: list[CheckResult], legend: str) -> list[str]:
    verified = [c.name for c in checks if c.hardware_verified]
    detected = sum(1 for c in checks if c.evidence == Evidence.DETECTED)
    inferred = sum(1 for c in checks if c.evidence == Evidence.CONFIGURED)
    head = (
        f"**Verified on AMD hardware (ran on GPU and passed):** {len(verified)} check(s): {', '.join(verified)}"
        if verified
        else "**Verified on AMD hardware:** none. No check ran on the GPU and passed."
    )
    return [
        "## Evidence",
        "",
        head,
        "",
        f"Not hardware-verified: {detected} detected, not run · {inferred} inferred from AMD docs.",
        "",
        f"> {legend}",
        "",
    ]


def _findings(title: str, findings: list[Finding], limit: int = 200) -> list[str]:
    out = [f"## {title}", ""]
    if not findings:
        return [*out, "No findings.", ""]
    out += ["| Category | Severity | Location | Message | HIP equivalent |", "|---|---|---|---|---|"]
    for f in findings[:limit]:
        loc = f"`{f.file}:{f.line}`" if f.line else f"`{f.file}`"
        out.append(f"| {f.category} | {f.severity} | {loc} | {_esc(f.message)} | {_esc(f.hip_equivalent)} |")
    if len(findings) > limit:
        out.append(f"\n_{len(findings) - limit} more findings in the JSON report._")
    return [*out, ""]


def diagnostic_markdown(r: DiagnosticReport) -> str:
    lines = ["# KernelVet System Diagnosis", "", *_meta(r.meta), f"**Overall status: {r.status}**", ""]
    lines += _host(r.host) + _gpus(r.gpus) + _software(r.software)
    by_cat: dict[str, list[CheckResult]] = {}
    for c in r.checks:
        by_cat.setdefault(c.category, []).append(c)
    titles = {
        "system": "System Checks",
        "hardware": "Hardware Checks",
        "software": "Software Checks",
        "runtime": "Runtime Checks (code run on the GPU, when one is available)",
    }
    for cat, checks in by_cat.items():
        lines += _checks(titles.get(cat, cat.title()), checks)
    if r.compatibility:
        lines += compatibility_section(r.compatibility)
    lines += _evidence(r.all_checks(), r.evidence_legend)
    if r.recommendations:
        lines += ["## Recommendations", "", *[f"- {rec}" for rec in r.recommendations], ""]
    return "\n".join(lines + _footer())


def compatibility_section(c: CompatibilityReport) -> list[str]:
    lines = _checks(
        f"Compatibility, inferred from AMD docs (matrix: ROCm {c.matrix_version}, retrieved {c.matrix_retrieved})",
        c.checks,
    )
    lines += [f"> {c.basis}", "", "Sources:", "", *[f"- <{u}>" for u in c.matrix_sources], ""]
    cited = [(x.name, x.source) for x in c.checks if x.source]
    if cited:
        lines += ["Exact references per check:", "", *[f"- {_esc(n)}: {_esc(s)}" for n, s in cited], ""]
    return lines


def compatibility_markdown(c: CompatibilityReport, meta: ReportMeta | None = None) -> str:
    lines = [
        "# KernelVet Compatibility Report",
        "",
        *_meta(meta or c.meta),
        f"**Status: {c.status}**",
        "",
    ]
    lines += [*compatibility_section(c), f"> {c.evidence_legend}", ""]
    return "\n".join(lines + _footer())


def migration_markdown(r: MigrationReport) -> str:
    lines = [f"# KernelVet Migration Report: `{r.project}`", "", *_meta(r.meta)]
    lines += ["## Verification stages", "", "| Stage | Status |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in stage_rows(r.stages)]
    lines += ["", f"> {verification_note(r.stages)}", "", f"> **{r.conversion_disclaimer}**", ""]
    if r.analysis:
        a = r.analysis
        inv = a.inventory
        lines += [
            "## Project inventory",
            "",
            f"- Files scanned: {len(inv.files)} ({inv.total_lines} lines), CUDA-related: {len(inv.cuda_files)}",
            f"- Build systems: {', '.join(b.kind for b in inv.build_systems) or 'none detected'}",
            f"- Tests: {len(inv.test_files)} file(s)",
            "",
            "### Dependencies",
            "",
        ]
        if inv.dependencies:
            lines += ["| Dependency | Kind | HIP alternative | Note |", "|---|---|---|---|"]
            lines += [
                f"| {d.name} | {d.kind} | {_esc(d.hip_alternative)} | {_esc(d.note)} |"
                for d in inv.dependencies
            ]
        else:
            lines.append("No NVIDIA library dependencies detected.")
        lines += [
            "",
            "## Migration readiness",
            "",
            f"**Score: {a.readiness.score}/100** — {a.readiness.explanation}",
            "",
            *[f"- {k}: {v}" for k, v in a.readiness.by_category.items()],
            "",
        ]
        lines += _findings("Analysis findings", a.findings)
    if r.plan:
        p = r.plan
        lines += [
            "## Migration plan",
            "",
            f"Translator: **{p.translator or 'none available'}** — {p.translator_reason}",
            "",
        ]
        for s in p.steps:
            lines.append(f"{s.order}. **{s.title}**{' (blocking)' if s.blocking else ''} — {_esc(s.detail)}")
        lines += ["", f"Build recommendation: {p.build_recommendation}", ""]
    if r.conversion:
        cv = r.conversion
        lines += [
            "## Source conversion",
            "",
            f"Translator: `{cv.translator}` {cv.translator_version or ''}"
            f"{' (dry run)' if cv.dry_run else ''}",
            "",
            "| Source | Status | Warnings | Errors |",
            "|---|---|---|---|",
        ]
        lines += [f"| `{f.source}` | {f.status} | {len(f.warnings)} | {len(f.errors)} |" for f in cv.files]
        lines.append("")
    if r.audit:
        lines += [
            f"> {r.audit.note}",
            "",
            *_findings(f"Audit findings ({r.audit.files_scanned} files)", r.audit.findings),
        ]
    if r.build:
        b = r.build
        lines += [
            "## Build",
            "",
            f"- Status: **{b.status}** ({b.method})",
            f"- Command: `{' '.join(b.command)}`",
        ]
        if b.reason:
            lines.append(f"- Note: {b.reason}")
        if b.errors:
            lines += ["", "```", *b.errors[:30], "```"]
        lines.append("")
    if r.tests:
        t = r.tests
        lines += [
            "## Tests",
            "",
            f"- Status: **{t.status}**; AMD GPU present: {'yes' if t.gpu_present else 'no'}",
        ]
        if t.reason:
            lines.append(f"- Note: {t.reason}")
        if t.cases:
            lines += ["", "| Test | Status | GPU | Reference match |", "|---|---|---|---|"]
            lines += [
                f"| {c.name} | {c.status} | {gpu_cell(c)} | "
                f"{'' if c.expected_output_match is None else c.expected_output_match} |"
                for c in t.cases
            ]
        lines.append("")
    if r.recommendations:
        lines += ["## Recommendations", "", *[f"- {x}" for x in r.recommendations], ""]
    return "\n".join(lines + _footer())


def benchmark_markdown(r: BenchmarkReport) -> str:
    lines = ["# KernelVet Benchmarks", "", *_meta(r.meta), f"> {r.note}", ""]
    lines += _host(r.host) + _gpus(r.gpus)
    for b in r.results:
        lines += [f"## {b.name} ({b.backend}) — {b.status}", ""]
        if b.reason:
            lines += [b.reason, ""]
        if b.profiler:
            lines += [f"Profiler: {b.profiler}", ""]
        if b.measurements:
            lines += ["| Measurement | Value | Unit | Iterations |", "|---|---|---|---|"]
            lines += [f"| {m.name} | {m.value:.4g} | {m.unit} | {m.iterations} |" for m in b.measurements]
            lines.append("")
    return "\n".join(lines + _footer())
