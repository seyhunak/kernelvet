"""CUDA → HIP migration workflow in the TUI."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from kernelvet.diagnose.software import detect_libraries
from kernelvet.execution.compiler import build_project
from kernelvet.execution.runner import run_tests
from kernelvet.execution.sandbox import SandboxConfig
from kernelvet.history import HistoryError
from kernelvet.migration import (
    HipifyUnavailable,
    analyze_project,
    audit_project,
    build_plan,
    convert_project,
)
from kernelvet.migration.hipify import choose_translator
from kernelvet.migration.workflow import assemble_report, compute_stages, save_state
from kernelvet.models.migration import Finding, FindingCategory, StageStatus
from kernelvet.models.report import MigrationReport
from kernelvet.reporting import OutputFormat, render_text
from kernelvet.reporting.common import STAGE_LABELS, verification_note
from kernelvet.tui.screens.base import Pane
from kernelvet.tui.widgets.common import KeyValue, Panel, badge, category_text, reset_table
from kernelvet.tui.workers import ui_progress
from kernelvet.utils.paths import PathError, existing_dir

ACTIONS = ["mg-scan", "mg-plan", "mg-convert", "mg-audit", "mg-build", "mg-test", "mg-report"]


class MigrationPane(Pane):
    TITLE = "CUDA → HIP Migration"
    HELP = """\
## CUDA → HIP Migration

1. Enter the **CUDA project** directory and press **Scan** – a read-only inventory and analysis.
2. **Plan** – an actionable plan (translator choice, dependencies, manual changes, blockers).
3. Enter an **output** directory and press **Convert** – AMD HIPIFY translates sources into
   that directory. The original project is never modified.
4. **Audit** – static scan of the converted tree for leftover CUDA and risky patterns.
5. **Build** – compile with `hipcc` (or the project's HIP CMake build).
6. **Test** – run the project's tests; only this stage, on an AMD GPU, validates runtime behaviour.
7. **Report** – export the combined report (JSON/Markdown/HTML/SARIF).

Each action that cannot run on this host is disabled, and the reason is shown under the buttons.
The readiness score measures *source conversion coverage only* – not correctness.

Filter findings by category with the selector on the Findings tab.
"""

    def compose(self) -> ComposeResult:
        with Panel(title="Project"):
            with Horizontal(classes="field"):
                yield Label("CUDA project ", classes="label")
                yield Input(placeholder="/path/to/cuda-project", id="mg-project")
            with Horizontal(classes="field"):
                yield Label("HIP output   ", classes="label")
                yield Input(
                    placeholder="/path/to/hip-project (must not be inside the CUDA project)", id="mg-output"
                )
            with Horizontal(classes="buttons"):
                yield Button("Scan", id="mg-scan", variant="primary")
                yield Button("Plan", id="mg-plan", disabled=True)
                yield Button("Convert", id="mg-convert", disabled=True)
                yield Button("Audit", id="mg-audit", disabled=True)
                yield Button("Build", id="mg-build", disabled=True)
                yield Button("Test", id="mg-test", disabled=True)
                yield Button("Export report", id="mg-report", disabled=True)
            yield Static("", id="mg-availability", classes="note")
        with TabbedContent(id="mg-tabs"):
            with TabPane("Inventory", id="tab-inventory"):
                yield KeyValue(Text("Scan a project to see its inventory.", style="dim"), id="mg-summary")
                yield DataTable(id="mg-deps", cursor_type="row")
                yield DataTable(id="mg-files", cursor_type="row", zebra_stripes=True)
            with TabPane("Findings", id="tab-findings"):
                yield Select(
                    [(c.value, c.value) for c in FindingCategory], prompt="All categories", id="mg-filter"
                )
                yield DataTable(id="mg-findings", cursor_type="row", zebra_stripes=True)
                yield Static("", id="mg-finding-detail")
            with TabPane("Plan", id="tab-plan"):
                yield DataTable(id="mg-plan-table", cursor_type="row")
                yield Static("", id="mg-plan-detail")
            with TabPane("Results", id="tab-results"):
                yield DataTable(id="mg-stages")
                yield Static("", id="mg-verification")
                yield DataTable(id="mg-results", cursor_type="row")
            with TabPane("Log", id="tab-log"):
                yield RichLog(id="mg-log", wrap=True, markup=False, max_lines=2000)

    def on_mount(self) -> None:
        self._findings: list[Finding] = []
        self.refresh_availability()
        self.render_stages()

    def on_activated(self) -> None:
        self.refresh_availability()

    # ------------------------------------------------------------------ availability

    def refresh_availability(self) -> None:
        ctx, s = self.state.ctx, self.state.settings
        notes: list[str] = []
        choice = choose_translator(ctx, s.hipify_preference)
        has_hipcc = bool(ctx.which("hipcc")) or (s.isolation == "docker" and bool(ctx.which("docker")))
        m = self.state.migration
        analyzed = m.analysis is not None
        converted = m.conversion is not None and not m.conversion.dry_run
        out_set = bool(self.query_one("#mg-output", Input).value.strip())
        self._enable("mg-plan", analyzed)
        self._enable("mg-convert", analyzed and choice.name is not None and out_set)
        self._enable("mg-audit", converted)
        self._enable("mg-build", converted and has_hipcc)
        self._enable("mg-test", m.build is not None and m.build.status is StageStatus.PASSED)
        self._enable("mg-report", analyzed)
        if choice.name is None:
            notes.append(f"Convert unavailable: {choice.reason}.")
        else:
            notes.append(f"Translator: {choice.name} ({choice.reason}).")
        if not has_hipcc:
            notes.append(
                "Build unavailable: hipcc not found (or set isolation=docker with Docker installed)."
            )
        if not self.state.amd_gpus:
            notes.append(
                "No AMD GPU detected: tests can run only if the build works, and GPU validation stays unverified."
            )
        if analyzed and not out_set:
            notes.append("Enter an output directory to enable Convert.")
        self.query_one("#mg-availability", Static).update(Text("\n".join(notes), style="dim"))

    def _enable(self, wid: str, on_: bool) -> None:
        self.query_one(f"#{wid}", Button).disabled = not on_

    @on(Input.Changed, "#mg-output")
    def _output_changed(self) -> None:
        self.refresh_availability()

    @on(Input.Submitted, "#mg-project")
    def _project_submitted(self) -> None:
        self.scan()

    def _busy(self, busy: bool) -> None:
        for wid in ACTIONS:
            if busy:
                self.query_one(f"#{wid}", Button).disabled = True
        if not busy:
            self._enable("mg-scan", True)
            self.refresh_availability()

    def log_line(self, text: str) -> None:
        self.query_one("#mg-log", RichLog).write(text)

    def _progress(self, message: str, _fraction: float) -> None:
        self.log_line(message)

    # ------------------------------------------------------------------ actions

    def on_button_pressed(self, event: Button.Pressed) -> None:
        handlers = {
            "mg-scan": self.scan,
            "mg-plan": self.plan,
            "mg-convert": self.convert,
            "mg-audit": self.audit,
            "mg-build": self.build,
            "mg-test": self.test,
            "mg-report": self.export_report,
        }
        h = handlers.get(event.button.id or "")
        if h:
            h()

    def scan(self) -> None:
        raw = self.query_one("#mg-project", Input).value.strip()
        try:
            root = existing_dir(raw or ".")
        except PathError as exc:
            self.report_error("Invalid project", exc)
            return
        self._busy(True)
        self.log_line(f"Scanning {root}")
        self._scan_worker(root)

    @work(thread=True, exclusive=True, group="migration", exit_on_error=False)
    def _scan_worker(self, root: Path) -> None:
        try:
            analysis = analyze_project(root)
        except Exception as exc:
            self.app.call_from_thread(self._error, "Scan failed", exc)
            return
        self.app.call_from_thread(self._scanned, root, analysis)

    def _scanned(self, root: Path, analysis) -> None:
        m = self.state.migration
        m.project, m.analysis = str(root), analysis
        m.plan = m.conversion = m.audit = m.build = m.tests = None
        inv = analysis.inventory
        r = analysis.readiness
        self.query_one("#mg-summary", KeyValue).set_rows(
            [
                ("Project", str(root)),
                ("Files", f"{len(inv.files)} ({inv.total_lines} lines), {len(inv.cuda_files)} CUDA-related"),
                (
                    "Build systems",
                    ", ".join(f"{b.kind} ({', '.join(b.files)})" for b in inv.build_systems) or "none",
                ),
                ("Tests", f"{len(inv.test_files)} file(s)"),
                (
                    "Readiness",
                    Text(f"{r.score}/100", style="bold")
                    + Text("  source conversion coverage only — not correctness", style="dim"),
                ),
                ("Findings", ", ".join(f"{k}: {v}" for k, v in r.by_category.items())),
            ]
        )
        deps = self.query_one("#mg-deps", DataTable)
        reset_table(deps, "Dependency", "Kind", "HIP alternative", "Note", "Evidence")
        for d in inv.dependencies:
            deps.add_row(d.name, d.kind, d.hip_alternative or "—", d.note or "", ", ".join(d.evidence[:3]))
        files = self.query_one("#mg-files", DataTable)
        reset_table(files, "File", "Kind", "Lines", "CUDA calls", "Kernels", "Test")
        for f in inv.files:
            files.add_row(
                f.path,
                f.kind.value,
                str(f.lines),
                str(f.cuda_api_calls),
                str(f.kernels),
                "yes" if f.is_test else "",
            )
        self._findings = analysis.findings
        self.render_findings()
        if not self.query_one("#mg-output", Input).value.strip():
            self.query_one("#mg-output", Input).value = str(root.parent / f"{root.name}-hip")
        self.log_line(f"Found {len(analysis.findings)} findings; readiness {r.score}/100")
        self._save_history(
            "analyzed", f"readiness {r.score}/100, {len(analysis.findings)} findings", str(root)
        )
        self._busy(False)
        self.render_stages()

    def render_findings(self) -> None:
        sel = self.query_one("#mg-filter", Select).value
        t = self.query_one("#mg-findings", DataTable)
        reset_table(t, "Category", "Severity", "Location", "Message", "HIP")
        for i, f in enumerate(self._findings):
            if isinstance(sel, str) and f.category.value != sel:
                continue
            t.add_row(
                category_text(f.category),
                f.severity.value,
                f"{f.file}:{f.line}" if f.line else f.file,
                f.message,
                f.hip_equivalent or "",
                key=str(i),
            )

    @on(Select.Changed, "#mg-filter")
    def _filter(self) -> None:
        self.render_findings()

    @on(DataTable.RowHighlighted, "#mg-findings")
    def _finding(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is None or event.row_key.value is None:
            return
        f = self._findings[int(event.row_key.value)]
        text = Text(f"{f.file}:{f.line or ''}  ", style="bold")
        text.append_text(category_text(f.category))
        text.append(f"  {f.rule_id}\n", style="dim")
        text.append(f.message + "\n")
        if f.snippet:
            text.append(f"  {f.snippet}\n", style="italic")
        if f.hip_equivalent:
            text.append(f"HIP: {f.hip_equivalent}\n", style="green")
        if f.suggestion:
            text.append(f"Suggestion: {f.suggestion}\n", style="yellow")
        self.query_one("#mg-finding-detail", Static).update(text)

    def plan(self) -> None:
        m = self.state.migration
        if m.analysis is None:
            return
        p = build_plan(
            m.analysis,
            self.state.ctx,
            preference=self.state.settings.hipify_preference,
            offload_arch=self.state.gfx_targets,
        )
        m.plan = p
        t = self.query_one("#mg-plan-table", DataTable)
        reset_table(t, "#", "Step", "Detail", "Files")
        for s in p.steps:
            t.add_row(
                str(s.order),
                Text(s.title, style="bold red" if s.blocking else "bold"),
                s.detail,
                str(len(s.files)) if s.files else "",
            )
        detail = Text()
        detail.append(
            f"Translator: {p.translator or 'none available'} — {p.translator_reason}\n", style="bold"
        )
        detail.append(
            f"Files to migrate: {len(p.files_to_migrate)} · copied unchanged: {len(p.files_to_copy)}\n"
        )
        detail.append(f"Blockers: {len(p.blockers)} · manual changes: {len(p.manual_changes)}\n")
        detail.append(f"Build: {p.build_recommendation}\n", style="dim")
        if p.api_conversions:
            sample = list(p.api_conversions.items())[:12]
            detail.append(
                "API conversions (sample): " + ", ".join(f"{a}→{b}" for a, b in sample), style="dim"
            )
        self.query_one("#mg-plan-detail", Static).update(detail)
        self.query_one("#mg-tabs", TabbedContent).active = "tab-plan"
        self.log_line(f"Plan: {len(p.steps)} steps, {len(p.blockers)} blockers")
        self._save_history("planned", f"{len(p.steps)} steps, {len(p.blockers)} blockers", m.project)

    def convert(self) -> None:
        m = self.state.migration
        out = self.query_one("#mg-output", Input).value.strip()
        if m.analysis is None or not out:
            return
        self._busy(True)
        self.query_one("#mg-tabs", TabbedContent).active = "tab-log"
        self._convert_worker(out)

    @work(thread=True, exclusive=True, group="migration", exit_on_error=False)
    def _convert_worker(self, out: str) -> None:
        m = self.state.migration
        s = self.state.settings
        assert m.analysis is not None  # the Convert button needs an analysis first
        try:
            result = convert_project(
                self.state.ctx,
                m.analysis.inventory,
                out,
                preference=s.hipify_preference,
                timeout=s.build_timeout,
                progress=ui_progress(self.app, self._progress),
            )
            save_state(result.output, "conversion", result)
        except (HipifyUnavailable, PathError) as exc:
            self.app.call_from_thread(self._error, "Convert", exc)
            return
        except Exception as exc:
            self.app.call_from_thread(self._error, "Convert failed", exc)
            return

        def done() -> None:
            m.output, m.conversion = result.output, result
            m.audit = m.build = m.tests = None
            failed = len(result.failed)
            self.log_line(f"Converted into {result.output} — {failed} file(s) failed; log: {result.log_path}")
            self._results_table()
            self._save_history("converted", f"{len(result.files)} files, {failed} failed", result.output)
            self._busy(False)
            self.render_stages()
            self.app.notify(
                f"{failed} failed" if failed else "All files converted",
                title="Conversion",
                severity="warning" if failed else "information",
            )

        self.app.call_from_thread(done)

    def audit(self) -> None:
        m = self.state.migration
        if not m.output:
            return
        self._busy(True)
        self._audit_worker(m.output)

    @work(thread=True, exclusive=True, group="migration", exit_on_error=False)
    def _audit_worker(self, out: str) -> None:
        ctx = self.state.ctx
        rocm = ctx.rocm_path()
        try:
            result = audit_project(
                out, installed_libraries=detect_libraries(ctx, rocm) if ctx.exists(rocm) else None
            )
            save_state(out, "audit", result)
        except Exception as exc:
            self.app.call_from_thread(self._error, "Audit failed", exc)
            return

        def done() -> None:
            self.state.migration.audit = result
            self._findings = result.findings
            self.render_findings()
            counts = Counter(f.category.value for f in result.findings)
            self.log_line(f"Audit: {len(result.findings)} findings {dict(counts)}")
            self.query_one("#mg-tabs", TabbedContent).active = "tab-findings"
            self._save_history("audited", f"{len(result.findings)} findings", out)
            self._busy(False)
            self.render_stages()

        self.app.call_from_thread(done)

    def build(self) -> None:
        m = self.state.migration
        if not m.output:
            return
        self._busy(True)
        self.query_one("#mg-tabs", TabbedContent).active = "tab-log"
        self.log_line("Building with hipcc…")
        self._build_worker(m.output)

    @work(thread=True, exclusive=True, group="migration", exit_on_error=False)
    def _build_worker(self, out: str) -> None:
        s = self.state.settings
        try:
            result = build_project(
                self.state.ctx,
                out,
                gfx_targets=self.state.gfx_targets,
                isolation=s.isolation,
                sandbox=SandboxConfig(image=s.docker_image),
                timeout=s.build_timeout,
            )
            save_state(out, "build", result)
        except Exception as exc:
            self.app.call_from_thread(self._error, "Build failed", exc)
            return

        def done() -> None:
            self.state.migration.build = result
            self.state.migration.tests = None
            self.log_line(f"Build {result.status} ({result.method}) {result.reason or ''}")
            for e in result.errors[:40]:
                self.log_line("  " + e)
            self._results_table()
            self._save_history("built", f"build {result.status}", out)
            self._busy(False)
            self.render_stages()

        self.app.call_from_thread(done)

    def test(self) -> None:
        m = self.state.migration
        if not m.output:
            return
        self._busy(True)
        self.log_line("Running tests…")
        self._test_worker(m.output)

    @work(thread=True, exclusive=True, group="migration", exit_on_error=False)
    def _test_worker(self, out: str) -> None:
        s = self.state.settings
        try:
            result = run_tests(
                self.state.ctx,
                out,
                self.state.migration.build,
                isolation=s.isolation,
                sandbox=SandboxConfig(image=s.docker_image),
                timeout=s.test_timeout,
            )
            save_state(out, "tests", result)
        except Exception as exc:
            self.app.call_from_thread(self._error, "Tests failed", exc)
            return

        def done() -> None:
            self.state.migration.tests = result
            self.log_line(
                f"Tests {result.status}; AMD GPU present: {result.gpu_present}. {result.reason or ''}"
            )
            self._results_table()
            self._save_history("tested", f"tests {result.status}", out)
            self._busy(False)
            self.render_stages()

        self.app.call_from_thread(done)

    def export_report(self) -> None:
        m = self.state.migration
        report = self._report()
        if report is None:
            return
        d = self.state.history.reports_dir
        try:
            d.mkdir(parents=True, exist_ok=True)
            base = (
                d
                / f"migration-{Path(m.output or m.project).name}-{report.meta.generated_at.replace(':', '')}"
            )
            written = []
            for fmt, ext in (
                (OutputFormat.JSON, ".json"),
                (OutputFormat.MARKDOWN, ".md"),
                (OutputFormat.HTML, ".html"),
                (OutputFormat.SARIF, ".sarif"),
            ):
                p = base.with_suffix(ext)
                p.write_text(render_text(report, fmt))
                written.append(p.name)
        except OSError as exc:
            self.report_error("Export failed", exc)
            return
        self.app.notify(f"{d}: {', '.join(written)}", title="Migration report exported", timeout=8)
        reports = self.rapp.pane("reports")
        if hasattr(reports, "refresh_list"):
            reports.refresh_list()

    # ------------------------------------------------------------------ helpers

    def _report(self) -> MigrationReport | None:
        m = self.state.migration
        if m.analysis is None:
            return None
        if m.output and Path(m.output).is_dir():
            r = assemble_report(m.output, analysis=m.analysis)
            r.plan = m.plan
            return r
        r = MigrationReport(project=m.project, analysis=m.analysis, plan=m.plan)
        return r

    def render_stages(self) -> None:
        m = self.state.migration
        stages = compute_stages(m.conversion, m.audit, m.build, m.tests)
        t = self.query_one("#mg-stages", DataTable)
        reset_table(t, "Verification stage", "Status")
        for key, label in STAGE_LABELS.items():
            t.add_row(label, badge(getattr(stages, key)))
        self.query_one("#mg-verification", Static).update(Text(verification_note(stages), style="italic"))

    def _results_table(self) -> None:
        m = self.state.migration
        t = self.query_one("#mg-results", DataTable)
        reset_table(t, "Item", "Status", "Detail")
        if m.conversion:
            for f in m.conversion.files:
                if f.status != "copied":
                    t.add_row(f.source, f.status, "; ".join((f.errors or f.warnings)[:2]))
        if m.build:
            t.add_row("build", badge(m.build.status), m.build.reason or " ".join(m.build.command)[:120])
        if m.tests:
            for c in m.tests.cases:
                t.add_row(f"test: {c.name}", badge(c.status), f"GPU: {'yes' if c.ran_on_gpu else 'no'}")
            if not m.tests.cases:
                t.add_row("tests", badge(m.tests.status), m.tests.reason or "")

    def _error(self, title: str, exc: BaseException) -> None:
        self.log_line(f"{title}: {exc}")
        self.report_error(title, exc)
        self._busy(False)

    def _save_history(self, stage: str, summary: str, path: str) -> None:
        if not self.state.settings.history or not path:
            return
        try:
            self.state.history.record_project(path, stage, summary)
        except HistoryError as exc:
            self.report_error("History not saved", exc)
