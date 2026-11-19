"""Diagnose: run the full diagnostic with live progress, inspect and export results."""

from __future__ import annotations

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Button, Checkbox, DataTable, ProgressBar, RichLog, Static

from kernelvet.diagnose.engine import DiagnoseOptions, run_diagnosis
from kernelvet.history import HistoryError
from kernelvet.models.common import CheckResult, CheckStatus
from kernelvet.models.report import DiagnosticReport
from kernelvet.reporting import OutputFormat, render_text
from kernelvet.reporting.common import software_rows
from kernelvet.tui.screens.base import Pane
from kernelvet.tui.widgets.common import KeyValue, Panel, badge, reset_table
from kernelvet.tui.workers import ui_progress


class DiagnosePane(Pane):
    TITLE = "Diagnose"
    BINDINGS = [
        Binding("r", "run", "Run diagnosis"),
        Binding("e", "export('json')", "Export JSON"),
        Binding("m", "export('markdown')", "Export MD"),
        Binding("h", "export('html')", "Export HTML"),
    ]
    HELP = """\
## Diagnose

Runs the complete diagnostic on this machine:

1. **Hardware** – GPUs from `amd-smi`, `rocminfo` and `/sys`, plus `/dev/kfd` access.
2. **Software** – ROCm, HIP runtime/compiler, amdgpu driver, ROCm libraries, tools, PyTorch.
3. **Runtime** – compiles a HIP probe with `hipcc` and **executes** it: init, device enumeration,
   allocation, transfers, a vector-add kernel and result verification. PyTorch tensor ops on the GPU.
4. **Compatibility** – against the bundled AMD ROCm compatibility snapshot.

The *Evidence* column says how each result was obtained: `detected` (static), `executed`
(ran on the GPU), `configured` (reference data) or `none` (not performed).
SKIPPED never counts as success.

Select a row to see details and the suggested fix.

Keys: **r** run · **e** export JSON · **m** export Markdown · **h** export HTML.
"""

    def compose(self) -> ComposeResult:
        with Horizontal(classes="buttons"):
            yield Button("Run diagnosis", id="dg-run", variant="primary")
            yield Checkbox("HIP probe", value=True, id="dg-runtime")
            yield Checkbox("PyTorch checks", value=True, id="dg-torch")
            yield Button("Export JSON", id="dg-json", disabled=True)
            yield Button("Export Markdown", id="dg-md", disabled=True)
            yield Button("Export HTML", id="dg-html", disabled=True)
        yield ProgressBar(total=100, show_eta=False, id="dg-progress")
        yield Static(Text("Press r to run a diagnosis.", style="dim"), id="dg-status")
        with Horizontal(classes="row"):
            with Panel(title="Hardware & host", classes="half"):
                yield KeyValue("", id="dg-hw")
            with Panel(title="Software inventory", classes="half"):
                yield KeyValue("", id="dg-sw")
        with Panel(title="Checks"):
            yield DataTable(id="dg-checks", cursor_type="row", zebra_stripes=True)
        with Panel(title="Check detail"):
            yield Static(Text("Select a check to see its detail.", style="dim"), id="dg-detail")
        with Panel(title="Progress log"):
            yield RichLog(id="dg-log", max_lines=500, wrap=True, markup=False)

    def on_mount(self) -> None:
        self._checks: list[CheckResult] = []
        cb = self.query_one("#dg-runtime", Checkbox)
        cb.value = self.state.settings.runtime_checks
        self.query_one("#dg-torch", Checkbox).value = self.state.settings.torch_checks
        if self.state.diagnosis:
            self.show_report(self.state.diagnosis)

    def action_run(self) -> None:
        self.start()

    def start(self) -> None:
        for wid in ("dg-run", "dg-json", "dg-md", "dg-html"):
            self.query_one(f"#{wid}", Button).disabled = True
        self.query_one("#dg-progress", ProgressBar).update(progress=0)
        self.query_one("#dg-log", RichLog).clear()
        self._run_worker(
            self.query_one("#dg-runtime", Checkbox).value, self.query_one("#dg-torch", Checkbox).value
        )

    @work(thread=True, exclusive=True, group="diagnose", exit_on_error=False)
    def _run_worker(self, runtime: bool, torch: bool) -> None:
        s = self.state.settings
        opts = DiagnoseOptions(
            runtime=runtime, torch=torch, python=s.python_executable, matrix_path=s.compatibility_matrix
        )
        try:
            report = run_diagnosis(self.state.ctx, opts, ui_progress(self.app, self._on_progress))
        except Exception as exc:
            self.app.call_from_thread(self._failed, exc)
            return
        self.app.call_from_thread(self._finished, report)

    def _on_progress(self, message: str, fraction: float) -> None:
        self.query_one("#dg-progress", ProgressBar).update(progress=fraction * 100)
        self.query_one("#dg-status", Static).update(Text(message, style="cyan"))
        self.query_one("#dg-log", RichLog).write(f"[{int(fraction * 100):3d}%] {message}")

    def _failed(self, exc: BaseException) -> None:
        self.query_one("#dg-run", Button).disabled = False
        self.query_one("#dg-status", Static).update(Text(f"Diagnosis failed: {exc}", style="red"))
        self.report_error("Diagnosis failed", exc)

    def _finished(self, report: DiagnosticReport) -> None:
        st = self.state
        st.diagnosis = report
        st.host, st.gpus, st.software = report.host, report.gpus, report.software
        st.compatibility = report.compatibility
        if st.settings.history:
            counts = {s: sum(1 for c in report.all_checks() if c.status == s) for s in CheckStatus}
            try:
                st.history.record_run(
                    "diagnosis",
                    report.status.value,
                    ", ".join(f"{n} {s.value}" for s, n in counts.items() if n),
                    report,
                )
            except HistoryError as exc:
                self.report_error("History not saved", exc)
        self.show_report(report)
        self.query_one("#dg-run", Button).disabled = False
        self.app.notify(
            f"Overall status: {report.status.value}",
            title="Diagnosis complete",
            severity="information" if report.status == "HEALTHY" else "warning",
        )

    def show_report(self, report: DiagnosticReport) -> None:
        for wid in ("dg-json", "dg-md", "dg-html"):
            self.query_one(f"#{wid}", Button).disabled = False
        h = report.host
        amd = [g for g in report.gpus if g.vendor_is_amd]
        self.query_one("#dg-hw", KeyValue).set_rows(
            [
                ("OS", f"{h.os_pretty or h.os_name}"),
                ("Kernel", h.kernel or "unknown"),
                ("Arch", h.architecture),
                (
                    "AMD GPUs",
                    ", ".join(f"{g.marketing_name or '?'} ({g.gfx_target or '?'})" for g in amd) or "none",
                ),
                ("Overall", badge(report.status)),
            ]
        )
        self.query_one("#dg-sw", KeyValue).set_rows(list(software_rows(report.software)))
        table = self.query_one("#dg-checks", DataTable)
        reset_table(table, "Category", "Check", "Status", "Evidence", "Result")
        self._checks = report.all_checks()
        for i, c in enumerate(self._checks):
            table.add_row(c.category, c.name, badge(c.status), c.evidence.value, c.summary, key=str(i))
        self.query_one("#dg-status", Static).update(
            Text(f"Diagnosis from {report.meta.generated_at}: ", style="dim") + badge(report.status)
        )

    @on(DataTable.RowHighlighted, "#dg-checks")
    def _row(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is None or event.row_key.value is None:
            return
        c = self._checks[int(event.row_key.value)]
        text = Text()
        text.append(f"{c.name} ", style="bold")
        text.append_text(badge(c.status))
        text.append(f"   [{c.category} · {c.evidence.value}]\n", style="dim")
        text.append(c.summary + "\n")
        if c.detail:
            text.append(c.detail + "\n", style="dim")
        if c.remediation:
            text.append("Fix: ", style="bold yellow")
            text.append(c.remediation + "\n")
        if c.source:
            text.append(f"Source: {c.source}\n", style="dim italic")
        self.query_one("#dg-detail", Static).update(text)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "dg-run":
            self.start()
        elif bid in ("dg-json", "dg-md", "dg-html"):
            self.action_export({"dg-json": "json", "dg-md": "markdown", "dg-html": "html"}[bid])

    def action_export(self, fmt: str) -> None:
        report = self.state.diagnosis
        if report is None:
            self.app.notify("Run a diagnosis first.", severity="warning")
            return
        f = OutputFormat(fmt)
        d = self.state.history.reports_dir
        try:
            d.mkdir(parents=True, exist_ok=True)
            stamp = report.meta.generated_at.replace(":", "").replace("+", "_")
            suffix = {"json": ".json", "markdown": ".md", "html": ".html"}[fmt]
            path = d / f"diagnosis-{stamp}{suffix}"
            path.write_text(render_text(report, f))
        except OSError as exc:
            self.report_error("Export failed", exc)
            return
        self.app.notify(str(path), title=f"Exported {fmt}")
        reports = self.rapp.pane("reports")
        if hasattr(reports, "refresh_list"):
            reports.refresh_list()
