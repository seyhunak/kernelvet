"""Benchmarks, Reports and Settings panes."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Input,
    Label,
    Markdown,
    ProgressBar,
    Select,
    Static,
)

from kernelvet.benchmark.engine import ITERATIONS_MAX, ITERATIONS_MIN, BenchmarkOptions, run_benchmarks
from kernelvet.config import Settings, config_path, save_settings
from kernelvet.history import HistoryError
from kernelvet.models.report import BenchmarkReport
from kernelvet.reporting import OutputFormat, render_text
from kernelvet.reporting.loader import REPORT_KINDS, ReportFileError, SavedReport, load_report_file
from kernelvet.tui.screens.base import Pane
from kernelvet.tui.widgets.common import Panel, badge, reset_table
from kernelvet.tui.workers import ui_progress


class BenchmarksPane(Pane):
    TITLE = "Benchmarks"
    BINDINGS = [Binding("r", "run", "Run benchmarks")]
    HELP = """\
## Benchmarks

Real measurements on this host only:

* **HIP** – kernel launch latency, pinned host↔device bandwidth, allocation latency,
  vector-add bandwidth and a tiled SGEMM (GFLOP/s). Compiled with `hipcc` at run time.
  Optionally wrapped in `rocprofv3` for kernel traces.
* **PyTorch** – FP32/FP16 matmul TFLOP/s, element-wise bandwidth, tensor copy bandwidth,
  and an optional synthetic MLP inference latency/throughput.

Anything that cannot run is SKIPPED with a reason and carries no numbers.
KernelVet never compares AMD and NVIDIA performance.

Keys: **r** run.
"""

    def compose(self) -> ComposeResult:
        with Horizontal(classes="buttons"):
            yield Button("Run benchmarks", id="bm-run", variant="primary")
            yield Checkbox("HIP", value=True, id="bm-hip")
            yield Checkbox("PyTorch", value=True, id="bm-torch")
            yield Checkbox("Inference", value=False, id="bm-inference")
            yield Checkbox("rocprofv3", value=False, id="bm-profile")
            with Horizontal(classes="labelled"):  # keep the label next to its field when bars wrap
                yield Label("iterations")
                yield Input("20", id="bm-iters", type="integer")
        yield ProgressBar(total=100, show_eta=False, id="bm-progress")
        yield Static(Text("Press r to run benchmarks.", style="dim"), id="bm-status")
        with Panel(title="Results"):
            yield DataTable(id="bm-table", zebra_stripes=True)
        yield Static("", id="bm-note", classes="note")

    def on_mount(self) -> None:
        if self.state.benchmarks:
            self.show(self.state.benchmarks)

    def action_run(self) -> None:
        self.start()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "bm-run":
            self.start()

    def start(self) -> None:
        raw = self.query_one("#bm-iters", Input).value.strip()
        try:
            iters = int(raw)
        except ValueError:
            iters = 0
        if not ITERATIONS_MIN <= iters <= ITERATIONS_MAX:  # same limits as `kernelvet benchmark --iterations`
            self.app.notify(
                f"iterations must be a whole number from {ITERATIONS_MIN} to {ITERATIONS_MAX} (got {raw or 'nothing'})",
                title="Benchmarks",
                severity="warning",
            )
            return
        opts = BenchmarkOptions(
            hip=self.query_one("#bm-hip", Checkbox).value,
            torch=self.query_one("#bm-torch", Checkbox).value,
            inference=self.query_one("#bm-inference", Checkbox).value,
            profile=self.query_one("#bm-profile", Checkbox).value,
            iterations=iters,
            python=self.state.settings.python_executable,
        )
        if problem := opts.problem():
            self.app.notify(problem, title="Benchmarks", severity="warning")
            return
        self.query_one("#bm-run", Button).disabled = True
        self._worker(opts)

    @work(thread=True, exclusive=True, group="benchmark", exit_on_error=False)
    def _worker(self, opts: BenchmarkOptions) -> None:
        try:
            report = run_benchmarks(self.state.ctx, opts, ui_progress(self.app, self._progress))
        except Exception as exc:
            self.app.call_from_thread(self._failed, exc)
            return
        self.app.call_from_thread(self._done, report)

    def _progress(self, message: str, fraction: float) -> None:
        self.query_one("#bm-progress", ProgressBar).update(progress=fraction * 100)
        self.query_one("#bm-status", Static).update(Text(message, style="cyan"))

    def _failed(self, exc: BaseException) -> None:
        self.query_one("#bm-run", Button).disabled = False
        self.report_error("Benchmark failed", exc)

    def _done(self, report: BenchmarkReport) -> None:
        self.state.benchmarks = report
        self.query_one("#bm-run", Button).disabled = False
        if self.state.settings.history:
            try:
                self.state.history.record_run(
                    "benchmark",
                    "/".join(r.status for r in report.results),
                    ", ".join(f"{r.id}: {r.status}" for r in report.results),
                    report,
                )
            except HistoryError as exc:
                self.report_error("History not saved", exc)
        self.show(report)

    def show(self, report: BenchmarkReport) -> None:
        t = self.query_one("#bm-table", DataTable)
        reset_table(t, "Benchmark", "Backend", "Status", "Measurement", "Value", "Unit", "Iter.")
        for b in report.results:
            if not b.measurements:
                t.add_row(b.name, b.backend, badge(b.status), b.reason or "", "", "", "")
            for i, m in enumerate(b.measurements):
                t.add_row(
                    b.name if i == 0 else "",
                    b.backend if i == 0 else "",
                    badge(b.status) if i == 0 else "",
                    m.name,
                    f"{m.value:.4g}",
                    m.unit,
                    str(m.iterations),
                )
        profilers = [b.profiler for b in report.results if b.profiler]
        self.query_one("#bm-status", Static).update(
            Text(f"Completed {report.meta.generated_at}", style="dim")
        )
        self.query_one("#bm-note", Static).update(
            Text(
                report.note + ("\nProfiler: " + "; ".join(profilers) if profilers else ""), style="dim italic"
            )
        )


class ReportsPane(Pane):
    TITLE = "Reports"
    BINDINGS = [Binding("ctrl+r", "refresh", "Refresh")]
    HELP = """\
## Reports

Saved reports live in the KernelVet data directory (`$XDG_DATA_HOME/kernelvet/reports`).
Diagnoses and benchmarks are saved automatically when history is enabled; use the export
buttons on the Diagnose and Migration screens for Markdown, HTML and SARIF.

Select a JSON report to preview it, then convert it to Markdown or HTML next to the original.

Keys: **ctrl+r** refresh the list.
"""

    def compose(self) -> ComposeResult:
        with Horizontal(classes="buttons"):
            yield Button("Refresh", id="rp-refresh")
            yield Button("Save as Markdown", id="rp-md", disabled=True)
            yield Button("Save as HTML", id="rp-html", disabled=True)
        yield Static("", id="rp-dir", classes="note")
        with Panel(title="Saved reports"):
            yield DataTable(id="rp-table", cursor_type="row", zebra_stripes=True)
        with Panel(title="Preview"):
            yield Markdown("Select a report.", id="rp-preview")

    def on_mount(self) -> None:
        self._selected: Path | None = None
        self.refresh_list()

    def on_activated(self) -> None:
        self.refresh_list()

    def action_refresh(self) -> None:
        self.refresh_list()

    def refresh_list(self) -> None:
        d = self.state.history.reports_dir
        self.query_one("#rp-dir", Static).update(Text(f"Directory: {d}", style="dim"))
        t = self.query_one("#rp-table", DataTable)
        reset_table(t, "File", "Kind", "Size")
        if not d.exists():
            return
        files = [p for p in sorted(d.iterdir(), reverse=True) if p.is_file()]
        for p in files:
            kind = next((k for k in REPORT_KINDS if k in p.name), "?")
            t.add_row(p.name, kind, f"{p.stat().st_size // 1024 or 1} KiB", key=str(p))
        # keep the cursor on the report the user picked: a refresh (e.g. after "Save as Markdown")
        # must not move it, or the next button would act on a different file
        if self._selected in files:
            t.move_cursor(row=files.index(self._selected))

    def _load(self, p: Path) -> SavedReport | None:
        try:
            return load_report_file(p)
        except ReportFileError:
            return None

    @on(DataTable.RowHighlighted, "#rp-table")
    def _row(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is None or event.row_key.value is None:
            return
        p = Path(event.row_key.value)
        self._selected = p
        md = self.query_one("#rp-preview", Markdown)
        is_json = p.suffix == ".json"
        self.query_one("#rp-md", Button).disabled = not is_json
        self.query_one("#rp-html", Button).disabled = not is_json
        if p.suffix == ".md":
            md.update(p.read_text(errors="replace")[:200_000])
        elif is_json:
            r = self._load(p)
            md.update(render_text(r, OutputFormat.MARKDOWN) if r else "Not a KernelVet report.")
        else:
            md.update(f"`{p.name}` — open it with an external viewer.")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "rp-refresh":
            self.refresh_list()
        elif event.button.id in ("rp-md", "rp-html") and self._selected:
            r = self._load(self._selected)
            if r is None:
                self.report_error("Convert", "not a KernelVet JSON report")
                return
            fmt = OutputFormat.MARKDOWN if event.button.id == "rp-md" else OutputFormat.HTML
            out = self._selected.with_suffix(".md" if fmt is OutputFormat.MARKDOWN else ".html")
            try:
                out.write_text(render_text(r, fmt))
            except OSError as exc:
                self.report_error("Convert", exc)
                return
            self.app.notify(str(out), title="Saved")
            self.refresh_list()


class SettingsPane(Pane):
    TITLE = "Settings"
    BINDINGS = [Binding("ctrl+s", "save", "Save"), Binding("ctrl+r", "reload", "Reload")]
    HELP = """\
## Settings

Edits `~/.config/kernelvet/config.toml` (shown below). Values from `KERNELVET_*`
environment variables override the file for the current process.

* **runtime_checks / torch_checks** – whether Diagnose executes code on the GPU.
* **hipify_preference** – `auto` prefers hipify-clang when CUDA headers are present.
* **isolation** – `docker` runs migrated builds/tests in a locked-down container.
* **offload_arch** – comma-separated gfx targets for builds (default: detected GPUs).
* **compatibility_matrix** – path to a custom matrix JSON.

Press **ctrl+s** (or **Save**) to write the file. Invalid values are rejected and nothing
is written. **ctrl+r** (or **Reload**) discards unsaved edits. An empty field means the default.
"""

    BOOL = ("runtime_checks", "torch_checks", "history")
    CHOICE = {"hipify_preference": ("auto", "hipify-clang", "hipify-perl"), "isolation": ("none", "docker")}

    def compose(self) -> ComposeResult:
        # the buttons sit above the form: twelve fields are taller than most terminals
        with Horizontal(classes="buttons"):
            yield Button("Save", id="st-save", variant="primary")
            yield Button("Reload", id="st-reload")
        yield Static("", id="st-path", classes="note")
        with Panel(title="Configuration"):
            for name, field in Settings.model_fields.items():
                with Horizontal(classes="field"):
                    yield Label(f"{name:<22}", classes="label")
                    if name in self.BOOL:
                        yield Checkbox("", id=f"st-{name}")
                    elif name in self.CHOICE:
                        yield Select(
                            [(v, v) for v in self.CHOICE[name]],
                            allow_blank=False,
                            id=f"st-{name}",
                            value=self.CHOICE[name][0],
                        )
                    else:
                        yield Input(id=f"st-{name}", placeholder=field.description or "")

    def on_mount(self) -> None:
        self.load_values()

    def load_values(self) -> None:
        s = self.state.settings
        self.query_one("#st-path", Static).update(Text(f"Config file: {config_path()}", style="dim"))
        for name in Settings.model_fields:
            v = getattr(s, name)
            w = self.query_one(f"#st-{name}")
            if isinstance(w, Checkbox):
                w.value = bool(v)
            elif isinstance(w, Select):
                w.value = v
            elif isinstance(w, Input):
                w.value = "" if v is None else str(v)

    def collect(self) -> Settings:
        data: dict[str, object] = {}
        for name in Settings.model_fields:
            w = self.query_one(f"#st-{name}")
            if isinstance(w, Checkbox | Select):
                data[name] = w.value
            elif isinstance(w, Input):
                data[name] = w.value.strip() or None
        data = {k: v for k, v in data.items() if v is not None or Settings.model_fields[k].default is None}
        return Settings(**data)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "st-reload":
            self.action_reload()
        elif event.button.id == "st-save":
            self.action_save()

    def action_reload(self) -> None:
        self.load_values()
        self.app.notify("Unsaved edits discarded", title="Settings reloaded")

    def action_save(self) -> None:
        try:
            new = self.collect()
            path = save_settings(new)
        except ValidationError as exc:
            err = exc.errors()[0]
            self.report_error("Invalid setting", f"{err['loc'][0]}: {err['msg']}")
            return
        except OSError as exc:
            self.report_error("Could not save", exc)
            return
        self.state.settings = new
        self.app.notify(str(path), title="Settings saved")
