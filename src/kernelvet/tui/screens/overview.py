"""Overview: devices, ROCm/HIP availability, health summary and history."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, DataTable, Static

from kernelvet.compatibility.engine import evaluate
from kernelvet.compatibility.matrix import load_matrix
from kernelvet.diagnose.engine import collect_inventory
from kernelvet.history import HistoryError, HistoryStore
from kernelvet.reporting.common import fmt_bytes, gpu_label
from kernelvet.tui.screens.base import Pane
from kernelvet.tui.widgets.common import KeyValue, Panel, badge, reset_table

R = TypeVar("R")


class OverviewPane(Pane):
    _history_warned = False
    TITLE = "Overview"
    HELP = """\
## Overview

A quick picture of this machine from **static detection** (nothing runs on the GPU here).

* **Environment** – detected AMD GPUs, ROCm and HIP compiler availability.
* **Health** – the result of the last full diagnosis (press **2** then **r** to run one).
* **Compatibility** – this host against AMD's published ROCm compatibility data.
* **History** – recent diagnostic runs and migration projects stored locally.

Keys: **ctrl+r** refresh inventory.
"""

    def compose(self) -> ComposeResult:
        with Horizontal(classes="row"):
            with Panel(title="Environment", classes="half"):
                yield KeyValue("Collecting inventory…", id="ov-env")
            with Panel(title="Health", classes="half"):
                yield KeyValue("No diagnosis run in this session yet.", id="ov-health")
        with Panel(title="GPU devices"):
            yield DataTable(id="ov-gpus", cursor_type="row", zebra_stripes=True)
        with Panel(title="Recent diagnostic runs"):
            yield DataTable(id="ov-runs", cursor_type="row")
        with Panel(title="Recent migration projects"):
            yield DataTable(id="ov-projects", cursor_type="row")
        with Horizontal(classes="buttons"):
            yield Button("Refresh inventory", id="ov-refresh", variant="primary")
            yield Button("Run full diagnosis →", id="ov-diagnose")
        yield Static("", id="ov-note", classes="note")

    def on_mount(self) -> None:
        self.refresh_history()
        self.load_inventory()

    def on_activated(self) -> None:
        self.refresh_history()
        self.render_health()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ov-refresh":
            self.load_inventory()
        elif event.button.id == "ov-diagnose":
            self.rapp.show_pane("diagnose")
            self.rapp.pane("diagnose").start()  # type: ignore[attr-defined]

    @work(thread=True, exclusive=True, group="inventory", exit_on_error=False)
    def load_inventory(self) -> None:
        self.app.call_from_thread(self.query_one("#ov-note", Static).update, "Collecting inventory…")
        try:
            host, gpus, sw = collect_inventory(self.state.ctx, torch=False)
            compat = evaluate(host, gpus, sw, load_matrix(self.state.settings.compatibility_matrix))
        except Exception as exc:  # surface any unexpected failure in the UI
            self.app.call_from_thread(self.report_error, "Inventory failed", exc)
            return
        self.app.call_from_thread(self._apply_inventory, host, gpus, sw, compat)

    def _apply_inventory(self, host, gpus, sw, compat) -> None:
        st = self.state
        st.host, st.gpus, st.software = host, gpus, sw
        if st.compatibility is None:
            st.compatibility = compat
        amd = st.amd_gpus
        env = self.query_one("#ov-env", KeyValue)
        env.set_rows(
            [
                ("Host", f"{host.os_pretty or host.os_name} · {host.architecture}"),
                ("Kernel", host.kernel or "unknown"),
                ("AMD GPUs", str(len(amd)) if amd else Text("none detected", style="yellow")),
                ("Architectures", ", ".join(sorted({g.gfx_target or "?" for g in amd})) or "—"),
                ("VRAM", ", ".join(fmt_bytes(g.vram_bytes) for g in amd) or "—"),
                ("ROCm", sw.rocm_version or Text("not detected", style="yellow")),
                (
                    "HIP compiler",
                    Text("available", style="green")
                    if sw.has("hipcc")
                    else Text("not installed", style="yellow"),
                ),
                (
                    "HIPIFY",
                    ", ".join(t for t in ("hipify-clang", "hipify-perl") if sw.has(t))
                    or Text("not installed", style="yellow"),
                ),
                ("Compatibility", badge(compat.status)),
            ]
        )
        t = self.query_one("#ov-gpus", DataTable)
        reset_table(t, "#", "Device", "Target", "CUs", "VRAM", "Detected by")
        for g in gpus:
            t.add_row(
                str(g.index),
                gpu_label(g),
                g.gfx_target or "?",
                str(g.compute_units or "?"),
                fmt_bytes(g.vram_bytes),
                ", ".join(g.sources),
            )
        if not gpus:
            t.add_row("—", Text("No GPU detected", style="yellow"), "", "", "", "")
        self.query_one("#ov-note", Static).update(
            Text("Static detection only. Run a diagnosis to execute code on the GPU.", style="dim")
        )
        self.render_health()
        for pane_id in ("gpus", "compat"):
            pane = self.rapp.pane(pane_id)
            if hasattr(pane, "render_state"):
                pane.render_state()

    def render_health(self) -> None:
        r = self.state.diagnosis
        kv = self.query_one("#ov-health", KeyValue)
        if r is None:
            runs = self._read_history(lambda h: h.recent_runs(1, kind="diagnosis"))
            if runs:
                kv.set_rows(
                    [
                        ("Last diagnosis", runs[0].created_at),
                        ("Status", badge(runs[0].status)),
                        ("Checks", runs[0].summary),
                        ("", Text("from history; run again for current state", style="dim")),
                    ]
                )
            else:
                kv.update(Text("No diagnosis yet. Press 2 then r to run one.", style="dim"))
            return
        checks = r.all_checks()
        verified = r.hardware_verified_checks
        kv.set_rows(
            [
                ("Overall", badge(r.status)),
                (
                    "PASS / WARN / FAIL / SKIP",
                    " / ".join(
                        str(sum(1 for c in checks if c.status == s))
                        for s in ("PASS", "WARNING", "FAIL", "SKIPPED")
                    ),
                ),
                (
                    "Verified on AMD hardware",
                    Text(f"{len(verified)} check(s) ran on GPU and passed", style="green")
                    if verified
                    else Text("none; results are detected or inferred only", style="yellow"),
                ),
                ("Generated", r.meta.generated_at),
            ]
        )

    def _read_history(self, read: Callable[[HistoryStore], list[R]]) -> list[R]:
        """History is optional: an unusable database shows one notice and empty tables, never a crash."""
        try:
            return read(self.state.history)
        except HistoryError as exc:
            if not self._history_warned:
                self._history_warned = True
                self.report_error("History unavailable", exc)
            return []

    def refresh_history(self) -> None:
        runs = self.query_one("#ov-runs", DataTable)
        reset_table(runs, "When", "Kind", "Status", "Summary")
        for r in self._read_history(lambda h: h.recent_runs(8)):
            runs.add_row(
                r.created_at,
                r.kind,
                badge(r.status) if r.status in ("HEALTHY", "REVIEW REQUIRED", "UNHEALTHY") else r.status,
                r.summary,
            )
        projects = self.query_one("#ov-projects", DataTable)
        reset_table(projects, "Updated", "Project", "Stage", "Summary")
        for p in self._read_history(lambda h: h.recent_projects(8)):
            projects.add_row(p.updated_at, p.path, p.stage, p.summary)
