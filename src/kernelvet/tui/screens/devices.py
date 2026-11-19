"""GPU Devices and Compatibility panes."""

from __future__ import annotations

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Button, DataTable, Static

from kernelvet.compatibility.engine import evaluate
from kernelvet.compatibility.matrix import load_matrix
from kernelvet.diagnose.engine import collect_inventory
from kernelvet.reporting.common import fmt_bytes, gpu_label
from kernelvet.tui.screens.base import Pane
from kernelvet.tui.widgets.common import KeyValue, Panel, badge, reset_table


class GpuPane(Pane):
    TITLE = "GPU Devices"
    BINDINGS = [Binding("ctrl+r", "refresh", "Rescan")]
    HELP = """\
## GPU Devices

Every GPU reported by `amd-smi static --json`, `rocminfo` and `/sys/class/drm`,
merged by PCI address. **Detected by** lists the tools that saw the device.

**Accessible** tells whether this user can open `/dev/kfd` (needed to run HIP code;
fix with `sudo usermod -aG render,video $USER`).

This view is static detection. To prove the GPU executes code, run **Diagnose**.

Keys: **ctrl+r** rescan.
"""

    def compose(self) -> ComposeResult:
        with Horizontal(classes="buttons"):
            yield Button("Rescan devices", id="gpu-scan", variant="primary")
        with Panel(title="Devices"):
            yield DataTable(id="gpu-table", cursor_type="row", zebra_stripes=True)
        with Panel(title="Device detail"):
            yield KeyValue(Text("Select a device.", style="dim"), id="gpu-detail")

    def on_mount(self) -> None:
        self.render_state()

    def render_state(self) -> None:
        t = self.query_one("#gpu-table", DataTable)
        reset_table(t, "#", "Device", "Vendor", "Target", "CUs", "VRAM", "PCI", "Accessible", "Detected by")
        for g in self.state.gpus:
            acc = "n/a" if g.accessible is None else ("yes" if g.accessible else "no")
            t.add_row(
                str(g.index),
                gpu_label(g),
                "AMD" if g.vendor_is_amd else (g.vendor or "?"),
                g.gfx_target or "?",
                str(g.compute_units or "?"),
                fmt_bytes(g.vram_bytes),
                g.pci_bus or "",
                acc,
                ", ".join(g.sources),
                key=str(g.index),
            )
        if not self.state.gpus:
            t.add_row("—", Text("No GPU detected", style="yellow"), "", "", "", "", "", "", "")

    def action_refresh(self) -> None:
        self.scan()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "gpu-scan":
            self.scan()

    @work(thread=True, exclusive=True, group="inventory", exit_on_error=False)
    def scan(self) -> None:
        try:
            host, gpus, sw = collect_inventory(self.state.ctx)
        except Exception as exc:
            self.app.call_from_thread(self.report_error, "Scan failed", exc)
            return

        def apply() -> None:
            self.state.host, self.state.gpus, self.state.software = host, gpus, sw
            self.render_state()
            self.app.notify(
                f"{len([g for g in gpus if g.vendor_is_amd])} AMD GPU(s) detected", title="GPU scan"
            )

        self.app.call_from_thread(apply)

    @on(DataTable.RowHighlighted, "#gpu-table")
    def _row(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is None or event.row_key.value is None:
            return
        g = next((x for x in self.state.gpus if str(x.index) == event.row_key.value), None)
        if g is None:
            return
        self.query_one("#gpu-detail", KeyValue).set_rows(
            [
                ("Name", gpu_label(g)),
                ("Vendor", g.vendor or "?"),
                ("gfx target", g.gfx_target or "?"),
                ("Compute units", str(g.compute_units or "?")),
                ("VRAM", fmt_bytes(g.vram_bytes)),
                ("PCI bus", g.pci_bus or "?"),
                ("Device ID", g.device_id or "?"),
                ("UUID", g.uuid or "?"),
                ("HSA node", str(g.node_id) if g.node_id is not None else "?"),
                ("Detected by", ", ".join(g.sources)),
                ("Evidence", g.evidence.value),
            ]
        )


class CompatPane(Pane):
    TITLE = "Compatibility"
    BINDINGS = [Binding("r", "run", "Re-evaluate")]
    HELP = """\
## Compatibility

Validates this host against a snapshot of AMD's published ROCm documentation
(system requirements + compatibility matrix). The snapshot's ROCm version, retrieval
date and source URLs are shown below every result.

* **PASS** – the detected value is explicitly listed.
* **WARNING** – not listed, ambiguous, or the snapshot doesn't cover your ROCm release.
* **FAIL** – a known incompatibility (e.g. an unsupported GPU, macOS).
* **SKIPPED** – the input could not be detected.

KernelVet is not AMD and is not a certification authority. Always confirm with the
live AMD documentation. Use a custom matrix via Settings → compatibility_matrix.

Keys: **r** re-evaluate.
"""

    def compose(self) -> ComposeResult:
        with Horizontal(classes="buttons"):
            yield Button("Re-evaluate", id="cp-run", variant="primary")
        yield Static("", id="cp-status")
        with Panel(title="Results"):
            yield DataTable(id="cp-table", cursor_type="row", zebra_stripes=True)
        with Panel(title="Detail"):
            yield Static(Text("Select a result.", style="dim"), id="cp-detail")
        with Panel(title="Reference data"):
            yield Static("", id="cp-sources")

    def on_mount(self) -> None:
        self.render_state()

    def render_state(self) -> None:
        c = self.state.compatibility
        t = self.query_one("#cp-table", DataTable)
        reset_table(t, "Check", "Status", "Result")
        if c is None:
            self.query_one("#cp-status", Static).update(Text("Waiting for inventory…", style="dim"))
            return
        for i, chk in enumerate(c.checks):
            t.add_row(chk.name, badge(chk.status), chk.summary, key=str(i))
        self.query_one("#cp-status", Static).update(Text("Status: ") + badge(c.status))
        m = load_matrix(self.state.settings.compatibility_matrix)
        src = Text()
        src.append(
            f"AMD ROCm {m.rocm_version} documentation snapshot, retrieved {m.retrieved}\n", style="bold"
        )
        for s in m.sources.values():
            src.append(f"• {s.title}: {s.url}\n")
        for n in m.notes:
            src.append(f"Note: {n}\n", style="dim")
        self.query_one("#cp-sources", Static).update(src)

    def action_run(self) -> None:
        self.evaluate()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cp-run":
            self.evaluate()

    @work(thread=True, exclusive=True, group="compat", exit_on_error=False)
    def evaluate(self) -> None:
        st = self.state
        try:
            host, gpus, sw = collect_inventory(
                st.ctx, torch=st.settings.torch_checks, python=st.settings.python_executable
            )
            report = evaluate(host, gpus, sw, load_matrix(st.settings.compatibility_matrix))
        except Exception as exc:
            self.app.call_from_thread(self.report_error, "Compatibility failed", exc)
            return

        def apply() -> None:
            st.host, st.gpus, st.software, st.compatibility = host, gpus, sw, report
            self.render_state()

        self.app.call_from_thread(apply)

    @on(DataTable.RowHighlighted, "#cp-table")
    def _row(self, event: DataTable.RowHighlighted) -> None:
        c = self.state.compatibility
        if c is None or event.row_key is None or event.row_key.value is None:
            return
        chk = c.checks[int(event.row_key.value)]
        text = Text(f"{chk.name}\n", style="bold")
        text.append(chk.summary + "\n")
        if chk.detail:
            text.append(chk.detail + "\n", style="dim")
        if chk.remediation:
            text.append("Fix: " + chk.remediation + "\n", style="yellow")
        if chk.source:
            text.append("Source: " + chk.source, style="dim italic")
        self.query_one("#cp-detail", Static).update(text)
