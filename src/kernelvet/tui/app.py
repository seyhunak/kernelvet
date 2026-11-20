"""KernelVet Textual application."""

from __future__ import annotations

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import ContentSwitcher, Footer, Header, Label, ListItem, ListView

from kernelvet import __version__
from kernelvet.config import ConfigError, Settings, load_settings
from kernelvet.history import HistoryStore
from kernelvet.tui.screens.base import Pane
from kernelvet.tui.screens.devices import CompatPane, GpuPane
from kernelvet.tui.screens.diagnose import DiagnosePane
from kernelvet.tui.screens.help import HelpScreen
from kernelvet.tui.screens.migration import MigrationPane
from kernelvet.tui.screens.overview import OverviewPane
from kernelvet.tui.screens.tools import BenchmarksPane, ReportsPane, SettingsPane
from kernelvet.tui.state import SessionState
from kernelvet.utils.proc import HostContext

SECTIONS: list[tuple[str, str, type[Pane]]] = [
    ("overview", "Overview", OverviewPane),
    ("diagnose", "Diagnose", DiagnosePane),
    ("gpus", "GPU Devices", GpuPane),
    ("compat", "Compatibility", CompatPane),
    ("migration", "CUDA → HIP Migration", MigrationPane),
    ("benchmarks", "Benchmarks", BenchmarksPane),
    ("reports", "Reports", ReportsPane),
    ("settings", "Settings", SettingsPane),
]

CSS = """
Screen { background: $background; }
#sidebar { width: 28; dock: left; border-right: vkey $panel-lighten-2; background: $panel; }
#sidebar > ListItem { padding: 0 1; height: 1; }
#sidebar > ListItem.-highlight { background: $accent 40%; }
#brand { padding: 1 1 0 1; color: $accent; text-style: bold; }
#tagline { padding: 0 1 1 1; color: $text-muted; }
#content { height: 1fr; }
Pane { padding: 0 1; }
.row { height: auto; }
.half { width: 1fr; }
/* Below 106 columns there is no room for two key/value panels side by side
   (values like "not detected" were cut to "not dete…"), so stack them. */
Screen.-cols-2 .row { layout: vertical; }
Screen.-cols-2 .row > .half { width: 100%; }
.field { height: auto; }
.field > .label { width: 24; padding: 1 0 0 0; color: $text-muted; }
.field > Input { width: 1fr; }
.field > Select { width: 40; }
.small { width: 10; }
/* Button bars are grids, so they wrap onto more rows instead of running off the
   right edge. The column count follows the terminal width (see BUTTON_COLUMNS). */
.buttons { layout: grid; grid-size: 2; grid-gutter: 0 1; grid-rows: auto; height: auto; margin: 0 0 1 0; }
.buttons > * { width: 100%; min-width: 0; margin: 0; }
.buttons > .labelled { height: auto; }
.buttons > .labelled > Label { width: auto; padding: 1 1 0 1; color: $text-muted; }
.buttons > .labelled > Input { width: 1fr; }
Screen.-cols-3 .buttons { grid-size: 3; }
Screen.-cols-4 .buttons { grid-size: 4; }
Screen.-cols-5 .buttons { grid-size: 5; }
Screen.-cols-6 .buttons { grid-size: 6; }
.note { color: $text-muted; margin: 0 0 1 0; }
DataTable { height: auto; max-height: 22; }
#dg-log, #mg-log { height: 12; }
ProgressBar { margin: 0 0 1 0; }
"""


# Terminal width at which each button-bar column count starts. Content is the terminal
# minus the 28-column sidebar and padding (~32 columns); each cell needs ~24 columns, the
# width of the longest control ("Export Markdown" on a button, "PyTorch checks" plus its checkbox).
BUTTON_COLUMNS = [(0, "-cols-2"), (106, "-cols-3"), (131, "-cols-4"), (156, "-cols-5"), (181, "-cols-6")]


class KernelVetApp(App[None]):
    TITLE = "KernelVet"
    SUB_TITLE = "Diagnose. Migrate. Verify."
    CSS = CSS
    HORIZONTAL_BREAKPOINTS = BUTTON_COLUMNS
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("question_mark,f1", "help", "Help"),
        *[
            Binding(str(i + 1), f"goto('{sid}')", label, show=False)
            for i, (sid, label, _) in enumerate(SECTIONS)
        ],
    ]

    def __init__(
        self,
        *,
        ctx: HostContext | None = None,
        settings: Settings | None = None,
        history: HistoryStore | None = None,
    ) -> None:
        super().__init__()
        self._config_error: str | None = None
        if settings is None:
            try:
                settings = load_settings()
            except ConfigError as exc:
                self._config_error = str(exc)
                settings = Settings()
        self.state = SessionState(
            ctx=ctx or HostContext(timeout=settings.command_timeout),
            settings=settings,
            history=history or HistoryStore(),
        )
        self.theme = "textual-dark"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal():
            with ListView(id="sidebar"):
                for i, (sid, label, _) in enumerate(SECTIONS):
                    yield ListItem(Label(f"{i + 1}  {label}"), id=f"nav-{sid}")
            with ContentSwitcher(initial="overview", id="content"):
                for sid, _label, cls in SECTIONS:
                    yield cls(id=sid)
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = f"Diagnose. Migrate. Verify. · v{__version__}"
        if self._config_error:
            self.notify(
                self._config_error + " (using defaults)", title="Config error", severity="error", timeout=10
            )

    # ------------------------------------------------------------------ navigation

    def pane(self, sid: str) -> Pane:
        return self.query_one(f"#{sid}", Pane)

    @property
    def current_pane(self) -> Pane:
        current = self.query_one("#content", ContentSwitcher).current or "overview"
        return self.pane(current)

    def show_pane(self, sid: str, *, focus: bool = False) -> None:
        """Show a section. With ``focus``, keyboard focus moves into it too.

        Focus also moves when it sits in a section that is being hidden: otherwise the
        footer and the keys would still belong to the section the user left.
        """
        switcher = self.query_one("#content", ContentSwitcher)
        if switcher.current != sid:
            switcher.current = sid
        nav = self.query_one("#sidebar", ListView)
        idx = [s for s, _, _ in SECTIONS].index(sid)
        if nav.index != idx:
            nav.index = idx
        pane = self.pane(sid)
        focused = self.focused
        in_other_pane = focused is not None and any(
            isinstance(node, Pane) and node is not pane for node in focused.ancestors_with_self
        )
        if focus or in_other_pane:
            pane.focus()
        pane.on_activated()

    def action_goto(self, sid: str) -> None:
        # number keys jump to a section and hand it the keyboard, so its own keys
        # (e.g. "r" to run) work straight away; arrowing through the sidebar keeps focus there
        self.show_pane(sid, focus=True)

    @on(ListView.Highlighted, "#sidebar")
    def _nav(self, event: ListView.Highlighted) -> None:
        if event.item is not None and event.item.id:
            self.show_pane(event.item.id.removeprefix("nav-"))

    def action_help(self) -> None:
        pane = self.current_pane
        self.push_screen(HelpScreen(pane.HELP or f"## {pane.TITLE}"))
