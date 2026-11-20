"""Base class for the content panes behind the sidebar navigation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from textual.containers import VerticalScroll

if TYPE_CHECKING:
    from kernelvet.tui.app import KernelVetApp
    from kernelvet.tui.state import SessionState


class Pane(VerticalScroll):
    """One navigation section. ``HELP`` is shown by the context-sensitive help (?)."""

    TITLE = ""
    HELP = ""

    @property
    def rapp(self) -> KernelVetApp:
        from kernelvet.tui.app import KernelVetApp

        return cast(KernelVetApp, self.app) if isinstance(self.app, KernelVetApp) else self.app  # type: ignore[return-value]

    @property
    def state(self) -> SessionState:
        return self.rapp.state

    def on_activated(self) -> None:
        """Called when the user navigates to this pane."""

    def report_error(self, title: str, exc: BaseException | str) -> None:
        self.app.notify(f"{exc}", title=title, severity="error", timeout=8)
