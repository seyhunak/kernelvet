"""Background work helpers.

Long-running operations (diagnostics, HIPIFY, builds, tests, benchmarks) run in
Textual thread workers so the event loop never blocks. Results and progress
are marshalled back to the UI thread with ``App.call_from_thread``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from textual.app import App

    from kernelvet.utils.proc import ProgressCallback


def ui_progress(app: App, fn: Callable[[str, float], None]) -> ProgressCallback:
    """Wrap a UI-thread progress handler so worker threads can call it safely."""

    def cb(message: str, fraction: float) -> None:
        app.call_from_thread(fn, message, fraction)

    return cb
