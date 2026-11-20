"""Small reusable TUI widgets and helpers."""

from __future__ import annotations

from rich.text import Text
from textual.containers import Vertical
from textual.widgets import DataTable, Static

from kernelvet.models.common import CheckStatus, OverallStatus
from kernelvet.models.migration import FindingCategory, StageStatus

STATUS_COLORS = {
    "PASS": "green",
    "WARNING": "yellow",
    "FAIL": "red",
    "SKIPPED": "grey50",
    "PASSED": "green",
    "FAILED": "red",
    "UNAVAILABLE": "yellow",
    "NOT_RUN": "grey50",
    "HEALTHY": "green",
    "REVIEW REQUIRED": "yellow",
    "UNHEALTHY": "red",
}
ICONS = {
    "PASS": "✔",
    "WARNING": "▲",
    "FAIL": "✖",
    "SKIPPED": "○",
    "PASSED": "✔",
    "FAILED": "✖",
    "UNAVAILABLE": "▲",
    "NOT_RUN": "○",
}
CATEGORY_COLORS = {
    FindingCategory.AUTO_CONVERTIBLE: "green",
    FindingCategory.MANUAL_INTERVENTION: "yellow",
    FindingCategory.UNSUPPORTED: "red",
    FindingCategory.REQUIRES_VERIFICATION: "cyan",
}


def badge(status: CheckStatus | StageStatus | OverallStatus | str) -> Text:
    s = str(status.value if hasattr(status, "value") else status)
    icon = ICONS.get(s, "")
    return Text(f"{icon} {s}".strip(), style=f"bold {STATUS_COLORS.get(s, 'white')}")


def category_text(cat: FindingCategory) -> Text:
    return Text(cat.value, style=CATEGORY_COLORS[cat])


class Panel(Vertical):
    """A bordered section with a title."""

    DEFAULT_CSS = """
    Panel { border: round $panel-lighten-2; padding: 0 1; height: auto; margin-bottom: 1; }
    Panel:focus-within { border: round $accent; }
    """

    def __init__(self, *children, title: str = "", **kw) -> None:
        super().__init__(*children, **kw)
        self.border_title = title


class KeyValue(Static):
    """Two-column key/value block rendered as a Rich table."""

    def set_rows(self, rows: list[tuple[str, str | Text]]) -> None:
        from rich.table import Table

        t = Table.grid(padding=(0, 2))
        t.add_column(style="cyan", no_wrap=True)
        t.add_column()
        for k, v in rows:
            t.add_row(k, v)
        self.update(t)


def reset_table(table: DataTable, *columns: str) -> None:
    table.clear(columns=True)
    table.add_columns(*columns)
