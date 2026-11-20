"""Context-sensitive help modal."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Markdown

GLOBAL_KEYS = """
---

### Global keys

| Key | Action |
|---|---|
| `1`–`8` | Jump to a section |
| `tab` / `shift+tab` | Move focus |
| `↑` `↓` | Move in tables and the sidebar |
| `?` / `f1` | This help |
| `ctrl+p` | Command palette |
| `q` / `ctrl+c` | Quit (running background work is cancelled) |

KernelVet is an independent open-source community project. It is not affiliated with,
sponsored by or endorsed by AMD, and it is not an AMD product.
"""


class HelpScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape,q,question_mark,f1", "dismiss", "Close")]
    DEFAULT_CSS = """
    HelpScreen { align: center middle; }
    HelpScreen > VerticalScroll { width: 90; max-width: 95%; height: 80%; border: thick $accent;
                                  background: $surface; padding: 1 2; }
    """

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Markdown(self.text + GLOBAL_KEYS)

    def action_dismiss(self) -> None:  # type: ignore[override]
        self.dismiss(None)
