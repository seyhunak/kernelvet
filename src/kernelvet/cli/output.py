"""CLI output plumbing.

* Machine-readable output (JSON/Markdown/HTML/SARIF) goes to stdout, or to a
  file with ``--output``; nothing else is ever written to stdout in that mode.
* Progress, warnings and confirmations always go to stderr.
"""

from __future__ import annotations

import contextlib
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import typer
from pydantic import BaseModel
from rich.console import Console, RenderableType

from kernelvet.config import ConfigError, Settings, load_settings
from kernelvet.models.common import ExitCode
from kernelvet.reporting import OutputFormat, UnsupportedFormat, format_from_path, render_text
from kernelvet.utils.proc import ProgressCallback

out = Console(soft_wrap=False)
err = Console(stderr=True)


def settings_or_exit() -> Settings:
    try:
        return load_settings()
    except ConfigError as exc:
        err.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(ExitCode.USAGE) from exc


def resolve_format(fmt: OutputFormat | None, output: Path | None) -> OutputFormat:
    if fmt is not None:
        return fmt
    if output is not None:
        guessed = format_from_path(str(output))
        if guessed is None:
            err.print(f"[red]cannot infer a format from {output}; pass --format[/red]")
            raise typer.Exit(ExitCode.USAGE)
        return guessed
    return OutputFormat.TERMINAL


def emit(
    report: BaseModel, fmt: OutputFormat, output: Path | None, terminal: Callable[[], RenderableType]
) -> None:
    if fmt is OutputFormat.TERMINAL:
        if output is not None:
            err.print("[red]--output needs a file format (json, markdown, html, sarif)[/red]")
            raise typer.Exit(ExitCode.USAGE)
        out.print(terminal())
        return
    try:
        text = render_text(report, fmt)
    except UnsupportedFormat as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(ExitCode.USAGE) from exc
    if output is None:
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
        sys.stdout.flush()
    else:
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text)
        except OSError as exc:
            err.print(f"[red]could not write {output}: {exc}[/red]")
            raise typer.Exit(ExitCode.FAILURE) from exc
        err.print(f"[green]wrote[/green] {output} ({fmt.value})")


def write_json(data: object) -> None:
    """Plain JSON on stdout: no colour codes, no wrapping, safe to pipe."""
    sys.stdout.write(json.dumps(data, indent=2, default=str) + "\n")
    sys.stdout.flush()


@contextlib.contextmanager
def progress(enabled: bool = True) -> Iterator[ProgressCallback]:
    """A stderr spinner; disabled automatically when stderr is not a terminal."""
    if not enabled or not err.is_terminal:
        yield lambda _m, _f: None
        return
    with err.status("Starting…", spinner="dots") as status:

        def update(message: str, fraction: float) -> None:
            status.update(f"[cyan]{message}[/cyan] [dim]{int(fraction * 100)}%[/dim]")

        yield update


FormatOpt = typer.Option(
    None,
    "--format",
    "-f",
    help="terminal, json, markdown, html (sarif for migration findings)",
    case_sensitive=False,
)
OutputOpt = typer.Option(
    None,
    "--output",
    "-o",
    help="Write the report to a file (format inferred from extension)",
    dir_okay=False,
    writable=True,
)
