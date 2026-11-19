"""Typer application: the `kernelvet` command."""

from __future__ import annotations

import typer

from kernelvet import __version__
from kernelvet.cli.commands import diagnose as diag
from kernelvet.cli.commands import migrate, misc
from kernelvet.models.report import DISCLAIMER, TRADEMARKS

EPILOG = (
    "Exit codes: 0 healthy/passed · 1 failure found · 2 usage error · 3 review required "
    "(warnings or unverified stages) · 4 required tool unavailable.\n\n" + DISCLAIMER + "\n\n" + TRADEMARKS
)

app = typer.Typer(
    name="kernelvet",
    help=(
        "KernelVet — Diagnose. Migrate. Verify.\n\n"
        "Unofficial open-source diagnostics and CUDA-to-HIP migration toolkit for AMD GPUs."
    ),
    epilog=EPILOG,
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _version(value: bool) -> None:
    if value:
        print(f"kernelvet {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version, is_eager=True, help="Show the version and exit"
    ),
) -> None:
    """KernelVet command-line interface."""


@app.command()
def tui() -> None:
    """Launch the interactive terminal UI."""
    from kernelvet.tui.app import KernelVetApp

    KernelVetApp().run()


app.command("diagnose")(diag.diagnose)
app.command("doctor")(diag.doctor)
app.command("compatibility")(diag.compatibility)
app.command("benchmark")(misc.benchmark)
app.command("schema")(misc.schema)
app.command("history")(misc.history)
app.command("export")(misc.export)
app.add_typer(migrate.app, name="migrate")
app.add_typer(diag.gpu_app, name="gpu")
app.add_typer(diag.system_app, name="system")
app.add_typer(misc.config_app, name="config")
