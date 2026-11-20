"""`kernelvet config`, `benchmark`, `schema`, `history`."""

from __future__ import annotations

import typing
from pathlib import Path

import typer
from pydantic import BaseModel, ValidationError
from rich.table import Table

from kernelvet.benchmark.engine import ITERATIONS_MAX, ITERATIONS_MIN, BenchmarkOptions, run_benchmarks
from kernelvet.cli.output import (
    FormatOpt,
    OutputOpt,
    emit,
    err,
    out,
    progress,
    resolve_format,
    settings_or_exit,
    write_json,
)
from kernelvet.config import Settings, config_path, data_dir, save_settings
from kernelvet.history import HistoryError, HistoryStore
from kernelvet.models.common import ExitCode
from kernelvet.models.report import (
    BenchmarkReport,
    CompatibilityReport,
    DiagnosticReport,
    MigrationReport,
)
from kernelvet.reporting import OutputFormat
from kernelvet.reporting.loader import ReportFileError, load_report_file
from kernelvet.reporting.terminal import kv_table, render_benchmarks
from kernelvet.utils.proc import HostContext

config_app = typer.Typer(help="Show and edit configuration.", no_args_is_help=True)


@config_app.command("show")
def config_show(json_out: bool = typer.Option(False, "--json", help="Emit JSON")) -> None:
    """Show the effective configuration (defaults + config file + KERNELVET_* env vars)."""
    s = settings_or_exit()
    if json_out:
        write_json({"path": str(config_path()), "exists": config_path().exists(), "settings": s.model_dump()})
        return
    rows = [(k, "" if v is None else str(v)) for k, v in s.model_dump().items()]
    out.print(
        kv_table(
            f"Configuration ({config_path()}{'' if config_path().exists() else ', not created yet'})", rows
        )
    )
    out.print(f"[dim]Data directory: {data_dir()}[/dim]")


@config_app.command("path")
def config_path_cmd() -> None:
    """Print the config file path."""
    print(config_path())


def _config_value(key: str, value: str) -> object:
    """What `config set KEY VALUE` stores.

    An empty VALUE resets the setting to its default. "none"/"null" clear a setting
    that can be unset; everywhere else the text is validated as typed, so
    `isolation none` selects the "none" choice.
    """
    field = Settings.model_fields[key]
    if value == "":
        return field.get_default(call_default_factory=True)
    if value.lower() in ("none", "null") and type(None) in typing.get_args(field.annotation):
        return None
    return value


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Setting name (see `kernelvet config show`)"),
    value: str = typer.Argument(..., help='New value; "" resets the setting to its default'),
) -> None:
    """Set a configuration value and save it to the config file."""
    s = settings_or_exit()
    if key not in Settings.model_fields:
        err.print(f"[red]unknown setting {key!r}[/red]; valid: {', '.join(Settings.model_fields)}")
        raise typer.Exit(ExitCode.USAGE)
    data = s.model_dump()
    data[key] = _config_value(key, value)
    try:
        new = Settings(**data)
    except ValidationError as exc:
        err.print(f"[red]invalid value for {key}:[/red] {exc.errors()[0]['msg']}")
        raise typer.Exit(ExitCode.USAGE) from exc
    path = save_settings(new)
    err.print(f"[green]saved[/green] {key} = {getattr(new, key)!r} to {path}")


def benchmark(
    hip: bool = typer.Option(True, "--hip/--no-hip", help="HIP kernel, memory and SGEMM benchmarks"),
    torch: bool = typer.Option(True, "--torch/--no-torch", help="PyTorch tensor benchmarks"),
    inference: bool = typer.Option(False, "--inference", help="Also run a synthetic MLP inference benchmark"),
    iterations: int = typer.Option(
        20, min=ITERATIONS_MIN, max=ITERATIONS_MAX, help="Timed iterations per measurement"
    ),
    profile: bool = typer.Option(False, "--profile", help="Run the HIP benchmark under rocprofv3"),
    profile_dir: Path | None = typer.Option(None, help="Where rocprofv3 writes its output"),
    python: str | None = typer.Option(None, help="Python interpreter for PyTorch benchmarks"),
    fmt: OutputFormat | None = FormatOpt,
    output: Path | None = OutputOpt,
) -> None:
    """Measure HIP and PyTorch performance on this host (real runs only; never simulated)."""
    s = settings_or_exit()
    fmt = resolve_format(fmt, output)
    opts = BenchmarkOptions(
        hip=hip,
        torch=torch,
        inference=inference,
        iterations=iterations,
        profile=profile,
        profile_dir=str(profile_dir) if profile_dir else None,
        python=python or s.python_executable,
    )
    if problem := opts.problem():
        err.print(f"[red]{problem}[/red]")
        raise typer.Exit(ExitCode.USAGE)
    with progress() as cb:
        report = run_benchmarks(HostContext(timeout=s.command_timeout), opts, cb)
    if s.history:
        try:
            HistoryStore().record_run(
                "benchmark",
                "/".join(r.status for r in report.results) or "none",
                ", ".join(f"{r.id}: {r.status}" for r in report.results),
                report,
            )
        except HistoryError as exc:  # history is best-effort: never lose the results over it
            err.print(f"[yellow]could not record history: {exc}[/yellow]")
    emit(report, fmt, output, lambda: render_benchmarks(report))
    statuses = {r.status for r in report.results}
    if "FAIL" in statuses:
        raise typer.Exit(ExitCode.FAILURE)
    raise typer.Exit(ExitCode.OK if statuses == {"PASS"} else ExitCode.REVIEW)


SCHEMAS: dict[str, type[BaseModel]] = {
    "diagnosis": DiagnosticReport,
    "compatibility": CompatibilityReport,
    "migration": MigrationReport,
    "benchmark": BenchmarkReport,
}


def schema(kind: str = typer.Argument(..., help="diagnosis, compatibility, migration or benchmark")) -> None:
    """Print the JSON Schema of a report type."""
    model = SCHEMAS.get(kind)
    if model is None:
        err.print(f"[red]unknown report type {kind!r}[/red]; choose from {', '.join(SCHEMAS)}")
        raise typer.Exit(ExitCode.USAGE)
    write_json(model.model_json_schema(mode="serialization"))  # describes the JSON we emit


def history(
    limit: int = typer.Option(10, min=1, max=500, help="How many runs and projects to show"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON"),
) -> None:
    """Show recent diagnostic runs and migration projects."""
    store = HistoryStore()
    try:
        runs = store.recent_runs(limit)
        projects = store.recent_projects(limit)
    except HistoryError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(ExitCode.USAGE) from exc
    if json_out:
        write_json({"runs": [r.__dict__ for r in runs], "projects": [p.__dict__ for p in projects]})
        return
    t = Table(title="Recent runs", title_justify="left", title_style="bold")
    for col in ("When", "Kind", "Status", "Summary", "Report"):
        t.add_column(col)
    for r in runs:
        t.add_row(r.created_at, r.kind, r.status, r.summary, r.report_path or "")
    out.print(t)
    p = Table(title="Migration projects", title_justify="left", title_style="bold")
    for col in ("Updated", "Project", "Stage", "Summary"):
        p.add_column(col)
    for pr in projects:
        p.add_row(pr.updated_at, pr.path, pr.stage, pr.summary)
    out.print(p)


def export(
    source: Path = typer.Argument(..., exists=True, dir_okay=False, help="A saved JSON report"),
    fmt: OutputFormat | None = FormatOpt,
    output: Path | None = OutputOpt,
) -> None:
    """Re-render a saved JSON report as Markdown, HTML, SARIF or terminal output."""
    from kernelvet.reporting.terminal import render_compatibility, render_diagnosis, render_migration

    fmt = resolve_format(fmt, output)
    try:
        report = load_report_file(source)
    except ReportFileError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(ExitCode.USAGE) from exc

    def term() -> object:
        if isinstance(report, DiagnosticReport):
            return render_diagnosis(report)
        if isinstance(report, CompatibilityReport):
            return render_compatibility(report)
        if isinstance(report, MigrationReport):
            return render_migration(report)
        return render_benchmarks(report)

    emit(report, fmt, output, term)  # type: ignore[arg-type]
