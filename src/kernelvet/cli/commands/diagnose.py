"""`kernelvet diagnose`, `doctor`, `compatibility`, `gpu list`, `system info`."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table
from rich.text import Text

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
from kernelvet.compatibility.engine import evaluate
from kernelvet.compatibility.matrix import CompatibilityMatrix, MatrixError, load_matrix
from kernelvet.config import Settings
from kernelvet.diagnose.engine import DiagnoseOptions, collect_inventory, run_diagnosis
from kernelvet.history import HistoryError, HistoryStore
from kernelvet.models.common import CheckStatus, ExitCode
from kernelvet.models.report import DiagnosticReport
from kernelvet.reporting import OutputFormat
from kernelvet.reporting.terminal import (
    OVERALL_STYLE,
    gpu_table,
    host_table,
    render_compatibility,
    render_diagnosis,
    software_table,
    status_text,
)
from kernelvet.utils.proc import HostContext


def _ctx(timeout: float) -> HostContext:
    return HostContext(timeout=timeout)


def _matrix_error(exc: MatrixError) -> typer.Exit:
    err.print(f"[red]compatibility matrix error:[/red] {exc}")
    return typer.Exit(ExitCode.USAGE)


def _diagnose_or_exit(s: Settings, opts: DiagnoseOptions) -> DiagnosticReport:
    try:
        with progress() as cb:
            return run_diagnosis(_ctx(s.command_timeout), opts, cb)
    except MatrixError as exc:
        raise _matrix_error(exc) from exc


def _matrix_or_exit(path: str | None) -> CompatibilityMatrix:
    try:
        return load_matrix(path)
    except MatrixError as exc:
        raise _matrix_error(exc) from exc


def _json_or_terminal(fmt: OutputFormat | None, json_out: bool) -> bool:
    """`gpu list` and `system info` print a table or JSON; `--json` is kept as a shorthand for `-f json`."""
    if fmt in (None, OutputFormat.TERMINAL):
        return json_out
    if fmt is OutputFormat.JSON:
        return True
    err.print(f"[red]{fmt.value} is not supported here; use terminal or json[/red]")
    raise typer.Exit(ExitCode.USAGE)


def _record(report: DiagnosticReport, enabled: bool) -> None:
    if not enabled:
        return
    counts = {s: sum(1 for c in report.all_checks() if c.status == s) for s in CheckStatus}
    summary = ", ".join(f"{n} {s.value}" for s, n in counts.items() if n)
    try:
        HistoryStore().record_run("diagnosis", report.status.value, summary, report)
    except HistoryError as exc:  # history is best-effort: never lose the report over it
        err.print(f"[yellow]could not record history: {exc}[/yellow]")


def diagnose(
    fmt: OutputFormat | None = FormatOpt,
    output: Path | None = OutputOpt,
    runtime: bool = typer.Option(
        True, "--runtime/--no-runtime", help="Compile and run the HIP probe on the GPU"
    ),
    torch: bool = typer.Option(True, "--torch/--no-torch", help="Detect PyTorch and run GPU tensor checks"),
    python: str | None = typer.Option(None, help="Python interpreter to inspect for PyTorch"),
    matrix: Path | None = typer.Option(
        None, help="Custom compatibility matrix JSON", exists=True, dir_okay=False
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show check details"),
    history: bool = typer.Option(True, "--history/--no-history", help="Record this run in local history"),
) -> None:
    """Inspect this machine: hardware, software, runtime execution and compatibility."""
    s = settings_or_exit()
    fmt = resolve_format(fmt, output)
    opts = DiagnoseOptions(
        runtime=runtime and s.runtime_checks,
        torch=torch and s.torch_checks,
        python=python or s.python_executable,
        matrix_path=str(matrix) if matrix else s.compatibility_matrix,
    )
    report = _diagnose_or_exit(s, opts)
    _record(report, history and s.history)
    emit(report, fmt, output, lambda: render_diagnosis(report, verbose=verbose))
    raise typer.Exit(ExitCode.for_status(report.status))


def doctor(
    python: str | None = typer.Option(None, help="Python interpreter to inspect for PyTorch"),
    runtime: bool = typer.Option(True, "--runtime/--no-runtime", help="Run GPU execution checks"),
) -> None:
    """Quick overall environment health check with the most important fixes."""
    s = settings_or_exit()
    opts = DiagnoseOptions(
        runtime=runtime and s.runtime_checks,
        torch=s.torch_checks,
        python=python or s.python_executable,
        matrix_path=s.compatibility_matrix,
    )
    report = _diagnose_or_exit(s, opts)
    _record(report, s.history)
    checks = report.all_checks()
    counts = {st: sum(1 for c in checks if c.status == st) for st in CheckStatus}
    out.print(Text("KernelVet", style="bold") + Text("  Diagnose. Migrate. Verify.", style="dim"))
    host = report.host
    amd = [g for g in report.gpus if g.vendor_is_amd]
    out.print(f"Host: {host.os_pretty or host.os_name} · kernel {host.kernel} · {host.architecture}")
    out.print(
        f"GPUs: {', '.join(g.marketing_name or g.gfx_target or '?' for g in amd) or 'no AMD GPU detected'}"
    )
    out.print(f"ROCm: {report.software.rocm_version or 'not detected'}")
    out.print(
        "Checks: "
        + "  ".join(
            f"[{('green', 'yellow', 'red', 'dim')[i]}]{counts[st]} {st.value}[/]"
            for i, st in enumerate(
                (CheckStatus.PASS, CheckStatus.WARNING, CheckStatus.FAIL, CheckStatus.SKIPPED)
            )
        )
    )
    problems = [c for c in checks if c.status in (CheckStatus.FAIL, CheckStatus.WARNING)]
    if problems:
        t = Table(title="Needs attention", title_justify="left", title_style="bold")
        t.add_column("Check")
        t.add_column("Status")
        t.add_column("What to do")
        for c in sorted(problems, key=lambda c: c.status != CheckStatus.FAIL)[:12]:
            t.add_row(c.name, status_text(c.status), c.remediation or c.summary)
        out.print(t)
    executed = [c for c in checks if c.evidence == "executed" and c.status == CheckStatus.PASS]
    out.print(
        Text(
            f"GPU execution verified: {'yes' if executed else 'no'}", style="green" if executed else "yellow"
        )
    )
    out.print(Text(f"Overall Status: {report.status.value}", style=OVERALL_STYLE[report.status]))
    out.print("[dim]Run `kernelvet diagnose -v` for full details.[/dim]")
    raise typer.Exit(ExitCode.for_status(report.status))


def compatibility(
    fmt: OutputFormat | None = FormatOpt,
    output: Path | None = OutputOpt,
    torch: bool = typer.Option(True, "--torch/--no-torch", help="Include PyTorch compatibility"),
    python: str | None = typer.Option(None, help="Python interpreter to inspect for PyTorch"),
    matrix: Path | None = typer.Option(
        None, help="Custom compatibility matrix JSON", exists=True, dir_okay=False
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show check details"),
) -> None:
    """Validate this machine against AMD's published ROCm compatibility data (static, no GPU execution)."""
    s = settings_or_exit()
    fmt = resolve_format(fmt, output)
    m = _matrix_or_exit(str(matrix) if matrix else s.compatibility_matrix)
    with progress() as cb:
        cb("Collecting inventory", 0.3)
        host, gpus, sw = collect_inventory(
            _ctx(s.command_timeout), torch=torch and s.torch_checks, python=python or s.python_executable
        )
        cb("Evaluating matrix", 0.8)
        report = evaluate(host, gpus, sw, m)
    emit(report, fmt, output, lambda: render_compatibility(report, verbose=verbose))
    raise typer.Exit(ExitCode.for_status(report.status))


ListFormatOpt = typer.Option(None, "--format", "-f", help="terminal or json", case_sensitive=False)

gpu_app = typer.Typer(help="GPU device commands.", no_args_is_help=True)
system_app = typer.Typer(help="Host system commands.", no_args_is_help=True)


@gpu_app.command("list")
def gpu_list(
    fmt: OutputFormat | None = ListFormatOpt,
    json_out: bool = typer.Option(False, "--json", help="Same as --format json"),
) -> None:
    """List GPUs detected by amd-smi, rocminfo and /sys (static detection)."""
    as_json = _json_or_terminal(fmt, json_out)
    s = settings_or_exit()
    _host, gpus, _sw = collect_inventory(_ctx(s.command_timeout))
    if as_json:
        write_json([g.model_dump() for g in gpus])
    else:
        out.print(gpu_table(gpus))
    raise typer.Exit(ExitCode.OK if any(g.vendor_is_amd for g in gpus) else ExitCode.FAILURE)


@system_app.command("info")
def system_info(
    fmt: OutputFormat | None = ListFormatOpt,
    json_out: bool = typer.Option(False, "--json", help="Same as --format json"),
    torch: bool = typer.Option(False, "--torch", help="Also inspect PyTorch"),
    python: str | None = typer.Option(None, help="Python interpreter to inspect for PyTorch (with --torch)"),
) -> None:
    """Show OS, kernel, ROCm, HIP, driver and tool versions (static detection)."""
    as_json = _json_or_terminal(fmt, json_out)
    s = settings_or_exit()
    host, _gpus, sw = collect_inventory(
        _ctx(s.command_timeout), torch=torch, python=python or s.python_executable
    )
    if as_json:
        write_json({"host": host.model_dump(), "software": sw.model_dump()})
        return
    out.print(host_table(host))
    out.print(software_table(sw))
    t = Table(title="Tools", title_justify="left", title_style="bold")
    for col in ("Tool", "Available", "Version", "Path"):
        t.add_column(col)
    for tool in sw.tools:
        t.add_row(
            tool.name,
            Text("yes", style="green") if tool.available else Text("no", style="dim"),
            tool.version or "",
            tool.path or "",
        )
    out.print(t)
