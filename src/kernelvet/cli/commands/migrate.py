"""`kernelvet migrate ...` — CUDA-to-HIP migration workflow."""

from __future__ import annotations

from pathlib import Path

import typer

from kernelvet.cli.output import (
    FormatOpt,
    OutputOpt,
    emit,
    err,
    progress,
    resolve_format,
    settings_or_exit,
)
from kernelvet.diagnose.hardware import discover_gpus
from kernelvet.diagnose.software import detect_libraries
from kernelvet.execution.compiler import build_project
from kernelvet.execution.runner import run_tests
from kernelvet.execution.sandbox import SandboxConfig
from kernelvet.history import HistoryError, HistoryStore
from kernelvet.migration import (
    HipifyUnavailable,
    analyze_project,
    audit_project,
    build_plan,
    convert_project,
)
from kernelvet.migration.hipify import Translator
from kernelvet.migration.workflow import (
    assemble_report,
    compute_stages,
    load_state,
    recommendations,
    save_state,
)
from kernelvet.models.common import ExitCode
from kernelvet.models.migration import (
    AuditResult,
    BuildResult,
    ConversionResult,
    FindingCategory,
    StageStatus,
)
from kernelvet.models.report import MigrationReport
from kernelvet.reporting import OutputFormat
from kernelvet.reporting.terminal import render_migration
from kernelvet.utils.paths import PathError, existing_dir
from kernelvet.utils.proc import HostContext

app = typer.Typer(
    help="Analyze, plan, convert (via AMD HIPIFY), audit, build and test CUDA→HIP migrations.",
    no_args_is_help=True,
)

ProjectArg = typer.Argument(..., help="Project directory", show_default=False)
TranslatorOpt = typer.Option(None, "--translator", help="Which HIPIFY to use (default: config value)")
IsolationOpt = typer.Option(None, "--isolation", help="none or docker (default from config)")


def _dir(path: Path) -> Path:
    try:
        return existing_dir(path)
    except PathError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(ExitCode.USAGE) from exc


def _record(project: Path, stage: str, summary: str, report: MigrationReport | None = None) -> None:
    s = settings_or_exit()
    if not s.history:
        return
    try:
        HistoryStore().record_project(str(project), stage, summary, report)
    except HistoryError as exc:  # history is best-effort: never lose the result over it
        err.print(f"[yellow]could not record history: {exc}[/yellow]")


def _targets(ctx: HostContext, explicit: list[str] | None) -> list[str]:
    s = settings_or_exit()
    if explicit:
        return explicit
    if s.offload_arch:
        return [a.strip() for a in s.offload_arch.split(",") if a.strip()]
    return sorted(
        {g.gfx_target.split(":")[0] for g in discover_gpus(ctx).gpus if g.vendor_is_amd and g.gfx_target}
    )


@app.command()
def analyze(
    project: Path = ProjectArg, fmt: OutputFormat | None = FormatOpt, output: Path | None = OutputOpt
) -> None:
    """Scan a CUDA project and classify every CUDA construct (read-only)."""
    root = _dir(project)
    fmt = resolve_format(fmt, output)
    with progress() as cb:
        cb("Scanning project", 0.3)
        analysis = analyze_project(root)
    report = MigrationReport(project=str(root), analysis=analysis)
    _record(root, "analyzed", f"readiness {analysis.readiness.score}/100, {len(analysis.findings)} findings")
    emit(report, fmt, output, lambda: render_migration(report))


@app.command()
def plan(
    project: Path = ProjectArg,
    fmt: OutputFormat | None = FormatOpt,
    output: Path | None = OutputOpt,
    translator: Translator | None = TranslatorOpt,
    offload_arch: list[str] | None = typer.Option(None, "--offload-arch", help="Target gfx architecture(s)"),
) -> None:
    """Produce a migration plan before anything is modified (a dry run of the whole migration)."""
    s = settings_or_exit()
    root = _dir(project)
    fmt = resolve_format(fmt, output)
    ctx = HostContext(timeout=s.command_timeout)
    with progress() as cb:
        cb("Analyzing project", 0.3)
        analysis = analyze_project(root)
        cb("Planning", 0.8)
        p = build_plan(
            analysis,
            ctx,
            preference=translator or s.hipify_preference,
            offload_arch=_targets(ctx, offload_arch),
        )
    report = MigrationReport(project=str(root), analysis=analysis, plan=p)
    report.recommendations = recommendations(report)
    _record(root, "planned", f"{len(p.steps)} steps, {len(p.blockers)} blockers")
    emit(report, fmt, output, lambda: render_migration(report))
    raise typer.Exit(ExitCode.REVIEW if p.blockers or p.translator is None else ExitCode.OK)


@app.command()
def convert(
    project: Path = ProjectArg,
    output_dir: Path = typer.Option(
        ..., "--output", "-o", help="Directory for the migrated HIP project", file_okay=False
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be converted; write nothing"),
    force: bool = typer.Option(False, "--force", help="Allow writing into a non-empty output directory"),
    translator: Translator | None = TranslatorOpt,
    fmt: OutputFormat | None = FormatOpt,
    report_file: Path | None = typer.Option(None, "--report", help="Also write a report file"),
) -> None:
    """Convert CUDA sources to HIP with AMD HIPIFY into a separate directory (sources are never modified)."""
    s = settings_or_exit()
    root = _dir(project)
    ctx = HostContext(timeout=s.command_timeout)
    analysis = analyze_project(root)
    try:
        with progress() as cb:
            result = convert_project(
                ctx,
                analysis.inventory,
                output_dir,
                preference=translator or s.hipify_preference,
                dry_run=dry_run,
                force=force,
                timeout=s.build_timeout,
                progress=cb,
            )
    except HipifyUnavailable as exc:
        err.print(f"[red]HIPIFY unavailable:[/red] {exc}")
        err.print(
            "KernelVet does not translate CUDA itself; install hipify-clang/hipify-perl "
            "(ROCm package `hipify-clang`) and retry."
        )
        raise typer.Exit(ExitCode.UNAVAILABLE) from exc
    except PathError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(ExitCode.USAGE) from exc
    if not dry_run:
        save_state(result.output, "conversion", result)
    report = MigrationReport(
        project=result.output,
        analysis=analysis,
        conversion=result,
        stages=compute_stages(result, None, None, None),
    )
    report.recommendations = recommendations(report)
    if not dry_run:
        _record(
            Path(result.output),
            "converted",
            f"{len(result.files)} files, {len(result.failed)} failed ({result.translator})",
            report,
        )
    fmt = resolve_format(fmt, report_file) if (fmt or report_file) else OutputFormat.TERMINAL
    emit(report, fmt, report_file, lambda: render_migration(report))
    if not dry_run:
        err.print(f"Migrated project: {result.output}  ·  log: {result.log_path}")
        err.print("Next: kernelvet migrate audit " + result.output)
    raise typer.Exit(ExitCode.FAILURE if result.failed else ExitCode.OK)


@app.command()
def audit(
    project: Path = ProjectArg, fmt: OutputFormat | None = FormatOpt, output: Path | None = OutputOpt
) -> None:
    """Statically audit a migrated HIP project for leftover CUDA and risky patterns."""
    s = settings_or_exit()
    root = _dir(project)
    fmt = resolve_format(fmt, output)
    ctx = HostContext(timeout=s.command_timeout)
    rocm = ctx.rocm_path()
    libs = detect_libraries(ctx, rocm) if ctx.exists(rocm) else None
    with progress() as cb:
        cb("Auditing", 0.5)
        result = audit_project(root, installed_libraries=libs)
    if libs is None:
        err.print("[yellow]ROCm not found: missing-library checks were skipped[/yellow]")
    if (root / ".kernelvet").is_dir():
        save_state(root, "audit", result)
    report = MigrationReport(
        project=str(root),
        audit=result,
        stages=compute_stages(load_state(root, "conversion", ConversionResult), result, None, None),
    )
    report.recommendations = recommendations(report)
    _record(root, "audited", f"{len(result.findings)} findings")
    emit(report, fmt, output, lambda: render_migration(report))
    cats = {f.category for f in result.findings}
    if cats & {FindingCategory.MANUAL_INTERVENTION, FindingCategory.UNSUPPORTED}:
        raise typer.Exit(ExitCode.FAILURE)
    raise typer.Exit(ExitCode.REVIEW if FindingCategory.REQUIRES_VERIFICATION in cats else ExitCode.OK)


@app.command()
def build(
    project: Path = ProjectArg,
    offload_arch: list[str] | None = typer.Option(None, "--offload-arch", help="gfx target(s), e.g. gfx942"),
    isolation: str | None = IsolationOpt,
    fmt: OutputFormat | None = FormatOpt,
    output: Path | None = OutputOpt,
) -> None:
    """Compile a migrated project with hipcc (or its HIP CMake build)."""
    s = settings_or_exit()
    root = _dir(project)
    fmt = resolve_format(fmt, output)
    ctx = HostContext(timeout=s.command_timeout)
    iso = isolation or s.isolation
    with progress() as cb:
        cb("Building", 0.3)
        result = build_project(
            ctx,
            root,
            gfx_targets=_targets(ctx, offload_arch),
            isolation=iso,
            sandbox=SandboxConfig(image=s.docker_image),
            timeout=s.build_timeout,
        )
    save_state(root, "build", result)
    report = MigrationReport(
        project=str(root),
        build=result,
        stages=compute_stages(
            load_state(root, "conversion", ConversionResult),
            load_state(root, "audit", AuditResult),
            result,
            None,
        ),
    )
    report.recommendations = recommendations(report)
    _record(root, "built", f"build {result.status}")
    emit(report, fmt, output, lambda: render_migration(report))
    raise typer.Exit(
        {StageStatus.PASSED: ExitCode.OK, StageStatus.UNAVAILABLE: ExitCode.UNAVAILABLE}.get(
            result.status, ExitCode.FAILURE
        )
    )


@app.command()
def test(
    project: Path = ProjectArg,
    run_all: bool = typer.Option(False, "--all", help="Run every built executable, not just tests"),
    isolation: str | None = IsolationOpt,
    fmt: OutputFormat | None = FormatOpt,
    output: Path | None = OutputOpt,
) -> None:
    """Run the migrated project's tests and record whether they ran on an AMD GPU."""
    s = settings_or_exit()
    root = _dir(project)
    fmt = resolve_format(fmt, output)
    ctx = HostContext(timeout=s.command_timeout)
    build_result = load_state(root, "build", BuildResult)
    with progress() as cb:
        cb("Running tests", 0.3)
        result = run_tests(
            ctx,
            root,
            build_result,
            run_all=run_all,
            isolation=isolation or s.isolation,
            sandbox=SandboxConfig(image=s.docker_image),
            timeout=s.test_timeout,
        )
    save_state(root, "tests", result)
    stages = compute_stages(
        load_state(root, "conversion", ConversionResult),
        load_state(root, "audit", AuditResult),
        build_result,
        result,
    )
    report = MigrationReport(project=str(root), build=build_result, tests=result, stages=stages)
    report.recommendations = recommendations(report)
    _record(root, "tested", f"tests {result.status}, GPU validation {stages.gpu_runtime_validation}")
    emit(report, fmt, output, lambda: render_migration(report))
    if result.status is StageStatus.FAILED:
        raise typer.Exit(ExitCode.FAILURE)
    if result.status is StageStatus.NOT_RUN:
        raise typer.Exit(ExitCode.UNAVAILABLE)
    raise typer.Exit(ExitCode.OK if stages.gpu_runtime_validation is StageStatus.PASSED else ExitCode.REVIEW)


@app.command()
def report(
    project: Path = ProjectArg, fmt: OutputFormat | None = FormatOpt, output: Path | None = OutputOpt
) -> None:
    """Assemble the full migration report from every stage run so far."""
    root = _dir(project)
    fmt = resolve_format(fmt, output)
    with progress() as cb:
        cb("Assembling report", 0.5)
        r = assemble_report(root)
    emit(r, fmt, output, lambda: render_migration(r))
    st = r.stages
    if any(getattr(st, k) is StageStatus.FAILED for k in type(st).model_fields):
        raise typer.Exit(ExitCode.FAILURE)
    raise typer.Exit(ExitCode.OK if st.gpu_runtime_validation is StageStatus.PASSED else ExitCode.REVIEW)
