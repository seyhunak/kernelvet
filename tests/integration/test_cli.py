"""CLI integration tests (Typer CliRunner). Hardware is faked; commands are real."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from typer.testing import CliRunner

from conftest import FakeRunner
from kernelvet import __version__
from kernelvet.cli.app import app
from kernelvet.config import Settings
from kernelvet.models.report import DiagnosticReport, MigrationReport
from kernelvet.utils.proc import HostContext

runner = CliRunner()


def flat(text: str) -> str:
    """Undo Rich's line wrapping so assertions don't depend on terminal width."""
    return " ".join(text.split())


@pytest.fixture
def fake_host(monkeypatch: pytest.MonkeyPatch, mi300x_ctx: HostContext) -> HostContext:
    """Route every HostContext the CLI creates to the fake MI300X host."""

    def factory(*_a: object, **_k: object) -> HostContext:
        return mi300x_ctx

    for mod in (
        "kernelvet.cli.commands.diagnose",
        "kernelvet.cli.commands.migrate",
        "kernelvet.cli.commands.misc",
    ):
        monkeypatch.setattr(f"{mod}.HostContext", factory)
    return mi300x_ctx


@pytest.fixture
def bare_host(monkeypatch: pytest.MonkeyPatch, empty_linux_ctx: HostContext) -> HostContext:
    def factory(*_a: object, **_k: object) -> HostContext:
        return empty_linux_ctx

    for mod in (
        "kernelvet.cli.commands.diagnose",
        "kernelvet.cli.commands.migrate",
        "kernelvet.cli.commands.misc",
    ):
        monkeypatch.setattr(f"{mod}.HostContext", factory)
    return empty_linux_ctx


def test_version_and_help() -> None:
    r = runner.invoke(app, ["--version"])
    assert r.exit_code == 0 and r.stdout.strip() == f"kernelvet {__version__}"
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    for cmd in (
        "tui",
        "diagnose",
        "doctor",
        "compatibility",
        "migrate",
        "gpu",
        "system",
        "config",
        "benchmark",
    ):
        assert cmd in r.stdout
    assert "Exit codes" in r.stdout
    r = runner.invoke(app, ["migrate", "--help"])
    for cmd in ("analyze", "plan", "convert", "audit", "build", "test", "report"):
        assert cmd in r.stdout


def test_diagnose_json_is_clean_and_schema_valid(fake_host: HostContext) -> None:
    r = runner.invoke(app, ["diagnose", "--format", "json"])
    assert r.exit_code == 0, r.stderr
    data = json.loads(r.stdout)  # stdout contains only the JSON document
    jsonschema.validate(data, DiagnosticReport.model_json_schema())
    assert data["status"] == "HEALTHY"


def test_diagnose_exit_codes(bare_host: HostContext) -> None:
    r = runner.invoke(app, ["diagnose", "-f", "json"])
    assert r.exit_code == 1
    assert json.loads(r.stdout)["status"] == "UNHEALTHY"


def test_diagnose_output_file_formats(fake_host: HostContext, tmp_path: Path) -> None:
    for name, marker in (
        ("r.json", '"kind": "diagnosis"'),
        ("r.md", "# KernelVet System Diagnosis"),
        ("r.html", "<!doctype html>"),
    ):
        r = runner.invoke(app, ["diagnose", "--output", str(tmp_path / name)])
        assert r.exit_code == 0 and r.stdout == ""
        assert marker in (tmp_path / name).read_text()
    r = runner.invoke(app, ["diagnose", "--output", str(tmp_path / "r.txt")])
    assert r.exit_code == 2


def test_diagnose_terminal_and_history(fake_host: HostContext) -> None:
    r = runner.invoke(app, ["diagnose", "--verbose"])
    assert r.exit_code == 0
    assert "KernelVet System Diagnosis" in r.stdout and "Overall Status: HEALTHY" in r.stdout
    h = runner.invoke(app, ["history", "--json"])
    runs = json.loads(h.stdout)["runs"]
    assert runs and runs[0]["kind"] == "diagnosis" and runs[0]["status"] == "HEALTHY"


def test_doctor(bare_host: HostContext) -> None:
    r = runner.invoke(app, ["doctor"])
    assert "Overall Status" in r.stdout and "GPU execution verified: no" in r.stdout
    assert r.exit_code == 1


def test_doctor_healthy(fake_host: HostContext) -> None:
    r = runner.invoke(app, ["doctor"])
    assert r.exit_code == 0 and "GPU execution verified: yes" in r.stdout


def test_compatibility(fake_host: HostContext) -> None:
    r = runner.invoke(app, ["compatibility", "-f", "json"])
    data = json.loads(r.stdout)
    assert {c["id"] for c in data["checks"]} >= {"compat.gpu", "compat.os", "compat.rocm"}
    assert data["matrix_sources"]
    assert r.exit_code == 0


def test_gpu_list_and_system_info(fake_host: HostContext) -> None:
    r = runner.invoke(app, ["gpu", "list", "--json"])
    assert r.exit_code == 0 and json.loads(r.stdout)[0]["gfx_target"] == "gfx942"
    r = runner.invoke(app, ["gpu", "list"])
    assert "MI300X" in r.stdout and "gfx942" in r.stdout
    r = runner.invoke(app, ["system", "info", "--json"])
    assert json.loads(r.stdout)["software"]["rocm_version"] == "10.1.0-25401"
    assert runner.invoke(app, ["system", "info"]).exit_code == 0


def test_gpu_list_without_gpu_fails(bare_host: HostContext) -> None:
    assert runner.invoke(app, ["gpu", "list"]).exit_code == 1


def test_config_commands() -> None:
    r = runner.invoke(app, ["config", "show", "--json"])
    assert r.exit_code == 0 and json.loads(r.stdout)["settings"]["isolation"] == "none"
    assert runner.invoke(app, ["config", "set", "isolation", "docker"]).exit_code == 0
    assert (
        json.loads(runner.invoke(app, ["config", "show", "--json"]).stdout)["settings"]["isolation"]
        == "docker"
    )
    assert runner.invoke(app, ["config", "set", "isolation", "vm"]).exit_code == 2
    assert runner.invoke(app, ["config", "set", "nope", "1"]).exit_code == 2
    assert "config.toml" in runner.invoke(app, ["config", "path"]).stdout
    assert runner.invoke(app, ["config", "show"]).exit_code == 0


def test_schema_command() -> None:
    r = runner.invoke(app, ["schema", "migration"])
    assert json.loads(r.stdout)["title"] == "MigrationReport"
    assert runner.invoke(app, ["schema", "nope"]).exit_code == 2


# --------------------------------------------------------------------------- migrate


def test_migrate_analyze_formats(cuda_project: Path, bare_host: HostContext) -> None:
    r = runner.invoke(app, ["migrate", "analyze", str(cuda_project), "-f", "json"])
    assert r.exit_code == 0
    data = json.loads(r.stdout)
    jsonschema.validate(data, MigrationReport.model_json_schema())
    assert data["analysis"]["readiness"]["score"] <= 60
    r = runner.invoke(app, ["migrate", "analyze", str(cuda_project), "-f", "sarif"])
    assert json.loads(r.stdout)["version"] == "2.1.0"
    assert "Findings" in runner.invoke(app, ["migrate", "analyze", str(cuda_project)]).stdout
    assert runner.invoke(app, ["migrate", "analyze", "/does/not/exist"]).exit_code == 2


def test_migrate_plan_review_when_blocked(cuda_project: Path, bare_host: HostContext) -> None:
    r = runner.invoke(app, ["migrate", "plan", str(cuda_project), "-f", "markdown"])
    assert r.exit_code == 3
    assert "## Migration plan" in r.stdout and "Install HIPIFY" in r.stdout


def test_migrate_convert_unavailable(cuda_project: Path, bare_host: HostContext, tmp_path: Path) -> None:
    r = runner.invoke(app, ["migrate", "convert", str(cuda_project), "--output", str(tmp_path / "out")])
    assert r.exit_code == 4
    assert "HIPIFY unavailable" in r.stderr
    assert not (tmp_path / "out").exists()


def test_migrate_rejects_unknown_translator(
    cuda_project: Path, bare_host: HostContext, tmp_path: Path
) -> None:
    for cmd in (["plan", str(cuda_project)], ["convert", str(cuda_project), "-o", str(tmp_path / "out")]):
        r = runner.invoke(app, ["migrate", *cmd, "--translator", "foo"])
        assert r.exit_code == 2
        assert not (tmp_path / "out").exists()


def test_migrate_convert_rejects_output_inside_source(
    cuda_project: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from conftest import fake_hipify

    ctx = HostContext(runner=fake_hipify(), root=tmp_path, system="Linux")
    monkeypatch.setattr("kernelvet.cli.commands.migrate.HostContext", lambda *a, **k: ctx)
    r = runner.invoke(app, ["migrate", "convert", str(cuda_project), "--output", str(cuda_project / "hip")])
    assert r.exit_code == 2 and "inside the source project" in flat(r.stderr)


def test_full_migration_workflow(cuda_project: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """convert → audit → build → test → report with fake HIPIFY/hipcc on a GPU-less host."""
    from conftest import fake_hipify

    r_ = fake_hipify()
    r_.tools["hipcc"] = "/opt/rocm/bin/hipcc"

    def hipcc(argv: list[str], kw: dict) -> object:
        from kernelvet.utils.proc import CommandResult

        (Path(kw["cwd"]) / argv[argv.index("-o") + 1]).write_text("#!fake")
        return CommandResult(argv, 0)

    r_.on(lambda a: a[0] == "hipcc", hipcc)
    from kernelvet.utils.proc import CommandResult

    r_.on(lambda a: a[0].startswith("./"), lambda a, kw: CommandResult(a, 0, stdout="PASS\n"))
    ctx = HostContext(runner=r_, root=tmp_path / "sysroot", system="Linux")
    (tmp_path / "sysroot").mkdir()
    monkeypatch.setattr("kernelvet.cli.commands.migrate.HostContext", lambda *a, **k: ctx)
    out = tmp_path / "hip"

    r = runner.invoke(app, ["migrate", "convert", str(cuda_project), "-o", str(out)])
    assert r.exit_code == 0, r.stderr
    assert (out / ".kernelvet/conversion.json").exists()
    r = runner.invoke(app, ["migrate", "audit", str(out), "-f", "json"])
    assert r.exit_code == 1  # inline PTX and NVTX remain: manual work needed
    r = runner.invoke(app, ["migrate", "build", str(out), "--offload-arch", "gfx942"])
    assert r.exit_code == 0, r.stdout
    r = runner.invoke(app, ["migrate", "test", str(out)])
    assert r.exit_code == 3  # tests passed, but no AMD GPU: not validated
    r = runner.invoke(app, ["migrate", "report", str(out), "-f", "json"])
    data = json.loads(r.stdout)
    assert data["stages"] == {
        "source_conversion": "PASSED",
        "static_audit": "FAILED",
        "compilation": "PASSED",
        "tests": "PASSED",
        "gpu_runtime_validation": "UNAVAILABLE",
    }
    assert r.exit_code == 1
    md = runner.invoke(app, ["migrate", "report", str(out), "-f", "markdown"]).stdout
    assert "5. AMD GPU runtime validation | UNAVAILABLE" in md
    h = json.loads(runner.invoke(app, ["history", "--json"]).stdout)
    assert any(p["stage"] == "tested" for p in h["projects"])


def test_migrate_test_without_build(hip_project: Path, bare_host: HostContext) -> None:
    assert runner.invoke(app, ["migrate", "test", str(hip_project)]).exit_code == 4


def test_benchmark_without_hardware(bare_host: HostContext) -> None:
    r = runner.invoke(app, ["benchmark", "-f", "json", "--iterations", "3"])
    data = json.loads(r.stdout)
    assert {x["status"] for x in data["results"]} == {"SKIPPED"}
    assert all(not x["measurements"] for x in data["results"])
    assert r.exit_code == 3


def test_benchmark_rejects_options_that_cannot_run(bare_host: HostContext) -> None:
    for args in (["--no-hip", "--no-torch"], ["--no-hip", "--profile"], ["--no-torch", "--inference"]):
        r = runner.invoke(app, ["benchmark", *args])
        assert r.exit_code == 2, args


def test_benchmark_reports_skipped_profiling(bare_host: HostContext) -> None:
    r = runner.invoke(app, ["benchmark", "--profile"])
    assert r.exit_code == 3
    assert "Profiler" in flat(r.stdout) and "not run" in flat(r.stdout)


def test_export_saved_report(fake_host: HostContext, tmp_path: Path) -> None:
    runner.invoke(app, ["diagnose", "-o", str(tmp_path / "d.json")])
    r = runner.invoke(app, ["export", str(tmp_path / "d.json"), "-f", "markdown"])
    assert r.exit_code == 0 and "# KernelVet System Diagnosis" in r.stdout
    (tmp_path / "bad.json").write_text('{"kind": "x"}')
    assert runner.invoke(app, ["export", str(tmp_path / "bad.json"), "-f", "json"]).exit_code == 2


def test_fake_runner_is_used(fake_host: HostContext) -> None:
    runner.invoke(app, ["gpu", "list"])
    assert isinstance(fake_host.runner, FakeRunner) and fake_host.runner.calls


@pytest.mark.parametrize("content", ["{not json", '{"schema": 1}'])
def test_bad_matrix_is_a_usage_error_not_a_traceback(
    fake_host: HostContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    bad = tmp_path / "matrix.json"
    bad.write_text(content)
    for args in (["compatibility", "--matrix", str(bad)], ["diagnose", "--matrix", str(bad), "-f", "json"]):
        r = runner.invoke(app, args)
        assert r.exit_code == 2, (args, r.exception)
        assert isinstance(r.exception, SystemExit) and r.stdout == ""
        assert "compatibility matrix error" in flat(r.stderr)
    monkeypatch.setenv("KERNELVET_COMPATIBILITY_MATRIX", str(bad))
    r = runner.invoke(app, ["doctor"])
    assert r.exit_code == 2 and "compatibility matrix error" in flat(r.stderr)
    monkeypatch.setenv("KERNELVET_COMPATIBILITY_MATRIX", str(tmp_path / "missing.json"))
    r = runner.invoke(app, ["doctor"])
    assert r.exit_code == 2 and "could not read" in flat(r.stderr)


def test_compatibility_json_carries_status_and_matches_schema(fake_host: HostContext) -> None:
    r = runner.invoke(app, ["compatibility", "-f", "json"])
    data = json.loads(r.stdout)
    assert data["kind"] == "compatibility" and "meta" in data
    assert data["status"] == "HEALTHY" and r.exit_code == 0
    schema = json.loads(runner.invoke(app, ["schema", "compatibility"]).stdout)
    assert "status" in schema["properties"]
    jsonschema.validate(data, schema)


def test_gpu_and_system_accept_format_json(fake_host: HostContext) -> None:
    for args in (["gpu", "list"], ["system", "info"]):
        a = runner.invoke(app, [*args, "-f", "json"])
        b = runner.invoke(app, [*args, "--json"])
        assert a.exit_code == b.exit_code == 0
        assert json.loads(a.stdout) == json.loads(b.stdout)
        r = runner.invoke(app, [*args, "-f", "html"])
        assert r.exit_code == 2 and "use terminal or json" in flat(r.stderr)
    r = runner.invoke(app, ["system", "info", "--torch", "--python", "python3", "-f", "json"])
    assert r.exit_code == 0 and "software" in json.loads(r.stdout)


def _corrupt_history() -> None:
    from kernelvet.config.settings import data_dir

    data_dir().mkdir(parents=True, exist_ok=True)
    (data_dir() / "history.db").write_text("not a database")


def test_corrupt_history_never_loses_a_report(fake_host: HostContext, cuda_project: Path) -> None:
    _corrupt_history()
    r = runner.invoke(app, ["diagnose", "-f", "json"])
    assert r.exit_code == 0 and json.loads(r.stdout)["status"] == "HEALTHY"
    assert "could not record history" in flat(r.stderr)
    for args in (["doctor"], ["benchmark"], ["migrate", "analyze", str(cuda_project)]):
        r = runner.invoke(app, args)
        assert not isinstance(r.exception, Exception) or isinstance(r.exception, SystemExit), args
        assert r.stdout and "could not record history" in flat(r.stderr), args
    for args in (["history"], ["history", "--json"]):
        r = runner.invoke(app, args)
        assert r.exit_code == 2 and r.stdout == "" and "Move or delete history.db" in flat(r.stderr)


def test_export_handles_compatibility_and_non_reports(fake_host: HostContext, tmp_path: Path) -> None:
    saved = tmp_path / "compat.json"
    assert runner.invoke(app, ["compatibility", "-o", str(saved)]).exit_code == 0
    r = runner.invoke(app, ["export", str(saved)])
    assert r.exit_code == 0 and "Compatibility" in r.stdout
    r = runner.invoke(app, ["export", str(saved), "-f", "markdown"])
    assert "# KernelVet Compatibility Report" in r.stdout and "**Status: HEALTHY**" in r.stdout
    not_object = tmp_path / "list.json"
    not_object.write_text("[]")
    r = runner.invoke(app, ["export", str(not_object)])
    assert r.exit_code == 2 and isinstance(r.exception, SystemExit)
    assert "is not a KernelVet report" in flat(r.stderr)


def test_config_set_can_choose_none_and_reset_any_setting() -> None:
    from kernelvet.config import load_settings

    def config_set(key: str, value: str) -> None:
        r = runner.invoke(app, ["config", "set", key, value])
        assert r.exit_code == 0, (key, value, r.stderr)

    # the README's `isolation docker` must be reversible from the CLI
    config_set("isolation", "docker")
    config_set("isolation", "none")
    assert load_settings().isolation == "none"
    # an empty value resets every kind of setting to its default
    for key, value in (
        ("isolation", "docker"),
        ("build_timeout", "60"),
        ("history", "false"),
        ("hipify_preference", "hipify-perl"),
        ("offload_arch", "gfx942"),
    ):
        config_set(key, value)
        config_set(key, "")
        assert getattr(load_settings(), key) == Settings.model_fields[key].default, key
    # "none"/"null" clear a setting that can be unset, and nothing else
    config_set("python_executable", "/usr/bin/python3")
    config_set("python_executable", "null")
    assert load_settings().python_executable is None
    r = runner.invoke(app, ["config", "set", "hipify_preference", "none"])
    assert r.exit_code == 2 and "invalid value for hipify_preference" in flat(r.stderr)
