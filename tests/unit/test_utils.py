from __future__ import annotations

import sys
from pathlib import Path

import pytest

from kernelvet.config import ConfigError, Settings, load_settings, save_settings
from kernelvet.history import HistoryStore
from kernelvet.models.common import (
    CheckResult,
    CheckStatus,
    ExitCode,
    OverallStatus,
    overall_status,
)
from kernelvet.utils.proc import CommandRunner
from kernelvet.utils.redact import redact, scrub_env
from kernelvet.utils.versions import is_prerelease, major_minor, parse_version, same_release


def c(status: CheckStatus) -> CheckResult:
    return CheckResult(id="x", name="x", category="t", status=status)


def test_overall_status_never_treats_skipped_as_success() -> None:
    assert overall_status([c(CheckStatus.PASS)]) is OverallStatus.HEALTHY
    assert overall_status([c(CheckStatus.PASS), c(CheckStatus.SKIPPED)]) is OverallStatus.REVIEW_REQUIRED
    assert overall_status([c(CheckStatus.WARNING)]) is OverallStatus.REVIEW_REQUIRED
    assert overall_status([c(CheckStatus.PASS), c(CheckStatus.FAIL)]) is OverallStatus.UNHEALTHY
    assert overall_status([]) is OverallStatus.REVIEW_REQUIRED
    assert ExitCode.for_status(OverallStatus.REVIEW_REQUIRED) == 3


def test_versions() -> None:
    assert parse_version("HIP version: 6.2.41134-65d174c3e") == (6, 2, 41134)
    assert major_minor("6.8.0-48-generic") == "6.8"
    assert same_release("10.1.0-25401", "10.1.0", depth=3)
    assert not same_release("10.1", None)
    assert parse_version("none") is None


def test_redaction() -> None:
    text = "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz0123456789 API_KEY: sk-1234567890abcdefghijklmn ok"
    out = redact(text)
    assert "hf_abc" not in out and "sk-123" not in out and out.endswith("ok")
    assert redact("Authorization: Bearer abcdefghijklmnopqrstuvwxyz").count("[REDACTED]") == 1
    env = scrub_env(
        {"PATH": "/bin", "GITHUB_TOKEN": "x", "AWS_SECRET_ACCESS_KEY": "y", "ROCM_PATH": "/opt/rocm"}
    )
    assert env == {"PATH": "/bin", "ROCM_PATH": "/opt/rocm"}


def test_runner_reports_missing_command() -> None:
    res = CommandRunner().run(["kernelvet-definitely-not-a-command"])
    assert res.not_found and not res.ok and "not installed" in res.describe_failure()


def test_runner_timeout() -> None:
    res = CommandRunner().run([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.5)
    assert res.timed_out and res.returncode is None


def test_runner_uses_argv_not_shell(tmp_path: Path) -> None:
    res = CommandRunner().run([sys.executable, "-c", "import sys; print(sys.argv[1])", "$(echo pwned); ls"])
    assert res.stdout.strip() == "$(echo pwned); ls"


def test_runner_hides_secrets_from_children(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_API_KEY", "super-secret")
    res = CommandRunner().run([sys.executable, "-c", "import os; print(os.environ.get('MY_API_KEY'))"])
    assert res.stdout.strip() == "None"


def test_settings_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "config.toml"
    s = Settings(command_timeout=12, isolation="docker", offload_arch="gfx942,gfx1100")
    save_settings(s, path)
    loaded = load_settings(path)
    assert loaded == s
    monkeypatch.setenv("KERNELVET_RUNTIME_CHECKS", "false")
    assert load_settings(path).runtime_checks is False


def test_settings_reject_bad_values(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('isolation = "vm"\n')
    with pytest.raises(ConfigError):
        load_settings(path)
    path.write_text("not = [toml")
    with pytest.raises(ConfigError):
        load_settings(path)


def test_history_store(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path)
    assert store.recent_runs() == [] and store.recent_projects() == []
    rec = store.record_run("diagnosis", "HEALTHY", "3 PASS", Settings())
    assert rec.report_path and Path(rec.report_path).exists()
    store.record_project("/p", "analyzed", "x")
    store.record_project("/p", "converted", "y")
    projects = store.recent_projects()
    assert len(projects) == 1 and projects[0].stage == "converted"
    assert store.recent_runs(kind="diagnosis")[0].status == "HEALTHY"
    assert store.saved_reports()


def test_unusable_history_raises_history_error(tmp_path: Path) -> None:
    from kernelvet.history import HistoryError

    store = HistoryStore(tmp_path)
    tmp_path.mkdir(exist_ok=True)
    store.db_path.write_text("not a database")
    for call in (
        lambda: store.recent_runs(),
        lambda: store.recent_projects(),
        lambda: store.record_run("diagnosis", "HEALTHY", "ok"),
        lambda: store.record_project("/p", "analyzed", "ok"),
    ):
        with pytest.raises(HistoryError, match=r"history\.db"):
            call()


def test_prerelease_versions() -> None:
    for v in ("10.1.0a20260823", "10.2.0rc1", "10.2.0.dev0", "10.2b3"):
        assert is_prerelease(v), v
    # releases, build suffixes and git hashes are not pre-releases
    for v in ("10.2.0", "7.17.26434-0000000", "6.2.41134-65d174c3e", "6.2.4-a1b2", None, ""):
        assert not is_prerelease(v), v
