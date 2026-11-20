from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import HOST, FakeRunner, mi300x_runner
from kernelvet.diagnose.engine import DiagnoseOptions, run_diagnosis, static_checks
from kernelvet.diagnose.hardware import (
    _bdf_from_id,
    discover_gpus,
    gpu_agents,
    parse_amd_smi_static,
    parse_rocminfo,
)
from kernelvet.diagnose.runtime import parse_probe_output, run_hip_probe, run_torch_probe
from kernelvet.diagnose.software import (
    _shebang_python,
    collect_software,
    detect_rocm_version,
    detect_tools,
    installed_libraries,
    locate_rocm,
)
from kernelvet.diagnose.system import collect_host, parse_os_release
from kernelvet.models.common import CheckStatus, Evidence, OverallStatus
from kernelvet.models.system import TorchInfo
from kernelvet.utils.proc import CommandResult, HostContext

# --------------------------------------------------------------------------- parsers


def test_parse_rocminfo_finds_gpu_agent_with_vram_and_bdf() -> None:
    agents = parse_rocminfo((HOST / "rocminfo.txt").read_text())
    assert len(agents) == 2
    gpus = gpu_agents(agents)
    assert len(gpus) == 1
    g = gpus[0]
    assert g.name == "gfx942"
    assert g.marketing_name == "AMD Instinct MI300X"
    assert g.compute_units == 304
    assert g.bdf == "0c:00.0"
    assert g.pool_bytes and max(g.pool_bytes) == 201310208 * 1024


def test_bdf_decoding() -> None:
    assert _bdf_from_id("3072") == "0c:00.0"
    assert _bdf_from_id("49920") == "c3:00.0"


def test_parse_amd_smi_static_handles_wrapped_values() -> None:
    entries = parse_amd_smi_static((HOST / "amd-smi-static.json").read_text())
    assert entries[0]["gfx_target"] == "gfx942"
    assert entries[0]["vram_bytes"] == 196592 * 1024**2
    assert entries[0]["pci_bus"] == "0c:00.0"
    assert entries[0]["compute_units"] == 304


def test_parse_amd_smi_static_gpu_data_envelope_and_plain_mb() -> None:
    text = '{"gpu_data": [{"gpu": 0, "asic": {"market_name": "X", "vendor_id": "0x1002"}, "vram": {"size": "8192 MB"}}]}'
    e = parse_amd_smi_static(text)[0]
    assert e["vram_bytes"] == 8192 * 1024**2


def test_parse_os_release_strips_quotes() -> None:
    info = parse_os_release('NAME="Ubuntu"\nVERSION_ID="24.04"\n# comment\nID=ubuntu\n')
    assert info == {"NAME": "Ubuntu", "VERSION_ID": "24.04", "ID": "ubuntu"}


# --------------------------------------------------------------------------- discovery


def test_discover_gpus_merges_sources_by_pci_bus(mi300x_ctx: HostContext) -> None:
    probe = discover_gpus(mi300x_ctx)
    assert len(probe.gpus) == 1
    g = probe.gpus[0]
    assert g.vendor_is_amd
    assert sorted(g.sources) == ["amd-smi", "rocminfo", "sysfs"]
    assert g.gfx_target == "gfx942"
    assert g.vram_bytes == 196592 * 1024**2
    assert g.accessible is True


def test_no_gpu_is_invented_without_any_source(empty_linux_ctx: HostContext) -> None:
    probe = discover_gpus(empty_linux_ctx)
    assert probe.gpus == []
    assert probe.kfd_accessible is False


def test_rocminfo_failure_is_recorded(tmp_path: Path) -> None:
    r = FakeRunner({"rocminfo": "/usr/bin/rocminfo"})
    r.on("rocminfo", CommandResult(["rocminfo"], 1, stderr="hsa api call failure at: rocminfo.cc"))
    ctx = HostContext(runner=r, root=tmp_path, system="Linux")
    probe = discover_gpus(ctx)
    assert probe.gpus == []
    assert any("exited with 1" in n for n in probe.notes)


def test_collect_software_reads_versions(mi300x_ctx: HostContext) -> None:
    sw = collect_software(mi300x_ctx)
    assert sw.rocm_version == "10.2.0"
    assert sw.hip_runtime_version == "7.17.26434-0000000"
    assert sw.hip_compiler_version.startswith("AMD clang version 24")
    assert sw.amdgpu_driver_version == "31.60.0"
    assert sw.amdgpu_module_loaded is True
    assert {"HIP runtime", "rocBLAS", "hipBLAS", "RCCL", "MIOpen", "hipCUB"} <= set(sw.rocm_libraries)
    assert sw.torch is not None and sw.torch.is_rocm_build
    assert sw.has("hipcc") and not sw.has("hipify-perl")


def test_rocm_10_core_layout_version(tmp_path: Path) -> None:
    # ROCm 10.x (TheRock): /opt/rocm has core-10.2/.info/version and no /opt/rocm/.info.
    for name in ("core-10.1", "core-10.2"):
        (tmp_path / f"opt/rocm/{name}/.info").mkdir(parents=True)
    (tmp_path / "opt/rocm/core-10.2/.info/version").write_text("10.2.0\n")
    (tmp_path / "opt/rocm/core-10.1/.info/version").write_text("10.1.0\n")
    ctx = HostContext(runner=FakeRunner({}), root=tmp_path, system="Linux")
    assert detect_rocm_version(ctx, "/opt/rocm") == "10.2.0"
    (tmp_path / "opt/rocm/core-10.2/.info/version").unlink()
    assert detect_rocm_version(ctx, "/opt/rocm") == "10.2"  # falls back to the directory name


SITE = "/venv/lib/python3.12/site-packages"
PIP_LONG_PATH_SCRIPT = "#!/bin/sh\n'''exec' \"/very/long/venv/bin/python3\" \"$0\" \"$@\"\n' '''\n"


def pip_rocm_ctx(root: Path, *, version: str = "10.2.0", probe: bool = True) -> HostContext:
    """ROCm from TheRock's Python packages in /venv (``pip install rocm[libraries,devel]``), no /opt/rocm."""
    (root / "etc").mkdir(parents=True, exist_ok=True)
    (root / "etc/os-release").write_text((HOST / "os-release").read_text())
    (root / "venv/bin").mkdir(parents=True, exist_ok=True)
    (root / "venv/bin/rocm-sdk").write_text("#!/venv/bin/python3\nimport sys\n")
    pkgs = {
        "_rocm_sdk_core": f"{SITE}/_rocm_sdk_core",
        "_rocm_sdk_devel": f"{SITE}/_rocm_sdk_devel",
        "_rocm_sdk_libraries_gfx120X_all": f"{SITE}/_rocm_sdk_libraries_gfx120X_all",
    }
    for rel, files in (
        ("_rocm_sdk_core/lib", ["libamdhip64.so.7"]),
        ("_rocm_sdk_libraries_gfx120X_all/lib", ["librocblas.so.5", "libhipblas.so.3"]),
    ):
        d = root / SITE.lstrip("/") / rel
        d.mkdir(parents=True)
        for f in files:
            (d / f).write_text("")
    (root / SITE.lstrip("/") / "_rocm_sdk_devel/include/hipcub").mkdir(parents=True)
    runner = FakeRunner({"rocm-sdk": "/venv/bin/rocm-sdk"})
    runner.on("rocm-sdk version", version + "\n" if version else "", rc=0 if version else 1)
    runner.on(
        lambda a: a[0] == "/venv/bin/python3" and "_rocm_sdk_" in a[-1],
        lambda a, kw: CommandResult(a, 0 if probe else 1, stdout=json.dumps(pkgs) if probe else ""),
    )
    return HostContext(runner=runner, root=root, env={}, system="Linux", timeout=5)


def test_rocm_from_pip_packages_is_found_with_rocm_sdk(tmp_path: Path) -> None:
    ctx = pip_rocm_ctx(tmp_path)
    sw = collect_software(ctx, torch=False)
    assert sw.rocm_version == "10.2.0"
    assert sw.rocm_source == "rocm-sdk"
    assert sw.rocm_path == f"{SITE}/_rocm_sdk_devel"
    assert {"HIP runtime", "rocBLAS", "hipBLAS", "hipCUB"} <= set(sw.rocm_libraries)
    assert installed_libraries(ctx) == sw.rocm_libraries
    # Read-only: `rocm-sdk version` and a package listing, never `rocm-sdk path` or `init`.
    assert isinstance(ctx.runner, FakeRunner)
    assert [c for c in ctx.runner.calls if c[0] == "rocm-sdk"] == [["rocm-sdk", "version"]] * 2
    report = run_diagnosis(ctx, DiagnoseOptions(runtime=False, torch=False))
    rocm = next(c for c in report.checks if c.id == "software.rocm")
    assert rocm.status is CheckStatus.PASS and "found with rocm-sdk" in rocm.summary, rocm.summary


def test_rocm_sdk_without_a_package_listing_still_gives_the_version(tmp_path: Path) -> None:
    sw = collect_software(pip_rocm_ctx(tmp_path, probe=False), torch=False)
    assert (sw.rocm_version, sw.rocm_source, sw.rocm_path, sw.rocm_libraries) == (
        "10.2.0",
        "rocm-sdk",
        None,
        [],
    )


def test_rocm_sdk_whose_version_fails_is_a_warning_not_a_pass(tmp_path: Path) -> None:
    report = run_diagnosis(pip_rocm_ctx(tmp_path, version=""), DiagnoseOptions(runtime=False, torch=False))
    rocm = next(c for c in report.checks if c.id == "software.rocm")
    assert rocm.status is CheckStatus.WARNING and "version could not be read" in rocm.summary, rocm.summary


def test_rocm_path_beats_rocm_sdk_and_rocm_sdk_beats_opt_rocm(tmp_path: Path) -> None:
    ctx = pip_rocm_ctx(tmp_path)
    (tmp_path / "opt/rocm/.info").mkdir(parents=True)
    (tmp_path / "opt/rocm/.info/version").write_text("7.14.1\n")
    inst = locate_rocm(ctx)
    assert inst is not None and inst.source == "rocm-sdk"
    ctx.env["ROCM_PATH"] = "/opt/rocm"
    inst = locate_rocm(ctx)
    assert inst is not None and (inst.source, inst.version) == ("ROCM_PATH", "7.14.1")
    ctx.env["ROCM_PATH"] = "/does/not/exist"  # a stale variable falls through to what is installed
    inst = locate_rocm(ctx)
    assert inst is not None and inst.source == "rocm-sdk"


@pytest.mark.parametrize(
    ("script", "python"),
    [
        ("#!/venv/bin/python3\n", "/venv/bin/python3"),
        ("#!/usr/bin/env python3.12\n", "python3.12"),
        (PIP_LONG_PATH_SCRIPT, "/very/long/venv/bin/python3"),
        ("#!/bin/bash\necho hi\n", None),
    ],
)
def test_console_script_interpreter(tmp_path: Path, script: str, python: str | None) -> None:
    (tmp_path / "rocm-sdk").write_text(script)
    ctx = HostContext(runner=FakeRunner({}), root=tmp_path, env={}, system="Linux")
    assert _shebang_python(ctx, "/rocm-sdk") == python


def test_missing_tools_are_reported_absent_not_guessed(empty_linux_ctx: HostContext) -> None:
    sw = collect_software(empty_linux_ctx)
    assert sw.rocm_version is None
    assert sw.hip_runtime_version is None
    assert sw.hip_compiler_version is None
    assert sw.rocm_libraries == []
    assert all(not t.available and t.version is None for t in detect_tools(empty_linux_ctx))


def test_collect_host_on_linux_root(mi300x_ctx: HostContext) -> None:
    host = collect_host(mi300x_ctx)
    assert host.os_id == "ubuntu" and host.os_version == "24.04"
    assert host.kernel == "6.8.0-48-generic"
    assert host.hostname_hash and len(host.hostname_hash) == 12


# --------------------------------------------------------------------------- runtime


def test_probe_output_parsing() -> None:
    steps, devices = parse_probe_output("STEP init OK x\nDEVICE 0 gfx942|MI300X|1\nSTEP kernel FAIL bad")
    assert steps == {"init": (True, "x"), "kernel": (False, "bad")}
    assert devices == ["0 gfx942|MI300X|1"]


def test_hip_probe_success_is_executed_evidence(mi300x_ctx: HostContext) -> None:
    checks = run_hip_probe(mi300x_ctx, gfx_targets=["gfx942"])
    assert len(checks) == 7
    assert all(c.status is CheckStatus.PASS and c.evidence is Evidence.EXECUTED for c in checks)
    compile_call = next(c for c in mi300x_ctx.runner.calls if c[0] == "hipcc" and "-o" in c)  # type: ignore[attr-defined]
    assert "--offload-arch=gfx942" in compile_call


def test_hip_probe_kernel_failure_skips_later_steps(tmp_path: Path) -> None:
    probe = "STEP init OK\nSTEP enumerate OK devices=1\nSTEP alloc OK\nSTEP h2d OK\nSTEP kernel FAIL invalid device function"
    ctx = HostContext(runner=mi300x_runner(probe=probe), root=tmp_path, system="Linux")
    by_id = {c.id: c for c in run_hip_probe(ctx, gfx_targets=["gfx942"])}
    assert by_id["runtime.kernel_exec"].status is CheckStatus.FAIL
    assert "offload-arch" in (by_id["runtime.kernel_exec"].remediation or "")
    assert by_id["runtime.memcpy_d2h"].status is CheckStatus.SKIPPED
    assert by_id["runtime.vector_add_verify"].status is CheckStatus.SKIPPED


def test_hip_probe_without_hipcc_never_passes(empty_linux_ctx: HostContext) -> None:
    checks = run_hip_probe(empty_linux_ctx)
    assert {c.status for c in checks} == {CheckStatus.SKIPPED}
    assert all(c.evidence is Evidence.NONE for c in checks)


def test_hip_probe_compile_failure_is_skipped_with_reason(tmp_path: Path) -> None:
    r = mi300x_runner()
    r.on(lambda a: a[0] == "hipcc" and "-o" in a, CommandResult(["hipcc"], 1, stderr="error: no such file"))
    checks = run_hip_probe(HostContext(runner=r, root=tmp_path, system="Linux"))
    assert {c.status for c in checks} == {CheckStatus.SKIPPED}
    assert "did not compile" in checks[0].summary


def test_torch_probe_cuda_build_is_not_counted(mi300x_ctx: HostContext) -> None:
    info = TorchInfo(installed=True, version="2.5.0+cu124", cuda_version="12.4")
    checks = run_torch_probe(mi300x_ctx, info)
    assert {c.status for c in checks} == {CheckStatus.SKIPPED}
    assert "not a ROCm build" in checks[0].summary


def test_torch_probe_gpu_unavailable(tmp_path: Path) -> None:
    r = mi300x_runner(torch_runtime={"steps": {"available": {"ok": False, "detail": "device_count=0"}}})
    ctx = HostContext(runner=r, root=tmp_path, system="Linux")
    checks = run_torch_probe(ctx, TorchInfo(installed=True, version="2.13.0", hip_version="10.1"))
    assert checks[0].status is CheckStatus.FAIL
    assert checks[1].status is CheckStatus.SKIPPED


# --------------------------------------------------------------------------- engine


def test_full_diagnosis_on_healthy_mi300x(mi300x_ctx: HostContext) -> None:
    report = run_diagnosis(mi300x_ctx, DiagnoseOptions())
    bad = [(c.id, c.status, c.summary) for c in report.all_checks() if c.status is not CheckStatus.PASS]
    assert bad == []
    assert report.status is OverallStatus.HEALTHY
    assert report.recommendations == []


def test_diagnosis_without_runtime_is_not_healthy(mi300x_ctx: HostContext) -> None:
    """Disabled runtime checks are absent; the skipped PyTorch compatibility check blocks HEALTHY."""
    report = run_diagnosis(mi300x_ctx, DiagnoseOptions(runtime=False, torch=False))
    assert not any(c.category == "runtime" for c in report.checks)
    assert report.status is OverallStatus.REVIEW_REQUIRED  # pytorch compat SKIPPED


def test_empty_host_is_unhealthy_and_fabricates_nothing(empty_linux_ctx: HostContext) -> None:
    report = run_diagnosis(empty_linux_ctx, DiagnoseOptions())
    assert report.status is OverallStatus.UNHEALTHY
    assert report.gpus == []
    assert report.software.rocm_version is None
    runtime = [c for c in report.checks if c.category == "runtime"]
    assert runtime and all(c.status is CheckStatus.SKIPPED for c in runtime)
    assert not any(
        c.evidence is Evidence.EXECUTED and c.status is CheckStatus.PASS for c in report.all_checks()
    )
    ids = {c.id: c.status for c in report.checks}
    assert ids["hardware.gpu_detected"] is CheckStatus.FAIL
    assert ids["software.rocm"] is CheckStatus.FAIL
    assert report.recommendations


def test_static_checks_flag_device_access(mi300x_ctx: HostContext) -> None:
    from kernelvet.diagnose.hardware import HardwareProbe

    host = collect_host(mi300x_ctx)
    sw = collect_software(mi300x_ctx, torch=False)
    checks = {c.id: c for c in static_checks(host, HardwareProbe(gpus=[], kfd_accessible=False), sw)}
    assert checks["hardware.device_access"].status is CheckStatus.FAIL
    assert "render" in (checks["hardware.device_access"].remediation or "")


def test_progress_callback_reaches_completion(mi300x_ctx: HostContext) -> None:
    seen: list[float] = []
    run_diagnosis(mi300x_ctx, DiagnoseOptions(torch=False), lambda _m, f: seen.append(f))
    assert seen[-1] == 1.0 and seen == sorted(seen)


def test_skipped_check_never_carries_evidence() -> None:
    from kernelvet.models.common import CheckResult

    for ev in Evidence:
        c = CheckResult(id="x", name="x", category="x", status=CheckStatus.SKIPPED, evidence=ev)
        assert c.evidence == Evidence.NONE
    passed = CheckResult(id="x", name="x", category="x", status=CheckStatus.PASS, evidence=Evidence.DETECTED)
    assert passed.evidence == Evidence.DETECTED


def test_skipped_checks_report_no_evidence_on_non_linux_host(tmp_path: Path) -> None:
    ctx = HostContext(runner=FakeRunner({}), root=tmp_path, env={}, system="Darwin")
    report = run_diagnosis(ctx, DiagnoseOptions(torch=False))
    skipped = [c for c in report.all_checks() if c.status == CheckStatus.SKIPPED]
    assert any(c.id == "hardware.device_access" for c in skipped)
    assert any(c.category == "compatibility" for c in skipped)
    assert all(c.evidence == Evidence.NONE for c in skipped)
