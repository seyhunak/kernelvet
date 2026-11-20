from __future__ import annotations

from pathlib import Path

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
from kernelvet.diagnose.software import collect_software, detect_tools
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
    assert sw.rocm_version == "10.1.0-25401"
    assert sw.hip_runtime_version == "10.1.25401-a1b2c3d4"
    assert sw.hip_compiler_version.startswith("AMD clang version 21")
    assert sw.amdgpu_driver_version == "31.60.0"
    assert sw.amdgpu_module_loaded is True
    assert {"HIP runtime", "rocBLAS", "hipBLAS", "RCCL", "MIOpen", "hipCUB"} <= set(sw.rocm_libraries)
    assert sw.torch is not None and sw.torch.is_rocm_build
    assert sw.has("hipcc") and not sw.has("hipify-perl")


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
