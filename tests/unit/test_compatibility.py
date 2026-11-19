from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernelvet.compatibility.engine import evaluate
from kernelvet.compatibility.matrix import load_matrix
from kernelvet.compatibility.validators import (
    CompatibilityInput,
    validate_driver,
    validate_gpu,
    validate_hip,
    validate_kernel,
    validate_os,
    validate_pytorch,
    validate_rocm,
)
from kernelvet.models.common import CheckStatus
from kernelvet.models.system import GpuDevice, HostInfo, SoftwareInfo, ToolInfo, TorchInfo

M = load_matrix()


def host(
    os_id: str = "ubuntu", ver: str = "24.04", kernel: str = "6.8.0-48-generic", name: str = "Ubuntu"
) -> HostInfo:
    return HostInfo(
        os_name=name,
        os_id=os_id,
        os_version=ver,
        os_pretty=f"{name} {ver}",
        kernel=kernel,
        architecture="x86_64",
        python_version="3.12",
    )


def gpu(gfx: str | None = "gfx942", name: str = "AMD Instinct MI300X") -> GpuDevice:
    return GpuDevice(index=0, vendor_is_amd=True, gfx_target=gfx, marketing_name=name)


def sw(
    rocm: str | None = "10.1.0",
    hip: str | None = "10.1.25401",
    driver: str | None = "31.60.0",
    torch: TorchInfo | None = None,
) -> SoftwareInfo:
    return SoftwareInfo(
        rocm_version=rocm,
        hip_runtime_version=hip,
        amdgpu_driver_version=driver,
        amdgpu_module_loaded=True,
        tools=[ToolInfo(name="hipcc", available=True)],
        torch=torch,
    )


def inp(
    h: HostInfo | None = None, g: list[GpuDevice] | None = None, s: SoftwareInfo | None = None
) -> CompatibilityInput:
    return CompatibilityInput(host=h or host(), gpus=[gpu()] if g is None else g, software=s or sw())


def test_matrix_records_provenance() -> None:
    assert M.rocm_version == "10.1.0"
    assert all(src.url.startswith("https://rocm.docs.amd.com/") for src in M.sources.values())
    assert M.retrieved
    assert M.notes
    assert all(e.source in M.sources for e in M.gpus)


@pytest.mark.parametrize(
    "gfx,expected",
    [
        ("gfx942", CheckStatus.PASS),
        ("gfx950", CheckStatus.PASS),
        ("gfx1100", CheckStatus.PASS),
        ("gfx906", CheckStatus.FAIL),  # listed as unsupported
        ("gfx1151", CheckStatus.WARNING),  # ambiguous in the published table
        ("gfx803", CheckStatus.WARNING),  # not listed
        (None, CheckStatus.WARNING),  # not detected
    ],
)
def test_gpu_support(gfx: str | None, expected: CheckStatus) -> None:
    assert validate_gpu(inp(g=[gpu(gfx)]), M).status is expected


def test_gpu_support_handles_feature_suffix() -> None:
    assert validate_gpu(inp(g=[gpu("gfx942:sramecc+:xnack-")]), M).status is CheckStatus.PASS


def test_no_gpu_means_skipped_not_pass() -> None:
    assert validate_gpu(inp(g=[]), M).status is CheckStatus.SKIPPED
    assert validate_driver(inp(g=[]), M).status is CheckStatus.SKIPPED


def test_os_support() -> None:
    assert validate_os(inp(), M).status is CheckStatus.PASS
    assert validate_os(inp(h=host("rhel", "9.6", "5.14.0-570.el9")), M).status is CheckStatus.PASS
    assert validate_os(inp(h=host("rocky", "9.5", "5.14.0-570")), M).status is CheckStatus.PASS
    assert validate_os(inp(h=host("fedora", "41")), M).status is CheckStatus.WARNING
    mac = HostInfo(os_name="macOS", architecture="arm64", python_version="3.12")
    assert validate_os(inp(h=mac), M).status is CheckStatus.FAIL


def test_os_restricted_to_specific_gpus() -> None:
    # RHEL 8.10 lists Instinct only; a Radeon RX 7900 XTX there is not supported
    r = validate_os(inp(h=host("rhel", "8.10", "4.18.0-553"), g=[gpu("gfx1100", "RX 7900 XTX")]), M)
    assert r.status is CheckStatus.WARNING
    assert validate_os(inp(h=host("rhel", "8.10", "4.18.0-553")), M).status is CheckStatus.PASS


def test_kernel() -> None:
    assert validate_kernel(inp(), M).status is CheckStatus.PASS
    assert validate_kernel(inp(h=host(kernel="6.5.0-14-generic")), M).status is CheckStatus.WARNING
    assert validate_kernel(inp(h=host("fedora", "41")), M).status is CheckStatus.SKIPPED


def test_rocm_version() -> None:
    assert validate_rocm(inp(), M).status is CheckStatus.PASS
    assert validate_rocm(inp(s=sw(rocm="10.1.0-25401")), M).status is CheckStatus.PASS
    assert validate_rocm(inp(s=sw(rocm="7.14.0")), M).status is CheckStatus.WARNING
    assert validate_rocm(inp(s=sw(rocm="6.2.4")), M).status is CheckStatus.WARNING
    newer = validate_rocm(inp(s=sw(rocm="11.0.0")), M)
    assert newer.status is CheckStatus.WARNING and "newer" in newer.summary
    assert validate_rocm(inp(s=sw(rocm=None)), M).status is CheckStatus.SKIPPED


def test_driver() -> None:
    assert validate_driver(inp(), M).status is CheckStatus.PASS
    assert validate_driver(inp(s=sw(driver="6.10.5")), M).status is CheckStatus.WARNING
    assert validate_driver(inp(s=sw(driver=None)), M).status is CheckStatus.WARNING
    unloaded = sw()
    unloaded.amdgpu_module_loaded = False
    assert validate_driver(inp(s=unloaded), M).status is CheckStatus.FAIL


def test_hip_consistency() -> None:
    assert validate_hip(inp(), M).status is CheckStatus.PASS
    assert validate_hip(inp(s=sw(hip="7.14.1")), M).status is CheckStatus.WARNING


@pytest.mark.parametrize(
    "torch,expected",
    [
        (TorchInfo(installed=True, version="2.13.0+rocm10.1", hip_version="10.1.25401"), CheckStatus.PASS),
        (TorchInfo(installed=True, version="2.9.0+rocm10.1", hip_version="10.1.25401"), CheckStatus.WARNING),
        (TorchInfo(installed=True, version="2.13.0+rocm7.14", hip_version="7.14.1"), CheckStatus.WARNING),
        (TorchInfo(installed=True, version="2.13.0+cu128", cuda_version="12.8"), CheckStatus.WARNING),
        (TorchInfo(installed=False), CheckStatus.SKIPPED),
        (None, CheckStatus.SKIPPED),
    ],
)
def test_pytorch(torch: TorchInfo | None, expected: CheckStatus) -> None:
    assert validate_pytorch(inp(s=sw(torch=torch)), M).status is expected


def test_evaluate_reports_sources_and_status() -> None:
    torch = TorchInfo(installed=True, version="2.13.0", hip_version="10.1.25401")
    r = evaluate(host(), [gpu()], sw(torch=torch), M)
    assert r.status.value == "HEALTHY"
    assert r.matrix_version == "10.1.0" and r.matrix_sources
    assert all(c.category == "compatibility" for c in r.checks)


def test_custom_matrix_file(tmp_path: Path) -> None:
    data = json.loads(
        (Path(__file__).parents[2] / "src/kernelvet/compatibility/data/rocm_matrix.json").read_text()
    )
    data["rocm_version"] = "7.14.0"
    p = tmp_path / "m.json"
    p.write_text(json.dumps(data))
    m = load_matrix(p)
    assert validate_rocm(inp(s=sw(rocm="7.14.0")), m).status is CheckStatus.PASS


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{not json", "is not valid JSON (line 1"),
        ('{"schema": 1}', "does not match the matrix schema: rocm_version: Field required"),
        ("[]", "does not match the matrix schema"),
    ],
)
def test_bad_matrix_file_raises_matrix_error(tmp_path: Path, content: str, message: str) -> None:
    from kernelvet.compatibility.matrix import MatrixError

    path = tmp_path / "matrix.json"
    path.write_text(content)
    with pytest.raises(MatrixError) as info:
        load_matrix(path)
    assert message in str(info.value) and str(path) in str(info.value)
    with pytest.raises(MatrixError, match="could not read"):
        load_matrix(tmp_path / "missing.json")


def test_compatibility_report_serializes_its_status() -> None:
    report = evaluate(
        HostInfo(os_name="Linux", architecture="x86_64", python_version="3.12"),
        [],
        SoftwareInfo(),
        load_matrix(),
    )
    data = json.loads(report.model_dump_json())
    assert data["kind"] == "compatibility" and data["meta"]["tool"] == "kernelvet"
    assert data["status"] == report.status.value
