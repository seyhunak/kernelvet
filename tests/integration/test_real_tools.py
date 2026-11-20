"""Tests against real AMD tooling. Skipped automatically when the tools/hardware are absent.

Run on a self-hosted AMD GPU runner with:  pytest -m "gpu or hipify"
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from conftest import tree_digest
from kernelvet.diagnose.engine import DiagnoseOptions, run_diagnosis
from kernelvet.diagnose.hardware import discover_gpus
from kernelvet.migration import audit_project, convert_project, scan_project
from kernelvet.models.common import CheckStatus, Evidence
from kernelvet.utils.proc import HostContext


def _has_amd_gpu() -> bool:
    if not shutil.which("rocminfo"):
        return False
    return any(g.vendor_is_amd for g in discover_gpus(HostContext()).gpus)


@pytest.mark.hipify
@pytest.mark.skipif(not shutil.which("hipify-perl"), reason="hipify-perl not installed")
def test_real_hipify_perl_conversion(cuda_project: Path, tmp_path: Path) -> None:
    before = tree_digest(cuda_project)
    res = convert_project(
        HostContext(), scan_project(cuda_project), tmp_path / "out", preference="hipify-perl"
    )
    assert tree_digest(cuda_project) == before
    converted = (tmp_path / "out/src/vector_add.cu").read_text()
    assert "hipMalloc" in converted and "cudaMalloc" not in converted
    assert not res.failed
    audit = audit_project(tmp_path / "out")
    assert any(f.rule_id == "cuda.inline_ptx" for f in audit.findings)  # PTX survives HIPIFY


@pytest.mark.gpu
@pytest.mark.skipif(not _has_amd_gpu(), reason="no AMD GPU with ROCm on this host")
def test_real_gpu_diagnosis_executes_probe() -> None:
    report = run_diagnosis(HostContext(), DiagnoseOptions(torch=False))
    runtime = {c.id: c for c in report.checks if c.category == "runtime"}
    assert runtime["runtime.kernel_exec"].evidence is Evidence.EXECUTED
    assert runtime["runtime.vector_add_verify"].status is CheckStatus.PASS
