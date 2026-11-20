"""Evaluate a detected system against the compatibility matrix."""

from __future__ import annotations

from kernelvet.compatibility.matrix import CompatibilityMatrix, load_matrix
from kernelvet.compatibility.validators import VALIDATORS, CompatibilityInput
from kernelvet.models.report import CompatibilityReport
from kernelvet.models.system import GpuDevice, HostInfo, SoftwareInfo


def evaluate(
    host: HostInfo,
    gpus: list[GpuDevice],
    software: SoftwareInfo,
    matrix: CompatibilityMatrix | None = None,
) -> CompatibilityReport:
    m = matrix or load_matrix()
    inp = CompatibilityInput(host=host, gpus=gpus, software=software)
    return CompatibilityReport(
        matrix_version=m.rocm_version,
        matrix_sources=m.source_urls(),
        matrix_retrieved=m.retrieved,
        checks=[v(inp, m) for v in VALIDATORS],
    )
