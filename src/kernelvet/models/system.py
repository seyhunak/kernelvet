"""Hardware and software inventory models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from kernelvet.models.common import Evidence


class HostInfo(BaseModel):
    os_name: str
    os_id: str | None = None  # e.g. "ubuntu", "rhel"
    os_version: str | None = None  # e.g. "24.04"
    os_pretty: str | None = None  # e.g. "Ubuntu 24.04.1 LTS"
    kernel: str | None = None
    architecture: str
    hostname_hash: str | None = Field(
        default=None, description="Short hash of the hostname; the raw hostname is never stored"
    )
    python_version: str
    in_container: bool = False


class GpuDevice(BaseModel):
    index: int
    vendor: str | None = None
    vendor_is_amd: bool = False
    marketing_name: str | None = None
    gfx_target: str | None = None  # e.g. "gfx942"
    compute_units: int | None = None
    vram_bytes: int | None = None
    pci_bus: str | None = None
    device_id: str | None = None
    uuid: str | None = None
    node_id: int | None = None
    accessible: bool | None = Field(
        default=None, description="Whether the current user can open the device nodes"
    )
    sources: list[str] = Field(default_factory=list, description="Tools that reported this GPU")
    evidence: Evidence = Evidence.DETECTED

    @property
    def vram_gib(self) -> float | None:
        return None if self.vram_bytes is None else self.vram_bytes / 1024**3


class ToolInfo(BaseModel):
    """A command-line tool or library and what we learned about it."""

    name: str
    available: bool
    path: str | None = None
    version: str | None = None
    detail: str | None = None


class TorchInfo(BaseModel):
    installed: bool
    version: str | None = None
    hip_version: str | None = None  # torch.version.hip, None for CUDA/CPU builds
    cuda_version: str | None = None
    gpu_available: bool | None = None
    device_count: int | None = None
    device_names: list[str] = Field(default_factory=list)
    python: str | None = None
    error: str | None = None

    @property
    def is_rocm_build(self) -> bool:
        return bool(self.hip_version)


class SoftwareInfo(BaseModel):
    rocm_version: str | None = None
    rocm_path: str | None = None
    # how ROCm was found: "ROCM_PATH", "HIP_PATH", "/opt/rocm" or "rocm-sdk" (TheRock Python packages)
    rocm_source: str | None = None
    hip_runtime_version: str | None = None
    hip_compiler_version: str | None = None
    amdgpu_driver_version: str | None = None
    amdgpu_module_loaded: bool | None = None
    tools: list[ToolInfo] = Field(default_factory=list)
    rocm_libraries: list[str] = Field(default_factory=list)
    torch: TorchInfo | None = None

    def tool(self, name: str) -> ToolInfo | None:
        return next((t for t in self.tools if t.name == name), None)

    def has(self, name: str) -> bool:
        t = self.tool(name)
        return bool(t and t.available)
