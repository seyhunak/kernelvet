"""ROCm, HIP, driver, library, tool and PyTorch discovery.

Static detection only: nothing here runs GPU code. Runtime execution lives in
:mod:`kernelvet.diagnose.runtime`.
"""

from __future__ import annotations

import json
import re
import sys

from kernelvet.models.system import SoftwareInfo, ToolInfo, TorchInfo
from kernelvet.utils.proc import HostContext
from kernelvet.utils.versions import version_str

# (tool, version argv or None)
TOOLS: list[tuple[str, list[str] | None]] = [
    ("rocminfo", None),
    ("amd-smi", ["amd-smi", "version"]),
    ("rocm-smi", None),
    ("hipcc", ["hipcc", "--version"]),
    ("hipconfig", ["hipconfig", "--version"]),
    ("hipify-clang", ["hipify-clang", "--version"]),
    ("hipify-perl", ["hipify-perl", "--version"]),
    ("rocprofv3", ["rocprofv3", "--version"]),
    ("rocprof", None),
    ("cmake", ["cmake", "--version"]),
    ("make", ["make", "--version"]),
    ("docker", ["docker", "--version"]),
]

# library family -> shared-object stem
ROCM_LIBRARIES: dict[str, str] = {
    "HIP runtime": "libamdhip64",
    "rocBLAS": "librocblas",
    "hipBLAS": "libhipblas",
    "hipBLASLt": "libhipblaslt",
    "MIOpen": "libMIOpen",
    "RCCL": "librccl",
    "rocFFT": "librocfft",
    "hipFFT": "libhipfft",
    "rocRAND": "librocrand",
    "hipRAND": "libhiprand",
    "rocSPARSE": "librocsparse",
    "hipSPARSE": "libhipsparse",
    "rocSOLVER": "librocsolver",
    "hipSOLVER": "libhipsolver",
    "MIGraphX": "libmigraphx",
    "rocprofiler-sdk": "librocprofiler-sdk",
    "roctracer": "libroctracer64",
}

HEADER_LIBRARIES: dict[str, str] = {
    "rocThrust": "include/thrust/system/hip",
    "hipCUB": "include/hipcub",
    "rocPRIM": "include/rocprim",
}


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def detect_tools(ctx: HostContext) -> list[ToolInfo]:
    out: list[ToolInfo] = []
    for name, argv in TOOLS:
        path = ctx.which(name)
        if not path:
            out.append(ToolInfo(name=name, available=False))
            continue
        version = None
        detail = None
        if argv:
            res = ctx.run(argv, timeout=min(ctx.timeout, 20))
            text = res.stdout + "\n" + res.stderr
            if res.ok or text.strip():
                version = _tool_version(name, text)
                detail = _first_line(text)[:200] or None
            else:
                detail = res.describe_failure()
        out.append(ToolInfo(name=name, available=True, path=path, version=version, detail=detail))
    return out


def _tool_version(name: str, text: str) -> str | None:
    if name == "hipcc":
        m = re.search(r"HIP version:\s*(\S+)", text)
        return m.group(1) if m else version_str(text)
    if name == "hipconfig":
        first = _first_line(text).split()
        return first[0] if first and version_str(first[0]) else version_str(text)
    if name == "amd-smi":
        m = re.search(r"amdsmi Library version:\s*([\d.]+)", text) or re.search(
            r"AMDSMI Tool:\s*([\d.+\w-]+)", text
        )
        return m.group(1) if m else version_str(text)
    return version_str(text)


def detect_rocm_version(ctx: HostContext, rocm_path: str) -> str | None:
    for rel in (".info/version", ".info/version-dev", ".info/version-utils"):
        text = ctx.read_text(f"{rocm_path}/{rel}")
        if text and text.strip():
            return text.strip().splitlines()[0]
    # /opt/rocm-6.2.0 style install directories
    for name in reversed(ctx.listdir("/opt")):
        m = re.fullmatch(r"rocm-(\d+\.\d+(?:\.\d+)?)", name)
        if m:
            return m.group(1)
    return None


def detect_hip_compiler(ctx: HostContext, tools: list[ToolInfo]) -> tuple[str | None, str | None]:
    """Return (HIP runtime version, compiler description) from hipconfig/hipcc."""
    runtime = None
    compiler = None
    hipconfig = next((t for t in tools if t.name == "hipconfig" and t.available), None)
    if hipconfig and hipconfig.version:
        runtime = hipconfig.version
    hipcc = next((t for t in tools if t.name == "hipcc" and t.available), None)
    if hipcc:
        res = ctx.run(["hipcc", "--version"], timeout=min(ctx.timeout, 20))
        text = res.stdout + res.stderr
        m = re.search(r"HIP version:\s*(\S+)", text)
        if m and not runtime:
            runtime = m.group(1)
        clang = re.search(r"(AMD clang version [^\n]+|clang version [^\n]+)", text)
        compiler = clang.group(1).strip() if clang else hipcc.version
    return runtime, compiler


def detect_driver(ctx: HostContext) -> tuple[str | None, bool | None]:
    if ctx.system != "Linux":
        return None, None
    loaded = ctx.exists("/sys/module/amdgpu")
    version = (ctx.read_text("/sys/module/amdgpu/version") or "").strip() or None
    return version, loaded


def detect_libraries(ctx: HostContext, rocm_path: str) -> list[str]:
    found: list[str] = []
    libdirs = [f"{rocm_path}/lib", f"{rocm_path}/lib64"]
    names = {n for d in libdirs for n in ctx.listdir(d)}
    for family, stem in ROCM_LIBRARIES.items():
        if any(n == f"{stem}.so" or n.startswith(f"{stem}.so") for n in names):
            found.append(family)
    for family, rel in HEADER_LIBRARIES.items():
        if ctx.exists(f"{rocm_path}/{rel}"):
            found.append(family)
    return found


TORCH_PROBE = r"""
import json, sys
out = {"installed": False, "python": sys.version.split()[0]}
try:
    import torch
except Exception as exc:  # ImportError or a broken install
    out["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(out)); sys.exit(0)
out["installed"] = True
out["version"] = torch.__version__
out["hip_version"] = getattr(torch.version, "hip", None)
out["cuda_version"] = getattr(torch.version, "cuda", None)
print(json.dumps(out))
"""


def detect_torch(ctx: HostContext, python: str | None = None) -> TorchInfo:
    """Inspect PyTorch in a child interpreter so a broken install can't crash us.

    Only reads version metadata; does not initialise the GPU runtime.
    """
    exe = python or sys.executable
    res = ctx.run([exe, "-c", TORCH_PROBE], timeout=max(ctx.timeout, 60))
    if not res.ok:
        return TorchInfo(installed=False, error=res.describe_failure())
    try:
        data = json.loads(res.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return TorchInfo(installed=False, error="unexpected output from PyTorch probe")
    return TorchInfo(**data)


def collect_software(ctx: HostContext, *, python: str | None = None, torch: bool = True) -> SoftwareInfo:
    rocm_path = ctx.rocm_path()
    rocm_present = ctx.exists(rocm_path)
    tools = detect_tools(ctx)
    runtime, compiler = detect_hip_compiler(ctx, tools)
    driver, loaded = detect_driver(ctx)
    return SoftwareInfo(
        rocm_version=detect_rocm_version(ctx, rocm_path) if rocm_present else None,
        rocm_path=rocm_path if rocm_present else None,
        hip_runtime_version=runtime,
        hip_compiler_version=compiler,
        amdgpu_driver_version=driver,
        amdgpu_module_loaded=loaded,
        tools=tools,
        rocm_libraries=detect_libraries(ctx, rocm_path) if rocm_present else [],
        torch=detect_torch(ctx, python) if torch else None,
    )


def torch_summary(t: TorchInfo | None) -> str:
    if t is None:
        return "not checked"
    if not t.installed:
        return "not installed"
    if t.hip_version:
        return f"{t.version} (ROCm/HIP {t.hip_version})"
    if t.cuda_version:
        return f"{t.version} (CUDA {t.cuda_version} build, not ROCm)"
    return f"{t.version} (CPU-only build)"


def dump(info: SoftwareInfo) -> str:  # pragma: no cover - debugging helper
    return json.dumps(info.model_dump(), indent=2)
