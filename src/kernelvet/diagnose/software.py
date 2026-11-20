"""ROCm, HIP, driver, library, tool and PyTorch discovery.

Static detection only: nothing here runs GPU code. Runtime execution lives in
:mod:`kernelvet.diagnose.runtime`.
"""

from __future__ import annotations

import json
import posixpath
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field

from kernelvet.models.system import SoftwareInfo, ToolInfo, TorchInfo
from kernelvet.utils.proc import HostContext
from kernelvet.utils.versions import version_str

# (tool, version argv or None)
TOOLS: list[tuple[str, list[str] | None]] = [
    ("rocminfo", None),
    ("amd-smi", ["amd-smi", "version"]),
    ("rocm-smi", None),
    ("rocm-sdk", None),
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
    # ROCm 10.x (TheRock) layout: /opt/rocm/core-10.2/.info/version, no /opt/rocm/.info
    cores = sorted(
        (
            (tuple(int(p) for p in m.group(1).split(".")), name, m.group(1))
            for name in ctx.listdir(rocm_path)
            if (m := re.fullmatch(r"core-(\d+\.\d+(?:\.\d+)?)", name))
        ),
        reverse=True,
    )
    if cores:
        _, name, from_dir = cores[0]
        text = ctx.read_text(f"{rocm_path}/{name}/.info/version")
        return text.strip().splitlines()[0] if text and text.strip() else from_dir
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


def detect_libraries(ctx: HostContext, rocm_path: str | Sequence[str]) -> list[str]:
    roots = [rocm_path] if isinstance(rocm_path, str) else list(rocm_path)
    found: list[str] = []
    names = {n for r in roots for d in (f"{r}/lib", f"{r}/lib64") for n in ctx.listdir(d)}
    for family, stem in ROCM_LIBRARIES.items():
        if any(n == f"{stem}.so" or n.startswith(f"{stem}.so") for n in names):
            found.append(family)
    for family, rel in HEADER_LIBRARIES.items():
        if any(ctx.exists(f"{r}/{rel}") for r in roots):
            found.append(family)
    return found


# Lists the TheRock Python packages (_rocm_sdk_core, _rocm_sdk_devel, _rocm_sdk_libraries_<family>)
# without importing them. `rocm-sdk path --root` would also give the devel root, but it needs
# rocm[devel] and creates the devel links on first use, and diagnostics never write to the host.
ROCM_SDK_PROBE = r"""
import json, os, pkgutil
out = {}
for m in pkgutil.iter_modules():
    base = getattr(m.module_finder, "path", None)
    if m.ispkg and base and m.name.startswith("_rocm_sdk_") and m.name not in out:
        out[m.name] = os.path.join(base, m.name)
print(json.dumps(out))
"""


@dataclass
class RocmInstall:
    """Where ROCm was found. ``source`` says how: ROCM_PATH, HIP_PATH, /opt/rocm or rocm-sdk."""

    source: str
    path: str | None = None
    version: str | None = None
    roots: list[str] = field(default_factory=list)  # directories holding lib/ and include/


def _shebang_python(ctx: HostContext, script: str) -> str | None:
    """The interpreter a pip console script runs with (plain or pip's long-path /bin/sh form)."""
    head = (ctx.read_text(script, limit=1024) or "").splitlines()
    if not head or not head[0].startswith("#!"):
        return None
    words = head[0][2:].split()
    if words and posixpath.basename(words[0]) == "env":
        words = words[1:]
    if words and posixpath.basename(words[0]).startswith("python"):
        return words[0]
    # pip's form for long paths: #!/bin/sh, then '''exec' "/long/venv/bin/python3" "$0" "$@"
    for line in head[1:3]:
        m = re.search(r"exec' \"([^\"]+)\"", line)
        if m:
            return m.group(1)
    return None


def _rocm_sdk_install(ctx: HostContext) -> RocmInstall | None:
    """ROCm installed as TheRock Python packages (``pip install rocm[...]``), found via rocm-sdk."""
    script = ctx.which("rocm-sdk")
    if not script:
        return None
    res = ctx.run(["rocm-sdk", "version"], timeout=min(ctx.timeout, 30))
    version = _first_line(res.stdout) if res.ok else ""
    inst = RocmInstall(source="rocm-sdk", version=version if version_str(version) else None)
    python = _shebang_python(ctx, script)
    if python:
        probe = ctx.run([python, "-c", ROCM_SDK_PROBE], timeout=min(ctx.timeout, 30))
        try:
            pkgs = json.loads(probe.stdout.strip().splitlines()[-1]) if probe.ok else {}
        except (json.JSONDecodeError, IndexError):
            pkgs = {}
        if isinstance(pkgs, dict):
            order = ["_rocm_sdk_devel", "_rocm_sdk_core"]
            names = order + sorted(n for n in pkgs if n.startswith("_rocm_sdk_libraries"))
            inst.roots = [str(pkgs[n]) for n in names if isinstance(pkgs.get(n), str)]
            inst.path = next((str(pkgs[n]) for n in order if isinstance(pkgs.get(n), str)), None)
    return inst


def locate_rocm(ctx: HostContext) -> RocmInstall | None:
    """Find the ROCm install the tools on this host would use, or ``None``.

    An explicit ``ROCM_PATH`` / ``HIP_PATH`` wins. Then ``rocm-sdk`` on PATH: an activated
    virtual environment with TheRock's Python packages puts its hipcc first on PATH, so it is
    the install in use. Then ``/opt/rocm``.
    """
    for var in ("ROCM_PATH", "HIP_PATH"):
        p = ctx.env.get(var)
        if p and ctx.exists(p):
            return RocmInstall(source=var, path=p, version=detect_rocm_version(ctx, p), roots=[p])
    sdk = _rocm_sdk_install(ctx)
    if sdk is not None:
        return sdk
    if ctx.exists("/opt/rocm"):
        return RocmInstall(
            source="/opt/rocm",
            path="/opt/rocm",
            version=detect_rocm_version(ctx, "/opt/rocm"),
            roots=["/opt/rocm"],
        )
    return None


def installed_libraries(ctx: HostContext) -> list[str] | None:
    """ROCm library families present, or ``None`` when ROCm itself was not found."""
    inst = locate_rocm(ctx)
    if inst is None or not inst.roots:
        return None
    return detect_libraries(ctx, inst.roots)


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
    rocm = locate_rocm(ctx)
    tools = detect_tools(ctx)
    runtime, compiler = detect_hip_compiler(ctx, tools)
    driver, loaded = detect_driver(ctx)
    return SoftwareInfo(
        rocm_version=rocm.version if rocm else None,
        rocm_path=rocm.path if rocm else None,
        rocm_source=rocm.source if rocm else None,
        hip_runtime_version=runtime,
        hip_compiler_version=compiler,
        amdgpu_driver_version=driver,
        amdgpu_module_loaded=loaded,
        tools=tools,
        rocm_libraries=detect_libraries(ctx, rocm.roots) if rocm and rocm.roots else [],
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
