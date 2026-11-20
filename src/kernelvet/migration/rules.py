"""CUDA construct rules used by the analyzer (pre-conversion) and auditor (post-conversion).

Each rule is a regex over comment-stripped source lines. Rules describe *what
HIPIFY and HIP do with a construct*; the translation itself is always left to
AMD HIPIFY. Where HIP behaviour differs from CUDA in ways a mechanical
translation can't see (wavefront size 64 on CDNA, 64-bit lane masks, managed
memory semantics), the rule says REQUIRES_VERIFICATION rather than pretending
the conversion is complete.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from kernelvet.models.migration import Finding, FindingCategory, Severity

AUTO = FindingCategory.AUTO_CONVERTIBLE
MANUAL = FindingCategory.MANUAL_INTERVENTION
UNSUP = FindingCategory.UNSUPPORTED
VERIFY = FindingCategory.REQUIRES_VERIFICATION


# CUDA names whose HIP equivalent is not a plain cuda -> hip prefix swap
# (as HIPIFY translates them). Checked before the generic rename.
HIP_RENAMES: dict[str, str] = {
    "cudaDeviceProp": "hipDeviceProp_t",
    "cudaThreadSynchronize": "hipDeviceSynchronize",
    "cudaThreadExit": "hipDeviceReset",
    "cudaThreadGetCacheConfig": "hipDeviceGetCacheConfig",
    "cudaThreadSetCacheConfig": "hipDeviceSetCacheConfig",
}


@dataclass(frozen=True)
class Rule:
    id: str
    pattern: re.Pattern[str]
    category: FindingCategory
    severity: Severity
    message: str
    hip: str | None = None
    suggestion: str | None = None
    phases: frozenset[str] = field(default_factory=lambda: frozenset({"analyze"}))
    targets: frozenset[str] = field(default_factory=lambda: frozenset({"source"}))  # source | build

    def hip_for(self, symbol: str) -> str | None:
        if self.hip and "{sym}" in self.hip:
            return self.hip.replace("{sym}", symbol)
        if self.hip == "cuda->hip":
            return HIP_RENAMES.get(symbol) or re.sub(r"^cuda", "hip", symbol)
        if self.hip and self.hip.startswith("prefix:"):
            old, new = self.hip[len("prefix:") :].split("->")
            return new + symbol[len(old) :] if symbol.startswith(old) else None
        return self.hip


def _r(
    id: str,
    pat: str,
    cat: FindingCategory,
    sev: Severity,
    msg: str,
    hip: str | None = None,
    sug: str | None = None,
    phases: tuple[str, ...] = ("analyze",),
    targets: tuple[str, ...] = ("source",),
    flags: int = 0,
) -> Rule:
    return Rule(id, re.compile(pat, flags), cat, sev, msg, hip, sug, frozenset(phases), frozenset(targets))


A, AU, BOTH = ("analyze",), ("audit",), ("analyze", "audit")
LOW, MED, HIGH, INFO = Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.INFO

RULES: list[Rule] = [
    # ---- runtime API: mechanical translation ---------------------------------------------
    _r(
        "cuda.runtime_api",
        r"\b(cuda(?!Graph|MallocManaged|LaunchCooperative|Ipc)[A-Z]\w*)\s*\(",
        AUTO,
        INFO,
        "CUDA runtime API call",
        "cuda->hip",
        "HIPIFY renames cuda* runtime calls to hip*",
    ),
    _r(
        "cuda.runtime_type",
        r"\b(cuda(?:Error_t|Stream_t|Event_t|DeviceProp|MemcpyKind|Array_t|"
        r"ChannelFormatDesc|Extent|PitchedPtr|Memcpy3DParms|StreamCaptureMode))\b",
        AUTO,
        INFO,
        "CUDA runtime type",
        "cuda->hip",
    ),
    _r(
        "cuda.include_runtime",
        r"#\s*include\s*[<\"](cuda_runtime(?:_api)?\.h|cuda\.h|device_launch_parameters\.h)[>\"]",
        AUTO,
        INFO,
        "CUDA runtime header",
        "hip/hip_runtime.h",
    ),
    _r(
        "cuda.kernel_launch",
        r"(<<<)[^;]*?>>>",
        AUTO,
        INFO,
        "Triple-chevron kernel launch",
        "<<<>>> (HIP) or hipLaunchKernelGGL",
        "HIP supports <<<>>>; hipify-perl may emit hipLaunchKernelGGL",
    ),
    _r(
        "cuda.fp16",
        r"#\s*include\s*[<\"](cuda_fp16\.h|cuda_bf16\.h)[>\"]",
        AUTO,
        INFO,
        "CUDA half/bfloat16 header",
        "hip/hip_fp16.h / hip/hip_bf16.h",
    ),
    _r(
        "cuda.driver_api",
        r"\b(cu(?:Init|Device\w+|Ctx\w+|Module\w+|Launch\w*|MemAlloc\w*|Memcpy\w*|"
        r"MemFree\w*|Stream\w+|Event\w+|Func\w+))\s*\(",
        AUTO,
        LOW,
        "CUDA driver API call",
        "prefix:cu->hip",
        "Driver API maps to hip* module/launch APIs",
    ),
    # ---- needs verification after translation -----------------------------------------------
    _r(
        "cuda.managed_memory",
        r"\b(cudaMallocManaged|hipMallocManaged)\s*\(",
        VERIFY,
        MED,
        "Managed (unified) memory",
        "hipMallocManaged",
        "Behaviour depends on GPU and XNACK/HMM support; verify performance and correctness on target GPUs",
        phases=BOTH,
    ),
    _r(
        "cuda.graphs",
        r"\b(cudaGraph\w*|cudaStreamBeginCapture)\s*\(",
        VERIFY,
        MED,
        "CUDA Graphs API",
        "cuda->hip",
        "hipGraph* exists but coverage differs; verify every graph API used",
    ),
    _r(
        "cuda.cooperative",
        r"\b(cudaLaunchCooperativeKernel\w*|cooperative_groups|this_grid\(\)\.sync)",
        VERIFY,
        MED,
        "Cooperative groups / cooperative launch",
        "hip/hip_cooperative_groups.h",
        "Grid-wide sync requires hardware/runtime support; verify on target GPU",
    ),
    _r(
        "cuda.ipc",
        r"\b(cudaIpc\w+)\s*\(",
        VERIFY,
        MED,
        "CUDA IPC",
        "cuda->hip",
        "Verify multi-process IPC on ROCm",
    ),
    _r(
        "cuda.warp_sync_intrinsic",
        r"\b(__(?:shfl(?:_up|_down|_xor)?_sync|ballot_sync|any_sync|all_sync|"
        r"match_any_sync|match_all_sync|activemask|syncwarp))\b",
        VERIFY,
        HIGH,
        "Warp-synchronous intrinsic",
        None,
        "AMD wavefronts are 64 lanes on CDNA (32 or 64 on RDNA); lane masks are 64-bit. "
        "Check masks and lane arithmetic",
        phases=BOTH,
    ),
    _r(
        "cuda.full_mask",
        r"(0x[fF]{8}\b(?!\s*[fF])|0xffffffffu?\b)",
        VERIFY,
        HIGH,
        "32-bit full lane mask literal",
        "__activemask() or a 64-bit mask",
        "A 32-bit full mask covers only half of a 64-lane wavefront",
        phases=BOTH,
    ),
    _r(
        "cuda.warp_size_literal",
        r"(?:\bWARP_SIZE\s+32\b|\bwarp_?size\s*=\s*32\b|threadIdx\.x\s*(?:%|&)\s*(?:32|31)\b|"
        r"threadIdx\.x\s*(?:/|>>)\s*(?:32|5)\b|\blane(?:_?id)?\s*=\s*\w+\s*(?:%|&)\s*(?:32|31)\b)",
        VERIFY,
        HIGH,
        "Hard-coded warp size of 32",
        "warpSize",
        "Use the device-side warpSize (64 on CDNA) instead of 32",
        phases=BOTH,
        flags=re.IGNORECASE,
    ),
    _r(
        "cuda.arch_macro",
        r"\b(__CUDA_ARCH__|__CUDACC__|CUDART_VERSION)\b",
        VERIFY,
        MED,
        "CUDA architecture/compiler macro",
        "__HIP_DEVICE_COMPILE__ / __HIPCC__ / HIP_VERSION",
        "Code paths keyed on CUDA compute capability need an AMD-specific equivalent",
        phases=BOTH,
    ),
    _r(
        "cuda.compute_capability",
        r"\.(major|minor)\s*(?:>=|<=|==|>|<)\s*\d",
        VERIFY,
        MED,
        "Compute-capability check",
        "hipDeviceProp_t::gcnArchName",
        "CUDA compute capability numbers have no meaning on AMD GPUs; check gcnArchName instead",
        phases=BOTH,
    ),
    _r(
        "cuda.nvrtc",
        r"\b(nvrtc\w+)\s*\(",
        VERIFY,
        MED,
        "NVRTC runtime compilation",
        "prefix:nvrtc->hiprtc",
        "hipRTC compiles HIP source, not PTX; verify generated kernel source",
    ),
    _r(
        "cuda.memcpy_symbol",
        r"\bhipMemcpy(?:To|From)Symbol(?:Async)?\s*\(\s*(?!HIP_SYMBOL)",
        VERIFY,
        MED,
        "hipMemcpyToSymbol without HIP_SYMBOL()",
        "HIP_SYMBOL(sym)",
        "Wrap device symbols with HIP_SYMBOL() to be portable",
        phases=AU,
    ),
    # ---- libraries ------------------------------------------------------------------------------
    _r(
        "lib.cublas",
        r"\b(cublas\w+)\b",
        AUTO,
        LOW,
        "cuBLAS usage",
        "prefix:cublas->hipblas",
        "HIPIFY maps cuBLAS to hipBLAS; link hipblas",
    ),
    _r("lib.curand", r"\b(curand\w+)\b", AUTO, LOW, "cuRAND usage", "prefix:curand->hiprand"),
    _r("lib.cufft", r"\b(cufft\w+)\b", AUTO, LOW, "cuFFT usage", "prefix:cufft->hipfft"),
    _r("lib.cusparse", r"\b(cusparse\w+)\b", AUTO, LOW, "cuSPARSE usage", "prefix:cusparse->hipsparse"),
    _r("lib.cusolver", r"\b(cusolver\w+)\b", AUTO, LOW, "cuSOLVER usage", "prefix:cusolver->hipsolver"),
    _r(
        "lib.nccl",
        r"\b(nccl[A-Z]\w*)\s*\(",
        AUTO,
        LOW,
        "NCCL collective",
        "RCCL (same nccl* API)",
        "Link RCCL; API names are unchanged",
    ),
    _r(
        "lib.thrust",
        r"#\s*include\s*[<\"](thrust/[^>\"]+)[>\"]",
        AUTO,
        LOW,
        "Thrust",
        "rocThrust (same headers)",
    ),
    _r("lib.cub", r"(#\s*include\s*[<\"]cub/[^>\"]+[>\"]|\bcub::)", AUTO, LOW, "CUB", "hipCUB (hipcub::)"),
    _r(
        "lib.cudnn",
        r"\b(cudnn\w+)\b",
        MANUAL,
        HIGH,
        "cuDNN usage",
        "MIOpen",
        "MIOpen has a different API and descriptor model; HIPIFY coverage is partial — port by hand",
    ),
    _r(
        "lib.nvtx",
        r"\b(nvtx\w+)\b|nvToolsExt",
        MANUAL,
        LOW,
        "NVTX ranges",
        "roctx (roctracer)",
        "Replace nvtxRangePush/Pop with roctxRangePush/Pop",
        phases=BOTH,
    ),
    _r(
        "lib.nvml",
        r"\b(nvml\w+)\b",
        MANUAL,
        MED,
        "NVML device management",
        "AMD SMI library (amdsmi_*)",
        "Port NVML queries to the AMD SMI C/Python API",
        phases=BOTH,
    ),
    _r(
        "lib.wmma",
        r"\b(nvcuda::wmma|wmma::\w+)",
        MANUAL,
        HIGH,
        "WMMA tensor-core API",
        "rocWMMA",
        "Port to rocWMMA; tile shapes and supported types differ by GPU",
        phases=BOTH,
    ),
    _r(
        "lib.cutlass",
        r"#\s*include\s*[<\"](cutlass/[^>\"]+)[>\"]",
        UNSUP,
        HIGH,
        "CUTLASS",
        "Composable Kernel or hipBLASLt",
        "CUTLASS has no HIP port; re-implement with Composable Kernel or hipBLASLt",
        phases=BOTH,
    ),
    _r(
        "lib.tensorrt",
        r"#\s*include\s*[<\"](NvInfer\w*\.h)[>\"]|\bnvinfer1::",
        UNSUP,
        HIGH,
        "TensorRT",
        "MIGraphX",
        "TensorRT is NVIDIA-only; use MIGraphX or ONNX Runtime with the MIGraphX/ROCm EP",
        phases=BOTH,
    ),
    _r(
        "lib.libcudacxx",
        r"#\s*include\s*[<\"](cuda/(?:std/)?[^>\"]+)[>\"]",
        MANUAL,
        MED,
        "libcu++ (cuda::std)",
        None,
        "libcu++ is not part of HIP; use std:: or hip/ equivalents",
        phases=BOTH,
    ),
    # ---- not translatable ---------------------------------------------------------------------
    _r(
        "cuda.inline_ptx",
        r"\basm\s*(?:volatile\s*)?\(\s*\"[^\"]*(?:\b(?:mov|ld|st|add|mul|mad|fma|cvt|"
        r"shfl|vote|bar|mma|cp\.async|red|atom)\.[a-z0-9.]+|%%(?:laneid|warpid|clock|smid))",
        UNSUP,
        HIGH,
        "Inline PTX assembly",
        None,
        "PTX is NVIDIA-specific; rewrite with HIP intrinsics or AMDGCN builtins",
        phases=BOTH,
    ),
    _r(
        "cuda.texture_reference",
        r"\b(texture|surface)\s*<",
        UNSUP,
        HIGH,
        "Texture/surface reference (legacy)",
        "hipTextureObject_t",
        "Legacy texture references are removed; use texture objects",
        phases=BOTH,
    ),
    _r(
        "cuda.nv_intrinsic",
        r"\b(__nv_\w+|__nanosleep|__prof_trigger)\s*\(",
        MANUAL,
        MED,
        "NVIDIA-specific device intrinsic",
        None,
        "Find an AMDGCN/HIP equivalent or remove",
        phases=BOTH,
    ),
    # ---- post-conversion leftovers (audit) ------------------------------------------------------
    _r(
        "audit.leftover_runtime_call",
        r"\b(cuda[A-Z]\w*)\s*\(",
        MANUAL,
        HIGH,
        "Unconverted CUDA runtime call",
        "cuda->hip",
        "HIPIFY did not translate this call; port it by hand or check the HIPIFY log",
        phases=AU,
    ),
    _r(
        "audit.leftover_cuda_include",
        r"#\s*include\s*[<\"]((?:cuda|cublas|cudnn|cufft|curand|cusparse|cusolver|nvrtc)"
        r"[\w/]*\.h)[>\"]",
        MANUAL,
        HIGH,
        "CUDA/NVIDIA header still included",
        None,
        "Replace with the HIP/ROCm header",
        phases=AU,
    ),
    _r(
        "audit.leftover_nvidia_lib",
        r"\b((?:cublas|cufft|curand|cusparse|cusolver)[A-Z]\w*)\b",
        MANUAL,
        HIGH,
        "NVIDIA library symbol still referenced",
        None,
        "Replace with the hip* library equivalent",
        phases=AU,
    ),
    _r(
        "audit.leftover_cudnn",
        r"\b(cudnn\w+)\b",
        MANUAL,
        HIGH,
        "cuDNN symbol still referenced",
        "MIOpen",
        "Port to MIOpen",
        phases=AU,
    ),
    # ---- build systems --------------------------------------------------------------------------
    _r(
        "build.cuda_language",
        r"(project\s*\([^)]*\bCUDA\b|enable_language\s*\(\s*CUDA)",
        MANUAL,
        MED,
        "CMake CUDA language",
        "enable_language(HIP) / project(... HIP)",
        "Switch the language to HIP and set CMAKE_HIP_ARCHITECTURES",
        phases=BOTH,
        targets=("build",),
        flags=re.IGNORECASE,
    ),
    _r(
        "build.find_cuda",
        r"find_package\s*\(\s*(CUDA|CUDAToolkit)\b",
        MANUAL,
        MED,
        "CMake finds the CUDA toolkit",
        "find_package(hip) and per-library packages (hipblas, ...)",
        phases=BOTH,
        targets=("build",),
        flags=re.IGNORECASE,
    ),
    _r(
        "build.cuda_targets",
        r"\b(CUDA::\w+|cuda_add_(?:executable|library))",
        MANUAL,
        MED,
        "CMake CUDA targets/macros",
        "hip::host / hip::device / roc::hipblas",
        phases=BOTH,
        targets=("build",),
    ),
    _r(
        "build.cuda_arch",
        r"(CUDA_ARCHITECTURES|CMAKE_CUDA_ARCHITECTURES|-arch(?:=|\s+)sm_\d+|"
        r"-gencode\b|\bsm_\d{2,3}\b|compute_\d{2,3}\b)",
        MANUAL,
        MED,
        "NVIDIA GPU architecture flags",
        "--offload-arch=gfxXXX / CMAKE_HIP_ARCHITECTURES",
        phases=BOTH,
        targets=("build",),
    ),
    _r(
        "build.nvcc",
        r"\b(nvcc)\b",
        MANUAL,
        MED,
        "nvcc compiler invocation",
        "hipcc",
        phases=BOTH,
        targets=("build",),
    ),
    _r(
        "build.cuda_link",
        r"(-l(?:cudart|cublas|cublasLt|cudnn|cufft|curand|cusparse|cusolver|nccl|nvToolsExt|nvidia-ml)\b)",
        MANUAL,
        MED,
        "Linking NVIDIA libraries",
        "-lamdhip64 / -lhipblas / -lrccl ...",
        phases=BOTH,
        targets=("build",),
    ),
    _r(
        "build.cuda_path",
        r"(/usr/local/cuda|CUDA_HOME|CUDA_PATH|CUDA_TOOLKIT_ROOT_DIR)",
        MANUAL,
        LOW,
        "CUDA toolkit path",
        "ROCM_PATH (/opt/rocm)",
        phases=BOTH,
        targets=("build",),
    ),
    _r(
        "build.torch_cuda_ext",
        r"\b(CUDAExtension|BuildExtension)\b",
        VERIFY,
        MED,
        "PyTorch C++/CUDA extension",
        None,
        "torch.utils.cpp_extension hipifies CUDAExtension on ROCm builds of PyTorch; verify the build",
        phases=BOTH,
        targets=("build",),
    ),
]


# --------------------------------------------------------------------------- scanning helpers

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"//[^\n]*")


def strip_comments(text: str, kind: str = "source") -> str:
    """Remove comments but keep line numbering intact."""
    if kind == "build":
        # CMake / Make / shell: '#' comments
        return "\n".join(
            line.split("#", 1)[0] if not line.lstrip().startswith("#include") else line
            for line in text.splitlines()
        )

    def blank(m: re.Match[str]) -> str:
        return "\n" * m.group(0).count("\n")

    text = _BLOCK_COMMENT.sub(blank, text)
    return _LINE_COMMENT.sub("", text)


def iter_matches(
    text: str, rel: str, *, phase: str, target: str, rules: list[Rule] | None = None
) -> Iterator[Finding]:
    clean = strip_comments(text, target)
    raw_lines = text.splitlines()
    for lineno, line in enumerate(clean.splitlines(), start=1):
        if not line.strip():
            continue
        for rule in rules or RULES:
            if phase not in rule.phases or target not in rule.targets:
                continue
            for m in rule.pattern.finditer(line):
                sym = next((g for g in m.groups() if g), m.group(0)) if m.groups() else m.group(0)
                yield Finding(
                    rule_id=rule.id,
                    category=rule.category,
                    severity=rule.severity,
                    message=f"{rule.message}: {sym.strip()}",
                    file=rel,
                    line=lineno,
                    column=m.start() + 1,
                    snippet=(raw_lines[lineno - 1].strip()[:200] if lineno - 1 < len(raw_lines) else None),
                    symbol=sym.strip(),
                    hip_equivalent=rule.hip_for(sym.strip()),
                    suggestion=rule.suggestion,
                )


CUDA_USAGE = re.compile(
    r"(#\s*include\s*[<\"](?:cuda|cublas|cudnn|cufft|curand|cusparse|cusolver|nccl|thrust/|cub/|nvrtc)|"
    r"\bcuda[A-Z]\w*\s*\(|__global__|__device__|<<<|\bcublas\w+|\bcudnn\w+)"
)


def uses_cuda(text: str) -> bool:
    return bool(CUDA_USAGE.search(strip_comments(text)))
