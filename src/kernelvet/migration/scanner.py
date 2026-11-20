"""CUDA project scanner: builds a file inventory without modifying anything."""

from __future__ import annotations

import os
import re
from pathlib import Path

from kernelvet.migration.compile_db import CompileDbError, find_compile_db, load_compile_db
from kernelvet.migration.rules import strip_comments, uses_cuda
from kernelvet.models.migration import (
    BuildSystem,
    CompileDbSummary,
    Dependency,
    FileKind,
    ProjectInventory,
    SourceFile,
)
from kernelvet.utils.paths import existing_dir, within

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "build",
    "_build",
    "cmake-build-debug",
    "cmake-build-release",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
}
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_FILES = 20_000

SOURCE_EXT = {".cu": FileKind.CUDA_SOURCE, ".cuh": FileKind.CUDA_HEADER, ".hip": FileKind.HIP_SOURCE}
CPP_EXT = {".cpp", ".cc", ".cxx", ".c", ".c++"}
HDR_EXT = {".h", ".hpp", ".hh", ".hxx", ".inl", ".inc"}
SCRIPT_NAMES = {"setup.py", "build.sh", "build.py", "configure", "meson.build", "BUILD", "BUILD.bazel"}

TEST_PATH = re.compile(
    r"(^|/)(tests?|testing|unittests?)(/|$)|(^|/)(test_[^/]+|[^/]+_test\.\w+|[^/]+_tests?\.\w+)$",
    re.IGNORECASE,
)

# name, kind, regex over sources/build files, HIP alternative, note
DEPENDENCIES: list[tuple[str, str, re.Pattern[str], str | None, str | None]] = [
    (
        "CUDA Runtime",
        "cuda-toolkit",
        re.compile(r"cuda_runtime|cudart|\bcuda[A-Z]\w+\s*\("),
        "HIP runtime",
        None,
    ),
    (
        "CUDA Driver API",
        "cuda-toolkit",
        re.compile(r"#\s*include\s*[<\"]cuda\.h|\bcu(?:Init|ModuleLoad|LaunchKernel)\b"),
        "HIP module API",
        None,
    ),
    ("cuBLAS", "nvidia-library", re.compile(r"cublas"), "hipBLAS (rocBLAS backend)", None),
    ("cuBLASLt", "nvidia-library", re.compile(r"cublasLt"), "hipBLASLt", None),
    ("cuDNN", "nvidia-library", re.compile(r"cudnn"), "MIOpen", "API differs; manual port"),
    ("NCCL", "nvidia-library", re.compile(r"\bnccl|NCCL"), "RCCL", "API-compatible"),
    ("cuFFT", "nvidia-library", re.compile(r"cufft"), "hipFFT (rocFFT backend)", None),
    ("cuRAND", "nvidia-library", re.compile(r"curand"), "hipRAND (rocRAND backend)", None),
    ("cuSPARSE", "nvidia-library", re.compile(r"cusparse"), "hipSPARSE (rocSPARSE backend)", None),
    ("cuSOLVER", "nvidia-library", re.compile(r"cusolver"), "hipSOLVER (rocSOLVER backend)", None),
    ("cuTENSOR", "nvidia-library", re.compile(r"cutensor"), "hipTensor", "verify API coverage"),
    ("Thrust", "nvidia-library", re.compile(r"thrust/"), "rocThrust", None),
    ("CUB", "nvidia-library", re.compile(r"cub/|\bcub::"), "hipCUB", None),
    ("NVTX", "nvidia-library", re.compile(r"nvToolsExt|nvtx3?/|\bnvtx\w+"), "roctx", None),
    ("NVML", "nvidia-library", re.compile(r"\bnvml\w*|nvidia-ml"), "AMD SMI", "manual port"),
    ("NVRTC", "nvidia-library", re.compile(r"\bnvrtc\w*"), "hipRTC", None),
    ("CUTLASS", "nvidia-library", re.compile(r"cutlass/"), "Composable Kernel / hipBLASLt", "no HIP port"),
    ("TensorRT", "nvidia-library", re.compile(r"NvInfer|nvinfer1"), "MIGraphX", "no HIP port"),
    ("OptiX", "nvidia-library", re.compile(r"optix\w*\.h|\boptix[A-Z]"), None, "no AMD equivalent in ROCm"),
    ("libcu++", "nvidia-library", re.compile(r"#\s*include\s*[<\"]cuda/"), None, "use std:: equivalents"),
    (
        "PyTorch C++/CUDA extension",
        "external",
        re.compile(r"CUDAExtension|torch/extension\.h"),
        "PyTorch ROCm build",
        "torch.utils.cpp_extension hipifies automatically",
    ),
]


def classify(rel: str, text: str | None) -> FileKind:
    name = os.path.basename(rel)
    ext = os.path.splitext(name)[1].lower()
    if ext in SOURCE_EXT:
        return SOURCE_EXT[ext]
    if name == "CMakeLists.txt" or ext == ".cmake":
        return FileKind.CMAKE
    if name in ("Makefile", "makefile", "GNUmakefile") or ext == ".mk":
        return FileKind.MAKEFILE
    if name in SCRIPT_NAMES or ext == ".sh":
        return FileKind.BUILD_SCRIPT
    if ext in CPP_EXT:
        return FileKind.CUDA_RELATED_CPP if text and uses_cuda(text) else FileKind.CPP
    if ext in HDR_EXT:
        return FileKind.CUDA_RELATED_HEADER if text and uses_cuda(text) else FileKind.CPP
    return FileKind.OTHER


TEXT_KINDS = {
    FileKind.CUDA_SOURCE,
    FileKind.CUDA_HEADER,
    FileKind.HIP_SOURCE,
    FileKind.CMAKE,
    FileKind.MAKEFILE,
    FileKind.BUILD_SCRIPT,
}


def iter_project_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".kernelvet"))
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            if p.is_symlink() and not within(root, p):
                continue  # never follow symlinks out of the project
            out.append(p)
            if len(out) >= MAX_FILES:
                return out
    return out


def read_text(p: Path) -> str | None:
    try:
        if p.stat().st_size > MAX_FILE_BYTES:
            return None
        data = p.read_bytes()
    except OSError:
        return None
    if b"\0" in data[:4096]:
        return None  # binary
    return data.decode("utf-8", errors="replace")


_KERNEL = re.compile(r"__global__\s+")
_CUDA_CALL = re.compile(r"\bcuda[A-Z]\w*\s*\(")


def _build_system(kind: FileKind, rel: str, text: str) -> BuildSystem:
    clean = strip_comments(text, "build")
    cuda_lang = bool(
        re.search(
            r"(project\s*\([^)]*\bCUDA\b|enable_language\s*\(\s*CUDA|\bnvcc\b|\.cu\b)", clean, re.IGNORECASE
        )
    )
    arch = sorted(set(re.findall(r"\b(?:sm|compute)_\d{2,3}\b", clean))) + sorted(
        set(re.findall(r"CUDA_ARCHITECTURES\s+\"?([\d;]+)\"?", clean))
    )
    notes = []
    if re.search(r"find_package\s*\(\s*(CUDA|CUDAToolkit)", clean, re.IGNORECASE):
        notes.append("find_package(CUDA/CUDAToolkit)")
    if "CUDAExtension" in clean:
        notes.append("PyTorch CUDAExtension")
    k = {FileKind.CMAKE: "cmake", FileKind.MAKEFILE: "make"}.get(kind, "script")
    return BuildSystem(kind=k, files=[rel], uses_cuda_language=cuda_lang, cuda_arch_flags=arch, notes=notes)


def scan_project(path: str | Path, compile_db: str | Path | None = None) -> ProjectInventory:
    root = existing_dir(path)
    inv = ProjectInventory(root=str(root))
    builds: dict[str, BuildSystem] = {}
    dep_hits: dict[str, list[str]] = {}
    for p in iter_project_files(root):
        rel = p.relative_to(root).as_posix()
        ext = p.suffix.lower()
        maybe_text = (
            ext in SOURCE_EXT
            or ext in CPP_EXT
            or ext in HDR_EXT
            or p.name in ("CMakeLists.txt", "Makefile", "makefile", "GNUmakefile")
            or ext in (".cmake", ".mk", ".sh", ".py")
        )
        text = read_text(p) if maybe_text else None
        if maybe_text and text is None:
            inv.skipped.append(f"{rel}: unreadable, binary or larger than {MAX_FILE_BYTES // 1024**2} MiB")
        kind = classify(rel, text)
        if kind is FileKind.OTHER and p.name not in SCRIPT_NAMES and ext != ".py":
            continue
        if ext == ".py" and kind is FileKind.OTHER:
            if not (text and ("CUDAExtension" in text or "cpp_extension" in text)):
                continue
            kind = FileKind.BUILD_SCRIPT
        lines = text.count("\n") + 1 if text else 0
        is_test = bool(TEST_PATH.search(rel))
        sf = SourceFile(
            path=rel,
            kind=kind,
            lines=lines,
            is_test=is_test,
            cuda_api_calls=len(_CUDA_CALL.findall(strip_comments(text))) if text else 0,
            kernels=len(_KERNEL.findall(text)) if text else 0,
        )
        inv.files.append(sf)
        inv.total_lines += lines
        if is_test and kind not in (FileKind.CMAKE, FileKind.MAKEFILE):
            inv.test_files.append(rel)
        if text and kind in (FileKind.CMAKE, FileKind.MAKEFILE, FileKind.BUILD_SCRIPT):
            bs = _build_system(kind, rel, text)
            if bs.kind in builds:
                existing = builds[bs.kind]
                existing.files.append(rel)
                existing.uses_cuda_language |= bs.uses_cuda_language
                existing.cuda_arch_flags = sorted(set(existing.cuda_arch_flags + bs.cuda_arch_flags))
                existing.notes = sorted(set(existing.notes + bs.notes))
            else:
                builds[bs.kind] = bs
        if text and kind is not FileKind.CPP:
            clean = strip_comments(text, "build" if kind in (FileKind.CMAKE, FileKind.MAKEFILE) else "source")
            for name, _k, pat, _alt, _note in DEPENDENCIES:
                for i, line in enumerate(clean.splitlines(), 1):
                    if pat.search(line):
                        dep_hits.setdefault(name, []).append(f"{rel}:{i}")
                        break
    inv.build_systems = list(builds.values())
    for name, dep_kind, _pat, alt, note in DEPENDENCIES:
        if name in dep_hits:
            inv.dependencies.append(
                Dependency(
                    name=name, kind=dep_kind, evidence=dep_hits[name][:20], hip_alternative=alt, note=note
                )
            )
    inv.compile_db = summarize_compile_db(root, inv, compile_db)
    return inv


# Files that are compiled on their own and so should have a compile_commands.json entry.
_TRANSLATION_UNIT_KINDS = (FileKind.CUDA_SOURCE, FileKind.CUDA_RELATED_CPP)


def summarize_compile_db(
    root: Path, inv: ProjectInventory, explicit: str | Path | None = None
) -> CompileDbSummary | None:
    path = Path(explicit).expanduser().resolve() if explicit else find_compile_db(root)
    if path is None:
        return None
    shown = path.relative_to(root.resolve()).as_posix() if within(root, path) else str(path)
    if not path.is_file():
        return CompileDbSummary(path=shown, error="file not found")
    try:
        db = load_compile_db(path)
    except CompileDbError as exc:
        return CompileDbSummary(path=shown, error=str(exc))
    units = [f.path for f in inv.files if f.kind in _TRANSLATION_UNIT_KINDS]
    covered = [u for u in units if db.covers(root / u)]
    return CompileDbSummary(
        path=shown,
        entries=len(db.entries),
        covered=covered,
        uncovered=[u for u in units if u not in covered],
    )
