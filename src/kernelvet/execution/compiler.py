"""Compile a migrated HIP project.

Strategy:

1. If the project's CMake already targets HIP (``enable_language(HIP)`` or
   ``LANGUAGES ... HIP``) and ``cmake`` is available, configure and build it.
2. Otherwise build directly with ``hipcc``: every translation unit containing
   ``main`` becomes an executable linked with the shared (non-main) units and
   with the ROCm libraries its includes require. A HIP ``CMakeLists`` that
   reproduces this build is generated under ``.kernelvet/`` for the user.

All output lives in ``<project>/.kernelvet/build``; sources are never edited.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from kernelvet.execution.sandbox import SandboxConfig, docker_argv
from kernelvet.migration.scanner import HDR_EXT, read_text, scan_project
from kernelvet.models.migration import BuildResult, FileKind, StageStatus
from kernelvet.utils.proc import CommandResult, HostContext

STATE_DIR = ".kernelvet"
BUILD_SUBDIR = f"{STATE_DIR}/build"

COMPILE_KINDS = {FileKind.CUDA_SOURCE, FileKind.HIP_SOURCE, FileKind.CUDA_RELATED_CPP, FileKind.CPP}
_MAIN = re.compile(r"\bint\s+main\s*\(")
_HIP_CMAKE = re.compile(r"(enable_language\s*\(\s*HIP\b|project\s*\([^)]*\bHIP\b)", re.IGNORECASE)
_CUDA_CMAKE = re.compile(r"(enable_language\s*\(\s*CUDA\b|project\s*\([^)]*\bCUDA\b)", re.IGNORECASE)

LIB_FLAGS: dict[str, str] = {
    "hipblas": "-lhipblas",
    "hipblaslt": "-lhipblaslt",
    "rocblas": "-lrocblas",
    "hipfft": "-lhipfft",
    "hiprand": "-lhiprand",
    "rocrand": "-lrocrand",
    "hipsparse": "-lhipsparse",
    "hipsolver": "-lhipsolver",
    "rccl": "-lrccl",
    "miopen": "-lMIOpen",
    "roctracer": "-lroctx64",
}
_INC = re.compile(r"#\s*include\s*[<\"](\w+)/")
_ERROR_LINE = re.compile(r"(error:|undefined reference|ld(?:\.lld)?: error|fatal error)", re.IGNORECASE)


def _errors(text: str, limit: int = 60) -> list[str]:
    return [line.strip() for line in text.splitlines() if _ERROR_LINE.search(line)][:limit]


def _arch_flags(targets: list[str]) -> list[str]:
    return [f"--offload-arch={t}" for t in sorted({t.split(":")[0] for t in targets if t.startswith("gfx")})]


@dataclass
class DirectBuild:
    units: list[str] = field(default_factory=list)  # shared translation units (no main)
    mains: list[str] = field(default_factory=list)  # translation units defining main()
    libs: list[str] = field(default_factory=list)  # -l flags inferred from includes
    commands: list[list[str]] = field(default_factory=list)
    executables: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


COMPILE_FLAGS_FILE = f"{STATE_DIR}/compile_flags.json"  # written by `migrate convert`


def _load_compile_flags(project: Path) -> dict[str, list[str]]:
    """Per-file flags from the CUDA project's compile_commands.json, if convert saved them."""
    try:
        data = json.loads((project / COMPILE_FLAGS_FILE).read_text())
    except (OSError, ValueError):
        return {}
    files = data.get("files") if isinstance(data, dict) else None
    if not isinstance(files, dict):
        return {}
    return {str(k): [str(f) for f in v] for k, v in files.items() if isinstance(v, list)}


def _macro(flag: str) -> str:
    return flag[2:].split("=", 1)[0]


def merge_compile_flags(sources: list[str], table: dict[str, list[str]], notes: list[str]) -> list[str]:
    """Combine the database flags of every file compiled in one hipcc command.

    The first file (the one with ``main``) wins: its ``-std`` and macro values are
    kept, and a conflicting ``-D`` from another file is dropped with a note.
    Duplicate flags are removed while keeping include order.
    """
    out: list[str] = []
    seen: set[tuple[str, ...]] = set()
    macros: dict[str, str] = {}
    std: str | None = None
    for src in sources:
        flags = table.get(src, [])
        i = 0
        while i < len(flags):
            flag = flags[i]
            item: tuple[str, ...] = (flag,)
            if flag in ("-isystem", "-iquote", "-idirafter", "-include") and i + 1 < len(flags):
                item = (flag, flags[i + 1])
                i += 1
            i += 1
            if item[0].startswith("-std="):
                if std is None:
                    std = item[0]
                    out.append(std)
                continue
            if item[0].startswith(("-D", "-U")):
                name = _macro(item[0])
                if name in macros and macros[name] != item[0]:
                    notes.append(
                        f"conflicting {macros[name]} and {item[0]} across files compiled together; kept {macros[name]}"
                    )
                    continue
                macros[name] = item[0]
            if item not in seen:
                seen.add(item)
                out += list(item)
    return out


def plan_direct_build(project: Path, targets: list[str]) -> DirectBuild:
    """Plan a direct hipcc build. All paths are project-relative."""
    inv = scan_project(project)
    plan = DirectBuild()
    libs: set[str] = set()
    for f in inv.files:
        if f.kind not in COMPILE_KINDS or Path(f.path).suffix.lower() in HDR_EXT:
            continue
        text = read_text(project / f.path) or ""
        for m in _INC.finditer(text):
            if m.group(1).lower() in LIB_FLAGS:
                libs.add(LIB_FLAGS[m.group(1).lower()])
        (plan.mains if _MAIN.search(text) else plan.units).append(f.path)
    plan.libs = sorted(libs)
    incdirs = sorted(
        {str(Path(f.path).parent) for f in inv.files if Path(f.path).suffix in (".h", ".hpp", ".cuh", ".hh")}
    )
    inc_flags = ["-I."] + [f"-I{d}" for d in incdirs if d != "."]
    arch = _arch_flags(targets)
    table = _load_compile_flags(project)
    if table:
        compiled = plan.mains + plan.units
        known = [s for s in compiled if s in table]
        plan.notes.append(
            f"include paths and defines from compile_commands.json for {len(known)} of {len(compiled)} "
            "translation unit(s); the rest use guessed include paths"
        )
        outside = sorted({f for fl in table.values() for f in fl if f.startswith("-I/")})
        if outside:
            plan.notes.append(
                f"{len(outside)} include path(s) from compile_commands.json are outside the project "
                "and are not visible inside the Docker sandbox"
            )
    if not plan.mains:
        plan.notes.append("no translation unit defines main(); compiled objects only")
        for u in plan.units:
            obj = f"{BUILD_SUBDIR}/{Path(u).with_suffix('.o').as_posix().replace('/', '_')}"
            db = merge_compile_flags([u], table, plan.notes)
            plan.commands.append(["hipcc", "-O2", *arch, *db, *inc_flags, "-c", "-x", "hip", u, "-o", obj])
        return plan
    for main_src in plan.mains:
        exe = f"{BUILD_SUBDIR}/{Path(main_src).stem}"
        db = merge_compile_flags([main_src, *plan.units], table, plan.notes)
        plan.commands.append(
            [
                "hipcc",
                "-O2",
                *arch,
                *db,
                *inc_flags,
                "-x",
                "hip",
                main_src,
                *plan.units,
                "-o",
                exe,
                *plan.libs,
            ]
        )
        plan.executables.append(exe)
    return plan


def generated_cmake(
    project_name: str, units: list[str], mains: list[str], libs: list[str], targets: list[str]
) -> str:
    arch = ";".join(sorted({t.split(":")[0] for t in targets})) or "gfx942"
    lines = [
        "# Generated by KernelVet. Review before use.",
        "cmake_minimum_required(VERSION 3.21)",
        f"project({project_name} LANGUAGES CXX HIP)",
        f'set(CMAKE_HIP_ARCHITECTURES "{arch}" CACHE STRING "AMD GPU targets")',
        "find_package(hip REQUIRED)",
    ]
    lib_pkgs = {
        "-lhipblas": "hipblas",
        "-lhipfft": "hipfft",
        "-lhiprand": "hiprand",
        "-lhipsparse": "hipsparse",
        "-lhipsolver": "hipsolver",
        "-lrccl": "rccl",
        "-lMIOpen": "miopen",
        "-lrocblas": "rocblas",
    }
    for flag in libs:
        if flag in lib_pkgs:
            lines.append(f"find_package({lib_pkgs[flag]} REQUIRED)")
    srcs = " ".join(units)
    lines += [
        f"set_source_files_properties({' '.join(units + mains)} PROPERTIES LANGUAGE HIP)"
        if units or mains
        else ""
    ]
    if units:
        lines += [
            f"add_library(kernelvet_common STATIC {srcs})",
            "target_include_directories(kernelvet_common PUBLIC . include)",
            "target_link_libraries(kernelvet_common PUBLIC hip::device)",
        ]
    for m in mains:
        name = Path(m).stem
        lines.append(f"add_executable({name} {m})")
        deps = ["kernelvet_common"] if units else ["hip::device"]
        deps += [f"roc::{lib_pkgs[f]}" for f in libs if f in lib_pkgs]
        lines.append(f"target_link_libraries({name} PRIVATE {' '.join(deps)})")
    return "\n".join(line for line in lines if line) + "\n"


def build_project(
    ctx: HostContext,
    path: str | Path,
    *,
    gfx_targets: list[str] | None = None,
    isolation: str = "none",
    sandbox: SandboxConfig | None = None,
    timeout: float = 900.0,
) -> BuildResult:
    project = Path(path).resolve()
    targets = gfx_targets or []
    docker = isolation == "docker"
    if docker and not ctx.which("docker"):
        return BuildResult(
            project=str(project),
            status=StageStatus.UNAVAILABLE,
            method="none",
            reason="isolation=docker but docker is not installed",
        )
    if not docker and not ctx.which("hipcc"):
        return BuildResult(
            project=str(project),
            status=StageStatus.UNAVAILABLE,
            method="none",
            reason="hipcc not found; install the ROCm HIP SDK or use isolation=docker",
        )
    (project / BUILD_SUBDIR).mkdir(parents=True, exist_ok=True)
    cfg = sandbox or SandboxConfig()

    def run(cmd: list[str]) -> CommandResult:
        argv = docker_argv(project, cmd, cfg=cfg) if docker else cmd
        return ctx.run(argv, timeout=timeout, cwd=project)

    cmake_text = read_text(project / "CMakeLists.txt") if (project / "CMakeLists.txt").exists() else None
    t0 = time.monotonic()
    if cmake_text and _HIP_CMAKE.search(cmake_text) and (docker or ctx.which("cmake")):
        bdir = BUILD_SUBDIR + "/cmake"
        configure = ["cmake", "-S", ".", "-B", bdir, "-DCMAKE_BUILD_TYPE=Release"]
        if targets:
            configure.append(
                "-DCMAKE_HIP_ARCHITECTURES=" + ";".join(sorted({t.split(":")[0] for t in targets}))
            )
        r1 = run(configure)
        if not r1.ok:
            return BuildResult(
                project=str(project),
                status=StageStatus.FAILED,
                method="cmake",
                command=configure,
                returncode=r1.returncode,
                stdout=r1.stdout[-20000:],
                stderr=r1.stderr[-20000:],
                errors=_errors(r1.stdout + r1.stderr) or [r1.describe_failure()],
                duration_s=time.monotonic() - t0,
                isolated=docker,
            )
        build = ["cmake", "--build", bdir, "--parallel"]
        r2 = run(build)
        artifacts = [
            str(p.relative_to(project))
            for p in (project / bdir).rglob("*")
            if p.is_file() and p.stat().st_mode & 0o111 and "CMakeFiles" not in p.parts
        ]
        return BuildResult(
            project=str(project),
            status=StageStatus.PASSED if r2.ok else StageStatus.FAILED,
            method="cmake",
            command=build,
            returncode=r2.returncode,
            stdout=r2.stdout[-20000:],
            stderr=r2.stderr[-20000:],
            errors=[] if r2.ok else (_errors(r2.stdout + r2.stderr) or [r2.describe_failure()]),
            artifacts=artifacts,
            duration_s=time.monotonic() - t0,
            isolated=docker,
        )

    plan = plan_direct_build(project, targets)
    reason_parts = []
    if cmake_text and _CUDA_CMAKE.search(cmake_text):
        reason_parts.append(
            "CMakeLists.txt still declares the CUDA language, so it was not used; built directly with hipcc"
        )
    if not targets:
        reason_parts.append("no --offload-arch given and no GPU detected: hipcc uses its default targets")
    reason_parts += plan.notes
    if not plan.commands:
        return BuildResult(
            project=str(project),
            status=StageStatus.FAILED,
            method="hipcc",
            reason="no HIP/C++ translation units found to compile",
            duration_s=0.0,
            isolated=docker,
        )
    try:
        gen = generated_cmake(project.name, plan.units, plan.mains, plan.libs, targets)
        (project / STATE_DIR / "CMakeLists.generated.txt").write_text(gen)
        reason_parts.append(f"generated a HIP CMakeLists at {STATE_DIR}/CMakeLists.generated.txt")
    except OSError:
        pass
    commands, exes = plan.commands, plan.executables
    stdout, stderr, errors = [], [], []
    rc: int | None = 0
    for cmd in commands:
        r = run(cmd)
        stdout.append(f"$ {' '.join(cmd)}\n{r.stdout}")
        stderr.append(r.stderr)
        if not r.ok:
            rc = r.returncode
            errors += _errors(r.stdout + r.stderr) or [r.describe_failure()]
    built = [e for e in exes if (project / e).exists()]
    ok = not errors
    return BuildResult(
        project=str(project),
        status=StageStatus.PASSED if ok else StageStatus.FAILED,
        method="hipcc",
        command=commands[0] if len(commands) == 1 else ["hipcc", f"({len(commands)} commands)"],
        returncode=rc,
        stdout="\n".join(stdout)[-20000:],
        stderr="\n".join(stderr)[-20000:],
        artifacts=built,
        errors=errors,
        duration_s=time.monotonic() - t0,
        isolated=docker,
        reason="; ".join(reason_parts) or None,
    )
