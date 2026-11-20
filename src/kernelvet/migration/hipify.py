"""AMD HIPIFY integration (``hipify-clang`` and ``hipify-perl``).

KernelVet never translates CUDA itself. This module picks a translator, runs it
file by file into a separate output tree, copies everything else verbatim and
records every warning, error and HIPIFY statistic.

Guarantees:

* the source tree is opened read-only and never written;
* a file HIPIFY fails on is copied unchanged and marked ``failed`` (so the
  audit flags its leftover CUDA), never silently dropped;
* the full log is written to ``<output>/.kernelvet/conversion.log``.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from kernelvet.migration.scanner import iter_project_files
from kernelvet.models.migration import ConversionResult, FileConversion, FileKind, ProjectInventory
from kernelvet.utils.paths import safe_output_dir
from kernelvet.utils.proc import HostContext, ProgressCallback, noop_progress
from kernelvet.utils.versions import version_str

TRANSLATE_KINDS = {
    FileKind.CUDA_SOURCE,
    FileKind.CUDA_HEADER,
    FileKind.CUDA_RELATED_CPP,
    FileKind.CUDA_RELATED_HEADER,
}
LOG_DIR = ".kernelvet"


class HipifyUnavailable(RuntimeError):
    pass


class Translator(StrEnum):
    AUTO = "auto"
    HIPIFY_CLANG = "hipify-clang"
    HIPIFY_PERL = "hipify-perl"


@dataclass
class TranslatorChoice:
    name: str | None
    reason: str
    cuda_path: str | None = None


def find_cuda_headers(ctx: HostContext) -> str | None:
    """hipify-clang parses CUDA with clang and needs CUDA toolkit headers."""
    for key in ("CUDA_PATH", "CUDA_HOME", "CUDA_TOOLKIT_ROOT_DIR"):
        p = ctx.env.get(key)
        if p and ctx.exists(f"{p}/include/cuda_runtime.h"):
            return p
    for p in ("/usr/local/cuda", "/opt/cuda", "/usr"):
        if ctx.exists(f"{p}/include/cuda_runtime.h"):
            return p
    return None


def choose_translator(
    ctx: HostContext, preference: str = "auto", *, has_compile_db: bool = False
) -> TranslatorChoice:
    if preference not in {t.value for t in Translator}:
        raise ValueError(
            f"unknown translator {preference!r}; choose from {', '.join(t.value for t in Translator)}"
        )
    clang = ctx.which("hipify-clang")
    perl = ctx.which("hipify-perl")
    cuda = find_cuda_headers(ctx)
    if preference == "hipify-clang":
        if not clang:
            return TranslatorChoice(None, "hipify-clang requested but not installed")
        if not cuda:
            return TranslatorChoice(
                None,
                "hipify-clang requested but no CUDA headers found "
                "(set CUDA_PATH); it needs them to parse CUDA",
            )
        return TranslatorChoice("hipify-clang", "requested in configuration", cuda)
    if preference == "hipify-perl":
        if not perl:
            return TranslatorChoice(None, "hipify-perl requested but not installed")
        return TranslatorChoice("hipify-perl", "requested in configuration")
    if clang and cuda:
        why = "hipify-clang is installed and CUDA headers were found; it parses code semantically"
        if has_compile_db:
            why += " (compile_commands.json available for include paths)"
        return TranslatorChoice("hipify-clang", why, cuda)
    if perl:
        why = "hipify-perl is installed (regex-based; no CUDA toolkit needed)"
        if clang and not cuda:
            why += "; hipify-clang is installed but no CUDA headers were found"
        return TranslatorChoice("hipify-perl", why)
    return TranslatorChoice(
        None,
        "neither hipify-clang nor hipify-perl is installed (install the `hipify-clang` package from ROCm)",
    )


_STATS = re.compile(
    r"CONVERTED refs count:\s*(\d+).*?UNCONVERTED refs count:\s*(\d+)", re.IGNORECASE | re.DOTALL
)
_WARN = re.compile(r"(?i)\bwarning\b[:\s]")
_ERR = re.compile(r"(?i)\b(error|fatal)\b[:\s]")


def parse_hipify_output(text: str) -> tuple[int | None, int | None, list[str], list[str]]:
    converted = unconverted = None
    m = _STATS.search(text)
    if m:
        converted, unconverted = int(m.group(1)), int(m.group(2))
    warnings, errors = [], []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if _ERR.search(s):
            errors.append(s[:400])
        elif _WARN.search(s):
            warnings.append(s[:400])
    return converted, unconverted, warnings, errors


def _include_flags(root: Path, inv: ProjectInventory) -> list[str]:
    dirs = sorted(
        {
            str((root / f.path).parent)
            for f in inv.files
            if f.kind in (FileKind.CUDA_HEADER, FileKind.CUDA_RELATED_HEADER, FileKind.CPP)
        }
    )
    flags = [f"-I{root}"]
    flags += [f"-I{d}" for d in dirs if d != str(root)]
    for name in ("include", "inc"):
        if (root / name).is_dir():
            flags.append(f"-I{root / name}")
    return flags


def translator_argv(choice: TranslatorChoice, src: Path, out: Path, includes: list[str]) -> list[str]:
    if choice.name == "hipify-perl":
        return ["hipify-perl", "-print-stats", f"-o={out}", str(src)]
    argv = ["hipify-clang", str(src), "-o", str(out), "--print-stats"]
    if choice.cuda_path:
        argv.append(f"--cuda-path={choice.cuda_path}")
    return [*argv, "--", "-std=c++17", *includes]


def translator_version(ctx: HostContext, name: str) -> str | None:
    res = ctx.run([name, "--version"], timeout=20)
    return version_str(res.stdout + res.stderr) if (res.stdout or res.stderr) else None


def convert_project(
    ctx: HostContext,
    inv: ProjectInventory,
    output: str | Path,
    *,
    preference: str = "auto",
    dry_run: bool = False,
    force: bool = False,
    timeout: float = 300.0,
    progress: ProgressCallback = noop_progress,
) -> ConversionResult:
    root = Path(inv.root)
    out_root = safe_output_dir(root, output, allow_existing=force)
    choice = choose_translator(ctx, preference, has_compile_db=(root / "compile_commands.json").exists())
    if choice.name is None:
        raise HipifyUnavailable(choice.reason)

    kinds = {f.path: f.kind for f in inv.files}
    result = ConversionResult(
        project=str(root),
        output=str(out_root),
        translator=choice.name,
        translator_version=None if dry_run else translator_version(ctx, choice.name),
        dry_run=dry_run,
    )
    includes = _include_flags(root, inv)
    files = [p for p in iter_project_files(root)]
    log_lines: list[str] = [
        f"translator: {choice.name} ({choice.reason})",
        f"source: {root}",
        f"output: {out_root}",
        "",
    ]
    for i, src in enumerate(files):
        rel = src.relative_to(root).as_posix()
        dst = out_root / rel
        kind = kinds.get(rel)
        progress(f"{'Planning' if dry_run else 'Converting'} {rel}", (i + 1) / max(len(files), 1))
        if kind in TRANSLATE_KINDS:
            argv = translator_argv(choice, src, dst, includes)
            if dry_run:
                result.files.append(
                    FileConversion(source=rel, output=str(dst), translator=choice.name, status="planned")
                )
                log_lines.append("would run: " + " ".join(argv))
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            res = ctx.run(argv, timeout=timeout, cwd=root)
            converted, unconverted, warnings, errors = parse_hipify_output(res.stdout + "\n" + res.stderr)
            ok = res.ok and dst.exists()
            if not ok:
                if not res.ok:
                    errors.insert(0, res.describe_failure())
                if not dst.exists():
                    errors.append("translator produced no output; original copied unchanged for review")
                    shutil.copy2(src, dst)
            result.files.append(
                FileConversion(
                    source=rel,
                    output=str(dst),
                    translator=choice.name,
                    status="converted" if ok else "failed",
                    returncode=res.returncode,
                    warnings=warnings,
                    errors=errors,
                    replacements=converted,
                    unconverted=unconverted,
                )
            )
            log_lines += [
                f"$ {' '.join(argv)}",
                f"exit: {res.returncode}",
                res.stdout.strip(),
                res.stderr.strip(),
                "",
            ]
        else:
            if dry_run:
                result.files.append(
                    FileConversion(source=rel, output=str(dst), translator="copy", status="planned-copy")
                )
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_symlink():
                result.files.append(
                    FileConversion(
                        source=rel,
                        output=str(dst),
                        translator="copy",
                        status="skipped",
                        warnings=["symlink not recreated in the output tree"],
                    )
                )
                continue
            shutil.copy2(src, dst)
            result.files.append(
                FileConversion(source=rel, output=str(dst), translator="copy", status="copied")
            )

    if not dry_run:
        log_dir = out_root / LOG_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "conversion.log").write_text("\n".join(log_lines) + "\n")
        (log_dir / "conversion.json").write_text(result.model_dump_json(indent=2))
        (log_dir / "source.json").write_text(json.dumps({"source_project": str(root)}))
        result.log_path = str(log_dir / "conversion.log")
    return result
