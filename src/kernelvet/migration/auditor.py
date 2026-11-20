"""Static audit of a migrated (HIP) tree.

Finds leftover CUDA, NVIDIA library references, CUDA build configuration,
risky substitutions and architecture assumptions. A clean audit does **not**
mean the code is correct; only compiling and running tests can show that.
"""

from __future__ import annotations

import re
from pathlib import Path

from kernelvet.migration.analyzer import BUILD_KINDS, SOURCE_KINDS
from kernelvet.migration.rules import RULES, Rule, iter_matches, strip_comments
from kernelvet.migration.scanner import read_text, scan_project
from kernelvet.models.migration import AuditResult, FileKind, Finding, FindingCategory, Severity

# HIP header -> library family name used by diagnose.software.detect_libraries
HIP_LIB_HEADERS: dict[str, str] = {
    "hipblas": "hipBLAS",
    "hipblaslt": "hipBLASLt",
    "hipfft": "hipFFT",
    "hiprand": "hipRAND",
    "hipsparse": "hipSPARSE",
    "hipsolver": "hipSOLVER",
    "miopen": "MIOpen",
    "rccl": "RCCL",
    "hipcub": "hipCUB",
    "rocrand": "rocRAND",
    "rocblas": "rocBLAS",
}
_INCLUDE = re.compile(r"#\s*include\s*[<\"]([\w/]+)/?[\w.]*[>\"]")
AUDIT_RULES: list[Rule] = [r for r in RULES if "audit" in r.phases]


def audit_project(path: str | Path, *, installed_libraries: list[str] | None = None) -> AuditResult:
    inv = scan_project(path)
    root = Path(inv.root)
    findings: list[Finding] = []
    scanned = 0
    referenced: dict[str, str] = {}
    for f in inv.files:
        if f.kind in SOURCE_KINDS or f.kind is FileKind.CPP:
            target = "source"
        elif f.kind in BUILD_KINDS:
            target = "build"
        else:
            continue
        text = read_text(root / f.path)
        if text is None:
            continue
        scanned += 1
        findings.extend(iter_matches(text, f.path, phase="audit", target=target, rules=AUDIT_RULES))
        if target == "source":
            for i, line in enumerate(strip_comments(text).splitlines(), 1):
                for m in _INCLUDE.finditer(line):
                    head = m.group(1).split("/")[0].lower()
                    if head in HIP_LIB_HEADERS:
                        referenced.setdefault(HIP_LIB_HEADERS[head], f"{f.path}:{i}")
    if installed_libraries is not None:
        installed = set(installed_libraries)
        for lib, where in sorted(referenced.items()):
            if lib not in installed:
                file, _, line = where.rpartition(":")
                findings.append(
                    Finding(
                        rule_id="audit.missing_dependency",
                        category=FindingCategory.MANUAL_INTERVENTION,
                        severity=Severity.HIGH,
                        message=f"{lib} is used but was not found in the ROCm install",
                        file=file,
                        line=int(line),
                        symbol=lib,
                        suggestion=f"Install the {lib} package for your ROCm release",
                    )
                )
    findings.sort(key=lambda x: (x.file, x.line or 0, x.rule_id))
    return AuditResult(project=str(root), files_scanned=scanned, findings=findings)


def migrated_source_of(path: str | Path) -> str | None:
    """Return the CUDA project a migrated tree came from (written by convert_project)."""
    meta = Path(path) / ".kernelvet" / "source.json"
    try:
        import json

        return json.loads(meta.read_text()).get("source_project")
    except (OSError, ValueError):
        return None
