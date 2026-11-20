"""Migration readiness analysis over a scanned inventory."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from kernelvet.migration.rules import iter_matches
from kernelvet.migration.scanner import read_text, scan_project
from kernelvet.models.migration import (
    AnalysisResult,
    FileKind,
    Finding,
    FindingCategory,
    ProjectInventory,
    ReadinessScore,
)

SOURCE_KINDS = {
    FileKind.CUDA_SOURCE,
    FileKind.CUDA_HEADER,
    FileKind.CUDA_RELATED_CPP,
    FileKind.CUDA_RELATED_HEADER,
    FileKind.HIP_SOURCE,
}
BUILD_KINDS = {FileKind.CMAKE, FileKind.MAKEFILE, FileKind.BUILD_SCRIPT}

WEIGHTS = {
    FindingCategory.AUTO_CONVERTIBLE: 1.0,
    FindingCategory.REQUIRES_VERIFICATION: 0.5,
    FindingCategory.MANUAL_INTERVENTION: 0.0,
    FindingCategory.UNSUPPORTED: 0.0,
}


def scan_findings(root: Path, inv: ProjectInventory, phase: str) -> list[Finding]:
    findings: list[Finding] = []
    for f in inv.files:
        if f.kind in SOURCE_KINDS:
            target = "source"
        elif f.kind in BUILD_KINDS:
            target = "build"
        else:
            continue
        text = read_text(root / f.path)
        if text is None:
            continue
        findings.extend(iter_matches(text, f.path, phase=phase, target=target))
    return findings


def readiness(findings: list[Finding]) -> ReadinessScore:
    counts = Counter(f.category for f in findings)
    by_cat = {c.value: counts.get(c, 0) for c in FindingCategory}
    total = sum(counts.values())
    if total == 0:
        return ReadinessScore(
            score=0,
            total_findings=0,
            by_category=by_cat,
            explanation="No CUDA constructs detected; nothing to convert.",
        )
    weighted = sum(WEIGHTS[c] * n for c, n in counts.items())
    score = round(100 * weighted / total)
    # any unsupported construct caps the score: the project can't be fully converted mechanically
    if counts.get(FindingCategory.UNSUPPORTED):
        score = min(score, 60)
    return ReadinessScore(score=score, total_findings=total, by_category=by_cat)


def analyze_project(path: str | Path, *, compile_db: str | Path | None = None) -> AnalysisResult:
    inv = scan_project(path, compile_db)
    findings = scan_findings(Path(inv.root), inv, phase="analyze")
    return AnalysisResult(inventory=inv, findings=findings, readiness=readiness(findings))


def summarize(findings: list[Finding]) -> dict[str, int]:
    return dict(Counter(f.category.value for f in findings))
