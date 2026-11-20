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
    ReadinessLevel,
    ReadinessScore,
    ScoreComponent,
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


# Upper limit on the score when any source construct has no HIP equivalent: such a
# project cannot be converted mechanically, whatever share of the rest can.
UNSUPPORTED_CAP = 60

# Most to least convertible, for the breakdown.
CATEGORY_ORDER = (
    FindingCategory.AUTO_CONVERTIBLE,
    FindingCategory.REQUIRES_VERIFICATION,
    FindingCategory.MANUAL_INTERVENTION,
    FindingCategory.UNSUPPORTED,
)

LEVEL_TEXT = {
    ReadinessLevel.MECHANICAL: "every detected construct has a mechanical HIPIFY translation",
    ReadinessLevel.REVIEW: "everything converts, but some constructs behave differently on AMD GPUs "
    "and need review",
    ReadinessLevel.MANUAL_WORK: "some code has to be rewritten by hand",
    ReadinessLevel.BLOCKED: "some code has no HIP equivalent and needs a redesign",
}


def _weighted(counts: Counter[FindingCategory]) -> int | None:
    total = sum(counts.values())
    if not total:
        return None
    return round(100 * sum(WEIGHTS[c] * n for c, n in counts.items()) / total)


def readiness(findings: list[Finding], inv: ProjectInventory | None = None) -> ReadinessScore:
    """Score how much of the CUDA *source* HIPIFY can translate mechanically.

    * Only source findings are scored. Build-system findings (CMake, Make,
      setup.py) always need manual porting and are reported as ``build_items``.
    * Each rule counts once per file (a "construct"), so 40 cudaMalloc calls
      don't outweigh one inline PTX block. ``occurrence_coverage`` gives the
      per-occurrence figure for comparison.
    * Any UNSUPPORTED construct caps the score at ``UNSUPPORTED_CAP``.
    """
    by_cat = {c.value: 0 for c in FindingCategory}
    for f in findings:
        by_cat[f.category.value] += 1
    build_files = {sf.path for sf in inv.files if sf.kind in BUILD_KINDS} if inv else set()
    source = [f for f in findings if f.file not in build_files]
    build = [f for f in findings if f.file in build_files]

    distinct = {(f.category, f.rule_id, f.file) for f in source}
    constructs = Counter(c for c, _, _ in distinct)
    occurrences = Counter(f.category for f in source)
    files = {c: len({f.file for f in source if f.category is c}) for c in FindingCategory}
    components = [
        ScoreComponent(
            category=c,
            weight=WEIGHTS[c],
            constructs=constructs.get(c, 0),
            occurrences=occurrences.get(c, 0),
            files=files[c],
        )
        for c in CATEGORY_ORDER
    ]
    build_items = len({(f.rule_id, f.file) for f in build})

    if not distinct:
        return ReadinessScore(
            score=None,
            level=ReadinessLevel.NO_CUDA,
            components=components,
            build_items=build_items,
            total_findings=len(findings),
            by_category=by_cat,
            explanation="No CUDA constructs detected in the source code; nothing to convert.",
        )

    coverage = _weighted(constructs) or 0  # distinct is non-empty here
    if constructs.get(FindingCategory.UNSUPPORTED):
        level = ReadinessLevel.BLOCKED
    elif constructs.get(FindingCategory.MANUAL_INTERVENTION):
        level = ReadinessLevel.MANUAL_WORK
    elif constructs.get(FindingCategory.REQUIRES_VERIFICATION):
        level = ReadinessLevel.REVIEW
    else:
        level = ReadinessLevel.MECHANICAL

    score, cap, cap_reason = coverage, None, None
    if level is ReadinessLevel.BLOCKED and coverage > UNSUPPORTED_CAP:
        cap = UNSUPPORTED_CAP
        n = constructs[FindingCategory.UNSUPPORTED]
        cap_reason = f"{n} unsupported construct(s) with no HIP equivalent"
        score = cap

    n_total = len(distinct)
    explanation = (
        f"{coverage}% of {n_total} distinct CUDA construct(s) in the source have a mechanical HIP "
        f"translation (each rule counted once per file; needs-review counts half). "
        f"Level: {level.value}, {LEVEL_TEXT[level]}."
    )
    if cap is not None:
        explanation += f" Score capped at {cap}: {cap_reason}."
    if build_items:
        explanation += f" Plus {build_items} build-system change(s), not scored."
    explanation += " This is source conversion coverage only, not correctness or runtime compatibility."

    return ReadinessScore(
        score=score,
        level=level,
        coverage=coverage,
        occurrence_coverage=_weighted(occurrences),
        cap=cap,
        cap_reason=cap_reason,
        components=components,
        build_items=build_items,
        total_findings=len(findings),
        by_category=by_cat,
        explanation=explanation,
    )


def analyze_project(path: str | Path, *, compile_db: str | Path | None = None) -> AnalysisResult:
    inv = scan_project(path, compile_db)
    findings = scan_findings(Path(inv.root), inv, phase="analyze")
    return AnalysisResult(inventory=inv, findings=findings, readiness=readiness(findings, inv))


def summarize(findings: list[Finding]) -> dict[str, int]:
    return dict(Counter(f.category.value for f in findings))
