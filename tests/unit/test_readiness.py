"""Readiness scoring: what is counted, how, and what it says about itself."""

from __future__ import annotations

from kernelvet.migration.analyzer import UNSUPPORTED_CAP, readiness
from kernelvet.models.migration import (
    READINESS_LIMITATIONS,
    FileKind,
    Finding,
    FindingCategory,
    ProjectInventory,
    ReadinessLevel,
    Severity,
    SourceFile,
)

AUTO = FindingCategory.AUTO_CONVERTIBLE
VERIFY = FindingCategory.REQUIRES_VERIFICATION
MANUAL = FindingCategory.MANUAL_INTERVENTION
UNSUP = FindingCategory.UNSUPPORTED


def f(category: FindingCategory, rule: str, file: str = "src/a.cu", line: int = 1) -> Finding:
    return Finding(
        rule_id=rule, category=category, severity=Severity.INFO, message=rule, file=file, line=line
    )


def inventory(*build_files: str) -> ProjectInventory:
    files = [
        SourceFile(path="src/a.cu", kind=FileKind.CUDA_SOURCE, lines=1),
        SourceFile(path="src/b.cu", kind=FileKind.CUDA_SOURCE, lines=1),
    ]
    files += [SourceFile(path=p, kind=FileKind.CMAKE, lines=1) for p in build_files]
    return ProjectInventory(root="/p", files=files)


def test_repeated_occurrences_do_not_inflate_the_score() -> None:
    # 40 cudaMalloc calls and one construct that needs a rewrite
    findings = [f(AUTO, "cuda-malloc", line=i) for i in range(40)] + [f(MANUAL, "cudnn")]
    r = readiness(findings, inventory())
    assert r.coverage == 50  # 1 converted construct of 2
    assert r.occurrence_coverage == round(100 * 40 / 41)  # the old, per-occurrence view
    assert r.score == 50 and r.level is ReadinessLevel.MANUAL_WORK
    auto = next(c for c in r.components if c.category is AUTO)
    assert (auto.constructs, auto.occurrences, auto.files) == (1, 40, 1)


def test_same_rule_in_two_files_counts_twice() -> None:
    r = readiness([f(AUTO, "cuda-malloc", "src/a.cu"), f(AUTO, "cuda-malloc", "src/b.cu")], inventory())
    auto = next(c for c in r.components if c.category is AUTO)
    assert (
        auto.constructs == 2 and auto.files == 2 and r.level is ReadinessLevel.MECHANICAL and r.score == 100
    )


def test_build_system_findings_are_listed_not_scored() -> None:
    findings = [
        f(AUTO, "cuda-malloc"),
        f(MANUAL, "cmake-cuda-language", "CMakeLists.txt"),
        f(MANUAL, "nvcc", "CMakeLists.txt"),
    ]
    r = readiness(findings, inventory("CMakeLists.txt"))
    assert r.score == 100 and r.level is ReadinessLevel.MECHANICAL
    assert r.build_items == 2 and "2 build-system change(s), not scored" in r.explanation
    assert r.by_category["MANUAL_INTERVENTION"] == 2  # still visible in the totals


def test_review_level_and_half_weight() -> None:
    r = readiness([f(AUTO, "cuda-malloc"), f(VERIFY, "warp-size-32")], inventory())
    assert r.level is ReadinessLevel.REVIEW and r.score == 75 and r.cap is None


def test_unsupported_caps_score_and_sets_blocked() -> None:
    findings = [f(AUTO, f"rule{i}") for i in range(9)] + [f(UNSUP, "inline-ptx")]
    r = readiness(findings, inventory())
    assert r.coverage == 90 and r.score == UNSUPPORTED_CAP and r.cap == UNSUPPORTED_CAP
    assert r.level is ReadinessLevel.BLOCKED and "1 unsupported construct" in (r.cap_reason or "")


def test_cap_is_not_reported_when_coverage_is_already_lower() -> None:
    r = readiness([f(UNSUP, "inline-ptx"), f(MANUAL, "cudnn")], inventory())
    assert r.score == 0 and r.cap is None and r.level is ReadinessLevel.BLOCKED


def test_only_build_findings_means_no_cuda_source() -> None:
    r = readiness([f(MANUAL, "nvcc", "CMakeLists.txt")], inventory("CMakeLists.txt"))
    assert r.score is None and r.level is ReadinessLevel.NO_CUDA and r.build_items == 1


def test_every_score_carries_its_limitations() -> None:
    r = readiness([f(AUTO, "cuda-malloc")], inventory())
    assert r.limitations == READINESS_LIMITATIONS
    assert any("not calibrated" in x for x in r.limitations)
    assert any("only constructs with a rule" in x for x in r.limitations)
