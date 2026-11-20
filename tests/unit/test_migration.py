from __future__ import annotations

from pathlib import Path

import pytest

from conftest import FakeRunner, fake_hipify, tree_digest
from kernelvet.migration import (
    HipifyUnavailable,
    analyze_project,
    audit_project,
    build_plan,
    choose_translator,
    convert_project,
    scan_project,
)
from kernelvet.migration.hipify import parse_hipify_output, translator_argv
from kernelvet.migration.rules import iter_matches, strip_comments, uses_cuda
from kernelvet.migration.workflow import assemble_report, compute_stages, save_state
from kernelvet.models.migration import (
    FileKind,
    FindingCategory,
    StageStatus,
    TestCaseResult,
    TestRunResult,
)
from kernelvet.utils.paths import PathError, safe_output_dir
from kernelvet.utils.proc import HostContext

# --------------------------------------------------------------------------- scanner


def test_scanner_inventory(cuda_project: Path) -> None:
    inv = scan_project(cuda_project)
    kinds = {f.path: f.kind for f in inv.files}
    assert kinds["src/vector_add.cu"] is FileKind.CUDA_SOURCE
    assert kinds["include/common.cuh"] is FileKind.CUDA_HEADER
    assert kinds["src/gemm.cpp"] is FileKind.CUDA_RELATED_CPP
    assert kinds["CMakeLists.txt"] is FileKind.CMAKE
    assert kinds["Makefile"] is FileKind.MAKEFILE
    assert "README.md" not in kinds
    assert inv.test_files == ["tests/test_vector_add.cu"]
    deps = {d.name: d for d in inv.dependencies}
    assert {"CUDA Runtime", "cuBLAS", "NVTX"} <= set(deps)
    assert deps["cuBLAS"].hip_alternative.startswith("hipBLAS")
    builds = {b.kind: b for b in inv.build_systems}
    assert builds["cmake"].uses_cuda_language and builds["make"].cuda_arch_flags == ["sm_80"]
    vec = next(f for f in inv.files if f.path == "src/vector_add.cu")
    assert vec.kernels == 1 and vec.cuda_api_calls >= 8


def test_scanner_skips_build_dirs_and_binaries(cuda_project: Path) -> None:
    (cuda_project / "build").mkdir()
    (cuda_project / "build" / "x.cu").write_text("__global__ void k(){}")
    (cuda_project / "blob.cu").write_bytes(b"\x00\x01binary")
    inv = scan_project(cuda_project)
    assert not any(f.path.startswith("build/") for f in inv.files)
    assert any("blob.cu" in s for s in inv.skipped)


def test_scanner_rejects_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(PathError):
        scan_project(tmp_path / "nope")


# --------------------------------------------------------------------------- rules & analyzer


def test_comments_are_ignored_but_lines_kept() -> None:
    src = "int a;\n// cudaMalloc(&p, 1);\n/* cudaFree(p);\n */ cudaMemcpy(a, b, n, k);\n"
    clean = strip_comments(src)
    assert clean.count("\n") == src.count("\n")
    found = list(iter_matches(src, "f.cu", phase="analyze", target="source"))
    assert [(f.symbol, f.line) for f in found if f.rule_id == "cuda.runtime_api"] == [("cudaMemcpy", 4)]


def test_uses_cuda() -> None:
    assert uses_cuda("#include <cuda_runtime.h>")
    assert uses_cuda("__global__ void k() {}")
    assert not uses_cuda("// cudaMalloc in a comment\nint main() {}")


def test_analysis_classifies_constructs(cuda_project: Path) -> None:
    a = analyze_project(cuda_project)
    by_rule: dict[str, set[FindingCategory]] = {}
    for f in a.findings:
        by_rule.setdefault(f.rule_id, set()).add(f.category)
    assert by_rule["cuda.inline_ptx"] == {FindingCategory.UNSUPPORTED}
    assert by_rule["cuda.warp_sync_intrinsic"] == {FindingCategory.REQUIRES_VERIFICATION}
    assert by_rule["cuda.full_mask"] == {FindingCategory.REQUIRES_VERIFICATION}
    assert by_rule["cuda.warp_size_literal"] == {FindingCategory.REQUIRES_VERIFICATION}
    assert by_rule["cuda.managed_memory"] == {FindingCategory.REQUIRES_VERIFICATION}
    assert by_rule["lib.cublas"] == {FindingCategory.AUTO_CONVERTIBLE}
    assert by_rule["lib.nvtx"] == {FindingCategory.MANUAL_INTERVENTION}
    assert by_rule["build.cuda_language"] == {FindingCategory.MANUAL_INTERVENTION}
    cublas = next(f for f in a.findings if f.symbol == "cublasSgemm")
    assert cublas.hip_equivalent == "hipblasSgemm" and cublas.file == "src/gemm.cpp" and cublas.line
    malloc = next(f for f in a.findings if f.symbol == "cudaMalloc")
    assert malloc.hip_equivalent == "hipMalloc"


def test_readiness_is_capped_by_unsupported_and_explained(cuda_project: Path) -> None:
    r = analyze_project(cuda_project).readiness
    assert 0 < r.score <= 60
    assert "not correctness" in r.explanation
    assert r.by_category["UNSUPPORTED"] == 1


def test_readiness_without_cuda(tmp_path: Path) -> None:
    (tmp_path / "main.cpp").write_text("int main() { return 0; }\n")
    r = analyze_project(tmp_path).readiness
    assert r.score == 0 and r.total_findings == 0


# --------------------------------------------------------------------------- planner / translator choice


def ctx_with(tools: dict[str, str], tmp_path: Path, cuda_headers: bool = False) -> HostContext:
    root = tmp_path / "sysroot"
    if cuda_headers:
        (root / "usr/local/cuda/include").mkdir(parents=True)
        (root / "usr/local/cuda/include/cuda_runtime.h").write_text("")
    root.mkdir(parents=True, exist_ok=True)
    return HostContext(runner=FakeRunner(tools), root=root, env={}, system="Linux")


def test_choose_translator(tmp_path: Path) -> None:
    both = {"hipify-clang": "/x/hipify-clang", "hipify-perl": "/x/hipify-perl"}
    assert choose_translator(ctx_with({}, tmp_path)).name is None
    assert choose_translator(ctx_with({"hipify-perl": "/x"}, tmp_path)).name == "hipify-perl"
    no_headers = choose_translator(ctx_with(both, tmp_path))
    assert no_headers.name == "hipify-perl" and "no CUDA headers" in no_headers.reason
    c = choose_translator(ctx_with(both, tmp_path / "b", cuda_headers=True))
    assert c.name == "hipify-clang" and c.cuda_path == "/usr/local/cuda"
    assert choose_translator(ctx_with({"hipify-perl": "/x"}, tmp_path / "c"), "hipify-clang").name is None


def test_plan_without_hipify_is_blocking(cuda_project: Path, tmp_path: Path) -> None:
    p = build_plan(analyze_project(cuda_project), ctx_with({}, tmp_path), offload_arch=["gfx942"])
    assert p.translator is None
    assert p.steps[0].title == "Install HIPIFY" and p.steps[0].blocking
    assert any(s.title == "Resolve blockers" and s.blocking for s in p.steps)
    assert "src/reduce.cu" in p.files_to_migrate and "CMakeLists.txt" in p.files_to_copy
    assert p.api_conversions["cudaMalloc"] == "hipMalloc"
    assert "gfx942" in p.build_recommendation
    assert {d.name for d in p.dependency_replacements} >= {"cuBLAS", "NVTX"}


# --------------------------------------------------------------------------- HIPIFY conversion


def test_convert_preserves_source_and_structure(cuda_project: Path, tmp_path: Path) -> None:
    before = tree_digest(cuda_project)
    ctx = HostContext(runner=fake_hipify(fail={"gemm.cpp"}), root=tmp_path, system="Linux")
    out = tmp_path / "hip-out"
    res = convert_project(ctx, scan_project(cuda_project), out)
    assert tree_digest(cuda_project) == before  # original untouched
    assert (out / "src/vector_add.cu").read_text().count("hipMalloc") >= 3
    assert (out / "CMakeLists.txt").read_text() == (cuda_project / "CMakeLists.txt").read_text()
    assert (out / "README.md").exists()
    status = {f.source: f for f in res.files}
    assert status["src/vector_add.cu"].status == "converted"
    assert status["src/vector_add.cu"].replacements == 7 and status["src/vector_add.cu"].unconverted == 1
    assert status["src/vector_add.cu"].warnings
    failed = status["src/gemm.cpp"]
    assert failed.status == "failed" and failed.errors
    # failed file is preserved (copied unchanged) rather than silently dropped
    assert (out / "src/gemm.cpp").read_text() == (cuda_project / "src/gemm.cpp").read_text()
    assert res.log_path and Path(res.log_path).exists()
    assert (out / ".kernelvet/conversion.json").exists()


def test_convert_dry_run_writes_nothing(cuda_project: Path, tmp_path: Path) -> None:
    ctx = HostContext(runner=fake_hipify(), root=tmp_path, system="Linux")
    out = tmp_path / "dry"
    res = convert_project(ctx, scan_project(cuda_project), out, dry_run=True)
    assert not out.exists()
    assert {f.status for f in res.files} == {"planned", "planned-copy"}


def test_convert_without_hipify_raises(cuda_project: Path, tmp_path: Path) -> None:
    ctx = HostContext(runner=FakeRunner({}), root=tmp_path, system="Linux")
    with pytest.raises(HipifyUnavailable):
        convert_project(ctx, scan_project(cuda_project), tmp_path / "out")


@pytest.mark.parametrize("rel", [".", "sub", "src/inner"])
def test_output_dir_cannot_overlap_source(cuda_project: Path, rel: str) -> None:
    with pytest.raises(PathError):
        safe_output_dir(cuda_project, cuda_project / rel)
    with pytest.raises(PathError):
        safe_output_dir(cuda_project / "src", cuda_project)


def test_output_dir_must_be_empty_unless_forced(cuda_project: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "x").write_text("")
    with pytest.raises(PathError):
        safe_output_dir(cuda_project, out)
    assert safe_output_dir(cuda_project, out, allow_existing=True) == out.resolve()


def test_parse_hipify_output() -> None:
    conv, unconv, warns, errs = parse_hipify_output(
        "CONVERTED refs count: 12\nUNCONVERTED refs count: 2\nwarning: x\nerror: y\n"
    )
    assert (conv, unconv, len(warns), len(errs)) == (12, 2, 1, 1)


def test_hipify_clang_argv_includes_cuda_path(tmp_path: Path) -> None:
    from kernelvet.migration.hipify import TranslatorChoice

    argv = translator_argv(
        TranslatorChoice("hipify-clang", "", "/usr/local/cuda"), tmp_path / "a.cu", tmp_path / "o.cu", ["-I."]
    )
    assert argv[0] == "hipify-clang" and "--cuda-path=/usr/local/cuda" in argv and argv[-1] == "-I."


# --------------------------------------------------------------------------- audit


def test_audit_finds_leftovers(hip_project: Path) -> None:
    res = audit_project(hip_project, installed_libraries=["HIP runtime", "rocBLAS"])
    rules = {f.rule_id for f in res.findings}
    assert {
        "audit.leftover_runtime_call",
        "audit.leftover_cuda_include",
        "audit.leftover_nvidia_lib",
        "cuda.warp_size_literal",
        "cuda.full_mask",
        "cuda.memcpy_symbol",
        "build.cuda_language",
        "audit.missing_dependency",
    } <= rules
    leftover = next(f for f in res.findings if f.rule_id == "audit.leftover_runtime_call")
    assert leftover.file == "src/main.hip" and leftover.line == 21
    assert "semantically correct" in res.note


def test_audit_clean_hip_code(tmp_path: Path) -> None:
    (tmp_path / "k.hip").write_text(
        "#include <hip/hip_runtime.h>\n__global__ void k(float* x) { x[threadIdx.x] *= 2; }\n"
        "int main() { float* d; hipMalloc(&d, 4); hipFree(d); }\n"
    )
    assert audit_project(tmp_path, installed_libraries=[]).findings == []


# --------------------------------------------------------------------------- workflow / stages


def test_gpu_validation_requires_amd_gpu() -> None:
    passed_cpu = TestRunResult(
        project="p",
        status=StageStatus.PASSED,
        gpu_present=False,
        cases=[TestCaseResult(name="t", command=["./t"], status=StageStatus.PASSED)],
    )
    s = compute_stages(None, None, None, passed_cpu)
    assert s.tests is StageStatus.PASSED
    assert s.gpu_runtime_validation is StageStatus.UNAVAILABLE

    passed_gpu = TestRunResult(
        project="p",
        status=StageStatus.PASSED,
        gpu_present=True,
        cases=[TestCaseResult(name="t", command=["./t"], status=StageStatus.PASSED, ran_on_gpu=True)],
    )
    assert compute_stages(None, None, None, passed_gpu).gpu_runtime_validation is StageStatus.PASSED


def test_assemble_report_reads_persisted_state(cuda_project: Path, tmp_path: Path) -> None:
    ctx = HostContext(runner=fake_hipify(), root=tmp_path, system="Linux")
    out = tmp_path / "hip-out"
    res = convert_project(ctx, scan_project(cuda_project), out)
    save_state(out, "conversion", res)
    save_state(out, "audit", audit_project(out))
    report = assemble_report(out)
    assert report.analysis is not None and report.analysis.inventory.root == str(cuda_project.resolve())
    assert report.stages.source_conversion is StageStatus.PASSED
    assert report.stages.static_audit is StageStatus.FAILED  # inline PTX etc. remain
    assert report.stages.compilation is StageStatus.NOT_RUN
    assert report.recommendations


def test_hip_names_that_are_not_a_prefix_swap() -> None:
    src = (
        "cudaDeviceProp prop;\n"
        "cudaThreadSynchronize();\n"
        "cudaThreadExit();\n"
        "cudaStream_t s;\n"
        "cudaMalloc(&p, n);\n"
    )
    hip = {f.symbol: f.hip_equivalent for f in iter_matches(src, "a.cu", phase="analyze", target="source")}
    assert hip["cudaDeviceProp"] == "hipDeviceProp_t"
    assert hip["cudaThreadSynchronize"] == "hipDeviceSynchronize"
    assert hip["cudaThreadExit"] == "hipDeviceReset"
    # the generic prefix swap still applies everywhere else
    assert hip["cudaStream_t"] == "hipStream_t"
    assert hip["cudaMalloc"] == "hipMalloc"


def test_choose_translator_rejects_unknown_preference(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown translator"):
        choose_translator(ctx_with({"hipify-perl": "/x"}, tmp_path), "foo")
