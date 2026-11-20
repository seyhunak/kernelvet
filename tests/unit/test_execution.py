from __future__ import annotations

from pathlib import Path

from conftest import FakeRunner, make_sysroot, mi300x_runner
from kernelvet.benchmark.engine import BenchmarkOptions, parse_hip_bench, run_benchmarks
from kernelvet.execution.compiler import build_project, generated_cmake, plan_direct_build
from kernelvet.execution.runner import run_tests
from kernelvet.execution.sandbox import SandboxConfig, docker_argv
from kernelvet.models.migration import BuildResult, StageStatus
from kernelvet.utils.proc import CommandResult, HostContext


def hip_tree(root: Path) -> Path:
    (root / "src").mkdir(parents=True)
    (root / "include").mkdir()
    (root / "tests").mkdir()
    (root / "include/util.h").write_text("#pragma once\nvoid scale(float*, int);\n")
    (root / "src/scale.hip").write_text(
        '#include <hip/hip_runtime.h>\n#include "util.h"\n'
        "__global__ void k(float* x){x[threadIdx.x]*=2;}\nvoid scale(float* x,int n){}\n"
    )
    (root / "src/app.cpp").write_text(
        "#include <hip/hip_runtime.h>\n#include <hipblas/hipblas.h>\nint main() { return 0; }\n"
    )
    (root / "tests/test_scale.hip").write_text(
        '#include <hip/hip_runtime.h>\n#include <cstdio>\nint main() { std::printf("PASS\\n"); return 0; }\n'
    )
    return root


def test_plan_direct_build(tmp_path: Path) -> None:
    plan = plan_direct_build(hip_tree(tmp_path / "p"), ["gfx942:sramecc+"])
    assert sorted(plan.mains) == ["src/app.cpp", "tests/test_scale.hip"]
    assert plan.units == ["src/scale.hip"]
    assert plan.libs == ["-lhipblas"]
    app = next(c for c in plan.commands if "src/app.cpp" in c)
    assert "--offload-arch=gfx942" in app and "-Iinclude" in app and app[-1] == "-lhipblas"
    assert "src/scale.hip" in app
    cmake = generated_cmake("p", plan.units, plan.mains, plan.libs, ["gfx942"])
    assert "project(p LANGUAGES CXX HIP)" in cmake and "find_package(hipblas REQUIRED)" in cmake


def fake_hipcc(fail: bool = False) -> FakeRunner:
    r = FakeRunner({"hipcc": "/opt/rocm/bin/hipcc"})

    def compile_(argv: list[str], kw: dict) -> CommandResult:
        if fail:
            return CommandResult(
                argv, 1, stderr="src/app.cpp:3:1: error: use of undeclared identifier 'cudaFree'"
            )
        out = Path(kw["cwd"]) / argv[argv.index("-o") + 1]
        out.write_text("#!fake")
        return CommandResult(argv, 0)

    r.on(lambda a: a[0] == "hipcc", compile_)
    return r


def test_build_unavailable_without_hipcc(tmp_path: Path) -> None:
    ctx = HostContext(runner=FakeRunner({}), root=tmp_path, system="Linux")
    res = build_project(ctx, hip_tree(tmp_path / "p"))
    assert res.status is StageStatus.UNAVAILABLE and "hipcc" in (res.reason or "")


def test_build_success_and_generated_cmake(tmp_path: Path) -> None:
    proj = hip_tree(tmp_path / "p")
    res = build_project(
        HostContext(runner=fake_hipcc(), root=tmp_path, system="Linux"), proj, gfx_targets=["gfx942"]
    )
    assert res.status is StageStatus.PASSED and res.method == "hipcc"
    assert sorted(Path(a).name for a in res.artifacts) == ["app", "test_scale"]
    assert (proj / ".kernelvet/CMakeLists.generated.txt").exists()
    assert not (proj / "CMakeLists.txt").exists()  # sources/build files untouched


def test_build_failure_collects_errors(tmp_path: Path) -> None:
    res = build_project(
        HostContext(runner=fake_hipcc(fail=True), root=tmp_path, system="Linux"),
        hip_tree(tmp_path / "p"),
        gfx_targets=["gfx942"],
    )
    assert res.status is StageStatus.FAILED
    assert any("undeclared identifier" in e for e in res.errors)


def test_build_notes_cuda_cmake(tmp_path: Path) -> None:
    proj = hip_tree(tmp_path / "p")
    (proj / "CMakeLists.txt").write_text("project(x LANGUAGES CXX CUDA)\n")
    res = build_project(
        HostContext(runner=fake_hipcc(), root=tmp_path, system="Linux"), proj, gfx_targets=["gfx942"]
    )
    assert "CUDA language" in (res.reason or "")


def test_docker_isolation_flags(tmp_path: Path) -> None:
    argv = docker_argv(tmp_path, ["hipcc", "x.hip"], cfg=SandboxConfig(image="img"), gpu=True)
    assert argv[:3] == ["docker", "run", "--rm"]
    for flag in ("--network=none", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--device=/dev/kfd"):
        assert flag in argv
    assert argv[-3:] == ["img", "hipcc", "x.hip"]
    assert "--device=/dev/kfd" not in docker_argv(tmp_path, ["x"], cfg=SandboxConfig())


def test_tests_not_run_without_successful_build(tmp_path: Path) -> None:
    ctx = HostContext(runner=FakeRunner({}), root=tmp_path, system="Linux")
    assert run_tests(ctx, tmp_path, None).status is StageStatus.NOT_RUN
    failed = BuildResult(project=str(tmp_path), status=StageStatus.FAILED, method="hipcc")
    assert run_tests(ctx, tmp_path, failed).status is StageStatus.NOT_RUN


def _built(tmp_path: Path) -> tuple[Path, BuildResult]:
    proj = hip_tree(tmp_path / "p")
    res = build_project(
        HostContext(runner=fake_hipcc(), root=tmp_path, system="Linux"), proj, gfx_targets=["gfx942"]
    )
    return proj, res


def test_tests_on_cpu_only_host_do_not_claim_gpu(tmp_path: Path) -> None:
    proj, build = _built(tmp_path)
    r = FakeRunner({})
    r.on(lambda a: a[0].startswith("./"), lambda a, kw: CommandResult(a, 0, stdout="PASS\n"))
    res = run_tests(HostContext(runner=r, root=tmp_path / "empty", system="Linux"), proj, build)
    assert res.status is StageStatus.PASSED
    assert [c.name for c in res.cases] == ["test_scale"]  # only test executables by default
    assert not res.gpu_present and not res.cases[0].ran_on_gpu
    assert "do not validate GPU execution" in (res.reason or "")


def test_tests_compare_reference_output_on_gpu_host(tmp_path: Path) -> None:
    proj, build = _built(tmp_path)
    (proj / "tests/expected").mkdir()
    (proj / "tests/expected/test_scale.out").write_text("PASS\n")
    sysroot = make_sysroot(tmp_path / "root")
    r = mi300x_runner()
    r.on(lambda a: a[0].startswith("./"), lambda a, kw: CommandResult(a, 0, stdout="PASS\n"))
    res = run_tests(HostContext(runner=r, root=sysroot, system="Linux"), proj, build)
    assert res.gpu_present and res.cases[0].expected_output_match is True and res.cases[0].ran_on_gpu

    r.on(lambda a: a[0].startswith("./"), lambda a, kw: CommandResult(a, 0, stdout="WRONG\n"))
    res = run_tests(HostContext(runner=r, root=sysroot, system="Linux"), proj, build)
    assert res.status is StageStatus.FAILED and res.cases[0].expected_output_match is False


# --------------------------------------------------------------------------- benchmarks


def test_parse_hip_bench() -> None:
    m = parse_hip_bench("DEVICE gfx942|x\nRESULT h2d_bandwidth GB/s 55.1 20\nRESULT bad line\n")
    assert len(m) == 1 and m[0].value == 55.1 and m[0].unit == "GB/s"


def test_benchmarks_skip_without_hardware(empty_linux_ctx: HostContext) -> None:
    report = run_benchmarks(empty_linux_ctx, BenchmarkOptions())
    assert {r.status for r in report.results} == {"SKIPPED"}
    assert all(not r.measurements for r in report.results)


def test_benchmarks_record_only_measured_numbers(mi300x_ctx: HostContext, tmp_path: Path) -> None:
    r = mi300x_ctx.runner
    assert isinstance(r, FakeRunner)
    r.on(
        lambda a: a[0].endswith("/bench"),
        CommandResult(["bench"], 0, stdout="RESULT vector_add_bandwidth GB/s 4200.5 20\n"),
    )
    report = run_benchmarks(mi300x_ctx, BenchmarkOptions(torch=False))
    hip = report.results[0]
    assert hip.status == "PASS" and [m.name for m in hip.measurements] == ["vector_add_bandwidth"]


def test_benchmark_options_problem() -> None:
    assert BenchmarkOptions().problem() is None
    assert "nothing to benchmark" in (BenchmarkOptions(hip=False, torch=False).problem() or "")
    assert "--profile" in (BenchmarkOptions(hip=False, profile=True).problem() or "")
    assert "--inference" in (BenchmarkOptions(torch=False, inference=True).problem() or "")


def test_profiler_output_outlives_the_run(mi300x_ctx: HostContext) -> None:
    r = mi300x_ctx.runner
    assert isinstance(r, FakeRunner)
    r.tools["rocprofv3"] = "/opt/rocm/bin/rocprofv3"

    def rocprof(argv: list[str], kw: dict) -> CommandResult:
        out = Path(argv[argv.index("-d") + 1])
        (out / "kernelvet_kernel_stats.csv").write_text("Name,Calls\nvector_add,20\n")
        return CommandResult(argv, 0, stdout="RESULT vector_add_bandwidth GB/s 4200.5 20\n")

    r.on(lambda a: a[0] == "rocprofv3" and "-d" in a, rocprof)
    hip = run_benchmarks(mi300x_ctx, BenchmarkOptions(torch=False, profile=True)).results[0]
    assert hip.status == "PASS"
    assert hip.profiler and hip.profiler.startswith("rocprofv3 (output: ")
    pdir = Path(hip.profiler.removeprefix("rocprofv3 (output: ").rstrip(")"))
    assert (pdir / "kernelvet_kernel_stats.csv").exists()  # not inside the deleted build folder


def test_skipped_benchmarks_mention_requested_options(empty_linux_ctx: HostContext) -> None:
    report = run_benchmarks(empty_linux_ctx, BenchmarkOptions(profile=True, inference=True))
    hip, torch = report.results
    assert hip.status == "SKIPPED" and hip.profiler and "not run" in hip.profiler
    assert "--profile" in (hip.reason or "")
    assert torch.status == "SKIPPED" and "--inference" in (torch.reason or "")
