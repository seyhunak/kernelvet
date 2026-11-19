"""Run a migrated project's tests and record what actually happened.

* CMake builds with ``enable_testing()`` run through ``ctest``.
* Direct hipcc builds run each built executable whose source lives in a test
  directory (or every executable with ``--all``).
* If ``<name>.expected`` / ``tests/expected/<name>.out`` exists, stdout is
  compared against it (reference output).

``ran_on_gpu`` is only set when an AMD GPU was detected on the host; a passing
test on a GPU-less host is never reported as GPU validation.
"""

from __future__ import annotations

import time
from pathlib import Path

from kernelvet.diagnose.hardware import discover_gpus
from kernelvet.execution.compiler import BUILD_SUBDIR
from kernelvet.execution.sandbox import SandboxConfig, docker_argv
from kernelvet.migration.scanner import TEST_PATH, scan_project
from kernelvet.models.migration import BuildResult, StageStatus, TestCaseResult, TestRunResult
from kernelvet.utils.proc import HostContext


def _reference_for(project: Path, name: str, source: str | None) -> Path | None:
    candidates = [project / "tests" / "expected" / f"{name}.out", project / f"{name}.expected"]
    if source:
        candidates.append(project / Path(source).with_suffix(".expected"))
    return next((c for c in candidates if c.is_file()), None)


def _normalise(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def run_tests(
    ctx: HostContext,
    path: str | Path,
    build: BuildResult | None,
    *,
    run_all: bool = False,
    isolation: str = "none",
    sandbox: SandboxConfig | None = None,
    timeout: float = 300.0,
) -> TestRunResult:
    project = Path(path).resolve()
    if build is None or build.status is not StageStatus.PASSED:
        return TestRunResult(
            project=str(project),
            status=StageStatus.NOT_RUN,
            reason="the project has not been built successfully; run `kernelvet migrate build` first",
        )
    gpus = [g for g in discover_gpus(ctx).gpus if g.vendor_is_amd]
    gpu_present = bool(gpus)
    docker = isolation == "docker"
    cfg = sandbox or SandboxConfig()

    def argv_for(cmd: list[str]) -> list[str]:
        return docker_argv(project, cmd, cfg=cfg, gpu=True) if docker else cmd

    cases: list[TestCaseResult] = []
    if build.method == "cmake" and (project / BUILD_SUBDIR / "cmake" / "CTestTestfile.cmake").exists():
        cmd = ["ctest", "--test-dir", f"{BUILD_SUBDIR}/cmake", "--output-on-failure"]
        t0 = time.monotonic()
        r = ctx.run(argv_for(cmd), timeout=timeout, cwd=project)
        cases.append(
            TestCaseResult(
                name="ctest",
                command=cmd,
                status=StageStatus.PASSED if r.ok else StageStatus.FAILED,
                returncode=r.returncode,
                stdout=r.stdout[-20000:],
                stderr=r.stderr[-20000:],
                duration_s=time.monotonic() - t0,
                ran_on_gpu=gpu_present and r.ok,
            )
        )
    else:
        inv = scan_project(project)
        sources = {Path(f.path).stem: f.path for f in inv.files}
        for exe in build.artifacts:
            name = Path(exe).name
            src = sources.get(name)
            is_test = bool(src and TEST_PATH.search(src)) or name.startswith("test")
            if not (is_test or run_all):
                continue
            cmd = [f"./{exe}"]
            t0 = time.monotonic()
            r = ctx.run(argv_for(cmd), timeout=timeout, cwd=project)
            ref = _reference_for(project, name, src)
            match = None
            if ref is not None:
                match = _normalise(r.stdout) == _normalise(ref.read_text(errors="replace"))
            ok = r.ok and match is not False
            cases.append(
                TestCaseResult(
                    name=name,
                    command=cmd,
                    status=StageStatus.PASSED if ok else StageStatus.FAILED,
                    returncode=r.returncode,
                    stdout=r.stdout[-20000:],
                    stderr=r.stderr[-20000:],
                    duration_s=time.monotonic() - t0,
                    expected_output_match=match,
                    ran_on_gpu=gpu_present and r.ok,
                )
            )
    if not cases:
        return TestRunResult(
            project=str(project),
            status=StageStatus.NOT_RUN,
            gpu_present=gpu_present,
            reason="no test executables found (sources under tests/ or named test_*); "
            "use --all to run every built executable",
        )
    failed = any(c.status is StageStatus.FAILED for c in cases)
    reason = None if gpu_present else ("no AMD GPU detected: results do not validate GPU execution")
    return TestRunResult(
        project=str(project),
        status=StageStatus.FAILED if failed else StageStatus.PASSED,
        cases=cases,
        gpu_present=gpu_present,
        reason=reason,
    )
