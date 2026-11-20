"""Run a migrated project's tests and record what actually happened.

* CMake builds with ``enable_testing()`` run through ``ctest``.
* Direct hipcc builds run each built executable whose source lives in a test
  directory (or every executable with ``--all``).
* If ``<name>.expected`` / ``tests/expected/<name>.out`` exists, stdout is
  compared against it (reference output).

GPU execution is observed, not assumed. Every test runs with the HIP runtime's
API log switched on (``AMD_LOG_LEVEL=3``, ``AMD_LOG_MASK=0x1``), which writes
each HIP API call and its result to stderr. A test only counts as having run on
the GPU (``gpu_evidence = executed``, ``ran_on_gpu``) when an AMD GPU is present
*and* its log shows at least one kernel launch that returned ``hipSuccess``. A
passing test that never launched a kernel (CPU-only code, or a binary that does
not use the HIP runtime) is ``detected``: the GPU is there, but nothing shows the
test used it. Detecting a GPU is never treated as executing on it.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from kernelvet.diagnose.hardware import discover_gpus
from kernelvet.execution.compiler import BUILD_SUBDIR
from kernelvet.execution.sandbox import SandboxConfig, docker_argv
from kernelvet.migration.scanner import TEST_PATH, scan_project
from kernelvet.models.common import Evidence
from kernelvet.models.migration import BuildResult, StageStatus, TestCaseResult, TestRunResult
from kernelvet.utils.proc import HostContext


def _reference_for(project: Path, name: str, source: str | None) -> Path | None:
    candidates = [project / "tests" / "expected" / f"{name}.out", project / f"{name}.expected"]
    if source:
        candidates.append(project / Path(source).with_suffix(".expected"))
    return next((c for c in candidates if c.is_file()), None)


# HIP runtime (CLR) API logging: ``<api>: Returned <hipError> : ...`` for every call.
HIP_LOG_ENV = {"AMD_LOG_LEVEL": "3", "AMD_LOG_MASK": "0x1"}

# Launch APIs only. hipLaunchHostFunc and friends run host code and are deliberately excluded.
_KERNEL_LAUNCH_OK = re.compile(
    r"\b(?:hip(?:Ext|Hcc)?(?:Module)?Launch(?:Cooperative)?Kernel\w*"
    r"|hipExtLaunchMultiKernelMultiDevice|hipGraphLaunch|hipLaunchByPtr)"
    r"\s*:\s*Returned\s+hipSuccess\b"
)


# CLR log lines look like ``:3:hip_memory.cpp :1234: 5678 us: [pid:1 tid:0x..] hipMalloc ( ... )``.
_HIP_LOG_LINE = re.compile(r"^:\d:.*$\n?", re.MULTILINE)


def strip_hip_log(text: str) -> str:
    """Drop HIP runtime log lines so stored output shows the test's own messages."""
    return _HIP_LOG_LINE.sub("", text)


def count_kernel_launches(log: str) -> int:
    """Successful kernel launches recorded in a HIP runtime API log."""
    return len(_KERNEL_LAUNCH_OK.findall(log))


def _gpu_evidence(gpu_present: bool, launches: int) -> Evidence:
    if not gpu_present:
        return Evidence.NONE
    return Evidence.EXECUTED if launches else Evidence.DETECTED


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
        return docker_argv(project, cmd, cfg=cfg, gpu=True, env=HIP_LOG_ENV) if docker else cmd

    run_env = {**ctx.env, **HIP_LOG_ENV}

    cases: list[TestCaseResult] = []
    if build.method == "cmake" and (project / BUILD_SUBDIR / "cmake" / "CTestTestfile.cmake").exists():
        # -V prints every test's output, so the HIP log of passing tests is captured too.
        cmd = ["ctest", "--test-dir", f"{BUILD_SUBDIR}/cmake", "--output-on-failure", "-V"]
        t0 = time.monotonic()
        r = ctx.run(argv_for(cmd), timeout=timeout, cwd=project, env=run_env)
        launches = count_kernel_launches(r.stdout + r.stderr)
        evidence = _gpu_evidence(gpu_present, launches)
        cases.append(
            TestCaseResult(
                name="ctest",
                command=cmd,
                status=StageStatus.PASSED if r.ok else StageStatus.FAILED,
                returncode=r.returncode,
                stdout=strip_hip_log(r.stdout)[-20000:],
                stderr=strip_hip_log(r.stderr)[-20000:],
                duration_s=time.monotonic() - t0,
                gpu_evidence=evidence,
                gpu_kernel_launches=launches,
                ran_on_gpu=r.ok and evidence is Evidence.EXECUTED,
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
            r = ctx.run(argv_for(cmd), timeout=timeout, cwd=project, env=run_env)
            launches = count_kernel_launches(r.stderr)
            evidence = _gpu_evidence(gpu_present, launches)
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
                    stdout=strip_hip_log(r.stdout)[-20000:],
                    stderr=strip_hip_log(r.stderr)[-20000:],
                    duration_s=time.monotonic() - t0,
                    expected_output_match=match,
                    gpu_evidence=evidence,
                    gpu_kernel_launches=launches,
                    ran_on_gpu=ok and evidence is Evidence.EXECUTED,
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
    if not gpu_present:
        reason: str | None = "no AMD GPU detected: results do not validate GPU execution"
    elif not any(c.ran_on_gpu for c in cases):
        reason = (
            "an AMD GPU is present, but no passing test launched a HIP kernel (checked in the HIP "
            "runtime log): results do not validate GPU execution"
        )
    else:
        reason = None
    return TestRunResult(
        project=str(project),
        status=StageStatus.FAILED if failed else StageStatus.PASSED,
        cases=cases,
        gpu_present=gpu_present,
        reason=reason,
    )
