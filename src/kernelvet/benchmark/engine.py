"""Benchmark runner. Measurements only come from real executions on this host."""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kernelvet.benchmark.workloads import HIP_BENCH_SOURCE, TORCH_BENCH_SOURCE
from kernelvet.config.settings import data_dir
from kernelvet.diagnose.engine import collect_inventory
from kernelvet.diagnose.runtime import offload_arch_flags
from kernelvet.models.report import BenchmarkMeasurement, BenchmarkReport, BenchmarkResult
from kernelvet.utils.proc import HostContext, ProgressCallback, noop_progress

ITERATIONS_MIN, ITERATIONS_MAX = 3, 1000  # timed iterations per measurement


@dataclass
class BenchmarkOptions:
    hip: bool = True
    torch: bool = True
    inference: bool = False
    iterations: int = 20
    profile: bool = False  # wrap the HIP benchmark in rocprofv3 when available
    profile_dir: str | None = None
    python: str | None = None

    def problem(self) -> str | None:
        """Why this combination of options cannot run anything useful, or None."""
        if not self.hip and not self.torch:
            return "nothing to benchmark: both the HIP and the PyTorch benchmarks are disabled"
        if self.profile and not self.hip:
            return "--profile profiles the HIP benchmark, which is disabled"
        if self.inference and not self.torch:
            return "--inference is part of the PyTorch benchmark, which is disabled"
        return None


def default_profile_dir() -> Path:
    """A new, lasting folder for rocprofv3 output (the build folder is temporary)."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return data_dir() / "profiles" / stamp


def parse_hip_bench(stdout: str) -> list[BenchmarkMeasurement]:
    out = []
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0] == "RESULT":
            try:
                out.append(
                    BenchmarkMeasurement(
                        name=parts[1], unit=parts[2], value=float(parts[3]), iterations=int(parts[4])
                    )
                )
            except ValueError:
                continue
    return out


def run_hip_benchmark(ctx: HostContext, opts: BenchmarkOptions, gfx_targets: list[str]) -> BenchmarkResult:
    res = BenchmarkResult(
        id="hip.core", name="HIP kernels, memory and SGEMM", backend="hip", status="SKIPPED"
    )
    if not ctx.which("hipcc"):
        res.reason = "hipcc not found"
        return res
    if not gfx_targets:
        res.reason = "no AMD GPU detected"
        return res
    with tempfile.TemporaryDirectory(prefix="kernelvet-bench-") as tmp:
        src = Path(tmp) / "bench.hip"
        exe = Path(tmp) / "bench"
        src.write_text("#include <chrono>\n#include <cstdlib>\n" + HIP_BENCH_SOURCE)
        build = ctx.run(
            ["hipcc", "-O3", *offload_arch_flags(gfx_targets), "-o", str(exe), str(src)], timeout=300
        )
        if not build.ok:
            res.status, res.reason = "FAIL", f"benchmark did not compile: {build.describe_failure()}"
            return res
        argv = [str(exe), str(opts.iterations)]
        if opts.profile:
            if ctx.which("rocprofv3"):
                pdir = Path(opts.profile_dir or default_profile_dir()).resolve()
                pdir.mkdir(parents=True, exist_ok=True)
                argv = [
                    "rocprofv3",
                    "--kernel-trace",
                    "--stats",
                    "-d",
                    str(pdir),
                    "-o",
                    "kernelvet",
                    "--",
                    *argv,
                ]
                res.profiler = f"rocprofv3 (output: {pdir})"
            else:
                res.profiler = "requested but rocprofv3 is not installed"
        run = ctx.run(argv, timeout=600)
    res.measurements = parse_hip_bench(run.stdout)
    if run.ok and res.measurements:
        res.status = "PASS"
    else:
        res.status = "FAIL"
        res.reason = run.describe_failure() + (
            "; " + " ".join(line for line in run.stdout.splitlines() if line.startswith("ERROR"))
        )
    return res


def run_torch_benchmark(ctx: HostContext, opts: BenchmarkOptions, torch_rocm: bool) -> BenchmarkResult:
    res = BenchmarkResult(
        id="torch.core", name="PyTorch GPU tensor operations", backend="pytorch", status="SKIPPED"
    )
    if not torch_rocm:
        res.reason = "no ROCm build of PyTorch detected"
        return res
    run = ctx.run(
        [
            opts.python or sys.executable,
            "-c",
            TORCH_BENCH_SOURCE,
            str(opts.iterations),
            "1" if opts.inference else "0",
        ],
        timeout=900,
    )
    try:
        data = json.loads(run.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        res.status, res.reason = "FAIL", run.describe_failure()
        return res
    if "error" in data:
        res.status, res.reason = "FAIL", data["error"]
        return res
    res.measurements = [BenchmarkMeasurement(**r) for r in data["results"]]
    res.status = "PASS"
    if data.get("device"):
        res.name += f" on {data['device']}"
    return res


def run_benchmarks(
    ctx: HostContext | None = None,
    opts: BenchmarkOptions | None = None,
    progress: ProgressCallback = noop_progress,
) -> BenchmarkReport:
    ctx = ctx or HostContext()
    opts = opts or BenchmarkOptions()
    progress("Collecting configuration metadata", 0.05)
    host, gpus, sw = collect_inventory(ctx, torch=opts.torch, python=opts.python)
    targets = [g.gfx_target for g in gpus if g.vendor_is_amd and g.gfx_target]
    report = BenchmarkReport(host=host, gpus=gpus, software=sw)
    if opts.hip:
        progress("Compiling and running HIP benchmarks", 0.2)
        hip = run_hip_benchmark(ctx, opts, [t for t in targets if t])
        if opts.profile and hip.profiler is None:
            # the benchmark stopped before rocprofv3 could run; say so instead of dropping the request
            hip.profiler = f"requested but not run (HIP benchmark {hip.status})"
            hip.reason = _with_note(hip.reason, "requested profiling (--profile) was not run")
        report.results.append(hip)
    if opts.torch:
        progress("Running PyTorch benchmarks", 0.6)
        torch = run_torch_benchmark(ctx, opts, bool(sw.torch and sw.torch.is_rocm_build))
        if opts.inference and torch.status != "PASS":
            torch.reason = _with_note(
                torch.reason, "the requested inference benchmark (--inference) produced no results"
            )
        report.results.append(torch)
    progress("Done", 1.0)
    return report


def _with_note(reason: str | None, note: str) -> str:
    return f"{reason}; {note}" if reason else note
