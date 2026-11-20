"""Runtime verification: actually execute code on the GPU.

Two probes:

* a HIP C++ program compiled with ``hipcc`` that initialises the runtime,
  enumerates devices, allocates memory, copies data both ways, launches a
  vector-add kernel and verifies the result on the host;
* a PyTorch script that runs tensor operations on the GPU and compares them
  with a CPU reference.

Each probe step becomes its own check with ``Evidence.EXECUTED``. When a probe
cannot run, every step is SKIPPED with the reason. Nothing is assumed to pass.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

from kernelvet.models.common import CheckResult, CheckStatus, Evidence
from kernelvet.models.system import TorchInfo
from kernelvet.utils.proc import HostContext

HIP_PROBE_SOURCE = r"""
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define REPORT(step, ok, ...) do { std::printf("STEP %s %s ", step, (ok) ? "OK" : "FAIL"); \
    std::printf(__VA_ARGS__); std::printf("\n"); std::fflush(stdout); } while (0)

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    REPORT("init", err == hipSuccess, "%s", hipGetErrorString(err));
    if (err != hipSuccess) return 2;
    REPORT("enumerate", count > 0, "devices=%d", count);
    if (count == 0) return 3;
    for (int d = 0; d < count; ++d) {
        hipDeviceProp_t p;
        if (hipGetDeviceProperties(&p, d) == hipSuccess)
            std::printf("DEVICE %d %s|%s|%zu\n", d, p.gcnArchName, p.name, (size_t)p.totalGlobalMem);
    }
    err = hipSetDevice(0);
    if (err != hipSuccess) { REPORT("alloc", false, "hipSetDevice: %s", hipGetErrorString(err)); return 4; }

    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);
    std::vector<float> a(n), b(n), c(n, 0.f);
    for (int i = 0; i < n; ++i) { a[i] = (float)i * 0.5f; b[i] = (float)(n - i); }
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    err = hipMalloc(&da, bytes);
    if (err == hipSuccess) err = hipMalloc(&db, bytes);
    if (err == hipSuccess) err = hipMalloc(&dc, bytes);
    REPORT("alloc", err == hipSuccess, "%zu bytes x3: %s", bytes, hipGetErrorString(err));
    if (err != hipSuccess) return 5;

    err = hipMemcpy(da, a.data(), bytes, hipMemcpyHostToDevice);
    if (err == hipSuccess) err = hipMemcpy(db, b.data(), bytes, hipMemcpyHostToDevice);
    REPORT("h2d", err == hipSuccess, "%s", hipGetErrorString(err));
    if (err != hipSuccess) return 6;

    vector_add<<<(n + 255) / 256, 256>>>(da, db, dc, n);
    err = hipGetLastError();
    if (err == hipSuccess) err = hipDeviceSynchronize();
    REPORT("kernel", err == hipSuccess, "%s", hipGetErrorString(err));
    if (err != hipSuccess) return 7;

    err = hipMemcpy(c.data(), dc, bytes, hipMemcpyDeviceToHost);
    REPORT("d2h", err == hipSuccess, "%s", hipGetErrorString(err));
    if (err != hipSuccess) return 8;

    int bad = 0;
    for (int i = 0; i < n; ++i) if (std::fabs(c[i] - (a[i] + b[i])) > 1e-3f) ++bad;
    REPORT("verify", bad == 0, "mismatches=%d of %d", bad, n);
    hipFree(da); hipFree(db); hipFree(dc);
    return bad == 0 ? 0 : 9;
}
"""

HIP_STEPS: list[tuple[str, str, str]] = [
    ("init", "runtime.hip_init", "HIP Initialization"),
    ("enumerate", "runtime.device_enum", "Device Enumeration"),
    ("alloc", "runtime.memory_alloc", "GPU Memory Allocation"),
    ("h2d", "runtime.memcpy_h2d", "Host-to-Device Transfer"),
    ("kernel", "runtime.kernel_exec", "Kernel Execution"),
    ("d2h", "runtime.memcpy_d2h", "Device-to-Host Transfer"),
    ("verify", "runtime.vector_add_verify", "Vector Add Result Verification"),
]


def _skip_all(reason: str, remediation: str | None = None) -> list[CheckResult]:
    return [
        CheckResult(
            id=cid,
            name=name,
            category="runtime",
            status=CheckStatus.SKIPPED,
            evidence=Evidence.NONE,
            summary=reason,
            remediation=remediation,
        )
        for _, cid, name in HIP_STEPS
    ]


def parse_probe_output(stdout: str) -> tuple[dict[str, tuple[bool, str]], list[str]]:
    steps: dict[str, tuple[bool, str]] = {}
    devices: list[str] = []
    for line in stdout.splitlines():
        parts = line.split(" ", 3)
        if len(parts) >= 3 and parts[0] == "STEP":
            steps[parts[1]] = (parts[2] == "OK", parts[3] if len(parts) > 3 else "")
        elif line.startswith("DEVICE "):
            devices.append(line[len("DEVICE ") :])
    return steps, devices


def offload_arch_flags(targets: list[str]) -> list[str]:
    uniq = sorted({t.split(":")[0] for t in targets if t and t.startswith("gfx")})
    return [f"--offload-arch={t}" for t in uniq]


def run_hip_probe(
    ctx: HostContext,
    *,
    gfx_targets: list[str] | None = None,
    workdir: Path | None = None,
    compile_timeout: float = 180.0,
    run_timeout: float = 60.0,
) -> list[CheckResult]:
    if not ctx.which("hipcc"):
        return _skip_all(
            "hipcc not found; the HIP probe needs a ROCm toolchain",
            "Install ROCm (HIP SDK) and ensure hipcc is on PATH",
        )
    with tempfile.TemporaryDirectory(prefix="kernelvet-probe-", dir=workdir) as tmp:
        src = Path(tmp) / "probe.hip"
        exe = Path(tmp) / "probe"
        src.write_text(HIP_PROBE_SOURCE)
        argv = ["hipcc", "-O2", *offload_arch_flags(gfx_targets or []), "-o", str(exe), str(src)]
        t0 = time.monotonic()
        build = ctx.run(argv, timeout=compile_timeout)
        if not build.ok:
            return _skip_all(
                f"HIP probe did not compile: {build.describe_failure()}",
                "Check that hipcc works: `hipcc --version`; see the compiler output in the JSON report",
            )
        compile_ms = (time.monotonic() - t0) * 1000
        t1 = time.monotonic()
        res = ctx.run([str(exe)], timeout=run_timeout)
        run_ms = (time.monotonic() - t1) * 1000

    steps, devices = parse_probe_output(res.stdout)
    checks: list[CheckResult] = []
    blocked_by: str | None = None
    for key, cid, name in HIP_STEPS:
        if key in steps:
            ok, detail = steps[key]
            checks.append(
                CheckResult(
                    id=cid,
                    name=name,
                    category="runtime",
                    status=CheckStatus.PASS if ok else CheckStatus.FAIL,
                    evidence=Evidence.EXECUTED,
                    summary=detail.strip() or ("ok" if ok else "failed"),
                    detail=(
                        f"compiled in {compile_ms:.0f} ms, ran in {run_ms:.0f} ms"
                        + (f"; devices: {'; '.join(devices)}" if devices and key == "enumerate" else "")
                    ),
                    remediation=None if ok else _runtime_remediation(key),
                    duration_ms=run_ms,
                )
            )
            if not ok:
                blocked_by = name
        else:
            reason = (
                f"not reached: {blocked_by} failed"
                if blocked_by
                else f"probe ended early ({res.describe_failure()})"
            )
            checks.append(
                CheckResult(
                    id=cid,
                    name=name,
                    category="runtime",
                    status=CheckStatus.SKIPPED if blocked_by else CheckStatus.FAIL,
                    evidence=Evidence.EXECUTED if not blocked_by else Evidence.NONE,
                    summary=reason,
                    remediation=None if blocked_by else _runtime_remediation(key),
                )
            )
            blocked_by = blocked_by or name
    return checks


def _runtime_remediation(step: str) -> str:
    return {
        "init": "HIP could not initialise. Check `rocminfo` works, that /dev/kfd exists, and that "
        "your user is in the `render` and `video` groups",
        "enumerate": "No HIP devices visible. Check HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES and "
        "that the amdgpu driver is loaded",
        "alloc": "Device memory allocation failed; check free VRAM with `amd-smi metric --mem-usage`",
        "kernel": "The kernel failed to launch or run. The binary may lack code for this GPU "
        "architecture; rebuild with --offload-arch=<your gfx target>",
    }.get(step, "See the probe output in the JSON report")


TORCH_RUNTIME_PROBE = r"""
import json, sys, time
out = {"steps": {}}
def step(name, ok, detail=""):
    out["steps"][name] = {"ok": bool(ok), "detail": str(detail)}
try:
    import torch
    avail = torch.cuda.is_available()
    step("available", avail, f"device_count={torch.cuda.device_count()}")
    if not avail:
        print(json.dumps(out)); sys.exit(0)
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    out["devices"] = names
    torch.manual_seed(0)
    a = torch.randn(512, 512)
    b = torch.randn(512, 512)
    t0 = time.perf_counter()
    ga, gb = a.to("cuda"), b.to("cuda")
    gc = ga @ gb
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    ref = a @ b
    diff = (gc.cpu() - ref).abs().max().item()
    step("tensor_ops", diff < 1e-2, f"512x512 matmul on {names[0]}, max |diff| vs CPU = {diff:.2e}, {dt:.1f} ms")
except Exception as exc:
    step("error", False, f"{type(exc).__name__}: {exc}")
print(json.dumps(out))
"""


def run_torch_probe(
    ctx: HostContext, torch_info: TorchInfo | None, python: str | None = None
) -> list[CheckResult]:
    def mk(
        cid: str,
        name: str,
        status: CheckStatus,
        summary: str,
        ev: Evidence = Evidence.NONE,
        remediation: str | None = None,
    ) -> CheckResult:
        return CheckResult(
            id=cid,
            name=name,
            category="runtime",
            status=status,
            evidence=ev,
            summary=summary,
            remediation=remediation,
        )

    avail_id, avail_name = "runtime.torch_gpu", "PyTorch GPU Visibility"
    ops_id, ops_name = "runtime.torch_tensor_ops", "PyTorch GPU Tensor Operations"
    if torch_info is None or not torch_info.installed:
        reason = "PyTorch is not installed in the checked interpreter"
        return [
            mk(avail_id, avail_name, CheckStatus.SKIPPED, reason),
            mk(ops_id, ops_name, CheckStatus.SKIPPED, reason),
        ]
    if not torch_info.is_rocm_build:
        build = f"CUDA {torch_info.cuda_version}" if torch_info.cuda_version else "CPU-only"
        reason = f"PyTorch {torch_info.version} is a {build} build, not a ROCm build"
        fix = "Install a ROCm build of PyTorch, e.g. from https://pytorch.org/get-started/locally/"
        return [
            mk(avail_id, avail_name, CheckStatus.SKIPPED, reason, remediation=fix),
            mk(ops_id, ops_name, CheckStatus.SKIPPED, reason),
        ]
    res = ctx.run([python or sys.executable, "-c", TORCH_RUNTIME_PROBE], timeout=180)
    try:
        data = json.loads(res.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        msg = res.describe_failure()
        return [
            mk(avail_id, avail_name, CheckStatus.FAIL, msg, Evidence.EXECUTED),
            mk(ops_id, ops_name, CheckStatus.SKIPPED, "not reached"),
        ]
    steps = data.get("steps", {})
    checks = []
    a = steps.get("available")
    if a is None:
        err = steps.get("error", {}).get("detail", "unknown error")
        checks.append(mk(avail_id, avail_name, CheckStatus.FAIL, err, Evidence.EXECUTED))
    else:
        checks.append(
            mk(
                avail_id,
                avail_name,
                CheckStatus.PASS if a["ok"] else CheckStatus.FAIL,
                a["detail"],
                Evidence.EXECUTED,
                None
                if a["ok"]
                else "torch.cuda.is_available() is False; check HIP_VISIBLE_DEVICES "
                "and that the PyTorch ROCm version matches the installed ROCm",
            )
        )
    t = steps.get("tensor_ops")
    if t is not None:
        checks.append(
            mk(
                ops_id,
                ops_name,
                CheckStatus.PASS if t["ok"] else CheckStatus.FAIL,
                t["detail"],
                Evidence.EXECUTED,
            )
        )
    elif "error" in steps:
        checks.append(mk(ops_id, ops_name, CheckStatus.FAIL, steps["error"]["detail"], Evidence.EXECUTED))
    else:
        checks.append(mk(ops_id, ops_name, CheckStatus.SKIPPED, "not reached: GPU not available"))
    return checks
