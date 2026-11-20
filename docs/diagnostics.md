# Diagnostics and compatibility

[Docs home](README.md) · [Commands](commands.md) · [Reports](reports.md)

`kernelvet diagnose` answers one question: **can this machine really run ROCm
workloads?** It looks at the system, then runs code on the GPU to prove it.
`kernelvet doctor` runs the same checks and prints only a summary with the top
fixes.

## Statuses

Every check ends as exactly one of:

| Status | Meaning |
| --- | --- |
| **PASS** | the condition was verified |
| **WARNING** | verification was incomplete, or there is a potential issue |
| **FAIL** | a known failure or incompatibility |
| **SKIPPED** | the check could not be performed, for example because a tool or GPU is missing; the reason is shown |

SKIPPED is never counted as success. The overall status is:

* **UNHEALTHY** if any check failed,
* **REVIEW REQUIRED** if any check is a warning or was skipped,
* **HEALTHY** only if every check passed.

## Hardware-verified vs inferred results

Each check also records how it knows its result:

| Shown as | JSON `evidence` | Meaning | Verified on AMD hardware? |
| --- | --- | --- | --- |
| **ran on GPU** | `executed` | KernelVet compiled and ran code on the GPU in this run | **Yes**, when the check is PASS |
| **detected, not run** | `detected` | read from tools, files or `/sys`; nothing was executed | No |
| **inferred from AMD docs** | `configured` | compared with AMD's published compatibility data | No: a PASS means "listed by AMD" |
| **none** | `none` | not performed (every SKIPPED check) | No |

`diagnose` and `doctor` end with a line saying how many checks were verified
on AMD hardware, or that none were.

## What is checked

| Category | Examples of checks | How |
| --- | --- | --- |
| System | OS, kernel, architecture (`system.platform`) | `/etc/os-release`, `uname` |
| Hardware | GPUs found, `/dev/kfd` and `/dev/dri` access (`hardware.gpu_detected`, `hardware.device_access`) | `amd-smi`, `rocminfo`, `/sys/class/drm` |
| Software | ROCm install, `rocminfo`, `amdgpu` driver (`software.rocm`, `software.rocminfo`, `software.amdgpu`) | ROCm location (below), tool versions, kernel module |
| Runtime | the HIP probe and PyTorch on the GPU (below) | code runs on the device |
| Compatibility | GPU, OS, kernel, ROCm, driver, HIP, PyTorch (`compat.*`) | the bundled snapshot of AMD's documentation |

### Where ROCm is looked for

`software.rocm` uses the first of these that exists:

1. `ROCM_PATH`, then `HIP_PATH`, when they point at an existing directory.
2. `rocm-sdk` on PATH: ROCm installed as AMD's TheRock Python packages, for
   example `pip install "rocm[libraries,devel]"` in a virtual environment.
   Activate that environment first. KernelVet runs `rocm-sdk version` for the
   version and lists the `_rocm_sdk_*` package folders to find the libraries.
   It does not run `rocm-sdk path` or `rocm-sdk init`, because those set up
   the devel links on first use, and diagnostics never write to your system.
3. `/opt/rocm`. ROCm 10.x keeps its version in `/opt/rocm/core-X.Y/.info/version`.

A nightly (pre-release) build, such as `10.2.0a20261118` from `rocm-sdk version`,
gets WARNING in `compat.rocm`, because AMD's compatibility matrix covers
released versions only.

### The HIP probe (runtime checks)

With `--runtime` (the default), KernelVet compiles a small HIP program with
`hipcc` for the detected GPU targets and runs it. Each step is its own check:

1. HIP initialization (`runtime.hip_init`)
2. device enumeration (`runtime.device_enum`)
3. GPU memory allocation (`runtime.memory_alloc`)
4. host-to-device transfer (`runtime.memcpy_h2d`)
5. kernel execution (`runtime.kernel_exec`)
6. device-to-host transfer (`runtime.memcpy_d2h`)
7. vector-add result verification (`runtime.vector_add_verify`)

When a step fails, every later step is SKIPPED with "not reached", so the first
failure is the place to look. If `hipcc` is missing or the probe doesn't
compile, all seven are SKIPPED with the reason.

### PyTorch checks

With `--torch` (the default), KernelVet inspects the chosen Python. If it has a
**ROCm** build of PyTorch, it checks that the GPU is visible
(`runtime.torch_gpu`) and runs real tensor operations on it
(`runtime.torch_tensor_ops`). A CUDA or CPU-only build is reported, with a link
to install the ROCm build.

## Compatibility

`kernelvet compatibility`, and the compatibility section of `diagnose`, compare
detected versions with KernelVet's dated snapshot of AMD's documentation. Nothing
runs on the GPU.

| Check | PASS when | Otherwise |
| --- | --- | --- |
| `compat.gpu` | the gfx target is listed as supported | FAIL if listed as unsupported; WARNING if unlisted or ambiguous |
| `compat.os` | the distribution and version are listed for your GPU class, and any per-GPU limit allows your GPU | FAIL on macOS; WARNING if unlisted or restricted |
| `compat.kernel` | the kernel's major.minor is listed for that OS | WARNING |
| `compat.rocm` | the installed ROCm matches the snapshot's release | WARNING if older, newer or unknown |
| `compat.driver` | the `amdgpu` version is listed | FAIL if the module isn't loaded on a GPU host; WARNING if unlisted |
| `compat.hip` | the HIP version matches the ROCm install (ROCm 10.x ships HIP 7.x: 10.2.0 ships HIP 7.17.0, looked up in the snapshot) | WARNING on mixed installs, or for a 10.x release the snapshot doesn't list |
| `compat.pytorch` | a ROCm PyTorch build whose version is listed | WARNING |

**Per-GPU OS limits.** AMD supports some distributions only on certain GPUs.
Examples: Debian 13 (MI355X, MI350X, MI325X, MI300X), Rocky Linux 9 (MI300X,
MI300A), Oracle Linux 8 (MI300X), RHEL 10.1 and 9.7 (not Radeon PRO V620),
Ubuntu 22.04 (not Instinct MI350P). These
are checked per product, because GPUs with the same target can differ. When the
GPU's product can't be identified and the answer depends on it, the result is
WARNING.

### Compatibility data

The snapshot is
[`rocm_matrix.json`](https://github.com/seyhunak/kernelvet/blob/main/src/kernelvet/compatibility/data/rocm_matrix.json),
transcribed by hand from AMD's system-requirements, compatibility-matrix and
release-history pages.

* It records the ROCm version it describes, its retrieval date and every source
  URL.
* Every rule cites a reference: the exact AMD page, section and row or footnote,
  the ROCm version that page describes, and when it was transcribed and last
  re-checked. Each check's `source` shows these references.
* Where AMD's pages disagree, both are recorded and the disagreement is
  written into the file's `notes`. The compatibility matrix is the current
  source. The system-requirements page still describes ROCm 7.14, so what only
  it lists (RHEL 10.1 and 9.7, kernel 6.17 on Ubuntu 24.04, MI210 on Debian 12,
  ...) counts as PASS only when the installed ROCm is 7.14. On any other host,
  or when ROCm is not detected, it is a WARNING: "listed only on AMD's
  system-requirements page for ROCm 7.14.x".
* A GPU that AMD doesn't list is not officially supported, and gets WARNING.

Always confirm important decisions against the live
[AMD ROCm documentation](https://rocm.docs.amd.com/). KernelVet is not AMD and
is not a certification authority.

## On a machine without ROCm

On macOS or a CPU-only Linux box the tool still runs. Hardware and runtime
checks are SKIPPED, `compat.os` FAILs on macOS, and the overall status is
UNHEALTHY or REVIEW REQUIRED. That is the correct result, not a bug.
