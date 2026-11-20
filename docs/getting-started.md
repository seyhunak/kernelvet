# Getting started

[Docs home](README.md) · [Use cases](use-cases.md) · [Commands](commands.md)

## 1. Install

You need Python 3.11 or newer.

```bash
pipx install kernelvet          # or: uv tool install kernelvet
kernelvet --version
```

`pip install kernelvet` also works inside a virtual environment. KernelVet runs
anywhere, including macOS and CPU-only Linux; there, the GPU checks report
SKIPPED or UNAVAILABLE.

KernelVet calls AMD's tools but does not bundle them. Install the ones you need
from [ROCm](https://rocm.docs.amd.com/):

| Tool | Needed for |
| --- | --- |
| `rocminfo`, `amd-smi` | GPU detection |
| `hipcc` | the runtime probe, migration builds and HIP benchmarks |
| `hipify-clang` or `hipify-perl` | `migrate convert` |
| `rocprofv3` | `benchmark --profile` |
| a ROCm build of PyTorch | the PyTorch checks and benchmarks |

You never need root.

## 2. Check the machine

```bash
kernelvet doctor
```

`doctor` prints the host, the GPUs it found, a count of PASS / WARNING / FAIL /
SKIPPED checks, the checks that need attention with how to fix each one, and an
overall status: **HEALTHY**, **REVIEW REQUIRED** or **UNHEALTHY**.

For every check with its details:

```bash
kernelvet diagnose -v
```

The last line says how many checks were **verified on AMD hardware**. Only
checks that ran code on the GPU and passed count. A check that compares versions
with AMD's documentation is "inferred from AMD docs", not verified. See
[Diagnostics and compatibility](diagnostics.md).

## 3. Save a report

```bash
kernelvet diagnose -o report.html     # self-contained page, opens offline
kernelvet diagnose -f json > diagnosis.json
```

See [Reports and automation](reports.md) for every format.

## 4. Try a migration

The repository includes a small CUDA project. It is not part of the installed
package, so if you installed with pipx, clone the repository first. Work on a
copy:

```bash
git clone https://github.com/seyhunak/kernelvet.git && cd kernelvet
cp -R tests/fixtures/cuda/vector_project /tmp/vector_project
kernelvet migrate analyze /tmp/vector_project
kernelvet migrate plan /tmp/vector_project
```

`analyze` and `plan` work on any machine. Converting needs HIPIFY, building
needs `hipcc`, and GPU validation needs an AMD GPU.

**The sample project is meant to fail the build.** It contains code that
HIPIFY cannot translate: inline PTX assembly, a 32-bit warp mask, NVTX calls
and a cuBLAS include that becomes `<hipblas.h>`, which ROCm 10.x no longer
ships. The audit points at each of them, and the build then fails on them. That
failure is expected: it shows what KernelVet catches before you spend time on
a broken build. The full walk-through is in
[How CUDA-to-HIP migration works](migration.md).

## 5. Or use the terminal UI

```bash
kernelvet tui
```

Press `1`–`8` to switch sections, `?` for help, `q` to quit.

## Next

* [Use cases](use-cases.md): workflows for common situations.
* [Command reference](commands.md): every command and option.
* [Configuration](configuration.md): timeouts, Docker isolation, offload targets.
