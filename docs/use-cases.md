# Use cases

[Docs home](README.md) · [Getting started](getting-started.md) · [Commands](commands.md)

Each section is a short recipe. The command details are in the
[Command reference](commands.md).

## "Is my new AMD GPU machine set up correctly?"

```bash
kernelvet doctor
```

Fix the listed problems from the top, then run it again until it reports
HEALTHY. `doctor` compiles and runs a small HIP program on every AMD GPU, so a
HEALTHY result means code really ran, not only that packages are installed.

## "PyTorch can't see my GPU"

```bash
kernelvet diagnose -v --python /path/to/venv/bin/python
```

Check, in order:

1. `hardware.device_access`: can your user open `/dev/kfd` and `/dev/dri`?
   (Usually the `render` and `video` groups.)
2. `runtime.*`: does plain HIP work? If the HIP probe fails, PyTorch will too.
3. `runtime.torch_gpu`: is the installed PyTorch a **ROCm** build? A CUDA or
   CPU-only wheel is reported as such, with a link to the ROCm build.
4. `compat.pytorch`: is that PyTorch version listed for your ROCm?

## "Is my GPU / OS / kernel combination supported?"

```bash
kernelvet compatibility -v
```

Every check cites the AMD page, section and ROCm version it used. Some
distributions are supported only on certain GPUs; KernelVet checks those per
product. Results are **inferred from AMD's documentation**, not tested, so
confirm anything important on the live
[AMD ROCm documentation](https://rocm.docs.amd.com/). See
[Diagnostics and compatibility](diagnostics.md#compatibility).

## "I want to see how hard porting our CUDA code would be"

No ROCm needed:

```bash
kernelvet migrate analyze ./our-cuda-project -o assessment.html
kernelvet migrate plan ./our-cuda-project
```

The report lists every CUDA construct, grouped as auto-convertible, needs
verification, manual work, or unsupported, plus the NVIDIA libraries in use and
their ROCm counterparts. The readiness score says how much HIPIFY can convert
mechanically. It does not predict whether the result will be correct.
`UNSUPPORTED` findings (inline PTX, CUTLASS, TensorRT, texture references) are
the work to scope first.

## "Port a CUDA project to HIP"

```bash
kernelvet migrate convert ./proj -o ./proj-hip
kernelvet migrate audit   ./proj-hip
# fix what the audit reports, then:
kernelvet migrate build   ./proj-hip --offload-arch gfx942
kernelvet migrate test    ./proj-hip
kernelvet migrate report  ./proj-hip -o migration.html
```

Repeat audit → build → test as you fix things. The job is done when stage 5,
*AMD GPU runtime validation*, is PASSED. See
[How CUDA-to-HIP migration works](migration.md).

## "Build untrusted or third-party code safely"

```bash
kernelvet config set isolation docker
kernelvet migrate build ./proj-hip
kernelvet migrate test  ./proj-hip
```

Builds and tests run in a container with no network and all Linux capabilities
dropped. The GPU devices are passed through only when a run needs them. See
[Configuration](configuration.md#docker-isolation).

## "Show CUDA leftovers in code review"

```bash
kernelvet migrate audit ./proj-hip -f sarif -o kernelvet.sarif
```

Upload the SARIF file to any code-scanning tool (GitHub code scanning, an IDE
SARIF viewer). Every result is marked as static analysis, not hardware
verified. See [Reports and automation](reports.md#sarif).

## "Fail a pipeline when a GPU host is broken"

```bash
kernelvet diagnose -f json -o diagnosis.json
```

The exit code is 0 for HEALTHY, 3 for REVIEW REQUIRED and 1 for UNHEALTHY, so
a script can decide how strict to be. See
[Reports and automation](reports.md#scripting).

## "Measure what this machine actually delivers"

```bash
kernelvet benchmark --inference -o bench.html
kernelvet benchmark --profile          # also capture a rocprofv3 trace
```

Numbers come only from runs on this host. A benchmark that can't run is
SKIPPED with no number. KernelVet never compares against other vendors.

## "Keep an eye on a machine over time"

```bash
kernelvet history
```

Diagnostic runs, benchmarks and migration projects are recorded locally in
SQLite. Turn it off with `kernelvet config set history false`.

## "Share a result with someone"

```bash
kernelvet diagnose -f json -o diagnosis.json   # keep the JSON
kernelvet export diagnosis.json -o diagnosis.html
```

The HTML page is self-contained (inline CSS, no scripts) and opens offline.
Reports include the evidence legend and disclaimers, so the reader can tell
what ran on hardware and what was inferred.
