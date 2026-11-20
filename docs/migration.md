# How CUDA-to-HIP migration works

[Docs home](README.md) · [Commands](commands.md) · [Use cases](use-cases.md)

KernelVet does **not** translate CUDA itself. The translation is always done by
AMD **HIPIFY** (`hipify-clang` or `hipify-perl`). KernelVet's job is everything
around it:

* understand the project before anything changes,
* run HIPIFY safely into a separate directory,
* then show, stage by stage, how far the result can actually be trusted.

> **Automated conversion is not guaranteed to produce working HIP code.** A
> successful `convert` only means HIPIFY processed the files. Review every
> converted file by hand, compile it, and run its tests on an AMD GPU before you
> rely on it.

## The pipeline

```
 CUDA project (read-only)                     HIP project (new directory)
 ┌──────────┐   ┌──────┐   ┌─────────┐       ┌───────┐   ┌───────┐   ┌──────┐   ┌────────┐
 │ analyze  │──▶│ plan │──▶│ convert │──────▶│ audit │──▶│ build │──▶│ test │──▶│ report │
 └──────────┘   └──────┘   └─────────┘       └───────┘   └───────┘   └──────┘   └────────┘
   classify      dry run     AMD HIPIFY        static      hipcc or    run tests,  combine
   + score                   file by file      scan        HIP CMake   GPU check   all stages
```

| Step | Command | Touches | Needs |
| --- | --- | --- | --- |
| 1 | `kernelvet migrate analyze CUDA_DIR` | reads `CUDA_DIR` | nothing |
| 2 | `kernelvet migrate plan CUDA_DIR` | reads `CUDA_DIR` | nothing (shows which HIPIFY it would use) |
| 3 | `kernelvet migrate convert CUDA_DIR -o HIP_DIR` | writes `HIP_DIR` only | `hipify-clang` or `hipify-perl` |
| 4 | `kernelvet migrate audit HIP_DIR` | reads `HIP_DIR` | nothing |
| 5 | `kernelvet migrate build HIP_DIR` | writes `HIP_DIR/.kernelvet/build` | `hipcc` (or Docker) |
| 6 | `kernelvet migrate test HIP_DIR` | runs built executables | a successful build; an AMD GPU and a test that launches a kernel for GPU validation |
| 7 | `kernelvet migrate report HIP_DIR` | reads saved state | nothing |

Each step can be run on its own, and the same steps are available in the TUI
(`kernelvet tui` → *CUDA → HIP Migration*).

## 1. Analyze: classify every CUDA construct

`migrate analyze` scans the project without changing anything.

**Inventory.** Every file is classified: CUDA sources and headers, C++ files
that use CUDA, CMake and Make builds, and tests. Build directories and VCS
folders are skipped.

**Dependencies.** NVIDIA libraries are detected and mapped to their ROCm
counterparts:

| NVIDIA | ROCm / HIP | How HIPIFY handles it |
| --- | --- | --- |
| CUDA runtime / driver API | HIP runtime | converted (`cuda*` → `hip*`) |
| cuBLAS, cuRAND, cuFFT, cuSPARSE, cuSOLVER | hipBLAS, hipRAND, hipFFT, hipSPARSE, hipSOLVER | converted |
| NCCL | RCCL (same `nccl*` API) | converted |
| Thrust, CUB | rocThrust, hipCUB | converted |
| cuDNN | MIOpen | manual: the APIs differ |
| NVTX, NVML | roctx, AMD SMI library | manual |
| WMMA | rocWMMA | manual |
| CUTLASS, TensorRT | Composable Kernel / hipBLASLt, MIGraphX | unsupported: needs a redesign |

**Findings.** Each match of a rule in
[`rules.py`](https://github.com/seyhunak/kernelvet/blob/main/src/kernelvet/migration/rules.py)
becomes a finding with a `file:line`, a severity, a message and the HIP
equivalent. Every finding is in one of four categories:

| Category | Meaning | Examples |
| --- | --- | --- |
| `AUTO_CONVERTIBLE` | HIPIFY converts it mechanically | `cudaMalloc`, `<<<grid, block>>>` launches, `cuda_runtime.h`, cuBLAS calls |
| `REQUIRES_VERIFICATION` | It converts, but HIP behaves differently and a person must check it | hard-coded warp size 32 (AMD CDNA wavefronts are 64), 32-bit `0xffffffff` lane masks, warp-sync intrinsics, managed memory, CUDA Graphs, cooperative groups, `__CUDA_ARCH__` checks, NVRTC |
| `MANUAL_INTERVENTION` | Someone has to rewrite it | cuDNN, NVTX, NVML, WMMA, libcu++, NVIDIA-only intrinsics, and CUDA settings in CMake/Make files (`LANGUAGES CUDA`, `nvcc`, `-arch=sm_80`, `-lcudart`, `CUDA_HOME`) |
| `UNSUPPORTED` | No HIP equivalent | inline PTX assembly, legacy texture references, CUTLASS, TensorRT |

### Readiness score

The score (0–100) measures **how much of the CUDA source HIPIFY can convert
mechanically**. It doesn't say whether the result will compile, be correct or
be fast.

How it is computed:

1. **Only source code is scored.** Findings in CMake, Make and `setup.py`
   files (CUDA language, `nvcc`, `sm_XX`, NVIDIA libraries) are always manual
   work. They are listed as *build-system changes*, not mixed into the score.
2. **Each construct counts once per file.** A construct is one rule matched in
   one file, so 40 `cudaMalloc` calls in a file count as one, the same as one
   inline-PTX block. Otherwise a project full of trivial calls would look ready
   while hiding the one piece of code that needs a redesign.
3. **Weights:** `AUTO_CONVERTIBLE` 1, `REQUIRES_VERIFICATION` 0.5,
   `MANUAL_INTERVENTION` and `UNSUPPORTED` 0. The result is the *coverage*.
4. **Cap:** if any `UNSUPPORTED` construct remains, the score is capped at 60,
   because such a project can't be converted mechanically however much of the
   rest can.

Each report also shows:

| Field | Meaning |
| --- | --- |
| **level** | what kind of work is left: `mechanical` (all converts), `review` (converts, but behaves differently on AMD GPUs), `manual-work` (some code must be rewritten), `blocked` (some code has no HIP equivalent), or `no-cuda` |
| **coverage** | the score before the cap |
| **occurrence coverage** | the same weights, counting every match. A large gap from *coverage* means many repeated simple calls |
| **breakdown** | per category: constructs, occurrences and files affected |
| **build-system changes** | distinct build-file findings, not scored |

A project with no CUDA in its source has no score (`n/a`, `null` in JSON),
not 0.

**What the score can't tell you.** Every report lists these, and JSON has them
as `limitations`:

* Only constructs with a rule in
  [`rules.py`](https://github.com/seyhunak/kernelvet/blob/main/src/kernelvet/migration/rules.py)
  are seen. CUDA usage without a rule is invisible, which makes coverage look
  better than it is.
* The weights are a fixed convention, not calibrated against real migrations.
* Each construct counts once per file, whatever its size: the score is about
  *kinds* of work, not amount of code.
* Code behind macros, disabled `#ifdef` branches or generated at build time is
  matched as text, not evaluated.

Use the score to compare projects and to track progress. Use the level and
the `MANUAL_INTERVENTION` / `UNSUPPORTED` findings to scope the actual work.

## 2. Plan: a dry run

`migrate plan` turns the analysis into numbered steps: which translator will be
used and why, which files it will process, which dependencies need swapping,
which manual changes and blockers exist, and the `--offload-arch` targets.
Nothing is written. It exits with code 3 when there are blockers or no
translator is installed.

**Translator choice** (`--translator` or the `hipify_preference` setting):

| Setting | Behaviour |
| --- | --- |
| `auto` (default) | `hipify-clang` if it is installed **and** CUDA headers are found (it parses CUDA semantically with clang); otherwise `hipify-perl` (regex-based, needs no CUDA toolkit) |
| `hipify-clang` | always `hipify-clang`; fails if it or the CUDA headers are missing |
| `hipify-perl` | always `hipify-perl` |

### Using compile_commands.json

A compilation database (`compile_commands.json`) records the exact command used
to compile each file. It is the only reliable source of a file's include
paths, `-D` defines and language standard. Without one, KernelVet guesses
include paths from the directory layout, which misses generated headers,
third-party include trees and defines that select code paths.

Generate one from the CUDA project, for example:

```bash
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON   # CMake
bear -- make                                             # Make, with Bear
```

KernelVet looks for it in the project root and in `build/`, `build-*`,
`cmake-build-*`, `out/` and one level below them. `--compile-commands FILE`
on `analyze`, `plan` and `convert` points at one somewhere else.

* `analyze` and `plan` show which CUDA translation units it covers. Files it
  doesn't list fall back to guessed flags. If a CMake project has no database,
  the plan suggests exporting one.
* `convert` with `hipify-clang` parses each listed file with that file's own
  flags. Each file's result records whether its flags came from
  `compile_commands.json` or were guessed. `hipify-perl` works on text and
  doesn't use compiler flags.
* The flags are saved to `HIP_DIR/.kernelvet/compile_flags.json`, with project
  paths rewritten relative to the HIP project, and the direct `hipcc` build
  reuses them.

Only flags that clang understands and that change what the code means are
used: `-I`, `-isystem`, `-iquote`, `-idirafter`, `-D`, `-U`, `-std` and
`-include`, with nvcc's long spellings (`--include-path`, `--define-macro`, …)
normalised. nvcc-only flags (`-arch`, `-gencode`, `-Xcompiler`,
`--expt-relaxed-constexpr`) and optimisation or warning flags are dropped.
For this reason the database is not passed to `hipify-clang` with `-p`: clang
would reject the nvcc flags in it.

## 3. Convert: AMD HIPIFY runs

`migrate convert CUDA_DIR -o HIP_DIR` runs HIPIFY file by file.

* **The CUDA project is never modified.** The output directory is refused if it
  is the source, is inside the source, or contains it, and a non-empty output
  directory is refused unless you pass `--force`.
* Files that are not CUDA are copied unchanged, so the HIP project is complete.
* A file HIPIFY fails on is copied **unchanged** and marked `failed`, so the
  audit flags its leftover CUDA. Nothing is dropped silently.
* HIPIFY's warnings, errors and converted/unconverted counts are recorded per
  file, and the full log is saved to `HIP_DIR/.kernelvet/conversion.log`.
* `--dry-run` shows what would happen without writing.
* Without HIPIFY installed, `convert` exits with code 4 and writes nothing.

## 4. Audit: static check of the result

`migrate audit HIP_DIR` scans the converted tree for:

* leftover CUDA runtime calls and `cuda*`/NVIDIA headers,
* NVIDIA library symbols (cuBLAS, cuDNN, …) that HIPIFY did not convert,
* CUDA settings still in CMake or Make files,
* risky patterns: warp size 32, 32-bit lane masks, `hipMemcpyToSymbol` without
  `HIP_SYMBOL()`, compute-capability checks,
* ROCm libraries that the code includes but that are not installed.
* flat ROCm library includes such as `<hipblas.h>` or `<hipcub.hpp>`, which
  HIPIFY emits but ROCm 10.x no longer ships; use `<hipblas/hipblas.h>`, which
  works on ROCm 5.x and later.

It exits 1 if anything needs manual work or is unsupported, 3 if something
needs verification, and 0 if it found nothing. **A clean audit does not mean
the code is correct**; only compiling and running tests can show that.

## 5. Build

`migrate build HIP_DIR [--offload-arch gfx942]`:

1. If the project's `CMakeLists.txt` already targets HIP (`enable_language(HIP)`
   or `LANGUAGES … HIP`) and `cmake` is available, KernelVet configures and
   builds it.
2. Otherwise it builds directly with `hipcc`, with the include paths and
   defines from `compile_commands.json` when `convert` saved them (see
   [Using compile_commands.json](#using-compile_commandsjson)). Every source with a `main` becomes
   an executable, linked with the shared sources and with the ROCm libraries
   its includes need. A HIP `CMakeLists` that reproduces this build is written
   to `.kernelvet/CMakeLists.generated.txt` for you to adopt.

A `CMakeLists.txt` that still declares the CUDA language is not used. All build
output goes to `HIP_DIR/.kernelvet/build`. With `isolation = docker`, the build
runs in a container with no network and all Linux capabilities dropped.

## 6. Test

`migrate test HIP_DIR` needs a successful build. It runs:

* `ctest` for CMake builds that call `enable_testing()`, or
* each built executable whose source is in a test directory or is named
  `test*` (`--all` runs every executable).

If `<name>.expected` or `tests/expected/<name>.out` exists, the output is
compared with it.

### GPU detection vs GPU execution

A GPU being **detected** on the machine doesn't mean a test **executed** on it.
A test binary can pass without ever touching the GPU, for example when it only
runs CPU code. So KernelVet observes execution instead of assuming it:

* Every test runs with the HIP runtime's API log switched on
  (`AMD_LOG_LEVEL=3`, `AMD_LOG_MASK=0x1`, also inside Docker). This log records
  every HIP call and its result.
* A test **ran on GPU** only when an AMD GPU is present *and* its log shows at
  least one kernel launch (`hipLaunchKernel`, `hipModuleLaunchKernel`,
  `hipGraphLaunch`, …) that returned `hipSuccess`. The number of launches is
  recorded.
* A passing test with no successful kernel launch is shown as **GPU detected,
  no kernel launch seen**, and doesn't count towards stage 5.
* Without an AMD GPU, every test is **no AMD GPU**.

In JSON, each test has `gpu_evidence` (`executed`, `detected` or `none`, the
same values as diagnostics), `gpu_kernel_launches` and `ran_on_gpu`. The HIP
log lines are removed from the stored test output.

For `ctest` runs, all tests are run as one case, so the evidence applies to the
whole `ctest` run, not to each test inside it.

**Limits.** A successful launch shows that a kernel was dispatched to the GPU.
It doesn't prove the kernel computed the right result; reference output files
and the tests' own checks do that. A program that doesn't use the HIP runtime
(for example one that calls the GPU only through OpenCL) is never shown as
running on the GPU.

## 7. Report

`migrate report HIP_DIR` combines every saved stage, and re-runs the read-only
analysis of the original project when it is still on disk. Use `-o report.html`
for a shareable page or `-f sarif` for code-scanning tools. See
[Reports and automation](reports.md).

## The five verification stages

Every migration report shows five separate stages:

| Stage | PASSED when | Otherwise |
| --- | --- | --- |
| 1. Source conversion | HIPIFY processed every file | FAILED if any file failed |
| 2. Static audit | no `MANUAL_INTERVENTION` or `UNSUPPORTED` findings remain | FAILED |
| 3. Compilation | the build succeeded | FAILED, or UNAVAILABLE without `hipcc` |
| 4. Tests | the tests passed | FAILED, or NOT_RUN |
| 5. AMD GPU runtime validation | the tests passed **and** at least one passing test was seen launching a HIP kernel on an AMD GPU | FAILED if tests failed on a GPU host; **NOT_RUN** if tests passed but none launched a kernel; **UNAVAILABLE** on a host without an AMD GPU |

Passing an earlier stage is never shown as passing a later one, and stage 5 is
never PASSED on a machine without an AMD GPU.

## Where state is kept

Each stage saves its result inside the HIP project, never in the CUDA project:

```
HIP_DIR/.kernelvet/
├── source.json       which CUDA project this came from   (convert)
├── conversion.json   per-file HIPIFY results              (convert)
├── conversion.log    full HIPIFY output                   (convert)
├── audit.json        audit findings                       (audit)
├── build.json        build command, status, errors        (build)
├── build/            build output                         (build)
├── CMakeLists.generated.txt                                (build, direct hipcc builds)
└── tests.json        test results                         (test)
```

That is how later stages and `migrate report` pick up earlier ones, in the CLI
and the TUI alike.

## What you still have to do

1. Fix every `MANUAL_INTERVENTION` and `UNSUPPORTED` finding.
2. Check every `REQUIRES_VERIFICATION` finding. Wavefront size, lane masks and
   managed memory are the usual causes of code that compiles but gives wrong
   results.
3. Port the build: swap the CUDA language, `nvcc`, `sm_XX` flags and NVIDIA
   libraries for HIP, `hipcc`, `--offload-arch=gfxXXX` and ROCm libraries. The
   generated `CMakeLists` is a starting point.
4. Compile and run the tests on an AMD GPU until stage 5 is PASSED.
5. Compare performance yourself. KernelVet does not compare against NVIDIA.

## On a machine without ROCm

| Command | Result |
| --- | --- |
| `analyze`, `plan`, `audit`, `report` | work fully |
| `convert` | exits 4: HIPIFY is not installed; nothing is written |
| `build` | exits 4: `hipcc` is not installed (unless `isolation = docker`) |
| `test` | exits 4 (NOT_RUN) without a build; with a build but no AMD GPU, exits 3 and stage 5 is UNAVAILABLE |

A walk-through with the sample project in `tests/fixtures/cuda/vector_project`
is in [examples/README.md](https://github.com/seyhunak/kernelvet/blob/main/examples/README.md).
