# Command reference

[Docs home](README.md) · [Migration](migration.md) · [Use cases](use-cases.md)

Every command prints help with `--help`, for example `kernelvet migrate convert --help`.

| Command | What it does |
| --- | --- |
| [`kernelvet doctor`](#doctor) | short health summary with the most important fixes |
| [`kernelvet diagnose`](#diagnose) | every check: hardware, software, a HIP program run on the GPU, PyTorch, compatibility |
| [`kernelvet compatibility`](#compatibility) | compare this host with AMD's published ROCm support data |
| [`kernelvet gpu list`](#gpu-list) | list detected GPUs |
| [`kernelvet system info`](#system-info) | OS, kernel, ROCm, HIP, driver and tool versions |
| [`kernelvet benchmark`](#benchmark) | measure HIP and PyTorch performance on this host |
| [`kernelvet migrate …`](#migrate) | analyze, plan, convert, audit, build, test and report a CUDA→HIP migration |
| [`kernelvet export`](#export) | re-render a saved JSON report in another format |
| [`kernelvet schema`](#schema) | print the JSON Schema of a report type |
| [`kernelvet history`](#history) | recent runs and migration projects |
| [`kernelvet config …`](#config) | show and change settings |
| [`kernelvet tui`](#tui) | the interactive terminal UI |

`kernelvet --version` (`-V`) prints the version. `--install-completion` and
`--show-completion` set up shell completion.

## Shared options and behaviour

* **`--format`, `-f`**: `terminal` (default), `json`, `markdown`, `html`, or
  `sarif` (migration findings only).
* **`--output`, `-o FILE`**: write the report to a file. The format is inferred
  from the extension (`.json`, `.md`, `.html`, `.sarif`) unless `-f` is given.
* With a machine-readable format, stdout carries **only** the report. Progress
  and messages go to stderr, so `kernelvet diagnose -f json | jq …` is safe.
* **`--python PATH`**: the Python interpreter to inspect for PyTorch (default:
  the `python_executable` setting, else the interpreter KernelVet runs in).

### Exit codes

| Code | Meaning |
| --- | --- |
| 0 | healthy / the stage passed |
| 1 | a failure was found |
| 2 | usage error (bad option, unreadable config or matrix) |
| 3 | review required: warnings, skipped checks or unverified stages |
| 4 | a required tool is unavailable (no HIPIFY, no `hipcc`, nothing to test) |
| 70 | internal error |

---

## doctor

```bash
kernelvet doctor [--python PATH] [--runtime/--no-runtime]
```

Runs the full diagnosis and prints a short summary: host, GPUs, PASS / WARNING /
FAIL / SKIPPED counts, up to 12 checks that need attention with their fixes, how
many checks were verified on AMD hardware, and the overall status.
`--no-runtime` skips the HIP program run on the GPU.

**Exit:** 0 healthy · 3 review required · 1 unhealthy.

## diagnose

```bash
kernelvet diagnose [-f FORMAT] [-o FILE] [-v] [--runtime/--no-runtime]
                   [--torch/--no-torch] [--python PATH] [--matrix FILE]
                   [--history/--no-history]
```

| Option | Default | Effect |
| --- | --- | --- |
| `--runtime/--no-runtime` | on | compile and run the HIP probe program on each AMD GPU |
| `--torch/--no-torch` | on | detect PyTorch and run tensor operations on the GPU |
| `--matrix FILE` | bundled | use a custom compatibility matrix |
| `-v`, `--verbose` | off | show each check's details |
| `--history/--no-history` | on | record the run in local history |

What each check does is in [Diagnostics and compatibility](diagnostics.md).

**Exit:** 0 healthy · 3 review required · 1 unhealthy.

## compatibility

```bash
kernelvet compatibility [-f FORMAT] [-o FILE] [-v] [--torch/--no-torch]
                        [--python PATH] [--matrix FILE]
```

Compares the detected GPU, OS, kernel, ROCm, `amdgpu` driver, HIP and PyTorch
versions with KernelVet's dated snapshot of AMD's documentation. Nothing runs
on the GPU, so a PASS means "listed by AMD", not "tested on this hardware".
Each check cites the AMD page, section, ROCm version and retrieval date it used.

**Exit:** 0 all PASS · 3 review required · 1 a known incompatibility.

## gpu list

```bash
kernelvet gpu list [--json | -f json]
```

Lists GPUs found by `amd-smi`, `rocminfo` and `/sys`: name, gfx target, compute
units, VRAM, PCI bus and which tools found each one. This is static detection;
nothing runs on the GPU.

**Exit:** 0 if an AMD GPU was found · 1 otherwise.

## system info

```bash
kernelvet system info [--json | -f json] [--torch] [--python PATH]
```

Prints the OS, kernel, architecture, ROCm path and version, HIP runtime and
compiler, `amdgpu` driver, available tools (AMD SMI, HIPIFY, profilers) and ROCm
libraries. `--torch` also inspects PyTorch.

## benchmark

```bash
kernelvet benchmark [--hip/--no-hip] [--torch/--no-torch] [--inference]
                    [--iterations N] [--profile] [--profile-dir DIR]
                    [--python PATH] [-f FORMAT] [-o FILE]
```

| Option | Default | Effect |
| --- | --- | --- |
| `--hip/--no-hip` | on | HIP kernel, memory-bandwidth and SGEMM benchmarks |
| `--torch/--no-torch` | on | PyTorch tensor benchmarks |
| `--inference` | off | also a synthetic MLP inference benchmark |
| `--iterations N` | 20 | timed iterations per measurement |
| `--profile` | off | run the HIP benchmark under `rocprofv3` |
| `--profile-dir DIR` | `profiles/<timestamp>` in the data directory | where `rocprofv3` writes its output |

Numbers come only from real runs on this host. A benchmark that cannot run is
SKIPPED and carries no numbers. KernelVet never compares against another vendor.

**Exit:** 0 all PASS · 3 something skipped · 1 a benchmark failed.

---

## migrate

The pipeline is explained in [How CUDA-to-HIP migration works](migration.md).
`CUDA_DIR` is the original project; `HIP_DIR` is the converted copy.

### migrate analyze

```bash
kernelvet migrate analyze CUDA_DIR [--compile-commands FILE] [-f FORMAT] [-o FILE]
```

Read-only. Builds the inventory (files, build systems, NVIDIA dependencies,
tests), classifies every CUDA construct as `AUTO_CONVERTIBLE`,
`REQUIRES_VERIFICATION`, `MANUAL_INTERVENTION` or `UNSUPPORTED`, and computes the
readiness score. It also reports the project's `compile_commands.json` and
which translation units it covers; see
[Using compile_commands.json](migration.md#using-compile_commandsjson).
`-f sarif` exports the findings for code-scanning tools.

### migrate plan

```bash
kernelvet migrate plan CUDA_DIR [--translator auto|hipify-clang|hipify-perl]
                       [--offload-arch gfx942 ...] [--compile-commands FILE]
                       [-f FORMAT] [-o FILE]
```

A dry run of the whole migration: the translator and why it was chosen, the
files to convert, dependency swaps, manual changes and blockers.

**Exit:** 0 · 3 if there are blockers or no HIPIFY is installed.

### migrate convert

```bash
kernelvet migrate convert CUDA_DIR -o HIP_DIR [--dry-run] [--force]
                          [--translator auto|hipify-clang|hipify-perl]
                          [--compile-commands FILE] [-f FORMAT] [--report FILE]
```

| Option | Effect |
| --- | --- |
| `-o`, `--output HIP_DIR` | required: where the HIP project is written |
| `--dry-run` | show what would be converted; write nothing |
| `--force` | allow a non-empty output directory |
| `--translator` | override the `hipify_preference` setting |
| `--report FILE` | also write a report file (format from the extension) |
| `--compile-commands FILE` | the CUDA project's `compile_commands.json` (default: found in the project or a `build/` directory) |

Runs AMD HIPIFY file by file. The CUDA project is never modified, and an output
directory that is, contains or is inside the source is refused. The output
reminds you that converted code is not guaranteed to work.

**Exit:** 0 · 1 if HIPIFY failed on any file · 2 bad output directory · 4 HIPIFY not installed.

### migrate audit

```bash
kernelvet migrate audit HIP_DIR [-f FORMAT] [-o FILE]
```

Static scan of the converted tree: leftover CUDA, NVIDIA libraries, CUDA build
settings, risky patterns, ROCm libraries that are used but not installed.

**Exit:** 0 nothing found · 3 something needs verification · 1 manual work or unsupported code remains.

### migrate build

```bash
kernelvet migrate build HIP_DIR [--offload-arch gfx942 ...] [--isolation none|docker]
                        [-f FORMAT] [-o FILE]
```

Builds with the project's HIP CMake build if it has one, otherwise directly with
`hipcc`. `--offload-arch` defaults to the `offload_arch` setting, then to the
detected GPU targets. `--isolation docker` builds in a container with no
network.

**Exit:** 0 built · 1 build failed · 4 `hipcc` (or Docker) not available.

### migrate test

```bash
kernelvet migrate test HIP_DIR [--all] [--isolation none|docker] [-f FORMAT] [-o FILE]
```

Runs `ctest`, or the built test executables (`--all`: every executable).
Compares output with reference files when present. For each test it records
whether the test was **seen launching a HIP kernel** on an AMD GPU, as opposed
to a GPU just being present. See
[GPU detection vs GPU execution](migration.md#gpu-detection-vs-gpu-execution).

**Exit:** 0 tests passed and ran kernels on an AMD GPU · 3 passed but GPU execution not shown (no AMD GPU, or no kernel launched) · 1 failed · 4 not built yet.

### migrate report

```bash
kernelvet migrate report HIP_DIR [-f FORMAT] [-o FILE]
```

Combines every saved stage into one report with the five verification stages.

**Exit:** 0 GPU validation passed · 3 not yet validated · 1 a stage failed.

---

## export

```bash
kernelvet export REPORT.json [-f FORMAT] [-o FILE]
```

Re-renders a saved JSON report (diagnosis, compatibility, migration or
benchmark) as terminal, Markdown, HTML or SARIF output.

## schema

```bash
kernelvet schema diagnosis|compatibility|migration|benchmark
```

Prints the JSON Schema of that report type, for validating JSON output.

## history

```bash
kernelvet history [--limit N] [--json]
```

Shows recent diagnostic and benchmark runs and migration projects from the local
SQLite history (default limit 10). Turn history off with `history = false`.

## config

```bash
kernelvet config show [--json]     # effective settings, the config file path and the data directory
kernelvet config path              # the config file location
kernelvet config set KEY VALUE     # change one setting; "" resets it, none clears an optional one
```

Every setting is listed in [Configuration](configuration.md).

## tui

```bash
kernelvet tui
```

A keyboard-driven terminal UI over the same engine, with eight sections:
Overview, Diagnose, GPU Devices, Compatibility, CUDA → HIP Migration,
Benchmarks, Reports and Settings. Press `1`–`8` to switch sections, `?` or
`F1` for help on the current section, and `q` to quit. Long tasks run in the
background.
