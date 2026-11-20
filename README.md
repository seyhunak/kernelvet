# KernelVet

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/seyhunak/kernelvet/blob/main/LICENSE)
[![Python 3.11 | 3.12 | 3.13 | 3.14](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://github.com/seyhunak/kernelvet/blob/main/pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-2a6db2.svg)](https://mypy-lang.org/)
[![Platform: Linux + ROCm](https://img.shields.io/badge/platform-Linux%20%2B%20ROCm-555555.svg)](https://rocm.docs.amd.com/)
[![Last commit](https://img.shields.io/github/last-commit/seyhunak/kernelvet/main.svg)](https://github.com/seyhunak/kernelvet/commits/main)

**Diagnose. Migrate. Verify.**

KernelVet is an open-source, terminal-native toolkit for AMD GPU developers. It
checks whether a machine can really run ROCm workloads, and helps move CUDA
projects to HIP using AMD's own tools. A command-line interface (`kernelvet`) and
a keyboard-driven terminal UI (`kernelvet tui`) share one engine.

**Documentation:** [Getting started](https://github.com/seyhunak/kernelvet/blob/main/docs/getting-started.md) ·
[Use cases](https://github.com/seyhunak/kernelvet/blob/main/docs/use-cases.md) ·
[Commands](https://github.com/seyhunak/kernelvet/blob/main/docs/commands.md) ·
[CUDA→HIP migration](https://github.com/seyhunak/kernelvet/blob/main/docs/migration.md) ·
[Diagnostics](https://github.com/seyhunak/kernelvet/blob/main/docs/diagnostics.md) ·
[Reports](https://github.com/seyhunak/kernelvet/blob/main/docs/reports.md) ·
[Configuration](https://github.com/seyhunak/kernelvet/blob/main/docs/configuration.md)

> KernelVet is an independent open-source community project. It is **not
> affiliated with, sponsored by or endorsed by AMD**, is not an AMD product, and is
> not a certification authority. Always confirm support questions against the
> live [AMD ROCm documentation](https://rocm.docs.amd.com/). See
> [Disclaimer and trademarks](#disclaimer-and-trademarks).

## What it does

| Area | What you get |
| --- | --- |
| **Diagnose** | Detects the OS, kernel, GPUs (`amd-smi`, `rocminfo`, sysfs), ROCm, the HIP compiler, the `amdgpu` driver, ROCm libraries and PyTorch. Then it **compiles and runs** a small HIP program on each GPU and runs real PyTorch tensor operations. |
| **Compatibility** | Compares the host with a dated snapshot of AMD's published compatibility data. Each check cites the exact AMD page, section and row its data came from. These results are **inferred from documentation**; nothing runs on the GPU. |
| **Migrate** | Scans a CUDA project, classifies every CUDA construct, scores readiness, plans the migration, converts it with **AMD HIPIFY** into a separate directory, audits the result, builds it with `hipcc` and runs its tests. Converted code is **not guaranteed to work**; see [HIPIFY conversion is not a guarantee](#hipify-conversion-is-not-a-guarantee). |
| **Benchmark** | Times a HIP bandwidth kernel and PyTorch matmuls on the GPU. `--profile` runs the HIP benchmark under `rocprofv3`; its output is saved to `--profile-dir`, or by default to `profiles/<timestamp>` in the data directory. Results come only from real runs. |
| **Report** | Writes terminal, JSON, Markdown, self-contained HTML and SARIF (for findings). |
| **TUI** | Eight sections: Overview, Diagnose, GPU Devices, Compatibility, CUDA→HIP Migration, Benchmarks, Reports, Settings. Long tasks run in background workers. |

## The honesty rules

KernelVet is built to never claim more than it observed.

* Every check is **PASS**, **WARNING**, **FAIL** or **SKIPPED**.
* Every check also records *how* it knows its result. See
  [Hardware-verified vs inferred results](#hardware-verified-vs-inferred-results).
* If a tool is missing or there is no GPU, the check is **SKIPPED** and says why.
  It is never reported as PASS.
* The overall status is:
  * **UNHEALTHY** if any check fails.
  * **REVIEW REQUIRED** if anything is a warning or was skipped.
  * **HEALTHY** only if every check passed.
* A CUDA migration goes through five separate verification stages:
  1. static conversion
  2. static audit
  3. compilation
  4. tests
  5. GPU validation

  Passing one stage is never shown as passing a later one.
* GPU validation passes only when the tests passed **and** at least one passing
  test was seen launching a HIP kernel on an AMD GPU. KernelVet reads that from
  the HIP runtime's own log; a GPU merely being present doesn't count. On a
  CPU-only machine it is reported as **UNAVAILABLE**.
* Benchmarks are never estimated or simulated.

## Hardware-verified vs inferred results

Only some checks run on AMD hardware. Every check carries an `evidence` value,
and the CLI, TUI and every report show it in plain words:

| Shown as | JSON `evidence` | What it means | Verified on AMD hardware? |
| --- | --- | --- | --- |
| **ran on GPU** | `executed` | KernelVet compiled and ran code on the GPU during this run (the HIP probe, PyTorch tensor ops). | **Yes**, but only when the check is PASS. |
| **detected, not run** | `detected` | Read from `amd-smi`, `rocminfo`, `/sys`, files or tool versions. Nothing was executed. | No |
| **inferred from AMD docs** | `configured` | Compared with AMD's published compatibility data (every `compat.*` check) or configuration. | No. A PASS means "listed by AMD", not "tested here". |
| **none** | `none` | The check was not performed (always the case for SKIPPED). | No |

* `kernelvet diagnose` and `kernelvet doctor` end with a line saying how many
  checks were verified on AMD hardware. When none were, they say so.
* JSON reports add `hardware_verified` and `evidence_label` to every check,
  `hardware_verified_checks` (the IDs of checks that ran on the GPU and
  passed) to a diagnosis, and an `evidence_legend`. Compatibility reports also
  carry a `basis` statement.
* SARIF results are static source analysis. Each result carries
  `"evidence": "static-analysis"` and `"hardwareVerified": false`.
* KernelVet itself has not yet been tested on real AMD GPU hardware. The
  hardware paths have been exercised only against constructed fixtures (see
  [What has and has not been verified](#what-has-and-has-not-been-verified)).

## Install

You need Python 3.11 or newer. The tool itself runs anywhere, including
CPU-only machines and macOS. On those machines the GPU checks simply report
SKIPPED or UNAVAILABLE.

```bash
pipx install kernelvet
```

[pipx](https://pipx.pypa.io/) installs the `kernelvet` command in its own
environment. `uv tool install kernelvet` does the same. Plain `pip install
kernelvet` works inside a virtual environment; outside one, Homebrew Python and
recent Debian/Ubuntu refuse it with an "externally-managed-environment" error.

The latest `main`, before it is released:

```bash
pipx install "git+https://github.com/seyhunak/kernelvet.git"
```

From a checkout:

```bash
pip install -e ".[dev]"
```

KernelVet calls AMD's tools; it does not bundle them. Install whichever you
need from ROCm:

* `rocminfo` and `amd-smi`: detection
* `hipcc`: the runtime probe, builds and benchmarks
* `hipify-clang` or `hipify-perl`: conversion
* `rocprofv3`: profiling

You never need root to run KernelVet.

## Quick start

```bash
kernelvet doctor                       # short health summary with the top fixes
kernelvet diagnose -v                  # every check, with details
kernelvet diagnose --format json > diagnosis.json
kernelvet diagnose -o report.html      # format inferred from the extension
kernelvet compatibility                # static check against AMD's published data
kernelvet gpu list
kernelvet system info --torch
kernelvet tui
```

### Migrating a CUDA project

```bash
kernelvet migrate analyze ./my-cuda-app           # read-only classification + readiness score
kernelvet migrate plan    ./my-cuda-app           # dry run: what will happen, file by file
kernelvet migrate convert ./my-cuda-app -o ./my-cuda-app-hip
kernelvet migrate audit   ./my-cuda-app-hip       # leftover CUDA, risky patterns, file:line
kernelvet migrate build   ./my-cuda-app-hip
kernelvet migrate test    ./my-cuda-app-hip
kernelvet migrate report  ./my-cuda-app-hip --format sarif -o findings.sarif
```

The analyzer sorts every finding into one of four categories:

* `AUTO_CONVERTIBLE`
* `REQUIRES_VERIFICATION`
* `MANUAL_INTERVENTION`
* `UNSUPPORTED`

`convert` never modifies the CUDA project. It refuses an output directory that:

* is the source directory,
* is inside the source, or
* contains the source.

It also refuses a non-empty output directory unless you pass `--force`. Each
stage saves its state in `<hip-project>/.kernelvet/`, so later stages and
`report` can pick it up. This includes the full HIPIFY log in `conversion.log`.

### HIPIFY conversion is not a guarantee

KernelVet does not translate CUDA itself; AMD HIPIFY does the conversion.
**Automated conversion is not guaranteed to produce working HIP code.** A
successful `convert` only means HIPIFY processed the files. Before you rely on
the result:

1. **Review** every converted file by hand. `migrate audit` points at leftover
   CUDA and risky patterns, but it cannot prove the code is correct.
2. **Compile** it with `migrate build`.
3. **Run the tests on an AMD GPU** with `migrate test`. Only this stage
   validates runtime behaviour.

`migrate convert` and every migration report (terminal, JSON, Markdown, HTML
and SARIF) repeat this warning.

`build` and `test` can run inside Docker:

```bash
kernelvet config set isolation docker
```

The container has no network and drops every Linux capability.
`/dev/kfd` and `/dev/dri` are passed through only for GPU runs.

## Output and exit codes

* With `--format json`, stdout contains **only** the JSON document. Progress
  goes to stderr, so piping into `jq` is safe.
* `kernelvet schema diagnosis|compatibility|migration|benchmark` prints the JSON
  Schema of each report type.

| Exit code | Meaning |
| --- | --- |
| 0 | Healthy / stage passed |
| 1 | A failure was found |
| 2 | Usage error |
| 3 | Review required (warnings, skipped checks or unverified stages) |
| 4 | A required tool is unavailable (e.g. no HIPIFY installed) |
| 70 | Internal error |

## Configuration

`kernelvet config show` prints the effective configuration. Settings are
resolved in this order:

1. the defaults,
2. the config file (`kernelvet config path`, usually
   `~/.config/kernelvet/config.toml`),
3. `KERNELVET_<FIELD>` environment variables.

`kernelvet config set KEY VALUE` changes one setting, for example
`kernelvet config set isolation docker`. An empty value resets the setting to
its default (`kernelvet config set isolation ""`). `none` clears a setting that
can be left unset, such as `offload_arch`.

`KERNELVET_CONFIG_DIR` and `KERNELVET_DATA_DIR` move the config and data
directories. History (SQLite) and saved reports live in the data directory.
Turn history off with `history = false`.

See [`examples/config.toml`](https://github.com/seyhunak/kernelvet/blob/main/examples/config.toml) for every option.

## Compatibility data

The bundled matrix is
[`src/kernelvet/compatibility/data/rocm_matrix.json`](https://github.com/seyhunak/kernelvet/blob/main/src/kernelvet/compatibility/data/rocm_matrix.json).

* It was transcribed by hand from AMD's system-requirements page,
  compatibility-matrix page and release-history page.
* It records the ROCm version (10.1.0), the date it was retrieved, and every
  source URL.
* Every rule carries a `ref` that points into a `references` table. Each
  reference gives the exact AMD page, section and row or footnote, its URL with
  the section anchor, the ROCm version that page describes, the transcription
  date, and the date it was last re-checked (`verification`). A reference that
  could not be re-checked is marked `unverified` and says why. Checks cite these
  references in their `source` field.
* AMD's pages sometimes disagree with each other. Those conflicts are written
  into the file's `notes` instead of being resolved silently.
* A value earns PASS only if AMD lists it explicitly. Unlisted values are
  WARNING, and known incompatibilities are FAIL.
* OS support that AMD limits to certain GPUs (for example Debian 13, Rocky
  Linux 9, Oracle Linux, RHEL 8.10, SLES 16.0, and RHEL 10.1/9.7 without the
  Radeon PRO V620) is checked per product, because GPUs with the same LLVM
  target can differ. When KernelVet cannot tell which product a GPU is and the
  answer depends on it, the result is WARNING, never PASS.

To use your own matrix, pass `--matrix FILE` or set `compatibility_matrix`.

## What has and has not been verified

The test suite runs on CPU-only machines:

* It replays recorded command output through a fake command runner.
* It builds fake filesystem roots (`/sys`, `/opt/rocm`, `/etc/os-release`) for
  each test.
* The host fixtures in `tests/fixtures/host/` were **constructed** to match the
  documented output formats. They were not recorded on real hardware; see
  [`tests/fixtures/host/README.md`](https://github.com/seyhunak/kernelvet/blob/main/tests/fixtures/host/README.md).

The tests in `tests/integration/test_real_tools.py` are **skipped** unless the
real tools are present. They are marked `hipify` (HIPIFY installed) and `gpu`
(AMD GPU present). Run them on a ROCm machine with:

```bash
pytest -m "gpu or hipify"
```

Bug reports with real `rocminfo` / `amd-smi` output are very welcome.

## Documentation

Full documentation is in [`docs/`](https://github.com/seyhunak/kernelvet/blob/main/docs/README.md):

* [Getting started](https://github.com/seyhunak/kernelvet/blob/main/docs/getting-started.md): install and first checks
* [Use cases](https://github.com/seyhunak/kernelvet/blob/main/docs/use-cases.md): recipes for common situations
* [Command reference](https://github.com/seyhunak/kernelvet/blob/main/docs/commands.md): every command, option and exit code
* [How CUDA-to-HIP migration works](https://github.com/seyhunak/kernelvet/blob/main/docs/migration.md): the pipeline, finding categories and verification stages
* [Diagnostics and compatibility](https://github.com/seyhunak/kernelvet/blob/main/docs/diagnostics.md): what each check does and how much it proves
* [Reports and automation](https://github.com/seyhunak/kernelvet/blob/main/docs/reports.md): output formats, JSON, SARIF and scripting
* [Configuration](https://github.com/seyhunak/kernelvet/blob/main/docs/configuration.md): settings, environment variables and Docker isolation
* [Architecture](https://github.com/seyhunak/kernelvet/blob/main/docs/architecture.md)
* [Contributing](https://github.com/seyhunak/kernelvet/blob/main/CONTRIBUTING.md)
* [Security policy](https://github.com/seyhunak/kernelvet/blob/main/SECURITY.md)
* [Code of Conduct](https://github.com/seyhunak/kernelvet/blob/main/CODE_OF_CONDUCT.md)
* [Notes for AI coding agents](https://github.com/seyhunak/kernelvet/blob/main/AGENTS.md)

## License

Apache License 2.0. See [LICENSE](https://github.com/seyhunak/kernelvet/blob/main/LICENSE)
(the unmodified license text) and [NOTICE](https://github.com/seyhunak/kernelvet/blob/main/NOTICE)
(copyright and trademark notices).

## Disclaimer and trademarks

KernelVet is an independent, community-maintained open-source project. It is
not affiliated with, sponsored by, or endorsed by Advanced Micro Devices, Inc.
(AMD), and it is not an AMD product. Its results are not an official AMD
statement of support or certification. For authoritative information, use the
[AMD ROCm documentation](https://rocm.docs.amd.com/).

### Trademarks

* AMD, AMD ROCm, AMD Instinct, AMD CDNA, AMD RDNA, Radeon and Ryzen are
  trademarks of Advanced Micro Devices, Inc.
* NVIDIA and CUDA are trademarks or registered trademarks of NVIDIA
  Corporation in the U.S. and other countries.
* Other names are used for identification purposes only and may be trademarks
  of their respective owners.

KernelVet uses these names only to say which hardware and software it works
with (nominative use). Their use does not imply any affiliation with, or
endorsement by, AMD or NVIDIA. KernelVet does not use AMD or NVIDIA logos. The
same notice is in [NOTICE](https://github.com/seyhunak/kernelvet/blob/main/NOTICE),
in `kernelvet --help` and at the end of every Markdown and HTML report.
