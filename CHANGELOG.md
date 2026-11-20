# Changelog

All notable changes to KernelVet are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Before 1.0, a minor version may
include breaking changes.

## [Unreleased]

## [0.3.0] - 2026-11-20

### Changed

* **Compatibility data updated to ROCm 10.2.0** (released 2026-11-18), from
  the 10.1.0 snapshot. Transcribed on 2026-11-20 from AMD's versioned 10.2.0
  compatibility matrix, with every reference re-pointed at it.
  * New: Instinct MI350P, Radeon PRO W7900D, RHEL 10.3 and 9.9, SLES 16.1,
    amdgpu 31.70.0, PyTorch 2.15 and JAX 0.11.2.
  * Removed, because the 10.2.0 matrix no longer lists them: amdgpu 30.20.1,
    30.20.0 and 30.10.2 for Instinct, 31.50.0 for Radeon, and RHEL 10.2/9.8 for
    Radeon (Radeon now lists RHEL 10.3 and 9.9 only).
  * Products that AMD's 10.1.0 selector already listed but the old snapshot
    missed are now included: MI350P, RX 9050, RX 9050 (4GB), RX 7600
    (`gfx1102`, previously WARNING as ambiguous), Radeon AI PRO R9700S and
    R9600.
  * MI350P has its own OS rows: not Ubuntu 22.04, RHEL 10.0, RHEL 9.4,
    Debian 12, Oracle Linux or Rocky Linux.
  * Ryzen APU targets stay WARNING. The matrix names each APU's target, but an
    iGPU name can map to more than one target (Radeon 840M is `gfx1152` or
    `gfx1153`), and KernelVet does not check the APU OS and driver rows.
  * The 7.14-era system-requirements page has not changed; its rules are kept,
    and either page still counts.

* **Readiness scoring is less biased and explains itself.**
  * Only source code is scored. Build-file findings are reported as
    `build_items`, because they always need manual porting.
  * Each rule counts once per file, so repeated trivial calls no longer
    outweigh the code that needs a redesign. The per-occurrence figure is kept
    as `occurrence_coverage`.
  * New fields: `level` (`mechanical` / `review` / `manual-work` / `blocked` /
    `no-cuda`), `coverage` before the cap, `cap` and `cap_reason`, a
    per-category breakdown (`components`), and `limitations`.
  * Terminal, Markdown/HTML and TUI reports show the breakdown and the limits.
  * A project with no CUDA source now has `score: null` instead of 0. JSON
    consumers should handle `null`.

### Added

* **`compile_commands.json` support.** KernelVet finds the CUDA project's
  compilation database (in the root or a `build*/`, `cmake-build-*` or `out/`
  directory), or takes one with `--compile-commands FILE` on
  `migrate analyze`, `plan` and `convert`.
  * `hipify-clang` parses each listed file with its own include paths, defines
    and `-std`, instead of guessed include paths.
  * The flags are saved to `.kernelvet/compile_flags.json`, relative to the HIP
    project, and reused by the direct `hipcc` build.
  * nvcc-only flags (`-arch`, `-gencode`, `-Xcompiler`, ...) are filtered out.
  * Reports and the plan show coverage. Each converted file records whether its
    flags came from the database or were guessed.
  * Before this change, a database in the project root was only mentioned in
    the translator reason and never read.

* User documentation in `docs/`: getting started, use cases, a command
  reference, how CUDA-to-HIP migration works, diagnostics and compatibility,
  reports and automation, and configuration. Linked from the README.

### Fixed

* **`compat.hip` and `compat.pytorch` no longer report ROCm 10.x as a mixed
  install.** ROCm 10.x numbers HIP separately (ROCm 10.2.0 ships HIP 7.17.0),
  so comparing the HIP and ROCm version numbers warned on every correct 10.x
  install. The matrix now records the HIP version each 10.x release ships
  (`hip_versions`: 10.0.0 → 7.15.0, 10.1.0 → 7.16.0, 10.2.0 → 7.17.0, from AMD's
  "ROCm Core SDK components" tables). A 10.x release the snapshot does not list
  is a WARNING saying it could not be checked, never a PASS.
* **GPU validation now requires observed GPU execution.** `migrate test` used
  to mark every passing test as "ran on GPU" whenever an AMD GPU was detected,
  so a test that never touched the GPU could pass stage 5. Tests now run with
  the HIP runtime's API log on (`AMD_LOG_LEVEL=3`, `AMD_LOG_MASK=0x1`, passed
  into Docker too). A test counts as running on the GPU only if the log shows a
  kernel launch that returned `hipSuccess`. Each test records `gpu_evidence`
  (`executed` / `detected` / `none`) and `gpu_kernel_launches`. Reports show
  "ran on GPU (N kernel launches)", "GPU detected, no kernel launch seen" or
  "no AMD GPU". Passing tests with no kernel launch leave stage 5 `NOT_RUN`,
  with a recommendation. `ctest` now runs with `-V` so the log of passing tests
  is captured.
* **ROCm 10.x version detection.** ROCm 10.x (TheRock) keeps the version file
  at `/opt/rocm/core-X.Y/.info/version` and has no `/opt/rocm/.info`, so
  `diagnose` reported "version could not be read" and `system info` showed
  "ROCm not detected". KernelVet now reads the newest `core-X.Y` directory's
  version file, falling back to the version in the directory name.
* **The audit flags flat ROCm library includes.** HIPIFY emits
  `#include <hipblas.h>`, `<rocrand.h>`, `<hipcub.hpp>` and similar. ROCm 10.x
  ships only the subdirectory form (`<hipblas/hipblas.h>`), so the build failed
  with "file not found". New audit rules `audit.flat_rocm_include` and
  `audit.flat_rocm_include_hpp` (manual intervention) name the replacement,
  which also works on ROCm 5.x and later.
* Docker isolation now starts the container with `--platform=linux/amd64`.
  AMD's ROCm images are x86-64 only, so on Arm hosts (Apple Silicon)
  `migrate build`/`migrate test` with `isolation = docker` failed to pull the
  image with "no matching manifest". On x86-64 hosts nothing changes.

## [0.2.0] - 2026-11-20

### Added

- Python 3.14 is tested in CI and listed in the package classifiers.
- This changelog.
- `NOTICE` file with the copyright line, the non-affiliation disclaimer and
  trademark notices for AMD, AMD ROCm and NVIDIA CUDA. It ships in the sdist and
  in the wheel's license files.
- Compatibility matrix provenance (matrix `schema` 2). Every rule has a `ref`
  into a new `references` table. Each reference gives the AMD page, section and
  row or footnote, an anchored URL, the ROCm version the page describes, the
  transcription date and the date it was last re-checked. On 2026-11-20 every
  rule was re-checked against AMD's pages and all references are `verified`.
  The loader rejects undefined refs, and an `unverified` reference without a
  `note`. Compatibility checks now cite the exact references behind each result
  in `source`. Schema 1 matrices still load.
- JSON reports: every check has `hardware_verified` (true only for a PASS that
  ran on the GPU) and `evidence_label`. Diagnoses add `hardware_verified_checks`
  and `evidence_legend`. Compatibility reports add `basis` and
  `evidence_legend`. Migration reports add `conversion_disclaimer`.
- The matrix now records every per-OS GPU limit from both AMD pages, by
  product. New: Debian 12 (not MI100), Oracle Linux 10 and 9 (MI355X, MI350X,
  MI325X, MI300X only), Oracle Linux 8 (MI300X only), SLES 16.0 (not MI100),
  and RHEL 10.1/9.7 (not Radeon PRO V620). Debian 13, Rocky Linux 9 and
  RHEL 8.10 are now checked per product, so MI300A on Debian 13 and MI325X on
  Rocky Linux 9 or RHEL 8.10 are no longer reported as PASS. Where the 10.1.0
  matrix and the 7.14 system requirements disagree, a GPU counts if either
  page lists it, and the disagreement is written into the rule's `note`. A
  GPU whose product cannot be identified gets WARNING when the answer depends
  on it.
- SARIF runs say their results are static analysis
  (`"evidence": "static-analysis"`, `"hardwareVerified": false`) and carry the
  HIPIFY disclaimer.

### Changed

- Evidence is shown in plain words everywhere: "ran on GPU" (`executed`),
  "detected, not run" (`detected`), "inferred from AMD docs" (`configured`) and
  "none". A legend accompanies the CLI, TUI, Markdown and HTML output. The JSON
  `evidence` values are unchanged.
- `diagnose` and `doctor` end with how many checks were verified on AMD
  hardware, replacing "GPU execution verified: yes/no". Compatibility output
  says its results are inferred from AMD's documentation, not tested on the
  hardware.
- `migrate convert`, the migration TUI and every migration report say that
  HIPIFY output is not guaranteed to work and needs manual review, compilation
  and runtime testing on an AMD GPU.
- The trademark notice now also names Ryzen and NVIDIA, and says "trademarks or
  registered trademarks of NVIDIA Corporation".
- The matrix's `matrix` source URL now points at the versioned ROCm 10.1.0 page.
  AMD's `latest` URL has moved on to 10.2.0, and is kept as `latest_url`.

- CI and release workflows use `actions/checkout@v7`, `actions/setup-python@v7`,
  `actions/upload-artifact@v7` and `actions/download-artifact@v8`.
- Dependabot version-update PRs are turned off (`.github/dependabot.yml`
  removed). Dependency updates are done by hand.

### Fixed

- SARIF export no longer uses `PurePath.as_uri()`, which is deprecated in
  Python 3.14. The generated `file://` URI is unchanged.
- `LICENSE` is now the exact Apache-2.0 text from apache.org. The previous copy
  merged the "Work" definition into "Object" and reworded the warranty
  disclaimer, so GitHub reported the license as "Other". The copyright line
  moved to `NOTICE`.
- Markdown/HTML diagnosis reports no longer title the runtime section
  "executed on the GPU" when its checks were skipped.

### Known limitations

- Still not tested on real AMD GPU hardware. CI has no GPU runner, so the
  hardware paths have been exercised only against constructed fixtures.
- The bundled compatibility data describes ROCm 10.1.0. AMD has since
  published 10.2.0; hosts running it get a WARNING saying the snapshot is older.

## [0.1.1] - 2026-11-19

### Changed

- The README recommends `pipx install kernelvet` (or `uv tool install
  kernelvet`). Plain `pip install` outside a virtual environment fails on
  Homebrew Python and recent Debian/Ubuntu. Republished so the PyPI project
  page shows this. No code changes.

## [0.1.0] - 2026-11-19

First public release.

### Added

- `kernelvet diagnose`: checks the OS, kernel, `amdgpu` driver, ROCm, the HIP
  compiler, ROCm libraries and PyTorch, then compiles and runs a small HIP
  program on each detected AMD GPU and runs real PyTorch tensor operations.
- Compatibility checks against a dated, sourced snapshot of AMD's published
  ROCm compatibility data.
- CUDA-to-HIP migration: analyze, plan, convert with AMD HIPIFY into a
  separate directory, audit, build with `hipcc` and run tests, with five
  separate verification stages.
- HIP and PyTorch benchmarks, optionally under `rocprofv3`.
- Reports in terminal, JSON, Markdown, self-contained HTML and SARIF.
- A keyboard-driven terminal UI (`kernelvet tui`) over the same engine.
- Every check is PASS, WARNING, FAIL or SKIPPED and records its evidence
  (`detected`, `executed`, `configured` or `none`). Missing tools or GPUs are
  SKIPPED, never PASS.

### Known limitations

- Not yet tested on real AMD GPU hardware: CI has no GPU runner, so the
  hardware paths have only been exercised against recorded-format fixtures.

[Unreleased]: https://github.com/seyhunak/kernelvet/compare/v0.3.0...HEAD
[0.3.0]: https://pypi.org/project/kernelvet/0.3.0/
[0.2.0]: https://pypi.org/project/kernelvet/0.2.0/
[0.1.1]: https://pypi.org/project/kernelvet/0.1.1/
[0.1.0]: https://pypi.org/project/kernelvet/0.1.0/
