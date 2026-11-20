# Changelog

All notable changes to KernelVet are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Before 1.0, a minor version may
include breaking changes.

## [Unreleased]

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

[Unreleased]: https://github.com/seyhunak/kernelvet/compare/v0.2.0...HEAD
[0.2.0]: https://pypi.org/project/kernelvet/0.2.0/
[0.1.1]: https://pypi.org/project/kernelvet/0.1.1/
[0.1.0]: https://pypi.org/project/kernelvet/0.1.0/
