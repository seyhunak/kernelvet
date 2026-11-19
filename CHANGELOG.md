# Changelog

All notable changes to KernelVet are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Before 1.0, a minor version may
include breaking changes.

## [Unreleased]

### Added

- Python 3.14 is tested in CI and listed in the package classifiers.
- This changelog.

### Changed

- CI and release workflows use `actions/checkout@v7`, `actions/setup-python@v7`,
  `actions/upload-artifact@v7` and `actions/download-artifact@v8`.
- Dependabot version-update PRs are turned off (`.github/dependabot.yml`
  removed). Dependency updates are done by hand.

### Fixed

- SARIF export no longer uses `PurePath.as_uri()`, which is deprecated in
  Python 3.14. The generated `file://` URI is unchanged.

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

[Unreleased]: https://github.com/seyhunak/kernelvet/compare/v0.1.1...HEAD
[0.1.1]: https://pypi.org/project/kernelvet/0.1.1/
[0.1.0]: https://pypi.org/project/kernelvet/0.1.0/
