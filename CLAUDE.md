# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

KernelVet is a Python 3.11+ CLI (Typer) and TUI (Textual) for AMD GPU
diagnostics and CUDA→HIP migration. The package is in `src/kernelvet`; the
console script is `kernelvet`. The project's invariants are in
[AGENTS.md](AGENTS.md); read it before changing behaviour. The module map is in
[docs/architecture.md](docs/architecture.md).

## Commands

```bash
uv venv -p 3.11 .venv && uv pip install -e ".[dev]"   # or: python3.11 -m venv .venv && pip install -e ".[dev]"
.venv/bin/ruff format src tests && .venv/bin/ruff check src tests
.venv/bin/mypy src
.venv/bin/pytest -q                                   # ~10 s, CPU-only
.venv/bin/pytest tests/unit/test_migration.py -k readiness
.venv/bin/pytest -m "gpu or hipify"                   # real tools; skipped without ROCm
scripts/smoke_test.sh .venv/bin/kernelvet             # CLI end to end on a GPU-less host (what CI runs on the wheel)
```

The development machine is usually macOS without ROCm. Every runtime check is
SKIPPED there, and `migrate convert` exits 4 (no HIPIFY). That is correct
behaviour, not a bug. To reproduce the Linux CI run:

```bash
docker run --rm -v "$PWD":/src:ro python:3.11-slim bash -c 'cp -r /src /w && cd /w && pip install -q -e ".[dev]" && pytest -q'
```

## How the pieces fit

* The CLI (`cli/commands/*`) and TUI (`tui/screens/*`) are thin front ends.
  Each one calls the same engine functions (`run_diagnosis`, `evaluate`,
  `analyze_project`, `convert_project`, `audit_project`, `build_project`,
  `run_tests`, `run_benchmarks`), which return Pydantic models from `models/`.
  Rendering is in `reporting/`.
* All host access goes through `utils/proc.HostContext`: subprocesses via
  `.run()` (argv only, with a timeout), and file reads via
  `.path()`/`.read_text()`/`.exists()` relative to `ctx.root`. Tests replace
  both with `FakeRunner` and `make_sysroot` (`tests/conftest.py`). Never call
  `subprocess` or open `/sys`/`/opt/rocm` paths directly.
* Migration stages share state through `<hip-output>/.kernelvet/*.json`
  (`migration/workflow.save_state`/`load_state`). `compute_stages` derives the
  five verification stages from those files. GPU validation passes only if the
  tests passed and a passing test's HIP runtime log (`AMD_LOG_LEVEL=3`) showed a
  successful kernel launch on a detected AMD GPU (`execution/runner.py`). A GPU
  being present is not enough.
* The CLI tests patch `HostContext` in each command module (see the
  `fake_host`/`bare_host` fixtures in `tests/integration/test_cli.py`). A new
  command module that builds its own `HostContext` must be added to that patch
  list.
* The TUI's long tasks are `@work(thread=True, exclusive=True, exit_on_error=False)`
  workers that update widgets only through `self.app.call_from_thread(...)`.
* The compatibility data is
  `src/kernelvet/compatibility/data/rocm_matrix.json`, a hand-transcribed and
  dated snapshot of AMD docs. Any edit must keep the `sources`, `retrieved`
  and `notes` fields accurate.

## Gotchas

* `ruff format` owns line length (110). E501 is ignored on purpose.
* Rich wraps CLI output at 80 columns under `CliRunner`. Assert on `flat(r.stdout)`
  (whitespace-collapsed), not on exact lines.
* Tests that modify a fixture project must use the `cuda_project`/`hip_project`
  fixtures, which copy it. Writing into `tests/fixtures/` leaks state into
  later tests.
* `KERNELVET_CONFIG_DIR`/`KERNELVET_DATA_DIR` isolate config and history. An
  autouse fixture sets both for every test.
* The previous Rust/WASM implementation is in git history on `main`. Nothing
  here depends on it.
