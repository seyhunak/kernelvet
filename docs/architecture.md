# Architecture

KernelVet is one Python package, `src/kernelvet`. The CLI (Typer) and the TUI
(Textual) are two thin front ends over the same engine. Neither front end has
its own logic: each one builds options, calls an engine function and renders
the Pydantic model it gets back.

```
            ┌──────────── cli/ (Typer) ────────────┐   ┌──── tui/ (Textual) ────┐
            │ commands/diagnose.py  migrate.py ... │   │ screens/*  workers     │
            └───────────────┬──────────────────────┘   └───────────┬────────────┘
                            │      options + progress callback     │
   ┌────────────────────────┴──────────────────────────────────────┴───────────┐
   │ diagnose/   compatibility/   migration/   execution/   benchmark/         │
   │                         (engine, pure functions)                          │
   └───────────────┬───────────────────────────────────────────┬───────────────┘
                   │ HostContext (runner, sysroot, env, timeout) │ models/ (Pydantic)
   ┌───────────────┴──────────────┐                    ┌────────┴──────────────┐
   │ utils/proc.py  CommandRunner │                    │ reporting/  history/  │
   │ argv only · timeout · scrub  │                    │ json md html sarif    │
   └──────────────────────────────┘                    └───────────────────────┘
```

## Packages

| Package | Responsibility |
| --- | --- |
| `models/` | Every data structure that crosses a boundary: checks, host/GPU/software inventory, migration findings and stages, reports. JSON output and JSON Schemas come straight from these models. |
| `utils/proc.py` | `CommandRunner` runs external tools: argv only (`shell=False`), a timeout on every call, a scrubbed environment and redacted output. `HostContext` bundles a runner with a *sysroot* (the base used for every file read), the environment, the platform and the default timeout. |
| `utils/paths.py` | Output-path validation. It guarantees a migration never writes inside, over, or around the CUDA source tree. |
| `utils/redact.py` | Removes secrets (tokens, keys, passwords, credentials in URLs) from command output, logs and reports. |
| `diagnose/` | Collects the host (`system.py`), GPUs (`hardware.py`: amd-smi + rocminfo + sysfs, merged by PCI bus) and software (`software.py`). Runs the HIP and PyTorch probes on the device (`runtime.py`). `engine.py` orchestrates all of it. |
| `compatibility/` | The bundled AMD matrix snapshot (`data/rocm_matrix.json`), its loader, one validator per dimension (GPU, OS, kernel, ROCm, driver, HIP, PyTorch) and `evaluate()`. |
| `migration/` | Rule catalogue (`rules.py`), project scanner, analyzer and readiness score, planner, HIPIFY integration, post-conversion auditor and the workflow that turns per-stage state into the five verification stages. |
| `execution/` | Building (`compiler.py`: a HIP-aware CMake build, otherwise a planned direct `hipcc` build), running tests (`runner.py`) and the optional Docker sandbox (`sandbox.py`). |
| `benchmark/` | HIP and PyTorch workloads, compiled and run for real; an optional `rocprofv3` pass. |
| `reporting/` | Renderers: Rich terminal, JSON, Markdown, self-contained HTML, SARIF 2.1.0. `loader.py` reads a saved JSON report back into its model. |
| `history/` | A SQLite store of past runs and projects, plus saved JSON reports. History is optional: an unusable store raises `HistoryError`, and callers warn instead of failing. |
| `config/` | `Settings` built from defaults, then the TOML file, then `KERNELVET_*` environment variables. |

## Key decisions

**Static vs executed evidence.** Every `CheckResult` carries an `evidence` field.
A GPU that `rocminfo` lists is `detected`. A GPU that ran the probe kernel and
returned the correct result is `executed`. Compatibility checks are
`configured`: they are inferred from AMD's documentation. The JSON values are a
stable API. People see the labels from `Evidence.label` instead: "ran on GPU",
"detected, not run", "inferred from AMD docs" and "none". The legend is
`EVIDENCE_LEGEND`, in `models/common.py`. `CheckResult.hardware_verified` is
true only for an `executed` PASS. That is the only thing the CLI, TUI and
reports count as "verified on AMD hardware", and
`DiagnosticReport.hardware_verified_checks` lists those checks.

**Matrix provenance.** Every rule in `rocm_matrix.json` has a `ref` list. It
points into `references`, which gives the page, section, row, anchored URL,
ROCm version, retrieval date and verification status. The loader rejects
undefined refs, and also an `unverified` reference that has no `note`.
Validators pass the refs behind each decision to `CompatibilityMatrix.cite()`,
so a check's `source` names the exact AMD section. Schema 1 files (no refs)
still load and cite page-level URLs.

**Per-product OS limits.** `os_gpu_restrictions` name products (`gpus` or
`except_gpus`), not LLVM targets, because MI300X and MI300A (both gfx942)
differ on Debian 13 and Oracle Linux. `CompatibilityMatrix.identify_gpu`
matches the detected marketing name against the products for that target, word
by word with the longest match winning, so "MI250" never matches "MI250X". An
unidentified GPU passes only if every product with its target is allowed;
otherwise the result is WARNING. The schema 1 `gfx` form still works.

**Skip, don't guess.** If a tool or device is missing, the checks that need it
become SKIPPED, each with a reason. When one step of the HIP probe fails
(init → enumerate → alloc → h2d → kernel → d2h → verify), every later step is
SKIPPED, not FAIL. This makes the first failure the obvious place to look.

**Never write to the CUDA tree.** `convert` writes only into a validated output
directory. Per-stage state (`conversion.json`, `audit.json`,
`build.json`, `tests.json`, plus `conversion.log`) lives in
`<output>/.kernelvet/`. If HIPIFY fails on a file, the original is copied
unchanged and marked `failed`, so the audit flags its leftover CUDA.

**We don't translate CUDA ourselves.** Translation is always done by
`hipify-clang` or `hipify-perl`. KernelVet's own rules only *classify* code,
before conversion (analyze) and after it (audit). HIPIFY output is not
guaranteed to work. `HIPIFY_DISCLAIMER` (in `models/report.py`) appears in
`migrate convert` output, in `MigrationReport.conversion_disclaimer` and in
every migration export.

**Translator choice.** `hipify-clang` is chosen only when CUDA headers are found,
because it parses CUDA with clang. Otherwise `hipify-perl` is used. The reason
for the choice is recorded in the log.

**Background work in the TUI.** Every long operation runs in a
`@work(thread=True, exclusive=True)` worker. The worker reports progress and
results through `app.call_from_thread`, so the UI stays responsive and keyboard
navigation keeps working.

## Testing seams

* `HostContext(runner=FakeRunner(...), root=tmp_sysroot)` replays recorded
  command output and fake `/sys`, `/dev`, `/opt/rocm` and `/etc` trees, so the
  whole diagnose and compatibility pipeline runs on any CI machine.
* Fixtures under `tests/fixtures/cuda/` are small but realistic CUDA projects.
  Tests always work on *copies* of them.
* `tests/integration/test_real_tools.py` runs the real HIPIFY and the real GPU
  probe. It is skipped unless those tools are present (`-m "gpu or hipify"`).
