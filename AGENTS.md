# AGENTS.md

Guidance for AI coding agents working in this repository. Read
[docs/architecture.md](docs/architecture.md) first.

## Commands

```bash
pip install -e ".[dev]"
ruff format src tests && ruff check src tests && mypy src
pytest -q                                   # full suite, CPU-only, ~10 s
pytest tests/unit/test_diagnose.py::test_name
kernelvet --help
```

## Invariants you must keep

* **Honest results.**
  * A check reaches PASS only on evidence.
  * Missing tools or devices make a check SKIPPED, or a verification stage
    UNAVAILABLE.
  * Never add a mock or placeholder that makes a feature *look* like it works.
  * Benchmark numbers come only from real runs.
* **`evidence` matters.** `executed` means code really ran on the GPU;
  `detected` means static inspection. A GPU being present is never evidence
  that something ran on it: migration tests count as GPU-executed only when the
  HIP runtime log shows a successful kernel launch (`execution/runner.py`).
* **Subprocesses** go through `HostContext.run`:
  * argv lists only, never shell strings,
  * a timeout on every call,
  * environment scrubbed and output redacted (`utils/redact.py`).
* **File reads** of system paths go through `HostContext.path` / `read_text`
  / `exists`, so tests can point them at a fake sysroot.
* **The CUDA source tree is read-only.** Output paths are checked by
  `utils/paths.safe_output_dir`. Migration state goes into
  `<output>/.kernelvet/`.
* **CUDA is never translated by our own code.** Conversion is HIPIFY's job;
  our rules only classify.
* **`--format json` keeps stdout clean.** Progress and messages go to stderr
  (`cli/output.err`).
* **Exit codes:** 0 ok, 1 failure, 2 usage, 3 review, 4 unavailable, 70 internal.

## Tests

* Tests must pass without ROCm, a GPU or HIPIFY.
* Use `FakeRunner` and `make_sysroot` from `tests/conftest.py`.
* Copy fixture projects before modifying them (`cuda_project` / `hip_project`
  fixtures do this).
* Tests that need real tools carry `@pytest.mark.gpu` or
  `@pytest.mark.hipify` and skip themselves when the tool is absent.
