## What and why

<!-- What does this change, and what problem does it solve? -->

## How it was verified

- [ ] `ruff format --check src tests && ruff check src tests`
- [ ] `mypy src`
- [ ] `pytest`
- [ ] Real-tool tests (`pytest -m "gpu or hipify"`) on: <!-- GPU / ROCm version, or "not run" -->

## Checklist

- [ ] No check reports PASS without evidence; missing tools/devices give SKIPPED or UNAVAILABLE
- [ ] External commands use argv lists with timeouts (no shell strings, no sudo)
- [ ] Nothing writes into the user's CUDA source tree
- [ ] Compatibility data changes cite the AMD source URL and update `retrieved`
- [ ] Docs / README updated if behaviour or output changed
