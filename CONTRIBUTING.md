# Contributing to KernelVet

Thanks for helping. Bug reports, real hardware output, compatibility data fixes
and code are all welcome.

## Development setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Before you open a pull request, run the same checks CI runs:

```bash
ruff format --check src tests
ruff check src tests
mypy src
pytest
```

Useful pytest invocations:

```bash
pytest tests/unit/test_migration.py -k readiness   # one area
pytest -m "gpu or hipify"                          # real-tool tests (ROCm machine only)
```

## Ground rules

1. **Never fabricate results.** If the code cannot observe something, the check
   is SKIPPED (or UNAVAILABLE for a verification stage), with a reason. PRs that
   make a check pass without evidence will not be merged.
2. **Keep static and executed evidence apart.** Set `evidence` correctly on
   every new `CheckResult`.
3. **External commands** go through `HostContext.run` (or `CommandRunner.run`)
   and nothing else:
   * argv lists only, never shell strings,
   * always a timeout,
   * never `sudo`,
   * never modify drivers or system configuration.
4. **Never write into a user's CUDA project.** All output goes into a separate,
   validated directory (see `utils/paths.py`).
5. **Compatibility data must be cited.** If you change `rocm_matrix.json`:
   * update `retrieved` and the source URLs,
   * cite in the PR the AMD page each change comes from,
   * record disagreements between AMD pages in `notes`; don't pick a winner
     silently.
6. **Tests must not need a GPU.** Use `FakeRunner` and `make_sysroot` from
   `tests/conftest.py`. Fixture files you construct by hand must be labelled
   as constructed. Real recordings are better; include the tool version and
   strip hostnames and serial numbers.

## Adding a migration rule

Rules live in `src/kernelvet/migration/rules.py`. Each rule has:

* a category: `AUTO_CONVERTIBLE`, `REQUIRES_VERIFICATION`,
  `MANUAL_INTERVENTION` or `UNSUPPORTED`,
* a severity,
* a phase (`analyze` and/or `audit`),
* a target (`source` or `build`),
* a regex, plus the HIP equivalent if there is one.

Add a positive and a negative test in `tests/unit/test_migration.py`. The
regex runs on comment-stripped text, and line numbers must still match the
original file.

## Commits and pull requests

* Use small, focused PRs with a clear description of the observed behaviour.
* Use conventional commit prefixes (`feat:`, `fix:`, `docs:`, `test:`,
  `refactor:`, `ci:`).
* If you change output formats, update the JSON Schema tests and the
  README.

## Releasing (maintainers)

[`release.yml`](.github/workflows/release.yml) publishes to PyPI when a
`v*` tag is pushed. It runs the full CI first and uploads exactly the wheel
and sdist that CI built and smoke-tested. It then creates a GitHub release
with those files attached. Publishing uses PyPI
[trusted publishing](https://docs.pypi.org/trusted-publishers/), so the
repository holds no PyPI token.

One-time setup:

1. On pypi.org, under *Your projects → Publishing*, add a pending
   publisher. Use project `kernelvet`, owner `seyhunak`, repository
   `kernelvet`, workflow `release.yml` and environment `pypi`.
2. Optionally, do the same on test.pypi.org with environment `testpypi`.
3. In the GitHub repository settings, create the environments `pypi` and
   `testpypi`. Adding yourself as a required reviewer on `pypi` makes every
   release wait for your approval.

To cut a release:

1. Set `__version__` in `src/kernelvet/__init__.py`, move the `[Unreleased]`
   entries in `CHANGELOG.md` under the new version and date, then commit and
   push.
2. Optionally, rehearse: *Actions → Release → Run workflow* publishes that
   version to TestPyPI. TestPyPI never accepts the same version twice.
3. Tag and push. The workflow fails if the tag does not match the version.

   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

A version on PyPI can never be uploaded again, even after it is deleted. If
a release is broken, publish a new version.

By contributing, you agree that your contributions are licensed under the
Apache License 2.0.
