# Examples

## `config.toml`

An annotated configuration file listing every option. `kernelvet config path`
prints where your own copy goes.

## Walk-through: migrating the sample CUDA project

`tests/fixtures/cuda/vector_project` is a small CUDA project. It has:

* kernels, a cuBLAS GEMM and a shared header,
* a test,
* a CMake build and a Make build.

Run it through the whole pipeline. Work on a copy: KernelVet never modifies
the source, but the fixture is shared with the test suite.

```bash
cp -R tests/fixtures/cuda/vector_project /tmp/vector_project

kernelvet migrate analyze /tmp/vector_project          # findings by category + readiness score
kernelvet migrate plan    /tmp/vector_project          # dry run
kernelvet migrate convert /tmp/vector_project -o /tmp/vector_project-hip   # needs hipify-perl or hipify-clang
kernelvet migrate audit   /tmp/vector_project-hip
kernelvet migrate build   /tmp/vector_project-hip      # needs hipcc
kernelvet migrate test    /tmp/vector_project-hip      # GPU validation needs an AMD GPU
kernelvet migrate report  /tmp/vector_project-hip -o migration.html
kernelvet migrate report  /tmp/vector_project-hip -f sarif -o findings.sarif
```

What to expect on a machine without ROCm:

* `convert` exits with code 4, explains that HIPIFY is missing, and writes
  nothing.
* `kernelvet migrate report /tmp/vector_project` still shows the analysis,
  with all five verification stages `NOT_RUN`.
* On a machine with HIPIFY and `hipcc` but no AMD GPU, the last stage is
  `UNAVAILABLE`. It is never `PASSED`.

## Using JSON output in scripts

```bash
kernelvet diagnose -f json --no-runtime 2>/dev/null \
  | jq -r '.checks[] | select(.status=="FAIL") | "\(.name): \(.remediation)"'
```

`kernelvet schema diagnosis` prints the JSON Schema you can validate against.
