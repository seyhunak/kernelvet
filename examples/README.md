# Examples

## `config.toml`

An annotated configuration file listing every option. `kernelvet config path`
prints where your own copy goes.

## Walk-through: migrating the sample CUDA project

`tests/fixtures/cuda/vector_project` is a small CUDA project. It has:

* kernels, a cuBLAS GEMM and a shared header,
* a test,
* a CMake build and a Make build.

It also contains code that HIPIFY cannot translate, on purpose, so the audit
has something to find: inline PTX assembly, a 32-bit warp mask, warp-size-32
assumptions, NVTX calls and a cuBLAS include that becomes `<hipblas.h>`
(ROCm 10.x ships only `<hipblas/hipblas.h>`). **It is not expected to build
after conversion.**

The repository's test fixtures are not in the installed package. Clone the
repository to run this walk-through.

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
* With HIPIFY and `hipcc` (checked in the `rocm/dev-ubuntu-24.04` image,
  ROCm 10.2.0, no GPU): `convert` exits 0, `audit` exits 1 with 26 findings
  (including `cuda.inline_ptx` UNSUPPORTED, `cuda.full_mask`, `lib.nvtx` and
  `audit.flat_rocm_include`), and `build` exits 1 on `'hipblas.h' file not
  found` and HIP's 64-bit lane-mask assertion. The stages end as source
  conversion `PASSED`, static audit `FAILED`, compilation `FAILED`, tests and
  GPU validation `NOT_RUN`.
* Fixing those findings by hand is the porting work. Only after that can the
  build pass, and only on an AMD GPU can the last stage pass. On a machine
  without an AMD GPU, stage 5 is `UNAVAILABLE`, never `PASSED`.

## Using JSON output in scripts

```bash
kernelvet diagnose -f json --no-runtime 2>/dev/null \
  | jq -r '.checks[] | select(.status=="FAIL") | "\(.name): \(.remediation)"'
```

`kernelvet schema diagnosis` prints the JSON Schema you can validate against.
