# Host fixtures

These files are **constructed** to follow the documented output formats of
`rocminfo`, `amd-smi static --json`, `hipcc --version` and `/etc/os-release`.
They were not recorded from a real machine and are used only by the test
suite (through a fake command runner). KernelVet never reads them at runtime.

The one exception is `mi300x/hipcc-version.txt`, which is the real
`hipcc --version` output of the `rocm/dev-ubuntu-24.04` image (ROCm 10.2.0,
HIP 7.17.26434), recorded on 2026-11-20 in a container without a GPU.
