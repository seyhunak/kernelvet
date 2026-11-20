# Host fixtures

These files are **constructed** to follow the documented output formats of
`rocminfo`, `amd-smi static --json`, `hipcc --version` and `/etc/os-release`.
They were not recorded from a real machine and are used only by the test
suite (through a fake command runner). KernelVet never reads them at runtime.
