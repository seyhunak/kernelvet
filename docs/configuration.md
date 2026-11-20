# Configuration

[Docs home](README.md) · [Commands](commands.md)

Settings are resolved in this order, later ones winning:

1. built-in defaults,
2. the config file: `kernelvet config path` prints its location, usually
   `~/.config/kernelvet/config.toml` (or `$XDG_CONFIG_HOME/kernelvet/config.toml`),
3. environment variables named `KERNELVET_<SETTING>`, for example
   `KERNELVET_ISOLATION=docker`.

```bash
kernelvet config show                        # effective settings
kernelvet config set isolation docker        # change one setting
kernelvet config set isolation ""            # reset to the default
kernelvet config set offload_arch none       # clear an optional setting
```

An annotated file with every option is in
[`examples/config.toml`](https://github.com/seyhunak/kernelvet/blob/main/examples/config.toml).

## Settings

| Setting | Default | Used by | Meaning |
| --- | --- | --- | --- |
| `command_timeout` | `30.0` | all detection | timeout in seconds for each detection command (`rocminfo`, `amd-smi`, …) |
| `build_timeout` | `900.0` | `migrate convert`, `migrate build` | timeout in seconds for HIPIFY and builds |
| `test_timeout` | `300.0` | `migrate test` | timeout in seconds per test executable |
| `runtime_checks` | `true` | `diagnose`, `doctor` | compile and run the HIP probe on the GPU |
| `torch_checks` | `true` | `diagnose`, `doctor`, `compatibility` | detect PyTorch and run tensor checks on the GPU |
| `python_executable` | unset | PyTorch checks, benchmarks | Python to inspect for PyTorch (default: the interpreter KernelVet runs in) |
| `hipify_preference` | `auto` | `migrate plan`, `migrate convert` | `auto`, `hipify-clang` or `hipify-perl`; see [translator choice](migration.md#2-plan-a-dry-run) |
| `isolation` | `none` | `migrate build`, `migrate test` | `docker` runs builds and tests in a container |
| `docker_image` | `rocm/dev-ubuntu-24.04:latest` | Docker isolation | the image used for isolated builds and tests |
| `offload_arch` | unset | `migrate plan`, `migrate build` | `--offload-arch` targets, comma-separated (default: the detected GPU targets) |
| `compatibility_matrix` | unset | compatibility checks | path to a custom matrix JSON (same schema as the bundled one) |
| `history` | `true` | `diagnose`, `doctor`, `benchmark`, `migrate` | record runs in the local SQLite history |

Command-line options override settings for that run, for example
`--no-runtime`, `--translator hipify-perl` or `--isolation docker`.

## Directories

| What | Default | Override |
| --- | --- | --- |
| config file | `~/.config/kernelvet/config.toml` | `KERNELVET_CONFIG_DIR` |
| history database, saved reports, profiler output | `~/.local/share/kernelvet/` | `KERNELVET_DATA_DIR` |

`kernelvet config show` also prints the data directory.

## Docker isolation

With `isolation = docker`, `migrate build` and `migrate test` run inside
`docker_image` with:

* no network,
* every Linux capability dropped,
* `/dev/kfd` and `/dev/dri` passed through only for runs that need the GPU.

Only Docker itself is needed on the host; `hipcc` comes from the image.

## A custom compatibility matrix

`--matrix FILE` or the `compatibility_matrix` setting replaces the bundled
snapshot of AMD's ROCm support data. Use the bundled
[`rocm_matrix.json`](https://github.com/seyhunak/kernelvet/blob/main/src/kernelvet/compatibility/data/rocm_matrix.json)
as the template. The loader rejects a file that doesn't match the schema, a
rule pointing at a reference that doesn't exist, and an `unverified` reference
with no `note`. See [Diagnostics and compatibility](diagnostics.md#compatibility-data).

KernelVet never needs root and never changes drivers or system configuration.
