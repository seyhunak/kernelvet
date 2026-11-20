# KernelVet documentation

KernelVet is an unofficial, independent toolkit for AMD GPU developers. It checks
whether a machine can really run ROCm workloads, and it helps move CUDA projects
to HIP using AMD's own tools. It is not affiliated with AMD.

## Start here

| Page | Read it when you want to… |
| --- | --- |
| [Getting started](getting-started.md) | install KernelVet and run your first check |
| [Use cases](use-cases.md) | find the workflow that matches what you are trying to do |
| [Command reference](commands.md) | look up a command, its options and its exit codes |

## Guides

| Page | What it covers |
| --- | --- |
| [How CUDA-to-HIP migration works](migration.md) | the analyze → plan → convert → audit → build → test → report pipeline, the finding categories, the five verification stages and their limits |
| [Diagnostics and compatibility](diagnostics.md) | what `diagnose`, `doctor` and `compatibility` check, the runtime probe, and the difference between hardware-verified and inferred results |
| [Reports and automation](reports.md) | terminal, JSON, Markdown, HTML and SARIF output, exit codes, JSON Schemas and scripting |
| [Configuration](configuration.md) | every setting, environment variables, history and Docker isolation |

## For contributors

| Page | What it covers |
| --- | --- |
| [Architecture](architecture.md) | module map and key design decisions |
| [Contributing](https://github.com/seyhunak/kernelvet/blob/main/CONTRIBUTING.md) | development setup and the rules every change must keep |
| [Notes for AI coding agents](https://github.com/seyhunak/kernelvet/blob/main/AGENTS.md) | invariants for automated contributors |

Back to the [project README](https://github.com/seyhunak/kernelvet/blob/main/README.md).
