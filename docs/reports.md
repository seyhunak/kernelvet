# Reports and automation

[Docs home](README.md) · [Commands](commands.md) · [Diagnostics](diagnostics.md)

## Formats

Every reporting command takes `--format`/`-f` and `--output`/`-o FILE`. With
`-o`, the format is inferred from the extension unless `-f` is given.

| Format | Extension | Best for |
| --- | --- | --- |
| `terminal` | — | reading in a terminal (the default) |
| `json` | `.json` | scripts, storage, re-rendering with `export` |
| `markdown` | `.md` | pull requests, issues, wikis |
| `html` | `.html` | sharing: one self-contained file, inline CSS, no scripts, opens offline |
| `sarif` | `.sarif` | code-scanning tools; migration findings only |

```bash
kernelvet diagnose -o report.html
kernelvet migrate report ./proj-hip -f markdown -o MIGRATION.md
```

## What every report contains

* **Status and evidence for each check.** The evidence is one of: ran on GPU,
  detected (not run), inferred from AMD docs, or none. See
  [hardware-verified vs inferred](diagnostics.md#hardware-verified-vs-inferred-results).
* **An evidence legend**, and for diagnoses the number of checks verified on
  AMD hardware.
* **Sources.** Compatibility results cite the exact AMD page, section, ROCm
  version and retrieval date.
* **Disclaimers.** Compatibility reports state their basis: AMD's published
  data, nothing run on the GPU. Migration reports state that automated
  conversion is not guaranteed to produce working code.

## JSON

With a machine-readable format, stdout carries **only** the report. Progress,
warnings and disclaimers go to stderr.

```bash
kernelvet diagnose -f json --no-runtime 2>/dev/null \
  | jq -r '.checks[] | select(.status=="FAIL") | "\(.name): \(.remediation)"'
```

Useful fields on each check: `id`, `status`, `evidence`, `evidence_label`,
`hardware_verified`, `remediation` and `source`.

Get the JSON Schema of each report type to validate against:

```bash
kernelvet schema diagnosis       # or: compatibility, migration, benchmark
```

## SARIF

```bash
kernelvet migrate analyze ./proj -f sarif -o findings.sarif
kernelvet migrate audit ./proj-hip -f sarif -o audit.sarif
```

Each finding becomes a SARIF result with its file and line, rule ID and the HIP
equivalent. Every result has `"evidence": "static-analysis"` and
`"hardwareVerified": false`, because nothing in a SARIF report ran on a GPU.

## Re-rendering a saved report

```bash
kernelvet export diagnosis.json -o diagnosis.html
kernelvet export migration.json -f sarif -o findings.sarif
```

`export` accepts diagnosis, compatibility, migration and benchmark JSON.

## Scripting

| Exit code | Meaning |
| --- | --- |
| 0 | healthy / the stage passed |
| 1 | a failure was found |
| 2 | usage error |
| 3 | review required: warnings, skipped checks or unverified stages |
| 4 | a required tool is unavailable |
| 70 | internal error |

Example: fail on real problems but tolerate warnings.

```bash
kernelvet diagnose -f json -o diagnosis.json
rc=$?
if [ "$rc" -eq 1 ] || [ "$rc" -ge 4 ]; then
  echo "GPU host is unhealthy"; exit 1
fi
```

For a migration, `kernelvet migrate report HIP_DIR` exits 0 only when stage 5
(AMD GPU runtime validation) passed.

## History

Diagnostic and benchmark runs and migration projects are recorded in a local
SQLite database in the data directory (see [Configuration](configuration.md#directories)).
`kernelvet history` lists them. `history = false` turns this off;
`diagnose --no-history` skips one run.
