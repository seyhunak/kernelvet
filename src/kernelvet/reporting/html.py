"""Self-contained HTML export (inline CSS, no scripts, opens offline)."""

from __future__ import annotations

import html

from pydantic import BaseModel

from kernelvet.models.report import (
    BenchmarkReport,
    CompatibilityReport,
    DiagnosticReport,
    MigrationReport,
)
from kernelvet.reporting.markdown import (
    benchmark_markdown,
    compatibility_markdown,
    diagnostic_markdown,
    migration_markdown,
)

CSS = """
:root{--bg:#0f1115;--panel:#161a21;--fg:#e6e6e6;--muted:#9aa4b2;--line:#2a313c;
--pass:#3fb950;--warn:#d29922;--fail:#f85149;--skip:#8b949e;--accent:#ed1c24}
@media (prefers-color-scheme: light){:root{--bg:#fff;--panel:#f6f8fa;--fg:#1f2328;--muted:#59636e;--line:#d1d9e0}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 ui-sans-serif,system-ui,sans-serif;margin:0;padding:24px}
main{max-width:1100px;margin:auto}
h1{border-bottom:3px solid var(--accent);padding-bottom:6px}
h2{margin-top:28px;border-bottom:1px solid var(--line);padding-bottom:4px}
table{border-collapse:collapse;width:100%;margin:8px 0;background:var(--panel)}
th,td{border:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600}
code{font-family:ui-monospace,monospace;font-size:12.5px}
blockquote{border-left:3px solid var(--line);margin:8px 0;padding:4px 12px;color:var(--muted)}
pre{background:var(--panel);padding:10px;overflow:auto;border:1px solid var(--line)}
.PASS{color:var(--pass);font-weight:600}.WARNING{color:var(--warn);font-weight:600}
.FAIL{color:var(--fail);font-weight:600}.SKIPPED{color:var(--skip);font-weight:600}
"""


def _inline(text: str) -> str:
    """Escape, then render `code` and **bold** spans."""
    out = html.escape(text)
    parts = out.split("`")
    out = "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))
    parts = out.split("**")
    out = "".join(f"<strong>{p}</strong>" if i % 2 else p for i, p in enumerate(parts))
    for status in ("PASS", "WARNING", "FAIL", "SKIPPED"):
        out = out.replace(f" {status} ", f' <span class="{status}">{status}</span> ')
    return out.replace("&lt;br&gt;", "<br>")


def markdown_to_html(md: str, title: str) -> str:
    """Convert the Markdown subset our exporters emit. Not a general Markdown parser."""
    body: list[str] = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            j = i + 1
            block = []
            while j < len(lines) and not lines[j].startswith("```"):
                block.append(html.escape(lines[j]))
                j += 1
            body.append("<pre>" + "\n".join(block) + "</pre>")
            i = j + 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            cells = [[c.strip().replace("\\|", "|") for c in _split_row(r)] for r in rows]
            head, data = cells[0], [r for r in cells[1:] if not all(set(c) <= {"-", ":"} for c in r)]
            body.append(
                "<table><thead><tr>"
                + "".join(f"<th>{_inline(c)}</th>" for c in head)
                + "</tr></thead><tbody>"
                + "".join("<tr>" + "".join(f"<td>{_cell(c)}</td>" for c in r) + "</tr>" for r in data)
                + "</tbody></table>"
            )
            continue
        if line.startswith("# "):
            body.append(f"<h1>{_inline(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{_inline(line[3:])}</h2>")
        elif line.startswith("### "):
            body.append(f"<h3>{_inline(line[4:])}</h3>")
        elif line.startswith("> "):
            body.append(f"<blockquote>{_inline(line[2:])}</blockquote>")
        elif line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{_inline(lines[i][2:])}</li>")
                i += 1
            body.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif line.strip():
            text = line.strip("_") if line.startswith("_") and line.endswith("_") else line
            body.append(f"<p>{_inline(text)}</p>")
        i += 1
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
        f"<body><main>{''.join(body)}</main></body></html>\n"
    )


def _split_row(row: str) -> list[str]:
    cells, cur, esc = [], "", False
    for ch in row.strip().strip("|"):
        if ch == "\\" and not esc:
            esc = True
            cur += ch
            continue
        if ch == "|" and not esc:
            cells.append(cur)
            cur = ""
        else:
            cur += ch
        esc = False
    cells.append(cur)
    return cells


def _cell(text: str) -> str:
    for icon, status in (
        ("✅ PASS", "PASS"),
        ("⚠️ WARNING", "WARNING"),
        ("❌ FAIL", "FAIL"),
        ("⏭️ SKIPPED", "SKIPPED"),
    ):
        if text == icon:
            return f'<span class="{status}">{html.escape(icon)}</span>'
    return _inline(text)


def to_html(report: BaseModel) -> str:
    if isinstance(report, DiagnosticReport):
        return markdown_to_html(diagnostic_markdown(report), "KernelVet System Diagnosis")
    if isinstance(report, MigrationReport):
        return markdown_to_html(migration_markdown(report), "KernelVet Migration Report")
    if isinstance(report, BenchmarkReport):
        return markdown_to_html(benchmark_markdown(report), "KernelVet Benchmarks")
    if isinstance(report, CompatibilityReport):
        return markdown_to_html(compatibility_markdown(report), "KernelVet Compatibility")
    raise TypeError(f"no HTML exporter for {type(report).__name__}")
