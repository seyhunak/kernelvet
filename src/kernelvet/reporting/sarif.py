"""SARIF 2.1.0 export for static-analysis findings (migration analysis and audit)."""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Any

from kernelvet import __version__
from kernelvet.models.migration import Finding
from kernelvet.models.report import MigrationReport

SARIF_SCHEMA = "https://json.schemastore.org/sarif-2.1.0.json"


def findings_to_sarif(
    findings: list[Finding], *, tool_name: str = "kernelvet", base_uri: str | None = None
) -> dict[str, Any]:
    rules: dict[str, dict[str, Any]] = {}
    results = []
    for f in findings:
        rules.setdefault(
            f.rule_id,
            {
                "id": f.rule_id,
                "name": f.rule_id.replace(".", "_"),
                "shortDescription": {"text": f.message.split(":")[0][:120]},
                "properties": {"category": str(f.category)},
            },
        )
        loc: dict[str, Any] = {"artifactLocation": {"uri": str(PurePosixPath(f.file))}}
        if base_uri:
            loc["artifactLocation"]["uriBaseId"] = "SRCROOT"
        if f.line:
            region: dict[str, Any] = {"startLine": f.line}
            if f.column:
                region["startColumn"] = f.column
            if f.snippet:
                region["snippet"] = {"text": f.snippet}
            loc["region"] = region
        result: dict[str, Any] = {
            "ruleId": f.rule_id,
            "level": f.severity.sarif_level,
            "message": {"text": f.message + (f" Suggested: {f.suggestion}" if f.suggestion else "")},
            "locations": [{"physicalLocation": loc}],
            "properties": {"category": str(f.category), "severity": str(f.severity)},
        }
        if f.hip_equivalent:
            result["properties"]["hipEquivalent"] = f.hip_equivalent
        results.append(result)
    run: dict[str, Any] = {
        "tool": {
            "driver": {
                "name": tool_name,
                "version": __version__,
                "informationUri": "https://github.com/seyhunak/kernelvet",
                "rules": list(rules.values()),
            }
        },
        "results": results,
    }
    if base_uri:
        run["originalUriBaseIds"] = {"SRCROOT": {"uri": base_uri.rstrip("/") + "/"}}
    return {"$schema": SARIF_SCHEMA, "version": "2.1.0", "runs": [run]}


def migration_to_sarif(report: MigrationReport) -> str:
    """Audit findings (migrated tree) when present, otherwise analysis findings (CUDA tree).

    The two refer to different directory trees, so they are never mixed in one run.
    """
    findings: list[Finding] = []
    root: str | None = None
    if report.audit:
        findings, root = report.audit.findings, report.audit.project
    elif report.analysis:
        findings, root = report.analysis.findings, report.analysis.inventory.root
    base = PurePosixPath(root).as_uri() if root and root.startswith("/") else None
    return json.dumps(findings_to_sarif(findings, base_uri=base), indent=2)
