"""Compatibility reference data.

The bundled matrix (``data/rocm_matrix.json``) is a hand-transcribed snapshot of
AMD's published ROCm documentation. It records its source URLs, retrieval date
and known discrepancies, and can be replaced with ``--matrix path.json`` or the
``compatibility_matrix`` setting without changing any code.

Provenance (schema 2): every rule carries ``ref``, a list of keys into
``references``. Each reference names the exact AMD page, section and row, the
ROCm version that page describes, the date it was transcribed and the date it
was last re-checked. A reference that could not be re-checked is marked
``"verification": "unverified"`` and must say why in ``note``. Schema 1 files
(no ``references``) still load; their checks cite the page-level ``sources``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, ValidationError, model_validator

GpuClass = Literal["instinct", "radeon"]


class MatrixError(ValueError):
    """A compatibility matrix file that cannot be read, is not JSON, or does not match the schema."""


class MatrixSource(BaseModel):
    title: str
    url: str
    page_date: str | None = None
    describes: str | None = None
    latest_url: str | None = None
    retrieved: str | None = None
    note: str | None = None


class MatrixReference(BaseModel):
    """The exact place in AMD's documentation that one or more rules were transcribed from."""

    source: str  # key into CompatibilityMatrix.sources
    url: str  # page URL including the section anchor
    page: str
    section: str
    detail: str  # table, row or footnote within the section
    rocm_version: str  # the ROCm release that page describes
    retrieved: str  # date the rule was transcribed
    verification: Literal["verified", "unverified"]
    verified_on: str | None = None  # date it was last re-checked against the page
    note: str | None = None

    @model_validator(mode="after")
    def _verification_is_explained(self) -> Self:
        if self.verification == "verified" and not self.verified_on:
            raise ValueError("a verified reference needs verified_on")
        if self.verification == "unverified" and not self.note:
            raise ValueError("an unverified reference needs a note saying why")
        return self

    def cite(self) -> str:
        flag = "" if self.verification == "verified" else ", UNVERIFIED"
        return (
            f"AMD docs: {self.page} > {self.section} > {self.detail} "
            f"(ROCm {self.rocm_version}, retrieved {self.retrieved}{flag}) {self.url}"
        )


class GpuEntry(BaseModel):
    name: str
    gpu_class: GpuClass = Field(alias="class")
    arch: str
    gfx: str
    status: Literal["supported", "deprecated", "unsupported"]
    source: str
    note: str | None = None
    ref: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class OsEntry(BaseModel):
    id: str
    version: str
    label: str
    kernels: list[str] = Field(default_factory=list)
    source: str
    ref: list[str] = Field(default_factory=list)


class OsRestriction(BaseModel):
    os: str  # "debian:13"
    gfx: list[str]
    note: str
    source: str
    ref: list[str] = Field(default_factory=list)


class FrameworkEntry(BaseModel):
    versions: list[str]
    source: str
    ref: list[str] = Field(default_factory=list)


class DriverTable(BaseModel):
    instinct: list[str] = Field(default_factory=list)
    radeon: list[str] = Field(default_factory=list)
    source: str
    ref: list[str] = Field(default_factory=list)


class CompatibilityMatrix(BaseModel):
    schema_version: int = Field(alias="schema")
    rocm_version: str
    retrieved: str
    sources: dict[str, MatrixSource]
    references: dict[str, MatrixReference] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    release_versions: list[str] = Field(default_factory=list)
    release_versions_ref: list[str] = Field(default_factory=list)
    gpus: list[GpuEntry]
    targets_uncertain: dict[str, str] = Field(default_factory=dict)
    targets_uncertain_ref: list[str] = Field(default_factory=list)
    os: dict[GpuClass, list[OsEntry]]
    os_gpu_restrictions: list[OsRestriction] = Field(default_factory=list)
    amdgpu_driver: DriverTable
    frameworks: dict[str, FrameworkEntry] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _refs_resolve(self) -> Self:
        for key, r in self.references.items():
            if r.source not in self.sources:
                raise ValueError(f"reference {key!r} names unknown source {r.source!r}")
        unknown = sorted({k for refs in self.rule_refs().values() for k in refs} - set(self.references))
        if unknown:
            raise ValueError(f"rules refer to undefined references: {', '.join(unknown)}")
        return self

    def rule_refs(self) -> dict[str, list[str]]:
        """Every rule in the matrix, labelled, with its ``ref`` list."""
        out: dict[str, list[str]] = {
            "release_versions": self.release_versions_ref,
            "targets_uncertain": self.targets_uncertain_ref,
            "amdgpu_driver": self.amdgpu_driver.ref,
        }
        out.update({f"gpus[{g.name}]": g.ref for g in self.gpus})
        out.update({f"os.{c}[{e.label}]": e.ref for c, entries in self.os.items() for e in entries})
        out.update({f"os_gpu_restrictions[{r.os}]": r.ref for r in self.os_gpu_restrictions})
        out.update({f"frameworks.{k}": f.ref for k, f in self.frameworks.items()})
        return out

    def source_url(self, key: str) -> str:
        urls = [self.sources[k].url for k in key.split("+") if k in self.sources]
        return ", ".join(urls) if urls else key

    def cite(self, refs: list[str], fallback_source: str) -> str:
        """The exact AMD documentation references for a rule, or the page-level source URLs for a
        matrix that has no ``references`` (schema 1)."""
        found = [self.references[k] for k in dict.fromkeys(refs) if k in self.references]
        if not found:
            return self.source_url(fallback_source)
        return "; ".join(r.cite() for r in found)

    def entries_for_gfx(self, gfx: str) -> list[GpuEntry]:
        return [g for g in self.gpus if g.gfx == gfx]

    def class_for_gfx(self, gfx: str | None) -> GpuClass | None:
        if not gfx:
            return None
        entries = self.entries_for_gfx(gfx)
        return entries[0].gpu_class if entries else None

    def source_urls(self) -> list[str]:
        return [s.url for s in self.sources.values()]


@lru_cache(maxsize=8)
def _load_cached(path: str | None) -> CompatibilityMatrix:
    where = path or "the bundled compatibility matrix"
    try:
        if path:
            raw = Path(path).read_text()
        else:
            raw = resources.files("kernelvet.compatibility").joinpath("data/rocm_matrix.json").read_text()
        return CompatibilityMatrix.model_validate(json.loads(raw))
    except OSError as exc:
        raise MatrixError(f"could not read {where}: {exc.strerror or exc}") from exc
    except json.JSONDecodeError as exc:
        raise MatrixError(
            f"{where} is not valid JSON (line {exc.lineno}, column {exc.colno}: {exc.msg})"
        ) from exc
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(x) for x in e['loc']) or '(top level)'}: {e['msg']}" for e in exc.errors()[:5]
        )
        more = f" (and {exc.error_count() - 5} more)" if exc.error_count() > 5 else ""
        raise MatrixError(f"{where} does not match the matrix schema: {problems}{more}") from exc


def load_matrix(path: str | Path | None = None) -> CompatibilityMatrix:
    return _load_cached(str(path) if path else None)
