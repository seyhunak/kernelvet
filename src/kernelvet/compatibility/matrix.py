"""Compatibility reference data.

The bundled matrix (``data/rocm_matrix.json``) is a hand-transcribed snapshot of
AMD's published ROCm documentation. It records its source URLs, retrieval date
and known discrepancies, and can be replaced with ``--matrix path.json`` or the
``compatibility_matrix`` setting without changing any code.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

GpuClass = Literal["instinct", "radeon"]


class MatrixError(ValueError):
    """A compatibility matrix file that cannot be read, is not JSON, or does not match the schema."""


class MatrixSource(BaseModel):
    title: str
    url: str
    page_date: str | None = None
    describes: str | None = None


class GpuEntry(BaseModel):
    name: str
    gpu_class: GpuClass = Field(alias="class")
    arch: str
    gfx: str
    status: Literal["supported", "deprecated", "unsupported"]
    source: str
    note: str | None = None

    model_config = {"populate_by_name": True}


class OsEntry(BaseModel):
    id: str
    version: str
    label: str
    kernels: list[str] = Field(default_factory=list)
    source: str


class OsRestriction(BaseModel):
    os: str  # "debian:13"
    gfx: list[str]
    note: str
    source: str


class FrameworkEntry(BaseModel):
    versions: list[str]
    source: str


class DriverTable(BaseModel):
    instinct: list[str] = Field(default_factory=list)
    radeon: list[str] = Field(default_factory=list)
    source: str


class CompatibilityMatrix(BaseModel):
    schema_version: int = Field(alias="schema")
    rocm_version: str
    retrieved: str
    sources: dict[str, MatrixSource]
    notes: list[str] = Field(default_factory=list)
    release_versions: list[str] = Field(default_factory=list)
    gpus: list[GpuEntry]
    targets_uncertain: dict[str, str] = Field(default_factory=dict)
    os: dict[GpuClass, list[OsEntry]]
    os_gpu_restrictions: list[OsRestriction] = Field(default_factory=list)
    amdgpu_driver: DriverTable
    frameworks: dict[str, FrameworkEntry] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    def source_url(self, key: str) -> str:
        urls = [self.sources[k].url for k in key.split("+") if k in self.sources]
        return ", ".join(urls) if urls else key

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
