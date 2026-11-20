"""Filesystem path validation for migration inputs and outputs."""

from __future__ import annotations

from pathlib import Path


class PathError(ValueError):
    """Raised for an unusable or unsafe path. The message is user-facing."""


def existing_dir(path: str | Path, what: str = "project") -> Path:
    p = Path(path).expanduser()
    if not p.exists():
        raise PathError(f"{what} directory does not exist: {p}")
    if not p.is_dir():
        raise PathError(f"{what} path is not a directory: {p}")
    return p.resolve()


def safe_output_dir(source: Path, output: str | Path, *, allow_existing: bool = False) -> Path:
    """Validate a migration output directory.

    The output must never be the source tree, inside it, or contain it: original
    CUDA projects are never modified.
    """
    src = source.resolve()
    out = Path(output).expanduser().resolve()
    if out == src:
        raise PathError("output directory must differ from the source project")
    if out.is_relative_to(src):
        raise PathError(f"output directory {out} is inside the source project {src}")
    if src.is_relative_to(out):
        raise PathError(f"output directory {out} contains the source project {src}")
    if out.exists():
        if not out.is_dir():
            raise PathError(f"output path exists and is not a directory: {out}")
        if any(out.iterdir()) and not allow_existing:
            raise PathError(f"output directory {out} is not empty; pass --force to write into it")
    return out


def within(root: Path, candidate: Path) -> bool:
    """True if ``candidate`` resolves inside ``root`` (guards against symlink escapes)."""
    try:
        return candidate.resolve().is_relative_to(root.resolve())
    except OSError:
        return False
