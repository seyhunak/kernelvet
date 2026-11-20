"""Lenient version parsing for tool output."""

from __future__ import annotations

import re

_VERSION = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")


def parse_version(text: str | None) -> tuple[int, ...] | None:
    """Extract the first dotted version from ``text``: "6.2.41134-65d174c3e" -> (6, 2, 41134)."""
    if not text:
        return None
    m = _VERSION.search(text)
    if not m:
        return None
    return tuple(int(g) for g in m.groups() if g is not None)


def major_minor(text: str | None) -> str | None:
    v = parse_version(text)
    return None if v is None else f"{v[0]}.{v[1]}"


def version_str(text: str | None) -> str | None:
    v = parse_version(text)
    return None if v is None else ".".join(str(p) for p in v)


def same_release(a: str | None, b: str | None, depth: int = 2) -> bool:
    va, vb = parse_version(a), parse_version(b)
    if va is None or vb is None:
        return False
    return va[:depth] == vb[:depth]
