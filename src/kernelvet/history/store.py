"""Local diagnostic and migration history (SQLite, stdlib only).

Lives in ``$XDG_DATA_HOME/kernelvet/history.db``; full JSON reports are saved
next to it under ``reports/``. Disabled with ``history = false`` in the config.
"""

from __future__ import annotations

import functools
import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import ParamSpec, TypeVar

from pydantic import BaseModel

from kernelvet.config.settings import data_dir

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    summary TEXT NOT NULL,
    report_path TEXT
);
CREATE TABLE IF NOT EXISTS projects (
    path TEXT PRIMARY KEY,
    updated_at TEXT NOT NULL,
    stage TEXT NOT NULL,
    summary TEXT NOT NULL,
    report_path TEXT
);
"""


P = ParamSpec("P")
T = TypeVar("T")


class HistoryError(Exception):
    """The history database or report folder cannot be used (unreadable, corrupt, read-only, ...)."""


def _guarded(method: Callable[P, T]) -> Callable[P, T]:
    """Turn SQLite and filesystem errors into one HistoryError that says which file is at fault."""

    @functools.wraps(method)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return method(*args, **kwargs)
        except (sqlite3.Error, OSError) as exc:
            store = args[0]
            assert isinstance(store, HistoryStore)
            raise HistoryError(
                f"history in {store.root} is not usable ({exc}). "
                f"Move or delete {store.db_path.name} to start a new history, or set history = false."
            ) from exc

    return wrapper


@dataclass
class RunRecord:
    id: int
    kind: str
    created_at: str
    status: str
    summary: str
    report_path: str | None


@dataclass
class ProjectRecord:
    path: str
    updated_at: str
    stage: str
    summary: str
    report_path: str | None


class HistoryStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or data_dir()
        self.db_path = self.root / "history.db"
        self.reports_dir = self.root / "reports"

    def _connect(self) -> sqlite3.Connection:
        self.root.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.executescript(SCHEMA)
        return conn

    def save_report(self, kind: str, report: BaseModel) -> Path:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        path = self.reports_dir / f"{stamp}-{kind}.json"
        path.write_text(report.model_dump_json(indent=2))
        return path

    @_guarded
    def record_run(self, kind: str, status: str, summary: str, report: BaseModel | None = None) -> RunRecord:
        path = str(self.save_report(kind, report)) if report is not None else None
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                "INSERT INTO runs(kind, created_at, status, summary, report_path) VALUES (?,?,?,?,?)",
                (kind, now, status, summary, path),
            )
            rid = cur.lastrowid or 0
        return RunRecord(rid, kind, now, status, summary, path)

    @_guarded
    def record_project(self, path: str, stage: str, summary: str, report: BaseModel | None = None) -> None:
        rpath = str(self.save_report("migration", report)) if report is not None else None
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO projects(path, updated_at, stage, summary, report_path) VALUES (?,?,?,?,?) "
                "ON CONFLICT(path) DO UPDATE SET updated_at=excluded.updated_at, stage=excluded.stage, "
                "summary=excluded.summary, report_path=COALESCE(excluded.report_path, projects.report_path)",
                (path, now, stage, summary, rpath),
            )

    @_guarded
    def recent_runs(self, limit: int = 10, kind: str | None = None) -> list[RunRecord]:
        if not self.db_path.exists():
            return []
        q = "SELECT id, kind, created_at, status, summary, report_path FROM runs"
        args: tuple[object, ...] = ()
        if kind:
            q += " WHERE kind = ?"
            args = (kind,)
        q += " ORDER BY id DESC LIMIT ?"
        with closing(self._connect()) as conn:
            return [RunRecord(*row) for row in conn.execute(q, (*args, limit))]

    @_guarded
    def recent_projects(self, limit: int = 10) -> list[ProjectRecord]:
        if not self.db_path.exists():
            return []
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT path, updated_at, stage, summary, report_path FROM projects "
                "ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            )
            return [ProjectRecord(*row) for row in rows]

    def saved_reports(self) -> list[Path]:
        if not self.reports_dir.exists():
            return []
        return sorted(self.reports_dir.glob("*.json"), reverse=True)
