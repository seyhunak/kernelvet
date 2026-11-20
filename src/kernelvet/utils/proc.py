"""Safe subprocess execution and an injectable host context.

Every external command goes through :class:`CommandRunner`:

* arguments are always an argv list, never a shell string;
* every call has a timeout;
* the child environment is scrubbed of obvious secrets;
* a missing executable is reported, not raised.

:class:`HostContext` bundles the runner with the filesystem root and the
platform, so diagnostics can be tested against recorded command output without
touching the real machine.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from kernelvet.utils.redact import redact, scrub_env

DEFAULT_TIMEOUT = 30.0


@dataclass
class CommandResult:
    argv: list[str]
    returncode: int | None  # None when the command could not be started or timed out
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    timed_out: bool = False
    not_found: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def describe_failure(self) -> str:
        if self.not_found:
            return f"`{self.argv[0]}` is not installed or not on PATH"
        if self.timed_out:
            return f"`{' '.join(self.argv)}` timed out after {self.duration_s:.0f}s"
        if self.error:
            return f"`{self.argv[0]}` could not run: {self.error}"
        tail = (self.stderr or self.stdout).strip().splitlines()[-3:]
        return f"`{' '.join(self.argv)}` exited with {self.returncode}" + (
            ": " + " | ".join(tail) if tail else ""
        )


class CommandRunner:
    """Runs real commands. Subclass or replace for tests."""

    def which(self, name: str) -> str | None:
        return shutil.which(name)

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout: float = DEFAULT_TIMEOUT,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        input_text: str | None = None,
    ) -> CommandResult:
        args = [str(a) for a in argv]
        if not args:
            raise ValueError("empty argv")
        child_env = scrub_env(dict(os.environ if env is None else env))
        start = time.monotonic()
        try:
            proc = subprocess.run(
                args,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
                cwd=str(cwd) if cwd else None,
                env=child_env,
                input=input_text,
                check=False,
                shell=False,
            )
        except FileNotFoundError:
            return CommandResult(args, None, not_found=True, duration_s=time.monotonic() - start)
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                args,
                None,
                stdout=_to_text(exc.stdout),
                stderr=_to_text(exc.stderr),
                timed_out=True,
                duration_s=time.monotonic() - start,
            )
        except (PermissionError, OSError) as exc:
            return CommandResult(args, None, error=str(exc), duration_s=time.monotonic() - start)
        return CommandResult(
            args,
            proc.returncode,
            stdout=redact(proc.stdout),
            stderr=redact(proc.stderr),
            duration_s=time.monotonic() - start,
        )


def _to_text(data: bytes | str | None) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return redact(data.decode(errors="replace"))
    return redact(data)


@dataclass
class HostContext:
    """Everything diagnostics need to know about "the machine".

    ``root`` lets tests point file reads (``/etc/os-release``, ``/sys``,
    ``/opt/rocm``) at a fixture directory.
    """

    runner: CommandRunner = field(default_factory=CommandRunner)
    root: Path = Path("/")
    env: dict[str, str] = field(default_factory=lambda: dict(os.environ))
    system: str = field(default_factory=platform.system)  # "Linux", "Darwin", "Windows"
    machine: str = field(default_factory=platform.machine)
    release: str = field(default_factory=platform.release)
    timeout: float = DEFAULT_TIMEOUT

    def path(self, absolute: str) -> Path:
        """Map an absolute host path onto ``root``."""
        return self.root / absolute.lstrip("/")

    def read_text(self, absolute: str, limit: int = 1_000_000) -> str | None:
        p = self.path(absolute)
        try:
            with p.open("r", errors="replace") as fh:
                return fh.read(limit)
        except (OSError, ValueError):
            return None

    def exists(self, absolute: str) -> bool:
        try:
            return self.path(absolute).exists()
        except OSError:
            return False

    def listdir(self, absolute: str) -> list[str]:
        try:
            return sorted(os.listdir(self.path(absolute)))
        except OSError:
            return []

    def which(self, name: str) -> str | None:
        return self.runner.which(name)

    def run(self, argv: Sequence[str], **kw: object) -> CommandResult:
        kw.setdefault("timeout", self.timeout)
        return self.runner.run(argv, **kw)  # type: ignore[arg-type]

    def rocm_path(self) -> str:
        return self.env.get("ROCM_PATH") or self.env.get("HIP_PATH") or "/opt/rocm"


ProgressCallback = Callable[[str, float], None]
"""Called with (message, fraction 0..1). Used by CLI spinners and TUI workers."""


def noop_progress(_message: str, _fraction: float) -> None:
    return None
