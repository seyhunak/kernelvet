"""Read a project's ``compile_commands.json`` (a JSON compilation database).

CMake (``-DCMAKE_EXPORT_COMPILE_COMMANDS=ON``), Bear and most build tools can
write one. It records the exact command used to compile each translation unit,
so it is the only reliable source of a file's include paths, macro definitions
and language standard. Guessing them from the directory layout misses
generated headers, third-party include trees and ``-D`` switches that select
code paths.

KernelVet uses it in two places:

* ``hipify-clang`` parses each file with that file's own flags.
* The flags are saved, rewritten relative to the migrated project, so the
  direct ``hipcc`` build compiles with the same include paths and defines.

Only flags that clang understands and that describe *what the code means* are
kept: ``-I``, ``-isystem``, ``-iquote``, ``-idirafter``, ``-D``, ``-U``,
``-std`` and ``-include``, with nvcc's long spellings (``--include-path``,
``--define-macro``, ...) normalised. Everything else (``-arch``, ``-gencode``,
``-Xcompiler``, ``--expt-relaxed-constexpr``, optimisation and warning flags) is
dropped: those flags are either nvcc-only, which clang would reject, or
irrelevant to translation. The database itself is never modified.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path

DB_NAME = "compile_commands.json"

# Build directories searched for a database, in order, below the project root.
_BUILD_DIR_GLOBS = ("build", "build-*", "build_*", "cmake-build-*", "out", "_build", ".build")

# Flags that take a path or value, as (spelling, kind). "path" values are made absolute.
_SEPARATE = {
    "-I": "path",
    "-isystem": "path",
    "-iquote": "path",
    "-idirafter": "path",
    "-include": "path",
    "-D": "value",
    "-U": "value",
    "--include-path": "path",
    "--system-include": "path",
    "--pre-include": "path",
    "--define-macro": "value",
    "--undefine-macro": "value",
    "--std": "value",
    "-std": "value",
}
# Canonical clang spelling of each accepted flag.
_CANONICAL = {
    "--include-path": "-I",
    "--system-include": "-isystem",
    "--pre-include": "-include",
    "--define-macro": "-D",
    "--undefine-macro": "-U",
    "--std": "-std",
}
_JOINED = ("-I", "-D", "-U")  # -Ifoo, -DFOO=1


class CompileDbError(ValueError):
    """The file exists but is not a usable compilation database."""


@dataclass
class CompileEntry:
    file: Path  # absolute
    directory: Path  # absolute
    flags: list[str] = field(default_factory=list)  # portable flags, paths absolute


@dataclass
class CompileDatabase:
    path: Path
    entries: dict[Path, CompileEntry] = field(default_factory=dict)  # keyed by absolute file
    skipped: int = 0  # entries without a usable file/command

    def flags_for(self, source: Path) -> list[str] | None:
        entry = self.entries.get(source.resolve())
        return list(entry.flags) if entry else None

    def covers(self, source: Path) -> bool:
        return source.resolve() in self.entries


def find_compile_db(root: Path) -> Path | None:
    """``compile_commands.json`` in the project root or a common build directory."""
    candidates = [root / DB_NAME]
    for pattern in _BUILD_DIR_GLOBS:
        for d in sorted(root.glob(pattern)):
            if d.is_dir():
                candidates.append(d / DB_NAME)
                candidates += sorted(d.glob(f"*/{DB_NAME}"))  # build/Release/...
    return next((c for c in candidates if c.is_file()), None)


def _arguments(entry: dict[str, object]) -> list[str]:
    args = entry.get("arguments")
    if isinstance(args, list) and all(isinstance(a, str) for a in args):
        return [str(a) for a in args]
    command = entry.get("command")
    if isinstance(command, str):
        try:
            return shlex.split(command)
        except ValueError:
            return []
    return []


def _absolute(value: str, directory: Path) -> str:
    p = Path(value)
    return str(p if p.is_absolute() else (directory / p).resolve())


def portable_flags(arguments: list[str], directory: Path) -> list[str]:
    """Keep the meaning-carrying flags clang accepts; resolve their paths against ``directory``."""
    out: list[str] = []
    i = 1  # arguments[0] is the compiler
    while i < len(arguments):
        arg = arguments[i]
        i += 1
        name, value = arg, None
        if "=" in arg and arg.split("=", 1)[0] in _SEPARATE and arg.startswith("--"):
            name, value = arg.split("=", 1)  # --include-path=dir, --std=c++17
        elif arg.startswith("-std="):
            name, value = "-std", arg[len("-std=") :]
        elif arg not in _SEPARATE and any(arg.startswith(j) and len(arg) > len(j) for j in _JOINED):
            name, value = arg[:2], arg[2:]  # -Idir, -DFOO=1
        if name not in _SEPARATE:
            continue
        if value is None:
            if i >= len(arguments):
                break
            value = arguments[i]
            i += 1
        canonical = _CANONICAL.get(name, name)
        if _SEPARATE[name] == "path":
            value = _absolute(value, directory)
        if canonical == "-std":
            out.append(f"-std={value}")
        elif canonical in _JOINED:
            out.append(f"{canonical}{value}")
        else:
            out += [canonical, value]
    return out


def load_compile_db(path: Path) -> CompileDatabase:
    try:
        data = json.loads(path.read_text(errors="replace"))
    except (OSError, ValueError) as exc:
        raise CompileDbError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(data, list):
        raise CompileDbError(f"{path}: expected a JSON array of compile commands")
    db = CompileDatabase(path=path.resolve())
    for raw in data:
        if not isinstance(raw, dict) or not isinstance(raw.get("file"), str):
            db.skipped += 1
            continue
        directory = Path(str(raw.get("directory") or path.parent))
        if not directory.is_absolute():
            directory = (path.parent / directory).resolve()
        args = _arguments(raw)
        if not args:
            db.skipped += 1
            continue
        # Resolved, so symlinked paths (e.g. macOS /tmp -> /private/tmp) match the scanned tree.
        file = Path(_absolute(str(raw["file"]), directory)).resolve()
        db.entries[file] = CompileEntry(file=file, directory=directory, flags=portable_flags(args, directory))
    return db


def rebase_flags(flags: list[str], source_root: Path) -> list[str]:
    """Rewrite absolute paths under ``source_root`` relative to the project root.

    Relative paths work both on the host and inside the Docker sandbox, where the
    project is mounted at a different location. Paths outside the project are
    kept absolute.
    """
    src = source_root.resolve()
    out: list[str] = []
    expect_path = False
    for flag in flags:
        if expect_path:
            out.append(_relative(flag, src))
            expect_path = False
            continue
        if flag in ("-isystem", "-iquote", "-idirafter", "-include"):
            out.append(flag)
            expect_path = True
        elif flag.startswith("-I"):
            out.append("-I" + _relative(flag[2:], src))
        else:
            out.append(flag)
    return out


def _relative(value: str, root: Path) -> str:
    p = Path(value)
    try:
        rel = p.resolve().relative_to(root)
    except ValueError:
        return value
    return rel.as_posix() or "."
