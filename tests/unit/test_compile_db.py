"""compile_commands.json: loading, flag filtering, and use by convert, plan and the direct build."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import FakeRunner
from kernelvet.execution.compiler import merge_compile_flags, plan_direct_build
from kernelvet.migration.analyzer import analyze_project
from kernelvet.migration.compile_db import (
    find_compile_db,
    load_compile_db,
    portable_flags,
    rebase_flags,
)
from kernelvet.migration.hipify import TranslatorChoice, convert_project, translator_argv
from kernelvet.migration.planner import build_plan
from kernelvet.migration.scanner import scan_project
from kernelvet.utils.proc import CommandResult, HostContext

NVCC = "/usr/local/cuda/bin/nvcc"


def write_db(project: Path, entries: list[dict[str, object]], where: str = "build") -> Path:
    d = project / where
    d.mkdir(parents=True, exist_ok=True)
    path = d / "compile_commands.json"
    path.write_text(json.dumps(entries))
    return path


def entry(project: Path, file: str, *flags: str) -> dict[str, object]:
    return {
        "directory": str(project / "build"),
        "arguments": [NVCC, *flags, "-c", str(project / file), "-o", "x.o"],
        "file": str(project / file),
    }


# --------------------------------------------------------------------------- flag filtering


def test_portable_flags_keeps_meaning_and_drops_nvcc_only_flags(tmp_path: Path) -> None:
    args = [
        NVCC,
        "-I../include",
        "-I",
        "/opt/third_party/include",
        "--include-path=gen",
        "-isystem",
        "/usr/include/eigen3",
        "-DUSE_FAST=1",
        "-D",
        "TILE=32",
        "--define-macro=MODE=2",
        "-UNDEBUG",
        "--std",
        "c++17",
        "-arch=sm_80",
        "-gencode",
        "arch=compute_80,code=sm_80",
        "--expt-relaxed-constexpr",
        "-Xcompiler",
        "-fPIC",
        "-O3",
        "-x",
        "cu",
        "-c",
        "a.cu",
    ]
    build = tmp_path / "build"
    assert portable_flags(args, build) == [
        f"-I{(tmp_path / 'include').resolve()}",
        "-I/opt/third_party/include",
        f"-I{(build / 'gen').resolve()}",
        "-isystem",
        "/usr/include/eigen3",
        "-DUSE_FAST=1",
        "-DTILE=32",
        "-DMODE=2",
        "-UNDEBUG",
        "-std=c++17",
    ]


def test_load_compile_db_accepts_command_strings_and_skips_bad_entries(tmp_path: Path) -> None:
    path = tmp_path / "compile_commands.json"
    path.write_text(
        json.dumps(
            [
                {
                    "directory": str(tmp_path),
                    "command": f'{NVCC} -Iinc -DNAME="a b" -c src/a.cu',
                    "file": "src/a.cu",
                },
                {"directory": str(tmp_path), "file": "src/b.cu"},  # no command
                "not an object",
            ]
        )
    )
    db = load_compile_db(path)
    assert db.skipped == 2
    assert db.flags_for(tmp_path / "src/a.cu") == [f"-I{(tmp_path / 'inc').resolve()}", "-DNAME=a b"]
    assert db.flags_for(tmp_path / "src/b.cu") is None


def test_find_compile_db_looks_in_root_then_build_dirs(tmp_path: Path) -> None:
    assert find_compile_db(tmp_path) is None
    nested = write_db(tmp_path, [], where="cmake-build-release/Release")
    assert find_compile_db(tmp_path) == nested
    top = write_db(tmp_path, [], where=".")
    assert find_compile_db(tmp_path) == top


def test_rebase_flags_makes_project_paths_relative(tmp_path: Path) -> None:
    flags = [f"-I{tmp_path / 'include'}", "-I/opt/x", "-isystem", str(tmp_path / "third"), "-DA=1"]
    assert rebase_flags(flags, tmp_path) == ["-Iinclude", "-I/opt/x", "-isystem", "third", "-DA=1"]


# --------------------------------------------------------------------------- inventory and plan


def test_inventory_reports_coverage_and_errors(cuda_project: Path) -> None:
    write_db(cuda_project, [entry(cuda_project, "src/vector_add.cu", "-Iinclude")])
    db = scan_project(cuda_project).compile_db
    assert db is not None and db.path == "build/compile_commands.json" and db.entries == 1
    assert db.covered == ["src/vector_add.cu"]
    assert "src/reduce.cu" in db.uncovered

    (cuda_project / "build/compile_commands.json").write_text("{nope")
    broken = scan_project(cuda_project).compile_db
    assert broken is not None and broken.error and "not valid JSON" in broken.error


def test_explicit_compile_db_path_wins(cuda_project: Path, tmp_path: Path) -> None:
    elsewhere = write_db(tmp_path / "out-of-tree", [entry(cuda_project, "src/reduce.cu")], where=".")
    db = scan_project(cuda_project, elsewhere).compile_db
    assert db is not None and db.path == str(elsewhere.resolve()) and db.covered == ["src/reduce.cu"]


def test_plan_mentions_database_or_suggests_exporting_one(cuda_project: Path, tmp_path: Path) -> None:
    ctx = HostContext(runner=FakeRunner({"hipify-perl": "/x"}), root=tmp_path / "sys", env={}, system="Linux")
    (tmp_path / "sys").mkdir()
    titles = [s.title for s in build_plan(analyze_project(cuda_project), ctx).steps]
    assert "Optional: export compile_commands.json" in titles

    write_db(cuda_project, [entry(cuda_project, "src/vector_add.cu")])
    step = next(
        s for s in build_plan(analyze_project(cuda_project), ctx).steps if "compile_commands" in s.title
    )
    assert step.title == "Use compile_commands.json" and "src/reduce.cu" in step.files
    assert "hipify-perl does not use compiler flags" in step.detail


# --------------------------------------------------------------------------- hipify-clang


def test_translator_argv_puts_database_flags_first() -> None:
    choice = TranslatorChoice("hipify-clang", "", "/usr/local/cuda")
    argv = translator_argv(choice, Path("a.cu"), Path("o.cu"), ["-I."], ["-std=c++20", "-Iinc", "-DX=1"])
    assert argv[argv.index("--") + 1 :] == ["-std=c++20", "-Iinc", "-DX=1", "-I."]
    plain = translator_argv(choice, Path("a.cu"), Path("o.cu"), ["-I."])
    assert plain[plain.index("--") + 1 :] == ["-std=c++17", "-I."]


def clang_ctx(tmp_path: Path) -> tuple[HostContext, list[list[str]]]:
    sysroot = tmp_path / "sys"
    (sysroot / "usr/local/cuda/include").mkdir(parents=True)
    (sysroot / "usr/local/cuda/include/cuda_runtime.h").write_text("")
    r = FakeRunner({"hipify-clang": "/opt/rocm/bin/hipify-clang"})
    r.on("hipify-clang --version", "HIPIFY version 21.0.0")
    calls: list[list[str]] = []

    def run(argv: list[str], kw: dict[str, object]) -> CommandResult:
        calls.append(argv)
        out = Path(argv[argv.index("-o") + 1])
        out.write_text(Path(argv[1]).read_text().replace("cuda", "hip"))
        return CommandResult(argv, 0, stderr="CONVERTED refs count: 3\nUNCONVERTED refs count: 0")

    r.on(lambda a: a[0] == "hipify-clang" and "-o" in a, run)
    return HostContext(runner=r, root=sysroot, env={}, system="Linux"), calls


def test_convert_with_hipify_clang_uses_each_files_flags(cuda_project: Path, tmp_path: Path) -> None:
    write_db(
        cuda_project,
        [entry(cuda_project, "src/vector_add.cu", "-I../include", "-DTILE=64", "-std=c++20", "-arch=sm_80")],
    )
    ctx, calls = clang_ctx(tmp_path)
    out = tmp_path / "hip"
    res = convert_project(ctx, scan_project(cuda_project), out)

    by_file = {Path(c[1]).name: c[c.index("--") + 1 :] for c in calls}
    assert by_file["vector_add.cu"][:3] == [
        f"-I{(cuda_project / 'include').resolve()}",
        "-DTILE=64",
        "-std=c++20",
    ]
    assert not any("sm_80" in f for f in by_file["vector_add.cu"])
    assert by_file["reduce.cu"][0] == "-std=c++17"  # not in the database: guessed flags only

    flags_from = {f.source: f.flags_from for f in res.files if f.translator == "hipify-clang"}
    assert flags_from["src/vector_add.cu"] == "compile_commands.json"
    assert flags_from["src/reduce.cu"] == "guessed"
    assert res.compile_db and res.compile_db.endswith("compile_commands.json")

    saved = json.loads((out / ".kernelvet/compile_flags.json").read_text())
    assert saved["files"]["src/vector_add.cu"] == ["-Iinclude", "-DTILE=64", "-std=c++20"]
    assert "compile database:" in (out / ".kernelvet/conversion.log").read_text()


# --------------------------------------------------------------------------- direct hipcc build


def test_merge_compile_flags_dedupes_and_keeps_first_macro_value() -> None:
    table = {
        "main.hip": ["-std=c++20", "-Iinc", "-DMODE=1"],
        "lib.hip": ["-std=c++17", "-Iinc", "-Ithird", "-DMODE=2", "-isystem", "/opt/x"],
    }
    notes: list[str] = []
    assert merge_compile_flags(["main.hip", "lib.hip"], table, notes) == [
        "-std=c++20",
        "-Iinc",
        "-DMODE=1",
        "-Ithird",
        "-isystem",
        "/opt/x",
    ]
    assert notes and "kept -DMODE=1" in notes[0]


def test_direct_build_uses_saved_compile_flags(tmp_path: Path) -> None:
    proj = tmp_path / "hip"
    (proj / "src").mkdir(parents=True)
    (proj / "src/main.hip").write_text("#include <hip/hip_runtime.h>\nint main() { return 0; }\n")
    (proj / "src/lib.hip").write_text("#include <hip/hip_runtime.h>\nvoid f() {}\n")
    (proj / ".kernelvet").mkdir()
    (proj / ".kernelvet/compile_flags.json").write_text(
        json.dumps({"source": "x", "files": {"src/main.hip": ["-Igen", "-DTILE=64"], "src/other.cu": []}})
    )
    plan = plan_direct_build(proj, ["gfx942"])
    cmd = plan.commands[0]
    assert cmd[cmd.index("--offload-arch=gfx942") + 1 : cmd.index("--offload-arch=gfx942") + 3] == [
        "-Igen",
        "-DTILE=64",
    ]
    assert any("compile_commands.json for 1 of 2" in n for n in plan.notes)


def test_direct_build_without_saved_flags_is_unchanged(tmp_path: Path) -> None:
    proj = tmp_path / "hip"
    (proj / "src").mkdir(parents=True)
    (proj / "src/main.hip").write_text("int main() { return 0; }\n")
    plan = plan_direct_build(proj, ["gfx942"])
    assert not any("compile_commands" in n for n in plan.notes)
    assert "-DTILE=64" not in plan.commands[0]


def test_symlinked_project_path_still_matches(cuda_project: Path, tmp_path: Path) -> None:
    # On macOS /tmp is a symlink to /private/tmp; databases often record the unresolved path.
    link = tmp_path / "link"
    link.symlink_to(cuda_project)
    write_db(cuda_project, [entry(link, "src/vector_add.cu", "-I../include")])
    db = scan_project(link).compile_db
    assert db is not None and db.covered == ["src/vector_add.cu"]
