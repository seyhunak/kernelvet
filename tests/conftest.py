"""Shared test fixtures: an isolated data dir, a fake command runner and fake hosts."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from kernelvet.utils.proc import CommandResult, CommandRunner, HostContext

FIXTURES = Path(__file__).parent / "fixtures"
HOST = FIXTURES / "host" / "mi300x"
CUDA_PROJECT = FIXTURES / "cuda" / "vector_project"
HIP_PROJECT = FIXTURES / "hip" / "partial_migration"


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never touch the developer's real config or history."""
    base = tmp_path_factory.mktemp("kernelvet-home")
    monkeypatch.setenv("KERNELVET_DATA_DIR", str(base / "data"))
    monkeypatch.setenv("KERNELVET_CONFIG_DIR", str(base / "config"))
    for var in list(__import__("os").environ):
        if var.startswith("KERNELVET_") and var not in ("KERNELVET_DATA_DIR", "KERNELVET_CONFIG_DIR"):
            monkeypatch.delenv(var, raising=False)


Matcher = Callable[[list[str]], bool]
Responder = Callable[[list[str], dict], CommandResult]


class FakeRunner(CommandRunner):
    """Answers commands from a rule list; records every call; never spawns processes."""

    def __init__(self, tools: dict[str, str] | None = None) -> None:
        self.tools = dict(tools or {})
        self.rules: list[tuple[Matcher, Responder]] = []
        self.calls: list[list[str]] = []

    def which(self, name: str) -> str | None:
        return self.tools.get(name)

    def on(
        self, matcher: Matcher | str, response: Responder | str | CommandResult, rc: int = 0
    ) -> FakeRunner:
        m = matcher if callable(matcher) else (lambda argv, p=matcher: " ".join(argv).startswith(p))
        if isinstance(response, CommandResult):
            r: Responder = lambda argv, kw, res=response: res  # noqa: E731
        elif isinstance(response, str):
            r = lambda argv, kw, out=response, code=rc: CommandResult(argv, code, stdout=out)  # noqa: E731
        else:
            r = response
        self.rules.insert(0, (m, r))
        return self

    def run(self, argv: Sequence[str], **kw: object) -> CommandResult:
        args = [str(a) for a in argv]
        self.calls.append(args)
        for matcher, responder in self.rules:
            if matcher(args):
                return responder(args, kw)
        if args[0] not in self.tools and not args[0].startswith("/") and not args[0].startswith("./"):
            return CommandResult(args, None, not_found=True)
        return CommandResult(args, 1, stderr=f"fake: no rule for {' '.join(args)}")


PROBE_OK = "\n".join(
    [
        "STEP init OK no error",
        "STEP enumerate OK devices=1",
        "DEVICE 0 gfx942:sramecc+:xnack-|AMD Instinct MI300X|206141652992",
        "STEP alloc OK 4194304 bytes x3: no error",
        "STEP h2d OK no error",
        "STEP kernel OK no error",
        "STEP d2h OK no error",
        "STEP verify OK mismatches=0 of 1048576",
    ]
)

TORCH_ROCM = {
    "installed": True,
    "python": "3.12.3",
    "version": "2.15.0+rocm10.2",
    "hip_version": "7.17.26434",
    "cuda_version": None,
}
TORCH_RUNTIME_OK = {
    "steps": {
        "available": {"ok": True, "detail": "device_count=1"},
        "tensor_ops": {"ok": True, "detail": "512x512 matmul, max |diff| vs CPU = 1.2e-05"},
    },
    "devices": ["AMD Instinct MI300X"],
}

ALL_TOOLS = {t: f"/opt/rocm/bin/{t}" for t in ("rocminfo", "amd-smi", "hipcc", "hipconfig", "rocprofv3")}


def make_sysroot(
    root: Path,
    *,
    rocm_version: str = "10.2.0",
    driver: str | None = "31.60.0",
    gpu: bool = True,
    os_release: str | None = None,
) -> Path:
    (root / "etc").mkdir(parents=True, exist_ok=True)
    (root / "etc/os-release").write_text(os_release or (HOST / "os-release").read_text())
    if rocm_version:
        info = root / "opt/rocm/.info"
        info.mkdir(parents=True, exist_ok=True)
        (info / "version").write_text(rocm_version + "\n")
        lib = root / "opt/rocm/lib"
        lib.mkdir(parents=True, exist_ok=True)
        for so in (
            "libamdhip64.so.7",
            "librocblas.so.5",
            "libhipblas.so.3",
            "librccl.so.1",
            "libMIOpen.so.1",
        ):
            (lib / so).write_text("")
        (root / "opt/rocm/include/hipcub").mkdir(parents=True, exist_ok=True)
    if driver is not None:
        mod = root / "sys/module/amdgpu"
        mod.mkdir(parents=True, exist_ok=True)
        if driver:
            (mod / "version").write_text(driver + "\n")
    if gpu:
        dev = root / "sys/class/drm/card0/device"
        dev.mkdir(parents=True, exist_ok=True)
        (dev / "vendor").write_text("0x1002\n")
        (dev / "device").write_text("0x74a1\n")
        (dev / "mem_info_vram_total").write_text(str(196592 * 1024**2))
        (dev / "uevent").write_text("DRIVER=amdgpu\nPCI_SLOT_NAME=0000:0c:00.0\n")
        (root / "dev").mkdir(exist_ok=True)
        (root / "dev/kfd").write_text("")
    return root


def mi300x_runner(
    *, probe: str = PROBE_OK, torch: dict | None = None, torch_runtime: dict | None = None
) -> FakeRunner:
    r = FakeRunner(dict(ALL_TOOLS))
    r.on("amd-smi static --json", (HOST / "amd-smi-static.json").read_text())
    r.on("amd-smi version", "AMDSMI Tool: 26.0.0+abc | AMDSMI Library version: 26.0.0 | ROCm version: 10.2.0")
    r.on("rocminfo", (HOST / "rocminfo.txt").read_text())
    r.on("hipcc --version", (HOST / "hipcc-version.txt").read_text())
    r.on("hipconfig --version", "7.17.26434-0000000")
    r.on("rocprofv3 --version", "rocprofv3 1.0.0")

    def compile_probe(argv: list[str], kw: dict) -> CommandResult:
        Path(argv[argv.index("-o") + 1]).write_text("#!fake")
        return CommandResult(argv, 0)

    r.on(lambda a: a[0] == "hipcc" and "-o" in a, compile_probe)
    r.on(lambda a: a[0].endswith("/probe"), CommandResult(["probe"], 0, stdout=probe))
    t = torch if torch is not None else TORCH_ROCM
    r.on(
        lambda a: len(a) > 2 and a[1] == "-c" and 'out["hip_version"]' in a[2],
        lambda a, kw: CommandResult(a, 0, stdout=json.dumps(t)),
    )
    tr = torch_runtime if torch_runtime is not None else TORCH_RUNTIME_OK
    r.on(
        lambda a: len(a) > 2 and a[1] == "-c" and "tensor_ops" in a[2],
        lambda a, kw: CommandResult(a, 0, stdout=json.dumps(tr)),
    )
    return r


@pytest.fixture
def mi300x_ctx(tmp_path: Path) -> HostContext:
    root = make_sysroot(tmp_path / "root")
    return HostContext(
        runner=mi300x_runner(),
        root=root,
        env={"PATH": "/usr/bin"},
        system="Linux",
        machine="x86_64",
        release="6.8.0-48-generic",
        timeout=5,
    )


@pytest.fixture
def empty_linux_ctx(tmp_path: Path) -> HostContext:
    """A Linux box with no AMD GPU, no ROCm and no tools at all."""
    root = tmp_path / "root"
    (root / "etc").mkdir(parents=True)
    (root / "etc/os-release").write_text(
        'ID=ubuntu\nVERSION_ID="24.04"\nPRETTY_NAME="Ubuntu 24.04.1 LTS"\nNAME="Ubuntu"\n'
    )
    runner = FakeRunner({})
    runner.on(
        lambda a: len(a) > 2 and a[1] == "-c",
        lambda a, kw: CommandResult(a, 0, stdout=json.dumps({"installed": False, "python": "3.12"})),
    )
    return HostContext(
        runner=runner,
        root=root,
        env={},
        system="Linux",
        machine="x86_64",
        release="6.8.0-48-generic",
        timeout=5,
    )


@pytest.fixture
def cuda_project(tmp_path: Path) -> Path:
    dst = tmp_path / "cuda-project"
    shutil.copytree(CUDA_PROJECT, dst)
    return dst


@pytest.fixture
def hip_project(tmp_path: Path) -> Path:
    dst = tmp_path / "hip-project"
    shutil.copytree(HIP_PROJECT, dst)
    return dst


def tree_digest(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def fake_hipify(fail: set[str] | None = None) -> FakeRunner:
    """Stands in for hipify-perl: rewrites cuda* -> hip* and prints HIPIFY-style stats."""
    fail = fail or set()
    r = FakeRunner({"hipify-perl": "/opt/rocm/bin/hipify-perl"})
    r.on("hipify-perl --version", "HIPIFY version 21.0.0")

    def run(argv: list[str], kw: dict) -> CommandResult:
        src = Path(argv[-1])
        out = Path(next(a for a in argv if a.startswith("-o="))[3:])
        if src.name in fail:
            return CommandResult(argv, 1, stderr=f"error: could not parse {src.name}")
        text = src.read_text().replace("cuda", "hip").replace("cublas", "hipblas")
        out.write_text(text)
        return CommandResult(
            argv,
            0,
            stderr=f"[HIPIFY] info: file '{src}' statistics:\n  CONVERTED refs count: 7\n"
            "  UNCONVERTED refs count: 1\nwarning: 1 unconverted reference",
        )

    r.on(lambda a: a[0] == "hipify-perl" and any(x.startswith("-o=") for x in a), run)
    return r
