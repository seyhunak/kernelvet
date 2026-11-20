"""Optional Docker isolation for building and running migrated projects.

Migrated projects contain arbitrary user code and build scripts. With
``isolation = "docker"`` every build/test command runs in a throwaway container:
no network, all capabilities dropped, no privilege escalation, bounded
processes and memory, and only the project directory mounted. GPU device nodes
are passed through only for test execution.

AMD publishes its ROCm images for x86-64 only, and ROCm itself runs only on
x86-64 hosts. The container is therefore always started as ``linux/amd64``:
a no-op on an x86-64 host, and on an Arm host (Apple Silicon) it makes Docker
pull and emulate the x86-64 image instead of failing with "no matching
manifest".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CONTAINER_WORKDIR = "/work"


@dataclass
class SandboxConfig:
    image: str = "rocm/dev-ubuntu-24.04:latest"
    # ROCm images exist only for x86-64; None leaves the choice to Docker.
    platform: str | None = "linux/amd64"
    network: str = "none"
    memory: str = "8g"
    pids_limit: int = 1024


def docker_argv(
    project: Path,
    command: list[str],
    *,
    cfg: SandboxConfig,
    gpu: bool = False,
    workdir: str = CONTAINER_WORKDIR,
) -> list[str]:
    """Wrap ``command`` (relative to the project root) in ``docker run``."""
    argv = [
        "docker",
        "run",
        "--rm",
    ]
    if cfg.platform:
        argv.append(f"--platform={cfg.platform}")
    argv += [
        f"--network={cfg.network}",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        f"--pids-limit={cfg.pids_limit}",
        f"--memory={cfg.memory}",
        "-v",
        f"{project.resolve()}:{CONTAINER_WORKDIR}",
        "-w",
        workdir,
    ]
    if gpu:
        argv += ["--device=/dev/kfd", "--device=/dev/dri", "--group-add=video", "--group-add=render"]
    return [*argv, cfg.image, *command]
