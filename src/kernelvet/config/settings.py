"""User configuration (``~/.config/kernelvet/config.toml``).

Precedence: built-in defaults < config file < ``KERNELVET_*`` environment
variables < command-line flags.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError


def _xdg(var: str, fallback: str) -> Path:
    base = os.environ.get(var)
    return Path(base) if base else Path.home() / fallback


def config_dir() -> Path:
    return Path(os.environ.get("KERNELVET_CONFIG_DIR") or _xdg("XDG_CONFIG_HOME", ".config") / "kernelvet")


def data_dir() -> Path:
    return Path(os.environ.get("KERNELVET_DATA_DIR") or _xdg("XDG_DATA_HOME", ".local/share") / "kernelvet")


def config_path() -> Path:
    return config_dir() / "config.toml"


class Settings(BaseModel):
    command_timeout: float = Field(30.0, gt=0, description="Timeout for detection commands (s)")
    build_timeout: float = Field(900.0, gt=0, description="Timeout for builds (s)")
    test_timeout: float = Field(300.0, gt=0, description="Timeout per test executable (s)")
    runtime_checks: bool = Field(True, description="Compile and run the HIP probe during diagnose")
    torch_checks: bool = Field(True, description="Run PyTorch GPU tensor checks during diagnose")
    python_executable: str | None = Field(
        None, description="Python used for PyTorch checks (default: the current interpreter)"
    )
    hipify_preference: Literal["auto", "hipify-clang", "hipify-perl"] = "auto"
    isolation: Literal["none", "docker"] = Field(
        "none", description="Run migrated project builds/tests in Docker"
    )
    docker_image: str = Field("rocm/dev-ubuntu-24.04:latest", description="Image for isolated builds")
    offload_arch: str | None = Field(
        None, description="--offload-arch for hipcc (default: detected GPU targets)"
    )
    compatibility_matrix: str | None = Field(None, description="Path to a custom compatibility matrix JSON")
    history: bool = Field(True, description="Record diagnostic runs and migrations in SQLite")

    model_config = {"extra": "ignore"}


ENV_PREFIX = "KERNELVET_"


def _env_overrides() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in Settings.model_fields:
        raw = os.environ.get(ENV_PREFIX + name.upper())
        if raw is None:
            continue
        # pydantic coerces "true"/"0"/"30" to the field's type during validation
        out[name] = raw
    return out


class ConfigError(ValueError):
    pass


def load_settings(path: Path | None = None) -> Settings:
    path = path or config_path()
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = tomllib.loads(path.read_text())
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"could not read {path}: {exc}") from exc
    data.update(_env_overrides())
    try:
        return Settings(**data)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration in {path}: {exc}") from exc


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int | float):
        return repr(v)
    s = str(v).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def save_settings(settings: Settings, path: Path | None = None) -> Path:
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# KernelVet configuration", ""]
    for name, field in Settings.model_fields.items():
        value = getattr(settings, name)
        if field.description:
            lines.append(f"# {field.description}")
        if value is None:
            lines.append(f"# {name} =")
        else:
            lines.append(f"{name} = {_toml_value(value)}")
    path.write_text("\n".join(lines) + "\n")
    return path
