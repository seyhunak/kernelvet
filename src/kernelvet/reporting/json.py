"""JSON export. Output is the Pydantic model dump: lossless and schema-checked."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel


def to_json(report: BaseModel, *, indent: int | None = 2) -> str:
    return report.model_dump_json(indent=indent)


def json_schema(model: type[BaseModel]) -> dict[str, Any]:
    return model.model_json_schema()


def dumps_schema(model: type[BaseModel]) -> str:
    return json.dumps(json_schema(model), indent=2)
