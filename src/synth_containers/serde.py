from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel


JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | dict[str, "JsonValue"] | list["JsonValue"]
JsonObject = dict[str, JsonValue]


def jsonable(value: Any) -> JsonValue:
    """Recursively convert common Python/container contract objects into JSON-safe data."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return jsonable(value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseModel):
        return jsonable(value.model_dump(mode="python", exclude_none=False))
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    return str(value)


def canonical_json_text(value: Any, *, indent: int = 2, sort_keys: bool = True) -> str:
    return json.dumps(jsonable(value), indent=indent, sort_keys=sort_keys)


class JsonDataclassMixin:
    """Tiny mixin for the contract dataclasses used across the package."""

    def to_dict(self) -> dict[str, Any]:
        payload = jsonable(self)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected a dict payload for {type(self).__name__}, got {type(payload).__name__}")
        return payload

    def to_json(self, *, indent: int = 2, sort_keys: bool = True) -> str:
        return canonical_json_text(self, indent=indent, sort_keys=sort_keys)
