from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .serde import JsonDataclassMixin


class ResourceKind(StrEnum):
    DATA = "data"
    CODE = "code"
    RUNTIME = "runtime"
    STATE = "state"
    EVALUATION = "evaluation"
    TOOLING = "tooling"
    SECRET = "secret"
    CONFIG = "config"
    ARTIFACT = "artifact"
    OTHER = "other"

    @classmethod
    def parse(cls, value: Any) -> "ResourceKind":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower().replace("-", "_")
        aliases = {
            "dataset": cls.DATA,
            "dataset_split": cls.DATA,
            "world_bundle": cls.DATA,
            "dockerfile": cls.CODE,
            "build_context": cls.CODE,
            "container_image": cls.RUNTIME,
            "workspace": cls.STATE,
            "checkpoint": cls.STATE,
            "snapshot": cls.STATE,
            "rubric": cls.EVALUATION,
            "verifier": cls.EVALUATION,
            "tool_registry": cls.TOOLING,
            "secret_ref": cls.SECRET,
        }
        if text in aliases:
            return aliases[text]
        try:
            return cls(text)
        except ValueError:
            return cls.OTHER


@dataclass(slots=True)
class ResourceRef(JsonDataclassMixin):
    resource_id: str
    kind: ResourceKind | str = ResourceKind.OTHER
    name: str = ""
    subtype: str = ""
    uri: str = ""
    path: str = ""
    digest: str = ""
    version: str = ""
    media_type: str = ""
    size_bytes: int | None = None
    required: bool = True
    labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.kind = ResourceKind.parse(self.kind)

    def normalized_kind(self) -> ResourceKind:
        return ResourceKind.parse(self.kind)

    def to_dict(self) -> dict[str, Any]:
        payload = JsonDataclassMixin.to_dict(self)
        payload["kind"] = self.normalized_kind().value
        return payload
