from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .serde import JsonObject, JsonValue


def _json_object(payload: Mapping[str, JsonValue], key: str) -> JsonObject:
    if key not in payload:
        return {}
    value = payload[key]
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be an object")
    return dict(value)


def _string_list(payload: Mapping[str, JsonValue], key: str) -> list[str]:
    if key not in payload:
        return []
    value = payload[key]
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list")
    return [str(item) for item in value]


def _optional_text(payload: Mapping[str, JsonValue], key: str) -> str | None:
    if key not in payload or payload[key] is None:
        return None
    text = str(payload[key]).strip()
    return text or None


@dataclass(frozen=True, slots=True)
class PauseRequest:
    reason: str = "paused"
    metadata: JsonObject = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, JsonValue]) -> "PauseRequest":
        return cls(
            reason=_optional_text(payload, "reason") or "paused",
            metadata=_json_object(payload, "metadata"),
        )


@dataclass(frozen=True, slots=True)
class TerminateRequest:
    reason: str = "terminated"

    @classmethod
    def from_payload(cls, payload: Mapping[str, JsonValue]) -> "TerminateRequest":
        return cls(reason=_optional_text(payload, "reason") or "terminated")


@dataclass(frozen=True, slots=True)
class RuntimeCheckpointRequest:
    checkpoint_id: str | None = None
    checkpoint_uri: str | None = None
    checkpoint_version: str = "v1"
    label: str | None = None
    labels: list[str] = field(default_factory=list)
    source: str | None = None
    actor_ids: list[str] = field(default_factory=list)
    metadata: JsonObject = field(default_factory=dict)
    annotations: JsonObject = field(default_factory=dict)
    artifact_refs: list[JsonObject] = field(default_factory=list)
    restore_eligible: bool = True

    @classmethod
    def from_payload(cls, payload: Mapping[str, JsonValue]) -> "RuntimeCheckpointRequest":
        artifact_refs = payload["artifact_refs"] if "artifact_refs" in payload else []
        if not isinstance(artifact_refs, list):
            raise TypeError("artifact_refs must be a list")
        for item in artifact_refs:
            if not isinstance(item, dict):
                raise TypeError("artifact_refs entries must be objects")
        return cls(
            checkpoint_id=_optional_text(payload, "checkpoint_id"),
            checkpoint_uri=_optional_text(payload, "checkpoint_uri"),
            checkpoint_version=_optional_text(payload, "checkpoint_version") or "v1",
            label=_optional_text(payload, "label"),
            labels=_string_list(payload, "labels"),
            source=_optional_text(payload, "source"),
            actor_ids=_string_list(payload, "actor_ids"),
            metadata=_json_object(payload, "metadata"),
            annotations=_json_object(payload, "annotations"),
            artifact_refs=[dict(item) for item in artifact_refs],
            restore_eligible=bool(payload["restore_eligible"]) if "restore_eligible" in payload else True,
        )


@dataclass(frozen=True, slots=True)
class CheckpointLabelsRequest:
    labels: list[str] = field(default_factory=list)
    annotations: JsonObject = field(default_factory=dict)
    metadata: JsonObject = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, JsonValue]) -> "CheckpointLabelsRequest":
        return cls(
            labels=_string_list(payload, "labels"),
            annotations=_json_object(payload, "annotations"),
            metadata=_json_object(payload, "metadata"),
        )


@dataclass(frozen=True, slots=True)
class RuntimeResumeRequest:
    checkpoint_id: str | None = None
    target_rollout_id: str | None = None
    mode: str | None = None
    submission_mode: str | None = None
    overrides: JsonObject = field(default_factory=dict)
    branch_metadata: JsonObject = field(default_factory=dict)
    checkpoint_data_base64: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, JsonValue]) -> "RuntimeResumeRequest":
        return cls(
            checkpoint_id=_optional_text(payload, "checkpoint_id"),
            target_rollout_id=_optional_text(payload, "target_rollout_id"),
            mode=_optional_text(payload, "mode"),
            submission_mode=_optional_text(payload, "submission_mode"),
            overrides=_json_object(payload, "overrides"),
            branch_metadata=_json_object(payload, "branch_metadata"),
            checkpoint_data_base64=_optional_text(payload, "checkpoint_data_base64"),
        )
