"""Wire and code contracts for live annotation protocols.

Two contracts live here:

* the **protocol code contract** a caller-supplied module must satisfy to be
  installed (``PROTOCOL`` marker, a ``Protocol`` class with ``on_event``);
* the **emission contract** that module speaks back to the container, and
  the **stream kinds** the container publishes on the annotation stream.

Everything the protocol emits is normalized and validated here before it is
persisted. A malformed emission is recorded as ``annotation.protocol.error``
and dropped; it never aborts the rollout it was observing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

PROTOCOL_SCHEMA = "synth.live-annotation-protocol.v1"
STREAM_SCHEMA = "synth.live-annotation-stream.v1"
REVISION_PREFIX = "anprev_"

# Semantic kinds published on the annotation stream, in the order a consumer
# should expect to meet them. ``stream.subscribed`` is control, not listed.
KIND_BOUND = "annotation.protocol.bound"
KIND_FINDING = "annotation.finding"
KIND_RETRACTED = "annotation.finding.retracted"
KIND_METRIC = "annotation.metric"
KIND_MODEL_REQUESTED = "annotation.model.requested"
KIND_MODEL_COMPLETED = "annotation.model.completed"
KIND_MODEL_FAILED = "annotation.model.failed"
KIND_PROTOCOL_ERROR = "annotation.protocol.error"
# Consumer -> annotator control, acknowledged durably on the stream so every
# consumer sees what every other consumer asked for, in order.
KIND_CONTROL_RECEIVED = "annotation.control.received"
KIND_CONTROL_REFUSED = "annotation.control.refused"
# A mid-rollout protocol hot-swap: new revision, optionally carrying the old
# protocol's snapshot state.
KIND_PROTOCOL_REBOUND = "annotation.protocol.rebound"
KIND_CLOSED = "annotation.closed"
KIND_HIGH_WATER = "capture.high_water"
KIND_CAPTURE_CLOSED = "capture.closed"

STREAM_KINDS: tuple[str, ...] = (
    KIND_BOUND,
    KIND_FINDING,
    KIND_RETRACTED,
    KIND_METRIC,
    KIND_MODEL_REQUESTED,
    KIND_MODEL_COMPLETED,
    KIND_MODEL_FAILED,
    KIND_PROTOCOL_ERROR,
    KIND_CONTROL_RECEIVED,
    KIND_CONTROL_REFUSED,
    KIND_PROTOCOL_REBOUND,
    KIND_CLOSED,
    KIND_HIGH_WATER,
    KIND_CAPTURE_CLOSED,
)

CONTROL_SCHEMA = "synth.live-annotation-control.v1"
# Control ops a consumer may send to a running annotator.
CONTROL_PROTOCOL_UPDATE = "protocol.update"
CONTROL_MESSAGE = "message"
CONTROL_STOP = "stop"
CONTROL_OPS = frozenset({CONTROL_PROTOCOL_UPDATE, CONTROL_MESSAGE, CONTROL_STOP})
_MESSAGE_MAX_BYTES = 16_384

FINDING_STATUS_PROVISIONAL = "provisional"

# Emission ops a protocol may return from on_event / on_model_result / on_close.
OP_FINDING = "finding"
OP_RETRACT = "retract"
OP_METRIC = "metric"
OP_MODEL_REQUEST = "model_request"
EMISSION_OPS = frozenset({OP_FINDING, OP_RETRACT, OP_METRIC, OP_MODEL_REQUEST})

_ID_MAX = 128
_LABEL_MAX = 256
_DETAIL_MAX_BYTES = 16_384
_INSTRUCTIONS_MAX_BYTES = 32_768
_CONTEXT_MAX_BYTES = 131_072


class EmissionError(ValueError):
    """A protocol emission violated the contract. Recorded, never raised past the runner."""


@dataclass(frozen=True)
class ProtocolRevision:
    """One immutable installed protocol. Identity is content-addressed."""

    revision_id: str
    protocol_id: str
    code: bytes
    code_sha256: str
    configuration: dict[str, Any]
    configuration_digest: str
    source_revision: str | None
    installed_at: str

    def public(self) -> dict[str, Any]:
        """Identity without source bytes or configuration values."""

        return {
            "protocol_revision_id": self.revision_id,
            "protocol_id": self.protocol_id,
            "code_sha256": self.code_sha256,
            "configuration_digest": self.configuration_digest,
            "source_revision": self.source_revision,
            "installed_at": self.installed_at,
        }


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def protocol_revision_id(
    *,
    code: bytes,
    protocol_id: str,
    configuration: dict[str, Any],
    source_revision: str | None,
) -> tuple[str, str, str]:
    """Return ``(revision_id, code_sha256, configuration_digest)``.

    The revision covers code bytes, declared protocol id, configuration and the
    caller's source revision, exactly like a policy revision. Re-installing
    identical inputs is idempotent.
    """

    code_sha256 = hashlib.sha256(code).hexdigest()
    configuration_digest = "sha256:" + hashlib.sha256(
        canonical_json(configuration).encode("utf-8")
    ).hexdigest()
    canonical = canonical_json(
        {
            "code_sha256": code_sha256,
            "protocol_id": protocol_id,
            "configuration": configuration,
            "source_revision": source_revision,
        }
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    return f"{REVISION_PREFIX}{digest[:16]}", code_sha256, configuration_digest


_SECRET_SUFFIXES = ("apikey", "secret", "token", "password")


def contains_secret(value: Any) -> bool:
    """Same refusal shape as ``PUT /policy``: identity and configuration, never credentials."""

    def secret_key(key: Any) -> bool:
        normalized = str(key).replace("_", "").replace("-", "").lower()
        return normalized == "credential" or normalized.endswith(_SECRET_SUFFIXES)

    if isinstance(value, dict):
        return any(secret_key(key) or contains_secret(child) for key, child in value.items())
    if isinstance(value, list):
        return any(contains_secret(child) for child in value)
    return False


def _require_str(row: dict[str, Any], key: str, *, max_len: int, what: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EmissionError(f"{what}.{key}_required")
    if len(value) > max_len:
        raise EmissionError(f"{what}.{key}_too_long")
    return value


def _optional_step(row: dict[str, Any], what: str) -> int | None:
    step = row.get("step")
    if step is None:
        return None
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise EmissionError(f"{what}.step_must_be_non_negative_integer")
    return step


def _optional_confidence(row: dict[str, Any], what: str) -> float | None:
    value = row.get("confidence")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EmissionError(f"{what}.confidence_must_be_number")
    if not 0.0 <= float(value) <= 1.0:
        raise EmissionError(f"{what}.confidence_out_of_range")
    return float(value)


def _bounded_object(value: Any, *, max_bytes: int, what: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise EmissionError(f"{what}_must_be_object")
    encoded = canonical_json(value)
    if len(encoded.encode("utf-8")) > max_bytes:
        raise EmissionError(f"{what}_too_large")
    if contains_secret(value):
        raise EmissionError(f"{what}_credential_forbidden")
    return json.loads(encoded)


def _evidence(row: dict[str, Any], *, source_stream_id: str, what: str) -> dict[str, Any]:
    raw = row.get("evidence")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise EmissionError(f"{what}.evidence_must_be_object")
    sequences = raw.get("sequences") or []
    if not isinstance(sequences, list) or len(sequences) > 256:
        raise EmissionError(f"{what}.evidence.sequences_invalid")
    clean: list[int] = []
    for item in sequences:
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise EmissionError(f"{what}.evidence.sequences_must_be_positive_integers")
        clean.append(item)
    return {
        "stream_id": source_stream_id,
        "sequences": sorted(set(clean)),
    }


@dataclass
class Emission:
    """A normalized emission. ``payload`` is what the runner persists (plus runner fields)."""

    op: str
    payload: dict[str, Any] = field(default_factory=dict)


def normalize_emission(raw: Any, *, source_stream_id: str) -> Emission:
    """Validate one emission dict. Raises ``EmissionError`` on any contract breach."""

    if not isinstance(raw, dict):
        raise EmissionError("emission_must_be_object")
    op = raw.get("op")
    if op not in EMISSION_OPS:
        raise EmissionError(f"emission.op_unknown:{op!r}")
    if op == OP_FINDING:
        finding_id = _require_str(raw, "finding_id", max_len=_ID_MAX, what="finding")
        kind = _require_str(raw, "kind", max_len=64, what="finding")
        label = _require_str(raw, "label", max_len=_LABEL_MAX, what="finding")
        supersedes = raw.get("supersedes")
        if supersedes is not None and (not isinstance(supersedes, str) or not supersedes):
            raise EmissionError("finding.supersedes_must_be_string")
        payload = {
            "finding_id": finding_id,
            "kind": kind,
            "label": label,
            "status": FINDING_STATUS_PROVISIONAL,
            "step": _optional_step(raw, "finding"),
            "confidence": _optional_confidence(raw, "finding"),
            "evidence": _evidence(raw, source_stream_id=source_stream_id, what="finding"),
            "supersedes": supersedes,
            "detail": _bounded_object(raw.get("detail"), max_bytes=_DETAIL_MAX_BYTES, what="finding.detail"),
        }
        return Emission(op=op, payload=payload)
    if op == OP_RETRACT:
        return Emission(
            op=op,
            payload={
                "finding_id": _require_str(raw, "finding_id", max_len=_ID_MAX, what="retract"),
                "reason": _require_str(raw, "reason", max_len=_LABEL_MAX, what="retract"),
            },
        )
    if op == OP_METRIC:
        name = _require_str(raw, "name", max_len=64, what="metric")
        value = raw.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EmissionError("metric.value_must_be_number")
        return Emission(
            op=op,
            payload={"name": name, "value": float(value), "step": _optional_step(raw, "metric")},
        )
    # OP_MODEL_REQUEST
    request_id = _require_str(raw, "request_id", max_len=_ID_MAX, what="model_request")
    instructions = raw.get("instructions")
    context = raw.get("context")
    if not isinstance(instructions, str) or not instructions.strip():
        raise EmissionError("model_request.instructions_required")
    if len(instructions.encode("utf-8")) > _INSTRUCTIONS_MAX_BYTES:
        raise EmissionError("model_request.instructions_too_large")
    if not isinstance(context, str):
        raise EmissionError("model_request.context_must_be_string")
    if len(context.encode("utf-8")) > _CONTEXT_MAX_BYTES:
        raise EmissionError("model_request.context_too_large")
    schema = raw.get("schema")
    if schema is not None and not isinstance(schema, dict):
        raise EmissionError("model_request.schema_must_be_object")
    max_output_tokens = raw.get("max_output_tokens")
    if max_output_tokens is not None and (
        isinstance(max_output_tokens, bool)
        or not isinstance(max_output_tokens, int)
        or max_output_tokens < 1
    ):
        raise EmissionError("model_request.max_output_tokens_invalid")
    return Emission(
        op=op,
        payload={
            "request_id": request_id,
            "instructions": instructions,
            "context": context,
            "schema": schema,
            "max_output_tokens": max_output_tokens,
        },
    )


def text_digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


class ControlError(ValueError):
    """A consumer control message violated the contract. Returned to the sender and recorded."""


@dataclass
class Control:
    op: str
    control_id: str
    payload: dict[str, Any] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        row = {"op": self.op, "control_id": self.control_id}
        if self.op == CONTROL_PROTOCOL_UPDATE:
            row["protocol_revision_id"] = self.payload["protocol_revision_id"]
            row["carry_state"] = self.payload["carry_state"]
        elif self.op == CONTROL_STOP:
            row["reason"] = self.payload.get("reason")
        elif self.op == CONTROL_MESSAGE:
            row["message_digest"] = text_digest(canonical_json(self.payload["message"]))
            row["message_kind"] = self.payload["message"].get("type")
        return row


def normalize_control(raw: Any, *, default_id: str) -> Control:
    """Validate one consumer control message. Raises ``ControlError``."""

    if not isinstance(raw, dict):
        raise ControlError("control_must_be_object")
    schema = raw.get("schema")
    if schema is not None and schema != CONTROL_SCHEMA:
        raise ControlError(f"control.schema_unknown:{schema!r}")
    op = raw.get("op")
    if op not in CONTROL_OPS:
        raise ControlError(f"control.op_unknown:{op!r}")
    control_id = raw.get("control_id")
    if control_id is None:
        control_id = default_id
    if not isinstance(control_id, str) or not control_id.strip() or len(control_id) > _ID_MAX:
        raise ControlError("control.control_id_invalid")
    if op == CONTROL_PROTOCOL_UPDATE:
        revision = raw.get("protocol_revision_id")
        if not isinstance(revision, str) or not revision.startswith(REVISION_PREFIX):
            raise ControlError("control.protocol_revision_id_required")
        carry = raw.get("carry_state", True)
        if not isinstance(carry, bool):
            raise ControlError("control.carry_state_must_be_boolean")
        return Control(op=op, control_id=control_id, payload={"protocol_revision_id": revision, "carry_state": carry})
    if op == CONTROL_STOP:
        reason = raw.get("reason")
        if reason is not None and (not isinstance(reason, str) or len(reason) > _LABEL_MAX):
            raise ControlError("control.reason_invalid")
        return Control(op=op, control_id=control_id, payload={"reason": reason})
    message = raw.get("message")
    if not isinstance(message, dict) or not message:
        raise ControlError("control.message_must_be_object")
    if not isinstance(message.get("type"), str) or not message["type"]:
        raise ControlError("control.message.type_required")
    encoded = canonical_json(message)
    if len(encoded.encode("utf-8")) > _MESSAGE_MAX_BYTES:
        raise ControlError("control.message_too_large")
    if contains_secret(message):
        raise ControlError("control.message_credential_forbidden")
    return Control(op=op, control_id=control_id, payload={"message": json.loads(encoded)})
