"""Trace identity, provider-native aliases, and process-to-process trace context."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from synth_containers.serde import JsonDataclassMixin

from ..canonical import record_id


TRACE_SCHEMA_VERSION = "synth.trace.v5"
EVIDENCE_BUNDLE_SCHEMA_VERSION = "synth.trace-evidence-bundle.v5"
BUNDLE_SCHEMA_VERSION = "synth.trace-bundle.v1"

ENV_TRACE_ID = "SYNTH_TRACE_ID"
ENV_CAPTURE_ID = "SYNTH_CAPTURE_ID"
ENV_ACTOR_ID = "SYNTH_ACTOR_ID"
ENV_ACTOR_SESSION_ID = "SYNTH_ACTOR_SESSION_ID"
ENV_PARENT_ACTOR_ID = "SYNTH_PARENT_ACTOR_ID"
ENV_PARENT_ACTOR_SESSION_ID = "SYNTH_PARENT_ACTOR_SESSION_ID"
ENV_PARENT_SPAN_ID = "SYNTH_PARENT_SPAN_ID"
ENV_DELEGATION_ID = "SYNTH_DELEGATION_ID"
ENV_WORKFLOW_ADDRESS = "SYNTH_WORKFLOW_ADDRESS"
ENV_BINDING_PATH = "SYNTH_TRACE_BINDING_PATH"
ENV_COLLECTOR_URL = "SYNTH_TRACE_COLLECTOR_URL"
ENV_OUTPUT_DIR = "SYNTH_TRACE_OUTPUT_DIR"


class TraceKind(StrEnum):
    AGENT_ROLLOUT = "agent_rollout"
    EVALUATION_ATTEMPT = "evaluation_attempt"
    WORKFLOW_RUN = "workflow_run"
    MODEL_CALL_ONLY = "model_call_only"


class AliasNamespace(StrEnum):
    """Namespaces for provider/native identities preserved beside canonical IDs."""

    PROVIDER_REQUEST = "provider.request"
    PROVIDER_RESPONSE = "provider.response"
    CODEX_THREAD = "codex.thread"
    CODEX_TURN = "codex.turn"
    CODEX_ITEM = "codex.item"
    REB_SESSION = "reb.session"
    REB_CALL = "reb.call"
    REB_CANDIDATE = "reb.candidate"
    REB_TASK = "reb.task"
    GAMEBENCH_EPISODE = "gamebench.episode"
    GAMEBENCH_STEP = "gamebench.step"
    EXPERIMENTS_TRACE_V4 = "experiments.trace.v4"
    CONTAINERS_TRACE_V4 = "containers.rollout_trace.v4"
    CORRELATION = "synth.correlation"


@dataclass(frozen=True, slots=True)
class TraceIdentityV5(JsonDataclassMixin):
    """Platform identities a trace may carry. Only ``trace_id`` is authoritative."""

    rollout_id: str | None = None
    run_id: str | None = None
    trial_id: str | None = None
    episode_id: str | None = None
    correlation_id: str | None = None
    task_id: str | None = None
    task_instance_id: str | None = None
    benchmark: str | None = None
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class AliasV1(JsonDataclassMixin):
    """One provider/native identity attached to a canonical entity."""

    namespace: AliasNamespace | str
    value: str
    target_id: str
    target_kind: str = "trace"
    provenance: str = "observed"
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class TraceContextV1(JsonDataclassMixin):
    """Correlation context propagated across process and container boundaries.

    W3C ``traceparent`` is derived from the trace/span identity so external tools
    can join on it; the Synth fields remain authoritative.
    """

    trace_id: str
    capture_id: str
    actor_id: str
    actor_session_id: str
    parent_actor_id: str | None = None
    parent_actor_session_id: str | None = None
    parent_span_id: str | None = None
    delegation_id: str | None = None
    workflow_address: str | None = None
    binding_path: str | None = None
    collector_url: str | None = None
    output_dir: str | None = None
    w3c_traceparent: str | None = None

    def to_environment(self) -> dict[str, str]:
        values = {
            ENV_TRACE_ID: self.trace_id,
            ENV_CAPTURE_ID: self.capture_id,
            ENV_ACTOR_ID: self.actor_id,
            ENV_ACTOR_SESSION_ID: self.actor_session_id,
            ENV_PARENT_ACTOR_ID: self.parent_actor_id or "",
            ENV_PARENT_ACTOR_SESSION_ID: self.parent_actor_session_id or "",
            ENV_PARENT_SPAN_ID: self.parent_span_id or "",
            ENV_DELEGATION_ID: self.delegation_id or "",
            ENV_WORKFLOW_ADDRESS: self.workflow_address or "",
            ENV_BINDING_PATH: self.binding_path or "",
            ENV_COLLECTOR_URL: self.collector_url or "",
            ENV_OUTPUT_DIR: self.output_dir or "",
            "TRACEPARENT": self.w3c_traceparent or self.traceparent(),
        }
        return {key: value for key, value in values.items() if value}

    def traceparent(self) -> str:
        trace_hex = _hex_suffix(self.trace_id, 32)
        span_hex = _hex_suffix(self.parent_span_id or self.capture_id, 16)
        return f"00-{trace_hex}-{span_hex}-01"

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "TraceContextV1 | None":
        source = environ if environ is not None else os.environ
        trace_id = str(source.get(ENV_TRACE_ID) or "").strip()
        capture_id = str(source.get(ENV_CAPTURE_ID) or "").strip()
        actor_id = str(source.get(ENV_ACTOR_ID) or "").strip()
        session_id = str(source.get(ENV_ACTOR_SESSION_ID) or "").strip()
        if not (trace_id and capture_id and actor_id and session_id):
            return None
        return cls(
            trace_id=trace_id,
            capture_id=capture_id,
            actor_id=actor_id,
            actor_session_id=session_id,
            parent_actor_id=str(source.get(ENV_PARENT_ACTOR_ID) or "") or None,
            parent_actor_session_id=str(
                source.get(ENV_PARENT_ACTOR_SESSION_ID) or ""
            )
            or None,
            parent_span_id=str(source.get(ENV_PARENT_SPAN_ID) or "") or None,
            delegation_id=str(source.get(ENV_DELEGATION_ID) or "") or None,
            workflow_address=str(source.get(ENV_WORKFLOW_ADDRESS) or "") or None,
            binding_path=str(source.get(ENV_BINDING_PATH) or "") or None,
            collector_url=str(source.get(ENV_COLLECTOR_URL) or "") or None,
            output_dir=str(source.get(ENV_OUTPUT_DIR) or "") or None,
            w3c_traceparent=_bounded_traceparent(source.get("TRACEPARENT")),
        )


@dataclass(frozen=True, slots=True)
class TraceProvenanceV5(JsonDataclassMixin):
    """Who produced the trace and against which code, config, and images."""

    producer: str
    producer_version: str
    source_format: str = "synth.capture.raw.v1"
    producer_commit: str | None = None
    container_image_digest: str | None = None
    runtime_version: str | None = None
    model: str | None = None
    provider: str | None = None
    harness: str | None = None
    prompt_digest: str | None = None
    config_digest: str | None = None
    execution_host: str | None = None
    captured_at: str | None = None
    transformation_chain: tuple[str, ...] = ()
    aliases: tuple[AliasV1, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


def _hex_suffix(value: str, length: int) -> str:
    hexed = "".join(char for char in value.lower() if char in "0123456789abcdef")
    if len(hexed) >= length:
        return hexed[-length:]
    return hexed.rjust(length, "0")


def _bounded_traceparent(value: str | None) -> str | None:
    if value is None:
        return None
    candidate = str(value).strip()
    if len(candidate) != 55:
        return None
    pieces = candidate.split("-")
    if (
        len(pieces) != 4
        or [len(piece) for piece in pieces] != [2, 32, 16, 2]
        or pieces[0].lower() == "ff"
        or pieces[1] == "0" * 32
        or pieces[2] == "0" * 16
        or any(
        any(char not in "0123456789abcdefABCDEF" for char in piece)
        for piece in pieces
        )
    ):
        return None
    return candidate.lower()


def mint_trace_id(*, kind: str, key: Any) -> str:
    return record_id("trace", kind=kind, key=key)


def mint_capture_id(*, trace_id: str, key: Any) -> str:
    return record_id("cap", kind="capture_session", scope=(trace_id,), key=key)


def mint_actor_id(*, trace_id: str, name: str) -> str:
    return record_id("actor", kind="actor", scope=(trace_id,), key=name)


def mint_session_id(
    *,
    trace_id: str,
    actor_id: str,
    attempt: int = 0,
    nonce: str | None = None,
) -> str:
    return record_id(
        "sess",
        kind="session",
        scope=(trace_id, actor_id),
        key={"attempt": attempt, "nonce": nonce} if nonce else attempt,
    )


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "ENV_ACTOR_ID",
    "ENV_ACTOR_SESSION_ID",
    "ENV_BINDING_PATH",
    "ENV_CAPTURE_ID",
    "ENV_COLLECTOR_URL",
    "ENV_OUTPUT_DIR",
    "ENV_PARENT_ACTOR_ID",
    "ENV_PARENT_ACTOR_SESSION_ID",
    "ENV_PARENT_SPAN_ID",
    "ENV_DELEGATION_ID",
    "ENV_TRACE_ID",
    "ENV_WORKFLOW_ADDRESS",
    "EVIDENCE_BUNDLE_SCHEMA_VERSION",
    "TRACE_SCHEMA_VERSION",
    "AliasNamespace",
    "AliasV1",
    "TraceContextV1",
    "TraceIdentityV5",
    "TraceKind",
    "TraceProvenanceV5",
    "mint_actor_id",
    "mint_capture_id",
    "mint_session_id",
    "mint_trace_id",
]
