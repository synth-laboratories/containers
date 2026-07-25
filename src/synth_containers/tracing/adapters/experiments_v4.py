"""Import a hand-authored ``experiments.trace.v4`` record into a sealed V5 document.

The Experiments prototype wrote this shape before native capture existed. Importing it
makes those records citable without turning them into a second authority: the result
declares imported capture, no raw provider evidence, and harness-reported usage.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..canonical import bytes_digest, canonical_bytes, record_id, utc_now
from ..models.actors import (
    ActorKind,
    ActorV5,
    CoverageState,
    SessionCoverageV5,
    SessionV5,
)
from ..models.completeness import (
    CaptureStatus,
    TraceCompletenessV5,
    TraceLifecycleV5,
    TraceStatus,
)
from ..models.document import TraceCaptureSummaryV5, TraceDocumentV5
from ..models.events import EventOrderV1, EventType, EventV5
from ..models.identity import (
    AliasNamespace,
    AliasV1,
    TraceIdentityV5,
    TraceKind,
    TraceProvenanceV5,
)
from ..models.spans import SpanKind, SpanV5, UsageProvenance, UsageV5


IMPORTER_NAME = "experiments_trace_v4"
IMPORTER_VERSION = "1"


def import_experiments_trace_v4(
    payload: Mapping[str, Any],
    *,
    producer: str = "synth_containers.tracing.adapters.experiments_v4",
    imported_at: str = "1970-01-01T00:00:00Z",
) -> TraceDocumentV5:
    """Build a sealed V5 document from an ``experiments.trace.v4`` payload."""

    source_digest = bytes_digest(canonical_bytes(payload))
    correlation = str(payload.get("trace_id") or "")
    trace_id = record_id("trace", kind="imported_experiments_v4", key=source_digest)
    agent_actor_id = record_id("actor", kind="actor", scope=(trace_id,), key="policy")
    env_actor_id = record_id("actor", kind="actor", scope=(trace_id,), key="environment")
    session_id = record_id("sess", kind="session", scope=(trace_id, agent_actor_id), key=0)
    env_session_id = record_id("sess", kind="session", scope=(trace_id, env_actor_id), key=0)

    interaction = _mapping(payload.get("interaction"))
    environment = _mapping(payload.get("environment"))
    operations = _mapping(payload.get("operations"))
    timestamps = _mapping(payload.get("timestamps"))

    spans: list[SpanV5] = []
    events: list[EventV5] = []
    sequence = 0
    started_at = str(timestamps.get("started_at") or imported_at)
    ended_at = str(timestamps.get("completed_at") or started_at)

    turns = [
        item for item in list(interaction.get("react_turns") or []) if isinstance(item, Mapping)
    ]
    for index, turn in enumerate(turns):
        sequence += 1
        span_id = record_id("span", kind="agent_turn", scope=(trace_id,), key=index)
        spans.append(
            SpanV5(
                span_id=span_id,
                span_kind=SpanKind.AGENT_TURN,
                actor_id=agent_actor_id,
                session_id=session_id,
                started_at=started_at,
                detail={
                    "turn_index": index,
                    "llm_call": turn.get("llm_call"),
                    "batch_index": turn.get("batch_index"),
                    "action": turn.get("action"),
                    "invalid_parse": turn.get("invalid_parse"),
                },
            ).sealed()
        )
        events.append(
            EventV5(
                event_id=record_id("evt", kind="turn", scope=(trace_id,), key=index),
                event_type=EventType.ENV_ACTION_EXECUTED,
                actor_id=env_actor_id,
                session_id=env_session_id,
                occurred_at=started_at,
                span_id=span_id,
                order=EventOrderV1(chronological_sequence=sequence),
                payload={
                    "action": turn.get("action"),
                    "ply": turn.get("ply"),
                    "achievements": list(turn.get("achievements") or []),
                },
            ).sealed()
        )

    for index, event in enumerate(
        [item for item in list(interaction.get("nev") or []) if isinstance(item, Mapping)]
    ):
        sequence += 1
        events.append(
            EventV5(
                event_id=record_id("evt", kind="nev", scope=(trace_id,), key=index),
                event_type=EventType.ENV_TRANSITION,
                actor_id=env_actor_id,
                session_id=env_session_id,
                occurred_at=started_at,
                order=EventOrderV1(chronological_sequence=sequence),
                payload=dict(event),
            ).sealed()
        )

    usage = UsageV5(
        provenance=UsageProvenance.OBSERVED_HARNESS,
        prompt_tokens=_int(operations.get("input_tokens")),
        completion_tokens=_int(operations.get("completion_tokens")),
        requests=_int(operations.get("llm_calls")),
        source_refs=("experiments.trace.v4:operations",),
    )

    agent = ActorV5(
        actor_id=agent_actor_id,
        kind=ActorKind.AGENT,
        display_name="craftax react policy",
        role="policy",
        model=str((payload.get("policy") or {}).get("base_model") or "") or None,
    ).sealed()
    env = ActorV5(
        actor_id=env_actor_id,
        kind=ActorKind.ENVIRONMENT,
        display_name=str(environment.get("bundle") or "environment"),
        role="environment",
    ).sealed()
    coverage = SessionCoverageV5(
        model_calls=CoverageState.NOT_CAPTURED,
        agent_events=CoverageState.PARTIAL,
        environment_events=CoverageState.PARTIAL,
        usage=CoverageState.AGGREGATE_ONLY,
        raw_provider=CoverageState.UNAVAILABLE,
        reasons=("imported from experiments.trace.v4; no provider traffic was intercepted",),
    )

    return TraceDocumentV5(
        trace_id=trace_id,
        trace_kind=TraceKind.AGENT_ROLLOUT,
        identity=TraceIdentityV5(
            correlation_id=correlation or None,
            episode_id=str(payload.get("attempt_id") or "") or None,
            seed=_int(environment.get("seed")),
        ),
        lifecycle=TraceLifecycleV5(
            status=TraceStatus.COMPLETED, started_at=started_at, ended_at=ended_at
        ),
        capture=TraceCaptureSummaryV5(
            capture_id=record_id("cap", kind="imported", scope=(trace_id,), key=source_digest),
            binding_id="imported",
            binding_digest=source_digest,
            capture_profile="imported_experiments_v4",
            interception="none",
            mode="disabled",
        ),
        provenance=TraceProvenanceV5(
            producer=producer,
            producer_version=IMPORTER_VERSION,
            source_format="experiments.trace.v4",
            container_image_digest=str(environment.get("react_image_id") or "") or None,
            captured_at=imported_at,
            transformation_chain=(f"{IMPORTER_NAME}@{IMPORTER_VERSION}",),
            extra={"source_digest": source_digest},
        ),
        completeness=TraceCompletenessV5(
            capture_status=CaptureStatus.PARTIAL,
            # A completed V4 execution record is itself the terminal source fact.
            terminal_event_observed=True,
            model_calls=CoverageState.NOT_CAPTURED,
            raw_provider=CoverageState.UNAVAILABLE,
            agent_events=CoverageState.PARTIAL,
            environment_events=CoverageState.PARTIAL,
            usage=CoverageState.AGGREGATE_ONLY,
            reasons=("imported record; provider traffic was never intercepted",),
        ),
        actors=(agent, env),
        sessions=(
            SessionV5(
                session_id=session_id,
                actor_id=agent_actor_id,
                started_at=started_at,
                ended_at=ended_at,
                coverage=coverage,
            ).sealed(),
            SessionV5(
                session_id=env_session_id,
                actor_id=env_actor_id,
                started_at=started_at,
                ended_at=ended_at,
                coverage=coverage,
            ).sealed(),
        ),
        spans=tuple(spans),
        events=tuple(events),
        usage=usage,
        aliases=(
            AliasV1(
                namespace=AliasNamespace.EXPERIMENTS_TRACE_V4,
                value=correlation,
                target_id=trace_id,
                target_kind="trace",
            ),
        )
        if correlation
        else (),
    ).sealed()


def _mapping(value: Any) -> Mapping[str, Any]:
    """A foreign payload section, or an empty one when it is absent or the wrong shape."""
    return value if isinstance(value, Mapping) else {}


def _int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = ["IMPORTER_NAME", "IMPORTER_VERSION", "import_experiments_trace_v4"]
