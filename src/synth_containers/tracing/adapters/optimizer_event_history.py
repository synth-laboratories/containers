"""Import optimizer ``event_history`` records without discarding application events."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, Mapping

from synth_containers.rollout_tracing.v4 import chat_message_to_canonical

from ..canonical import bytes_digest, canonical_bytes, record_id
from ..models.actors import CoverageState
from ..models.completeness import CaptureStatus, TraceCompletenessV5
from ..models.events import EventOrderV1, EventType, EventV5
from ..models.identity import TraceProvenanceV5
from ..models.spans import UsageProvenance, UsageV5
from .v4 import import_rollout_trace_v4


IMPORTER_NAME = "optimizer_event_history"
IMPORTER_VERSION = "1"
IMPORTED_AT = "1970-01-01T00:00:00Z"


def import_optimizer_event_history(
    payload: Mapping[str, Any],
    *,
    imported_at: str = IMPORTED_AT,
) -> Any:
    """Build a deterministic, explicitly partial V5 trace from optimizer events.

    ``lm_call`` entries become canonical model-call spans while every mapping entry,
    including non-model application events, remains represented by a V5 event. The
    source format has no capture binding or raw-provider completeness proof, so the
    resulting document always declares partial coverage.
    """

    source_digest = bytes_digest(canonical_bytes(payload))
    raw_events = _event_history(payload)
    correlation_id = _correlation_id(payload)
    rollout_id = _text(payload.get("rollout_id") or payload.get("id"))
    if not rollout_id:
        trace_payload = payload.get("trace")
        if isinstance(trace_payload, Mapping):
            rollout_id = _text(trace_payload.get("rollout_id") or trace_payload.get("trace_id"))

    lm_entries: list[tuple[int, Mapping[str, Any]]] = []
    non_model_entries: list[tuple[int, Mapping[str, Any]]] = []
    invalid_indexes: list[int] = []
    for index, item in enumerate(raw_events):
        if not isinstance(item, Mapping):
            invalid_indexes.append(index)
            continue
        if _event_type(item) == "lm_call":
            lm_entries.append((index, item))
        else:
            non_model_entries.append((index, item))

    v4_spans = [
        _lm_call_span(
            _redacted_event(item),
            source_index=index,
            ordinal=ordinal,
        )
        for ordinal, (index, item) in enumerate(lm_entries)
    ]
    source_index_by_span_id = {
        str(span["span_id"]): index
        for span, (index, _) in zip(v4_spans, lm_entries, strict=True)
    }
    v4_payload = {
        "rollout_id": rollout_id,
        "trace_correlation_id": correlation_id,
        "spans": v4_spans,
    }
    document = import_rollout_trace_v4(
        v4_payload,
        producer="synth_containers.tracing.adapters.optimizer_event_history",
        imported_at=imported_at,
    )
    source_event_by_span_id = {
        str(span["span_id"]): item
        for span, (_, item) in zip(v4_spans, lm_entries, strict=True)
    }
    spans = tuple(
        replace(
            span,
            usage=_usage_from_event(
                source_event_by_span_id[str(span.detail["source_span_id"])],
                source_digest=source_digest,
            ),
            content_digest="",
        ).sealed()
        for span in document.spans
    )
    aggregate_usage = _aggregate_usage(spans, source_digest=source_digest)
    actor_id = document.actors[0].actor_id
    session_id = document.sessions[0].session_id
    span_by_source_index = {
        source_index_by_span_id[str(span.detail["source_span_id"])]: span
        for span in spans
        if str(span.detail.get("source_span_id") or "") in source_index_by_span_id
    }

    events: list[EventV5] = []
    for index, item in enumerate(raw_events):
        if not isinstance(item, Mapping):
            continue
        native_type = _event_type(item)
        span = span_by_source_index.get(index)
        redacted_native, redaction = _redact_event(item)
        event_type = (
            EventType.MODEL_CALL_FINISHED if native_type == "lm_call" else native_type
        )
        event_id = record_id(
            "evt",
            kind="optimizer_event_history",
            scope=(document.trace_id,),
            key={"index": index, "source_digest": source_digest},
        )
        events.append(
            EventV5(
                event_id=event_id,
                event_type=event_type,
                actor_id=actor_id,
                session_id=session_id,
                occurred_at=_occurred_at(item, fallback=imported_at),
                span_id=span.span_id if span is not None else None,
                order=EventOrderV1(
                    chronological_sequence=index + 1,
                    source_order_id=_source_order_id(item, index=index),
                ),
                payload={
                    "imported_event_index": index,
                    "native_event_type": native_type,
                    "native_event": redacted_native,
                    "redaction": redaction.to_dict(),
                },
            ).sealed()
        )

    reasons = [
        "optimizer event_history carries no capture binding",
        "optimizer event_history does not prove raw provider completeness",
    ]
    if invalid_indexes:
        reasons.append("non-object event_history entries were not importable")
    if non_model_entries:
        reasons.append(
            "non-model optimizer events were preserved as typed application-event payloads"
        )
    missing_ranges = tuple(str(index) for index in invalid_indexes)
    agent_events = (
        CoverageState.PARTIAL if non_model_entries else CoverageState.NOT_CAPTURED
    )
    environment_events = (
        CoverageState.PARTIAL
        if any(_is_environment_event(item) for _, item in non_model_entries)
        else CoverageState.NOT_CAPTURED
    )
    tool_events = (
        CoverageState.PARTIAL
        if any(_is_tool_event(item) for _, item in non_model_entries)
        else CoverageState.NOT_CAPTURED
    )
    usage_coverage = (
        CoverageState.NOT_CAPTURED
        if not lm_entries
        else (
            CoverageState.UNAVAILABLE
            if aggregate_usage.provenance == UsageProvenance.UNAVAILABLE
            else CoverageState.AGGREGATE_ONLY
        )
    )
    started_at, ended_at = _event_bounds(events, fallback=imported_at)
    session = replace(
        document.sessions[0],
        started_at=started_at,
        ended_at=ended_at,
        coverage=replace(
            document.sessions[0].coverage,
            agent_events=agent_events,
            environment_events=environment_events,
            tool_events=tool_events,
            usage=usage_coverage,
            reasons=tuple(reasons),
        ),
        content_digest="",
    ).sealed()
    return replace(
        document,
        identity=replace(
            document.identity,
            rollout_id=rollout_id or document.identity.rollout_id,
            correlation_id=correlation_id or document.identity.correlation_id,
        ),
        lifecycle=replace(
            document.lifecycle,
            started_at=started_at,
            ended_at=ended_at,
        ),
        provenance=TraceProvenanceV5(
            producer="synth_containers.tracing.adapters.optimizer_event_history",
            producer_version=IMPORTER_VERSION,
            source_format="optimizer.event_history",
            captured_at=imported_at,
            transformation_chain=(f"{IMPORTER_NAME}@{IMPORTER_VERSION}",),
            extra={"source_digest": source_digest},
        ),
        completeness=TraceCompletenessV5(
            capture_status=CaptureStatus.PARTIAL,
            terminal_event_observed=True,
            model_calls=(
                CoverageState.PARTIAL if lm_entries else CoverageState.NOT_CAPTURED
            ),
            raw_provider=CoverageState.UNAVAILABLE,
            agent_events=agent_events,
            environment_events=environment_events,
            tool_events=tool_events,
            usage=usage_coverage,
            expected_record_count=len(raw_events),
            captured_record_count=len(raw_events) - len(invalid_indexes),
            high_water_ordinal=(len(raw_events) - 1 if raw_events else None),
            missing_ranges=missing_ranges,
            reasons=tuple(reasons),
            metadata={
                "source_event_count": len(raw_events),
                "lm_call_count": len(lm_entries),
                "non_model_event_count": len(non_model_entries),
                "invalid_event_count": len(invalid_indexes),
            },
        ),
        sessions=(session,),
        spans=spans,
        events=tuple(events),
        usage=aggregate_usage,
        content_digest="",
    ).sealed()


def _event_history(payload: Mapping[str, Any]) -> list[Any]:
    candidates = [payload.get("event_history")]
    trace = payload.get("trace")
    if isinstance(trace, Mapping):
        candidates.append(trace.get("event_history"))
    session_trace = payload.get("session_trace")
    if isinstance(session_trace, Mapping):
        candidates.append(session_trace.get("event_history"))
    for candidate in candidates:
        if isinstance(candidate, list):
            return candidate
    raise ValueError("optimizer.event_history source requires an event_history array")


def _lm_call_span(
    event: Mapping[str, Any],
    *,
    source_index: int,
    ordinal: int,
) -> dict[str, Any]:
    request = event.get("llm_request")
    response = event.get("llm_response")
    request_payload = dict(request) if isinstance(request, Mapping) else {}
    response_payload = dict(response) if isinstance(response, Mapping) else {}
    request_payload["messages"] = [
        _canonical_message(item)
        for item in request_payload.get("messages") or ()
        if isinstance(item, Mapping)
    ]
    if not isinstance(response_payload.get("choices"), list):
        message = response_payload.get("message")
        response_payload["choices"] = (
            [
                {
                    "index": 0,
                    "message": _canonical_message(message),
                    "finish_reason": response_payload.get("finish_reason"),
                }
            ]
            if isinstance(message, Mapping)
            else []
        )
    usage = response_payload.get("usage")
    if isinstance(usage, Mapping):
        completion_details = usage.get("completion_tokens_details")
        reasoning_tokens = (
            completion_details.get("reasoning_tokens")
            if isinstance(completion_details, Mapping)
            else None
        )
        response_payload["usage"] = {
            "prompt_tokens": usage.get("prompt_tokens") or usage.get("input_tokens"),
            "completion_tokens": usage.get("completion_tokens") or usage.get("output_tokens"),
            "reasoning_tokens": usage.get("reasoning_tokens") or reasoning_tokens,
            "cached_tokens": usage.get("cached_tokens"),
        }
    source_span_id = _text(event.get("span_id") or event.get("event_id"))
    return {
        "span_id": source_span_id or f"optimizer-event-{source_index}",
        "call_index": _integer(
            event.get("sequence_index") if event.get("sequence_index") is not None
            else event.get("call_index"),
            fallback=ordinal,
        ),
        "parent_span_id": event.get("parent_span_id"),
        "request": request_payload,
        "response": response_payload,
        "api_format": event.get("api_format"),
        "metrics": dict(event.get("metrics") or {})
        if isinstance(event.get("metrics"), Mapping)
        else {},
        "metadata": {
            **(
                dict(event.get("metadata") or {})
                if isinstance(event.get("metadata"), Mapping)
                else {}
            ),
            "source_event_index": source_index,
        },
    }


def _canonical_message(message: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(message.get("parts"), list):
        return dict(message)
    return chat_message_to_canonical(message).to_dict()


def _usage_from_event(
    event: Mapping[str, Any],
    *,
    source_digest: str,
) -> UsageV5:
    response = event.get("llm_response")
    usage = response.get("usage") if isinstance(response, Mapping) else None
    if not isinstance(usage, Mapping):
        return UsageV5(
            provenance=UsageProvenance.UNAVAILABLE,
            requests=1,
            unavailable_fields=(
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            ),
            source_refs=(source_digest,),
        )
    completion_details = usage.get("completion_tokens_details")
    reasoning = (
        completion_details.get("reasoning_tokens")
        if isinstance(completion_details, Mapping)
        else None
    )
    prompt = _optional_integer(_first_present(usage, "prompt_tokens", "input_tokens"))
    completion = _optional_integer(
        _first_present(usage, "completion_tokens", "output_tokens")
    )
    total = _optional_integer(usage.get("total_tokens"))
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    values = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "reasoning_tokens": _optional_integer(
            usage.get("reasoning_tokens")
            if usage.get("reasoning_tokens") is not None
            else reasoning
        ),
        "cached_tokens": _optional_integer(usage.get("cached_tokens")),
        "total_tokens": total,
    }
    unavailable = tuple(
        name
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        if values[name] is None
    )
    return UsageV5(
        provenance=(
            UsageProvenance.OBSERVED_HARNESS
            if not unavailable
            else (
                UsageProvenance.PARTIAL
                if any(value is not None for value in values.values())
                else UsageProvenance.UNAVAILABLE
            )
        ),
        prompt_tokens=prompt,
        completion_tokens=completion,
        reasoning_tokens=values["reasoning_tokens"],
        cached_tokens=values["cached_tokens"],
        total_tokens=total,
        requests=1,
        unavailable_fields=unavailable,
        source_refs=(source_digest,),
    )


def _aggregate_usage(spans: tuple[Any, ...], *, source_digest: str) -> UsageV5:
    if not spans:
        return UsageV5()
    usages = [span.usage for span in spans if span.usage is not None]
    if not usages or all(
        usage.provenance == UsageProvenance.UNAVAILABLE for usage in usages
    ):
        return UsageV5(
            provenance=UsageProvenance.UNAVAILABLE,
            requests=len(spans),
            unavailable_fields=(
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            ),
            source_refs=(source_digest,),
        )

    def aggregate(name: str) -> int | None:
        values = [getattr(usage, name) for usage in usages]
        return sum(int(value) for value in values if value is not None) if any(
            value is not None for value in values
        ) else None

    unavailable = tuple(
        name
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        if any(getattr(usage, name) is None for usage in usages)
    )
    return UsageV5(
        provenance=(
            UsageProvenance.DERIVED
            if not unavailable
            and all(
                usage.provenance == UsageProvenance.OBSERVED_HARNESS
                for usage in usages
            )
            else UsageProvenance.PARTIAL
        ),
        prompt_tokens=aggregate("prompt_tokens"),
        completion_tokens=aggregate("completion_tokens"),
        reasoning_tokens=aggregate("reasoning_tokens"),
        cached_tokens=aggregate("cached_tokens"),
        total_tokens=aggregate("total_tokens"),
        requests=len(spans),
        unavailable_fields=unavailable,
        source_refs=(source_digest,),
    )


def _redacted_event(event: Mapping[str, Any]) -> Mapping[str, Any]:
    redacted, _ = _redact_event(event)
    if not isinstance(redacted, Mapping):
        raise TypeError("redacted optimizer event must remain a mapping")
    return redacted


def _redact_event(event: Mapping[str, Any]) -> tuple[Any, Any]:
    # A module-level capture import creates adapters -> capture.finalizer -> adapters
    # during package initialization. Redaction is a runtime transformation boundary.
    from ..capture.redaction import redact_payload

    return redact_payload(dict(event))


def _correlation_id(payload: Mapping[str, Any]) -> str | None:
    direct = _text(payload.get("trace_correlation_id"))
    if direct:
        return direct
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        direct = _text(metadata.get("trace_correlation_id") or metadata.get("correlation_id"))
        if direct:
            return direct
        identifiers = metadata.get("correlation_ids")
        if isinstance(identifiers, Mapping):
            return _text(
                identifiers.get("trace_correlation_id") or identifiers.get("correlation_id")
            ) or None
    return None


def _event_type(event: Mapping[str, Any]) -> str:
    return _text(event.get("event_type") or event.get("type")) or "optimizer.event"


def _source_order_id(event: Mapping[str, Any], *, index: int) -> str:
    return _text(
        event.get("event_id")
        or event.get("span_id")
        or event.get("sequence_index")
        or event.get("step_index")
    ) or str(index)


def _occurred_at(event: Mapping[str, Any], *, fallback: str) -> str:
    for field in ("occurred_at", "at", "created_at"):
        value = event.get(field)
        if isinstance(value, str) and value:
            return value
    return fallback


def _event_bounds(
    events: list[EventV5],
    *,
    fallback: str,
) -> tuple[str, str]:
    values = [event.occurred_at for event in events] or [fallback]

    def parsed(value: str) -> datetime:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if moment.utcoffset() is None:
            raise ValueError(
                f"optimizer event timestamp must include a timezone: {value!r}"
            )
        return moment.astimezone(UTC)

    return min(values, key=parsed), max(values, key=parsed)


def _is_environment_event(event: Mapping[str, Any]) -> bool:
    return _event_type(event).startswith(("environment.", "env_"))


def _is_tool_event(event: Mapping[str, Any]) -> bool:
    return _event_type(event).startswith(("tool.", "tool_"))


def _integer(value: Any, *, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _optional_integer(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_present(payload: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if payload.get(name) is not None:
            return payload[name]
    return None


def _text(value: Any) -> str:
    return str(value) if value not in (None, "") else ""


__all__ = [
    "IMPORTED_AT",
    "IMPORTER_NAME",
    "IMPORTER_VERSION",
    "import_optimizer_event_history",
]
