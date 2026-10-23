"""Promote Craftax native application events into canonical Trace V5 planes."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

from ..canonical import record_id
from ...gen_ai import request_observation
from ..models.actors import (
    ActorKind,
    ActorV5,
    CoverageState,
    SessionCoverageV5,
    SessionStatus,
    SessionV5,
)
from ..models.completeness import (
    CaptureStatus,
    TerminationV5,
    TraceCompletenessV5,
    TraceLifecycleV5,
    TraceStatus,
)
from ..models.document import TraceDocumentV5
from ..models.events import EventType, EventV5
from ..models.identity import AliasV1, mint_actor_id, mint_session_id
from ..models.messages import MessageNodeV5, MessagePartV5, MessageRole, PartType
from ..models.spans import SpanKind, SpanStatus, SpanV5, UsageProvenance, UsageV5


CRAFTAX_EXTENSION_SCHEMA_VERSION = "synth.trace-extension.craftax.v1"


def promote_craftax_document(
    document: TraceDocumentV5,
    *,
    source_digest: str,
) -> TraceDocumentV5:
    """Return a sealed V5 document with typed Craftax lanes, calls, and steps.

    Native records remain losslessly available in event payloads. This adds the
    common V5 entities consumers need so a viewer does not have to understand
    evals JSONL or infer facts from labels.
    """

    native_events = tuple(
        event for event in document.events if str(event.event_type).startswith("craftax.")
    )
    if not native_events:
        return document

    lane_names = tuple(
        sorted(
            {
                str(event.payload.get("lane") or "").strip()
                for event in native_events
                if str(event.payload.get("lane") or "").strip()
            }
        )
    )
    if not lane_names:
        return document

    root_actor = document.actors[0]
    root_session = document.sessions[0]
    lane_records = {
        lane: _lane_record(document, lane, native_events, source_digest) for lane in lane_names
    }

    actors: list[ActorV5] = [
        replace(
            root_actor,
            kind=ActorKind.ORCHESTRATOR,
            display_name="Craftax evaluation",
            role="orchestrator",
            harness="suites.nonproduct.craftax",
            content_digest="",
        ).sealed()
    ]
    sessions: list[SessionV5] = [
        replace(
            root_session,
            status=SessionStatus.COMPLETED,
            ended_at=document.lifecycle.ended_at,
            coverage=SessionCoverageV5(
                agent_events=CoverageState.COMPLETE,
                reasons=("child rollout sessions carry model/environment coverage",),
            ),
            content_digest="",
        ).sealed()
    ]
    for lane in lane_names:
        record = lane_records[lane]
        actors.append(record["actor"])
        sessions.append(record["session"])

    messages: list[MessageNodeV5] = []
    spans: list[SpanV5] = []
    event_span_ids: dict[str, str] = {}
    for lane in lane_names:
        record = lane_records[lane]
        lane_messages, lane_spans, lane_event_spans = _lane_entities(
            document,
            lane,
            actor_id=record["actor"].actor_id,
            session_id=record["session"].session_id,
        )
        messages.extend(lane_messages)
        spans.extend(lane_spans)
        event_span_ids.update(lane_event_spans)

    remapped_events = tuple(
        _remap_event(event, lane_records, event_span_ids) for event in document.events
    )
    rollouts = tuple(lane_records[lane]["extension"] for lane in lane_names)
    aggregate_usage = _aggregate_usage(rollouts, source_digest=source_digest)
    task_ids = {str(item.get("task_id") or "") for item in rollouts}
    models = {str(item.get("model") or "") for item in rollouts}
    providers = {str(item.get("provider") or "") for item in rollouts}
    seeds = {item.get("seed") for item in rollouts if isinstance(item.get("seed"), int)}
    terminal_reasons = tuple(sorted({str(item.get("stopped_on") or "") for item in rollouts}))

    extensions = dict(document.extensions)
    extensions["craftax"] = {
        "schema_version": CRAFTAX_EXTENSION_SCHEMA_VERSION,
        "rollouts": list(rollouts),
        "paired": len(seeds) == 1 and len(rollouts) > 1,
        "source_digest": source_digest,
        "cost_provenance": "estimated_from_declared_rates",
    }
    completeness = TraceCompletenessV5(
        capture_status=CaptureStatus.COMPLETE,
        terminal_event_observed=all(item.get("terminal_event_id") for item in rollouts),
        model_calls=CoverageState.COMPLETE,
        raw_provider=CoverageState.NOT_CAPTURED,
        agent_events=CoverageState.COMPLETE,
        environment_events=CoverageState.COMPLETE,
        tool_events=CoverageState.NOT_CAPTURED,
        usage=CoverageState.COMPLETE if aggregate_usage.prompt_tokens is not None and aggregate_usage.completion_tokens is not None else CoverageState.PARTIAL,
        expected_record_count=len(document.events),
        captured_record_count=len(document.events),
        reasons=(
            "model calls and usage were observed by the Craftax policy harness",
            "raw provider transport was not captured",
        ),
        metadata={"adapter": "craftax_native@2", "rollout_count": len(rollouts)},
    )
    lifecycle = TraceLifecycleV5(
        status=TraceStatus.COMPLETED,
        started_at=min(event.occurred_at for event in native_events),
        ended_at=max(event.occurred_at for event in native_events),
        termination=TerminationV5(
            reason=terminal_reasons[0] if len(terminal_reasons) == 1 else "multi_lane_complete",
            detail=", ".join(value for value in terminal_reasons if value),
        ),
    )
    provenance = replace(
        document.provenance,
        model=next(iter(models)) if len(models) == 1 else None,
        provider=next(iter(providers)) if len(providers) == 1 else None,
        harness="suites.nonproduct.craftax",
        transformation_chain=(*document.provenance.transformation_chain, "craftax_native@2"),
        extra={**document.provenance.extra, "rollout_count": len(rollouts)},
    )
    identity = replace(
        document.identity,
        task_id=next(iter(task_ids)) if len(task_ids) == 1 else document.identity.task_id,
        benchmark="craftax",
        seed=next(iter(seeds)) if len(seeds) == 1 else document.identity.seed,
    )
    return replace(
        document,
        trace_kind="evaluation_attempt",
        identity=identity,
        lifecycle=lifecycle,
        provenance=provenance,
        completeness=completeness,
        actors=tuple(actors),
        sessions=tuple(sessions),
        messages=tuple(messages),
        spans=tuple(spans),
        events=remapped_events,
        usage=aggregate_usage,
        extensions=extensions,
        content_digest="",
    ).sealed()


def _lane_record(
    document: TraceDocumentV5,
    lane: str,
    events: tuple[EventV5, ...],
    source_digest: str,
) -> dict[str, Any]:
    lane_events = tuple(event for event in events if event.payload.get("lane") == lane)
    opened = next(
        (
            event
            for event in lane_events
            if str(event.event_type) == "craftax.eval.phase"
            and event.payload.get("phase") == "rollout.opened"
        ),
        None,
    )
    terminal = next(
        (
            event
            for event in reversed(lane_events)
            if str(event.event_type) == "craftax.eval.run.terminal"
        ),
        None,
    )
    policy = dict(opened.payload.get("policy") or {}) if opened else {}
    rollout_id = str(
        (terminal.payload.get("rollout_id") if terminal else None)
        or (opened.payload.get("rollout_id") if opened else None)
        or lane
    )
    actor_id = mint_actor_id(trace_id=document.trace_id, name=f"craftax:{lane}")
    session_id = mint_session_id(
        trace_id=document.trace_id,
        actor_id=actor_id,
        nonce=rollout_id,
    )
    latest_usage: dict[str, Any] = {}
    for event in lane_events:
        usage = event.payload.get("usage")
        if isinstance(usage, dict) and int(usage.get("calls") or 0) >= int(
            latest_usage.get("calls") or 0
        ):
            latest_usage = dict(usage)
    actions = [
        {
            "step": int(event.payload.get("step_index") or 0),
            "action": str(event.payload.get("action") or ""),
            "transition": str(event.payload.get("transition") or ""),
            "reason": str((event.payload.get("payload") or {}).get("reason") or "") or None,
            "event_id": event.event_id,
        }
        for event in lane_events
        if event.payload.get("kind") == "action_applied"
    ]
    achievements = [
        {
            "step": int(event.payload.get("step_index") or 0),
            "name": str((event.payload.get("payload") or {}).get("achievement") or ""),
            "event_id": event.event_id,
        }
        for event in lane_events
        if event.payload.get("kind") == "achievement_unlocked"
    ]
    calls = [
        {
            "call_index": int(event.payload.get("call_index") or 0),
            "event_id": event.event_id,
            "prompt_tokens": _known_count(event.payload.get("prompt_tokens")),
            "completion_tokens": _known_count(event.payload.get("completion_tokens")),
        }
        for event in lane_events
        if event.payload.get("kind") == "policy.call"
    ]
    reward = terminal.payload.get("reward") if terminal else None
    env_steps = terminal.payload.get("env_steps") if terminal else len(actions)
    effort = str(policy.get("reasoning_effort") or "")
    model = str(policy.get("model") or "")
    provider = str(policy.get("provider") or "")
    display_name = " · ".join(value for value in (model, effort) if value) or lane
    actor = ActorV5(
        actor_id=actor_id,
        kind=ActorKind.AGENT,
        display_name=display_name,
        role="policy",
        subtype="craftax_react",
        parent_actor_id=document.actors[0].actor_id,
        harness="suites.nonproduct.craftax",
        runtime="gamebench/craftax-singleplayer",
        model=model or None,
        provider=provider or None,
        policy_id=str(policy.get("id") or "") or None,
        task_id=str((opened.payload.get("task_id") if opened else "") or "") or None,
        aliases=(
            AliasV1(
                namespace="gamebench.rollout",
                value=rollout_id,
                target_id=actor_id,
                target_kind="actor",
                provenance="observed",
            ),
        ),
        metadata={
            "lane": lane,
            "reasoning_effort": effort or None,
            "seed": policy.get("seed"),
            "reward": reward,
            "env_steps": env_steps,
        },
    ).sealed()
    session = SessionV5(
        session_id=session_id,
        actor_id=actor_id,
        started_at=min(event.occurred_at for event in lane_events),
        ended_at=max(event.occurred_at for event in lane_events),
        attempt_id=rollout_id,
        capture_id=document.capture.capture_id,
        parent_session_id=document.sessions[0].session_id,
        status=SessionStatus.COMPLETED,
        harness="suites.nonproduct.craftax",
        provider=provider or None,
        coverage=SessionCoverageV5(
            model_calls=CoverageState.COMPLETE,
            agent_events=CoverageState.COMPLETE,
            environment_events=CoverageState.COMPLETE,
            tool_events=CoverageState.NOT_CAPTURED,
            usage=CoverageState.COMPLETE,
            raw_provider=CoverageState.NOT_CAPTURED,
            reasons=("harness-native call and environment records",),
        ),
        aliases=(
            AliasV1(
                namespace="gamebench.rollout",
                value=rollout_id,
                target_id=session_id,
                target_kind="session",
                provenance="observed",
            ),
        ),
        metadata={"lane": lane, "reasoning_effort": effort or None},
    ).sealed()
    return {
        "actor": actor,
        "session": session,
        "extension": {
            "lane": lane,
            "rollout_id": rollout_id,
            "actor_id": actor_id,
            "session_id": session_id,
            "model": model or None,
            "provider": provider or None,
            "reasoning_effort": effort or None,
            "policy_kind": policy.get("kind"),
            "seed": policy.get("seed"),
            "task_id": (opened.payload.get("task_id") if opened else None),
            "reward": reward,
            "env_steps": env_steps,
            "stopped_on": (terminal.payload.get("stopped_on") if terminal else None),
            "terminal_event_id": terminal.event_id if terminal else None,
            "usage": latest_usage,
            "actions": actions,
            "achievements": achievements,
            "model_calls": calls,
            "source_digest": source_digest,
        },
    }


def _lane_entities(
    document: TraceDocumentV5,
    lane: str,
    *,
    actor_id: str,
    session_id: str,
) -> tuple[list[MessageNodeV5], list[SpanV5], dict[str, str]]:
    messages: list[MessageNodeV5] = []
    spans: list[SpanV5] = []
    event_spans: dict[str, str] = {}
    predecessor: str | None = None
    for event in document.events:
        if event.payload.get("lane") != lane:
            continue
        kind = event.payload.get("kind")
        if kind == "policy.call":
            call_index = int(event.payload.get("call_index") or 0)
            span_id = record_id(
                "span",
                kind="craftax_model_call",
                scope=(document.trace_id, session_id),
                key=call_index,
            )
            input_ids: list[str] = []
            prefix = event.payload.get("prefix")
            if isinstance(prefix, str) and prefix:
                prefix_id = record_id(
                    "msg",
                    kind="craftax_system",
                    scope=(session_id,),
                    key=event.payload.get("prefix_digest"),
                )
                messages.append(
                    _message(
                        prefix_id,
                        MessageRole.SYSTEM,
                        prefix,
                        actor_id,
                        session_id,
                        event.occurred_at,
                        predecessor,
                        metadata={"prefix_digest": event.payload.get("prefix_digest")},
                    )
                )
                predecessor = prefix_id
                input_ids.append(prefix_id)
            prompt_id = record_id("msg", kind="craftax_prompt", scope=(session_id,), key=call_index)
            messages.append(
                _message(
                    prompt_id,
                    MessageRole.USER,
                    str(event.payload.get("prompt") or ""),
                    actor_id,
                    session_id,
                    event.occurred_at,
                    predecessor,
                    metadata={
                        "call_index": call_index,
                        "observation": event.payload.get("observation"),
                    },
                )
            )
            predecessor = prompt_id
            input_ids.append(prompt_id)
            reply_id = record_id("msg", kind="craftax_reply", scope=(session_id,), key=call_index)
            reply = _message(
                reply_id,
                MessageRole.ASSISTANT,
                str(event.payload.get("reply") or ""),
                actor_id,
                session_id,
                event.occurred_at,
                predecessor,
                produced_by_span_id=span_id,
                metadata={
                    "call_index": call_index,
                    "finish_reason": event.payload.get("finish_reason"),
                },
            )
            messages.append(reply)
            predecessor = reply_id
            prompt_tokens = _known_count(event.payload.get("prompt_tokens"))
            completion_tokens = _known_count(event.payload.get("completion_tokens"))
            spans.append(
                SpanV5(
                    span_id=span_id,
                    span_kind=SpanKind.MODEL_CALL,
                    actor_id=actor_id,
                    session_id=session_id,
                    started_at=event.occurred_at,
                    ended_at=event.occurred_at,
                    status=SpanStatus.OK,
                    input_message_ids=tuple(input_ids),
                    output_message_ids=(reply_id,),
                    usage=UsageV5(
                        provenance=UsageProvenance.OBSERVED_HARNESS if prompt_tokens is not None and completion_tokens is not None else UsageProvenance.PARTIAL,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens if prompt_tokens is not None and completion_tokens is not None else None,
                        requests=1,
                        unavailable_fields=tuple(name for name, value in (("prompt_tokens", prompt_tokens), ("completion_tokens", completion_tokens), ("reasoning_tokens", None)) if value is None),
                        source_refs=(event.event_id,),
                    ),
                    detail={
                        "call_index": call_index,
                        "model": event.payload.get("model"),
                        "reasoning_effort": event.payload.get("reasoning_effort"),
                        "observation": event.payload.get("observation"),
                        "finish_reason": event.payload.get("finish_reason"),
                        "native_event_id": event.event_id,
                        **request_observation(
                            {
                                "model": event.payload.get("model"),
                                "temperature": event.payload.get("temperature"),
                                "top_p": event.payload.get("top_p"),
                                "top_k": event.payload.get("top_k"),
                                "max_tokens": event.payload.get("max_tokens"),
                                "reasoning_effort": event.payload.get(
                                    "reasoning_effort"
                                ),
                            }
                        ),
                    },
                    aliases=(
                        AliasV1(
                            namespace="craftax.policy_call",
                            value=f"{lane}:{call_index}",
                            target_id=span_id,
                            target_kind="span",
                            provenance="observed",
                        ),
                    ),
                ).sealed()
            )
            event_spans[event.event_id] = span_id
        elif kind == "action_applied":
            step = int(event.payload.get("step_index") or 0)
            span_id = record_id(
                "span",
                kind="craftax_environment_step",
                scope=(document.trace_id, session_id),
                key=step,
            )
            spans.append(
                SpanV5(
                    span_id=span_id,
                    span_kind=SpanKind.ENVIRONMENT_STEP,
                    actor_id=actor_id,
                    session_id=session_id,
                    started_at=event.occurred_at,
                    ended_at=event.occurred_at,
                    status=SpanStatus.OK,
                    turn_id=f"{session_id}:step:{step}",
                    detail={
                        "step_index": step,
                        "action": event.payload.get("action"),
                        "transition": event.payload.get("transition"),
                        "reason": (event.payload.get("payload") or {}).get("reason"),
                        "native_event_id": event.event_id,
                    },
                ).sealed()
            )
            event_spans[event.event_id] = span_id
    return messages, spans, event_spans


def _message(
    message_id: str,
    role: MessageRole,
    text: str,
    actor_id: str,
    session_id: str,
    occurred_at: str,
    predecessor: str | None,
    *,
    produced_by_span_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> MessageNodeV5:
    return MessageNodeV5(
        message_id=message_id,
        role=role,
        parts=(
            MessagePartV5(
                part_id=f"{message_id}:0",
                type=PartType.TEXT,
                text=text,
            ),
        ),
        sender_actor_id=actor_id,
        session_id=session_id,
        predecessor_message_ids=(predecessor,) if predecessor else (),
        produced_by_span_id=produced_by_span_id,
        occurred_at=occurred_at,
        metadata=dict(metadata or {}),
    ).sealed()


def _remap_event(
    event: EventV5,
    lane_records: dict[str, dict[str, Any]],
    event_span_ids: dict[str, str],
) -> EventV5:
    lane = str(event.payload.get("lane") or "")
    record = lane_records.get(lane)
    if record is None:
        return event
    native_type = str(event.event_type)
    native_kind = str(event.payload.get("kind") or "")
    event_type: EventType | str = native_type
    if native_kind == "policy.call":
        event_type = EventType.MODEL_CALL_FINISHED
    elif native_kind == "action_applied":
        event_type = EventType.ENV_ACTION_EXECUTED
    elif native_kind in {"achievement_unlocked", "reward_delta"}:
        event_type = EventType.ENV_REWARD
    elif native_type == "craftax.snapshot":
        event_type = EventType.ENV_OBSERVATION
    elif native_type == "craftax.eval.run.terminal":
        event_type = EventType.ENV_TERMINAL
    elif native_type == "craftax.eval.phase" and event.payload.get("phase") == "rollout.opened":
        event_type = EventType.SESSION_STARTED
    payload = {**event.payload, "native_event_type": native_type}
    return replace(
        event,
        event_type=event_type,
        actor_id=record["actor"].actor_id,
        session_id=record["session"].session_id,
        span_id=event_span_ids.get(event.event_id),
        payload=payload,
        content_digest="",
    ).sealed()


def _aggregate_usage(
    rollouts: tuple[dict[str, Any], ...],
    *,
    source_digest: str,
) -> UsageV5:
    latest = [item.get("usage") or {} for item in rollouts]
    def total(name):
        values = [_known_count(item.get(name)) for item in latest]
        return sum(value for value in values if value is not None) if values and all(value is not None for value in values) else None
    costs = [item.get("estimated_usd") for item in latest]
    known_costs = [known for value in costs if (known := _known_cost(value)) is not None]
    return UsageV5(
        provenance=UsageProvenance.PARTIAL,
        prompt_tokens=total("prompt_tokens"), completion_tokens=total("completion_tokens"),
        cached_tokens=total("cached_prompt_tokens"), total_tokens=total("total_tokens"),
        requests=total("calls"), cost_usd=sum(known_costs) if costs and len(costs) == len(known_costs) else None,
        unavailable_fields=("reasoning_tokens", "raw_provider"), source_refs=(source_digest,),
    )


def _known_count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _known_cost(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
        return float(value)
    return None



__all__ = ["CRAFTAX_EXTENSION_SCHEMA_VERSION", "promote_craftax_document"]
