"""Promote a container-sealed chat/classification rollout into messages + model_call spans.

Chat-style task platforms (Banking77, HealthBench, …) seal their rollouts as a
harbor document whose only content is the runtime event log: ``observation``
(``payload.system`` / ``payload.prompt`` / ``payload.text``, or a HealthBench-style
``payload.messages`` list of ``{role, content}`` turns), ``span.policy.opened``,
``action`` (``payload.text`` / ``payload.label``, or ``payload.content``),
``span.policy.closed``, ``reward_signal``. Annotators that reason about *LLM calls* need messages and a
``model_call`` span. This adapter derives them — under the container's own trace
id — and records the sealed digest it promoted from. Events are kept verbatim.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..canonical import record_id
from ..models.actors import ActorKind, ActorV5, CoverageState, SessionCoverageV5, SessionStatus, SessionV5
from ..models.document import TraceDocumentV5
from ..models.messages import MessageNodeV5, MessagePartV5, MessageRole, PartType
from ..models.spans import SpanKind, SpanStatus, SpanV5


def is_container_chat_rollout(document: TraceDocumentV5) -> bool:
    kinds = {str(event.event_type) for event in document.events}
    return (
        not document.messages
        and not document.spans
        and "observation" in kinds
        and "action" in kinds
        and any(str(e.event_type) == "observation" and isinstance(e.payload, dict) and ("prompt" in e.payload or "text" in e.payload or _conversation(e.payload)) for e in document.events)
    )


def _conversation(payload: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """``(role, content)`` turns from a HealthBench-style ``messages`` observation, else empty."""

    messages = payload.get("messages")
    if not isinstance(messages, list):
        return ()
    turns = []
    for item in messages:
        if not isinstance(item, dict) or not isinstance(item.get("content"), str):
            return ()
        turns.append((str(item.get("role") or "user"), item["content"]))
    return tuple(turns)


_ROLES = {"system": MessageRole.SYSTEM, "user": MessageRole.USER, "assistant": MessageRole.ASSISTANT}


def _ordered(document: TraceDocumentV5):
    return sorted(document.events, key=lambda item: (item.order.chronological_sequence if item.order.chronological_sequence is not None else 0, item.occurred_at, item.event_id))


def promote_chat_rollout(document: TraceDocumentV5, sealed_digest: str | None = None) -> TraceDocumentV5:
    if not is_container_chat_rollout(document):
        return document
    trace_id = document.trace_id
    orchestrator = next((a for a in document.actors if str(a.kind) == "orchestrator"), document.actors[0])
    opened = next((e for e in document.events if str(e.event_type) == "trace.opened"), None)
    policy_ref = (opened.payload.get("policy_ref") if opened and isinstance(opened.payload, dict) else None) or {}
    actor_id = record_id("actor", kind="chat_policy", scope=(trace_id,), key=policy_ref)
    session_id = record_id("sess", kind="chat_policy", scope=(trace_id, actor_id), key=policy_ref)
    started = document.lifecycle.started_at
    ended = document.lifecycle.ended_at or started
    actor = ActorV5(
        actor_id=actor_id,
        kind=ActorKind.AGENT,
        display_name=f"policy {policy_ref.get('config') or policy_ref.get('harness') or 'chat'}",
        role="policy",
        parent_actor_id=orchestrator.actor_id,
        harness=str(policy_ref.get("harness") or "") or None,
        policy_id=str(policy_ref.get("config") or "") or None,
        task_id=document.identity.task_id,
        metadata={"promoted_from": "chat_container"},
    ).sealed()
    session = SessionV5(
        session_id=session_id,
        actor_id=actor_id,
        started_at=started,
        ended_at=ended,
        status=SessionStatus.COMPLETED,
        harness=actor.harness,
        coverage=SessionCoverageV5(model_calls=CoverageState.PARTIAL, agent_events=CoverageState.COMPLETE, environment_events=CoverageState.COMPLETE, usage=CoverageState.NOT_CAPTURED, raw_provider=CoverageState.UNAVAILABLE, reasons=("model call derived from observation/action events; provider payload not captured",)),
    ).sealed()
    messages: list[MessageNodeV5] = []
    spans: list[SpanV5] = []
    predecessor: str | None = None
    system_id: str | None = None
    pending_prompt: tuple[str | tuple[str, ...], Any] | None = None
    call_index = 0

    def add(message_id: str, role: MessageRole, text: str, at: str, *, span_id: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        nonlocal predecessor
        messages.append(MessageNodeV5(message_id=message_id, role=role, parts=(MessagePartV5(part_id=f"{message_id}:0", type=PartType.TEXT, text=text),), sender_actor_id=actor_id, session_id=session_id, predecessor_message_ids=(predecessor,) if predecessor else (), produced_by_span_id=span_id, occurred_at=at, metadata=dict(metadata or {})).sealed())
        predecessor = message_id

    for event in _ordered(document):
        kind = str(event.event_type)
        payload = event.payload if isinstance(event.payload, dict) else {}
        if kind == "observation" and (turns := _conversation(payload)):
            inputs_for_call: list[str] = []
            for position, (role, content) in enumerate(turns):
                turn_id = record_id("msg", kind="chat_turn", scope=(session_id,), key=(call_index, position))
                add(turn_id, _ROLES.get(role.lower(), MessageRole.USER), content, event.occurred_at, metadata={"call_index": call_index, "turn_index": position, "source_event_id": event.event_id})
                inputs_for_call.append(turn_id)
            pending_prompt = (tuple(inputs_for_call), event)
        elif kind == "observation":
            system = payload.get("system")
            if isinstance(system, str) and system and system_id is None:
                system_id = record_id("msg", kind="chat_system", scope=(session_id,), key=system)
                add(system_id, MessageRole.SYSTEM, system, event.occurred_at, metadata={"source_event_id": event.event_id})
            prompt = payload.get("prompt") if isinstance(payload.get("prompt"), str) else payload.get("text")
            prompt_id = record_id("msg", kind="chat_prompt", scope=(session_id,), key=call_index)
            add(prompt_id, MessageRole.USER, str(prompt or ""), event.occurred_at, metadata={"call_index": call_index, "source_event_id": event.event_id, "query_text": payload.get("text")})
            pending_prompt = (prompt_id, event)
        elif kind == "action":
            span_id = record_id("span", kind="chat_model_call", scope=(trace_id, session_id), key=call_index)
            reply_id = record_id("msg", kind="chat_reply", scope=(session_id,), key=call_index)
            text = payload.get("text") if isinstance(payload.get("text"), str) else payload.get("content") if isinstance(payload.get("content"), str) else str(payload.get("label") or "")
            add(reply_id, MessageRole.ASSISTANT, text, event.occurred_at, span_id=span_id, metadata={"call_index": call_index, "source_event_id": event.event_id, "label": payload.get("label")})
            prompt_ids = (pending_prompt[0] if isinstance(pending_prompt[0], tuple) else (pending_prompt[0],)) if pending_prompt else ()
            inputs = tuple(i for i in ((system_id,) if system_id else ()) + prompt_ids if i)
            spans.append(SpanV5(span_id=span_id, span_kind=SpanKind.MODEL_CALL, actor_id=actor_id, session_id=session_id, started_at=pending_prompt[1].occurred_at if pending_prompt else event.occurred_at, ended_at=event.occurred_at, status=SpanStatus.OK, input_message_ids=inputs, output_message_ids=(reply_id,), detail={"call_index": call_index, "policy": policy_ref, "label": payload.get("label"), "source_event_id": event.event_id}).sealed())
            call_index += 1
            pending_prompt = None
    if not spans:
        return document
    extra = {**document.provenance.extra, "promoted_from_container_trace_digest": sealed_digest or document.content_digest, "container_trace_id": trace_id, "promotion": "chat_container"}
    promoted = replace(
        document,
        actors=tuple(document.actors) + (actor,),
        sessions=tuple(document.sessions) + (session,),
        messages=tuple(messages),
        spans=tuple(spans),
        provenance=replace(document.provenance, extra=extra, transformation_chain=tuple(document.provenance.transformation_chain) + ("chat_container_promotion@1",)),
        content_digest="",
    )
    return promoted.sealed()


def promote_container_rollout_any(document: TraceDocumentV5, sealed_digest: str | None = None) -> TraceDocumentV5:
    """Craftax lanes when the rollout is Craftax, chat promotion otherwise; unchanged if neither applies."""

    from .craftax_container import is_container_craftax_rollout, promote_container_rollout

    if is_container_craftax_rollout(document):
        return promote_container_rollout(document, sealed_digest)
    return promote_chat_rollout(document, sealed_digest)


__all__ = ["is_container_chat_rollout", "promote_chat_rollout", "promote_container_rollout_any"]
