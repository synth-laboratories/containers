"""Promote a container-sealed Craftax rollout into the lane-typed Craftax Trace V5.

The task platform seals every rollout as a harbor bundle whose document keeps
the runtime's native event kinds (``observation``, ``span.policy.plan``,
``action_applied`` …) and builds no spans or messages. The Craftax annotators
need the promoted shape ``promote_craftax_document`` produces from
``craftax.react-native-events.v1`` — ``model_call`` spans with observation/reply
messages and ``environment_step`` spans with transitions.

This adapter re-expresses the container document as those native events and
assembles the promoted document **under the container's own trace id**, so
evidence refs, selectors, and the annotation store all keep one identity. The
container's sealed digest is recorded in the promoted document's provenance.
"""

from __future__ import annotations

import json

import tempfile
from pathlib import Path
from typing import Any

from ..canonical import canonical_bytes, bytes_digest
from ..capture.binding import WorkloadKind
from ..models.document import TraceDocumentV5
from ..models.identity import TraceIdentityV5, TraceKind
from ..store.bundle import LocalTraceBundle
from ..validation.rehydrate import trace_document_from_payload
from . import native as native_mod
from .craftax import promote_craftax_document

SOURCE_FORMAT = "craftax.react-native-events.v1"
TRANSCRIPT_KINDS: frozenset[str] = frozenset(
    {
        "task_resolved",
        "action_applied",
        "resource_delta",
        "achievement_unlocked",
        "reward_delta",
        "state_transition",
        "entity_transition",
        "checkpoint_cadence",
        "episode_truncated",
        "terminal",
        "combat",
        "death",
    }
)


def is_container_craftax_rollout(document: TraceDocumentV5) -> bool:
    """A runtime-kind event log with policy plans and applied actions, and no lanes yet."""

    kinds = {str(event.event_type) for event in document.events}
    return (
        "span.policy.plan" in kinds
        and "action_applied" in kinds
        and not any(kind.startswith("craftax.") for kind in kinds)
    )


def _policy(document: TraceDocumentV5) -> dict[str, Any]:
    """Policy identity: the runtime's ``policy.session.opened`` record wins over the generic agent actor."""

    agent = next((actor for actor in document.actors if str(actor.kind) == "agent"), None)
    policy: dict[str, Any] = {"id": None, "kind": None, "model": None, "provider": None, "reasoning_effort": None}
    if agent is not None:
        policy.update({"id": agent.policy_id or agent.display_name, "kind": agent.harness, "model": agent.model, "provider": agent.provider})
    session_opened = next((e for e in document.events if str(e.event_type) == "policy.session.opened" and isinstance(e.payload, dict)), None)
    if session_opened is not None:
        payload = session_opened.payload
        policy.update({key: payload.get(source) for key, source in (("id", "config"), ("kind", "harness"), ("model", "model"), ("provider", "provider"), ("reasoning_effort", "reasoning_effort")) if payload.get(source)})
        policy["session_kind"] = payload.get("kind")
    extra = dict(document.provenance.extra or {})
    for key in ("policy_ref", "policy"):
        value = extra.get(key)
        if isinstance(value, dict):
            policy["id"] = policy["id"] or value.get("config")
            policy["kind"] = policy["kind"] or value.get("harness")
    return policy


def container_rollout_to_native_events(document: TraceDocumentV5) -> dict[str, Any]:
    """The ``craftax.react-native-events.v1`` payload equivalent to a container rollout document."""

    if not is_container_craftax_rollout(document):
        raise ValueError("document is not a container-sealed Craftax rollout")
    policy = _policy(document)
    seed = document.identity.seed
    if seed is None:
        instance = str(document.identity.task_instance_id or "")
        seed = int(instance.rsplit(":", 1)[-1]) if instance.rsplit(":", 1)[-1].isdigit() else 0
    rollout_id = document.identity.rollout_id or document.trace_id
    lane = f"container/craftax#{policy['id'] or policy['kind'] or 'policy'}#s{seed}"
    base = {"lane": lane, "rollout_id": rollout_id, "task_id": document.identity.task_id or "gamebench/craftax-singleplayer"}
    events: list[dict[str, Any]] = []
    prev: str | None = None

    def emit(event_type: str, occurred_at: str, payload: dict[str, Any]) -> None:
        nonlocal prev
        event_id = f"{lane}:{len(events)}"
        events.append({"event_id": event_id, "event_type": event_type, "occurred_at": occurred_at, "caused_by": [prev] if prev else [], "payload": {**base, **payload}})
        prev = event_id

    ordered = sorted(document.events, key=lambda item: (item.order.chronological_sequence if item.order.chronological_sequence is not None else 0, item.occurred_at, item.event_id))
    first_ts = ordered[0].occurred_at if ordered else document.lifecycle.started_at
    emit("craftax.eval.phase", first_ts, {"phase": "rollout.opened", "policy": {**policy, "env_seed": seed, "seed": seed, "graded": False}, "source": {"container_trace_id": document.trace_id, "container_trace_digest": document.content_digest}})
    latest_obs: str | None = None
    call_index = 0
    planned = 0
    steps = 0
    terminal_reason: str | None = None
    records = policy_call_records(document)
    for event in ordered:
        kind = str(event.event_type)
        payload = event.payload if isinstance(event.payload, dict) else {}
        if kind == "observation":
            readout = payload.get("readout") or {}
            latest_obs = (readout.get("observation_text") if isinstance(readout, dict) else None) or latest_obs
        elif kind == "span.policy.plan":
            actions = [str(item) for item in (payload.get("actions") or [])]
            planned += len(actions)
            record = records.get(call_index + 1) or records.get(call_index) or {}
            reasoning = record.get("reasoning")
            reply = ("THOUGHT: " + reasoning.strip().replace("\n", " ") + "\n" if reasoning else "") + "ACTIONS: " + ", ".join(actions)
            emit("craftax.transcript", event.occurred_at, {"kind": "policy.call", "call_index": call_index, "model": record.get("model") or policy.get("model") or policy.get("kind"), "reasoning_effort": policy.get("reasoning_effort"), "observation": "text", "prompt": latest_obs or "", "reply": reply, "finish_reason": "tool_call" if record.get("tool_arguments") else "plan", "prompt_tokens": int(record.get("prompt_tokens") or 0), "completion_tokens": int(record.get("completion_tokens") or 0), "reply_provenance": "span.policy.data" if record else "rendered from span.policy.plan", "container_event_id": event.event_id, "policy_data_event_id": record.get("event_id")})
            call_index += 1
        elif kind in TRANSCRIPT_KINDS:
            if kind == "action_applied":
                steps += 1
            if kind == "terminal":
                terminal_reason = (payload.get("payload") or {}).get("reason") if isinstance(payload.get("payload"), dict) else None
            emit("craftax.transcript", event.occurred_at, {"kind": kind, "action": payload.get("action"), "step_index": payload.get("step_index"), "tick": payload.get("tick"), "transition": payload.get("transition"), "payload": payload.get("payload") or {}, "message": payload.get("message"), "severity": payload.get("severity"), "episode_id": payload.get("episode_id"), "container_event_id": event.event_id})
    reward = None
    for event in reversed(ordered):
        if str(event.event_type) == "reward_signal" and isinstance(event.payload, dict):
            reward = event.payload.get("value", event.payload.get("reward"))
            break
    last_ts = ordered[-1].occurred_at if ordered else document.lifecycle.ended_at or first_ts
    emit("craftax.eval.run.terminal", last_ts, {"stopped_on": terminal_reason or "unknown", "terminated": True, "truncated": terminal_reason == "max_steps", "reward": reward, "env_steps": steps, "usage": {"calls": call_index, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "actions_planned": planned, "actions_taken": steps}})
    return {"run_id": f"container-{rollout_id}", "trace_correlation_id": rollout_id, "metadata": {"bridge": "container harbor document -> " + SOURCE_FORMAT, "container_trace_id": document.trace_id, "container_trace_digest": document.content_digest}, "events": events}


def policy_call_records(document: TraceDocumentV5) -> dict[int, dict[str, Any]]:
    """Final (non-delta) ``span.policy.data`` records by call number: reasoning, tool arguments, usage."""

    import json as _json

    records: dict[int, dict[str, Any]] = {}
    for event in document.events:
        if str(event.event_type) != "span.policy.data" or not isinstance(event.payload, dict):
            continue
        payload = event.payload
        if payload.get("delta") or payload.get("channel") is not None:
            continue
        call = payload.get("call")
        if not isinstance(call, int):
            continue
        reasoning = payload.get("reasoning")
        if isinstance(reasoning, str) and reasoning.startswith('"'):
            try:
                reasoning = _json.loads(reasoning)
            except ValueError:
                pass
        records[call] = {
            "event_id": event.event_id,
            "model": payload.get("model"),
            "provider": payload.get("provider"),
            "reasoning": reasoning if isinstance(reasoning, str) and reasoning.strip() else None,
            "assistant": payload.get("assistant") if isinstance(payload.get("assistant"), str) else None,
            "tool_arguments": payload.get("tool_arguments") if isinstance(payload.get("tool_arguments"), str) else None,
            "actions": list(payload.get("actions") or []),
            "prompt_tokens": payload.get("prompt_tokens"),
            "completion_tokens": payload.get("completion_tokens"),
            "usage": payload.get("usage") if isinstance(payload.get("usage"), dict) else None,
            "fallback": bool(payload.get("fallback")),
        }
    return records


def policy_call_steps(document: TraceDocumentV5) -> dict[int, dict[str, Any]]:
    """The environment steps applied after each ``span.policy.plan`` and before the next, by 0-based call index.

    A ``choose_actions`` tool call has no provider-side result: the environment
    steps that execute its action batch *are* the result. Each batch records the
    planned actions, the applied steps (``step_index``, ``action``, ``transition``
    and the engine's ``reason`` for a rejected/no-op step) and the terminal
    reason when the episode ended inside the batch.
    """

    ordered = sorted(document.events, key=lambda item: (item.order.chronological_sequence if item.order.chronological_sequence is not None else 0, item.occurred_at, item.event_id))
    batches: dict[int, dict[str, Any]] = {}
    call_index = -1
    for event in ordered:
        kind = str(event.event_type)
        payload = event.payload if isinstance(event.payload, dict) else {}
        detail = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
        if kind == "span.policy.plan":
            call_index += 1
            batches[call_index] = {"planned": [str(item) for item in (payload.get("actions") or [])], "steps": [], "terminal": None}
        elif call_index < 0:
            continue
        elif kind == "action_applied":
            step: dict[str, Any] = {"step_index": int(payload.get("step_index") or 0), "action": payload.get("action"), "transition": payload.get("transition")}
            if detail.get("reason"):
                step["reason"] = detail["reason"]
            batches[call_index]["steps"].append(step)
        elif kind == "terminal":
            batches[call_index]["terminal"] = detail.get("reason") or payload.get("transition") or "terminal"
    return batches


def _tool_result_message(message: Any, *, index: int, tool_call_id: str, batch: dict[str, Any], step_spans: dict[tuple[str, int], Any]) -> Any:
    """A ``tool`` message holding the synthetic ``choose_actions`` result: a compact JSON summary of the executed steps.

    The part's ``text`` is the same compact JSON as its ``structured`` payload, so
    ``part`` selectors resolve and quote against exactly what ``trace_get_message``
    and ``trace_get_tool_call`` render. Each step carries the ``span_id`` of its
    ``environment_step`` span so an annotator can open the full transition.
    """

    import json as _json
    from collections import Counter

    from ..canonical import record_id
    from ..models.messages import MessageNodeV5, MessagePartV5, MessageRole, PartType

    steps: list[dict[str, Any]] = []
    span_ids: list[str] = []
    last_span = None
    for step in batch["steps"]:
        span = step_spans.get((message.session_id, int(step["step_index"])))
        item = dict(step)
        if span is not None:
            item["span_id"] = span.span_id
            span_ids.append(span.span_id)
            last_span = span
        steps.append(item)
    outcomes = Counter(str(step.get("transition") or "unknown") for step in steps)
    summary: dict[str, Any] = {
        "tool_name": "choose_actions",
        "planned": len(batch["planned"]),
        "executed": len(steps),
        "outcomes": dict(sorted(outcomes.items())),
        "steps": steps,
        "terminal": batch.get("terminal"),
    }
    text = _json.dumps(summary, sort_keys=True, separators=(",", ":"))
    result_id = record_id("msg", kind="craftax_tool_result", scope=(message.session_id,), key=index)
    part = MessagePartV5(part_id=f"{result_id}:0", type=PartType.TOOL_RESULT, tool_call_id=tool_call_id, tool_name="choose_actions", text=text, structured=summary, is_error=False, visibility=message.visibility)
    return MessageNodeV5(
        message_id=result_id,
        role=MessageRole.TOOL,
        parts=(part,),
        sender_actor_id=message.sender_actor_id,
        session_id=message.session_id,
        predecessor_message_ids=(message.message_id,),
        occurred_at=(last_span.ended_at if last_span is not None else None) or message.occurred_at,
        visibility=message.visibility,
        metadata={"call_index": index, "tool_call_id": tool_call_id, "result_provenance": "synthesized from environment_step spans", "step_span_ids": span_ids},
    ).sealed()


def _content_action_plan(assistant: Any) -> str | None:
    """A plan the model wrote as content JSON (``{"actions": [...]}``) instead of
    calling ``choose_actions``. Some models (GLM 5.3 flash, for one) answer this
    way every call; the harness executes it identically, so promote it as the
    same tool call so tool-call annotators see one shape. Returns the compact
    JSON of the ``actions`` list, or None when the content is not such a plan."""

    if not isinstance(assistant, str) or "actions" not in assistant:
        return None
    text = assistant.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    actions = parsed.get("actions")
    if not isinstance(actions, list) or not actions or not all(isinstance(item, str) for item in actions):
        return None
    return json.dumps({"actions": actions}, separators=(",", ":"))


def _enrich_replies(promoted: TraceDocumentV5, records: dict[int, dict[str, Any]], batches: dict[int, dict[str, Any]] | None = None) -> TraceDocumentV5:
    """Give assistant replies their native structure: reasoning part + choose_actions tool call (+ its synthetic result) + rendered text."""

    from ..models.messages import MessagePartV5, PartType, ReasoningAvailability
    from dataclasses import replace

    if not records:
        return promoted
    batches = batches or {}
    step_spans = {(span.session_id, int(span.detail["step_index"])): span for span in promoted.spans if str(span.span_kind) == "environment_step" and isinstance(span.detail.get("step_index"), int)}
    messages = []
    changed = False
    for message in promoted.messages:
        if str(message.role) != "assistant" or not isinstance(message.metadata.get("call_index"), int):
            messages.append(message)
            continue
        index = int(message.metadata["call_index"])
        record = records.get(index + 1) or records.get(index)
        if not record:
            messages.append(message)
            continue
        parts: list[MessagePartV5] = []
        base = message.message_id
        metadata = {**message.metadata, "reply_provenance": "span.policy.data", "fallback": record.get("fallback")}
        tool_message = None
        if record.get("reasoning"):
            parts.append(MessagePartV5(part_id=f"{base}:reasoning", type=PartType.REASONING, text=record["reasoning"], reasoning_availability=ReasoningAvailability.CAPTURED))
        arguments_json = record.get("tool_arguments") or _content_action_plan(record.get("assistant"))
        if arguments_json:
            tool_call_id = f"{base}:choose_actions"
            if not record.get("tool_arguments"):
                metadata["tool_call_provenance"] = "content_json"
            parts.append(MessagePartV5(part_id=f"{base}:tool_call", type=PartType.TOOL_CALL, tool_call_id=tool_call_id, tool_name="choose_actions", arguments_json=arguments_json))
            batch = batches.get(index)
            if batch and batch["steps"]:
                tool_message = _tool_result_message(message, index=index, tool_call_id=tool_call_id, batch=batch, step_spans=step_spans)
                metadata["tool_result_message_id"] = tool_message.message_id
        if record.get("assistant"):
            parts.append(MessagePartV5(part_id=f"{base}:content", type=PartType.TEXT, text=record["assistant"]))
        parts.extend(message.parts)  # the rendered THOUGHT/ACTIONS text keeps text-based tooling working
        messages.append(replace(message, parts=tuple(parts), metadata=metadata, content_digest="").sealed())
        if tool_message is not None:
            messages.append(tool_message)
        changed = True
    if not changed:
        return promoted
    return replace(promoted, messages=tuple(messages), content_digest="").sealed()


def promote_container_rollout(document: TraceDocumentV5, sealed_digest: str | None = None) -> TraceDocumentV5:
    """Promoted Craftax document for a container rollout, keeping the container trace id.

    Returns the document unchanged when it is not a container Craftax rollout
    (already promoted, or a different task), so it is safe as a generic hook.
    """

    if not is_container_craftax_rollout(document):
        return document
    native = container_rollout_to_native_events(document)
    source_bytes = canonical_bytes(native)
    source_digest = bytes_digest(source_bytes)
    with tempfile.TemporaryDirectory(prefix="craftax-promote-") as tmp:
        bundle = LocalTraceBundle(Path(tmp), bundle_id=f"promote-{document.trace_id}")
        result = native_mod._assemble_events(
            native_mod._react_events(native),
            trace_id=document.trace_id,
            source_digest=source_digest,
            source_format=SOURCE_FORMAT,
            bundle=bundle,
            workload_kind=WorkloadKind.REACT,
            identity=TraceIdentityV5(rollout_id=document.identity.rollout_id, run_id=document.identity.run_id, correlation_id=document.identity.correlation_id or document.identity.rollout_id, task_id=document.identity.task_id, task_instance_id=document.identity.task_instance_id, benchmark=document.identity.benchmark or "craftax", seed=document.identity.seed),
            trace_kind=TraceKind.AGENT_ROLLOUT,
            document_adapter=lambda doc: promote_craftax_document(doc, source_digest=source_digest),
        )
        promoted = trace_document_from_payload(bundle.read_trace(result["trace_digest"]))
    policy = _policy(document)
    extra = {**promoted.provenance.extra, "promoted_from_container_trace_digest": sealed_digest or document.content_digest, "container_trace_id": document.trace_id, "policy": policy}
    from dataclasses import replace

    promoted = replace(promoted, provenance=replace(promoted.provenance, extra=extra, model=policy.get("model") or promoted.provenance.model, provider=policy.get("provider") or promoted.provenance.provider), content_digest="").sealed()
    return _enrich_replies(promoted, policy_call_records(document), policy_call_steps(document))


__all__ = ["SOURCE_FORMAT", "container_rollout_to_native_events", "is_container_craftax_rollout", "policy_call_records", "policy_call_steps", "promote_container_rollout"]
