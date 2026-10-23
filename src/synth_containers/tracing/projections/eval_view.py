"""Project a sealed ``TraceDocumentV5`` into ``eval_trace_view_v2`` and ``eval_index_v1`` rows.

Rules: nothing is inferred from another field; ``environment.action_executed`` /
``action_applied`` events are authoritative for executed actions; rejected actions stay
separate; fields with no evidence are absent. Events attach to a policy call only through
explicit evidence (the event's span, that span's parent/causes, a causing event, or a
payload ``call_index`` that matches a model-call span). Everything else becomes its own
``environment`` step.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from ..models.document import TraceDocumentV5
from ..models.eval_view import (
    EvalActionV2,
    EvalArtifactV2,
    EvalCostV2,
    EvalEnvironmentV2,
    EvalEvaluatorV2,
    EvalExecutedActionV2,
    EvalFrameV2,
    EvalIndexCostV1,
    EvalIndexRowV1,
    EvalIndexTaskV1,
    EvalIndexTokensV1,
    EvalIntegrityV2,
    EvalMediaV2,
    EvalRejectedActionV2,
    EvalReproduceV2,
    EvalRunV2,
    EvalStepContentV2,
    EvalStepTokensV2,
    EvalStepV2,
    EvalTaskV2,
    EvalToolCallV2,
    EvalTraceViewV2,
    EvalUsageV2,
)
from ..models.events import EventType, EventV5
from ..models.messages import PartType
from ..models.spans import SpanKind

_STATUS = {
    "completed": "completed", "failed": "failed", "interrupted": "interrupted",
}
_EXECUTED_TYPES = {str(EventType.ENV_ACTION_EXECUTED)}
_PROPOSED_TYPES = {str(EventType.ENV_ACTION_PROPOSED)}


def _ms(a: str | None, b: str | None) -> int | None:
    if not a or not b:
        return None
    try:
        da = datetime.fromisoformat(a)
        db = datetime.fromisoformat(b)
    except ValueError:
        return None
    return max(0, round((db - da).total_seconds() * 1000))


def _kind(event: EventV5) -> str:
    payload = event.payload or {}
    return str(payload.get("kind") or payload.get("native_event_type") or event.event_type)


def _inner(payload: dict[str, Any]) -> dict[str, Any]:
    inner = payload.get("payload")
    return inner if isinstance(inner, dict) else {}


def _is_executed(e: EventV5) -> bool:
    return str(e.event_type) in _EXECUTED_TYPES or _kind(e) == "action_applied"


def _is_rejected(e: EventV5) -> bool:
    return _kind(e) in {"action_rejected", "environment.action_rejected"} or str(e.status) == "rejected"


def _is_proposed(e: EventV5) -> bool:
    return str(e.event_type) in _PROPOSED_TYPES or _kind(e) == "action_proposed"


def _name(payload: dict[str, Any]) -> str | None:
    for key in ("action", "name"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _turn(payload: dict[str, Any]) -> int | None:
    for key in ("turn", "step_index"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _json_args(raw: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {"_raw": raw}
    return parsed if isinstance(parsed, dict) else {"_value": parsed}


def _join(texts: Iterable[str]) -> str | None:
    joined = "\n\n".join(t for t in texts if t)
    return joined or None


def project_eval_view(
    document: TraceDocumentV5,
    *,
    environment: str,
    media: Iterable[EvalMediaV2] | None = None,
    evaluator: EvalEvaluatorV2 | None = None,
    cost: EvalCostV2 | None = None,
    task: EvalTaskV2 | None = None,
    ext: dict[str, Any] | None = None,
) -> EvalTraceViewV2:
    """Build the viewer contract from a sealed V5 document."""

    if environment not in {"craftax", "dungeongrid", "terminal", "browser", "none"}:
        raise ValueError(
            f"code=unknown_renderer stage=project_eval_view environment={environment!r}: "
            "expected craftax|dungeongrid|terminal|browser|none"
        )
    started = document.lifecycle.started_at
    messages = {m.message_id: m for m in document.messages}
    spans = {s.span_id: s for s in document.spans}
    model_spans = [s for s in document.spans if str(s.span_kind) == str(SpanKind.MODEL_CALL)]
    model_ids = {s.span_id for s in model_spans}
    by_call_index: dict[tuple[str, int], str] = {}
    for s in model_spans:
        idx = (s.detail or {}).get("call_index")
        if isinstance(idx, int):
            by_call_index[(s.session_id, idx)] = s.span_id
    events_by_id = {e.event_id: e for e in document.events}

    def owner(event: EventV5, depth: int = 0) -> str | None:
        if event.span_id in model_ids:
            return event.span_id
        idx = (event.payload or {}).get("call_index")
        if isinstance(idx, int) and (event.session_id, idx) in by_call_index:
            return by_call_index[(event.session_id, idx)]
        span = spans.get(event.span_id or "")
        if span is not None:
            if span.parent_span_id in model_ids:
                return span.parent_span_id
            for cause in span.caused_by_span_ids:
                if cause in model_ids:
                    return cause
        if depth < 4:
            for cause_id in event.caused_by_event_ids:
                cause = events_by_id.get(cause_id)
                if cause is not None:
                    found = owner(cause, depth + 1)
                    if found:
                        return found
        return None

    attached: dict[str, list[EventV5]] = {}
    orphans: dict[str, list[EventV5]] = {}
    for event in document.events:
        if not (_is_executed(event) or _is_rejected(event) or _is_proposed(event)
                or _kind(event) in {"achievement_unlocked", "reward_delta"}
                or str(event.event_type) == str(EventType.ENV_REWARD)):
            continue
        target = owner(event)
        if target:
            attached.setdefault(target, []).append(event)
        else:
            orphans.setdefault(event.span_id or event.turn_id or event.event_id, []).append(event)

    def action_block(events: list[EventV5], extra_planned: list[str] | None = None):
        planned: list[str] = list(extra_planned or [])
        executed: list[EvalExecutedActionV2] = []
        rejected: list[EvalRejectedActionV2] = []
        reward: float | None = None
        achievements: list[str] = []
        for e in events:
            payload = e.payload or {}
            inner = _inner(payload)
            if _is_executed(e):
                name = _name(payload)
                if name:
                    executed.append(EvalExecutedActionV2(name=name, turn=_turn(payload)))
            elif _is_rejected(e):
                name = _name(payload) or "unspecified"
                reason = payload.get("reason") or inner.get("reason")
                rejected.append(EvalRejectedActionV2(name=name, reason=reason if isinstance(reason, str) and reason else None))
            elif _is_proposed(e):
                many = payload.get("actions")
                if isinstance(many, list):
                    planned.extend(str(x) for x in many)
                elif (proposed_name := _name(payload)) is not None:
                    planned.append(proposed_name)
            elif _kind(e) == "achievement_unlocked":
                ach = inner.get("achievement") or payload.get("achievement")
                if isinstance(ach, str) and ach:
                    achievements.append(ach)
            else:
                value = payload.get("delta", payload.get("reward"))
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    reward = (reward or 0.0) + float(value)
        action = None
        if planned or executed or rejected:
            action = EvalActionV2(
                planned=tuple(planned) if planned else None,
                executed=tuple(executed) if executed else None,
                rejected=tuple(rejected) if rejected else None,
            )
        return action, reward, tuple(achievements) or None

    rows: list[tuple[str, int, EvalStepV2]] = []  # (timestamp, order, step) pre-indexing

    def elapsed(ts: str | None) -> int | None:
        return _ms(started, ts)

    # system prompt: the first system-role message, verbatim
    system_prompt = next(
        (m.text() for m in document.messages if str(m.role) == "system" and m.text()), None
    )

    tool_result_step: dict[str, str] = {}
    for m in document.messages:
        if str(m.role) != "tool":
            continue
        results = [p for p in m.parts if str(p.type) == str(PartType.TOOL_RESULT)]
        for p in results:
            if p.tool_call_id:
                tool_result_step[p.tool_call_id] = m.message_id

    order = 0
    for span in model_spans:
        detail = span.detail or {}
        out_msgs = [messages[i] for i in span.output_message_ids if i in messages]
        parts = [p for m in out_msgs for p in m.parts]
        reasoning = _join(p.text or "" for p in parts if str(p.type) == str(PartType.REASONING))
        message = _join(p.text or "" for p in parts if str(p.type) == str(PartType.TEXT))
        obs = detail.get("observation")
        calls = tuple(
            EvalToolCallV2(
                id=p.tool_call_id or f"{span.span_id}:{n}",
                name=p.tool_name or "unknown",
                arguments=_json_args(p.arguments_json),
                result_ref=tool_result_step.get(p.tool_call_id or ""),
            )
            for n, p in enumerate(q for q in parts if str(q.type) == str(PartType.TOOL_CALL))
        )
        usage = span.usage
        tokens = None
        if usage is not None and any(
            v is not None for v in (usage.prompt_tokens, usage.completion_tokens, usage.cached_tokens)
        ):
            tokens = EvalStepTokensV2(
                input=usage.prompt_tokens, output=usage.completion_tokens, cache_read=usage.cached_tokens
            )
        planned_detail = detail.get("planned_actions")
        extra_planned = [str(x) for x in planned_detail] if isinstance(planned_detail, list) else None
        action, reward, achievements = action_block(attached.get(span.span_id, []), extra_planned)
        turns = [a.turn for a in (action.executed or ()) if a.turn is not None] if action else []
        raw = {k: detail[k] for k in ("finish_reason", "prefix_digest") if detail.get(k) is not None}
        rows.append((
            span.started_at, order,
            EvalStepV2(
                id=span.span_id, index=0, kind="policy_call",
                timestamp=span.started_at, elapsed_ms=elapsed(span.started_at),
                actor=span.actor_id if len({s.actor_id for s in model_spans}) > 1 else None,
                content=EvalStepContentV2(
                    reasoning=reasoning, message=message,
                    observation=obs if isinstance(obs, str) and obs else None,
                ),
                tokens=tokens, tool_calls=calls or None, action=action, reward=reward,
                achievements=achievements,
                turn_start=min(turns) if turns else None, turn_end=max(turns) if turns else None,
                raw=raw or None,
            ),
        ))
        order += 1

    for m in document.messages:
        if str(m.role) != "tool":
            continue
        text = _join(p.text or "" for p in m.parts if str(p.type) == str(PartType.TOOL_RESULT))
        ts = m.occurred_at or started
        rows.append((ts, order, EvalStepV2(
            id=m.message_id, index=0, kind="tool_result", timestamp=ts, elapsed_ms=elapsed(ts),
            content=EvalStepContentV2(observation=text),
        )))
        order += 1

    for key, events in orphans.items():
        action, reward, achievements = action_block(events)
        ts = events[0].occurred_at
        turns = [a.turn for a in (action.executed or ()) if a.turn is not None] if action else []
        rows.append((ts, order, EvalStepV2(
            id=key, index=0, kind="environment", timestamp=ts, elapsed_ms=elapsed(ts),
            content=EvalStepContentV2(), action=action, reward=reward, achievements=achievements,
            turn_start=min(turns) if turns else None, turn_end=max(turns) if turns else None,
        )))
        order += 1

    rows.sort(key=lambda r: (r[0], r[1]))
    steps = tuple(
        _with_index(step, i) for i, (_, _, step) in enumerate(rows, start=1)
    )
    step_by_turn: dict[int, str] = {}
    for step in steps:
        for a in (step.action.executed or ()) if step.action else ():
            if a.turn is not None:
                step_by_turn.setdefault(a.turn, step.id)

    frames = tuple(
        EvalFrameV2(
            turn=turn,
            step_id=step_by_turn.get(turn),
            state={k: v for k, v in (e.payload or {}).items() if k not in {"kind", "native_event_type"}} or None,
        )
        for e in document.events
        if str(e.event_type) == str(EventType.ENV_OBSERVATION)
        and (turn := _turn(e.payload or {})) is not None
    )

    prov, ident = document.provenance, document.identity
    model = prov.model or next((a.model for a in document.actors if a.model), None) \
        or next((s.detail.get("model") for s in model_spans if isinstance(s.detail.get("model"), str)), None)
    provider = prov.provider or next((a.provider for a in document.actors if a.provider), None)
    effort = next(
        (s.detail.get("reasoning_effort") for s in model_spans
         if isinstance(s.detail.get("reasoning_effort"), str)), None)
    u = document.usage
    usage = EvalUsageV2(
        calls=u.requests, input_tokens=u.prompt_tokens, output_tokens=u.completion_tokens,
        cache_read_tokens=u.cached_tokens,
    )
    duration = _ms(started, document.lifecycle.ended_at)
    if duration is None and u.wall_time_seconds is not None:
        duration = round(u.wall_time_seconds * 1000)

    if cost is None:
        cost = EvalCostV2(
            usd=u.cost_usd, basis="unknown",
            source="trace_v5.usage" if u.cost_usd is not None else "none",
        )
    media_list = tuple(media) if media is not None else None
    media_ids = {m.id for m in media_list or ()}
    artifacts = tuple(
        EvalArtifactV2(
            id=a.artifact_id, name=a.logical_name or a.artifact_id, media_type=a.media_type,
            bytes=a.size_bytes, media_id=a.artifact_id if a.artifact_id in media_ids else None,
        )
        for a in document.artifacts
    ) or None
    command = prov.extra.get("command") if isinstance(prov.extra, dict) else None
    reproduce = EvalReproduceV2(
        command=command if isinstance(command, str) else None,
        config_digest=prov.config_digest, container_image_digest=prov.container_image_digest,
    )
    if task is None:
        tid = ident.task_id or ident.task_instance_id
        task = EvalTaskV2(
            id=tid or document.trace_id, name=ident.task_instance_id or tid or document.trace_id,
            family=ident.benchmark or "unknown",
        )
    return EvalTraceViewV2(
        trace_id=document.trace_id,
        run_id=ident.run_id or ident.rollout_id or document.trace_id,
        task=task,
        run=EvalRunV2(
            model=model or "unknown", provider=provider, effort=effort, seed=ident.seed,
            status=_run_status(document),
            started_at=started, duration_ms=duration, usage=usage,
        ),
        cost=cost,
        integrity=EvalIntegrityV2(
            status="sealed" if document.content_digest else "provisional",
            content_digest=document.content_digest, source=prov.producer,
            capture_status=str(document.completeness.capture_status),
        ),
        system_prompt=system_prompt,
        steps=steps,
        evaluator=evaluator,
        environment=EvalEnvironmentV2(renderer=environment, version=1, frames=frames),
        media=media_list,
        artifacts=artifacts,
        reproduce=reproduce if (reproduce.command or reproduce.config_digest or reproduce.container_image_digest) else None,
        ext=ext,
    )


def _run_status(document: TraceDocumentV5) -> str:
    term = document.lifecycle.termination
    reason = (term.reason if term else "") or ""
    if "timeout" in reason or "timed_out" in reason:
        return "timed_out"
    return _STATUS.get(str(document.lifecycle.status), "interrupted")


def _with_index(step: EvalStepV2, index: int) -> EvalStepV2:
    from dataclasses import replace

    return replace(step, index=index)


def compute_media_status(view: EvalTraceViewV2) -> str:
    """``native`` when every frame resolves to a native item, ``partial`` if some, else ``none``."""

    native = {m.id for m in view.media or () if m.provenance == "native"}
    frames = view.environment.frames if view.environment else ()
    if not frames:
        return "none"
    hits = sum(1 for f in frames if f.media_id in native)
    if hits == len(frames):
        return "native"
    return "partial" if hits else "none"


def project_eval_index_row(
    view: EvalTraceViewV2, *, protocol: str | None = None, ext: dict[str, Any] | None = None
) -> dict[str, Any]:
    """One ``eval_index_v1`` run row from a view. ``protocol`` falls back to ``view.ext['protocol']``."""

    protocol = protocol or (view.ext or {}).get("protocol")
    if not isinstance(protocol, str) or not protocol:
        raise ValueError(
            f"code=missing_protocol stage=project_eval_index_row run_id={view.run_id}: "
            "pass protocol= or set ext.protocol on the view"
        )
    ev = view.evaluator
    usage = view.run.usage or EvalUsageV2()
    # eval_index_v1 requires these facts. The detailed view can retain unknowns,
    # but an index row must refuse them rather than invent experiment identity
    # or claim a measured zero-call run.
    if view.run.seed is None:
        raise ValueError(
            f"code=missing_seed stage=project_eval_index_row run_id={view.run_id}"
        )
    if usage.calls is None:
        raise ValueError(
            f"code=missing_calls stage=project_eval_index_row run_id={view.run_id}"
        )
    success = {"pass": True, "fail": False}.get(ev.verdict) if ev else None
    row = EvalIndexRowV1(
        id=view.run_id,
        task=EvalIndexTaskV1(view.task.id, view.task.name, view.task.family),
        model=view.run.model, provider=view.run.provider, effort=view.run.effort,
        seed=view.run.seed,
        protocol=protocol, status=view.run.status,
        score=ev.score if ev else None, reward=ev.reward if ev else None, success=success,
        steps=len(view.steps), calls=usage.calls,
        duration_ms=view.run.duration_ms,
        tokens=EvalIndexTokensV1(usage.input_tokens, usage.output_tokens, usage.cache_read_tokens),
        cost=EvalIndexCostV1(view.cost.usd, view.cost.basis),
        media_status=compute_media_status(view),
        flags=tuple(f.code for f in ev.flags) if ev else (),
        ext=ext,
    )
    return row.to_dict()


def to_atif(view: EvalTraceViewV2) -> dict[str, Any]:
    """Emit an ATIF v1.7 trajectory from the view (reasoning goes under ``extra.reasoning``)."""

    from ..adapters.atif import ATIF_SCHEMA_VERSION

    result_step = {s.id: s for s in view.steps if s.kind == "tool_result"}
    paired = {
        call.result_ref for s in view.steps for call in (s.tool_calls or ()) if call.result_ref in result_step
    }
    out: list[dict[str, Any]] = []

    def add(step: dict[str, Any]) -> None:
        step["step_id"] = len(out) + 1
        out.append(step)

    if view.system_prompt:
        add({"source": "system", "message": view.system_prompt, "timestamp": view.run.started_at})
    if view.task.prompt:
        add({"source": "user", "message": view.task.prompt, "timestamp": view.run.started_at})
    for s in view.steps:
        if s.id in paired:
            continue
        c = s.content
        entry: dict[str, Any] = {"message": c.message or c.observation or ""}
        if s.timestamp:
            entry["timestamp"] = s.timestamp
        extra: dict[str, Any] = {"view_step_id": s.id, "kind": s.kind}
        if s.kind == "policy_call":
            entry["source"] = "agent"
            entry["model_name"] = view.run.model
            if c.reasoning:
                extra["reasoning"] = c.reasoning
            if c.observation:
                extra["observation"] = c.observation
            if s.tokens:
                m = {k: v for k, v in (("prompt_tokens", s.tokens.input),
                                       ("completion_tokens", s.tokens.output),
                                       ("cached_tokens", s.tokens.cache_read)) if v is not None}
                if m:
                    entry["metrics"] = m
        elif s.kind == "user":
            entry["source"] = "user"
        else:
            entry["source"] = "system"
        if s.tool_calls:
            entry["tool_calls"] = [
                {"tool_call_id": t.id, "function_name": t.name, "arguments": t.arguments or {}}
                for t in s.tool_calls
            ]
            results = []
            for t in s.tool_calls:
                rs = result_step.get(t.result_ref or "")
                text = t.result if t.result is not None else (
                    (rs.content.observation or rs.content.message) if rs else None)
                if text is not None:
                    results.append({"source_call_id": t.id, "content": text})
            if results:
                entry["observation"] = {"results": results}
        if s.action:
            extra["action"] = s.action.to_dict()
        if s.reward is not None:
            extra["reward"] = s.reward
        if s.achievements:
            extra["achievements"] = list(s.achievements)
        entry["extra"] = extra
        add(entry)

    usage = view.run.usage
    final: dict[str, Any] = {"total_steps": len(out)}
    if usage:
        for key, val in (("total_prompt_tokens", usage.input_tokens),
                         ("total_completion_tokens", usage.output_tokens),
                         ("total_cached_tokens", usage.cache_read_tokens)):
            if val is not None:
                final[key] = val
    if view.cost.usd is not None:
        final["total_cost_usd"] = view.cost.usd
    agent: dict[str, Any] = {"name": view.run.model, "version": "unknown", "model_name": view.run.model}
    return {
        "schema_version": ATIF_SCHEMA_VERSION,
        "session_id": view.run_id,
        "agent": agent,
        "steps": out,
        "final_metrics": final,
    }


__all__ = [
    "compute_media_status", "project_eval_index_row", "project_eval_view", "to_atif",
]
