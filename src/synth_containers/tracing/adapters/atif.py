"""Harbor ATIF 1.5/1.7 import and projection with explicit loss accounting."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Mapping

from ..canonical import bytes_digest, canonical_bytes, canonical_text, record_id
from ..models.actors import ActorKind, ActorV5, CoverageState, SessionCoverageV5, SessionV5
from ..models.completeness import CaptureStatus, TraceCompletenessV5, TraceLifecycleV5, TraceStatus
from ..models.document import TraceCaptureSummaryV5, TraceDocumentV5
from ..models.events import EventOrderV1, EventType, EventV5
from ..models.identity import AliasNamespace, AliasV1, TraceIdentityV5, TraceKind, TraceProvenanceV5
from ..models.messages import MessageNodeV5, MessagePartV5, MessageRole, PartType
from ..models.spans import SpanKind, SpanStatus, SpanV5, UsageProvenance, UsageV5
from ..models.tokens import TokenCaptureProvenance, TokenCaptureV5, TokenSequenceRefV1


ATIF_SCHEMA_VERSION = "ATIF-v1.7"
_SUPPORTED_VERSIONS = {
    "ATIF-v1.5": "ATIF-v1.5",
    "ATIF-v1.6": "ATIF-v1.6",
    "ATIF-v1.7": "ATIF-v1.7",
    "1.5": "ATIF-v1.5",
    "1.6": "ATIF-v1.6",
    "1.7": "ATIF-v1.7",
}
_IMPORT_TIME = "1970-01-01T00:00:00Z"
_ATIF_SOURCE = {
    MessageRole.SYSTEM.value: "system",
    MessageRole.USER.value: "user",
    MessageRole.ASSISTANT.value: "agent",
}


def export_atif(document: TraceDocumentV5) -> dict[str, Any]:
    """Project a sealed V5 graph into a Harbor-valid ATIF-v1.7 trajectory."""

    if not document.content_digest:
        raise ValueError("ATIF export requires a sealed Trace V5 document")
    agents = [actor for actor in document.actors if str(actor.kind) == ActorKind.AGENT]
    root_actor = next((actor for actor in agents if not actor.parent_actor_id), None)
    if root_actor is None and agents:
        root_actor = agents[0]
    parent_by_actor: dict[str, str | None] = {
        actor.actor_id: actor.parent_actor_id for actor in agents
    }
    if root_actor is not None:
        for actor in agents:
            if actor.actor_id != root_actor.actor_id and not parent_by_actor[actor.actor_id]:
                parent_by_actor[actor.actor_id] = root_actor.actor_id

    children: dict[str, list[ActorV5]] = {}
    for actor in agents:
        parent_id = parent_by_actor.get(actor.actor_id)
        if parent_id:
            children.setdefault(parent_id, []).append(actor)

    run_session_id = (
        document.identity.run_id
        or document.identity.correlation_id
        or (document.sessions[0].session_id if document.sessions else document.trace_id)
    )

    def build(actor: ActorV5 | None, *, root: bool) -> dict[str, Any]:
        actor_id = actor.actor_id if actor is not None else None
        direct_children = children.get(actor_id or "", ())
        steps = _export_actor_steps(document, actor_id=actor_id, root=root)
        for child in direct_children:
            steps.append(
                {
                    "step_id": 0,
                    "source": "system",
                    "message": f"Delegated to {child.display_name}",
                    "observation": {
                        "results": [
                            {
                                "subagent_trajectory_ref": [
                                    {
                                        "trajectory_id": child.actor_id,
                                        "session_id": run_session_id,
                                        "extra": {"synth_actor_id": child.actor_id},
                                    }
                                ],
                                "extra": {"event": "subagent.delegation"},
                            }
                        ]
                    },
                    "extra": {
                        "synth_event": "subagent.delegation",
                        "synth_child_actor_id": child.actor_id,
                    },
                }
            )
        if not steps:
            steps.append(
                {
                    "step_id": 0,
                    "source": "system",
                    "message": "",
                    "extra": {"synth_placeholder": "ATIF requires at least one step"},
                }
            )
        steps.sort(key=_export_step_sort_key)
        for index, step in enumerate(steps, start=1):
            step["step_id"] = index
            step.pop("_sort", None)

        actor_name = actor.display_name if actor is not None else document.provenance.producer
        actor_version = (
            str((actor.metadata if actor is not None else {}).get("version") or "")
            or document.provenance.producer_version
            or "unknown"
        )
        agent: dict[str, Any] = {
            "name": actor_name or "agent",
            "version": actor_version,
            "extra": {
                "synth_actor_id": actor_id,
                "synth_actor_kind": str(actor.kind) if actor is not None else "agent",
                "synth_role": actor.role if actor is not None else "",
                "harness": actor.harness if actor is not None else document.provenance.harness,
                "runtime": actor.runtime if actor is not None else None,
                "provider": actor.provider if actor is not None else document.provenance.provider,
            },
        }
        model_name = (
            actor.model if actor is not None else None
        ) or document.provenance.model
        if model_name:
            agent["model_name"] = model_name
        tool_definitions = (
            (actor.metadata if actor is not None else {}).get("tool_definitions")
        )
        if isinstance(tool_definitions, list):
            agent["tool_definitions"] = tool_definitions
        agent["extra"] = _compact(agent["extra"])

        subagents = [build(child, root=False) for child in direct_children]
        trajectory_id = document.trace_id if root else str(actor_id)
        result: dict[str, Any] = {
            "schema_version": ATIF_SCHEMA_VERSION,
            "session_id": run_session_id,
            "trajectory_id": trajectory_id,
            "agent": agent,
            "steps": steps,
            "notes": (
                "Trace V5 projection; raw provider frames and evidence bundles "
                "remain in the source bundle."
            ),
            "final_metrics": _export_final_metrics(document, len(steps)) if root else {
                "total_steps": len(steps)
            },
            "extra": {
                "synth_trace_digest": document.content_digest,
                "synth_trace_schema": document.schema_version,
                "synth_actor_id": actor_id,
                "synth_entity_ids": [
                    *[message.message_id for message in document.messages],
                    *[event.event_id for event in document.events],
                    *[span.span_id for span in document.spans],
                ],
                "projection_losses": [
                    "raw provider frames and evidence bundles are not represented in ATIF",
                    "non-ATIF event types are represented as system observations",
                    "the V5 message DAG is linearized into ATIF step order",
                ],
            },
        }
        if subagents:
            result["subagent_trajectories"] = subagents
        return result

    return build(root_actor, root=True)


def _export_actor_steps(
    document: TraceDocumentV5,
    *,
    actor_id: str | None,
    root: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    non_agent_actor_ids = {
        actor.actor_id
        for actor in document.actors
        if str(actor.kind) != ActorKind.AGENT
    }
    for index, message in enumerate(document.messages):
        if actor_id is not None and message.sender_actor_id != actor_id:
            if not (root and message.sender_actor_id in non_agent_actor_ids):
                continue
        step = _message_step(document, message)
        step["_sort"] = (message.occurred_at or "", 0, index)
        records.append(step)
    for index, event in enumerate(document.events):
        if actor_id is not None and event.actor_id != actor_id:
            if not (root and event.actor_id in non_agent_actor_ids):
                continue
        records.append(
            {
                "step_id": 0,
                "timestamp": event.occurred_at,
                "source": "system",
                "message": str(event.event_type),
                "observation": {
                    "results": [
                        {
                            "content": canonical_text(event.payload),
                            "extra": {
                                "synth_event_id": event.event_id,
                                "synth_event_type": str(event.event_type),
                            },
                        }
                    ]
                },
                "extra": {
                    "synth_entity_id": event.event_id,
                    "synth_entity_kind": "event",
                    "synth_actor_id": event.actor_id,
                    "synth_session_id": event.session_id,
                    "synth_span_id": event.span_id,
                    "synth_status": str(event.status),
                },
                "_sort": (event.occurred_at or "", 1, index),
            }
        )
    _coalesce_tool_results(records)
    return records


def _coalesce_tool_results(records: list[dict[str, Any]]) -> None:
    """Put ATIF observations on the same step as their cited tool call.

    V5 permits a tool result to arrive in a later message. ATIF v1.7 validates
    ``source_call_id`` only against calls in that same step, so projection moves
    the observation to the call step. A standalone result remains observable but
    drops the invalid ATIF link and preserves it in ``extra`` as a declared loss.
    """

    call_steps: dict[str, dict[str, Any]] = {}
    for step in records:
        for call in step.get("tool_calls") or ():
            call_id = str(call.get("tool_call_id") or "")
            if call_id:
                call_steps[call_id] = step
    for step in records:
        observation = step.get("observation")
        if not isinstance(observation, dict):
            continue
        remaining: list[dict[str, Any]] = []
        for raw_result in observation.get("results") or ():
            result = dict(raw_result)
            call_id = str(result.get("source_call_id") or "")
            destination = call_steps.get(call_id)
            if call_id and destination is not None and destination is not step:
                target = destination.setdefault("observation", {"results": []})
                target.setdefault("results", []).append(result)
                continue
            if call_id and destination is None:
                result.pop("source_call_id", None)
                result["extra"] = {
                    **dict(result.get("extra") or {}),
                    "synth_unresolved_source_call_id": call_id,
                    "synth_projection_loss": "ATIF requires a same-step tool call",
                }
            remaining.append(result)
        if remaining:
            observation["results"] = remaining
        else:
            step.pop("observation", None)


def _message_step(document: TraceDocumentV5, message: MessageNodeV5) -> dict[str, Any]:
    source = _ATIF_SOURCE.get(str(message.role), "system")
    text_parts = [
        part.text
        for part in message.parts
        if str(part.type) == PartType.TEXT and part.text is not None
    ]
    structured_parts = [
        canonical_text(part.structured)
        for part in message.parts
        if part.structured is not None and str(part.type) not in {
            PartType.TOOL_CALL,
            PartType.TOOL_RESULT,
        }
    ]
    rendered_message = "\n".join([*text_parts, *structured_parts])
    step: dict[str, Any] = {
        "step_id": 0,
        "timestamp": message.occurred_at,
        "source": source,
        "message": rendered_message,
        "extra": {
            "synth_entity_id": message.message_id,
            "synth_entity_kind": "message",
            "synth_actor_id": message.sender_actor_id,
            "synth_session_id": message.session_id,
            "synth_turn_id": message.turn_id,
            "synth_thread_id": message.thread_id,
            "synth_predecessor_message_ids": list(message.predecessor_message_ids),
        },
    }
    if source != "agent":
        tool_results = _tool_results(message)
        if tool_results:
            step["observation"] = {"results": tool_results}
        return _compact(step)

    reasoning = "\n".join(
        part.text or ""
        for part in message.parts
        if str(part.type) == PartType.REASONING and part.text
    )
    tool_calls = _tool_calls(message)
    tool_results = _tool_results(message)
    model_spans = [
        span
        for span in document.spans
        if message.message_id in span.output_message_ids
        and str(span.span_kind) == SpanKind.MODEL_CALL
    ]
    if reasoning:
        step["reasoning_content"] = reasoning
    if tool_calls:
        step["tool_calls"] = tool_calls
    if tool_results:
        step["observation"] = {"results": tool_results}
    if model_spans:
        usage = UsageV5()
        for span in model_spans:
            if span.usage is not None:
                usage = usage.merged(span.usage)
        metrics = _export_metrics(usage, model_spans[0].token_capture)
        if metrics:
            step["metrics"] = metrics
        step["llm_call_count"] = len(model_spans)
        model_name = (
            model_spans[-1].detail.get("model")
            or (document.actor(message.sender_actor_id).model if document.actor(message.sender_actor_id) else None)
            or document.provenance.model
        )
        if model_name:
            step["model_name"] = str(model_name)
        effort = model_spans[-1].detail.get("reasoning_effort")
        if isinstance(effort, (str, float)):
            step["reasoning_effort"] = effort
        step["extra"]["synth_span_ids"] = [span.span_id for span in model_spans]
    return _compact(step)


def _tool_calls(message: MessageNodeV5) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for part in message.parts:
        if str(part.type) != PartType.TOOL_CALL:
            continue
        arguments: dict[str, Any] = {}
        if part.arguments_json:
            try:
                decoded = json.loads(part.arguments_json)
                arguments = decoded if isinstance(decoded, dict) else {"value": decoded}
            except json.JSONDecodeError:
                arguments = {"raw": part.arguments_json}
        calls.append(
            {
                "tool_call_id": part.tool_call_id or part.part_id,
                "function_name": part.tool_name or "unknown_tool",
                "arguments": arguments,
                "extra": {"synth_part_id": part.part_id},
            }
        )
    return calls


def _tool_results(message: MessageNodeV5) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for part in message.parts:
        if str(part.type) != PartType.TOOL_RESULT:
            continue
        content = part.text
        if content is None and part.structured is not None:
            content = canonical_text(part.structured)
        results.append(
            _compact(
                {
                    "source_call_id": part.tool_call_id,
                    "content": content or "",
                    "extra": {
                        "synth_part_id": part.part_id,
                        "is_error": part.is_error,
                    },
                }
            )
        )
    return results


def _export_metrics(
    usage: UsageV5,
    token_capture: TokenCaptureV5 | None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "cached_tokens": usage.cached_tokens,
        "cost_usd": usage.cost_usd,
        "extra": _compact(
            {
                "reasoning_tokens": usage.reasoning_tokens,
                "cache_write_tokens": usage.cache_write_tokens,
                "total_tokens": usage.total_tokens,
                "usage_provenance": str(usage.provenance),
            }
        ),
    }
    if token_capture is not None:
        if token_capture.prompt and token_capture.prompt.token_ids:
            metrics["prompt_token_ids"] = list(token_capture.prompt.token_ids)
        if token_capture.completion and token_capture.completion.token_ids:
            metrics["completion_token_ids"] = list(token_capture.completion.token_ids)
        if token_capture.completion_logprobs:
            metrics["logprobs"] = list(token_capture.completion_logprobs)
    return _compact(metrics)


def _export_final_metrics(document: TraceDocumentV5, step_count: int) -> dict[str, Any]:
    return _compact(
        {
            "total_prompt_tokens": document.usage.prompt_tokens,
            "total_completion_tokens": document.usage.completion_tokens,
            "total_cached_tokens": document.usage.cached_tokens,
            "total_cost_usd": document.usage.cost_usd,
            "total_steps": step_count,
            "extra": _compact(
                {
                    "reasoning_tokens": document.usage.reasoning_tokens,
                    "cache_write_tokens": document.usage.cache_write_tokens,
                    "total_tokens": document.usage.total_tokens,
                    "requests": document.usage.requests,
                    "wall_time_seconds": document.usage.wall_time_seconds,
                    "usage_provenance": str(document.usage.provenance),
                }
            ),
        }
    )


def _export_step_sort_key(step: Mapping[str, Any]) -> tuple[Any, ...]:
    marker = step.get("_sort")
    if isinstance(marker, tuple):
        timestamp, kind, index = marker
        return (0 if timestamp else 1, timestamp, kind, index)
    return (2, "", 0, str((step.get("extra") or {}).get("synth_child_actor_id") or ""))


def inspect_atif_import(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and inventory a foreign ATIF document before canonical assembly."""

    version = _normalized_version(payload.get("schema_version"))
    _validate_trajectory(payload, version=version, embedded=False, path="$")
    steps = tuple(item for item in payload["steps"] if isinstance(item, Mapping))
    return {
        "source_format": version,
        "source_digest": bytes_digest(canonical_bytes(payload)),
        "session_id": str(payload.get("session_id") or ""),
        "trajectory_id": str(payload.get("trajectory_id") or ""),
        "steps": steps,
        "subagent_count": _subagent_count(payload),
        "losses": (
            "foreign ATIF has no raw provider transport frames",
            "ATIF step order is preserved but provider-native event timing may be unavailable",
        ),
    }


def _normalized_version(value: Any) -> str:
    version = str(value or "ATIF-v1.5")
    normalized = _SUPPORTED_VERSIONS.get(version)
    if normalized is None:
        raise ValueError(f"unsupported ATIF version: {version}")
    return normalized


def _validate_trajectory(
    payload: Mapping[str, Any],
    *,
    version: str,
    embedded: bool,
    path: str,
) -> None:
    agent = payload.get("agent")
    if not isinstance(agent, Mapping):
        raise ValueError(f"{path}.agent must be an object")
    if not str(agent.get("name") or ""):
        raise ValueError(f"{path}.agent.name is required")
    if not str(agent.get("version") or ""):
        raise ValueError(f"{path}.agent.version is required")
    if embedded and version == ATIF_SCHEMA_VERSION and not str(payload.get("trajectory_id") or ""):
        raise ValueError(f"{path}.trajectory_id is required for embedded subagents")
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"{path}.steps must be a non-empty array")
    for index, raw in enumerate(steps):
        step_path = f"{path}.steps[{index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{step_path} must be an object")
        if raw.get("step_id") != index + 1:
            raise ValueError(
                f"{step_path}.step_id: expected {index + 1} (sequential from 1), "
                f"got {raw.get('step_id')!r}"
            )
        source = str(raw.get("source") or "")
        if source not in {"system", "user", "agent"}:
            raise ValueError(f"{step_path}.source must be system, user, or agent")
        message = raw.get("message")
        if not isinstance(message, (str, list)):
            raise ValueError(f"{step_path}.message must be a string or content-part array")
        if isinstance(message, list):
            if version == "ATIF-v1.5":
                raise ValueError(f"{step_path}.message content parts require ATIF-v1.6+")
            _validate_content_parts(message, f"{step_path}.message")
        timestamp = raw.get("timestamp")
        if timestamp is not None:
            try:
                datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError(f"{step_path}.timestamp is not ISO 8601") from error
        if source != "agent":
            for field in (
                "model_name",
                "reasoning_effort",
                "reasoning_content",
                "tool_calls",
                "metrics",
            ):
                if raw.get(field) is not None:
                    raise ValueError(f"{step_path}.{field} is agent-only")
        if raw.get("llm_call_count") == 0 and source == "agent":
            if raw.get("metrics") is not None or raw.get("reasoning_content") is not None:
                raise ValueError(
                    f"{step_path} deterministic dispatch cannot carry metrics or reasoning"
                )
        call_ids: set[str] = set()
        tool_calls = raw.get("tool_calls")
        if tool_calls is not None:
            if not isinstance(tool_calls, list):
                raise ValueError(f"{step_path}.tool_calls must be an array")
            for tool_index, call in enumerate(tool_calls):
                if not isinstance(call, Mapping):
                    raise ValueError(f"{step_path}.tool_calls[{tool_index}] must be an object")
                call_id = str(call.get("tool_call_id") or "")
                if not call_id or not str(call.get("function_name") or ""):
                    raise ValueError(
                        f"{step_path}.tool_calls[{tool_index}] requires IDs and function_name"
                    )
                if not isinstance(call.get("arguments"), Mapping):
                    raise ValueError(
                        f"{step_path}.tool_calls[{tool_index}].arguments must be an object"
                    )
                if call_id in call_ids:
                    raise ValueError(f"{step_path} contains duplicate tool_call_id {call_id!r}")
                call_ids.add(call_id)
        observation = raw.get("observation")
        if observation is not None:
            if not isinstance(observation, Mapping) or not isinstance(
                observation.get("results"), list
            ):
                raise ValueError(f"{step_path}.observation.results must be an array")
            for result_index, result in enumerate(observation["results"]):
                result_path = f"{step_path}.observation.results[{result_index}]"
                if not isinstance(result, Mapping):
                    raise ValueError(f"{result_path} must be an object")
                source_call_id = result.get("source_call_id")
                if source_call_id is not None and str(source_call_id) not in call_ids:
                    raise ValueError(
                        f"{result_path}.source_call_id does not name a tool call in this step"
                    )
                content = result.get("content")
                if isinstance(content, list):
                    if version == "ATIF-v1.5":
                        raise ValueError(f"{result_path}.content parts require ATIF-v1.6+")
                    _validate_content_parts(content, f"{result_path}.content")
                elif content is not None and not isinstance(content, str):
                    raise ValueError(f"{result_path}.content must be a string or parts")
                refs = result.get("subagent_trajectory_ref")
                if refs is not None:
                    if version != ATIF_SCHEMA_VERSION or not isinstance(refs, list):
                        raise ValueError(f"{result_path}.subagent_trajectory_ref requires v1.7")
                    for ref in refs:
                        if not isinstance(ref, Mapping) or not (
                            ref.get("trajectory_id") or ref.get("trajectory_path")
                        ):
                            raise ValueError(
                                f"{result_path} contains an unresolvable subagent reference"
                            )
    subagents = payload.get("subagent_trajectories")
    if subagents is None:
        return
    if version != ATIF_SCHEMA_VERSION or not isinstance(subagents, list):
        raise ValueError(f"{path}.subagent_trajectories requires ATIF-v1.7")
    seen: set[str] = set()
    for index, child in enumerate(subagents):
        child_path = f"{path}.subagent_trajectories[{index}]"
        if not isinstance(child, Mapping):
            raise ValueError(f"{child_path} must be an object")
        child_version = _normalized_version(child.get("schema_version"))
        child_id = str(child.get("trajectory_id") or "")
        if not child_id or child_id in seen:
            raise ValueError(f"{child_path}.trajectory_id must be present and unique")
        seen.add(child_id)
        _validate_trajectory(child, version=child_version, embedded=True, path=child_path)


def _validate_content_parts(parts: list[Any], path: str) -> None:
    for index, part in enumerate(parts):
        if not isinstance(part, Mapping):
            raise ValueError(f"{path}[{index}] must be an object")
        kind = str(part.get("type") or "")
        if kind == "text":
            if not isinstance(part.get("text"), str) or part.get("source") is not None:
                raise ValueError(f"{path}[{index}] is an invalid text part")
        elif kind == "image":
            source = part.get("source")
            if not isinstance(source, Mapping) or not str(source.get("path") or ""):
                raise ValueError(f"{path}[{index}] is an invalid image part")
            if str(source.get("media_type") or "") not in {
                "image/jpeg",
                "image/png",
                "image/gif",
                "image/webp",
            }:
                raise ValueError(f"{path}[{index}] has an invalid image media type")
            if part.get("text") is not None:
                raise ValueError(f"{path}[{index}] image part cannot contain text")
        else:
            raise ValueError(f"{path}[{index}].type must be text or image")


def import_atif(payload: Mapping[str, Any]) -> TraceDocumentV5:
    """Import ATIF into a sealed, deterministic, explicitly partial Trace V5."""

    from ..capture.redaction import redact_payload

    assessed = inspect_atif_import(payload)
    source_digest = str(assessed["source_digest"])
    redacted_value, redaction = redact_payload(payload)
    if not isinstance(redacted_value, dict):
        raise ValueError("ATIF import requires a JSON object")
    trace_id = record_id("trace", kind="imported_atif", key=source_digest)
    actors: list[ActorV5] = []
    sessions: list[SessionV5] = []
    messages: list[MessageNodeV5] = []
    spans: list[SpanV5] = []
    events: list[EventV5] = []
    aliases: list[AliasV1] = []
    timestamps: list[str] = []
    global_event_order = 0

    def visit(
        trajectory: Mapping[str, Any],
        *,
        parent_actor_id: str | None,
        parent_session_id: str | None,
        path: tuple[int, ...],
    ) -> tuple[str, str]:
        nonlocal global_event_order
        agent = trajectory["agent"]
        native_trajectory_id = str(
            trajectory.get("trajectory_id") or trajectory.get("session_id") or ".".join(map(str, path))
        )
        actor_id = record_id(
            "actor",
            kind="atif_actor",
            scope=(trace_id,),
            key={"trajectory_id": native_trajectory_id, "path": path},
        )
        native_session_id = str(trajectory.get("session_id") or "")
        session_id = record_id(
            "sess",
            kind="atif_session",
            scope=(trace_id, actor_id),
            key={"session_id": native_session_id, "path": path},
        )
        actor_aliases = (
            AliasV1(
                namespace="harbor.atif.trajectory",
                value=native_trajectory_id,
                target_id=actor_id,
                target_kind="actor",
            ),
        )
        actors.append(
            ActorV5(
                actor_id=actor_id,
                kind=ActorKind.AGENT,
                display_name=str(agent.get("name") or "ATIF agent"),
                role="agent",
                parent_actor_id=parent_actor_id,
                harness=str(agent.get("name") or ""),
                model=str(agent.get("model_name") or "") or None,
                aliases=actor_aliases,
                metadata=_compact(
                    {
                        "version": str(agent.get("version") or ""),
                        "tool_definitions": agent.get("tool_definitions"),
                        "extra": agent.get("extra"),
                    }
                ),
            ).sealed()
        )
        previous: tuple[str, ...] = ()
        local_timestamps: list[str] = []
        for index, raw in enumerate(trajectory["steps"]):
            timestamp = str(raw.get("timestamp") or _IMPORT_TIME)
            timestamps.append(timestamp)
            local_timestamps.append(timestamp)
            native_step_id = str(raw["step_id"])
            message_id = record_id(
                "msg",
                kind="atif_message",
                scope=(trace_id, actor_id),
                key=native_step_id,
            )
            parts = _import_message_parts(message_id, raw)
            role = {
                "system": MessageRole.SYSTEM,
                "user": MessageRole.USER,
                "agent": MessageRole.ASSISTANT,
            }[str(raw["source"])]
            message = MessageNodeV5(
                message_id=message_id,
                role=role,
                parts=parts,
                sender_actor_id=actor_id,
                session_id=session_id,
                predecessor_message_ids=previous,
                occurred_at=timestamp,
                aliases=(
                    AliasV1(
                        namespace="harbor.atif.step",
                        value=native_step_id,
                        target_id=message_id,
                        target_kind="message",
                    ),
                ),
                metadata=_compact(
                    {
                        "atif_trajectory_id": native_trajectory_id,
                        "is_copied_context": raw.get("is_copied_context"),
                        "extra": raw.get("extra"),
                    }
                ),
            ).sealed()
            messages.append(message)
            previous = (message_id,)

            if str(raw["source"]) == "agent":
                llm_call_count = raw.get("llm_call_count")
                metrics = raw.get("metrics")
                usage = _import_usage(metrics)
                token_capture = _import_token_capture(metrics)
                is_dispatch = llm_call_count == 0
                span_id = record_id(
                    "span",
                    kind="atif_step",
                    scope=(trace_id, actor_id),
                    key=native_step_id,
                )
                spans.append(
                    SpanV5(
                        span_id=span_id,
                        span_kind=SpanKind.APPLICATION if is_dispatch else SpanKind.MODEL_CALL,
                        actor_id=actor_id,
                        session_id=session_id,
                        started_at=timestamp,
                        ended_at=timestamp,
                        status=SpanStatus.OK,
                        output_message_ids=(message_id,),
                        usage=None if is_dispatch else usage,
                        token_capture=None if is_dispatch else token_capture,
                        detail=_compact(
                            {
                                "model": raw.get("model_name") or agent.get("model_name"),
                                "reasoning_effort": raw.get("reasoning_effort"),
                                "llm_call_count": llm_call_count,
                                "atif_step_id": raw["step_id"],
                                "aggregated_calls": bool(
                                    isinstance(llm_call_count, int) and llm_call_count > 1
                                ),
                            }
                        ),
                        aliases=(
                            AliasV1(
                                namespace="harbor.atif.step",
                                value=native_step_id,
                                target_id=span_id,
                                target_kind="span",
                            ),
                        ),
                    ).sealed()
                )

            for tool_index, call in enumerate(raw.get("tool_calls") or ()):
                global_event_order += 1
                call_id = str(call["tool_call_id"])
                event_id = record_id(
                    "evt",
                    kind="atif_tool_call",
                    scope=(trace_id, actor_id),
                    key={"step": native_step_id, "call": call_id, "index": tool_index},
                )
                events.append(
                    EventV5(
                        event_id=event_id,
                        event_type=EventType.TOOL_CALL_PROPOSED,
                        actor_id=actor_id,
                        session_id=session_id,
                        occurred_at=timestamp,
                        message_id=message_id,
                        order=EventOrderV1(chronological_sequence=global_event_order),
                        payload={
                            "tool_call_id": call_id,
                            "function_name": call["function_name"],
                            "arguments": dict(call["arguments"]),
                            "extra": call.get("extra"),
                        },
                        aliases=(
                            AliasV1(
                                namespace="harbor.atif.tool_call",
                                value=call_id,
                                target_id=event_id,
                                target_kind="event",
                            ),
                        ),
                    ).sealed()
                )
            observation = raw.get("observation")
            for result_index, result in enumerate(
                observation.get("results") if isinstance(observation, Mapping) else ()
            ):
                global_event_order += 1
                source_call_id = str(result.get("source_call_id") or "")
                event_id = record_id(
                    "evt",
                    kind="atif_observation",
                    scope=(trace_id, actor_id),
                    key={
                        "step": native_step_id,
                        "call": source_call_id,
                        "index": result_index,
                    },
                )
                events.append(
                    EventV5(
                        event_id=event_id,
                        event_type=(
                            EventType.TOOL_RESULT
                            if source_call_id
                            else EventType.ENV_OBSERVATION
                        ),
                        actor_id=actor_id,
                        session_id=session_id,
                        occurred_at=timestamp,
                        message_id=message_id,
                        order=EventOrderV1(chronological_sequence=global_event_order),
                        payload={
                            "source_call_id": source_call_id or None,
                            "content": result.get("content"),
                            "subagent_trajectory_ref": result.get(
                                "subagent_trajectory_ref"
                            ),
                            "extra": result.get("extra"),
                        },
                        aliases=(
                            AliasV1(
                                namespace="harbor.atif.tool_call",
                                value=source_call_id,
                                target_id=event_id,
                                target_kind="event",
                            ),
                        )
                        if source_call_id
                        else (),
                    ).sealed()
                )

        started = min(local_timestamps) if local_timestamps else _IMPORT_TIME
        ended = max(local_timestamps) if local_timestamps else started
        has_metrics = any(
            isinstance(step.get("metrics"), Mapping) for step in trajectory["steps"]
        )
        coverage = SessionCoverageV5(
            model_calls=CoverageState.PARTIAL,
            agent_events=CoverageState.PARTIAL,
            environment_events=CoverageState.PARTIAL,
            tool_events=CoverageState.PARTIAL,
            usage=CoverageState.COMPLETE if has_metrics else CoverageState.NOT_CAPTURED,
            raw_provider=CoverageState.UNAVAILABLE,
            reasons=("imported from ATIF; raw provider transport unavailable",),
        )
        sessions.append(
            SessionV5(
                session_id=session_id,
                actor_id=actor_id,
                started_at=started,
                ended_at=ended,
                parent_session_id=parent_session_id,
                status="completed",
                harness=str(agent.get("name") or ""),
                provider=str((agent.get("extra") or {}).get("provider") or "") or None,
                coverage=coverage,
                aliases=(
                    AliasV1(
                        namespace="harbor.atif.session",
                        value=native_session_id,
                        target_id=session_id,
                        target_kind="session",
                    ),
                )
                if native_session_id
                else (),
            ).sealed()
        )
        for child_index, child in enumerate(trajectory.get("subagent_trajectories") or ()):
            visit(
                child,
                parent_actor_id=actor_id,
                parent_session_id=session_id,
                path=(*path, child_index),
            )
        return actor_id, session_id

    visit(redacted_value, parent_actor_id=None, parent_session_id=None, path=(0,))
    started = min(timestamps) if timestamps else _IMPORT_TIME
    ended = max(timestamps) if timestamps else started
    root_session_native = str(redacted_value.get("session_id") or "")
    native_trajectory = str(redacted_value.get("trajectory_id") or "")
    if native_trajectory:
        aliases.append(
            AliasV1(
                namespace="harbor.atif.trajectory",
                value=native_trajectory,
                target_id=trace_id,
            )
        )
    if root_session_native:
        aliases.append(
            AliasV1(
                namespace=AliasNamespace.CORRELATION,
                value=root_session_native,
                target_id=trace_id,
            )
        )
    usage = _import_total_usage(redacted_value)
    return TraceDocumentV5(
        trace_id=trace_id,
        trace_kind=(
            TraceKind.WORKFLOW_RUN
            if assessed["subagent_count"]
            else TraceKind.AGENT_ROLLOUT
        ),
        identity=TraceIdentityV5(
            run_id=root_session_native or None,
            correlation_id=native_trajectory or root_session_native or None,
        ),
        lifecycle=TraceLifecycleV5(
            status=TraceStatus.COMPLETED,
            started_at=started,
            ended_at=ended,
        ),
        capture=TraceCaptureSummaryV5(
            capture_id=record_id(
                "cap",
                kind="imported",
                scope=(trace_id,),
                key=source_digest,
            ),
            binding_id="imported",
            binding_digest=source_digest,
            capture_profile="imported_atif",
            interception="none",
            mode="disabled",
        ),
        provenance=TraceProvenanceV5(
            producer="synth_containers.tracing.adapters.atif",
            producer_version="2",
            source_format=str(assessed["source_format"]),
            captured_at=_IMPORT_TIME,
            model=str((redacted_value.get("agent") or {}).get("model_name") or "") or None,
            harness=str((redacted_value.get("agent") or {}).get("name") or "") or None,
            transformation_chain=("atif_import@2",),
            extra={
                "source_digest": source_digest,
                "redaction": redaction.to_dict(),
            },
        ),
        completeness=TraceCompletenessV5(
            capture_status=CaptureStatus.PARTIAL,
            terminal_event_observed=bool(messages or events),
            model_calls=CoverageState.PARTIAL,
            raw_provider=CoverageState.UNAVAILABLE,
            agent_events=CoverageState.PARTIAL,
            environment_events=CoverageState.PARTIAL,
            tool_events=CoverageState.PARTIAL,
            usage=(
                CoverageState.COMPLETE
                if usage.provenance != UsageProvenance.UNAVAILABLE
                else CoverageState.NOT_CAPTURED
            ),
            reasons=tuple(assessed["losses"]),
        ),
        actors=tuple(actors),
        sessions=tuple(sessions),
        messages=tuple(messages),
        spans=tuple(spans),
        events=tuple(events),
        usage=usage,
        aliases=tuple(aliases),
        extensions={
            "atif_native": redacted_value,
            "atif_source_digest": source_digest,
        },
    ).sealed()


def _import_message_parts(
    message_id: str,
    step: Mapping[str, Any],
) -> tuple[MessagePartV5, ...]:
    parts: list[MessagePartV5] = []
    message = step["message"]
    if isinstance(message, str):
        parts.append(
            MessagePartV5(
                part_id=record_id("part", kind="atif_text", scope=(message_id,), key=0),
                type=PartType.TEXT,
                text=message,
            )
        )
    else:
        for index, part in enumerate(message):
            if str(part["type"]) == "text":
                parts.append(
                    MessagePartV5(
                        part_id=record_id(
                            "part", kind="atif_text", scope=(message_id,), key=index
                        ),
                        type=PartType.TEXT,
                        text=str(part["text"]),
                    )
                )
            else:
                source = dict(part["source"])
                parts.append(
                    MessagePartV5(
                        part_id=record_id(
                            "part", kind="atif_image", scope=(message_id,), key=index
                        ),
                        type=PartType.MEDIA,
                        media_type=str(source["media_type"]),
                        structured={"source": source},
                    )
                )
    reasoning = step.get("reasoning_content")
    if isinstance(reasoning, str):
        parts.append(
            MessagePartV5(
                part_id=record_id(
                    "part", kind="atif_reasoning", scope=(message_id,), key=len(parts)
                ),
                type=PartType.REASONING,
                text=reasoning,
            )
        )
    for index, call in enumerate(step.get("tool_calls") or ()):
        parts.append(
            MessagePartV5(
                part_id=record_id(
                    "part",
                    kind="atif_tool_call",
                    scope=(message_id,),
                    key={"index": index, "id": call["tool_call_id"]},
                ),
                type=PartType.TOOL_CALL,
                tool_call_id=str(call["tool_call_id"]),
                tool_name=str(call["function_name"]),
                arguments_json=canonical_text(call["arguments"]),
                structured={"extra": call.get("extra")} if call.get("extra") else None,
            )
        )
    return tuple(parts)


def _import_usage(metrics: Any) -> UsageV5 | None:
    if not isinstance(metrics, Mapping):
        return None
    extra = metrics.get("extra")
    detail = extra if isinstance(extra, Mapping) else {}
    return UsageV5(
        provenance=UsageProvenance.OBSERVED_HARNESS,
        prompt_tokens=_optional_int(metrics.get("prompt_tokens")),
        completion_tokens=_optional_int(metrics.get("completion_tokens")),
        reasoning_tokens=_optional_int(detail.get("reasoning_tokens")),
        cached_tokens=_optional_int(metrics.get("cached_tokens")),
        cache_write_tokens=_optional_int(detail.get("cache_write_tokens")),
        total_tokens=_optional_int(detail.get("total_tokens")),
        cost_usd=_optional_float(metrics.get("cost_usd")),
    )


def _import_token_capture(metrics: Any) -> TokenCaptureV5 | None:
    if not isinstance(metrics, Mapping):
        return None
    prompt_ids = metrics.get("prompt_token_ids")
    completion_ids = metrics.get("completion_token_ids")
    logprobs = metrics.get("logprobs")
    if not any(isinstance(item, list) and item for item in (prompt_ids, completion_ids, logprobs)):
        return None
    return TokenCaptureV5(
        provenance=TokenCaptureProvenance.IMPORTED,
        level="token_ids",
        prompt=(
            TokenSequenceRefV1(
                token_ids=tuple(int(item) for item in prompt_ids),
                count=len(prompt_ids),
            )
            if isinstance(prompt_ids, list)
            else None
        ),
        completion=(
            TokenSequenceRefV1(
                token_ids=tuple(int(item) for item in completion_ids),
                count=len(completion_ids),
            )
            if isinstance(completion_ids, list)
            else None
        ),
        completion_logprobs=(
            tuple(float(item) for item in logprobs)
            if isinstance(logprobs, list)
            else ()
        ),
        source_refs=("harbor.atif.metrics",),
    )


def _import_total_usage(payload: Mapping[str, Any]) -> UsageV5:
    final_metrics = payload.get("final_metrics")
    if isinstance(final_metrics, Mapping):
        extra = final_metrics.get("extra")
        detail = extra if isinstance(extra, Mapping) else {}
        return UsageV5(
            provenance=UsageProvenance.OBSERVED_HARNESS,
            prompt_tokens=_optional_int(final_metrics.get("total_prompt_tokens")),
            completion_tokens=_optional_int(final_metrics.get("total_completion_tokens")),
            reasoning_tokens=_optional_int(detail.get("reasoning_tokens")),
            cached_tokens=_optional_int(final_metrics.get("total_cached_tokens")),
            cache_write_tokens=_optional_int(detail.get("cache_write_tokens")),
            total_tokens=_optional_int(detail.get("total_tokens")),
            requests=_optional_int(detail.get("requests")),
            wall_time_seconds=_optional_float(detail.get("wall_time_seconds")),
            cost_usd=_optional_float(final_metrics.get("total_cost_usd")),
        )
    usage = UsageV5()
    observed = False
    for step in payload.get("steps") or ():
        step_usage = _import_usage(step.get("metrics"))
        if step_usage is not None:
            usage = usage.merged(step_usage)
            observed = True
    return usage if observed else UsageV5()


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _subagent_count(payload: Mapping[str, Any]) -> int:
    children = payload.get("subagent_trajectories")
    if not isinstance(children, list):
        return 0
    return len(children) + sum(
        _subagent_count(item) for item in children if isinstance(item, Mapping)
    )


def _compact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _compact(item)
            for key, item in value.items()
            if item is not None and item != {}
        }
    if isinstance(value, list):
        return [_compact(item) for item in value]
    return value


def atif_roundtrip_report(document: TraceDocumentV5) -> dict[str, Any]:
    exported = export_atif(document)
    imported = import_atif(exported)
    return {
        "trace_id": document.trace_id,
        "source_messages": len(document.messages),
        "source_events": len(document.events),
        "atif_steps": len(exported["steps"]),
        "imported_trace_digest": imported.content_digest,
        "digest": bytes_digest(canonical_bytes(exported)),
        "losses": exported["extra"]["projection_losses"],
    }


__all__ = [
    "ATIF_SCHEMA_VERSION",
    "atif_roundtrip_report",
    "export_atif",
    "inspect_atif_import",
    "import_atif",
]
