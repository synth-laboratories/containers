"""Import Codex stdout JSONL as native agent events.

This is agent-event coverage, not provider model-call interception. The importer says
so explicitly: the coverage it declares is ``imported_agent_events`` and provider call
coverage stays ``not_captured`` unless a proxy also observed the traffic. Claiming
Responses/SSE coverage from JSONL alone is the overclaim this contract exists to stop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..models.coordination import InteractionKind, coordination_event_type
from ..models.identity import AliasNamespace, AliasV1


IMPORTER_NAME = "codex_stdout_jsonl"
IMPORTER_VERSION = "2"

_EVENT_TYPE_BY_CODEX_KIND = {
    "agent_message": "codex.agent_message",
    "agent_reasoning": "codex.agent_reasoning",
    "exec_command_begin": "codex.command_started",
    "exec_command_end": "codex.command_finished",
    "patch_apply_begin": "codex.file_mutation_started",
    "patch_apply_end": "codex.file_mutation_finished",
    "mcp_tool_call_begin": "codex.tool_call_started",
    "mcp_tool_call_end": "codex.tool_call_finished",
    "task_started": "codex.turn_started",
    "task_complete": "codex.turn_finished",
    "token_count": "codex.usage_snapshot",
    "thread.started": "codex.thread_started",
    "turn.started": "codex.turn_started",
    "turn.completed": "codex.turn_finished",
}

_EVENT_TYPE_BY_ITEM = {
    ("agent_message", "completed"): "codex.agent_message",
    ("reasoning", "completed"): "codex.agent_reasoning",
    ("command_execution", "started"): "codex.command_started",
    ("command_execution", "completed"): "codex.command_finished",
    ("file_change", "started"): "codex.file_mutation_started",
    ("file_change", "completed"): "codex.file_mutation_finished",
    ("mcp_tool_call", "started"): "codex.tool_call_started",
    ("mcp_tool_call", "completed"): "codex.tool_call_finished",
    ("web_search", "started"): "codex.tool_call_started",
    ("web_search", "completed"): "codex.tool_call_finished",
}
_CODEX_INTERACTION_EDGE_KINDS = frozenset(
    {
        "interaction_edge",
        "rollout.interaction_edge",
    }
)


@dataclass(slots=True)
class CodexImport:
    """Events and aliases discovered in a Codex JSONL stream."""

    events: list[dict[str, Any]] = field(default_factory=list)
    aliases: list[AliasV1] = field(default_factory=list)
    usage_snapshots: list[dict[str, Any]] = field(default_factory=list)
    unknown_kinds: dict[str, int] = field(default_factory=dict)
    line_count: int = 0
    malformed_lines: int = 0


def import_codex_jsonl(source: Path | bytes, *, target_id: str) -> CodexImport:
    """Read a Codex stdout JSONL file or secret-safe payload into typed events."""

    result = CodexImport()
    if isinstance(source, bytes):
        text = source.decode("utf-8", errors="replace")
    else:
        if not source.exists():
            return result
        text = source.read_text(encoding="utf-8", errors="replace")
    seen_threads: set[str] = set()
    seen_turns: set[str] = set()
    seen_items: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        result.line_count += 1
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            result.malformed_lines += 1
            continue
        if not isinstance(record, Mapping):
            result.malformed_lines += 1
            continue
        if record.get("_synth_redacted_malformed_jsonl") is True:
            result.malformed_lines += 1
            continue
        message = record.get("msg") if isinstance(record.get("msg"), Mapping) else record
        kind = str(message.get("type") or record.get("type") or "")
        item = record.get("item") if isinstance(record.get("item"), Mapping) else None
        phase = kind.removeprefix("item.") if kind.startswith("item.") else ""
        item_kind = str(item.get("type") or "") if item is not None else ""
        event_type = (
            _EVENT_TYPE_BY_ITEM.get((item_kind, phase))
            if item is not None
            else None
        ) or _EVENT_TYPE_BY_CODEX_KIND.get(kind)
        if event_type is None:
            result.unknown_kinds[kind] = result.unknown_kinds.get(kind, 0) + 1
            event_type = f"codex.{kind or 'unknown'}"
        body = (
            {
                **{key: value for key, value in item.items() if key != "type"},
                "item_type": item_kind,
                "phase": phase,
            }
            if item is not None
            else {key: value for key, value in message.items() if key != "type"}
        )
        if kind in _CODEX_INTERACTION_EDGE_KINDS:
            interaction = message.get("interaction")
            if not isinstance(interaction, Mapping):
                result.malformed_lines += 1
                continue
            raw_interaction_kind = interaction.get("kind")
            try:
                interaction_kind = InteractionKind(str(raw_interaction_kind))
            except ValueError:
                result.unknown_kinds[
                    f"{kind}:{raw_interaction_kind}"
                ] = result.unknown_kinds.get(
                    f"{kind}:{raw_interaction_kind}",
                    0,
                ) + 1
                continue
            source = interaction.get("source")
            target = interaction.get("target")
            if (
                isinstance(source, Mapping)
                and source.get("basis") == "raw_source"
                and isinstance(target, Mapping)
                and target.get("basis") == "raw_source"
            ):
                event_type = coordination_event_type(interaction_kind)
                body = {"interaction": dict(interaction)}
            else:
                event_type = "codex.interaction_edge"
                body = {
                    "interaction": dict(interaction),
                    "coordination_projection": "native_identifiers_unresolved",
                }
        codex_id = str(
            (item or {}).get("id")
            or record.get("id")
            or message.get("id")
            or ""
        )
        result.events.append(
            {
                "event_type": event_type,
                "body": body,
                "codex_id": codex_id,
                "native_kind": kind,
            }
        )
        usage = record.get("usage") or message.get("usage")
        if event_type in {"codex.usage_snapshot", "codex.turn_finished"} and isinstance(
            usage, Mapping
        ):
            result.usage_snapshots.append(dict(usage))
        thread_id = str(record.get("thread_id") or message.get("thread_id") or "")
        if thread_id and thread_id not in seen_threads:
            seen_threads.add(thread_id)
            result.aliases.append(
                AliasV1(
                    namespace=AliasNamespace.CODEX_THREAD,
                    value=thread_id,
                    target_id=target_id,
                    target_kind="session",
                )
            )
        turn_id = str(record.get("turn_id") or message.get("turn_id") or "")
        if turn_id and turn_id not in seen_turns:
            seen_turns.add(turn_id)
            result.aliases.append(
                AliasV1(
                    namespace=AliasNamespace.CODEX_TURN,
                    value=turn_id,
                    target_id=target_id,
                    target_kind="session",
                )
            )
        item_id = str((item or {}).get("id") or "")
        if item_id and item_id not in seen_items:
            seen_items.add(item_id)
            result.aliases.append(
                AliasV1(
                    namespace=AliasNamespace.CODEX_ITEM,
                    value=item_id,
                    target_id=target_id,
                    target_kind="session",
                )
            )
    return result


__all__ = ["IMPORTER_NAME", "IMPORTER_VERSION", "CodexImport", "import_codex_jsonl"]
