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

from ..models.identity import AliasNamespace, AliasV1


IMPORTER_NAME = "codex_stdout_jsonl"
IMPORTER_VERSION = "1"

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
}


@dataclass(slots=True)
class CodexImport:
    """Events and aliases discovered in a Codex JSONL stream."""

    events: list[dict[str, Any]] = field(default_factory=list)
    aliases: list[AliasV1] = field(default_factory=list)
    usage_snapshots: list[dict[str, Any]] = field(default_factory=list)
    unknown_kinds: dict[str, int] = field(default_factory=dict)
    line_count: int = 0
    malformed_lines: int = 0


def import_codex_jsonl(path: Path, *, target_id: str) -> CodexImport:
    """Read a Codex stdout JSONL file into typed application events."""

    result = CodexImport()
    if not path.exists():
        return result
    seen_threads: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
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
        message = record.get("msg") if isinstance(record.get("msg"), Mapping) else record
        kind = str(message.get("type") or record.get("type") or "")
        event_type = _EVENT_TYPE_BY_CODEX_KIND.get(kind)
        if event_type is None:
            result.unknown_kinds[kind] = result.unknown_kinds.get(kind, 0) + 1
            event_type = f"codex.{kind or 'unknown'}"
        body = {key: value for key, value in message.items() if key != "type"}
        result.events.append(
            {
                "event_type": event_type,
                "body": body,
                "codex_id": str(record.get("id") or ""),
            }
        )
        if event_type == "codex.usage_snapshot":
            result.usage_snapshots.append(dict(body))
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
    return result


__all__ = ["IMPORTER_NAME", "IMPORTER_VERSION", "CodexImport", "import_codex_jsonl"]
