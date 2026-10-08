"""Read-only session filtering for a multi-lane annotation job.

This is a *tool view*, not a newly sealed trace. It keeps the source identity and
digest so selectors still refer to the original authority, while preventing a
scoped annotator from inspecting another rollout lane.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from ..models.document import TraceDocumentV5
from ..models.selectors import SelectorKind, TraceSelectorV1


def scoped_tool_document(
    document: TraceDocumentV5,
    session_ids: Sequence[str],
    span_ids: Sequence[str] = (),
) -> TraceDocumentV5:
    allowed = frozenset(session_ids)
    selected_spans = frozenset(span_ids)
    if not allowed and not selected_spans:
        return document
    known = {item.session_id for item in document.sessions}
    if not allowed <= known:
        raise ValueError(f"unknown annotation sessions: {sorted(allowed - known)}")
    known_spans = {item.span_id for item in document.spans}
    if not selected_spans <= known_spans:
        raise ValueError(f"unknown annotation spans: {sorted(selected_spans - known_spans)}")
    spans = tuple(
        item for item in document.spans
        if (not allowed or item.session_id in allowed)
        and (not selected_spans or item.span_id in selected_spans)
    )
    visible_sessions = {item.session_id for item in spans} if selected_spans else allowed
    sessions = tuple(item for item in document.sessions if item.session_id in visible_sessions)
    actors = {item.actor_id for item in sessions}
    visible_message_ids = {
        message_id
        for span in spans
        for message_id in (*span.input_message_ids, *span.output_message_ids)
    }
    messages = tuple(
        item for item in document.messages
        if item.session_id in visible_sessions
        and (not selected_spans or item.message_id in visible_message_ids)
    )
    events = tuple(
        item for item in document.events
        if item.session_id in visible_sessions
        and (not selected_spans or item.span_id in selected_spans or item.message_id in visible_message_ids)
    )
    return replace(
        document,
        actors=tuple(item for item in document.actors if item.actor_id in actors),
        sessions=sessions,
        messages=messages,
        branches=tuple(item for item in document.branches if item.session_id in visible_sessions),
        spans=spans,
        events=events,
        errors=tuple(item for item in document.errors if item.session_id in visible_sessions and not selected_spans),
        artifacts=(),
        aliases=(),
        links=(),
        coordination=None,
        extensions={},
    )


def selector_in_sessions(
    document: TraceDocumentV5,
    selector: TraceSelectorV1,
    session_ids: Sequence[str],
    span_ids: Sequence[str] = (),
) -> bool:
    allowed = frozenset(session_ids)
    selected_spans = frozenset(span_ids)
    if not allowed and not selected_spans:
        return True
    visible = scoped_tool_document(document, session_ids, span_ids) if selected_spans else document
    kind, entity_id = str(selector.kind), selector.entity_id
    if kind == SelectorKind.TRACE:
        return True
    if kind == SelectorKind.SESSION:
        return entity_id in {item.session_id for item in visible.sessions}
    if kind == SelectorKind.MESSAGE:
        item = visible.message(entity_id or "")
        return item is not None and (not allowed or item.session_id in allowed)
    if kind == SelectorKind.SPAN:
        item = visible.span(entity_id or "")
        return item is not None and (not allowed or item.session_id in allowed)
    if kind == SelectorKind.EVENT:
        item = visible.event(entity_id or "")
        return item is not None and (not allowed or item.session_id in allowed)
    if kind == SelectorKind.PART:
        return any(
            (not allowed or item.session_id in allowed) and any(part.part_id == selector.part_id for part in item.parts)
            for item in visible.messages
        )
    if kind == SelectorKind.ACTOR:
        return any(item.actor_id == entity_id and (not allowed or item.session_id in allowed) for item in visible.sessions)
    # Other selector types are intentionally not exposed by the scoped tool view.
    return False


__all__ = ["scoped_tool_document", "selector_in_sessions"]
