"""``TraceSelectorV1`` — stable citations into a sealed trace.

A selector resolves against exactly one sealed trace digest. If the trace digest
does not match, resolution fails rather than silently citing a different execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from synth_containers.serde import JsonDataclassMixin

from ..canonical import text_digest
from .document import TraceDocumentV5


SELECTOR_SCHEMA_VERSION = "synth.trace-selector.v1"


class SelectorKind(StrEnum):
    TRACE = "trace"
    ACTOR = "actor"
    SESSION = "session"
    BRANCH = "branch"
    SPAN = "span"
    EVENT = "event"
    MESSAGE = "message"
    PART = "part"
    ARTIFACT = "artifact"


class GroundingStatus(StrEnum):
    GROUNDED = "grounded"
    PARTIALLY_GROUNDED = "partially_grounded"
    SUMMARY_ONLY = "summary_only"
    UNINSPECTED = "uninspected"
    SOURCE_UNAVAILABLE = "source_unavailable"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class TextRangeV1(JsonDataclassMixin):
    start: int
    end: int
    unit: str = "character"


@dataclass(frozen=True, slots=True)
class TraceSelectorV1(JsonDataclassMixin):
    trace_id: str
    trace_digest: str
    kind: SelectorKind | str
    entity_id: str | None = None
    part_id: str | None = None
    json_pointer: str | None = None
    range: TextRangeV1 | None = None
    quote: str | None = None
    quote_digest: str | None = None
    entity_digest: str | None = None
    source_projection: str | None = None
    schema_version: str = SELECTOR_SCHEMA_VERSION

    def with_quote(self, quote: str) -> "TraceSelectorV1":
        from dataclasses import replace

        return replace(self, quote=quote, quote_digest=text_digest(quote))


@dataclass(frozen=True, slots=True)
class SelectorResolutionV1(JsonDataclassMixin):
    """Result of resolving one selector against a sealed document."""

    selector: TraceSelectorV1
    resolved: bool
    reason: str = ""
    entity_kind: str = ""
    entity_digest: str | None = None
    resolved_text: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


def resolve_selector(
    document: TraceDocumentV5,
    selector: TraceSelectorV1,
) -> SelectorResolutionV1:
    """Resolve a selector against a sealed trace document."""

    if not document.content_digest:
        return SelectorResolutionV1(selector=selector, resolved=False, reason="trace_not_sealed")
    if selector.trace_id != document.trace_id:
        return SelectorResolutionV1(selector=selector, resolved=False, reason="trace_id_mismatch")
    if selector.trace_digest != document.content_digest:
        return SelectorResolutionV1(
            selector=selector, resolved=False, reason="trace_digest_mismatch"
        )

    kind = str(selector.kind)
    if kind == SelectorKind.TRACE:
        return SelectorResolutionV1(
            selector=selector,
            resolved=True,
            entity_kind=kind,
            entity_digest=document.content_digest,
        )

    entity_id = selector.entity_id or ""
    if not entity_id:
        return SelectorResolutionV1(selector=selector, resolved=False, reason="entity_id_required")

    lookup = {
        SelectorKind.ACTOR.value: (document.actor, "content_digest"),
        SelectorKind.SESSION.value: (document.session, "content_digest"),
        SelectorKind.SPAN.value: (document.span, "content_digest"),
        SelectorKind.EVENT.value: (document.event, "content_digest"),
        SelectorKind.MESSAGE.value: (document.message, "content_digest"),
        SelectorKind.ARTIFACT.value: (document.artifact, "digest"),
    }
    if kind == SelectorKind.BRANCH:
        entity = next(
            (item for item in document.branches if item.branch_id == entity_id),
            None,
        )
        digest_attr = None
    elif kind == SelectorKind.PART:
        return _resolve_part(document, selector)
    elif kind in lookup:
        getter, digest_attr = lookup[kind]
        entity = getter(entity_id)
    else:
        return SelectorResolutionV1(selector=selector, resolved=False, reason="unsupported_kind")

    if entity is None:
        return SelectorResolutionV1(selector=selector, resolved=False, reason="entity_not_found")

    entity_digest = getattr(entity, digest_attr, None) if digest_attr else None
    if selector.entity_digest and entity_digest and selector.entity_digest != entity_digest:
        return SelectorResolutionV1(
            selector=selector, resolved=False, reason="entity_digest_mismatch"
        )
    return SelectorResolutionV1(
        selector=selector,
        resolved=True,
        entity_kind=kind,
        entity_digest=entity_digest,
    )


def _resolve_part(
    document: TraceDocumentV5,
    selector: TraceSelectorV1,
) -> SelectorResolutionV1:
    message = document.message(selector.entity_id or "")
    if message is None:
        return SelectorResolutionV1(selector=selector, resolved=False, reason="message_not_found")
    part = next((item for item in message.parts if item.part_id == selector.part_id), None)
    if part is None:
        return SelectorResolutionV1(selector=selector, resolved=False, reason="part_not_found")
    text = part.text or part.arguments_json or ""
    if selector.range is not None:
        text = text[selector.range.start : selector.range.end]
    if selector.quote is not None:
        expected = selector.quote_digest or text_digest(selector.quote)
        if text_digest(text) != expected and selector.quote not in (part.text or ""):
            return SelectorResolutionV1(
                selector=selector, resolved=False, reason="quote_mismatch", resolved_text=text
            )
    return SelectorResolutionV1(
        selector=selector,
        resolved=True,
        entity_kind=SelectorKind.PART.value,
        entity_digest=message.content_digest,
        resolved_text=text,
    )


def selector_for(
    document: TraceDocumentV5,
    *,
    kind: SelectorKind | str,
    entity_id: str | None = None,
    part_id: str | None = None,
    quote: str | None = None,
) -> TraceSelectorV1:
    """Build a selector bound to this document's sealed digest."""

    selector = TraceSelectorV1(
        trace_id=document.trace_id,
        trace_digest=document.content_digest,
        kind=kind,
        entity_id=entity_id,
        part_id=part_id,
    )
    if quote is not None:
        selector = selector.with_quote(quote)
    return selector


__all__ = [
    "SELECTOR_SCHEMA_VERSION",
    "GroundingStatus",
    "SelectorKind",
    "SelectorResolutionV1",
    "TextRangeV1",
    "TraceSelectorV1",
    "resolve_selector",
    "selector_for",
]
