"""``TraceSelectorV1`` — stable citations into a sealed trace.

A selector resolves against exactly one sealed trace digest. If the trace digest
does not match, resolution fails rather than silently citing a different execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from synth_containers.serde import JsonDataclassMixin

from ..canonical import canonical_payload, canonical_text, content_digest, text_digest
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
        return _finish_resolution(
            selector,
            entity=document,
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

    entity_digest = (
        getattr(entity, digest_attr, None)
        if digest_attr
        else content_digest(entity)
    )
    return _finish_resolution(
        selector,
        entity=entity,
        entity_kind=kind,
        entity_digest=entity_digest,
        default_text=entity.text() if kind == SelectorKind.MESSAGE else None,
    )


def _resolve_part(
    document: TraceDocumentV5,
    selector: TraceSelectorV1,
) -> SelectorResolutionV1:
    message = document.message(selector.entity_id or "")
    if message is None:
        return SelectorResolutionV1(selector=selector, resolved=False, reason="message_not_found")
    if not selector.part_id:
        return SelectorResolutionV1(selector=selector, resolved=False, reason="part_id_required")
    part = next((item for item in message.parts if item.part_id == selector.part_id), None)
    if part is None:
        return SelectorResolutionV1(selector=selector, resolved=False, reason="part_not_found")
    return _finish_resolution(
        selector=selector,
        entity=part,
        entity_kind=SelectorKind.PART.value,
        entity_digest=content_digest(part),
        default_text=part.text or part.arguments_json or "",
    )


def _finish_resolution(
    selector: TraceSelectorV1,
    *,
    entity: Any,
    entity_kind: str,
    entity_digest: str,
    default_text: str | None = None,
) -> SelectorResolutionV1:
    if selector.entity_digest is not None and selector.entity_digest != entity_digest:
        return SelectorResolutionV1(
            selector=selector,
            resolved=False,
            reason="entity_digest_mismatch",
            entity_kind=entity_kind,
            entity_digest=entity_digest,
        )

    text = default_text
    if selector.json_pointer is not None:
        pointer_ok, pointed, reason = _resolve_json_pointer(
            canonical_payload(entity), selector.json_pointer
        )
        if not pointer_ok:
            return SelectorResolutionV1(
                selector=selector,
                resolved=False,
                reason=reason,
                entity_kind=entity_kind,
                entity_digest=entity_digest,
            )
        text = pointed if isinstance(pointed, str) else canonical_text(pointed)
    elif text is None and (
        selector.range is not None
        or selector.quote is not None
        or selector.quote_digest is not None
    ):
        text = canonical_text(entity)

    if selector.range is not None:
        if selector.range.unit != "character":
            return SelectorResolutionV1(
                selector=selector,
                resolved=False,
                reason="range_unit_unsupported",
                entity_kind=entity_kind,
                entity_digest=entity_digest,
            )
        assert text is not None
        if (
            selector.range.start < 0
            or selector.range.end < selector.range.start
            or selector.range.end > len(text)
        ):
            return SelectorResolutionV1(
                selector=selector,
                resolved=False,
                reason="range_invalid",
                entity_kind=entity_kind,
                entity_digest=entity_digest,
                resolved_text=text,
            )
        text = text[selector.range.start : selector.range.end]

    if selector.quote is not None:
        quote_digest = text_digest(selector.quote)
        if (
            selector.quote_digest is not None
            and selector.quote_digest != quote_digest
        ):
            return SelectorResolutionV1(
                selector=selector,
                resolved=False,
                reason="quote_digest_mismatch",
                entity_kind=entity_kind,
                entity_digest=entity_digest,
                resolved_text=text,
            )
        assert text is not None
        exact_quote_required = selector.range is not None or selector.json_pointer is not None
        if (
            (exact_quote_required and text != selector.quote)
            or (not exact_quote_required and selector.quote not in text)
        ):
            return SelectorResolutionV1(
                selector=selector,
                resolved=False,
                reason="quote_mismatch",
                entity_kind=entity_kind,
                entity_digest=entity_digest,
                resolved_text=text,
            )
        text = selector.quote
    elif selector.quote_digest is not None:
        assert text is not None
        if text_digest(text) != selector.quote_digest:
            return SelectorResolutionV1(
                selector=selector,
                resolved=False,
                reason="quote_digest_mismatch",
                entity_kind=entity_kind,
                entity_digest=entity_digest,
                resolved_text=text,
            )

    return SelectorResolutionV1(
        selector=selector,
        resolved=True,
        entity_kind=entity_kind,
        entity_digest=entity_digest,
        resolved_text=text,
    )


def _resolve_json_pointer(value: Any, pointer: str) -> tuple[bool, Any, str]:
    if pointer == "":
        return True, value, ""
    if not pointer.startswith("/"):
        return False, None, "json_pointer_invalid"
    current = value
    for raw_token in pointer.split("/")[1:]:
        token = _decode_pointer_token(raw_token)
        if token is None:
            return False, None, "json_pointer_invalid"
        if isinstance(current, dict):
            if token not in current:
                return False, None, "json_pointer_not_found"
            current = current[token]
        elif isinstance(current, list):
            if not token.isdigit() or (len(token) > 1 and token.startswith("0")):
                return False, None, "json_pointer_not_found"
            index = int(token)
            if index >= len(current):
                return False, None, "json_pointer_not_found"
            current = current[index]
        else:
            return False, None, "json_pointer_not_found"
    return True, current, ""


def _decode_pointer_token(token: str) -> str | None:
    decoded: list[str] = []
    index = 0
    while index < len(token):
        character = token[index]
        if character != "~":
            decoded.append(character)
            index += 1
            continue
        if index + 1 >= len(token) or token[index + 1] not in {"0", "1"}:
            return None
        decoded.append("~" if token[index + 1] == "0" else "/")
        index += 2
    return "".join(decoded)


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
