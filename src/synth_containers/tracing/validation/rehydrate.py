"""Rehydrate sealed JSON payloads back into typed records.

Canonical serialization drops ``None`` values, so rehydration relies on dataclass
defaults rather than on the payload carrying every key. A rehydrated record must
re-digest to the digest it was stored under; ``rehydrate_trace`` checks that.
"""

from __future__ import annotations

import types
from dataclasses import MISSING, fields, is_dataclass
from typing import Any, Union, get_args, get_origin, get_type_hints

from ..canonical import content_digest
from ..models.document import TraceDocumentV5
from ..models.evidence import TraceEvidenceBundleV5


class RehydrationError(ValueError):
    """Raised when a stored payload cannot be rebuilt into its typed record."""


def build(record_type: type, payload: Any) -> Any:
    """Construct ``record_type`` from a canonical JSON payload."""

    if payload is None:
        return None
    if not is_dataclass(record_type):
        return payload
    if not isinstance(payload, dict):
        raise RehydrationError(f"expected an object for {record_type.__name__}")
    hints = get_type_hints(record_type)
    kwargs: dict[str, Any] = {}
    for field_info in fields(record_type):
        annotation = hints[field_info.name]
        if field_info.name in payload:
            kwargs[field_info.name] = _convert(annotation, payload[field_info.name])
            continue
        # Canonical serialization drops nulls, so an absent nullable field means None.
        # A field that has no default and cannot be None is genuinely missing.
        if _allows_none(annotation):
            kwargs[field_info.name] = None
    missing = [
        field_info.name
        for field_info in fields(record_type)
        if field_info.name not in kwargs
        and field_info.default is MISSING
        and field_info.default_factory is MISSING  # type: ignore[misc]
    ]
    if missing:
        raise RehydrationError(
            f"{record_type.__name__} payload is missing required fields: {', '.join(missing)}"
        )
    return record_type(**kwargs)


def _allows_none(annotation: Any) -> bool:
    return get_origin(annotation) in (Union, types.UnionType) and type(None) in get_args(annotation)


def _convert(annotation: Any, value: Any) -> Any:
    if value is None:
        return None
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        for candidate in get_args(annotation):
            if candidate is type(None):
                continue
            if is_dataclass(candidate):
                return build(candidate, value)
        return value
    if origin is tuple:
        args = get_args(annotation)
        if not args:
            return tuple(value)
        element = args[0]
        return tuple(_convert(element, item) for item in value)
    if origin is list:
        args = get_args(annotation)
        element = args[0] if args else Any
        return [_convert(element, item) for item in value]
    if origin is dict:
        return dict(value)
    if is_dataclass(annotation):
        return build(annotation, value)
    return value


def trace_document_from_payload(payload: dict[str, Any]) -> TraceDocumentV5:
    document = build(TraceDocumentV5, payload)
    if not isinstance(document, TraceDocumentV5):
        raise RehydrationError("payload did not rehydrate into a TraceDocumentV5")
    return document


def evidence_bundle_from_payload(payload: dict[str, Any]) -> TraceEvidenceBundleV5:
    bundle = build(TraceEvidenceBundleV5, payload)
    if not isinstance(bundle, TraceEvidenceBundleV5):
        raise RehydrationError("payload did not rehydrate into a TraceEvidenceBundleV5")
    return bundle


def rehydrate_trace(payload: dict[str, Any]) -> TraceDocumentV5:
    """Rehydrate and prove the record round-trips to its stored digest."""

    document = trace_document_from_payload(payload)
    expected = payload.get("content_digest")
    actual = content_digest(document)
    if expected and expected != actual:
        raise RehydrationError(
            f"rehydrated trace digest mismatch: stored {expected}, recomputed {actual}"
        )
    return document


__all__ = [
    "RehydrationError",
    "build",
    "evidence_bundle_from_payload",
    "rehydrate_trace",
    "trace_document_from_payload",
]
