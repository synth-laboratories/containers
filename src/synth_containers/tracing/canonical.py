"""Canonical serialization, content digests, and deterministic identity for Trace V5.

This module is the frozen contract every other tracing module depends on. Changing
any function here changes every trace digest, so treat it as a versioned surface.

Canonical form:

- UTF-8 JSON, keys sorted, no insignificant whitespace, ``NaN``/``Infinity`` rejected.
- ``None`` values are dropped; empty lists/objects are preserved.
- Digests are ``sha256:<hex>`` over the canonical bytes.
- A record's own ``content_digest`` field is excluded while digesting that record.
- Deterministic record IDs are ``<prefix>_<16 hex chars>`` over a typed identity key.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass, replace
from datetime import UTC, datetime
from typing import Any, TypeVar

from pydantic import BaseModel
from synth_containers.serde import jsonable


RecordT = TypeVar("RecordT")


CANONICAL_JSON_PROFILE = "synth.canonical-json.v1"
DIGEST_ALGORITHM = "sha256"
DIGEST_PREFIX = f"{DIGEST_ALGORITHM}:"
RECORD_ID_HEX_LENGTH = 16

_CONTENT_DIGEST_FIELD = "content_digest"


def _strip_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_nulls(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_strip_nulls(item) for item in value]
    return value


def canonical_payload(value: Any) -> Any:
    """Return the JSON-safe, null-stripped payload used for digests and storage."""

    _reject_unordered(value)
    return _strip_nulls(jsonable(value))


def _reject_unordered(value: Any, *, path: str = "$") -> None:
    if isinstance(value, (set, frozenset)):
        raise TypeError(f"unordered container is not canonical at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_unordered(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_unordered(item, path=f"{path}[{index}]")
    elif is_dataclass(value):
        for item in fields(value):
            _reject_unordered(getattr(value, item.name), path=f"{path}.{item.name}")
    elif isinstance(value, BaseModel):
        _reject_unordered(value.model_dump(mode="python"), path=path)


def canonical_bytes(value: Any) -> bytes:
    """Serialize to the canonical byte form used by every V5 digest."""

    return json.dumps(
        canonical_payload(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_text(value: Any) -> str:
    return canonical_bytes(value).decode("utf-8")


def readable_json(value: Any) -> str:
    """Human-operable rendering of the same payload; never used for digests."""

    return json.dumps(canonical_payload(value), sort_keys=True, indent=2, ensure_ascii=False)


def content_digest(value: Any) -> str:
    """Digest of a value, ignoring any ``content_digest`` field it already carries."""

    payload = canonical_payload(value)
    if isinstance(payload, dict):
        payload = {key: item for key, item in payload.items() if key != _CONTENT_DIGEST_FIELD}
    return DIGEST_PREFIX + hashlib.sha256(canonical_bytes(payload)).hexdigest()


def seal_record(record: RecordT) -> RecordT:
    """Return a copy of a dataclass record whose ``content_digest`` matches its content."""

    return replace(record, content_digest=content_digest(record))


def bytes_digest(payload: bytes) -> str:
    return DIGEST_PREFIX + hashlib.sha256(payload).hexdigest()


def text_digest(payload: str) -> str:
    return bytes_digest(payload.encode("utf-8"))


def digest_hex(digest: str) -> str:
    """Return the bare hex of a ``sha256:`` digest."""

    if not digest.startswith(DIGEST_PREFIX):
        raise ValueError(f"expected a {DIGEST_PREFIX} digest, got {digest!r}")
    return digest[len(DIGEST_PREFIX) :]


def short_digest(digest: str, *, length: int = 12) -> str:
    return digest_hex(digest)[:length]


def record_id(prefix: str, *, kind: str, scope: tuple[str, ...] = (), key: Any = None) -> str:
    """Mint a deterministic record ID from a typed identity key.

    The same ``(kind, scope, key)`` always produces the same ID, which is what makes
    repeated ingestion of one capture idempotent.
    """

    clean_prefix = prefix.strip().strip("_")
    if not clean_prefix:
        raise ValueError("record id prefix must not be empty")
    identity = {"kind": kind, "scope": list(scope), "key": key}
    hexed = hashlib.sha256(canonical_bytes(identity)).hexdigest()
    return f"{clean_prefix}_{hexed[:RECORD_ID_HEX_LENGTH]}"


def utc_now() -> str:
    """RFC3339 UTC timestamp with microsecond precision."""

    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def to_rfc3339(value: float | int | datetime) -> str:
    if isinstance(value, datetime):
        moment = value.astimezone(UTC)
    else:
        moment = datetime.fromtimestamp(float(value), tz=UTC)
    return moment.isoformat(timespec="microseconds").replace("+00:00", "Z")


__all__ = [
    "CANONICAL_JSON_PROFILE",
    "DIGEST_ALGORITHM",
    "DIGEST_PREFIX",
    "RECORD_ID_HEX_LENGTH",
    "bytes_digest",
    "canonical_bytes",
    "canonical_payload",
    "canonical_text",
    "content_digest",
    "digest_hex",
    "readable_json",
    "record_id",
    "seal_record",
    "short_digest",
    "text_digest",
    "to_rfc3339",
    "utc_now",
]
