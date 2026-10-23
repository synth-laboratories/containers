"""Scoped, unsigned replay positions; these are never execution/auth grants."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from typing import Annotated, Iterable, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

Identity = Annotated[
    str, StringConstraints(min_length=1, max_length=512, pattern=r"^[^\x00-\x1f\x7f]+$")
]


class StreamCursorError(ValueError):
    def __init__(self, code: str, *, status_code: int = 409) -> None:
        super().__init__(code)
        self.code = code
        self.status_code = status_code


class StreamScopeV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    stream_id: Identity
    subject_id_field: Identity
    subject_id: Identity
    kinds_digest: Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]


class StreamCursorV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal["synth.stream-cursor.v1"]
    scope: StreamScopeV1
    after: Annotated[int, Field(strict=True, ge=0)]
    chain_head: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

    def encode(self) -> str:
        raw = self.model_dump_json().encode()
        token = "sc1." + base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
        if len(token) > 4096:
            raise StreamCursorError("invalid_stream_cursor", status_code=422)
        return token

    @classmethod
    def decode(cls, token: str) -> StreamCursorV1:
        if (
            not isinstance(token, str)
            or len(token) > 4096
            or not re.fullmatch(r"sc1\.[A-Za-z0-9_-]+", token)
        ):
            raise StreamCursorError("invalid_stream_cursor", status_code=422)
        try:
            encoded = token[4:].encode("ascii")
            raw = base64.b64decode(
                encoded + b"=" * (-len(encoded) % 4), altchars=b"-_", validate=True
            )
            payload = json.loads(raw, object_pairs_hook=_unique_object)
            if (
                not isinstance(payload, dict)
                or payload.get("schema_version") != "synth.stream-cursor.v1"
            ):
                raise ValueError("cursor schema required")
            return cls.model_validate(payload)
        except (ValueError, UnicodeError, binascii.Error, ValidationError) as error:
            raise StreamCursorError("invalid_stream_cursor", status_code=422) from error


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate cursor field")
        result[key] = value
    return result


def stream_scope(
    stream_id: str,
    subject_id_field: str,
    subject_id: str,
    kinds: Iterable[str] | None,
) -> StreamScopeV1:
    allowed = None if kinds is None else sorted(set(kinds))
    raw = json.dumps(
        {"kinds": allowed}, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return StreamScopeV1(
        stream_id=stream_id,
        subject_id_field=subject_id_field,
        subject_id=subject_id,
        kinds_digest="sha256:" + hashlib.sha256(raw).hexdigest(),
    )


def normalize_stream_kinds(kinds: Iterable[str] | None) -> frozenset[str] | None:
    if kinds is None:
        return None
    result = set()
    for index, kind in enumerate(kinds):
        if index >= 256 or not isinstance(kind, str) or not 1 <= len(kind) <= 512:
            raise StreamCursorError("invalid_stream_filter", status_code=422)
        result.add(kind)
    return frozenset(result)


def resume_after(token: str, scope: StreamScopeV1) -> int:
    cursor = StreamCursorV1.decode(token)
    if cursor.scope != scope:
        raise StreamCursorError("stream_cursor_scope_mismatch")
    return cursor.after


class StreamWatermarkV1(BaseModel):
    """Transport progress over a filtered source prefix, not a native event/seal."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal["synth.stream-watermark.v1"]
    cursor: StreamCursorV1
    high_water: Annotated[int, Field(strict=True, ge=0)]
    closed: bool
    durability: Literal["memory", "local_fsync"]
    writer_state: Literal["available", "recovery_required"]

    @model_validator(mode="after")
    def validate_high_water(self) -> StreamWatermarkV1:
        if self.cursor.after > self.high_water:
            raise ValueError("Stream watermark cannot exceed observed source history")
        return self
