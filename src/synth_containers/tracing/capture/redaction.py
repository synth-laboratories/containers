"""Credential stripping and secret scanning applied before anything is persisted.

Redaction runs at capture ingress, not at export. A credential-bearing header never
reaches a spool segment, so no later projection can leak one. Sealing fails closed
when the scan still finds a secret shape in a body that is about to be written.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from synth_containers.serde import JsonDataclassMixin


REDACTION_PROFILE = "strict_headers_and_secrets"
REDACTED = "<redacted>"

DENIED_HEADERS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "api-key",
        "x-api-key",
        "x-goog-api-key",
        "openai-api-key",
        "anthropic-api-key",
        "cookie",
        "set-cookie",
        "x-auth-token",
        "x-reb-evaluator-token",
    }
)

ALLOWED_HEADERS = frozenset(
    {
        "accept",
        "content-type",
        "content-length",
        "content-encoding",
        "user-agent",
        "x-request-id",
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "openai-processing-ms",
        "openai-version",
    }
)

# Correlation headers carry execution topology, never credentials, so they survive
# redaction. Any denied header still wins: `x-reb-evaluator-token` is dropped.
CORRELATION_HEADER_PREFIXES = ("x-synth-trace-", "x-reb-score-")
CORRELATION_HEADERS = frozenset(
    {"traceparent", "tracestate", "baggage", "x-synth-call-correlation-id"}
)

DENIED_BODY_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "access_token",
        "refresh_token",
        "client_secret",
        "cookie",
        "password",
        "secret",
        "token",
        "bearer",
    }
)

_DENIED_BODY_KEY_SUFFIXES = (
    "_access_token",
    "_refresh_token",
    "_client_secret",
    "_api_key",
    "_apikey",
    "_authorization",
    "_password",
    "_secret",
    "_token",
    "_bearer",
    "_cookie",
)

_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9._\-]{16,}")),
    ("groq_key", re.compile(r"\bgsk_[A-Za-z0-9]{16,}")),
    ("google_key", re.compile(r"\bAIza[0-9A-Za-z._\-]{20,}")),
    ("tinker_key", re.compile(r"\btk-[A-Za-z0-9._\-]{16,}")),
    ("synth_key", re.compile(r"\bsk_(?:live|test|prod|dev)_[A-Za-z0-9._\-]{12,}")),
    ("trace_capability", re.compile(r"\bsk_trace_[A-Za-z0-9._\-]{16,}")),
)


class RedactionError(RuntimeError):
    """Raised when a body about to be persisted still contains a secret shape."""


@dataclass(frozen=True, slots=True)
class RedactionReportV1(JsonDataclassMixin):
    profile: str = REDACTION_PROFILE
    removed_headers: tuple[str, ...] = ()
    redacted_body_keys: tuple[str, ...] = ()
    matched_patterns: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def merged(self, other: "RedactionReportV1") -> "RedactionReportV1":
        return RedactionReportV1(
            profile=self.profile,
            removed_headers=tuple(sorted(set(self.removed_headers) | set(other.removed_headers))),
            redacted_body_keys=tuple(
                sorted(set(self.redacted_body_keys) | set(other.redacted_body_keys))
            ),
            matched_patterns=tuple(
                sorted(set(self.matched_patterns) | set(other.matched_patterns))
            ),
            metadata={**self.metadata, **other.metadata},
        )


def redact_headers(headers: Mapping[str, str]) -> tuple[dict[str, str], RedactionReportV1]:
    """Keep only allowlisted headers; every denied header is dropped, never masked."""

    kept: dict[str, str] = {}
    removed: list[str] = []
    for name, value in headers.items():
        lowered = name.lower()
        if lowered in DENIED_HEADERS or not _header_allowed(lowered):
            removed.append(lowered)
            continue
        kept[lowered] = value
    return kept, RedactionReportV1(removed_headers=tuple(sorted(set(removed))))


def _header_allowed(lowered: str) -> bool:
    return (
        lowered in ALLOWED_HEADERS
        or lowered in CORRELATION_HEADERS
        or lowered.startswith(CORRELATION_HEADER_PREFIXES)
    )


def scrub_text(text: str) -> tuple[str, tuple[str, ...]]:
    """Replace known secret shapes in free text and report which patterns matched."""

    matched: list[str] = []
    result = text
    for name, pattern in _SECRET_PATTERNS:
        if pattern.search(result):
            matched.append(name)
            result = pattern.sub(REDACTED, result)
    return result, tuple(matched)


def redact_payload(value: Any) -> tuple[Any, RedactionReportV1]:
    """Recursively redact a JSON-shaped payload for persistence."""

    keys: list[str] = []
    patterns: list[str] = []

    def visit(node: Any) -> Any:
        if isinstance(node, Mapping):
            output: dict[str, Any] = {}
            for key, item in node.items():
                normalized_key = _normalize_body_key(str(key))
                if _body_key_denied(normalized_key):
                    keys.append(normalized_key)
                    output[str(key)] = REDACTED
                else:
                    output[str(key)] = visit(item)
            return output
        if isinstance(node, (list, tuple)):
            return [visit(item) for item in node]
        if isinstance(node, str):
            scrubbed, matched = scrub_text(node)
            patterns.extend(matched)
            return scrubbed
        return node

    redacted = visit(value)
    report = RedactionReportV1(
        redacted_body_keys=tuple(sorted(set(keys))),
        matched_patterns=tuple(sorted(set(patterns))),
    )
    return redacted, report


def redact_json_source_bytes(
    payload: bytes,
    *,
    json_lines: bool = False,
) -> tuple[bytes, RedactionReportV1]:
    """Return a secret-safe canonical source artifact for a JSON import.

    The original byte digest remains provenance, but only this redacted
    representation may enter a bundle blob store. Malformed JSONL records are
    represented by their digest and size rather than persisted as opaque text.
    """

    from ..canonical import bytes_digest, canonical_bytes

    if not json_lines:
        loaded = json.loads(payload.decode("utf-8"))
        redacted, report = redact_payload(loaded)
        assert_no_secrets(redacted, where="redacted JSON import source")
        return canonical_bytes(redacted), report

    safe_lines: list[bytes] = []
    reports: list[RedactionReportV1] = []
    malformed_lines = 0
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            loaded = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            malformed_lines += 1
            safe_lines.append(
                canonical_bytes(
                    {
                        "_synth_redacted_malformed_jsonl": True,
                        "malformed_line": line_number,
                        "wire_byte_size": len(line),
                        "wire_digest": bytes_digest(line),
                    }
                )
            )
            continue
        redacted, report = redact_payload(loaded)
        assert_no_secrets(redacted, where=f"redacted JSONL import line {line_number}")
        safe_lines.append(canonical_bytes(redacted))
        reports.append(report)

    merged = RedactionReportV1()
    for report in reports:
        merged = merged.merged(report)
    merged = RedactionReportV1(
        profile=merged.profile,
        removed_headers=merged.removed_headers,
        redacted_body_keys=merged.redacted_body_keys,
        matched_patterns=merged.matched_patterns,
        metadata={
            **merged.metadata,
            "source_encoding": "canonical_jsonl",
            "malformed_lines_omitted": malformed_lines,
        },
    )
    safe = b"\n".join(safe_lines)
    if safe_lines:
        safe += b"\n"
    return safe, merged


def _normalize_body_key(key: str) -> str:
    snake_case = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    return re.sub(r"[^a-z0-9]+", "_", snake_case.lower()).strip("_")


def _body_key_denied(normalized_key: str) -> bool:
    return normalized_key in DENIED_BODY_KEYS or normalized_key.endswith(
        _DENIED_BODY_KEY_SUFFIXES
    )


def assert_no_secrets(value: Any, *, where: str) -> None:
    """Fail closed if a persisted payload still matches a known secret shape."""

    from ..canonical import canonical_text

    text = canonical_text(value)
    for name, pattern in _SECRET_PATTERNS:
        if pattern.search(text):
            raise RedactionError(f"{where}: unredacted secret shape {name!r} would be persisted")

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, item in node.items():
                lowered = str(key).lower()
                normalized = _normalize_body_key(str(key))
                if (
                    lowered in DENIED_HEADERS or _body_key_denied(normalized)
                ) and item != REDACTED:
                    raise RedactionError(
                        f"{where}: credential-bearing field {key!r} would be persisted"
                    )
                visit(item)
        elif isinstance(node, (list, tuple)):
            for item in node:
                visit(item)

    visit(value)


__all__ = [
    "ALLOWED_HEADERS",
    "CORRELATION_HEADERS",
    "CORRELATION_HEADER_PREFIXES",
    "DENIED_BODY_KEYS",
    "DENIED_HEADERS",
    "REDACTED",
    "REDACTION_PROFILE",
    "RedactionError",
    "RedactionReportV1",
    "assert_no_secrets",
    "redact_headers",
    "redact_json_source_bytes",
    "redact_payload",
    "scrub_text",
]
