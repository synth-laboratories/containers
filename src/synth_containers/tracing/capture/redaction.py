"""Credential stripping and secret scanning applied before anything is persisted.

Redaction runs at capture ingress, not at export. A credential-bearing header never
reaches a spool segment, so no later projection can leak one. Sealing fails closed
when the scan still finds a secret shape in a body that is about to be written.
"""

from __future__ import annotations

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
CORRELATION_HEADERS = frozenset({"traceparent", "tracestate", "baggage"})

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

_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9._\-]{16,}")),
    ("groq_key", re.compile(r"\bgsk_[A-Za-z0-9]{16,}")),
    ("google_key", re.compile(r"\bAIza[0-9A-Za-z._\-]{20,}")),
    ("tinker_key", re.compile(r"\btk-[A-Za-z0-9._\-]{16,}")),
    ("synth_key", re.compile(r"\bsk_(?:live|test|prod|dev)_[A-Za-z0-9._\-]{12,}")),
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
                if str(key).lower() in DENIED_BODY_KEYS:
                    keys.append(str(key).lower())
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


def assert_no_secrets(value: Any, *, where: str) -> None:
    """Fail closed if a persisted payload still matches a known secret shape."""

    from ..canonical import canonical_text

    text = canonical_text(value)
    for name, pattern in _SECRET_PATTERNS:
        if pattern.search(text):
            raise RedactionError(f"{where}: unredacted secret shape {name!r} would be persisted")
    lowered = text.lower()
    for header in DENIED_HEADERS:
        # Match the header as an object key. A redaction report legitimately lists the
        # same names as values, and naming what was stripped is not a leak.
        if f'"{header}":' in lowered:
            raise RedactionError(
                f"{where}: credential-bearing header {header!r} would be persisted"
            )


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
    "redact_payload",
    "scrub_text",
]
