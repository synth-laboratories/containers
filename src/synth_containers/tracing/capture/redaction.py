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
# A distinct marker for a value that matched a *secret shape* (not merely a
# credential-bearing field name). The fail-closed seal refuses to persist any
# body that still carries this marker: if a live credential shape transited the
# capture path, the record is dropped rather than stored with a partial scrub.
SECRET_REDACTED = "<redacted:secret>"

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
        "passwd",
        "secret",
        "secret_key",
        "private_key",
        "aws_secret_access_key",
        "credentials",
        "token",
        "bearer",
        "key",
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
    "_passwd",
    "_secret",
    "_secret_key",
    "_private_key",
    "_token",
    "_bearer",
    "_cookie",
    "_key",
    "_credentials",
)

# Attribution/identity fields whose names end in ``_key`` but carry no credential.
# They are exempt from the broad ``_key`` suffix denial (and only from that: their
# values are still scanned for secret shapes). ``prompt_cache_key`` attributes a
# provider call to its native Codex child; ``idempotency_key`` and ``trace_key``
# identify rollouts and capture sessions.
NON_SECRET_BODY_KEYS = frozenset(
    {
        "prompt_cache_key",
        "idempotency_key",
        "trace_key",
    }
)

_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9._\-]{16,}")),
    ("groq_key", re.compile(r"\bgsk_[A-Za-z0-9]{16,}")),
    ("google_key", re.compile(r"\bAIza[0-9A-Za-z._\-]{20,}")),
    ("tinker_key", re.compile(r"\btk-[A-Za-z0-9._\-]{16,}")),
    # Synth mints user keys as sk_synth_user_<48 hex> (backend
    # routes_api_keys.py:58,211) and demo keys as sk_demo_<urlsafe>
    # (demo/routes.py:256), alongside the sk_(live|test|prod|dev)_ shapes.
    (
        "synth_key",
        re.compile(r"\bsk_(?:live|test|prod|dev|synth_user|demo)_[A-Za-z0-9._\-]{12,}"),
    ),
    ("trace_capability", re.compile(r"\bsk_trace_[A-Za-z0-9._\-]{16,}")),
    # Third-party credential shapes coding-agent traces routinely carry.
    ("github_pat_classic", re.compile(r"\bghp_[A-Za-z0-9]{20,}")),
    ("github_pat_fine_grained", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}")),
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|AIPA)[0-9A-Z]{12,}")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    # Bare JWTs (three base64url segments); anchored on the "eyJ" header to
    # avoid matching ordinary dotted identifiers.
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_=-]+\.[A-Za-z0-9_=-]+\.[A-Za-z0-9_=-]+"),
    ),
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
            result = pattern.sub(SECRET_REDACTED, result)
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
    if normalized_key in NON_SECRET_BODY_KEYS:
        return False
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
    # Fail closed: a payload that required a secret-shape scrub carries the
    # SECRET_REDACTED marker. A live credential transited the capture path, so
    # the record is refused rather than persisted with a best-effort scrub.
    if SECRET_REDACTED in text:
        raise RedactionError(
            f"{where}: payload carried a secret credential shape (scrubbed) and is refused"
        )

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
    "NON_SECRET_BODY_KEYS",
    "DENIED_HEADERS",
    "REDACTED",
    "SECRET_REDACTED",
    "REDACTION_PROFILE",
    "RedactionError",
    "RedactionReportV1",
    "assert_no_secrets",
    "redact_headers",
    "redact_json_source_bytes",
    "redact_payload",
    "scrub_text",
]
