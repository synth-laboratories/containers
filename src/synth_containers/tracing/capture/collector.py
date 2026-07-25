"""Local append-only collector for facts a model reverse proxy cannot observe.

Tool executions, environment observations, executed actions, transitions, rewards,
and produced artifacts all arrive here under the same binding as the proxy traffic.
Without it, a proxy-only capture must declare ``model_calls_only`` coverage.

The collector is in-process by design for Push 1: both acceptance consumers own the
process that observes these facts, so a local HTTP hop would add a failure mode
without adding evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..canonical import record_id, utc_now
from ..models.actors import SessionStatus
from ..models.events import EventType
from .envelope import RawRecordType
from .redaction import RedactionReportV1, assert_no_secrets, redact_payload, scrub_text
from .session import CaptureSession


COLLECTOR_VERSION = "synth-trace-collector/1"


@dataclass(slots=True)
class CollectorStats:
    application_events: int = 0
    artifacts: int = 0
    sessions_finished: int = 0


class LocalCollector:
    """Appends application events and artifacts to the capture session's spool."""

    def __init__(self, session: CaptureSession) -> None:
        self.session = session
        self.stats = CollectorStats()

    @property
    def binding(self) -> Any:
        return self.session.binding

    def event(
        self,
        *,
        event_type: str,
        payload: dict[str, Any],
        actor_id: str | None = None,
        session_id: str | None = None,
        occurred_at: str | None = None,
        caused_by: tuple[str, ...] = (),
        structural: dict[str, Any] | None = None,
    ) -> str:
        """Append one application event and return its raw envelope id."""

        redacted, report = redact_payload(payload)
        envelope = self.session.append(
            RawRecordType.APPLICATION_EVENT,
            payload={
                "event_type": event_type,
                "body": redacted,
                "caused_by": list(caused_by),
                "structural": structural,
                "redaction": report.to_dict(),
            },
            actor_id=actor_id,
            session_id=session_id,
            occurred_at=occurred_at or utc_now(),
            producer_version=COLLECTOR_VERSION,
        )
        self.stats.application_events += 1
        return envelope.envelope_id

    def artifact(
        self,
        *,
        role: str,
        media_type: str,
        content: bytes,
        logical_name: str,
        visibility: str = "private",
        actor_id: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Store a digest-addressed body and append its artifact record.

        Artifact bodies are redacted before they are written, the same as events.
        A binary body that still matches a secret shape fails closed rather than
        being stored, because there is no safe way to rewrite it in place.
        """

        stored, redaction = _redact_artifact(content, media_type=media_type)
        assert_no_secrets(
            {"artifact": stored.decode("utf-8", errors="replace")},
            where=f"artifact {logical_name}",
        )
        digest, uri = self.session.store_blob(stored)
        artifact_id = record_id(
            "art",
            kind="artifact",
            scope=(self.session.binding.trace_id,),
            key={"digest": digest, "name": logical_name},
        )
        self.session.append(
            RawRecordType.ARTIFACT,
            payload={
                "artifact_id": artifact_id,
                "digest": digest,
                "media_type": media_type,
                "size_bytes": len(stored),
                "role": role,
                "logical_name": logical_name,
                "visibility": visibility,
                "uri": uri,
                "redaction": redaction.to_dict(),
            },
            actor_id=actor_id,
            session_id=session_id,
            producer_version=COLLECTOR_VERSION,
        )
        self.stats.artifacts += 1
        return artifact_id

    def finish_session(
        self,
        *,
        status: SessionStatus | str,
        actor_id: str,
        session_id: str,
        ended_at: str | None = None,
    ) -> tuple[str, str]:
        """Append one durable terminal child-session fact.

        The collector server owns idempotency and ordering against later child
        writes. This append point owns the raw fact from which the finalizer derives
        the immutable ``SessionV5`` lifecycle and canonical ``session.finished``
        event.
        """

        normalized = str(status)
        if normalized not in {
            str(SessionStatus.COMPLETED),
            str(SessionStatus.FAILED),
            str(SessionStatus.INTERRUPTED),
        }:
            raise ValueError("child session status must be terminal")
        terminal_at = ended_at or utc_now()
        envelope = self.session.append(
            RawRecordType.SESSION_FINISHED,
            payload={
                "event_type": str(EventType.SESSION_FINISHED),
                "body": {
                    "status": normalized,
                    "ended_at": terminal_at,
                },
                "caused_by": [],
                "structural": None,
                "redaction": RedactionReportV1().to_dict(),
            },
            actor_id=actor_id,
            session_id=session_id,
            occurred_at=terminal_at,
            producer_version=COLLECTOR_VERSION,
        )
        self.stats.sessions_finished += 1
        return envelope.envelope_id, terminal_at


def _redact_artifact(content: bytes, *, media_type: str) -> tuple[bytes, RedactionReportV1]:
    """Redact an artifact body according to what its media type allows."""

    lowered = media_type.lower()
    if lowered == "application/json" or lowered.endswith("+json"):
        try:
            payload = json.loads(content.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _redact_text(content)
        redacted, report = redact_payload(payload)
        return json.dumps(redacted, sort_keys=True).encode("utf-8"), report
    if lowered.startswith("text/") or lowered in {"application/x-ndjson", "application/x-sh"}:
        return _redact_text(content)
    return content, RedactionReportV1(metadata={"binary_media_type": media_type})


def _redact_text(content: bytes) -> tuple[bytes, RedactionReportV1]:
    scrubbed, matched = scrub_text(content.decode("utf-8", errors="replace"))
    return scrubbed.encode("utf-8"), RedactionReportV1(matched_patterns=matched)


__all__ = ["COLLECTOR_VERSION", "CollectorStats", "LocalCollector"]
