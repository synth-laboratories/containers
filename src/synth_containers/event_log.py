"""Append-only rollout event log. Consumer cursor is sequence; producer cursors stay internal."""

from __future__ import annotations

import json
import os
import re
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .tracing.capture.redaction import assert_no_secrets


CONTROL_SUBSCRIBED = "stream.subscribed"
SCHEMA_STREAM_EVENT = "synth.trace-stream-event.v1"
SCHEMA_EVENT_CHAIN = "synth.rollout.event-chain.v1"
_ROLLOUT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

STREAM_HEARTBEAT_INTERVAL_S = 5.0
STREAM_TERMINAL_GRACE_S = 5.0
MAX_STREAMS_PER_ROLLOUT = 2
STREAM_RETRY_AFTER_S = 5
DEFAULT_STREAM_RECONNECT = {
    "min_backoff_s": 1.0,
    "max_backoff_s": 30.0,
    "jitter": 0.2,
}


def validate_rollout_id(value: str) -> str:
    if not _ROLLOUT_ID.fullmatch(value):
        raise ValueError("rollout_id must be 1-128 URL-safe identifier characters")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(slots=True)
class LogEnvelope:
    kind: str
    payload: dict[str, Any]
    sequence: int | None
    control: bool
    ts: str
    digest: str

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "schema": SCHEMA_STREAM_EVENT,
            "kind": self.kind,
            "ts": self.ts,
            "control": self.control,
            "payload": deepcopy(self.payload),
            "digest": self.digest,
        }
        if self.sequence is not None:
            row["sequence"] = self.sequence
            row["event_id"] = str(self.sequence)
        else:
            row["event_id"] = self.kind
        return row

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "LogEnvelope":
        if row.get("schema") != SCHEMA_STREAM_EVENT:
            raise ValueError("event_log_schema_mismatch")
        sequence = row.get("sequence")
        if sequence is not None and (isinstance(sequence, bool) or not isinstance(sequence, int)):
            raise ValueError("event_log_sequence_must_be_integer_or_null")
        payload = row.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("event_log_payload_must_be_object")
        kind = row.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError("event_log_kind_required")
        control = row.get("control")
        if not isinstance(control, bool):
            raise ValueError("event_log_control_must_be_boolean")
        if control != (sequence is None):
            raise ValueError("event_log_control_sequence_mismatch")
        expected_event_id = kind if sequence is None else str(sequence)
        if row.get("event_id") != expected_event_id:
            raise ValueError("event_log_event_id_mismatch")
        digest = row.get("digest")
        expected = _digest(kind, sequence, payload)
        if digest != expected:
            raise ValueError("event_log_digest_mismatch")
        ts = row.get("ts")
        if not isinstance(ts, str) or not ts:
            raise ValueError("event_log_timestamp_required")
        return cls(
            kind=kind,
            payload=deepcopy(payload),
            sequence=sequence,
            control=control,
            ts=ts,
            digest=expected,
        )


def _digest(kind: str, sequence: int | None, payload: dict[str, Any]) -> str:
    import hashlib
    import json

    blob = json.dumps(
        {"kind": kind, "sequence": sequence, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def envelope_digest(kind: str, sequence: int | None, payload: dict[str, Any]) -> str:
    """Public name for the canonical envelope digest (see :func:`_digest`)."""

    return _digest(kind, sequence, payload)


def chain_genesis(rollout_id: str) -> str:
    """Genesis head of the per-rollout event chain (``synth.rollout.event-chain.v1``).

    Byte-exact definition:

    - ``genesis = sha256(utf8(rollout_id)).hexdigest()`` — 64 lowercase hex chars.
    - ``head(i) = sha256(ascii(head(i-1) + digest(i))).hexdigest()`` where
      ``digest(i)`` is the ``digest`` field of the i-th SEQUENCED
    (``control: false``) envelope in sequence order — 16 lowercase hex chars,
      itself the truncated sha256 of the canonical
      ``{"kind","sequence","payload"}`` object (see :func:`_digest`).

    Control records never enter the chain.  A consumer that drains every
    sequenced event can recompute the head from the envelope digests alone and
    compare it to the ``chain_head`` carried in the events-page cursor, the
    ``capture.closed`` payload, and the lite seal.
    """

    import hashlib

    return hashlib.sha256(rollout_id.encode("utf-8")).hexdigest()


def chain_extend(head: str, digest: str) -> str:
    """Fold one sequenced envelope digest into the chain head."""

    import hashlib

    return hashlib.sha256((head + digest).encode("ascii")).hexdigest()


def chain_head_for(rollout_id: str, digests: "list[str] | tuple[str, ...]") -> str:
    """Chain head over ``digests`` (sequenced-envelope digests, in order)."""

    head = chain_genesis(rollout_id)
    for digest in digests:
        head = chain_extend(head, digest)
    return head


def _normalized_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Freeze a JSON-safe copy so persisted and published bytes cannot drift."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # pragma: no cover - dict input guarantees this
        raise ValueError("event_log_payload_must_be_object")
    return decoded


@dataclass
class RolloutEventLog:
    rollout_id: str
    stream_id: str
    closed: bool = False
    last_snapshot_key: str = ""
    journal_path: Path | None = None
    closed_at: str | None = None
    last_acked: int = 0
    _high_water: int = 0
    _chain_head: str = ""
    _items: list[LogEnvelope] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @property
    def high_water(self) -> int:
        return self._high_water

    @property
    def chain_head(self) -> str:
        """Head of the per-rollout event chain (see :func:`chain_genesis`)."""

        return self._chain_head or chain_genesis(self.rollout_id)

    def append_control(self, kind: str, payload: dict[str, Any]) -> LogEnvelope:
        with self._lock:
            if self.closed:
                raise RuntimeError(f"event_log_closed:{self.rollout_id}")
            frozen_payload = _normalized_payload(payload)
            envelope = LogEnvelope(
                kind=kind,
                payload=frozen_payload,
                sequence=None,
                control=True,
                ts=_utc_now(),
                digest=_digest(kind, None, frozen_payload),
            )
            self._persist({"record": "envelope", "envelope": envelope.to_dict()})
            self._items.append(envelope)
            return envelope

    def append(self, kind: str, payload: dict[str, Any]) -> LogEnvelope:
        with self._lock:
            if self.closed:
                raise RuntimeError(f"event_log_closed:{self.rollout_id}")
            frozen_payload = _normalized_payload(payload)
            next_sequence = self._high_water + 1
            envelope = LogEnvelope(
                kind=kind,
                payload=frozen_payload,
                sequence=next_sequence,
                control=False,
                ts=_utc_now(),
                digest=_digest(kind, next_sequence, frozen_payload),
            )
            self._persist({"record": "envelope", "envelope": envelope.to_dict()})
            self._high_water = next_sequence
            self._chain_head = chain_extend(self.chain_head, envelope.digest)
            self._items.append(envelope)
            return envelope

    def mark_closed(self) -> None:
        with self._lock:
            if self.closed:
                return
            closed_at = _utc_now()
            self._persist(
                {
                    "record": "closed",
                    "high_water": self._high_water,
                    "closed_at": closed_at,
                    "chain_head": self.chain_head,
                }
            )
            self.closed = True
            self.closed_at = closed_at

    def seal_capture(self) -> None:
        """Append the capture watermark records and close the log.

        ``capture.closed`` carries the chain head over the evidence events
        (everything before the two ``capture.*`` records), so a consumer can
        verify its drained evidence stream against a producer-signed head.
        """

        with self._lock:
            evidence_high_water = self.high_water
            evidence_chain_head = self.chain_head
            self.append("capture.high_water", {"high_water": evidence_high_water})
            self.append(
                "capture.closed",
                {
                    "high_water": evidence_high_water,
                    "chain_head": evidence_chain_head,
                    "chain_schema": SCHEMA_EVENT_CHAIN,
                },
            )
            self.mark_closed()

    def record_ack(self, sequence: int) -> int:
        """Record the consumer's durably-processed high water; returns the ack head.

        Acks are monotonic and never run ahead of ``high_water`` (acking the
        future is clamped).  The ack head is durably stored in a sidecar next
        to the journal so retention decisions survive recovery.
        """

        with self._lock:
            if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
                raise ValueError("event_log_ack_must_be_non_negative_integer")
            acked = min(max(sequence, self.last_acked), self._high_water)
            if acked != self.last_acked:
                self.last_acked = acked
                self._persist_ack()
            return self.last_acked

    def _ack_path(self) -> Path | None:
        if self.journal_path is None:
            return None
        return self.journal_path.with_name(self.journal_path.name + ".ack.json")

    def _persist_ack(self) -> None:
        path = self._ack_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        encoded = json.dumps(
            {"rollout_id": self.rollout_id, "acked": self.last_acked, "ts": _utc_now()},
            sort_keys=True,
            separators=(",", ":"),
        )
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def subscribed_payload(self) -> dict[str, Any]:
        return {
            "type": CONTROL_SUBSCRIBED,
            "stream.id": self.stream_id,
            "rollout_id": self.rollout_id,
            "next_sequence": self._high_water + 1,
            "ready": True,
        }

    def after(self, sequence: int) -> list[LogEnvelope]:
        """Semantic events with sequence > `sequence`, plus control records when sequence == 0."""
        with self._lock:
            out: list[LogEnvelope] = []
            for item in self._items:
                if item.control:
                    if sequence <= 0:
                        out.append(item)
                    continue
                if item.sequence is not None and item.sequence > sequence:
                    out.append(item)
            return out

    def snapshot_key(self) -> str:
        return self.last_snapshot_key

    @property
    def persisted(self) -> bool:
        return self.journal_path is not None

    @staticmethod
    def frame_asset_path(storage_root: Path, rollout_id: str, step: int) -> Path:
        validate_rollout_id(rollout_id)
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("frame step must be a non-negative integer")
        rollout_key = __import__("hashlib").sha256(rollout_id.encode("utf-8")).hexdigest()
        return storage_root / "frame_assets" / rollout_key / f"{step}.png"

    def persist_frame(self, step: int, payload: bytes) -> str | None:
        """Durably store a PNG before its availability event becomes visible."""
        if self.journal_path is None or not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            return None
        storage_root = self.journal_path.parent.parent
        path = self.frame_asset_path(storage_root, self.rollout_id, step)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return f"/rollouts/{self.rollout_id}/frames/{step}.png"

    def _persist(self, row: dict[str, Any]) -> None:
        """Durably append before making a record visible to poll/SSE/WS consumers."""
        if self.journal_path is None:
            return
        assert_no_secrets(row, where=f"rollout_event_log:{self.rollout_id}")
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)
        with self.journal_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    @classmethod
    def recover(
        cls,
        *,
        rollout_id: str,
        stream_id: str,
        journal_path: Path,
    ) -> "RolloutEventLog":
        """Recover the exact sequence log, failing closed on corruption or gaps."""
        log = cls(rollout_id=rollout_id, stream_id=stream_id, journal_path=journal_path)
        if not journal_path.exists():
            return log
        closed = False
        expected_sequence = 1
        for line_number, line in enumerate(journal_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"event_log_malformed_json_line:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"event_log_record_must_be_object:{line_number}")
            record = row.get("record")
            if record == "closed":
                if row.get("high_water") != log._high_water:
                    raise ValueError(f"event_log_close_high_water_mismatch:{line_number}")
                declared_head = row.get("chain_head")
                if declared_head is not None and declared_head != log.chain_head:
                    # A journal whose per-event digests validate but whose
                    # recomputed chain differs from the sealed head has been
                    # rewritten; fail closed like every other corruption.
                    raise ValueError(f"event_log_chain_head_mismatch:{line_number}")
                closed = True
                closed_at = row.get("closed_at")
                if isinstance(closed_at, str) and closed_at:
                    log.closed_at = closed_at
                continue
            if record != "envelope" or not isinstance(row.get("envelope"), dict):
                raise ValueError(f"event_log_unknown_record:{line_number}")
            if closed:
                raise ValueError(f"event_log_record_after_close:{line_number}")
            envelope = LogEnvelope.from_dict(row["envelope"])
            if envelope.sequence is not None:
                if envelope.sequence != expected_sequence:
                    raise ValueError(f"event_log_sequence_gap:{line_number}")
                expected_sequence += 1
                log._high_water = envelope.sequence
                log._chain_head = chain_extend(log.chain_head, envelope.digest)
            log._items.append(envelope)
        log.closed = closed
        log._recover_ack()
        return log

    def _recover_ack(self) -> None:
        path = self._ack_path()
        if path is None or not path.exists():
            return
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("event_log_ack_sidecar_malformed") from exc
        acked = row.get("acked") if isinstance(row, dict) else None
        if isinstance(acked, bool) or not isinstance(acked, int) or acked < 0:
            raise ValueError("event_log_ack_sidecar_invalid")
        self.last_acked = min(acked, self._high_water)


def stream_descriptor(
    *,
    rollout_id: str,
    stream_id: str,
    bound_transport: str,
    retention: str = "run",
    reconnect: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_rollout_id(rollout_id)
    poll_url = f"/rollouts/{rollout_id}/events"
    sse_url = f"/rollouts/{rollout_id}/stream"
    ws_url = f"/rollouts/{rollout_id}/ws"
    policy = dict(reconnect or DEFAULT_STREAM_RECONNECT)
    return {
        "schema": "synth.rollout.stream.v1",
        "id": stream_id,
        "transports": {
            "poll": {"url": poll_url},
            "sse": {"url": sse_url} if bound_transport in {"sse", "websocket"} else None,
            "websocket": {"url": ws_url} if bound_transport == "websocket" else None,
        },
        "cursor": {"kind": "sequence", "producer_kind": None},
        "reward": {"url": f"/rollouts/{rollout_id}/reward"},
        "auth": {"mode": "none"},
        "retention": retention,
        "reconnect": {
            "min_backoff_s": float(policy["min_backoff_s"]),
            "max_backoff_s": float(policy["max_backoff_s"]),
            "jitter": float(policy["jitter"]),
        },
    }
