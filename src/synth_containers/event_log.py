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
    _high_water: int = 0
    _items: list[LogEnvelope] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @property
    def high_water(self) -> int:
        return self._high_water

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
            self._items.append(envelope)
            return envelope

    def mark_closed(self) -> None:
        with self._lock:
            if self.closed:
                return
            self._persist({"record": "closed", "high_water": self._high_water})
            self.closed = True

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
                closed = True
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
            log._items.append(envelope)
        log.closed = closed
        return log


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
