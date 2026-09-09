"""Append-only rollout event log. Consumer cursor is sequence; producer cursors stay internal."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import struct
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Iterable

from .tracing.capture.redaction import assert_no_secrets


CONTROL_SUBSCRIBED = "stream.subscribed"
SCHEMA_STREAM_EVENT = "synth.trace-stream-event.v1"
SCHEMA_EVENT_CHAIN = "synth.rollout.event-chain.v1"
SCHEMA_ENVELOPE_DIGEST = "synth.envelope-digest.v2"
_ROLLOUT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_FRAME_NAME = re.compile(r"^[A-Za-z0-9_]+$")

DEFAULT_STREAM_RECONNECT = {
    "min_backoff_s": 1.0,
    "max_backoff_s": 30.0,
    "jitter": 0.2,
}


def validate_rollout_id(value: str) -> str:
    if not _ROLLOUT_ID.fullmatch(value):
        raise ValueError("rollout_id must be 1-128 URL-safe identifier characters")
    return value


def _validate_frame_name(name: str) -> str:
    if ".." in name or "/" in name or "\\" in name or not _FRAME_NAME.fullmatch(name):
        raise ValueError("frame name must be a URL-safe token")
    return name


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


# Event kinds that must survive compaction verbatim. Anything describing how a
# rollout ended, or why it went wrong, is what someone reads a retained log to
# find out; the bulk -- per-step frames, actions, entity transitions -- is
# volume, not signal.
COMPACTION_KEEP_KINDS = frozenset(
    {
        "episode_truncated",
        "death",
        "terminal_success",
        "env.episode.closed",
        "status",
        "achievement_unlocked",
    }
)

# Payload fields that mark an envelope as carrying a failure, whatever its kind.
COMPACTION_ERROR_FIELDS = ("error", "failure", "failure_type", "invalid_parse", "sampler_failure")


def _is_error_envelope(envelope: "LogEnvelope") -> bool:
    if envelope.kind in COMPACTION_KEEP_KINDS:
        return True
    payload = envelope.payload
    return any(payload.get(field) for field in COMPACTION_ERROR_FIELDS)


@dataclass(slots=True)
class LogEnvelope:
    kind: str
    payload: dict[str, Any]
    sequence: int | None
    control: bool
    ts: str
    digest: str
    digest_schema: str | None = SCHEMA_ENVELOPE_DIGEST

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "schema": SCHEMA_STREAM_EVENT,
            "kind": self.kind,
            "ts": self.ts,
            "control": self.control,
            "payload": deepcopy(self.payload),
            "digest": self.digest,
        }
        if self.digest_schema is not None:
            row["digest_schema"] = self.digest_schema
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
        digest_schema = row.get("digest_schema")
        if digest_schema is None:
            # Journals persisted before the v2 contract remain recoverable. They
            # are never upgraded in place: their chain commits to their v1 digest.
            expected = _legacy_digest(kind, sequence, payload)
        elif digest_schema == SCHEMA_ENVELOPE_DIGEST:
            expected = _digest(kind, sequence, payload)
        else:
            raise ValueError("event_log_digest_schema_unsupported")
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
            digest_schema=digest_schema,
        )


def _legacy_digest(kind: str, sequence: int | None, payload: dict[str, Any]) -> str:
    """The unversioned v1 digest, retained only for persisted-journal recovery."""

    import hashlib

    blob = json.dumps(
        {"kind": kind, "sequence": sequence, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _canonical_v2(value: Any, out: bytearray) -> None:
    """Append the byte-exact ``synth.envelope-digest.v2`` encoding of ``value``.

    The tagged, length-delimited encoding deliberately does not depend on a JSON
    library's string escaping or floating-point rendering. Strings are raw UTF-8;
    finite floats are their IEEE-754 binary64 bits; object keys sort by UTF-8 bytes.
    """

    if value is None:
        out.extend(b"n")
    elif value is False:
        out.extend(b"f")
    elif value is True:
        out.extend(b"t")
    elif isinstance(value, int):
        if value < -(1 << 63) or value > (1 << 64) - 1:
            raise ValueError("event_log_integer_out_of_range")
        out.extend(b"i")
        out.extend(str(value).encode("ascii"))
        out.extend(b";")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("event_log_float_must_be_finite")
        out.extend(b"d")
        out.extend(struct.pack("!d", value).hex().encode("ascii"))
        out.extend(b";")
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        out.extend(b"s")
        out.extend(str(len(encoded)).encode("ascii"))
        out.extend(b":")
        out.extend(encoded)
    elif isinstance(value, list):
        out.extend(b"a")
        out.extend(str(len(value)).encode("ascii"))
        out.extend(b"[")
        for item in value:
            _canonical_v2(item, out)
        out.extend(b"]")
    elif isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("event_log_object_keys_must_be_strings")
        keys = sorted(value, key=lambda key: key.encode("utf-8"))
        out.extend(b"o")
        out.extend(str(len(keys)).encode("ascii"))
        out.extend(b"{")
        for key in keys:
            _canonical_v2(key, out)
            _canonical_v2(value[key], out)
        out.extend(b"}")
    else:  # pragma: no cover - payload normalization makes values JSON-native
        raise TypeError(f"event_log_value_not_json:{type(value).__name__}")


def canonical_envelope_bytes(
    kind: str, sequence: int | None, payload: dict[str, Any]
) -> bytes:
    """Return the versioned canonical digest preimage for one envelope."""

    out = bytearray(b"synth.envelope-digest.v2\0")
    _canonical_v2({"kind": kind, "sequence": sequence, "payload": payload}, out)
    return bytes(out)


def _digest(kind: str, sequence: int | None, payload: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(canonical_envelope_bytes(kind, sequence, payload)).hexdigest()[:16]


def envelope_digest(kind: str, sequence: int | None, payload: dict[str, Any]) -> str:
    """Digest under ``synth.envelope-digest.v2`` (see :func:`_digest`)."""

    return _digest(kind, sequence, payload)


def chain_genesis(rollout_id: str) -> str:
    """Genesis head of the per-rollout event chain (``synth.rollout.event-chain.v1``).

    Byte-exact definition:

    - ``genesis = sha256(utf8(rollout_id)).hexdigest()`` — 64 lowercase hex chars.
    - ``head(i) = sha256(ascii(head(i-1) + digest(i))).hexdigest()`` where
      ``digest(i)`` is the ``digest`` field of the i-th SEQUENCED
    (``control: false``) envelope in sequence order — 16 lowercase hex chars,
      itself the truncated sha256 of the versioned canonical envelope bytes
      (``synth.envelope-digest.v2``; see :func:`canonical_envelope_bytes`).

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
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    )
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
    _changed: threading.Condition = field(default_factory=threading.Condition, repr=False)

    @property
    def high_water(self) -> int:
        return self._high_water

    @property
    def chain_head(self) -> str:
        """Head of the per-rollout event chain (see :func:`chain_genesis`)."""

        return self._chain_head or chain_genesis(self.rollout_id)

    def _notify(self) -> None:
        with self._changed:
            self._changed.notify_all()

    def wake_readers(self) -> None:
        """Wake blocked ``wait_for_change`` callers without appending anything."""

        self._notify()

    def wait_for_change(self, after: int, timeout: float | None = None) -> bool:
        """Block until a sequence above ``after`` exists, the log closes, or ``timeout`` elapses.

        In-process tails (the live annotation runner) use this instead of a
        sleep loop. Returns True when there is something new to read.
        """

        with self._changed:
            if self._high_water > after or self.closed:
                return True
            self._changed.wait(timeout)
            return self._high_water > after or self.closed

    _released: bool = False
    _compacted: bool = False

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
        self._notify()
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
        self._notify()
        return envelope

    def compact(self) -> dict[str, Any]:
        """Shrink a closed log in RAM to errors plus a summary of the rest.

        Disk is untouched and authoritative: `_persist` fsynced every envelope,
        `recover` replays them, and `_rehydrate` restores full fidelity on
        demand. So this only decides what stays cheap to answer *without* a disk
        read -- and the questions worth answering cheaply are "how did it end"
        and "what went wrong", not "what was on frame 212".

        Compacting rather than releasing outright keeps a post-close poll useful
        (errors survive verbatim) while dropping the volume: on a 50-call
        episode the kept kinds are a few dozen envelopes against several hundred
        of per-step chatter.
        """

        if self._released:
            return {"compacted": False, "reason": "released"}
        with self._lock:
            if not self.closed:
                return {"compacted": False, "reason": "open"}
            if self._compacted:
                return {"compacted": False, "reason": "already"}
            kept: list[LogEnvelope] = []
            dropped = 0
            kinds: dict[str, int] = {}
            lo = hi = None
            for item in self._items:
                if item.control or _is_error_envelope(item):
                    kept.append(item)
                    continue
                dropped += 1
                kinds[item.kind] = kinds.get(item.kind, 0) + 1
                if item.sequence is not None:
                    lo = item.sequence if lo is None else min(lo, item.sequence)
                    hi = item.sequence if hi is None else max(hi, item.sequence)
            if not dropped:
                self._compacted = True
                return {"compacted": False, "reason": "nothing_to_drop"}
            summary = {
                "record": "compacted_summary",
                "dropped": dropped,
                "kinds": dict(sorted(kinds.items())),
                "covers": [lo, hi],
                "high_water": self._high_water,
                "note": "full fidelity remains on disk; a read reloads it",
            }
            # Deliberately a control record: control envelopes carry no sequence,
            # so inserting one cannot collide with a real sequence number or open
            # a gap that `recover` would fail closed on.
            kept.append(
                LogEnvelope(
                    kind="log.compacted",
                    payload=summary,
                    sequence=None,
                    control=True,
                    ts=_utc_now(),
                    digest=_digest("log.compacted", None, summary),
                )
            )
            self._items = kept
            self._compacted = True
            return {"compacted": True, **summary}

    def release(self) -> None:
        """Drop the in-RAM envelopes of a closed log; disk stays authoritative.

        Every rollout's full event list used to live in `PlatformState.logs`
        for the lifetime of the container, with nothing ever evicting it -- no
        pop, clear, del or prune anywhere. So gold's memory tracked *total
        rollouts processed*, not concurrent ones, and a long run grew without
        bound: RSS sat at 2.4 GiB after the producing runs had already been
        killed. That is what exhausted a 25 GB Docker VM and had the kernel
        SIGKILL the container mid-run (exit 137, OOMKilled=false because the
        container had no cgroup limit to trip).

        Releasing is lossless: `_persist` has already fsynced every envelope to
        the journal, and `recover` replays it failing closed on gaps. A reader
        arriving after release pays one disk read via `_rehydrate` and sees
        byte-identical envelopes.
        """

        with self._lock:
            if not self.closed or self._released:
                return
            self._items = []
            self._released = True

    def _rehydrate(self) -> None:
        """Reload a released or compacted log on access; caller must not hold the lock.

        Reads restore full fidelity rather than serving the compacted view,
        because readers are not all ours: `craftax_gold/gepa.py` builds its
        rollout response from `events_payload(rollout_id, 0, 10_000)`, and
        silently handing that a summary would truncate the response instead of
        failing. Compaction therefore reclaims memory only until something
        actually asks -- which for old rollouts is usually never.
        """
        restored = RolloutEventLog.recover(
            rollout_id=self.rollout_id,
            stream_id=self.stream_id,
            journal_path=self.journal_path,
        )
        with self._lock:
            if not (self._released or self._compacted):
                return
            self._items = restored._items
            self._released = False
            self._compacted = False

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
        self._notify()

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
        if self._released or self._compacted:
            self._rehydrate()
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
    def frame_asset_path(
        storage_root: Path, rollout_id: str, step: int, name: str | None = None
    ) -> Path:
        validate_rollout_id(rollout_id)
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("frame step must be a non-negative integer")
        filename = f"{step}.png"
        if name is not None:
            filename = f"{step}.{_validate_frame_name(name)}.png"
        rollout_key = __import__("hashlib").sha256(rollout_id.encode("utf-8")).hexdigest()
        return storage_root / "frame_assets" / rollout_key / filename

    def persist_frame(self, step: int, payload: bytes, *, name: str | None = None) -> str | None:
        """Durably store a PNG before its availability event becomes visible.

        ``name`` is a per-hero token (``agent_0``). The primary (active ego)
        frame keeps ``{step}.png`` / ``/rollouts/{id}/frames/{step}.png``.
        """
        if name is not None:
            _validate_frame_name(name)
        if self.journal_path is None or not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            return None
        storage_root = self.journal_path.parent.parent
        path = self.frame_asset_path(storage_root, self.rollout_id, step, name)
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
        if name is None:
            return f"/rollouts/{self.rollout_id}/frames/{step}.png"
        return f"/rollouts/{self.rollout_id}/frames/{step}/{name}.png"

    def _persist(self, row: dict[str, Any]) -> None:
        """Durably append before making a record visible to poll/SSE/WS consumers."""
        assert_no_secrets(row, where=f"rollout_event_log:{self.rollout_id}")
        if self.journal_path is None:
            return
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
        "reward": {
            "url": f"/rollouts/{rollout_id}/reward",
            "events": f"/rollouts/{rollout_id}/reward/events",
            "stream": f"/rollouts/{rollout_id}/reward/stream",
        },
        "auth": {"mode": "none"},
        "retention": retention,
        "reconnect": {
            "min_backoff_s": float(policy["min_backoff_s"]),
            "max_backoff_s": float(policy["max_backoff_s"]),
            "jitter": float(policy["jitter"]),
        },
    }


SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def format_sse(envelope: LogEnvelope, extra: dict[str, Any] | None = None) -> str:
    """One SSE record. ``id`` is the semantic sequence, or 0 for control records."""

    row = envelope.to_dict()
    if extra:
        row.update(extra)
    sse_id = envelope.sequence if envelope.sequence is not None else 0
    return (
        f"id: {sse_id}\n"
        f"event: {row['kind']}\n"
        f"data: {json.dumps(row, separators=(',', ':'))}\n\n"
    )


def poll_payload(
    log: RolloutEventLog,
    *,
    after: int,
    limit: int = 1000,
    kinds: Iterable[str] | None = None,
    subject_id_field: str = "rollout_id",
    subject_id: str | None = None,
) -> dict[str, Any]:
    """Page of envelopes after ``after``, optionally restricted to ``kinds``."""

    if isinstance(limit, bool) or limit < 1 or limit > 10_000:
        raise ValueError("invalid_page_limit")
    allowed = frozenset(kinds) if kinds is not None else None
    available = []
    for item in log.after(after):
        if allowed is not None and not item.control and item.kind not in allowed:
            continue
        available.append(item)
    controls = [item for item in available if item.sequence is None]
    evidence = [item for item in available if item.sequence is not None]
    page = [*controls, *evidence[:limit]]
    envelopes = [item.to_dict() for item in page]
    identity = subject_id if subject_id is not None else log.rollout_id
    for row in envelopes:
        row[subject_id_field] = identity
    return {
        subject_id_field: identity,
        "stream_id": log.stream_id,
        "cursor": {
            "kind": "sequence",
            "after": after,
            "high_water": log.high_water,
            "closed": log.closed,
            "next": max([after, *(item.sequence for item in page if item.sequence is not None)], default=after),
            "has_more": len(evidence) > limit,
        },
        "events": envelopes,
    }


async def iter_sse(
    log: RolloutEventLog,
    request: Any,
    *,
    after: int = 0,
    extra: dict[str, Any] | None = None,
    kinds: Iterable[str] | None = None,
) -> AsyncIterator[str]:
    """Replay then follow a log. Heartbeats keep Luna-idle streams open."""

    allowed = frozenset(kinds) if kinds is not None else None
    emitted_controls: set[str] = set()
    while not await request.is_disconnected():
        emitted = False
        for envelope in log.after(after):
            if envelope.sequence is not None:
                after = envelope.sequence
            if allowed is not None and not envelope.control and envelope.kind not in allowed:
                continue
            if envelope.control:
                control_key = f"{envelope.kind}:{envelope.digest}"
                if control_key in emitted_controls:
                    continue
                emitted_controls.add(control_key)
            yield format_sse(envelope, extra)
            emitted = True
        if log.closed:
            break
        if not emitted:
            yield ": heartbeat\n\n"
        await asyncio.sleep(0.05)
