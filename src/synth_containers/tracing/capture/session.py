"""``CaptureSession`` — the single append point every capture producer writes through.

The proxy and the collector share one monotonic ordinal and one spool so that model
calls and application events interleave in a single, replayable order.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from ..canonical import bytes_digest, record_id
from .binding import TraceCaptureBindingV1
from .envelope import RawCaptureEnvelopeV1, RawRecordType, make_envelope
from .spool import RawSpool


class CaptureSession:
    """Owns the raw spool, ordinal sequence, and content-addressed blob root."""

    def __init__(
        self,
        *,
        binding: TraceCaptureBindingV1,
        spool: RawSpool,
        blob_root: Path,
    ) -> None:
        self.binding = binding
        self.spool = spool
        self.blob_root = Path(blob_root)
        self.first_observed_at: str | None = None
        self.last_observed_at: str | None = None
        self._lock = threading.Lock()
        self._ordinal = 0

    def append(
        self,
        record_type: RawRecordType | str,
        *,
        payload: dict[str, Any],
        actor_id: str | None = None,
        session_id: str | None = None,
        call_id: str | None = None,
        upstream_attempt_id: str | None = None,
        sequence_in_call: int | None = None,
        occurred_at: str | None = None,
        producer_version: str = "1",
    ) -> RawCaptureEnvelopeV1:
        with self._lock:
            self._ordinal += 1
            ordinal = self._ordinal
        envelope = make_envelope(
            capture_id=self.binding.capture_id,
            ordinal=ordinal,
            record_type=record_type,
            actor_id=actor_id or self.binding.workload.root_actor_id,
            session_id=session_id or self.binding.workload.actor_session_id,
            payload=payload,
            call_id=call_id,
            upstream_attempt_id=upstream_attempt_id,
            sequence_in_call=sequence_in_call,
            occurred_at=occurred_at,
            producer_version=producer_version,
        )
        with self._lock:
            self.spool.append(envelope)
            if self.first_observed_at is None:
                self.first_observed_at = envelope.occurred_at
            self.last_observed_at = envelope.occurred_at
        return envelope

    def store_blob(self, content: bytes) -> tuple[str, str]:
        """Write a content-addressed body and return ``(digest, relative uri)``."""

        digest = bytes_digest(content)
        algorithm, _, hexed = digest.partition(":")
        path = self.blob_root / algorithm / hexed[:2] / hexed
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".tmp")
            temp.write_bytes(content)
            temp.replace(path)
            path.chmod(0o444)
        return digest, f"blobs/{algorithm}/{hexed[:2]}/{hexed}"

    def mint(self, prefix: str, *, kind: str, key: Any) -> str:
        return record_id(
            prefix,
            kind=kind,
            scope=(self.binding.trace_id, self.binding.capture_id),
            key=key,
        )


__all__ = ["CaptureSession"]
