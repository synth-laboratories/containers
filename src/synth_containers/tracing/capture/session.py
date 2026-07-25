"""``CaptureSession`` — the single append point every capture producer writes through.

The proxy and the collector share one monotonic ordinal and one spool so that model
calls and application events interleave in a single, replayable order.
"""

from __future__ import annotations

import threading
from typing import Any

from ..canonical import record_id
from ..store.base import BlobStore
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
        blobs: BlobStore,
    ) -> None:
        self.binding = binding
        self.spool = spool
        self.blobs = blobs
        self.first_observed_at: str | None = None
        self.last_observed_at: str | None = None
        self._lock = threading.Lock()
        self._ordinal = spool.high_water_ordinal

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
            # Ordinal allocation and durable append are one critical section. If
            # they were separate, thread B could append N+1 before thread A
            # appended N, advancing the spool high-water mark and rejecting N.
            self.spool.append(envelope)
            if self.first_observed_at is None:
                self.first_observed_at = envelope.occurred_at
            self.last_observed_at = envelope.occurred_at
        return envelope

    def store_blob(self, content: bytes) -> tuple[str, str]:
        """Write a content-addressed body and return ``(digest, relative uri)``."""

        digest = self.blobs.put(content)
        return digest, self.blobs.uri(digest)

    def mint(self, prefix: str, *, kind: str, key: Any) -> str:
        return record_id(
            prefix,
            kind=kind,
            scope=(self.binding.trace_id, self.binding.capture_id),
            key=key,
        )


__all__ = ["CaptureSession"]
