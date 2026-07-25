"""Append-only raw spool with immutable, digest-named segments.

Writing protocol:

1. records append to an open ``.partial`` file, flushed per record;
2. rotation or close reads the partial back, computes its digest, and atomically
   renames it to ``<sequence>-<digest12>.jsonl``;
3. the live manifest generation is rewritten atomically after each sealed segment.

A crash therefore leaves at most one ``.partial`` file. ``repair`` promotes its
complete leading lines into a sealed segment and reports the truncated tail, so an
interrupted capture can never publish a corrupt sealed trace.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator

from synth_containers.serde import JsonDataclassMixin

from ..canonical import bytes_digest, canonical_text, content_digest, short_digest, utc_now
from .envelope import RawCaptureEnvelopeV1
from .redaction import assert_no_secrets


SEGMENT_MANIFEST_SCHEMA_VERSION = "synth.trace-segment-manifest.v1"
LIVE_MANIFEST_SCHEMA_VERSION = "synth.trace-live-manifest.v1"
PARTIAL_SUFFIX = ".partial"


@dataclass(frozen=True, slots=True)
class TraceSegmentManifestV1(JsonDataclassMixin):
    """Immutable description of one sealed raw segment."""

    segment_id: str
    capture_id: str
    sequence: int
    digest: str
    record_count: int
    byte_size: int
    first_ordinal: int
    last_ordinal: int
    first_occurred_at: str
    last_occurred_at: str
    relative_path: str
    encoding: str = "jsonl"
    compression: str = "none"
    media_type: str = "application/x-ndjson"
    truncated_tail_bytes: int = 0
    schema_version: str = SEGMENT_MANIFEST_SCHEMA_VERSION
    content_digest: str = ""

    def sealed(self) -> "TraceSegmentManifestV1":
        return replace(self, content_digest=content_digest(self))


@dataclass(frozen=True, slots=True)
class LiveManifestV1(JsonDataclassMixin):
    """Versioned view of the segments captured so far; ``latest.json`` points here."""

    capture_id: str
    generation: int
    updated_at: str
    segments: tuple[TraceSegmentManifestV1, ...] = ()
    high_water_ordinal: int = -1
    open_partial: str | None = None
    schema_version: str = LIVE_MANIFEST_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)


class RawSpool:
    """Append-only raw record spool for one capture session."""

    def __init__(
        self,
        root: Path,
        *,
        capture_id: str,
        max_segment_records: int = 512,
    ) -> None:
        self.root = Path(root)
        self.capture_id = capture_id
        self.max_segment_records = max(1, int(max_segment_records))
        self.segments_dir = self.root / "segments"
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        self._sequence = 0
        self._generation = 0
        self._sealed: list[TraceSegmentManifestV1] = []
        self._open_path: Path | None = None
        self._open_handle: Any = None
        self._open_records: list[RawCaptureEnvelopeV1] = []
        self._high_water = -1

    # -- writing -----------------------------------------------------------------

    def append(self, envelope: RawCaptureEnvelopeV1) -> RawCaptureEnvelopeV1:
        """Append one raw record. The record is durable before this returns."""

        assert_no_secrets(envelope.payload, where=f"raw envelope {envelope.envelope_id}")
        if self._open_handle is None:
            self._open_segment()
        assert self._open_handle is not None
        self._open_handle.write(canonical_text(envelope) + "\n")
        self._open_handle.flush()
        os.fsync(self._open_handle.fileno())
        self._open_records.append(envelope)
        self._high_water = max(self._high_water, int(envelope.ordinal))
        if len(self._open_records) >= self.max_segment_records:
            self.rotate()
        return envelope

    def rotate(self) -> TraceSegmentManifestV1 | None:
        """Seal the open partial into an immutable segment and publish a manifest."""

        if self._open_handle is None or self._open_path is None:
            return None
        self._open_handle.flush()
        os.fsync(self._open_handle.fileno())
        self._open_handle.close()
        self._open_handle = None
        payload = self._open_path.read_bytes()
        records = list(self._open_records)
        self._open_records = []
        partial_path = self._open_path
        self._open_path = None
        if not records:
            partial_path.unlink(missing_ok=True)
            return None
        manifest = self._seal_segment(partial_path, payload, records)
        self._publish_manifest()
        return manifest

    def close(self) -> LiveManifestV1:
        self.rotate()
        return self._publish_manifest()

    def _open_segment(self) -> None:
        self._sequence += 1
        self._open_path = self.segments_dir / f"{self._sequence:06d}{PARTIAL_SUFFIX}"
        self._open_handle = self._open_path.open("w", encoding="utf-8")

    def _seal_segment(
        self,
        partial_path: Path,
        payload: bytes,
        records: list[RawCaptureEnvelopeV1],
    ) -> TraceSegmentManifestV1:
        digest = bytes_digest(payload)
        sequence = int(partial_path.stem)
        name = f"{sequence:06d}-{short_digest(digest)}.jsonl"
        final_path = self.segments_dir / name
        partial_path.replace(final_path)
        final_path.chmod(0o444)
        manifest = TraceSegmentManifestV1(
            segment_id=f"seg_{short_digest(digest, length=16)}",
            capture_id=self.capture_id,
            sequence=sequence,
            digest=digest,
            record_count=len(records),
            byte_size=len(payload),
            first_ordinal=int(records[0].ordinal),
            last_ordinal=int(records[-1].ordinal),
            first_occurred_at=records[0].occurred_at,
            last_occurred_at=records[-1].occurred_at,
            relative_path=f"segments/{name}",
        ).sealed()
        self._sealed.append(manifest)
        return manifest

    def _publish_manifest(self) -> LiveManifestV1:
        self._generation += 1
        manifest = LiveManifestV1(
            capture_id=self.capture_id,
            generation=self._generation,
            updated_at=utc_now(),
            segments=tuple(self._sealed),
            high_water_ordinal=self._high_water,
            open_partial=self._open_path.name if self._open_path else None,
        )
        manifests_dir = self.root / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        generation_path = manifests_dir / f"{self._generation:06d}.json"
        generation_path.write_text(canonical_text(manifest) + "\n", encoding="utf-8")
        latest = self.root / "latest.json"
        temp = latest.with_suffix(".json.tmp")
        temp.write_text(canonical_text(manifest) + "\n", encoding="utf-8")
        temp.replace(latest)
        return manifest

    # -- reading -----------------------------------------------------------------

    @property
    def segments(self) -> tuple[TraceSegmentManifestV1, ...]:
        return tuple(self._sealed)

    @property
    def high_water_ordinal(self) -> int:
        return self._high_water

    def records(self) -> Iterator[dict[str, Any]]:
        yield from read_segments(self.root, self.segments)


def read_segments(
    root: Path,
    segments: tuple[TraceSegmentManifestV1, ...],
) -> Iterator[dict[str, Any]]:
    """Read raw records back in canonical segment order, verifying each digest."""

    for segment in sorted(segments, key=lambda item: item.sequence):
        path = Path(root) / segment.relative_path
        payload = path.read_bytes()
        actual = bytes_digest(payload)
        if actual != segment.digest:
            raise ValueError(
                f"segment digest mismatch for {segment.relative_path}: "
                f"expected {segment.digest}, found {actual}"
            )
        for line in payload.decode("utf-8").splitlines():
            if line.strip():
                yield json.loads(line)


@dataclass(frozen=True, slots=True)
class SpoolRepairV1(JsonDataclassMixin):
    """Result of promoting an interrupted partial into a sealed segment."""

    capture_id: str
    repaired: bool
    recovered_records: int = 0
    truncated_tail_bytes: int = 0
    segment_digest: str | None = None
    reason: str = ""
    repaired_at: str = ""


def repair(root: Path, *, capture_id: str) -> SpoolRepairV1:
    """Recover an interrupted spool without ever publishing a corrupt record.

    Complete leading JSON lines are promoted into a sealed segment; an incomplete
    trailing line is dropped and its byte count is reported.
    """

    segments_dir = Path(root) / "segments"
    partials = sorted(segments_dir.glob(f"*{PARTIAL_SUFFIX}"))
    if not partials:
        return SpoolRepairV1(
            capture_id=capture_id,
            repaired=False,
            reason="no_partial_segment",
            repaired_at=utc_now(),
        )
    partial = partials[-1]
    payload = partial.read_bytes()
    text = payload.decode("utf-8", errors="replace")
    complete: list[str] = []
    consumed = 0
    for line in text.splitlines(keepends=True):
        if not line.endswith("\n"):
            break
        stripped = line.strip()
        if stripped:
            try:
                json.loads(stripped)
            except json.JSONDecodeError:
                break
            complete.append(stripped)
        consumed += len(line.encode("utf-8"))
    truncated = len(payload) - consumed
    if not complete:
        partial.unlink(missing_ok=True)
        return SpoolRepairV1(
            capture_id=capture_id,
            repaired=True,
            recovered_records=0,
            truncated_tail_bytes=truncated,
            reason="partial_had_no_complete_records",
            repaired_at=utc_now(),
        )
    recovered = ("\n".join(complete) + "\n").encode("utf-8")
    digest = bytes_digest(recovered)
    sequence = int(partial.stem)
    final_path = segments_dir / f"{sequence:06d}-{short_digest(digest)}.jsonl"
    final_path.write_bytes(recovered)
    final_path.chmod(0o444)
    partial.unlink(missing_ok=True)
    return SpoolRepairV1(
        capture_id=capture_id,
        repaired=True,
        recovered_records=len(complete),
        truncated_tail_bytes=truncated,
        segment_digest=digest,
        reason="promoted_complete_leading_records",
        repaired_at=utc_now(),
    )


__all__ = [
    "LIVE_MANIFEST_SCHEMA_VERSION",
    "SEGMENT_MANIFEST_SCHEMA_VERSION",
    "LiveManifestV1",
    "RawSpool",
    "SpoolRepairV1",
    "TraceSegmentManifestV1",
    "read_segments",
    "repair",
]
