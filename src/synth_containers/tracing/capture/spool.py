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
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterator

from synth_containers.serde import JsonDataclassMixin

from ..canonical import bytes_digest, canonical_text, content_digest, short_digest, utc_now
from .envelope import RawCaptureEnvelopeV1, validate_envelope_payload
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
    content_digest: str = ""

    def sealed(self) -> "LiveManifestV1":
        return replace(self, content_digest=content_digest(self))


@dataclass(frozen=True, slots=True)
class LiveManifestPointerV1(JsonDataclassMixin):
    """Mutable latest pointer to one immutable live-manifest generation."""

    capture_id: str
    generation: int
    manifest_digest: str
    relative_path: str
    updated_at: str
    schema_version: str = "synth.trace-live-manifest-pointer.v1"


class RawSpool:
    """Append-only raw record spool for one capture session."""

    def __init__(
        self,
        root: Path,
        *,
        capture_id: str,
        max_segment_records: int = 512,
        compression: str = "none",
    ) -> None:
        self.root = Path(root)
        self.capture_id = capture_id
        self.max_segment_records = max(1, int(max_segment_records))
        if compression != "none":
            raise ValueError(f"unsupported trace spool compression: {compression!r}")
        self.compression = compression
        self.segments_dir = self.root / "segments"
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        self._sequence = 0
        self._generation = 0
        self._sealed: list[TraceSegmentManifestV1] = []
        self._open_path: Path | None = None
        self._open_handle: Any = None
        self._open_records: list[RawCaptureEnvelopeV1] = []
        self._high_water = -1
        self._latest_manifest: LiveManifestV1 | None = None
        self._closed_manifest: LiveManifestV1 | None = None
        self._lock = threading.RLock()
        self._resume()

    def _resume(self) -> None:
        """Resume only from a verified immutable manifest generation."""

        manifest = load_live_manifest(self.root, expected_capture_id=self.capture_id)
        if manifest is None:
            partials = sorted(self.segments_dir.glob(f"*{PARTIAL_SUFFIX}"))
            if partials:
                raise RuntimeError(
                    "capture spool has an unmanifested partial; run synth-trace repair first"
                )
            return
        self._generation = manifest.generation
        self._latest_manifest = manifest
        self._sealed = list(manifest.segments)
        self._sequence = max((item.sequence for item in manifest.segments), default=0)
        self._high_water = manifest.high_water_ordinal
        partials = sorted(self.segments_dir.glob(f"*{PARTIAL_SUFFIX}"))
        if partials:
            raise RuntimeError(
                "capture spool has an interrupted partial; run synth-trace repair before resuming"
            )

    # -- writing -----------------------------------------------------------------

    def append(self, envelope: RawCaptureEnvelopeV1) -> RawCaptureEnvelopeV1:
        """Append one raw record. The record is durable before this returns."""

        with self._lock:
            if self._closed_manifest is not None:
                raise RuntimeError("capture spool is closed")
            assert_no_secrets(
                envelope.payload,
                where=f"raw envelope {envelope.envelope_id}",
            )
            if envelope.capture_id != self.capture_id:
                raise ValueError(
                    f"envelope capture_id {envelope.capture_id!r} does not match "
                    f"{self.capture_id!r}"
                )
            if int(envelope.ordinal) <= self._high_water:
                raise ValueError(
                    f"envelope ordinal {envelope.ordinal} is not above high-water "
                    f"{self._high_water}"
                )
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

        with self._lock:
            if self._closed_manifest is not None:
                return None
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
        with self._lock:
            if self._closed_manifest is not None:
                return self._closed_manifest
            self.rotate()
            self._closed_manifest = self._publish_manifest()
            return self._closed_manifest

    def freeze_existing(self) -> LiveManifestV1:
        """Seal a resumed terminal authority without publishing a new generation."""

        with self._lock:
            if self._closed_manifest is not None:
                return self._closed_manifest
            if self._open_handle is not None or self._open_records:
                raise RuntimeError("cannot freeze a capture spool with an open segment")
            if self._latest_manifest is None:
                raise RuntimeError("cannot freeze a capture spool without a manifest")
            self._closed_manifest = self._latest_manifest
            return self._closed_manifest

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
        ).sealed()
        manifests_dir = self.root / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        generation_path = manifests_dir / (
            f"{self._generation:06d}-{short_digest(manifest.content_digest)}.json"
        )
        _write_immutable(generation_path, canonical_text(manifest).encode("utf-8") + b"\n")
        pointer = LiveManifestPointerV1(
            capture_id=self.capture_id,
            generation=self._generation,
            manifest_digest=manifest.content_digest,
            relative_path=str(generation_path.relative_to(self.root)),
            updated_at=manifest.updated_at,
        )
        latest = self.root / "latest.json"
        temp = latest.with_suffix(".json.tmp")
        temp.write_text(canonical_text(pointer) + "\n", encoding="utf-8")
        temp.replace(latest)
        self._latest_manifest = manifest
        return manifest

    # -- reading -----------------------------------------------------------------

    @property
    def segments(self) -> tuple[TraceSegmentManifestV1, ...]:
        with self._lock:
            return tuple(self._sealed)

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed_manifest is not None

    @property
    def high_water_ordinal(self) -> int:
        with self._lock:
            return self._high_water

    def records(self) -> Iterator[dict[str, Any]]:
        yield from read_segments(self.root, self.segments)


def read_segments(
    root: Path,
    segments: tuple[TraceSegmentManifestV1, ...],
) -> Iterator[dict[str, Any]]:
    """Read raw records back in canonical segment order, verifying each digest."""

    root = Path(root).resolve()
    previous_sequence = 0
    previous_ordinal = -1
    expected_capture_id: str | None = None
    for segment in sorted(segments, key=lambda item: item.sequence):
        if segment.content_digest != content_digest(segment):
            raise ValueError(f"segment manifest digest mismatch for {segment.segment_id}")
        if segment.sequence <= previous_sequence:
            raise ValueError(f"segment sequence is not strictly increasing: {segment.sequence}")
        previous_sequence = segment.sequence
        if expected_capture_id is None:
            expected_capture_id = segment.capture_id
        elif segment.capture_id != expected_capture_id:
            raise ValueError("segment manifests mix capture ids")
        path = (root / segment.relative_path).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"segment path escapes capture root: {segment.relative_path}")
        payload = path.read_bytes()
        actual = bytes_digest(payload)
        if actual != segment.digest:
            raise ValueError(
                f"segment digest mismatch for {segment.relative_path}: "
                f"expected {segment.digest}, found {actual}"
            )
        if len(payload) != segment.byte_size:
            raise ValueError(f"segment byte size mismatch for {segment.relative_path}")
        records: list[dict[str, Any]] = []
        for line in payload.decode("utf-8").splitlines():
            if line.strip():
                loaded = json.loads(line)
                if not isinstance(loaded, dict):
                    raise ValueError("raw segment line must be a JSON object")
                envelope = validate_envelope_payload(
                    loaded,
                    expected_capture_id=segment.capture_id,
                    previous_ordinal=previous_ordinal,
                )
                previous_ordinal = envelope.ordinal
                records.append(loaded)
        if len(records) != segment.record_count:
            raise ValueError(f"segment record count mismatch for {segment.relative_path}")
        if records:
            if int(records[0]["ordinal"]) != segment.first_ordinal:
                raise ValueError(f"segment first ordinal mismatch for {segment.relative_path}")
            if int(records[-1]["ordinal"]) != segment.last_ordinal:
                raise ValueError(f"segment last ordinal mismatch for {segment.relative_path}")
            if str(records[0]["occurred_at"]) != segment.first_occurred_at:
                raise ValueError(f"segment first timestamp mismatch for {segment.relative_path}")
            if str(records[-1]["occurred_at"]) != segment.last_occurred_at:
                raise ValueError(f"segment last timestamp mismatch for {segment.relative_path}")
        yield from records


def load_live_manifest(
    root: Path,
    *,
    expected_capture_id: str | None = None,
) -> LiveManifestV1 | None:
    """Load and verify the immutable generation named by ``latest.json``."""

    root = Path(root)
    latest = root / "latest.json"
    if not latest.exists():
        return None
    payload = json.loads(latest.read_text(encoding="utf-8"))
    if "relative_path" in payload:
        pointer = LiveManifestPointerV1(
            capture_id=str(payload["capture_id"]),
            generation=int(payload["generation"]),
            manifest_digest=str(payload["manifest_digest"]),
            relative_path=str(payload["relative_path"]),
            updated_at=str(payload["updated_at"]),
            schema_version=str(
                payload.get("schema_version") or "synth.trace-live-manifest-pointer.v1"
            ),
        )
        manifest_payload = json.loads(
            _safe_relative_path(root, pointer.relative_path).read_text(encoding="utf-8")
        )
        actual = content_digest(manifest_payload)
        if actual != pointer.manifest_digest:
            raise ValueError(
                f"live manifest digest mismatch: expected {pointer.manifest_digest}, "
                f"found {actual}"
            )
    else:
        # Push-1 bundles wrote the manifest directly to latest.json.
        manifest_payload = payload
    segments = tuple(_segment_from_payload(item) for item in manifest_payload.get("segments") or [])
    manifest = LiveManifestV1(
        capture_id=str(manifest_payload["capture_id"]),
        generation=int(manifest_payload["generation"]),
        updated_at=str(manifest_payload["updated_at"]),
        segments=segments,
        high_water_ordinal=int(manifest_payload.get("high_water_ordinal", -1)),
        open_partial=manifest_payload.get("open_partial"),
        schema_version=str(
            manifest_payload.get("schema_version") or LIVE_MANIFEST_SCHEMA_VERSION
        ),
        metadata=dict(manifest_payload.get("metadata") or {}),
        content_digest=str(manifest_payload.get("content_digest") or ""),
    )
    if manifest.content_digest and content_digest(manifest) != manifest.content_digest:
        raise ValueError("live manifest content digest does not match its payload")
    if expected_capture_id and manifest.capture_id != expected_capture_id:
        raise ValueError(
            f"live manifest capture_id {manifest.capture_id!r} does not match "
            f"{expected_capture_id!r}"
        )
    return manifest


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
    segment: TraceSegmentManifestV1 | None = None
    live_manifest_digest: str | None = None


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
    if len(partials) > 1:
        return SpoolRepairV1(
            capture_id=capture_id,
            repaired=False,
            reason="multiple_partial_segments_require_manual_review",
            repaired_at=utc_now(),
        )
    partial = partials[0]
    payload = partial.read_bytes()
    text = payload.decode("utf-8", errors="replace")
    previous = load_live_manifest(Path(root), expected_capture_id=capture_id)
    previous_ordinal = previous.high_water_ordinal if previous else -1
    complete: list[str] = []
    consumed = 0
    for line in text.splitlines(keepends=True):
        if not line.endswith("\n"):
            break
        stripped = line.strip()
        if stripped:
            try:
                loaded = json.loads(stripped)
                if not isinstance(loaded, dict):
                    raise ValueError("raw envelope line is not an object")
                envelope = validate_envelope_payload(
                    loaded,
                    expected_capture_id=capture_id,
                    previous_ordinal=previous_ordinal,
                )
            except (json.JSONDecodeError, ValueError):
                break
            previous_ordinal = envelope.ordinal
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
    _write_immutable(final_path, recovered)
    partial.unlink(missing_ok=True)
    recovered_payloads = [json.loads(line) for line in complete]
    segment = TraceSegmentManifestV1(
        segment_id=f"seg_{short_digest(digest, length=16)}",
        capture_id=capture_id,
        sequence=sequence,
        digest=digest,
        record_count=len(recovered_payloads),
        byte_size=len(recovered),
        first_ordinal=int(recovered_payloads[0]["ordinal"]),
        last_ordinal=int(recovered_payloads[-1]["ordinal"]),
        first_occurred_at=str(recovered_payloads[0]["occurred_at"]),
        last_occurred_at=str(recovered_payloads[-1]["occurred_at"]),
        relative_path=f"segments/{final_path.name}",
        truncated_tail_bytes=truncated,
    ).sealed()
    prior_segments = tuple(previous.segments) if previous else ()
    if any(item.sequence == segment.sequence and item.digest != segment.digest for item in prior_segments):
        raise ValueError(f"repair would replace sealed segment sequence {segment.sequence}")
    segments = tuple(
        sorted(
            {
                item.digest: item
                for item in (*prior_segments, segment)
            }.values(),
            key=lambda item: item.sequence,
        )
    )
    generation = (previous.generation if previous else 0) + 1
    live = LiveManifestV1(
        capture_id=capture_id,
        generation=generation,
        updated_at=utc_now(),
        segments=segments,
        high_water_ordinal=max(item.last_ordinal for item in segments),
        open_partial=None,
        metadata={"repair": True, "truncated_tail_bytes": truncated},
    ).sealed()
    manifests_dir = Path(root) / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    generation_path = manifests_dir / (
        f"{generation:06d}-{short_digest(live.content_digest)}.json"
    )
    _write_immutable(generation_path, canonical_text(live).encode("utf-8") + b"\n")
    pointer = LiveManifestPointerV1(
        capture_id=capture_id,
        generation=generation,
        manifest_digest=live.content_digest,
        relative_path=str(generation_path.relative_to(Path(root))),
        updated_at=live.updated_at,
    )
    latest = Path(root) / "latest.json"
    temp = latest.with_suffix(".json.tmp")
    temp.write_text(canonical_text(pointer) + "\n", encoding="utf-8")
    temp.replace(latest)
    return SpoolRepairV1(
        capture_id=capture_id,
        repaired=True,
        recovered_records=len(complete),
        truncated_tail_bytes=truncated,
        segment_digest=digest,
        reason="promoted_complete_leading_records",
        repaired_at=utc_now(),
        segment=segment,
        live_manifest_digest=live.content_digest,
    )


def _segment_from_payload(payload: dict[str, Any]) -> TraceSegmentManifestV1:
    return TraceSegmentManifestV1(
        segment_id=str(payload["segment_id"]),
        capture_id=str(payload["capture_id"]),
        sequence=int(payload["sequence"]),
        digest=str(payload["digest"]),
        record_count=int(payload["record_count"]),
        byte_size=int(payload["byte_size"]),
        first_ordinal=int(payload["first_ordinal"]),
        last_ordinal=int(payload["last_ordinal"]),
        first_occurred_at=str(payload["first_occurred_at"]),
        last_occurred_at=str(payload["last_occurred_at"]),
        relative_path=str(payload["relative_path"]),
        encoding=str(payload.get("encoding") or "jsonl"),
        compression=str(payload.get("compression") or "none"),
        media_type=str(payload.get("media_type") or "application/x-ndjson"),
        truncated_tail_bytes=int(payload.get("truncated_tail_bytes") or 0),
        schema_version=str(payload.get("schema_version") or SEGMENT_MANIFEST_SCHEMA_VERSION),
        content_digest=str(payload.get("content_digest") or ""),
    )


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"immutable trace object already exists with other bytes: {path}")
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(payload)
    temp.replace(path)
    path.chmod(0o444)


def _safe_relative_path(root: Path, relative: str) -> Path:
    candidate = PurePosixPath(relative)
    if (
        not relative
        or "\\" in relative
        or "\x00" in relative
        or candidate.is_absolute()
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError(f"unsafe capture path: {relative!r}")
    resolved_root = Path(root).resolve()
    resolved = (resolved_root / candidate.as_posix()).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"capture path escapes root: {relative!r}")
    return resolved


__all__ = [
    "LIVE_MANIFEST_SCHEMA_VERSION",
    "SEGMENT_MANIFEST_SCHEMA_VERSION",
    "LiveManifestV1",
    "LiveManifestPointerV1",
    "RawSpool",
    "SpoolRepairV1",
    "TraceSegmentManifestV1",
    "read_segments",
    "load_live_manifest",
    "repair",
]
