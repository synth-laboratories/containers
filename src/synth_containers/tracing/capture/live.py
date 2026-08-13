"""Durable live readers for raw Trace V5 capture envelopes."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from time import sleep

from synth_containers.serde import JsonDataclassMixin

from ..canonical import content_digest
from .envelope import RawCaptureEnvelopeV1, validate_envelope_payload
from .spool import (
    PARTIAL_SUFFIX,
    RawSpool,
    RawSpoolSnapshotV1,
    load_live_manifest,
    read_segments,
)


LIVE_TRACE_PAGE_SCHEMA_VERSION = "synth.trace-live-page.v1"
LIVE_TRACE_STATUS_SCHEMA_VERSION = "synth.trace-live-status.v1"


@dataclass(frozen=True, slots=True)
class LiveTraceStatusV1(JsonDataclassMixin):
    capture_id: str
    generation: int
    high_water_ordinal: int
    sealed_record_count: int
    open_record_count: int
    open_partial: str | None
    closed: bool
    schema_version: str = LIVE_TRACE_STATUS_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class LiveTracePageV1(JsonDataclassMixin):
    capture_id: str
    after_ordinal: int
    high_water_ordinal: int
    next_after_ordinal: int
    records: tuple[RawCaptureEnvelopeV1, ...]
    closed: bool
    manifest_generation: int
    schema_version: str = LIVE_TRACE_PAGE_SCHEMA_VERSION
    content_digest: str = ""

    def sealed(self) -> "LiveTracePageV1":
        return replace(self, content_digest=content_digest(self))


def status_from_spool(spool: RawSpool) -> LiveTraceStatusV1:
    return _status_from_snapshot(spool.snapshot())


def page_from_spool(
    spool: RawSpool,
    *,
    after_ordinal: int = -1,
    limit: int = 256,
) -> LiveTracePageV1:
    snapshot = spool.snapshot()
    records = spool.read_after(after_ordinal, limit=limit)
    final_snapshot = spool.snapshot()
    next_after = records[-1].ordinal if records else after_ordinal
    return LiveTracePageV1(
        capture_id=final_snapshot.capture_id,
        after_ordinal=after_ordinal,
        high_water_ordinal=final_snapshot.high_water_ordinal,
        next_after_ordinal=next_after,
        records=records,
        closed=final_snapshot.closed,
        manifest_generation=max(snapshot.generation, final_snapshot.generation),
    ).sealed()


def read_live_page(
    root: Path,
    *,
    expected_capture_id: str | None = None,
    after_ordinal: int = -1,
    limit: int = 256,
) -> LiveTracePageV1:
    """Read sealed segments and the complete prefix of the active partial.

    Rotation can replace the partial between reads, so the filesystem snapshot is
    retried when its manifest generation changes. Corrupt complete lines fail
    loudly; only a final non-newline-terminated write is considered in flight.
    """

    capture_root = Path(root)
    normalized_limit = max(1, min(int(limit), 10_000))
    for attempt in range(3):
        manifest = load_live_manifest(
            capture_root,
            expected_capture_id=expected_capture_id,
        )
        capture_id = (
            manifest.capture_id
            if manifest is not None
            else expected_capture_id
        )
        if not capture_id:
            raise ValueError(
                "capture id is required before the first live manifest is published"
            )
        records = _read_filesystem_records(
            capture_root,
            capture_id=capture_id,
            manifest=manifest,
        )
        refreshed = load_live_manifest(
            capture_root,
            expected_capture_id=capture_id,
        )
        if (
            (manifest is None and refreshed is not None)
            or (
                manifest is not None
                and refreshed is not None
                and manifest.generation != refreshed.generation
            )
        ):
            if attempt < 2:
                continue
        selected = tuple(
            item for item in records if item.ordinal > after_ordinal
        )[:normalized_limit]
        high_water = records[-1].ordinal if records else (
            refreshed.high_water_ordinal if refreshed is not None else -1
        )
        active_manifest = refreshed or manifest
        closed = bool(
            active_manifest
            and active_manifest.metadata.get("closed") is True
            and not _partial_paths(capture_root)
        )
        return LiveTracePageV1(
            capture_id=capture_id,
            after_ordinal=after_ordinal,
            high_water_ordinal=high_water,
            next_after_ordinal=selected[-1].ordinal if selected else after_ordinal,
            records=selected,
            closed=closed,
            manifest_generation=active_manifest.generation if active_manifest else 0,
        ).sealed()
    raise RuntimeError("live capture changed continuously while reading")


def follow_live_pages(
    root: Path,
    *,
    expected_capture_id: str | None = None,
    after_ordinal: int = -1,
    limit: int = 256,
    poll_seconds: float = 0.25,
):
    """Yield non-empty filesystem pages until the durable capture closes."""

    cursor = after_ordinal
    while True:
        page = read_live_page(
            root,
            expected_capture_id=expected_capture_id,
            after_ordinal=cursor,
            limit=limit,
        )
        if page.records:
            cursor = page.next_after_ordinal
            yield page
            continue
        if page.closed:
            return
        sleep(max(0.01, float(poll_seconds)))


def parse_live_cursor(value: str | None, *, capture_id: str) -> int:
    if not value:
        return -1
    cursor = value.strip()
    if ":" in cursor:
        cursor_capture_id, ordinal_text = cursor.rsplit(":", 1)
        if cursor_capture_id != capture_id:
            raise ValueError("live cursor capture_id does not match this capture")
    else:
        ordinal_text = cursor
    ordinal = int(ordinal_text)
    if ordinal < -1:
        raise ValueError("live cursor ordinal must be at least -1")
    return ordinal


def sse_frame(envelope: RawCaptureEnvelopeV1) -> bytes:
    payload = json.dumps(
        envelope.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        f"id: {envelope.capture_id}:{envelope.ordinal}\n"
        "event: trace\n"
        f"data: {payload}\n\n"
    ).encode("utf-8")


def _status_from_snapshot(snapshot: RawSpoolSnapshotV1) -> LiveTraceStatusV1:
    return LiveTraceStatusV1(
        capture_id=snapshot.capture_id,
        generation=snapshot.generation,
        high_water_ordinal=snapshot.high_water_ordinal,
        sealed_record_count=snapshot.sealed_record_count,
        open_record_count=snapshot.open_record_count,
        open_partial=snapshot.open_partial,
        closed=snapshot.closed,
    )


def _read_filesystem_records(
    root: Path,
    *,
    capture_id: str,
    manifest,
) -> tuple[RawCaptureEnvelopeV1, ...]:
    records: list[RawCaptureEnvelopeV1] = []
    previous_ordinal = -1
    if manifest is not None:
        for payload in read_segments(root, manifest.segments):
            envelope = validate_envelope_payload(
                payload,
                expected_capture_id=capture_id,
                previous_ordinal=previous_ordinal,
            )
            records.append(envelope)
            previous_ordinal = envelope.ordinal
    partials = _partial_paths(root)
    if len(partials) > 1:
        raise ValueError("capture has multiple active partial segments")
    if partials:
        payload = partials[0].read_bytes()
        complete_prefix = payload[: payload.rfind(b"\n") + 1]
        try:
            text = complete_prefix.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("active partial contains invalid UTF-8") from exc
        for line in text.splitlines():
            if not line.strip():
                continue
            loaded = json.loads(line)
            if not isinstance(loaded, dict):
                raise ValueError("active partial line must be a JSON object")
            envelope = validate_envelope_payload(
                loaded,
                expected_capture_id=capture_id,
                previous_ordinal=previous_ordinal,
            )
            records.append(envelope)
            previous_ordinal = envelope.ordinal
    return tuple(records)


def _partial_paths(root: Path) -> tuple[Path, ...]:
    return tuple(sorted((Path(root) / "segments").glob(f"*{PARTIAL_SUFFIX}")))


__all__ = [
    "LIVE_TRACE_PAGE_SCHEMA_VERSION",
    "LIVE_TRACE_STATUS_SCHEMA_VERSION",
    "LiveTracePageV1",
    "LiveTraceStatusV1",
    "follow_live_pages",
    "page_from_spool",
    "parse_live_cursor",
    "read_live_page",
    "sse_frame",
    "status_from_spool",
]
