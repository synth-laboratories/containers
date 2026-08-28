"""Promote durable Harbor rollout logs into portable Trace V5 archives.

Harbor's in-process event journal is the primary local evidence record.  The
HTTP-compatible ``/trace`` endpoint intentionally keeps its compact seal for
stream reconciliation, while this module turns the same terminal evidence into
a self-contained bundle that can be independently validated and inspected by
Workshop (or ``synth-trace``) without a running container.
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..event_log import RolloutEventLog
from ..tracing.canonical import bytes_digest, canonical_bytes, record_id
from ..tracing.capture.binding import (
    BindingCaptureV1,
    BindingContainerV1,
    BindingContextV1,
    BindingWorkloadV1,
    CaptureMode,
    CapturePolicyV1,
    Interception,
    TokenCaptureLevel,
    WorkloadKind,
    mint_binding,
)
from ..tracing.capture.redaction import assert_no_secrets, redact_payload
from ..tracing.models.actors import (
    ActorKind,
    ActorV5,
    CoverageState,
    SessionCoverageV5,
    SessionStatus,
    SessionV5,
)
from ..tracing.models.completeness import (
    CaptureStatus,
    TerminationV5,
    TraceCompletenessV5,
    TraceLifecycleV5,
    TraceStatus,
)
from ..tracing.models.document import TraceCaptureSummaryV5, TraceDocumentV5
from ..tracing.models.artifacts import ArtifactRefV5, ArtifactRole
from ..tracing.models.events import EventOrderV1, EventStatus, EventType, EventV5
from ..tracing.models.identity import AliasV1, TraceIdentityV5, TraceKind, TraceProvenanceV5
from ..tracing.models.spans import UsageProvenance, UsageV5
from ..tracing.inspection import inspect_trace_input
from ..tracing.store.bundle import LocalTraceBundle
from .targets import TargetSpec


PROMOTION_SCHEMA = "synth.containers.harbor-trace-promotion.v1"
MAX_FRAME_BYTES = 16 * 1024 * 1024
MAX_FRAME_DIMENSION = 8192
MAX_FRAME_PIXELS = 4096 * 4096


@dataclass(frozen=True, slots=True)
class HarborTraceBundleRef:
    """The immutable local archive Workshop fetches by rollout identity."""

    archive_path: Path
    trace_id: str
    trace_digest: str
    bundle_digest: str
    archive_digest: str
    byte_size: int


def materialize_harbor_trace_bundle(
    *,
    output_path: Path,
    spec: TargetSpec,
    log: RolloutEventLog,
    seal: dict[str, Any],
    pin: dict[str, Any],
    status: str,
) -> HarborTraceBundleRef:
    """Write a redacted, self-contained archive from a terminal Harbor log.

    The compact seal and journal remain untouched.  A failed promotion therefore
    never turns a completed evaluation into lost evidence; callers may retry the
    deterministic promotion from the same durable log on process recovery.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source = _source_payload(log=log, seal=seal, pin=pin, spec=spec, status=status)
    original_bytes = canonical_bytes(source)
    redacted_source, redaction = redact_payload(source)
    assert_no_secrets(redacted_source, where="Harbor trace-bundle promotion")
    redacted_bytes = canonical_bytes(redacted_source)

    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.stem}.build-", dir=output_path.parent
    ) as temporary:
        bundle = LocalTraceBundle(Path(temporary) / "bundle", bundle_id=f"harbor-{log.rollout_id}")
        source_blob_digest = bundle.blobs.put(redacted_bytes)
        frame_artifacts, event_artifact_ids = _retain_frame_artifacts(
            bundle=bundle, log=log, source=redacted_source
        )
        document, binding = _document_from_source(
            source=redacted_source,
            source_blob_digest=source_blob_digest,
            original_source_digest=bytes_digest(original_bytes),
            stored_source_digest=bytes_digest(redacted_bytes),
            redaction=redaction.to_dict(),
            spec=spec,
            status=status,
            frame_artifacts=frame_artifacts,
            event_artifact_ids=event_artifact_ids,
        )
        bundle.write_binding(binding)
        bundle.write_trace(document, binding=binding, segments=())
        bundle.write_receipt(
            "harbor-rollout-promotion",
            {
                "schema_version": PROMOTION_SCHEMA,
                "rollout_id": log.rollout_id,
                "trace_id": document.trace_id,
                "trace_digest": document.content_digest,
                "lite_seal_digest": seal.get("content_digest"),
                "original_source_digest": bytes_digest(original_bytes),
                "stored_source_digest": bytes_digest(redacted_bytes),
                "source_blob_digest": source_blob_digest,
                "redaction": redaction.to_dict(),
                "event_count": len(document.events),
                "high_water": log.high_water,
                "frame_artifact_count": len(frame_artifacts),
                "frame_reference_count": len(event_artifact_ids),
            },
        )
        manifest = bundle.write_manifest(
            metadata={
                "promotion_schema": PROMOTION_SCHEMA,
                "producer": "synth-containers-harbor",
                "rollout_id": log.rollout_id,
                "lite_seal_digest": seal.get("content_digest"),
                "source_blob_digest": source_blob_digest,
                "original_source_digest": bytes_digest(original_bytes),
                "stored_source_digest": bytes_digest(redacted_bytes),
                "redaction": redaction.to_dict(),
                "frame_artifact_count": len(frame_artifacts),
                "frame_reference_count": len(event_artifact_ids),
            }
        )
        _verify_frame_artifact_bindings(bundle, document.to_dict())
        archive = bundle.archive_bytes()

    _atomic_write_bytes(output_path, archive)
    return HarborTraceBundleRef(
        archive_path=output_path,
        trace_id=document.trace_id,
        trace_digest=document.content_digest,
        bundle_digest=manifest.content_digest,
        archive_digest=bytes_digest(archive),
        byte_size=len(archive),
    )


def inspect_harbor_trace_bundle(archive_path: Path) -> HarborTraceBundleRef:
    """Verify a persisted archive enough to safely announce it to Workshop."""

    archive_path = Path(archive_path)
    archive = archive_path.read_bytes()
    inspection = inspect_trace_input(archive_path)
    if not (
        inspection.compatibility == "native"
        and inspection.self_contained is True
        and inspection.trusted
        and inspection.validation.valid
        and len(inspection.traces) == 1
    ):
        raise ValueError("harbor_trace_bundle_inspection_invalid")
    with tempfile.TemporaryDirectory(prefix=".harbor-trace-inspect-") as temporary:
        bundle = LocalTraceBundle.extract_archive(
            archive_path,
            Path(temporary) / "bundle",
        )
        valid, errors = bundle.verify_self_contained()
        if not valid:
            raise ValueError(f"harbor_trace_bundle_not_self_contained:{errors}")
        manifest = bundle.read_manifest()
        trace = inspection.traces[0]
        _verify_frame_artifact_bindings(bundle, bundle.read_trace(trace.trace_digest))
        return HarborTraceBundleRef(
            archive_path=archive_path,
            trace_id=trace.trace_id,
            trace_digest=trace.trace_digest,
            bundle_digest=str(manifest["content_digest"]),
            archive_digest=bytes_digest(archive),
            byte_size=len(archive),
        )


def _source_payload(
    *,
    log: RolloutEventLog,
    seal: dict[str, Any],
    pin: dict[str, Any],
    spec: TargetSpec,
    status: str,
) -> dict[str, Any]:
    events = []
    for item in log.after(0):
        if item.control:
            continue
        events.append(
            {
                "sequence": item.sequence,
                "kind": item.kind,
                "occurred_at": item.ts,
                "payload": _json_value(item.payload),
                "digest": item.digest,
            }
        )
    return {
        "schema_version": PROMOTION_SCHEMA,
        "rollout_id": log.rollout_id,
        "stream_id": log.stream_id,
        "target_id": spec.target_id,
        "status": status,
        "high_water": log.high_water,
        "lite_seal_digest": seal.get("content_digest"),
        "pin": _json_value(pin),
        "events": events,
    }


def _json_value(value: Any) -> Any:
    """Normalize event payloads to canonical JSON without retaining object reprs."""

    return json.loads(json.dumps(value, default=str, ensure_ascii=False, allow_nan=False))


def _retain_frame_artifacts(
    *, bundle: LocalTraceBundle, log: RolloutEventLog, source: dict[str, Any]
) -> tuple[tuple[ArtifactRefV5, ...], dict[int, tuple[str, ...]]]:
    """Embed every declared native PNG and bind its event to an artifact."""

    by_digest: dict[str, dict[str, Any]] = {}
    event_digests: dict[int, str] = {}
    rollout_id = str(source["rollout_id"])
    for item in list(source.get("events") or []):
        if str(item.get("kind") or "") != "frame":
            continue
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        if payload.get("format") != "png":
            continue
        sequence = item.get("sequence")
        step = payload.get("step")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise ValueError("native_frame_sequence_invalid")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError(f"native_frame_step_invalid:sequence={sequence}")
        expected_url = f"/rollouts/{rollout_id}/frames/{step}.png"
        if payload.get("url") != expected_url:
            raise ValueError(f"native_frame_url_invalid:sequence={sequence}:expected={expected_url}")
        if log.journal_path is None:
            raise ValueError(f"native_frame_artifact_missing:sequence={sequence}:step={step}")
        path = RolloutEventLog.frame_asset_path(
            log.journal_path.parent.parent, rollout_id, step
        )
        if not path.is_file():
            raise ValueError(f"native_frame_artifact_missing:sequence={sequence}:step={step}")
        body = path.read_bytes()
        width, height = _png_dimensions(body, sequence=sequence)
        stored = bundle.blobs.put_if_absent(
            body,
            media_type="image/png",
            metadata={"rollout_id": rollout_id, "step": str(step)},
        )
        digest = stored.digest
        event_digests[sequence] = digest
        entry = by_digest.setdefault(
            digest,
            {
                "size_bytes": len(body),
                "width": width,
                "height": height,
                "uri": stored.metadata.uri,
                "steps": [],
                "producer_digests": [],
                "produced_at": item.get("occurred_at"),
            },
        )
        entry["steps"].append(step)
        producer_digest = item.get("digest")
        if isinstance(producer_digest, str) and producer_digest not in entry["producer_digests"]:
            entry["producer_digests"].append(producer_digest)

    artifact_by_digest: dict[str, ArtifactRefV5] = {}
    for digest, entry in sorted(by_digest.items()):
        artifact = ArtifactRefV5(
            artifact_id=record_id(
                "artifact",
                kind="environment_frame_png",
                scope=(rollout_id,),
                key=digest,
            ),
            digest=digest,
            media_type="image/png",
            size_bytes=int(entry["size_bytes"]),
            role=ArtifactRole.SCREENSHOT,
            uri=str(entry["uri"]),
            producer="synth-containers",
            source_authority="durable_rollout_frame_asset",
            produced_at=str(entry["produced_at"] or "") or None,
            logical_name=f"frames/{digest.partition(':')[2]}.png",
            metadata={
                "environment_frame": True,
                "rollout_id": rollout_id,
                "steps": sorted(set(entry["steps"])),
                "width": int(entry["width"]),
                "height": int(entry["height"]),
                "producer_digests": entry["producer_digests"],
            },
        )
        artifact_by_digest[digest] = artifact
    return (
        tuple(artifact_by_digest[digest] for digest in sorted(artifact_by_digest)),
        {
            sequence: (artifact_by_digest[digest].artifact_id,)
            for sequence, digest in event_digests.items()
        },
    )


def _verify_frame_artifact_bindings(
    bundle: LocalTraceBundle, trace: dict[str, Any]
) -> None:
    """Reject URL-only or dangling native frames, including legacy cached bundles."""

    artifacts = {
        str(item.get("artifact_id")): item
        for item in trace.get("artifacts") or []
        if isinstance(item, dict) and item.get("artifact_id")
    }
    verified: set[str] = set()
    for event in trace.get("events") or []:
        if not isinstance(event, dict):
            continue
        detail = event.get("detail") if isinstance(event.get("detail"), dict) else {}
        if event.get("event_type") != "frame" or detail.get("format") != "png":
            continue
        artifact_ids = event.get("artifact_ids") or []
        if not artifact_ids:
            raise ValueError("harbor_trace_bundle_native_frame_not_embedded")
        for artifact_id in artifact_ids:
            artifact = artifacts.get(str(artifact_id))
            if artifact is None:
                raise ValueError(
                    f"harbor_trace_bundle_frame_artifact_missing:{artifact_id}"
                )
            if artifact.get("media_type") != "image/png":
                raise ValueError(
                    f"harbor_trace_bundle_frame_media_type_invalid:{artifact_id}"
                )
            digest = str(artifact.get("digest") or "")
            expected_uri = bundle.blobs.uri(digest)
            if artifact.get("uri") != expected_uri:
                raise ValueError(
                    f"harbor_trace_bundle_frame_uri_invalid:{artifact_id}"
                )
            if digest in verified:
                continue
            body = bundle.blobs.get(digest)
            if artifact.get("size_bytes") != len(body):
                raise ValueError(
                    f"harbor_trace_bundle_frame_size_invalid:{artifact_id}"
                )
            _png_dimensions(body, sequence=int(event.get("sequence") or 0))
            verified.add(digest)


def _png_dimensions(body: bytes, *, sequence: int) -> tuple[int, int]:
    if len(body) > MAX_FRAME_BYTES:
        raise ValueError(f"native_frame_too_large:sequence={sequence}:bytes={len(body)}")
    if len(body) < 24 or body[:8] != b"\x89PNG\r\n\x1a\n" or body[12:16] != b"IHDR":
        raise ValueError(f"native_frame_png_invalid:sequence={sequence}")
    width, height = struct.unpack(">II", body[16:24])
    if (
        width < 1
        or height < 1
        or width > MAX_FRAME_DIMENSION
        or height > MAX_FRAME_DIMENSION
        or width * height > MAX_FRAME_PIXELS
    ):
        raise ValueError(
            f"native_frame_dimensions_invalid:sequence={sequence}:width={width}:height={height}"
        )
    offset = 8
    compressed = bytearray()
    saw_iend = False
    while offset + 12 <= len(body):
        length = struct.unpack(">I", body[offset : offset + 4])[0]
        end = offset + 12 + length
        if end > len(body):
            raise ValueError(f"native_frame_png_truncated:sequence={sequence}")
        kind = body[offset + 4 : offset + 8]
        payload = body[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", body[offset + 8 + length : end])[0]
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != expected_crc:
            raise ValueError(f"native_frame_png_crc_invalid:sequence={sequence}")
        if kind == b"IDAT":
            compressed.extend(payload)
        if kind == b"IEND":
            saw_iend = True
            if end != len(body):
                raise ValueError(f"native_frame_png_trailing_bytes:sequence={sequence}")
            break
        offset = end
    if not saw_iend or not compressed:
        raise ValueError(f"native_frame_png_incomplete:sequence={sequence}")
    try:
        zlib.decompress(bytes(compressed))
    except zlib.error as exc:
        raise ValueError(f"native_frame_png_decode_failed:sequence={sequence}") from exc
    return width, height


def _document_from_source(
    *,
    source: dict[str, Any],
    source_blob_digest: str,
    original_source_digest: str,
    stored_source_digest: str,
    redaction: dict[str, Any],
    spec: TargetSpec,
    status: str,
    frame_artifacts: tuple[ArtifactRefV5, ...] = (),
    event_artifact_ids: dict[int, tuple[str, ...]] | None = None,
) -> tuple[TraceDocumentV5, Any]:
    rollout_id = str(source["rollout_id"])
    events_source = list(source.get("events") or [])
    trace_id = rollout_id
    categories = [_event_category(str(item.get("kind") or "")) for item in events_source]
    category_order = tuple(dict.fromkeys(categories or ["orchestrator"]))
    actor_ids = {
        category: record_id("actor", kind="harbor_rollout", scope=(trace_id,), key=category)
        for category in category_order
    }
    session_ids = {
        category: record_id("sess", kind="harbor_rollout", scope=(trace_id, actor_ids[category]), key=category)
        for category in category_order
    }
    capture_id = record_id(
        "cap", kind="harbor_rollout_promotion", scope=(trace_id,), key=stored_source_digest
    )
    pin = source.get("pin") if isinstance(source.get("pin"), dict) else {}
    policy_ref = pin.get("policy_ref") if isinstance(pin.get("policy_ref"), dict) else {}
    model = _first_string(policy_ref, "model", "model_id")
    provider = _first_string(policy_ref, "provider", "provider_id")
    started_at = str(events_source[0]["occurred_at"]) if events_source else "1970-01-01T00:00:00Z"
    ended_at = str(events_source[-1]["occurred_at"]) if events_source else started_at
    coverage = SessionCoverageV5(
        model_calls=CoverageState.PARTIAL,
        agent_events=CoverageState.COMPLETE,
        environment_events=CoverageState.PARTIAL,
        tool_events=CoverageState.PARTIAL,
        usage=CoverageState.NOT_CAPTURED,
        raw_provider=CoverageState.UNAVAILABLE,
        reasons=(
            "promoted from Harbor's durable application-event journal; native environment PNGs are embedded when declared",
        ),
    )
    binding = mint_binding(
        trace_id=trace_id,
        capture_id=capture_id,
        trace_kind=TraceKind.AGENT_ROLLOUT,
        policy=CapturePolicyV1(
            profile="harbor_terminal_promotion",
            raw_capture="durable_application_event_journal",
            token_level=TokenCaptureLevel.NONE,
            retention_class="local_only",
        ),
        workload=BindingWorkloadV1(
            kind=WorkloadKind.OTHER,
            root_actor_id=actor_ids[category_order[0]],
            actor_session_id=session_ids[category_order[0]],
            run_id=rollout_id,
            rollout_id=rollout_id,
            session_id=str(source.get("stream_id") or "") or None,
        ),
        capture=BindingCaptureV1(
            interception=Interception.APPLICATION,
            mode=CaptureMode.OBSERVE_AND_TRANSFORM,
            proxy_profile="harbor_durable_event_journal",
            output_artifact_root="local_only",
        ),
        container=BindingContainerV1(
            container_definition_id=spec.target_id,
            contract_version=PROMOTION_SCHEMA,
        ),
        context=BindingContextV1(
            task_instance_id=_first_string(pin, "task_instance_id"),
        ),
        metadata={
            "source_blob_digest": source_blob_digest,
            "original_source_digest": original_source_digest,
            "stored_source_digest": stored_source_digest,
        },
    )
    actors = tuple(
        ActorV5(
            actor_id=actor_ids[category],
            kind=_actor_kind(category),
            display_name=f"Harbor {category}",
            role=category,
            harness="harbor",
            model=model if category == "agent" else None,
            provider=provider if category == "agent" else None,
            policy_id=_first_string(policy_ref, "policy_id", "id") if category == "agent" else None,
            task_id=spec.target_id,
            visibility="private",
            metadata={"source": "durable_event_journal", "target_id": spec.target_id},
        ).sealed()
        for category in category_order
    )
    sessions = tuple(
        SessionV5(
            session_id=session_ids[category],
            actor_id=actor_ids[category],
            started_at=started_at,
            ended_at=ended_at,
            started_sequence=1 if events_source else None,
            ended_sequence=len(events_source) if events_source else None,
            status=SessionStatus.FAILED if _failed(status) else SessionStatus.COMPLETED,
            harness="harbor",
            provider=provider if category == "agent" else None,
            coverage=coverage,
            metadata={"category": category},
        ).sealed()
        for category in category_order
    )
    event_artifact_ids = event_artifact_ids or {}
    events = tuple(
        EventV5(
            event_id=record_id(
                "evt",
                kind="harbor_rollout_event",
                scope=(trace_id,),
                key={"sequence": item["sequence"], "digest": item.get("digest")},
            ),
            # Trace V5 explicitly preserves producer-defined event types.  A
            # blanket ``application.event`` kept every byte but erased the
            # semantic affordance the rollout inspector uses for Focus mode
            # and readable event titles.
            event_type=str(item.get("kind") or EventType.APPLICATION),
            actor_id=actor_ids[_event_category(str(item.get("kind") or ""))],
            session_id=session_ids[_event_category(str(item.get("kind") or ""))],
            occurred_at=str(item["occurred_at"]),
            order=EventOrderV1(chronological_sequence=int(item["sequence"])),
            payload=_promoted_event_payload(item),
            status=EventStatus.ERROR if _event_failed(str(item.get("kind") or ""), item.get("payload")) else EventStatus.OK,
            artifact_ids=event_artifact_ids.get(int(item["sequence"]), ()),
            aliases=(
                AliasV1(
                    namespace="synth.containers.rollout_event",
                    value=str(item["sequence"]),
                    target_id=trace_id,
                    target_kind="trace",
                ),
            ),
        ).sealed()
        for item in events_source
    )
    trace_status = TraceStatus.FAILED if _failed(status) else TraceStatus.COMPLETED
    document = TraceDocumentV5(
        trace_id=trace_id,
        trace_kind=TraceKind.AGENT_ROLLOUT,
        identity=TraceIdentityV5(
            rollout_id=rollout_id,
            run_id=rollout_id,
            correlation_id=str(source.get("stream_id") or "") or None,
            task_id=spec.target_id,
            task_instance_id=_first_string(pin, "task_instance_id"),
            benchmark=_first_string(pin, "environment_ref"),
        ),
        lifecycle=TraceLifecycleV5(
            status=trace_status,
            started_at=started_at,
            ended_at=ended_at,
            termination=TerminationV5(
                reason="harbor_terminal_rollout",
                detail=status,
            ),
        ),
        capture=TraceCaptureSummaryV5(
            capture_id=capture_id,
            binding_id=binding.binding_id,
            binding_digest=binding.content_digest,
            capture_profile="harbor_terminal_promotion",
            interception="application",
            mode="observe_and_transform",
            segment_digests=(),
            segment_count=0,
            raw_record_count=len(events),
        ),
        provenance=TraceProvenanceV5(
            producer="synth-containers-harbor",
            producer_version="v0.8",
            source_format=PROMOTION_SCHEMA,
            model=model,
            provider=provider,
            harness="harbor",
            captured_at=ended_at,
            transformation_chain=("harbor_terminal_promotion@1",),
            extra={
                "target_id": spec.target_id,
                "lite_seal_digest": source.get("lite_seal_digest"),
                "original_source_digest": original_source_digest,
                "stored_source_digest": stored_source_digest,
                "redaction": redaction,
            },
        ),
        completeness=TraceCompletenessV5(
            capture_status=CaptureStatus.COMPLETE,
            terminal_event_observed=bool(events),
            model_calls=CoverageState.PARTIAL,
            raw_provider=CoverageState.UNAVAILABLE,
            agent_events=CoverageState.COMPLETE,
            environment_events=CoverageState.PARTIAL,
            tool_events=CoverageState.PARTIAL,
            usage=CoverageState.NOT_CAPTURED,
            expected_record_count=len(events),
            captured_record_count=len(events),
            high_water_ordinal=int(source.get("high_water") or len(events)),
            reasons=(
                "Every retained Harbor application event is represented; native environment PNGs are embedded when declared, while raw provider transport and token-level usage were unavailable.",
            ),
        ),
        actors=actors,
        sessions=sessions,
        events=events,
        artifacts=frame_artifacts,
        usage=UsageV5(provenance=UsageProvenance.UNAVAILABLE),
        aliases=(
            AliasV1(
                namespace="synth.containers.rollout",
                value=rollout_id,
                target_id=trace_id,
            ),
        ),
        visibility="private",
        extensions={
            "harbor": {
                "promotion_schema": PROMOTION_SCHEMA,
                "source_blob_digest": source_blob_digest,
                "lite_seal_digest": source.get("lite_seal_digest"),
                "event_count": len(events),
                "high_water": source.get("high_water"),
                "frame_artifact_count": len(frame_artifacts),
                "frame_reference_count": len(event_artifact_ids),
            }
        },
    ).sealed()
    return document, binding


def _event_category(kind: str) -> str:
    normalized = kind.lower()
    if "verifier" in normalized or normalized.startswith("reward."):
        return "verifier"
    if normalized.startswith(("trial.", "policy.", "tool.", "stdout", "stderr")):
        return "agent"
    if normalized.startswith(("environment.", "env.", "world.")) or normalized in {
        "observation",
        "frame",
        "action",
        "action_applied",
        "entity_transition",
        "resource_delta",
        "reward_signal",
        "task_resolved",
    }:
        return "environment"
    return "orchestrator"


def _promoted_event_payload(item: dict[str, Any]) -> dict[str, Any]:
    """Keep native fields directly inspectable while retaining source proof."""

    native = item.get("payload")
    payload = dict(native) if isinstance(native, dict) else {"value": native}
    payload["source_event_type"] = str(item.get("kind") or "")
    payload["source_event_digest"] = item.get("digest")
    return payload


def _actor_kind(category: str) -> ActorKind:
    if category == "agent":
        return ActorKind.AGENT
    if category == "verifier":
        return ActorKind.VERIFIER
    if category == "environment":
        return ActorKind.ENVIRONMENT
    return ActorKind.ORCHESTRATOR


def _event_failed(kind: str, payload: Any) -> bool:
    normalized = kind.lower()
    if any(token in normalized for token in ("failed", "error", "refused", "timeout")):
        return True
    return isinstance(payload, dict) and str(payload.get("status") or "").lower() in {
        "failed",
        "error",
        "refused",
        "timeout",
    }


def _failed(status: str) -> bool:
    return status.lower() in {"failed", "error", "refused", "timeout", "truncated"}


def _first_string(value: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    path.chmod(0o444)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
