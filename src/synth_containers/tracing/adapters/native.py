"""Bundle-native importers for Codex, ReAct, and Jesterky application records."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from ..canonical import bytes_digest, canonical_bytes, record_id
from ..capture.binding import (
    BindingCaptureV1,
    BindingWorkloadV1,
    CaptureMode,
    CapturePolicyV1,
    TokenCaptureLevel,
    WorkloadKind,
    mint_binding,
)
from ..capture.collector import LocalCollector
from ..capture.coverage import CaptureScope, new_coverage_receipt
from ..capture.envelope import RawRecordType
from ..capture.finalizer import TraceFinalizer, application_event_id
from ..capture.session import CaptureSession
from ..capture.spool import RawSpool
from ..models.identity import (
    AliasV1,
    TraceIdentityV5,
    TraceKind,
    TraceProvenanceV5,
    mint_actor_id,
    mint_capture_id,
    mint_session_id,
    mint_trace_id,
)
from ..models.document import TraceDocumentV5
from ..models.spans import UsageProvenance, UsageV5
from ..store.bundle import LocalTraceBundle
from .codex_jsonl import import_codex_jsonl


IMPORTED_AT = "1970-01-01T00:00:00Z"


def import_native_to_bundle(
    source: Path,
    *,
    source_format: str,
    bundle: LocalTraceBundle,
) -> dict[str, Any]:
    """Import a supported native application-event record into a sealed bundle."""

    source = Path(source)
    normalized = source_format.lower().strip().replace("-", "_")
    source_bytes = source.read_bytes()
    source_digest = bytes_digest(source_bytes)
    stored_source_digest = bundle.blobs.put(source_bytes)
    if stored_source_digest != source_digest:
        raise ValueError("native source blob digest changed while importing")
    if normalized in {"codex", "codex_jsonl", "codex_stdout_jsonl"}:
        return _import_codex(source, source_digest=source_digest, bundle=bundle)
    if normalized in {"react", "craftax_react", "gamebench_react"}:
        payload = _json_object(source_bytes)
        return _import_events(
            payload,
            source_digest=source_digest,
            source_format="gamebench.react-native-events.v1",
            bundle=bundle,
            workload_kind=WorkloadKind.REACT,
            event_adapter=_react_events,
            identity=TraceIdentityV5(
                correlation_id=str(payload.get("trace_correlation_id") or "") or None,
            ),
        )
    if normalized in {"jesterky", "jesterky_manifest"}:
        payload = _json_object(source_bytes)
        run_id = str(payload.get("run_id") or "")
        return _import_events(
            payload,
            source_digest=source_digest,
            source_format=str(payload.get("schema_version") or "jesterky.run-manifest.v1"),
            bundle=bundle,
            workload_kind=WorkloadKind.JESTERKY,
            event_adapter=_jesterky_events,
            identity=TraceIdentityV5(run_id=run_id or None, correlation_id=run_id or None),
            trace_kind=TraceKind.WORKFLOW_RUN,
        )
    raise ValueError(f"unsupported native trace format: {source_format}")


def write_imported_document(
    document: TraceDocumentV5,
    *,
    source_digest: str,
    source_format: str,
    bundle: LocalTraceBundle,
) -> dict[str, Any]:
    """Bind a foreign canonical document to a portable local bundle."""

    if not document.actors or not document.sessions:
        raise ValueError("imported canonical trace requires at least one actor and session")
    capture_id = document.capture.capture_id
    binding = mint_binding(
        trace_id=document.trace_id,
        capture_id=capture_id,
        trace_kind=document.trace_kind,
        policy=CapturePolicyV1(
            profile=f"imported_{source_format}",
            raw_capture="source_artifact",
            token_level=TokenCaptureLevel.USAGE_ONLY,
        ),
        workload=BindingWorkloadV1(
            kind=WorkloadKind.OTHER,
            root_actor_id=document.actors[0].actor_id,
            actor_session_id=document.sessions[0].session_id,
            run_id=document.identity.run_id,
            rollout_id=document.identity.rollout_id,
        ),
        capture=BindingCaptureV1(
            interception="none",
            mode=CaptureMode.DISABLED,
            proxy_profile="foreign_import",
        ),
    )
    binding = replace(binding, created_at=IMPORTED_AT, content_digest="").sealed()
    rebound = replace(
        document,
        capture=replace(
            document.capture,
            binding_id=binding.binding_id,
            binding_digest=binding.content_digest,
            interception="none",
            mode=CaptureMode.DISABLED,
            segment_digests=(),
            segment_count=0,
            raw_record_count=0,
        ),
        content_digest="",
    ).sealed()
    bundle.write_binding(binding)
    bundle.write_trace(rebound, binding=binding, segments=())
    bundle.write_receipt(
        "foreign-import",
        {
            "source_digest": source_digest,
            "source_format": source_format,
            "trace_id": rebound.trace_id,
            "trace_digest": rebound.content_digest,
        },
    )
    bundle.write_manifest(
        metadata={
            "imported_source_digest": source_digest,
            "imported_source_format": source_format,
        }
    )
    return {
        "trace_id": rebound.trace_id,
        "trace_digest": rebound.content_digest,
        "capture_id": capture_id,
        "source_digest": source_digest,
        "source_format": source_format,
    }


def _import_codex(
    source: Path,
    *,
    source_digest: str,
    bundle: LocalTraceBundle,
) -> dict[str, Any]:
    trace_id = mint_trace_id(kind="imported_codex", key=source_digest)
    actor_id = mint_actor_id(trace_id=trace_id, name=f"imported-{WorkloadKind.CODEX}")
    session_id = mint_session_id(
        trace_id=trace_id,
        actor_id=actor_id,
        nonce=source_digest,
    )
    imported = import_codex_jsonl(source, target_id=session_id)
    events = [
        {
            "event_type": item["event_type"],
            "payload": {
                **dict(item["body"]),
                "native_kind": item["native_kind"],
                "codex_id": item["codex_id"],
            },
            "source_id": str(item["codex_id"] or index),
        }
        for index, item in enumerate(imported.events)
    ]
    usage = _codex_usage(imported.usage_snapshots, source_digest=source_digest)
    result = _assemble_events(
        events,
        trace_id=trace_id,
        source_digest=source_digest,
        source_format="codex.stdout-jsonl",
        bundle=bundle,
        workload_kind=WorkloadKind.CODEX,
        identity=TraceIdentityV5(),
        trace_kind=TraceKind.AGENT_ROLLOUT,
        usage=usage,
        aliases=tuple(imported.aliases),
    )
    return {
        **result,
        "line_count": imported.line_count,
        "malformed_lines": imported.malformed_lines,
        "unknown_kinds": imported.unknown_kinds,
    }


def _import_events(
    payload: Mapping[str, Any],
    *,
    source_digest: str,
    source_format: str,
    bundle: LocalTraceBundle,
    workload_kind: WorkloadKind | str,
    event_adapter: Any,
    identity: TraceIdentityV5,
    trace_kind: TraceKind | str = TraceKind.AGENT_ROLLOUT,
) -> dict[str, Any]:
    trace_id = mint_trace_id(kind=f"imported_{workload_kind}", key=source_digest)
    return _assemble_events(
        event_adapter(payload),
        trace_id=trace_id,
        source_digest=source_digest,
        source_format=source_format,
        bundle=bundle,
        workload_kind=workload_kind,
        identity=identity,
        trace_kind=trace_kind,
    )


def _assemble_events(
    events: list[dict[str, Any]],
    *,
    trace_id: str,
    source_digest: str,
    source_format: str,
    bundle: LocalTraceBundle,
    workload_kind: WorkloadKind | str,
    identity: TraceIdentityV5,
    trace_kind: TraceKind | str,
    usage: UsageV5 | None = None,
    aliases: tuple[AliasV1, ...] = (),
) -> dict[str, Any]:
    capture_id = mint_capture_id(trace_id=trace_id, key=source_digest)
    actor_id = mint_actor_id(trace_id=trace_id, name=f"imported-{workload_kind}")
    session_id = mint_session_id(
        trace_id=trace_id,
        actor_id=actor_id,
        nonce=source_digest,
    )
    policy = CapturePolicyV1(
        profile=f"imported_{workload_kind}",
        raw_capture="source_artifact",
        token_level=(
            TokenCaptureLevel.USAGE_ONLY if usage is not None else TokenCaptureLevel.NONE
        ),
        retention_class="local_only",
    )
    binding = mint_binding(
        trace_id=trace_id,
        capture_id=capture_id,
        trace_kind=trace_kind,
        policy=policy,
        workload=BindingWorkloadV1(
            kind=workload_kind,
            root_actor_id=actor_id,
            actor_session_id=session_id,
            run_id=identity.run_id,
        ),
        capture=BindingCaptureV1(
            interception="none",
            mode=CaptureMode.DISABLED,
            proxy_profile="native_import",
        ),
    )
    binding = replace(binding, created_at=IMPORTED_AT, content_digest="").sealed()
    capture_root = bundle.capture_root(trace_id)
    bundle.write_binding(binding)
    spool = RawSpool(
        capture_root,
        capture_id=capture_id,
        max_segment_records=policy.max_segment_records,
    )
    session = CaptureSession(binding=binding, spool=spool, blobs=bundle.blobs)
    collector = LocalCollector(session)
    envelope_ids: dict[str, str] = {}
    for index, item in enumerate(events):
        source_id = str(item.get("source_id") or index)
        caused_by = tuple(
            envelope_ids[parent]
            for parent in item.get("caused_by") or ()
            if parent in envelope_ids
        )
        envelope_id = collector.event(
            event_type=str(item["event_type"]),
            payload={
                **dict(item.get("payload") or {}),
                "native_source_id": source_id,
            },
            occurred_at=str(item.get("occurred_at") or IMPORTED_AT),
            caused_by=caused_by,
            structural=item.get("structural"),
        )
        envelope_ids[source_id] = envelope_id
    session.append(
        RawRecordType.CAPTURE_FINISHED,
        payload={"reason": "native_import", "source_digest": source_digest},
        occurred_at=IMPORTED_AT,
        producer_version="synth-trace-native-import/1",
    )
    spool.close()
    aliases = (
        *aliases,
        *(
            AliasV1(
                namespace="native.event",
                value=source_id,
                target_id=application_event_id(
                    trace_id=trace_id,
                    envelope_id=envelope_id,
                ),
                target_kind="event",
                provenance="imported",
            )
            for source_id, envelope_id in envelope_ids.items()
        ),
    )
    receipt = new_coverage_receipt(
        binding_id=binding.binding_id,
        binding_digest=binding.content_digest,
        capture_id=capture_id,
        scope=CaptureScope.IMPORTED_AGENT_EVENTS,
        requested_mode=CaptureMode.DISABLED,
        resolved_mode=CaptureMode.DISABLED,
        interception="none",
        proxy_config_digest=binding.capture.proxy_config_digest or "",
    )
    finalizer = TraceFinalizer(
        binding=binding,
        spool_root=capture_root,
        segments=spool.segments,
        provenance=TraceProvenanceV5(
            producer="synth_containers.tracing.adapters.native",
            producer_version="1",
            source_format=source_format,
            captured_at=IMPORTED_AT,
            transformation_chain=("native_application_import@1",),
            extra={"source_digest": source_digest},
        ),
        identity=identity,
        root_actor_name=f"imported {workload_kind}",
    )
    sealed = finalizer.seal(coverage=receipt, aliases=aliases)
    document = sealed.document
    if usage is not None:
        document = replace(document, usage=usage, content_digest="").sealed()
    bundle.write_trace(document, binding=binding, segments=sealed.segments)
    bundle.write_receipt(
        "native-import",
        {
            "source_digest": source_digest,
            "source_format": source_format,
            "trace_id": trace_id,
            "trace_digest": document.content_digest,
            "events": len(events),
        },
    )
    bundle.write_manifest(
        metadata={
            "imported_source_digest": source_digest,
            "imported_source_format": source_format,
        }
    )
    return {
        "trace_id": trace_id,
        "trace_digest": document.content_digest,
        "capture_id": capture_id,
        "actor_id": actor_id,
        "session_id": session_id,
        "events": len(events),
        "source_digest": source_digest,
        "source_format": source_format,
    }


def _react_events(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, item in enumerate(payload.get("events") or ()):
        if not isinstance(item, Mapping):
            continue
        output.append(
            {
                "event_type": str(item.get("event_type") or "react.event"),
                "payload": dict(item.get("payload") or {}),
                "source_id": str(item.get("event_id") or index),
                "caused_by": tuple(str(value) for value in item.get("caused_by") or ()),
                "occurred_at": item.get("occurred_at"),
            }
        )
    return output


def _jesterky_events(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    run_id = str(payload.get("run_id") or "jesterky-import")
    for index, item in enumerate(payload.get("events") or ()):
        if not isinstance(item, Mapping):
            continue
        kind = item.get("kind")
        native_kind = (
            str(kind.get("kind") or "event") if isinstance(kind, Mapping) else str(kind or "event")
        )
        address = item.get("addr") if isinstance(item.get("addr"), Mapping) else {}
        node_path = []
        for part in address.get("node_path") or ():
            if isinstance(part, Mapping):
                node_path.append(str(part.get("node") or part.get("key") or ""))
            else:
                node_path.append(str(part))
        output.append(
            {
                "event_type": f"jesterky.{native_kind}",
                "payload": {
                    **dict(item.get("payload") or {}),
                    "native_kind": native_kind,
                    "wall_ms": item.get("wall_ms"),
                },
                "source_id": f"{run_id}:{index}",
                "structural": {
                    "workflow_id": str(address.get("run_id") or run_id),
                    "node_path": node_path,
                    "iteration": int(address.get("iteration") or 0),
                    "local_sequence": int(address.get("local_seq") or 0),
                },
            }
        )
    return output


def _codex_usage(
    snapshots: list[dict[str, Any]],
    *,
    source_digest: str,
) -> UsageV5 | None:
    if not snapshots:
        return None

    def total(*names: str) -> int | None:
        found = [
            int(snapshot[name])
            for snapshot in snapshots
            for name in names
            if isinstance(snapshot.get(name), int)
        ]
        return sum(found) if found else None

    prompt = total("input_tokens", "prompt_tokens")
    completion = total("output_tokens", "completion_tokens")
    observed_total = total("total_tokens")
    return UsageV5(
        provenance=UsageProvenance.OBSERVED_HARNESS,
        prompt_tokens=prompt,
        completion_tokens=completion,
        reasoning_tokens=total("reasoning_tokens"),
        cached_tokens=total("cached_input_tokens", "cached_tokens"),
        total_tokens=(
            observed_total
            if observed_total is not None
            else (
                int(prompt or 0) + int(completion or 0)
                if prompt is not None or completion is not None
                else None
            )
        ),
        requests=len(snapshots),
        source_refs=(source_digest,),
    )


def _json_object(payload: bytes) -> Mapping[str, Any]:
    loaded = json.loads(payload.decode("utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("native trace source must be a JSON object")
    # Prove the payload is canonicalizable before it enters identity/digest logic.
    canonical_bytes(loaded)
    return loaded


__all__ = ["IMPORTED_AT", "import_native_to_bundle", "write_imported_document"]
