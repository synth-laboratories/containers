"""Stable storage inspection boundary for Trace V5 consumers.

The physical bundle layout is intentionally not part of this module's public
contract.  Desktop and other consumers can inspect directories, deterministic
bundle archives, or standalone sealed Trace V5 documents and receive the same
versioned result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import json
from pathlib import Path
import tempfile
from typing import Any
import zipfile

from synth_containers.serde import JsonDataclassMixin

from .canonical import bytes_digest, content_digest
from .models.document import TraceDocumentV5
from .models.identity import BUNDLE_SCHEMA_VERSION, TRACE_SCHEMA_VERSION
from .models.rollout_inspector import ROLLOUT_INSPECTOR_PROJECTION_SCHEMA_VERSION
from .projections.inspector import load_bundle
from .store.bundle import LocalTraceBundle, _safe_bundle_path
from .validation.rehydrate import rehydrate_trace
from .validation.validator import validate


TRACE_INSPECTION_SCHEMA_VERSION = "synth.trace-inspection.v1"
ROLLOUT_INSPECTOR_PROJECTION_FORMAT = ROLLOUT_INSPECTOR_PROJECTION_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class TraceInspectionIssueV1(JsonDataclassMixin):
    code: str
    message: str
    severity: str = "error"
    path: str | None = None


@dataclass(frozen=True, slots=True)
class TraceInspectionValidationV1(JsonDataclassMixin):
    valid: bool
    self_contained: bool | None
    issues: tuple[TraceInspectionIssueV1, ...] = ()


@dataclass(frozen=True, slots=True)
class InspectedTraceV1(JsonDataclassMixin):
    trace_id: str
    trace_digest: str
    schema_version: str
    trace_kind: str | None = None
    capture_id: str | None = None
    binding_digest: str | None = None
    source_format: str | None = None
    producer: str | None = None
    model: str | None = None
    provider: str | None = None
    harness: str | None = None
    benchmark: str | None = None
    task_id: str | None = None
    seed: int | None = None
    terminal_reason: str | None = None
    lifecycle_status: str | None = None
    capture_status: str | None = None
    reward: float | None = None
    cost_usd: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    started_at: str | None = None
    ended_at: str | None = None
    duration_ms: int | None = None
    span_count: int | None = None
    event_count: int | None = None
    artifact_count: int | None = None
    tool_call_count: int | None = None
    error_count: int | None = None
    available: bool = True
    verified: bool = False
    projectable: bool = False


@dataclass(frozen=True, slots=True)
class InspectedAssetV1(JsonDataclassMixin):
    path: str
    kind: str
    role: str | None
    media_type: str
    byte_size: int | None = None
    bytes_digest: str | None = None
    semantic_digest: str | None = None
    available: bool = True
    verified: bool = False


@dataclass(frozen=True, slots=True)
class InspectedProjectionV1(JsonDataclassMixin):
    path: str
    projection_digest: str | None = None
    format: str | None = None
    source_trace_digest: str | None = None
    schema_version: str | None = None
    available: bool = True
    verified: bool = False


@dataclass(frozen=True, slots=True)
class TraceInspectionV1(JsonDataclassMixin):
    input_kind: str
    compatibility: str
    source: str
    source_bytes_digest: str | None
    bundle_id: str | None
    bundle_digest: str | None
    archive_digest: str | None
    self_contained: bool | None
    trusted: bool
    validation: TraceInspectionValidationV1
    traces: tuple[InspectedTraceV1, ...] = ()
    assets: tuple[InspectedAssetV1, ...] = ()
    projections: tuple[InspectedProjectionV1, ...] = ()
    schema_version: str = TRACE_INSPECTION_SCHEMA_VERSION


def inspect_trace_input(
    source: str | Path,
    *,
    archive_output: str | Path | None = None,
) -> TraceInspectionV1:
    """Inspect a portable trace input without changing its bytes.

    The function returns an ``invalid`` result for expected input failures rather
    than leaking layout-specific exceptions to consumers.  When ``archive_output``
    is supplied for a trusted bundle, the exact verified object graph is also
    emitted as a deterministic portable ZIP.  The inspected source is never changed.
    """

    path = Path(source)
    if not path.exists():
        return _failure(
            path,
            input_kind="unknown",
            code="source_missing",
            message=f"trace input does not exist: {path}",
        )
    if path.is_dir():
        result = _inspect_bundle_directory(
            path,
            source=path,
            input_kind="bundle_directory",
        )
        return _finalize_archive(
            LocalTraceBundle(path),
            result,
            archive_output=archive_output,
        )
    try:
        source_digest = bytes_digest(path.read_bytes())
    except OSError as exc:
        return _failure(
            path,
            input_kind="unknown",
            code="source_unreadable",
            message=str(exc),
        )
    if zipfile.is_zipfile(path):
        with tempfile.TemporaryDirectory(prefix="synth-trace-inspect-") as temporary:
            extracted = Path(temporary) / "bundle"
            try:
                LocalTraceBundle.extract_archive(
                    path,
                    extracted,
                    require_self_contained=False,
                )
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                return _failure(
                    path,
                    input_kind="bundle_archive",
                    code="archive_invalid",
                    message=str(exc),
                    source_bytes_digest=source_digest,
                )
            result = _inspect_bundle_directory(
                extracted,
                source=path,
                input_kind="bundle_archive",
                source_bytes_digest=source_digest,
            )
            return _finalize_archive(
                LocalTraceBundle(extracted),
                result,
                archive_output=archive_output,
            )
    if path.suffix.lower() == ".zip":
        return _failure(
            path,
            input_kind="bundle_archive",
            code="archive_invalid",
            message="input has a ZIP extension but is not a readable ZIP archive",
            source_bytes_digest=source_digest,
        )
    return _inspect_standalone(path, source_digest=source_digest)


def _finalize_archive(
    bundle: LocalTraceBundle,
    result: TraceInspectionV1,
    *,
    archive_output: str | Path | None,
) -> TraceInspectionV1:
    if not result.trusted or not result.self_contained:
        return result
    try:
        if archive_output is not None:
            archive_digest = bundle.write_archive(Path(archive_output))
        else:
            archive_digest = bytes_digest(bundle.archive_bytes())
    except (OSError, ValueError) as exc:
        issue = TraceInspectionIssueV1(
            code="archive_materialization_failed",
            message=str(exc),
        )
        return replace(
            result,
            compatibility="invalid",
            trusted=False,
            validation=replace(
                result.validation,
                valid=False,
                issues=result.validation.issues + (issue,),
            ),
            traces=tuple(replace(item, projectable=False) for item in result.traces),
        )
    return replace(result, archive_digest=archive_digest)


def _inspect_standalone(path: Path, *, source_digest: str) -> TraceInspectionV1:
    asset = InspectedAssetV1(
        path=path.name,
        kind="trace",
        role="sealed_trace",
        media_type="application/json",
        byte_size=path.stat().st_size,
        bytes_digest=source_digest,
        available=True,
        verified=False,
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        if path.suffix.lower() != ".json":
            return TraceInspectionV1(
                input_kind="opaque_file",
                compatibility="opaque",
                source=str(path),
                source_bytes_digest=source_digest,
                bundle_id=None,
                bundle_digest=None,
                archive_digest=None,
                self_contained=None,
                trusted=False,
                validation=TraceInspectionValidationV1(valid=True, self_contained=None),
                assets=(asset,),
            )
        return _failure(
            path,
            input_kind="standalone_trace",
            code="json_invalid",
            message=str(exc),
            source_bytes_digest=source_digest,
            assets=(asset,),
        )
    if not isinstance(payload, dict):
        return _failure(
            path,
            input_kind="standalone_trace",
            code="trace_not_object",
            message="standalone trace JSON must be an object",
            source_bytes_digest=source_digest,
            assets=(asset,),
        )
    schema_version = str(payload.get("schema_version") or "")
    if schema_version != TRACE_SCHEMA_VERSION:
        return TraceInspectionV1(
            input_kind="opaque_file",
            compatibility="opaque",
            source=str(path),
            source_bytes_digest=source_digest,
            bundle_id=None,
            bundle_digest=None,
            archive_digest=None,
            self_contained=None,
            trusted=False,
            validation=TraceInspectionValidationV1(valid=True, self_contained=None),
            assets=(asset,),
        )
    try:
        document = rehydrate_trace(payload)
    except (TypeError, ValueError) as exc:
        return _failure(
            path,
            input_kind="standalone_trace",
            code="trace_invalid",
            message=str(exc),
            source_bytes_digest=source_digest,
            assets=(asset,),
        )
    receipt = validate(document)
    issues = tuple(
        TraceInspectionIssueV1(
            code=item.code,
            message=item.message,
            severity=str(item.severity),
        )
        for item in receipt.findings
    )
    valid = receipt.valid
    verified_asset = InspectedAssetV1(
        path=asset.path,
        kind=asset.kind,
        role=asset.role,
        media_type=asset.media_type,
        byte_size=asset.byte_size,
        bytes_digest=asset.bytes_digest,
        semantic_digest=document.content_digest,
        available=True,
        verified=valid,
    )
    return TraceInspectionV1(
        input_kind="standalone_trace",
        compatibility="native" if valid else "invalid",
        source=str(path),
        source_bytes_digest=source_digest,
        bundle_id=None,
        bundle_digest=None,
        archive_digest=None,
        self_contained=True,
        trusted=valid,
        validation=TraceInspectionValidationV1(
            valid=valid,
            self_contained=True,
            issues=issues,
        ),
        traces=(_trace_summary(document, verified=valid, projectable=valid),),
        assets=(verified_asset,),
    )


def _inspect_bundle_directory(
    root: Path,
    *,
    source: Path,
    input_kind: str,
    source_bytes_digest: str | None = None,
    archive_digest: str | None = None,
) -> TraceInspectionV1:
    bundle = LocalTraceBundle(root)
    issues: list[TraceInspectionIssueV1] = []
    try:
        pointer = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
        manifest = bundle.read_manifest()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _failure(
            source,
            input_kind=input_kind,
            code="bundle_manifest_invalid",
            message=str(exc),
            source_bytes_digest=source_bytes_digest,
            archive_digest=archive_digest,
        )
    manifest_digest = str(manifest.get("content_digest") or "")
    if not manifest_digest or content_digest(manifest) != manifest_digest:
        issues.append(
            TraceInspectionIssueV1(
                code="bundle_digest_mismatch",
                message="bundle manifest is not sealed under its declared content digest",
            )
        )
    if pointer.get("relative_path") and str(pointer.get("bundle_id") or "") != str(
        manifest.get("bundle_id") or ""
    ):
        issues.append(
            TraceInspectionIssueV1(
                code="bundle_id_mismatch",
                message="manifest pointer and immutable manifest name different bundle ids",
            )
        )
    try:
        self_contained, storage_failures = bundle.verify_self_contained()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        self_contained = False
        storage_failures = (f"verification_failed:{exc}",)
    for failure in storage_failures:
        issues.append(
            TraceInspectionIssueV1(
                code=_failure_code(failure),
                message=failure,
                path=_failure_path(failure),
            )
        )
    if not bool(manifest.get("self_contained", True)):
        self_contained = False
        issues.append(
            TraceInspectionIssueV1(
                code="bundle_declared_partial",
                message="bundle manifest declares that the bundle is not self-contained",
            )
        )

    assets = _bundle_assets(bundle, manifest)
    projections = _projection_descriptors(bundle, assets)
    traces: list[InspectedTraceV1] = []
    documents: dict[str, TraceDocumentV5] = {}
    for entry in manifest.get("traces") or ():
        trace_id = str(entry.get("trace_id") or "")
        trace_digest = str(entry.get("trace_digest") or "")
        try:
            document = rehydrate_trace(bundle.read_trace(trace_digest))
            documents[trace_digest] = document
            traces.append(_trace_summary(document, verified=False, projectable=False))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            issues.append(
                TraceInspectionIssueV1(
                    code="trace_unreadable",
                    message=str(exc),
                    path=str(entry.get("sealed_path") or "") or None,
                )
            )
            traces.append(
                InspectedTraceV1(
                    trace_id=trace_id,
                    trace_digest=trace_digest,
                    schema_version=TRACE_SCHEMA_VERSION,
                    capture_id=str(entry.get("capture_id") or "") or None,
                    available=False,
                )
            )

    layout_current = bool(pointer.get("relative_path")) and bool(manifest.get("objects"))
    schema_supported = str(manifest.get("schema_version") or BUNDLE_SCHEMA_VERSION) == (
        BUNDLE_SCHEMA_VERSION
    )
    partial = not self_contained and all(_is_partial_issue(item.code) for item in issues)
    structurally_invalid = any(
        item.severity == "error" and not _is_partial_issue(item.code) for item in issues
    )
    if schema_supported and self_contained and not structurally_invalid:
        try:
            inspected = load_bundle(root)
            evidence_by_digest = {item.trace.content_digest: item.evidence for item in inspected}
            validated: list[InspectedTraceV1] = []
            for trace in traces:
                document = documents.get(trace.trace_digest)
                if document is None:
                    validated.append(trace)
                    continue
                receipt = validate(document, evidence_by_digest.get(trace.trace_digest))
                issues.extend(
                    TraceInspectionIssueV1(
                        code=item.code,
                        message=item.message,
                        severity=str(item.severity),
                    )
                    for item in receipt.findings
                )
                validated.append(
                    _trace_summary(
                        document,
                        verified=receipt.valid,
                        projectable=receipt.valid,
                    )
                )
            traces = validated
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            issues.append(
                TraceInspectionIssueV1(
                    code="bundle_records_invalid",
                    message=str(exc),
                )
            )
    valid = (
        self_contained
        and not any(item.severity == "error" for item in issues)
        and all(item.verified for item in traces)
    )
    if not schema_supported and self_contained and not structurally_invalid:
        compatibility = "opaque"
    elif valid:
        compatibility = "native" if layout_current else "legacy_native"
    elif partial:
        compatibility = "partial"
    else:
        compatibility = "invalid"
    trusted = compatibility in {"native", "legacy_native"}
    if not trusted:
        traces = [
            InspectedTraceV1(
                **{
                    **item.to_dict(),
                    "projectable": False,
                }
            )
            for item in traces
        ]
    return TraceInspectionV1(
        input_kind=input_kind,
        compatibility=compatibility,
        source=str(source),
        source_bytes_digest=source_bytes_digest,
        bundle_id=str(manifest.get("bundle_id") or "") or None,
        bundle_digest=manifest_digest or None,
        archive_digest=archive_digest,
        self_contained=self_contained,
        trusted=trusted,
        validation=TraceInspectionValidationV1(
            valid=valid,
            self_contained=self_contained,
            issues=tuple(issues),
        ),
        traces=tuple(traces),
        assets=tuple(assets),
        projections=tuple(projections),
    )


def _trace_summary(
    document: TraceDocumentV5,
    *,
    verified: bool,
    projectable: bool,
) -> InspectedTraceV1:
    model = document.provenance.model or next(
        (item.model for item in document.actors if item.model),
        None,
    )
    provider = document.provenance.provider or next(
        (item.provider for item in document.actors if item.provider),
        None,
    )
    duration_ms = None
    if document.lifecycle.started_at and document.lifecycle.ended_at:
        started = datetime.fromisoformat(document.lifecycle.started_at.replace("Z", "+00:00"))
        ended = datetime.fromisoformat(document.lifecycle.ended_at.replace("Z", "+00:00"))
        duration_ms = max(0, round((ended - started).total_seconds() * 1000))
    reward = None
    craftax = document.extensions.get("craftax")
    if isinstance(craftax, dict):
        rollouts = craftax.get("rollouts")
        if isinstance(rollouts, list) and len(rollouts) == 1:
            value = rollouts[0].get("reward") if isinstance(rollouts[0], dict) else None
            reward = float(value) if isinstance(value, (int, float)) else None
    return InspectedTraceV1(
        trace_id=document.trace_id,
        trace_digest=document.content_digest,
        schema_version=document.schema_version,
        trace_kind=str(document.trace_kind),
        capture_id=document.capture.capture_id,
        binding_digest=document.capture.binding_digest,
        source_format=document.provenance.source_format,
        producer=document.provenance.producer,
        model=model,
        provider=provider,
        harness=document.provenance.harness,
        benchmark=document.identity.benchmark,
        task_id=document.identity.task_id,
        seed=document.identity.seed,
        terminal_reason=(
            document.lifecycle.termination.reason
            if document.lifecycle.termination is not None
            else None
        ),
        lifecycle_status=str(document.lifecycle.status),
        capture_status=str(document.completeness.capture_status),
        reward=reward,
        cost_usd=document.usage.cost_usd,
        prompt_tokens=document.usage.prompt_tokens,
        completion_tokens=document.usage.completion_tokens,
        started_at=document.lifecycle.started_at,
        ended_at=document.lifecycle.ended_at,
        duration_ms=duration_ms,
        span_count=len(document.spans),
        event_count=len(document.events),
        artifact_count=len(document.artifacts),
        tool_call_count=sum(
            1 for span in document.spans if str(span.span_kind) == "tool_execution"
        ),
        error_count=len(document.errors),
        available=True,
        verified=verified,
        projectable=projectable,
    )


def _bundle_assets(
    bundle: LocalTraceBundle,
    manifest: dict[str, Any],
) -> list[InspectedAssetV1]:
    objects = list(manifest.get("objects") or ())
    if not objects:
        objects = _legacy_assets(bundle, manifest)
    assets: list[InspectedAssetV1] = []
    for item in objects:
        relative = str(item.get("path") or "")
        available = False
        verified = False
        observed_size: int | None = None
        observed_digest: str | None = None
        try:
            path = _safe_bundle_path(bundle.root, relative)
            if path.is_file():
                body = path.read_bytes()
                available = True
                observed_size = len(body)
                observed_digest = bytes_digest(body)
                expected_size = item.get("byte_size")
                expected_digest = item.get("bytes_digest")
                verified = (expected_size is None or int(expected_size) == observed_size) and (
                    expected_digest is None or str(expected_digest) == observed_digest
                )
        except (OSError, ValueError):
            pass
        assets.append(
            InspectedAssetV1(
                path=relative,
                kind=str(item.get("kind") or "object"),
                role=str(item.get("role") or "") or None,
                media_type=str(item.get("media_type") or "application/octet-stream"),
                byte_size=(
                    int(item["byte_size"]) if item.get("byte_size") is not None else observed_size
                ),
                bytes_digest=str(item.get("bytes_digest") or "") or observed_digest,
                semantic_digest=str(item.get("semantic_digest") or "") or None,
                available=available,
                verified=verified,
            )
        )
    return sorted(assets, key=lambda item: item.path)


def _legacy_assets(bundle: LocalTraceBundle, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    assets: dict[str, dict[str, Any]] = {}

    def add(path: str, kind: str, role: str, media_type: str, semantic: str | None) -> None:
        assets[path] = {
            "path": path,
            "kind": kind,
            "role": role,
            "media_type": media_type,
            "semantic_digest": semantic,
        }

    for entry in manifest.get("traces") or ():
        add(
            str(entry.get("sealed_path") or ""),
            "trace",
            "sealed_trace",
            "application/json",
            str(entry.get("trace_digest") or "") or None,
        )
        add(
            str(entry.get("binding_path") or ""),
            "binding",
            "capture_binding",
            "application/json",
            str(entry.get("binding_digest") or "") or None,
        )
        for segment in entry.get("segments") or ():
            relative = str(segment.get("relative_path") or "")
            if not relative.startswith("traces/"):
                relative = f"traces/{entry.get('trace_id')}/{relative}"
            add(
                relative,
                "segment",
                "raw_segment",
                str(segment.get("media_type") or "application/x-ndjson"),
                str(segment.get("digest") or "") or None,
            )
    for entry in manifest.get("evidence") or ():
        add(
            str(entry.get("path") or ""),
            "evidence",
            "evidence_bundle",
            "application/json",
            str(entry.get("bundle_digest") or "") or None,
        )
    for relative in manifest.get("receipt_paths") or ():
        add(str(relative), "receipt", "receipt", "application/json", None)
    for digest in manifest.get("blob_digests") or ():
        try:
            relative = str(bundle.blobs.path_for(str(digest)).relative_to(bundle.root))
        except ValueError:
            continue
        add(relative, "blob", "blob", "application/octet-stream", str(digest))
    projection_root = bundle.root / "projections"
    if projection_root.exists():
        for path in projection_root.glob("*/*.json"):
            add(
                str(path.relative_to(bundle.root)),
                "projection",
                path.parent.name,
                "application/json",
                None,
            )
    return [item for path, item in sorted(assets.items()) if path]


def _projection_descriptors(
    bundle: LocalTraceBundle,
    assets: list[InspectedAssetV1],
) -> list[InspectedProjectionV1]:
    projections: list[InspectedProjectionV1] = []
    for asset in assets:
        if asset.kind != "projection":
            continue
        if not asset.available:
            projections.append(
                InspectedProjectionV1(path=asset.path, available=False, verified=False)
            )
            continue
        try:
            payload = json.loads(_safe_bundle_path(bundle.root, asset.path).read_text("utf-8"))
            manifest = payload["manifest"]
            digest = str(manifest.get("content_digest") or "")
            verified = bool(digest) and content_digest(manifest) == digest
            projections.append(
                InspectedProjectionV1(
                    path=asset.path,
                    projection_digest=digest or None,
                    format=str(manifest.get("format") or "") or None,
                    source_trace_digest=(str(manifest.get("source_trace_digest") or "") or None),
                    schema_version=str(manifest.get("schema_version") or "") or None,
                    available=True,
                    verified=verified and asset.verified,
                )
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            projections.append(
                InspectedProjectionV1(path=asset.path, available=True, verified=False)
            )
    return projections


def _failure_code(failure: str) -> str:
    return failure.split(":", 1)[0] or "verification_failed"


def _failure_path(failure: str) -> str | None:
    pieces = failure.split(":", 2)
    if len(pieces) < 2 or pieces[1].startswith("sha256"):
        return None
    return pieces[1] or None


def _is_partial_issue(code: str) -> bool:
    return code in {
        "bundle_declared_partial",
        "missing_object",
        "missing_blob",
        "missing_segment",
        "sealed_trace",
        "evidence",
        "trace_unreadable",
    }


def _failure(
    path: Path,
    *,
    input_kind: str,
    code: str,
    message: str,
    source_bytes_digest: str | None = None,
    archive_digest: str | None = None,
    assets: tuple[InspectedAssetV1, ...] = (),
) -> TraceInspectionV1:
    issue = TraceInspectionIssueV1(code=code, message=message)
    return TraceInspectionV1(
        input_kind=input_kind,
        compatibility="invalid",
        source=str(path),
        source_bytes_digest=source_bytes_digest,
        bundle_id=None,
        bundle_digest=None,
        archive_digest=archive_digest,
        self_contained=False if input_kind.startswith("bundle") else None,
        trusted=False,
        validation=TraceInspectionValidationV1(
            valid=False,
            self_contained=False if input_kind.startswith("bundle") else None,
            issues=(issue,),
        ),
        assets=assets,
    )


__all__ = [
    "ROLLOUT_INSPECTOR_PROJECTION_FORMAT",
    "TRACE_INSPECTION_SCHEMA_VERSION",
    "InspectedAssetV1",
    "InspectedProjectionV1",
    "InspectedTraceV1",
    "TraceInspectionIssueV1",
    "TraceInspectionV1",
    "TraceInspectionValidationV1",
    "inspect_trace_input",
]
