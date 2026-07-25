"""Stable byte-level interchange loaders for backend and SDK consumers."""

from __future__ import annotations

import json
from typing import Any

from .models.evidence import TraceEvidenceBundleV5
from .store.projection import catalog_projection as project_catalog
from .store.bundle import (
    BundleEvidenceEntryV1,
    BundleManifestV1,
    BundleObjectRefV1,
    _trace_entry_from_payload,
)
from .validation.rehydrate import evidence_bundle_from_payload, rehydrate_trace


def _object(payload: bytes) -> dict[str, Any]:
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("interchange document must be a JSON object")
    return value


def load_bundle_manifest(payload: bytes) -> BundleManifestV1:
    value = _object(payload)
    return BundleManifestV1(
        bundle_id=str(value["bundle_id"]),
        created_at=str(value["created_at"]),
        traces=tuple(_trace_entry_from_payload(item) for item in value.get("traces") or ()),
        evidence=tuple(
            BundleEvidenceEntryV1(
                bundle_id=str(item["bundle_id"]),
                bundle_digest=str(item["bundle_digest"]),
                trace_digest=str(item["trace_digest"]),
                path=str(item["path"]),
            )
            for item in value.get("evidence") or ()
        ),
        projection_digests=tuple(value.get("projection_digests") or ()),
        receipt_paths=tuple(value.get("receipt_paths") or ()),
        blob_digests=tuple(value.get("blob_digests") or ()),
        objects=tuple(
            BundleObjectRefV1(
                path=str(item["path"]),
                bytes_digest=str(item["bytes_digest"]),
                byte_size=int(item["byte_size"]),
                media_type=str(item["media_type"]),
                kind=str(item.get("kind") or _legacy_object_kind(str(item.get("role") or ""))),
                role=str(item.get("role") or "") or None,
                semantic_digest=item.get("semantic_digest"),
                immutable=bool(item.get("immutable", True)),
            )
            for item in value.get("objects") or ()
        ),
        self_contained=bool(value.get("self_contained", True)),
        component_schemas=dict(value.get("component_schemas") or {}),
        schema_version=str(value.get("schema_version") or "synth.trace-bundle.v1"),
        metadata=dict(value.get("metadata") or {}),
        content_digest=str(value.get("content_digest") or ""),
    )


def _legacy_object_kind(role: str) -> str:
    if role == "capture_binding":
        return "binding"
    if role == "sealed_trace":
        return "trace"
    if role.startswith("evidence"):
        return "evidence"
    if role.startswith("projection"):
        return "projection"
    if role == "receipt":
        return "receipt"
    if role == "blob":
        return "blob"
    return "segment"


def load_trace_document(payload: bytes) -> Any:
    return rehydrate_trace(_object(payload))


def load_evidence_bundle(payload: bytes) -> TraceEvidenceBundleV5:
    return evidence_bundle_from_payload(_object(payload))


def catalog_projection(payload: bytes) -> dict[str, Any]:
    """Return a SQL-neutral row/document projection of one sealed trace."""

    return project_catalog(load_trace_document(payload))


__all__ = [
    "catalog_projection",
    "load_bundle_manifest",
    "load_evidence_bundle",
    "load_trace_document",
]
