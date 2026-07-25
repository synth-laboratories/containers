"""The local trace bundle: a self-contained, portable directory of immutable objects.

Layout::

    <bundle>/
      manifest.json
      blobs/sha256/<prefix>/<digest>
      traces/<trace-id>/binding.json
      traces/<trace-id>/segments/<sequence>-<digest>.jsonl
      traces/<trace-id>/manifests/<generation>.json
      traces/<trace-id>/latest.json
      traces/<trace-id>/sealed/<trace-digest>.json
      evidence/<evidence-bundle-digest>.json
      projections/v4/<projection-digest>.json
      receipts/capture-coverage.json
      receipts/validation.json
      receipts/projection-v4.json
      catalog.sqlite3

Every object except ``latest.json`` and ``catalog.sqlite3`` is immutable. The catalog
is a projection: deleting it and calling ``rebuild_catalog`` reproduces it from the
manifest and the sealed objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from synth_containers.serde import JsonDataclassMixin

from ..canonical import (
    bytes_digest,
    canonical_bytes,
    canonical_text,
    content_digest,
    digest_hex,
    utc_now,
)
from ..capture.binding import TraceCaptureBindingV1
from ..capture.spool import TraceSegmentManifestV1
from ..models.document import TraceDocumentV5
from ..models.evidence import TraceEvidenceBundleV5
from ..models.identity import BUNDLE_SCHEMA_VERSION
from ..models.projection import ProjectionManifestV1
from .filesystem import FilesystemBlobStore
from .sqlite_catalog import SqliteCatalogStore


MANIFEST_FILE = "manifest.json"
CATALOG_FILE = "catalog.sqlite3"


@dataclass(frozen=True, slots=True)
class BundleTraceEntryV1(JsonDataclassMixin):
    trace_id: str
    trace_digest: str
    capture_id: str
    binding_digest: str
    sealed_path: str
    binding_path: str
    segments: tuple[TraceSegmentManifestV1, ...] = ()


@dataclass(frozen=True, slots=True)
class BundleEvidenceEntryV1(JsonDataclassMixin):
    bundle_id: str
    bundle_digest: str
    trace_digest: str
    path: str


@dataclass(frozen=True, slots=True)
class BundleManifestV1(JsonDataclassMixin):
    """Names every object a self-contained bundle must contain."""

    bundle_id: str
    created_at: str
    traces: tuple[BundleTraceEntryV1, ...] = ()
    evidence: tuple[BundleEvidenceEntryV1, ...] = ()
    projection_digests: tuple[str, ...] = ()
    receipt_paths: tuple[str, ...] = ()
    blob_digests: tuple[str, ...] = ()
    self_contained: bool = True
    component_schemas: dict[str, str] = field(default_factory=dict)
    schema_version: str = BUNDLE_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    content_digest: str = ""

    def sealed(self) -> "BundleManifestV1":
        return replace(self, content_digest=content_digest(self))


class LocalTraceBundle:
    """Reads and writes one bundle directory."""

    def __init__(self, root: Path, *, bundle_id: str | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.bundle_id = bundle_id or self.root.name
        self.blobs = FilesystemBlobStore(self.root / "blobs")
        self._traces: list[BundleTraceEntryV1] = []
        self._evidence: list[BundleEvidenceEntryV1] = []
        self._projections: list[str] = []
        self._receipts: list[str] = []

    # -- paths -------------------------------------------------------------------

    def trace_root(self, trace_id: str) -> Path:
        return self.root / "traces" / trace_id

    def capture_root(self, trace_id: str) -> Path:
        path = self.trace_root(trace_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def catalog_path(self) -> Path:
        return self.root / CATALOG_FILE

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_FILE

    # -- writing -----------------------------------------------------------------

    def write_binding(self, binding: TraceCaptureBindingV1) -> Path:
        return binding.write(self.capture_root(binding.trace_id))

    def write_trace(
        self,
        document: TraceDocumentV5,
        *,
        binding: TraceCaptureBindingV1,
        segments: tuple[TraceSegmentManifestV1, ...],
    ) -> Path:
        if not document.content_digest:
            raise ValueError("only a sealed trace document can be written to a bundle")
        sealed_dir = self.trace_root(document.trace_id) / "sealed"
        sealed_dir.mkdir(parents=True, exist_ok=True)
        path = sealed_dir / f"{digest_hex(document.content_digest)}.json"
        _write_immutable(path, canonical_bytes(document))
        self._traces.append(
            BundleTraceEntryV1(
                trace_id=document.trace_id,
                trace_digest=document.content_digest,
                capture_id=document.capture.capture_id,
                binding_digest=binding.content_digest,
                sealed_path=str(path.relative_to(self.root)),
                binding_path=str(
                    (self.trace_root(document.trace_id) / "binding.json").relative_to(self.root)
                ),
                segments=segments,
            )
        )
        return path

    def write_evidence(self, bundle: TraceEvidenceBundleV5) -> Path:
        if not bundle.content_digest:
            raise ValueError("only a sealed evidence bundle can be written")
        directory = self.root / "evidence"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{digest_hex(bundle.content_digest)}.json"
        _write_immutable(path, canonical_bytes(bundle))
        self._evidence.append(
            BundleEvidenceEntryV1(
                bundle_id=bundle.bundle_id,
                bundle_digest=bundle.content_digest,
                trace_digest=bundle.trace_ref.content_digest,
                path=str(path.relative_to(self.root)),
            )
        )
        return path

    def write_projection(
        self,
        manifest: ProjectionManifestV1,
        payload: Any,
        *,
        kind: str,
    ) -> tuple[Path, ProjectionManifestV1]:
        directory = self.root / "projections" / kind
        directory.mkdir(parents=True, exist_ok=True)
        body = canonical_bytes(payload)
        target_digest = self.blobs.put(body)
        sealed = replace(manifest, target_digest=target_digest).sealed()
        path = directory / f"{digest_hex(sealed.content_digest)}.json"
        _write_immutable(
            path,
            canonical_bytes({"manifest": sealed, "payload": payload}),
        )
        self._projections.append(sealed.content_digest)
        return path, sealed

    def write_receipt(self, name: str, payload: Any) -> Path:
        directory = self.root / "receipts"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.json"
        path.write_text(canonical_text(payload) + "\n", encoding="utf-8")
        relative = str(path.relative_to(self.root))
        if relative not in self._receipts:
            self._receipts.append(relative)
        return path

    def write_manifest(self, *, metadata: dict[str, Any] | None = None) -> BundleManifestV1:
        blob_digests = tuple(sorted(_all_blob_digests(self.root / "blobs")))
        manifest = BundleManifestV1(
            bundle_id=self.bundle_id,
            created_at=utc_now(),
            traces=tuple(self._traces),
            evidence=tuple(self._evidence),
            projection_digests=tuple(self._projections),
            receipt_paths=tuple(sorted(self._receipts)),
            blob_digests=blob_digests,
            component_schemas={
                "trace": "synth.trace.v5",
                "evidence": "synth.trace-evidence-bundle.v5",
                "binding": "synth.trace-capture-binding.v1",
                "raw_envelope": "synth.capture.raw.v1",
                "segment_manifest": "synth.trace-segment-manifest.v1",
                "coverage_receipt": "synth.capture-coverage-receipt.v1",
                "projection_manifest": "synth.projection-manifest.v1",
                "selector": "synth.trace-selector.v1",
            },
            metadata=dict(metadata or {}),
        ).sealed()
        self.manifest_path.write_text(canonical_text(manifest) + "\n", encoding="utf-8")
        return manifest

    # -- reading -----------------------------------------------------------------

    def read_manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def read_trace(self, trace_digest: str) -> dict[str, Any]:
        """Read a sealed trace by digest, proving the stored bytes match that digest."""

        for entry in self.read_manifest().get("traces") or []:
            if entry["trace_digest"] == trace_digest:
                return _read_sealed(self.root / entry["sealed_path"], trace_digest)
        raise FileNotFoundError(f"bundle manifest does not name trace {trace_digest}")

    def read_evidence(self, bundle_digest: str) -> dict[str, Any]:
        for entry in self.read_manifest().get("evidence") or []:
            if entry["bundle_digest"] == bundle_digest:
                return _read_sealed(self.root / entry["path"], bundle_digest)
        raise FileNotFoundError(f"bundle manifest does not name evidence {bundle_digest}")

    def open_catalog(self) -> SqliteCatalogStore:
        return SqliteCatalogStore(self.catalog_path)

    def verify_self_contained(self) -> tuple[bool, tuple[str, ...]]:
        """Prove every body the manifest names is present and matches its digest."""

        manifest = self.read_manifest()
        missing: list[str] = []
        for digest in manifest.get("blob_digests") or []:
            if not self.blobs.has(digest):
                missing.append(f"missing_blob:{digest}")
                continue
            try:
                self.blobs.get(digest)
            except ValueError as exc:
                missing.append(f"corrupt_blob:{digest}:{exc}")
        for entry in manifest.get("traces") or []:
            try:
                _read_sealed(self.root / entry["sealed_path"], entry["trace_digest"])
            except (FileNotFoundError, ValueError) as exc:
                missing.append(f"sealed_trace:{entry['trace_digest']}:{exc}")
            for segment in entry.get("segments") or []:
                path = self.root / "traces" / entry["trace_id"] / segment["relative_path"]
                if not path.exists():
                    missing.append(f"missing_segment:{segment['digest']}")
                elif bytes_digest(path.read_bytes()) != segment["digest"]:
                    missing.append(f"corrupt_segment:{segment['digest']}")
        for entry in manifest.get("evidence") or []:
            try:
                _read_sealed(self.root / entry["path"], entry["bundle_digest"])
            except (FileNotFoundError, ValueError) as exc:
                missing.append(f"evidence:{entry['bundle_digest']}:{exc}")
        return (not missing), tuple(missing)


def rebuild_catalog(bundle: LocalTraceBundle) -> dict[str, int]:
    """Rebuild the SQLite projection from the bundle's manifest and sealed objects."""

    manifest = bundle.read_manifest()
    catalog = bundle.open_catalog()
    catalog.reset()
    traces = 0
    evidence = 0
    for entry in manifest.get("traces") or []:
        payload = bundle.read_trace(entry["trace_digest"])
        catalog.index_trace(_document_from_payload(payload))
        traces += 1
    for entry in manifest.get("evidence") or []:
        payload = bundle.read_evidence(entry["bundle_digest"])
        catalog.index_evidence(_evidence_from_payload(payload))
        evidence += 1
    catalog.close()
    return {"traces": traces, "evidence_bundles": evidence}


def _document_from_payload(payload: dict[str, Any]) -> TraceDocumentV5:
    from ..validation.rehydrate import trace_document_from_payload

    return trace_document_from_payload(payload)


def _evidence_from_payload(payload: dict[str, Any]) -> TraceEvidenceBundleV5:
    from ..validation.rehydrate import evidence_bundle_from_payload

    return evidence_bundle_from_payload(payload)


def _read_sealed(path: Path, expected_digest: str) -> dict[str, Any]:
    """Read a sealed object and prove it re-digests to the digest it is stored under."""

    if not path.exists():
        raise FileNotFoundError(f"sealed object missing at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual = content_digest(payload)
    if actual != expected_digest:
        raise ValueError(
            f"sealed object digest mismatch: expected {expected_digest}, found {actual}"
        )
    return payload


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(payload)
    temp.replace(path)
    path.chmod(0o444)


def _all_blob_digests(root: Path) -> list[str]:
    digests: list[str] = []
    if not root.exists():
        return digests
    for algorithm_dir in sorted(root.iterdir()):
        if not algorithm_dir.is_dir():
            continue
        for prefix_dir in sorted(algorithm_dir.iterdir()):
            if not prefix_dir.is_dir():
                continue
            for blob in sorted(prefix_dir.iterdir()):
                if blob.is_file() and not blob.name.endswith(".tmp"):
                    digests.append(f"{algorithm_dir.name}:{blob.name}")
    return digests


__all__ = [
    "CATALOG_FILE",
    "MANIFEST_FILE",
    "BundleEvidenceEntryV1",
    "BundleManifestV1",
    "BundleTraceEntryV1",
    "LocalTraceBundle",
    "rebuild_catalog",
]
