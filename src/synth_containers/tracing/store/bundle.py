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
import io
import os
from pathlib import PurePosixPath
import shutil
import stat
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
import zipfile
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
class BundleObjectRefV1(JsonDataclassMixin):
    """Integrity declaration for one bundle object, independent of its schema."""

    path: str
    bytes_digest: str
    byte_size: int
    media_type: str
    kind: str
    role: str | None = None
    semantic_digest: str | None = None
    immutable: bool = True


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
    objects: tuple[BundleObjectRefV1, ...] = ()
    self_contained: bool = True
    component_schemas: dict[str, str] = field(default_factory=dict)
    schema_version: str = BUNDLE_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    content_digest: str = ""

    def sealed(self) -> "BundleManifestV1":
        return replace(self, content_digest=content_digest(self))


@dataclass(frozen=True, slots=True)
class BundleManifestPointerV1(JsonDataclassMixin):
    bundle_id: str
    manifest_digest: str
    relative_path: str
    updated_at: str
    generation: int
    schema_version: str = "synth.trace-bundle-manifest-pointer.v1"


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
        self._generation = 0
        self._load_state()

    def _load_state(self) -> None:
        if not self.manifest_path.exists():
            return
        payload = self.read_manifest()
        self.bundle_id = str(payload.get("bundle_id") or self.bundle_id)
        self._traces = [_trace_entry_from_payload(item) for item in payload.get("traces") or []]
        self._evidence = [
            BundleEvidenceEntryV1(
                bundle_id=str(item["bundle_id"]),
                bundle_digest=str(item["bundle_digest"]),
                trace_digest=str(item["trace_digest"]),
                path=str(item["path"]),
            )
            for item in payload.get("evidence") or []
        ]
        self._projections = [str(item) for item in payload.get("projection_digests") or []]
        self._receipts = [str(item) for item in payload.get("receipt_paths") or []]
        pointer = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._generation = int(pointer.get("generation") or 0)

    # -- paths -------------------------------------------------------------------

    def trace_root(self, trace_id: str) -> Path:
        return self.root / "traces" / _safe_component(trace_id, kind="trace_id")

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
        entry = BundleTraceEntryV1(
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
        if not any(
            item.trace_id == entry.trace_id and item.trace_digest == entry.trace_digest
            for item in self._traces
        ):
            self._traces.append(entry)
        return path

    def write_evidence(self, bundle: TraceEvidenceBundleV5) -> Path:
        if not bundle.content_digest:
            raise ValueError("only a sealed evidence bundle can be written")
        directory = self.root / "evidence"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{digest_hex(bundle.content_digest)}.json"
        _write_immutable(path, canonical_bytes(bundle))
        entry = BundleEvidenceEntryV1(
                bundle_id=bundle.bundle_id,
                bundle_digest=bundle.content_digest,
                trace_digest=bundle.trace_ref.content_digest,
                path=str(path.relative_to(self.root)),
            )
        if not any(item.bundle_digest == entry.bundle_digest for item in self._evidence):
            self._evidence.append(entry)
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
        if sealed.content_digest not in self._projections:
            self._projections.append(sealed.content_digest)
        return path, sealed

    def write_receipt(self, name: str, payload: Any) -> Path:
        directory = self.root / "receipts"
        directory.mkdir(parents=True, exist_ok=True)
        body = canonical_text(payload).encode("utf-8") + b"\n"
        digest = bytes_digest(body)
        path = directory / f"{name}-{digest_hex(digest)[:16]}.json"
        _write_immutable(path, body)
        relative = str(path.relative_to(self.root))
        if relative not in self._receipts:
            self._receipts.append(relative)
        return path

    def write_manifest(self, *, metadata: dict[str, Any] | None = None) -> BundleManifestV1:
        blob_digests = tuple(sorted(_all_blob_digests(self.root / "blobs")))
        objects = tuple(self._object_inventory())
        self._generation += 1
        manifest = BundleManifestV1(
            bundle_id=self.bundle_id,
            created_at=utc_now(),
            traces=tuple(self._traces),
            evidence=tuple(self._evidence),
            projection_digests=tuple(self._projections),
            receipt_paths=tuple(sorted(self._receipts)),
            blob_digests=blob_digests,
            objects=objects,
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
        manifests = self.root / "manifests"
        manifests.mkdir(parents=True, exist_ok=True)
        generation_path = manifests / (
            f"{self._generation:06d}-{digest_hex(manifest.content_digest)[:16]}.json"
        )
        _write_immutable(
            generation_path,
            canonical_text(manifest).encode("utf-8") + b"\n",
        )
        pointer = BundleManifestPointerV1(
            bundle_id=self.bundle_id,
            manifest_digest=manifest.content_digest,
            relative_path=str(generation_path.relative_to(self.root)),
            updated_at=utc_now(),
            generation=self._generation,
        )
        temp = self.manifest_path.with_suffix(".json.tmp")
        temp.write_text(canonical_text(pointer) + "\n", encoding="utf-8")
        temp.replace(self.manifest_path)
        return manifest

    # -- reading -----------------------------------------------------------------

    def read_manifest(self) -> dict[str, Any]:
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if "relative_path" not in payload:
            return payload
        manifest_path = self._declared_path(str(payload["relative_path"]))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = content_digest(manifest)
        if actual != payload["manifest_digest"]:
            raise ValueError(
                f"bundle manifest digest mismatch: expected {payload['manifest_digest']}, "
                f"found {actual}"
            )
        return manifest

    def read_trace(self, trace_digest: str) -> dict[str, Any]:
        """Read a sealed trace by digest, proving the stored bytes match that digest."""

        for entry in self.read_manifest().get("traces") or []:
            if entry["trace_digest"] == trace_digest:
                return _read_sealed(
                    self._declared_path(str(entry["sealed_path"])),
                    trace_digest,
                )
        raise FileNotFoundError(f"bundle manifest does not name trace {trace_digest}")

    def read_evidence(self, bundle_digest: str) -> dict[str, Any]:
        for entry in self.read_manifest().get("evidence") or []:
            if entry["bundle_digest"] == bundle_digest:
                return _read_sealed(
                    self._declared_path(str(entry["path"])),
                    bundle_digest,
                )
        raise FileNotFoundError(f"bundle manifest does not name evidence {bundle_digest}")

    def open_catalog(self) -> SqliteCatalogStore:
        return SqliteCatalogStore(self.catalog_path)

    def _declared_path(self, relative: str) -> Path:
        return _safe_bundle_path(self.root, relative)

    def verify_self_contained(self) -> tuple[bool, tuple[str, ...]]:
        """Prove every body the manifest names is present and matches its digest."""

        manifest = self.read_manifest()
        missing: list[str] = []
        object_refs = manifest.get("objects") or []
        if object_refs:
            for item in object_refs:
                try:
                    path = self._declared_path(str(item["path"]))
                except ValueError as exc:
                    missing.append(f"unsafe_object:{item.get('path')}:{exc}")
                    continue
                if not path.exists():
                    missing.append(f"missing_object:{item['path']}")
                    continue
                body = path.read_bytes()
                actual = bytes_digest(body)
                if actual != item["bytes_digest"]:
                    missing.append(
                        f"corrupt_object:{item['path']}:expected={item['bytes_digest']}:"
                        f"actual={actual}"
                    )
                if len(body) != int(item["byte_size"]):
                    missing.append(f"object_size:{item['path']}")
                semantic = item.get("semantic_digest")
                if semantic:
                    kind = str(item.get("kind") or "")
                    if kind == "blob" or (
                        kind == "segment" and str(item.get("role")) == "raw_segment"
                    ):
                        observed_semantic = bytes_digest(body)
                    else:
                        try:
                            observed_semantic = content_digest(json.loads(body.decode("utf-8")))
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            observed_semantic = ""
                    if observed_semantic != semantic:
                        missing.append(
                            f"semantic_digest:{item['path']}:expected={semantic}:"
                            f"actual={observed_semantic}"
                        )
            try:
                self.read_manifest()
            except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                missing.append(f"manifest_pointer:{exc}")
            return (not missing), tuple(missing)
        # Backward-compatible verification for Push 1 manifests.
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
                _read_sealed(
                    self._declared_path(str(entry["sealed_path"])),
                    entry["trace_digest"],
                )
            except (FileNotFoundError, ValueError) as exc:
                missing.append(f"sealed_trace:{entry['trace_digest']}:{exc}")
            for segment in entry.get("segments") or []:
                path = _safe_bundle_path(
                    self.trace_root(str(entry["trace_id"])),
                    str(segment["relative_path"]),
                )
                if not path.exists():
                    missing.append(f"missing_segment:{segment['digest']}")
                elif bytes_digest(path.read_bytes()) != segment["digest"]:
                    missing.append(f"corrupt_segment:{segment['digest']}")
        for entry in manifest.get("evidence") or []:
            try:
                _read_sealed(
                    self._declared_path(str(entry["path"])),
                    entry["bundle_digest"],
                )
            except (FileNotFoundError, ValueError) as exc:
                missing.append(f"evidence:{entry['bundle_digest']}:{exc}")
        return (not missing), tuple(missing)

    def archive_bytes(self) -> bytes:
        """Return a deterministic ZIP of the manifest and every declared object."""

        ok, errors = self.verify_self_contained()
        if not ok:
            raise ValueError(f"cannot archive incomplete bundle: {errors}")
        manifest = self.read_manifest()
        pointer = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        paths = {MANIFEST_FILE}
        if pointer.get("relative_path"):
            paths.add(str(pointer["relative_path"]))
        paths.update(str(item["path"]) for item in manifest.get("objects") or ())
        # Push 1 manifests lacked an object inventory; include all files except the
        # rebuildable catalog and temporary files for backward compatibility.
        if not manifest.get("objects"):
            paths.update(
                str(path.relative_to(self.root))
                for path in self.root.rglob("*")
                if path.is_file()
                and path.name != CATALOG_FILE
                and not path.name.endswith(".tmp")
            )
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for relative in sorted(paths):
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o444 << 16
                archive.writestr(info, self._declared_path(relative).read_bytes())
        return output.getvalue()

    @classmethod
    def extract_archive(
        cls,
        source: Path,
        target: Path,
        *,
        max_entries: int = 100_000,
        max_expanded_bytes: int = 8 * 1024 * 1024 * 1024,
        require_self_contained: bool = True,
    ) -> "LocalTraceBundle":
        """Safely extract a bundle ZIP before publishing its directory.

        Normal imports require a self-contained verified bundle.  Read-only
        inspection may set ``require_self_contained=False`` to classify partial
        bundles after the same path, collision, symlink, and expansion checks.
        """

        source = Path(source)
        target = Path(target)
        if target.exists():
            raise FileExistsError(f"archive target already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.import-", dir=target.parent)
        )
        published = False
        try:
            with zipfile.ZipFile(source, "r") as archive:
                infos = archive.infolist()
                if len(infos) > max_entries:
                    raise ValueError(f"archive has too many entries: {len(infos)}")
                expanded = sum(int(item.file_size) for item in infos)
                if expanded > max_expanded_bytes:
                    raise ValueError(
                        f"archive expands to {expanded} bytes, above "
                        f"{max_expanded_bytes}"
                    )
                seen: set[str] = set()
                for info in infos:
                    relative = _safe_archive_member(info.filename)
                    folded = relative.casefold()
                    if folded in seen:
                        raise ValueError(
                            f"archive contains duplicate/case-colliding path: {relative}"
                        )
                    seen.add(folded)
                    if info.flag_bits & 0x1:
                        raise ValueError(f"encrypted archive member is unsupported: {relative}")
                    mode = (info.external_attr >> 16) & 0o170000
                    if mode == stat.S_IFLNK:
                        raise ValueError(f"archive symlink is forbidden: {relative}")
                    destination = _safe_bundle_path(staging, relative)
                    if info.is_dir():
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info, "r") as incoming, destination.open("xb") as outgoing:
                        shutil.copyfileobj(incoming, outgoing)
            imported = cls(staging)
            if require_self_contained:
                ok, failures = imported.verify_self_contained()
                if not ok:
                    raise ValueError(f"imported bundle failed verification: {failures}")
            os.replace(staging, target)
            published = True
            return cls(target)
        finally:
            if not published:
                shutil.rmtree(staging, ignore_errors=True)

    def write_archive(self, target: Path) -> str:
        """Materialize a deterministic archive and return its raw-bytes digest."""

        body = self.archive_bytes()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != body:
            raise FileExistsError(f"archive target contains different bytes: {target}")
        if not target.exists():
            temp = target.with_suffix(target.suffix + ".tmp")
            temp.write_bytes(body)
            temp.replace(target)
        return bytes_digest(body)

    def _object_inventory(self) -> list[BundleObjectRefV1]:
        """Derive the portable object graph from files reachable by bundle state."""

        refs: dict[str, BundleObjectRefV1] = {}

        def add(
            path: Path,
            *,
            kind: str,
            role: str,
            media_type: str = "application/json",
            semantic_digest: str | None = None,
            immutable: bool = True,
        ) -> None:
            if not path.exists():
                raise FileNotFoundError(f"bundle object is missing before manifest: {path}")
            body = path.read_bytes()
            relative = str(path.relative_to(self.root))
            refs[relative] = BundleObjectRefV1(
                path=relative,
                bytes_digest=bytes_digest(body),
                byte_size=len(body),
                media_type=media_type,
                kind=kind,
                role=role,
                semantic_digest=semantic_digest,
                immutable=immutable,
            )

        for entry in self._traces:
            add(
                self.root / entry.binding_path,
                kind="binding",
                role="capture_binding",
                semantic_digest=entry.binding_digest,
            )
            add(
                self.root / entry.sealed_path,
                kind="trace",
                role="sealed_trace",
                semantic_digest=entry.trace_digest,
            )
            for segment in entry.segments:
                add(
                    self.trace_root(entry.trace_id) / segment.relative_path,
                    kind="segment",
                    role="raw_segment",
                    media_type=segment.media_type,
                    semantic_digest=segment.digest,
                )
            trace_root = self.trace_root(entry.trace_id)
            for path in sorted((trace_root / "manifests").glob("*.json")):
                add(path, kind="segment", role="capture_manifest")
            latest = trace_root / "latest.json"
            if latest.exists():
                add(
                    latest,
                    kind="segment",
                    role="capture_manifest_pointer",
                    immutable=False,
                )
        for entry in self._evidence:
            add(
                self.root / entry.path,
                kind="evidence",
                role="evidence_bundle",
                semantic_digest=entry.bundle_digest,
            )
        for path in sorted((self.root / "projections").glob("*/*.json")):
            add(path, kind="projection", role=path.parent.name)
        for relative in self._receipts:
            add(self.root / relative, kind="receipt", role="receipt")
        for digest in _all_blob_digests(self.root / "blobs"):
            path = self.blobs.path_for(digest)
            add(
                path,
                kind="blob",
                role="blob",
                media_type="application/octet-stream",
                semantic_digest=digest,
            )
        return [refs[path] for path in sorted(refs)]


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
        if path.read_bytes() != payload:
            raise FileExistsError(f"immutable bundle object has different bytes: {path}")
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(payload)
    temp.replace(path)
    path.chmod(0o444)


def _trace_entry_from_payload(payload: dict[str, Any]) -> BundleTraceEntryV1:
    return BundleTraceEntryV1(
        trace_id=str(payload["trace_id"]),
        trace_digest=str(payload["trace_digest"]),
        capture_id=str(payload["capture_id"]),
        binding_digest=str(payload["binding_digest"]),
        sealed_path=str(payload["sealed_path"]),
        binding_path=str(payload["binding_path"]),
        segments=tuple(
            TraceSegmentManifestV1(
                segment_id=str(item["segment_id"]),
                capture_id=str(item["capture_id"]),
                sequence=int(item["sequence"]),
                digest=str(item["digest"]),
                record_count=int(item["record_count"]),
                byte_size=int(item["byte_size"]),
                first_ordinal=int(item["first_ordinal"]),
                last_ordinal=int(item["last_ordinal"]),
                first_occurred_at=str(item["first_occurred_at"]),
                last_occurred_at=str(item["last_occurred_at"]),
                relative_path=str(item["relative_path"]),
                encoding=str(item.get("encoding") or "jsonl"),
                compression=str(item.get("compression") or "none"),
                media_type=str(item.get("media_type") or "application/x-ndjson"),
                truncated_tail_bytes=int(item.get("truncated_tail_bytes") or 0),
                schema_version=str(
                    item.get("schema_version") or "synth.trace-segment-manifest.v1"
                ),
                content_digest=str(item.get("content_digest") or ""),
            )
            for item in payload.get("segments") or []
        ),
    )


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


def _safe_component(value: str, *, kind: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"unsafe {kind}: {value!r}")
    return value


def _safe_archive_member(value: str) -> str:
    if not value or "\\" in value or "\x00" in value:
        raise ValueError(f"unsafe archive path: {value!r}")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"unsafe archive path: {value!r}")
    return candidate.as_posix()


def _safe_bundle_path(root: Path, relative: str) -> Path:
    if not relative or "\\" in relative or "\x00" in relative:
        raise ValueError(f"unsafe bundle path: {relative!r}")
    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"unsafe bundle path: {relative!r}")
    resolved_root = Path(root).resolve()
    resolved = (resolved_root / candidate.as_posix()).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"bundle path escapes root: {relative!r}")
    return resolved


__all__ = [
    "CATALOG_FILE",
    "MANIFEST_FILE",
    "BundleEvidenceEntryV1",
    "BundleManifestV1",
    "BundleManifestPointerV1",
    "BundleObjectRefV1",
    "BundleTraceEntryV1",
    "LocalTraceBundle",
    "rebuild_catalog",
]
