"""Store interfaces. Push 1 ships local implementations; Stage 2 adds cloud ones.

The interfaces exist now so a managed S3 ``BlobStore`` or a Factory Turso
``CatalogStore`` can be added later without changing trace identity: bodies are
content-addressed, and the catalog is a rebuildable projection of the manifests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, runtime_checkable

from synth_containers.serde import JsonDataclassMixin

from ..models.document import TraceDocumentV5
from ..models.evidence import TraceEvidenceBundleV5


@dataclass(frozen=True, slots=True)
class BlobMetadataV1(JsonDataclassMixin):
    digest: str
    byte_size: int
    media_type: str = "application/octet-stream"
    etag: str | None = None
    uri: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BlobPutResultV1(JsonDataclassMixin):
    digest: str
    created: bool
    metadata: BlobMetadataV1


@runtime_checkable
class BlobStore(Protocol):
    """Content-addressed immutable bodies."""

    def put(self, content: bytes) -> str:
        """Store bytes and return their ``sha256:`` digest."""

    def put_if_absent(
        self,
        content: bytes,
        *,
        media_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> BlobPutResultV1: ...

    def get(
        self,
        digest: str,
        *,
        byte_range: tuple[int, int] | None = None,
    ) -> bytes:
        """Return the bytes for a digest, verifying it on read."""

    def has(self, digest: str) -> bool: ...

    def head(self, digest: str) -> BlobMetadataV1: ...

    def uri(self, digest: str) -> str:
        """Return the store-relative locator for a digest."""


@runtime_checkable
class CatalogStore(Protocol):
    """Rebuildable structured-search projection over sealed traces and evidence."""

    def index_trace(self, document: TraceDocumentV5) -> None: ...

    def index_evidence(self, bundle: TraceEvidenceBundleV5) -> None: ...

    def traces(self) -> Iterable[dict[str, Any]]: ...

    def entities(
        self,
        *,
        trace_id: str | None = None,
        kind: str | None = None,
        limit: int = 100,
    ) -> Iterable[dict[str, Any]]: ...

    def relationships(self, *, trace_id: str | None = None) -> Iterable[dict[str, Any]]: ...

    def aliases(self, *, namespace: str | None = None) -> Iterable[dict[str, Any]]: ...

    def reset(self) -> None:
        """Drop the projection so it can be rebuilt from manifests."""


class TraceStore:
    """Composition of a blob store and a catalog store.

    The catalog is always derivable from the blobs plus the bundle manifest, which is
    what makes ``rebuild`` safe and what keeps the bundle useful without SQLite.
    """

    def __init__(self, *, blobs: BlobStore, catalog: CatalogStore) -> None:
        self.blobs = blobs
        self.catalog = catalog

    def put_trace(self, document: TraceDocumentV5) -> str:
        from ..canonical import canonical_bytes

        digest = self.blobs.put(canonical_bytes(document))
        self.catalog.index_trace(document)
        return digest

    def put_evidence(self, bundle: TraceEvidenceBundleV5) -> str:
        from ..canonical import canonical_bytes

        digest = self.blobs.put(canonical_bytes(bundle))
        self.catalog.index_evidence(bundle)
        return digest


__all__ = [
    "BlobMetadataV1",
    "BlobPutResultV1",
    "BlobStore",
    "CatalogStore",
    "TraceStore",
]
