"""Trace sources: how the annotation service finds a sealed document by digest.

The service never discovers or starts task containers. It is handed a loader
that resolves ``(trace_id, digest)`` from somewhere sealed traces already live —
a container's own ``LocalTraceBundle`` on disk, a directory of bundles, or a
host trace store. The digest is re-verified on load; a mismatch is a refusal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

from ..canonical import content_digest
from ..models.document import TraceDocumentV5
from ..store.bundle import LocalTraceBundle
from ..validation.rehydrate import trace_document_from_payload

TraceLoader = Callable[[str, str], Optional[TraceDocumentV5]]


def bundle_trace_loader(root: Path, *, promote: Callable[[TraceDocumentV5, str], TraceDocumentV5] | None = None) -> TraceLoader:
    """Resolve traces from one sealed bundle directory.

    ``promote`` may derive a richer document (for example the Craftax lane
    promotion); the loader then answers for *either* digest, and records the
    sealed source digest it came from on the returned document's provenance.
    """

    bundle = LocalTraceBundle(root)

    def load(trace_id: str, digest: str) -> TraceDocumentV5 | None:
        manifest = bundle.read_manifest()
        for entry in manifest.get("traces") or ():
            if str(entry.get("trace_id")) != trace_id:
                continue
            sealed_digest = str(entry.get("trace_digest") or "")
            if not sealed_digest:
                continue
            document = trace_document_from_payload(bundle.read_trace(sealed_digest))
            if content_digest(document) != sealed_digest:
                return None
            if document.content_digest == digest:
                return document
            if promote is not None:
                promoted = promote(document, sealed_digest)
                if promoted.content_digest == digest:
                    return promoted
        return None

    return load


def bundle_trace_refs(root: Path, *, promote: Callable[[TraceDocumentV5, str], TraceDocumentV5] | None = None) -> list[dict[str, str]]:
    """``{kind: trace_v5, id, digest}`` refs for every sealed trace in a bundle (promoted digest when asked)."""

    bundle = LocalTraceBundle(root)
    refs: list[dict[str, str]] = []
    for entry in bundle.read_manifest().get("traces") or ():
        trace_id = str(entry.get("trace_id") or "")
        sealed_digest = str(entry.get("trace_digest") or "")
        if not trace_id or not sealed_digest:
            continue
        digest = sealed_digest
        if promote is not None:
            document = trace_document_from_payload(bundle.read_trace(sealed_digest))
            digest = promote(document, sealed_digest).content_digest
        refs.append({"kind": "trace_v5", "id": trace_id, "digest": digest, "sealed_digest": sealed_digest})
    return refs


def chain_loaders(loaders: Iterable[TraceLoader]) -> TraceLoader:
    """First loader that answers wins; none answering means the trace is unavailable."""

    ordered = tuple(loaders)

    def load(trace_id: str, digest: str) -> TraceDocumentV5 | None:
        for loader in ordered:
            document = loader(trace_id, digest)
            if document is not None:
                return document
        return None

    return load


__all__ = ["TraceLoader", "bundle_trace_loader", "bundle_trace_refs", "chain_loaders"]
