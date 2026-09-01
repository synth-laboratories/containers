"""Per-sealed-trace lookup index and selector memo, keyed by the trace digest.

Resolving a citation against a sealed trace is a pure function of two immutable
inputs: the sealed document (identified by its content digest) and the selector.
``TraceDocumentV5`` finds entities by linear scan, so validating a proposal with
thousands of findings against a trace with thousands of spans is quadratic, and
every job re-resolves every citation the evidence head already carries.

``SealedTraceIndex`` fixes both without touching the models:

* ``IndexedTraceDocument`` is the same dataclass (same fields, same digest, same
  serialization) with dictionary lookups for ``span``/``message``/``event``/...;
  the real ``resolve_selector`` runs unchanged on top of it.
* ``resolve`` memoizes resolutions per selector, so a citation repeated across
  findings in one proposal, or across jobs on one trace, is resolved once.
* ``verified_bundles`` remembers which evidence revisions were validated clean
  against this trace, so a later revision only needs its appended records checked.

``SealedTraceCache`` bounds all of that with an LRU keyed by the trace digest.
Two traces never alias: a different digest is a different entry.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import fields
from typing import Any

from ..canonical import content_digest
from ..models.document import TraceDocumentV5
from ..models.selectors import SelectorResolutionV1, TraceSelectorV1, resolve_selector


class IndexedTraceDocument(TraceDocumentV5):
    """A sealed ``TraceDocumentV5`` whose entity lookups are O(1).

    No dataclass fields are added, so ``fields()``, ``asdict()``, ``content_digest``
    and ``to_dict()`` are identical to the plain document. The index is built
    lazily and kept out of the dataclass field set (a slot, never serialized).
    """

    __slots__ = ("_by_id",)

    _LOOKUPS = {
        "actor": ("actors", "actor_id"),
        "session": ("sessions", "session_id"),
        "span": ("spans", "span_id"),
        "event": ("events", "event_id"),
        "message": ("messages", "message_id"),
        "artifact": ("artifacts", "artifact_id"),
    }

    @classmethod
    def of(cls, document: TraceDocumentV5) -> "IndexedTraceDocument":
        if isinstance(document, cls):
            return document
        if not document.content_digest:
            raise ValueError("only sealed trace documents can be indexed")
        return cls(**{item.name: getattr(document, item.name) for item in fields(document)})

    def _table(self, kind: str) -> dict[str, Any]:
        try:
            tables = self._by_id
        except AttributeError:
            tables = {}
            object.__setattr__(self, "_by_id", tables)
        table = tables.get(kind)
        if table is None:
            collection, attribute = self._LOOKUPS[kind]
            table = {}
            for item in getattr(self, collection):
                # First match wins, exactly like the linear scan it replaces.
                table.setdefault(getattr(item, attribute), item)
            tables[kind] = table
        return table

    def actor(self, actor_id: str):  # type: ignore[override]
        return self._table("actor").get(actor_id)

    def session(self, session_id: str):  # type: ignore[override]
        return self._table("session").get(session_id)

    def span(self, span_id: str):  # type: ignore[override]
        return self._table("span").get(span_id)

    def event(self, event_id: str):  # type: ignore[override]
        return self._table("event").get(event_id)

    def message(self, message_id: str):  # type: ignore[override]
        return self._table("message").get(message_id)

    def artifact(self, artifact_id: str):  # type: ignore[override]
        return self._table("artifact").get(artifact_id)


class SealedTraceIndex:
    """Everything cacheable about resolving citations against one sealed trace."""

    def __init__(self, document: TraceDocumentV5, *, max_memo: int = 200_000) -> None:
        self.view = IndexedTraceDocument.of(document)
        self.trace_id = self.view.trace_id
        self.trace_digest = self.view.content_digest
        self.max_memo = max_memo
        self._resolutions: dict[TraceSelectorV1, SelectorResolutionV1] = {}
        self._selector_digests: dict[TraceSelectorV1, str] = {}
        self._verified_bundles: set[str] = set()
        self.hits = 0
        self.misses = 0

    def _bound(self, memo: dict[Any, Any]) -> None:
        if len(memo) >= self.max_memo:
            memo.clear()

    def resolve(self, selector: TraceSelectorV1) -> SelectorResolutionV1:
        """``resolve_selector`` with a per-selector memo; a foreign digest never hits."""

        if selector.trace_digest != self.trace_digest or selector.trace_id != self.trace_id:
            # Not this trace: never memoized, so nothing can alias across traces.
            return resolve_selector(self.view, selector)
        found = self._resolutions.get(selector)
        if found is not None:
            self.hits += 1
            return found
        self.misses += 1
        resolution = resolve_selector(self.view, selector)
        self._bound(self._resolutions)
        self._resolutions[selector] = resolution
        return resolution

    def selector_digest(self, selector: TraceSelectorV1) -> str:
        found = self._selector_digests.get(selector)
        if found is None:
            found = content_digest(selector)
            self._bound(self._selector_digests)
            self._selector_digests[selector] = found
        return found

    def bundle_verified(self, bundle_digest: str) -> bool:
        return bool(bundle_digest) and bundle_digest in self._verified_bundles

    def mark_bundle_verified(self, bundle_digest: str) -> None:
        if bundle_digest:
            self._verified_bundles.add(bundle_digest)

    def stats(self) -> dict[str, int]:
        return {
            "memoized_selectors": len(self._resolutions),
            "hits": self.hits,
            "misses": self.misses,
            "verified_bundles": len(self._verified_bundles),
        }


class SealedTraceCache:
    """Bounded LRU of ``SealedTraceIndex`` keyed by sealed trace digest."""

    def __init__(self, *, max_traces: int = 8, max_memo_per_trace: int = 200_000) -> None:
        self.max_traces = max(1, int(max_traces))
        self.max_memo_per_trace = max_memo_per_trace
        self._entries: OrderedDict[str, SealedTraceIndex] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, document: TraceDocumentV5) -> SealedTraceIndex:
        digest = document.content_digest
        if not digest:
            raise ValueError("only sealed trace documents can be cached")
        with self._lock:
            entry = self._entries.get(digest)
            if entry is not None:
                if entry.trace_id != document.trace_id:
                    raise ValueError(
                        f"sealed digest {digest} is claimed by traces {entry.trace_id!r} and {document.trace_id!r}"
                    )
                self._entries.move_to_end(digest)
                return entry
            entry = SealedTraceIndex(document, max_memo=self.max_memo_per_trace)
            self._entries[digest] = entry
            while len(self._entries) > self.max_traces:
                self._entries.popitem(last=False)
            return entry

    def __contains__(self, digest: str) -> bool:
        with self._lock:
            return digest in self._entries

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def digests(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


__all__ = ["IndexedTraceDocument", "SealedTraceCache", "SealedTraceIndex"]
