"""Durable local store for annotation jobs, receipts, execution traces, and evidence.

Layout::

    <root>/
      jobs.sqlite                      job index (idempotency key -> terminal job)
      catalog.sqlite                   SqliteCatalogStore over evidence bundles
      jobs/<job_id>/job.json           latest sealed job revision
      jobs/<job_id>/history/*.json     every sealed revision, append-only
      jobs/<job_id>/receipts/*.json
      jobs/<job_id>/execution_trace.json
      jobs/<job_id>/proposal.json      raw structured output (kept even when rejected)
      jobs/<job_id>/workspace/
      traces/<trace_id>/source/<digest>.json    materialized sealed authority
      traces/<trace_id>/evidence/<digest>.json  every evidence revision
      traces/<trace_id>/head.json               pointer to the current evidence head

Every read re-verifies the content digest of what it loads. A mismatch raises
``StoreCorruption``; nothing is silently repaired. ``rebuild_index`` recreates
both SQLite files from the sealed JSON, which is the only authority.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import sqlite3
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator, TypeVar

from ..canonical import content_digest, readable_json, utc_now
from ..models.document import TraceDocumentV5
from ..models.evidence import TraceEvidenceBundleV5
from ..models.standards import AnnotationStatus, AnnotationV1, ReceiptV1
from ..store.sqlite_catalog import SqliteCatalogStore
from ..validation.rehydrate import (
    build,
    evidence_bundle_from_payload,
    trace_document_from_payload,
)
from .jobs import AnnotationJobState, AnnotationJobV1
from .ledger import PaidLedger

CachedT = TypeVar("CachedT")

JOBS_SCHEMA = """
CREATE TABLE IF NOT EXISTS annotation_jobs (
    job_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    trace_digest TEXT NOT NULL,
    annotator_id TEXT NOT NULL,
    annotator_digest TEXT NOT NULL,
    mode TEXT NOT NULL,
    state TEXT NOT NULL,
    revision INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    bundle_digest TEXT,
    parent_job_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_key ON annotation_jobs (idempotency_key, state);
CREATE INDEX IF NOT EXISTS idx_jobs_trace ON annotation_jobs (trace_id, state);
CREATE TABLE IF NOT EXISTS annotation_index (
    annotation_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    trace_digest TEXT NOT NULL,
    bundle_digest TEXT NOT NULL,
    annotator_id TEXT NOT NULL,
    annotation_type TEXT NOT NULL,
    status TEXT,
    review_state TEXT,
    supersedes_id TEXT,
    job_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_annotation_trace ON annotation_index (trace_id);
CREATE INDEX IF NOT EXISTS idx_annotation_supersedes ON annotation_index (supersedes_id);
"""


class StoreCorruption(RuntimeError):
    """A sealed file no longer matches its digest. The store fails closed."""


class RevisionConflict(RuntimeError):
    """Two writers raced on the same job or evidence head."""


def _read_sealed_json(path: Path, *, expected_digest: str | None = None) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise StoreCorruption(f"unreadable sealed record {path}: {error}") from error
    if not isinstance(payload, dict):
        raise StoreCorruption(f"sealed record {path} is not an object")
    stored = str(payload.get("content_digest") or "")
    if not stored:
        raise StoreCorruption(f"sealed record {path} has no content digest")
    recomputed = content_digest(payload)
    if recomputed != stored:
        raise StoreCorruption(f"sealed record {path} digest {stored} != content {recomputed}")
    if expected_digest is not None and stored != expected_digest:
        raise StoreCorruption(f"sealed record {path} digest {stored} != expected {expected_digest}")
    return payload


def _write_immutable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    path.chmod(0o444)


def _write_mutable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    if path.exists():
        path.chmod(0o644)
    os.replace(tmp, path)


def _safe(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-:" else "_" for ch in value)
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"unsafe path component {value!r}")
    return cleaned.replace(":", "_")


class AnnotationStore:
    def __init__(self, root: Path, *, source_cache_size: int = 16, evidence_cache_size: int = 32, job_cache_size: int = 512) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "jobs").mkdir(exist_ok=True)
        (self.root / "traces").mkdir(exist_ok=True)
        self._db_lock = threading.RLock()
        self._jobs_db = self._connect()
        self.catalog = SqliteCatalogStore(self.root / "catalog.sqlite")
        self.ledger = PaidLedger(self.root / "paid_ledger")
        # Sealed files are immutable, so a digest-keyed LRU of what was already
        # read and re-verified is exact: a different digest is a different key,
        # and the head pointer (mutable, tiny) is always read from disk before
        # the digest it names is looked up. Each entry is pinned to the stat
        # signature of the file it came from; if the bytes on disk change out
        # of band the entry is dropped and the read re-verifies (and fails
        # closed) exactly as before.
        self._cache_lock = threading.Lock()
        self._source_cache: OrderedDict[tuple[str, str], tuple[tuple[int, int, int], TraceDocumentV5]] = OrderedDict()
        self._evidence_cache: OrderedDict[str, tuple[tuple[int, int, int], TraceEvidenceBundleV5]] = OrderedDict()
        self._job_cache: OrderedDict[str, tuple[tuple[int, int, int], AnnotationJobV1]] = OrderedDict()
        self.source_cache_size = max(0, int(source_cache_size))
        self.evidence_cache_size = max(0, int(evidence_cache_size))
        self.job_cache_size = max(0, int(job_cache_size))

    @staticmethod
    def _signature(path: Path) -> tuple[int, int, int] | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        return (stat.st_size, stat.st_mtime_ns, stat.st_ino)

    def _recall(self, cache: OrderedDict[Any, tuple[tuple[int, int, int], CachedT]], key: Any, path: Path) -> CachedT | None:
        with self._cache_lock:
            found = cache.get(key)
            if found is None:
                return None
            signature, value = found
            if signature != self._signature(path):
                cache.pop(key, None)
                return None
            cache.move_to_end(key)
            return value

    def _remember(self, cache: OrderedDict[Any, tuple[tuple[int, int, int], CachedT]], key: Any, value: CachedT, path: Path, size: int) -> None:
        signature = self._signature(path)
        if size <= 0 or signature is None:
            return
        with self._cache_lock:
            cache[key] = (signature, value)
            cache.move_to_end(key)
            while len(cache) > size:
                cache.popitem(last=False)

    def cache_stats(self) -> dict[str, int]:
        with self._cache_lock:
            return {"sources": len(self._source_cache), "evidence": len(self._evidence_cache), "jobs": len(self._job_cache)}

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.root / "jobs.sqlite", check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.executescript(JOBS_SCHEMA)
        connection.commit()
        return connection

    def _execute(self, sql: str, params: tuple[Any, ...] | list[Any] = ()) -> list[sqlite3.Row]:
        with self._db_lock:
            cursor = self._jobs_db.execute(sql, params)
            rows = cursor.fetchall()
            self._jobs_db.commit()
            return rows

    def close(self) -> None:
        self._jobs_db.close()
        self.catalog.close()

    # -- locking ------------------------------------------------------------------

    @contextlib.contextmanager
    def lock(self) -> Iterator[None]:
        """Process-wide exclusive lock for job creation and head updates."""

        path = self.root / ".lock"
        with path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    # -- source traces --------------------------------------------------------------

    def source_path(self, trace_id: str, digest: str) -> Path:
        return self.root / "traces" / _safe(trace_id) / "source" / f"{_safe(digest)}.json"

    def put_source(self, document: TraceDocumentV5) -> Path:
        if not document.content_digest or content_digest(document) != document.content_digest:
            raise ValueError("only sealed trace documents can be materialized")
        path = self.source_path(document.trace_id, document.content_digest)
        if not path.exists():
            _write_immutable(path, readable_json(document))
        self._remember(self._source_cache, (document.trace_id, document.content_digest), document, path, self.source_cache_size)
        return path

    def has_source(self, trace_id: str, digest: str) -> bool:
        return self.source_path(trace_id, digest).exists()

    def get_source(self, trace_id: str, digest: str) -> TraceDocumentV5 | None:
        path = self.source_path(trace_id, digest)
        cached = self._recall(self._source_cache, (trace_id, digest), path)
        if cached is not None:
            return cached
        if not path.exists():
            return None
        payload = _read_sealed_json(path, expected_digest=digest)
        document = trace_document_from_payload(payload)
        if document.content_digest != digest or content_digest(document) != digest:
            raise StoreCorruption(f"materialized trace {trace_id} does not re-seal to {digest}")
        self._remember(self._source_cache, (trace_id, digest), document, path, self.source_cache_size)
        return document

    def source_digests(self, trace_id: str) -> tuple[str, ...]:
        folder = self.root / "traces" / _safe(trace_id) / "source"
        if not folder.exists():
            return ()
        return tuple(sorted(path.stem.replace("sha256_", "sha256:") for path in folder.glob("*.json")))

    # -- jobs -------------------------------------------------------------------------

    def job_dir(self, job_id: str) -> Path:
        return self.root / "jobs" / _safe(job_id)

    def save_job(self, job: AnnotationJobV1) -> AnnotationJobV1:
        if not job.content_digest or content_digest(job) != job.content_digest:
            raise ValueError("job records must be sealed before saving")
        folder = self.job_dir(job.job_id)
        current = self.get_job(job.job_id)
        if current is not None:
            if job.revision != current.revision + 1:
                raise RevisionConflict(
                    f"job {job.job_id} revision {job.revision} does not follow {current.revision}"
                )
            if current.terminal:
                raise RevisionConflict(f"job {job.job_id} is terminal ({current.state})")
        elif job.revision != 1:
            raise RevisionConflict(f"new job {job.job_id} must start at revision 1")
        history = folder / "history" / f"{job.revision:04d}-{job.state}.json"
        if history.exists():
            raise RevisionConflict(f"job revision already recorded: {history.name}")
        text = readable_json(job)
        _write_immutable(history, text)
        _write_mutable(folder / "job.json", text)
        self._remember(self._job_cache, job.job_id, job, folder / "job.json", self.job_cache_size)
        self._execute(
            "INSERT OR REPLACE INTO annotation_jobs (job_id, idempotency_key, trace_id, "
            "trace_digest, annotator_id, annotator_digest, mode, state, revision, created_at, "
            "updated_at, content_digest, bundle_digest, parent_job_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job.job_id,
                job.idempotency_key,
                job.request.source_trace_id,
                job.request.source_trace_digest,
                job.request.annotator_id,
                job.request.annotator_digest,
                str(job.request.mode),
                str(job.state),
                job.revision,
                job.created_at,
                job.updated_at,
                job.content_digest,
                job.bundle_digest,
                job.request.parent_job_id,
            ),
        )
        return job

    def get_job(self, job_id: str) -> AnnotationJobV1 | None:
        path = self.job_dir(job_id) / "job.json"
        cached = self._recall(self._job_cache, job_id, path)
        if cached is not None:
            return cached
        if not path.exists():
            return None
        payload = _read_sealed_json(path)
        job = build(AnnotationJobV1, payload)
        self._remember(self._job_cache, job_id, job, path, self.job_cache_size)
        return job

    def job_history(self, job_id: str) -> tuple[AnnotationJobV1, ...]:
        folder = self.job_dir(job_id) / "history"
        if not folder.exists():
            return ()
        return tuple(
            build(AnnotationJobV1, _read_sealed_json(path)) for path in sorted(folder.glob("*.json"))
        )

    def find_cached_job(self, idempotency_key: str) -> AnnotationJobV1 | None:
        """The sealed or abstained job for this key, if one exists."""

        rows = self._execute(
            "SELECT job_id FROM annotation_jobs WHERE idempotency_key = ? AND state IN (?, ?) "
            "ORDER BY created_at ASC",
            (idempotency_key, AnnotationJobState.SEALED.value, AnnotationJobState.ABSTAINED.value),
        )
        for row in rows:
            job = self.get_job(str(row["job_id"]))
            if job is not None and job.cached_from_job_id is None:
                return job
        return None

    def find_active_job(self, idempotency_key: str) -> AnnotationJobV1 | None:
        rows = self._execute(
            "SELECT job_id FROM annotation_jobs WHERE idempotency_key = ? AND state IN (?, ?, ?) "
            "ORDER BY created_at ASC",
            (
                idempotency_key,
                AnnotationJobState.PREPARED.value,
                AnnotationJobState.RUNNING.value,
                AnnotationJobState.VALIDATING.value,
            ),
        )
        for row in rows:
            job = self.get_job(str(row["job_id"]))
            if job is not None:
                return job
        return None

    def list_jobs(
        self,
        *,
        trace_id: str | None = None,
        state: str | None = None,
        annotator_id: str | None = None,
    ) -> tuple[AnnotationJobV1, ...]:
        clauses: list[str] = []
        params: list[Any] = []
        if trace_id is not None:
            clauses.append("trace_id = ?")
            params.append(trace_id)
        if state is not None:
            clauses.append("state = ?")
            params.append(state)
        if annotator_id is not None:
            clauses.append("annotator_id = ?")
            params.append(annotator_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._execute(
            f"SELECT job_id FROM annotation_jobs {where} ORDER BY created_at ASC, job_id ASC",
            params,
        )
        jobs = [self.get_job(str(row["job_id"])) for row in rows]
        return tuple(job for job in jobs if job is not None)

    # -- per-job artifacts ------------------------------------------------------------

    def save_receipt(self, job_id: str, receipt: ReceiptV1) -> Path:
        path = self.job_dir(job_id) / "receipts" / f"{_safe(receipt.receipt_id)}.json"
        _write_immutable(path, readable_json(receipt))
        return path

    def receipts(self, job_id: str) -> tuple[ReceiptV1, ...]:
        folder = self.job_dir(job_id) / "receipts"
        if not folder.exists():
            return ()
        found = [build(ReceiptV1, _read_sealed_json(path)) for path in folder.glob("*.json")]
        return tuple(sorted(found, key=lambda item: (item.started_at, item.ended_at or "", item.receipt_id)))

    def save_execution_trace(self, job_id: str, document: TraceDocumentV5) -> Path:
        path = self.job_dir(job_id) / "execution_trace.json"
        _write_immutable(path, readable_json(document))
        return path

    def get_execution_trace(self, job_id: str) -> TraceDocumentV5 | None:
        path = self.job_dir(job_id) / "execution_trace.json"
        if not path.exists():
            return None
        return trace_document_from_payload(_read_sealed_json(path))

    def save_proposal(self, job_id: str, proposal: Any) -> Path:
        path = self.job_dir(job_id) / "proposal.json"
        _write_mutable(path, json.dumps(proposal, indent=2, sort_keys=True, ensure_ascii=False))
        return path

    def get_proposal(self, job_id: str) -> Any:
        path = self.job_dir(job_id) / "proposal.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_commit_intent(
        self,
        job_id: str,
        *,
        bundle: TraceEvidenceBundleV5,
        terminal_job: AnnotationJobV1,
        receipt: ReceiptV1,
        expected_prior_digest: str | None,
    ) -> Path:
        """Journal a validated terminal commit before changing the evidence head.

        The sealed bundle and terminal job make the evidence-head update replayable
        after a crash. The journal is immutable: a job has only one validated result.
        """

        path = self.job_dir(job_id) / "commit_intent.json"
        payload = {
            "expected_prior_digest": expected_prior_digest,
            "bundle": bundle.to_dict(),
            "terminal_job": terminal_job.to_dict(),
            "receipt": receipt.to_dict(),
        }
        if not path.exists():
            _write_immutable(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return path

    def get_commit_intent(
        self, job_id: str
    ) -> tuple[TraceEvidenceBundleV5, AnnotationJobV1, ReceiptV1, str | None] | None:
        path = self.job_dir(job_id) / "commit_intent.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return (
                evidence_bundle_from_payload(payload["bundle"]),
                build(AnnotationJobV1, payload["terminal_job"]),
                build(ReceiptV1, payload["receipt"]),
                payload.get("expected_prior_digest"),
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise StoreCorruption(f"unreadable commit intent for {job_id}: {error}") from error

    def ensure_evidence_committed(
        self,
        bundle: TraceEvidenceBundleV5,
        *,
        expected_prior_digest: str | None,
        job_id: str,
    ) -> TraceEvidenceBundleV5:
        """Commit a bundle once, or finish indexing an already-written revision."""

        path = self.evidence_path(bundle.trace_ref.trace_id, bundle.content_digest)
        if not path.exists():
            return self.put_evidence(
                bundle,
                expected_prior_digest=expected_prior_digest,
                job_id=job_id,
            )
        persisted = evidence_bundle_from_payload(
            _read_sealed_json(path, expected_digest=bundle.content_digest)
        )
        if persisted.content_digest != bundle.content_digest:
            raise StoreCorruption(f"commit intent bundle {bundle.content_digest} does not match persisted evidence")
        # A crash may have occurred after the immutable revision or head pointer was
        # written but before one or both indexes were updated. Indexing is idempotent.
        self.catalog.index_evidence(persisted)
        self._index_annotations(persisted, job_id=job_id)
        self._remember(self._evidence_cache, persisted.content_digest, persisted, path, self.evidence_cache_size)
        return persisted

    def workspace_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "workspace"

    # -- evidence ----------------------------------------------------------------------

    def _trace_dir(self, trace_id: str) -> Path:
        return self.root / "traces" / _safe(trace_id)

    def evidence_path(self, trace_id: str, bundle_digest: str) -> Path:
        return self._trace_dir(trace_id) / "evidence" / f"{_safe(bundle_digest)}.json"

    def evidence_head(self, trace_id: str) -> TraceEvidenceBundleV5 | None:
        pointer = self._trace_dir(trace_id) / "head.json"
        if not pointer.exists():
            return None
        try:
            head = json.loads(pointer.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise StoreCorruption(f"unreadable evidence head for {trace_id}: {error}") from error
        digest = str(head.get("bundle_digest") or "")
        if not digest:
            raise StoreCorruption(f"evidence head for {trace_id} names no bundle digest")
        return self.get_evidence(trace_id, digest)

    def get_evidence(self, trace_id: str, bundle_digest: str) -> TraceEvidenceBundleV5 | None:
        path = self.evidence_path(trace_id, bundle_digest)
        cached = self._recall(self._evidence_cache, bundle_digest, path)
        if cached is not None and cached.trace_ref.trace_id == trace_id:
            return cached
        if not path.exists():
            return None
        payload = _read_sealed_json(path, expected_digest=bundle_digest)
        bundle = evidence_bundle_from_payload(payload)
        if content_digest(bundle) != bundle_digest:
            raise StoreCorruption(f"evidence bundle {bundle_digest} does not re-seal")
        self._remember(self._evidence_cache, bundle_digest, bundle, path, self.evidence_cache_size)
        return bundle

    def evidence_bundles(self, trace_id: str) -> tuple[dict[str, Any], ...]:
        folder = self._trace_dir(trace_id) / "evidence"
        if not folder.exists():
            return ()
        head = None
        pointer = self._trace_dir(trace_id) / "head.json"
        if pointer.exists():
            head = json.loads(pointer.read_text(encoding="utf-8")).get("bundle_digest")
        rows: list[dict[str, Any]] = []
        for path in folder.glob("*.json"):
            payload = _read_sealed_json(path)
            rows.append(
                {
                    "bundle_id": payload.get("bundle_id"),
                    "bundle_digest": payload.get("content_digest"),
                    "created_at": payload.get("created_at"),
                    "trace_digest": (payload.get("trace_ref") or {}).get("content_digest"),
                    "supersedes_bundle_digest": (payload.get("metadata") or {}).get(
                        "supersedes_bundle_digest"
                    ),
                    "annotation_count": len(payload.get("annotations") or ()),
                    "verifier_result_count": len(payload.get("verifier_results") or ()),
                    "is_head": payload.get("content_digest") == head,
                }
            )
        rows.sort(key=lambda item: (str(item["created_at"]), str(item["bundle_digest"])))
        return tuple(rows)

    def put_evidence(
        self,
        bundle: TraceEvidenceBundleV5,
        *,
        expected_prior_digest: str | None,
        job_id: str | None = None,
    ) -> TraceEvidenceBundleV5:
        """Append a new evidence head; compare-and-set on the prior digest."""

        if not bundle.content_digest or content_digest(bundle) != bundle.content_digest:
            raise ValueError("evidence bundles must be sealed before saving")
        trace_id = bundle.trace_ref.trace_id
        pointer = self._trace_dir(trace_id) / "head.json"
        current_digest: str | None = None
        if pointer.exists():
            current_digest = json.loads(pointer.read_text(encoding="utf-8")).get("bundle_digest")
        if current_digest != expected_prior_digest:
            raise RevisionConflict(
                f"evidence head for {trace_id} is {current_digest}, expected {expected_prior_digest}"
            )
        declared_prior = bundle.metadata.get("supersedes_bundle_digest")
        if current_digest is not None and declared_prior != current_digest:
            raise RevisionConflict("new evidence revision does not supersede the current head")
        path = self.evidence_path(trace_id, bundle.content_digest)
        if not path.exists():
            _write_immutable(path, readable_json(bundle))
        _write_mutable(
            pointer,
            json.dumps(
                {
                    "trace_id": trace_id,
                    "trace_digest": bundle.trace_ref.content_digest,
                    "bundle_id": bundle.bundle_id,
                    "bundle_digest": bundle.content_digest,
                    "updated_at": utc_now(),
                },
                indent=2,
                sort_keys=True,
            ),
        )
        self.catalog.index_evidence(bundle)
        self._index_annotations(bundle, job_id=job_id)
        self._remember(self._evidence_cache, bundle.content_digest, bundle, path, self.evidence_cache_size)
        return bundle

    def _index_annotations(self, bundle: TraceEvidenceBundleV5, *, job_id: str | None) -> None:
        rows = [
            (
                annotation.annotation_id,
                bundle.trace_ref.trace_id,
                bundle.trace_ref.content_digest,
                bundle.content_digest,
                annotation.annotator_id,
                annotation.annotation_type,
                str(annotation.status) if annotation.status is not None else None,
                str(annotation.review_state) if annotation.review_state is not None else None,
                annotation.supersedes_id,
                job_id,
            )
            for annotation in bundle.annotations
        ]
        if not rows:
            return
        # One transaction for the whole revision: a commit per annotation was the
        # dominant persistence cost for large proposals.
        with self._db_lock:
            self._jobs_db.executemany(
                "INSERT OR IGNORE INTO annotation_index (annotation_id, trace_id, trace_digest, "
                "bundle_digest, annotator_id, annotation_type, status, review_state, "
                "supersedes_id, job_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self._jobs_db.commit()

    def annotations(
        self,
        trace_id: str,
        *,
        annotator_id: str | None = None,
        annotation_type: str | None = None,
        label: str | None = None,
        status: str | None = None,
        review_state: str | None = None,
        include_superseded: bool = False,
        target_entity_id: str | None = None,
    ) -> tuple[AnnotationV1, ...]:
        head = self.evidence_head(trace_id)
        if head is None:
            return ()
        superseded = {item.supersedes_id for item in head.annotations if item.supersedes_id}
        found: list[AnnotationV1] = []
        for annotation in head.annotations:
            if not include_superseded and annotation.annotation_id in superseded:
                continue
            if annotator_id is not None and annotation.annotator_id != annotator_id:
                continue
            if annotation_type is not None and annotation.annotation_type != annotation_type:
                continue
            if label is not None and label not in annotation.labels:
                continue
            if status is not None:
                current = str(annotation.status) if annotation.status is not None else AnnotationStatus.APPLIED.value
                if current != status:
                    continue
            if review_state is not None and (
                annotation.review_state is None or str(annotation.review_state) != review_state
            ):
                continue
            if target_entity_id is not None and annotation.target.entity_id != target_entity_id:
                continue
            found.append(annotation)
        return tuple(found)

    def get_annotation(self, annotation_id: str) -> tuple[AnnotationV1, str] | None:
        rows = self._execute(
            "SELECT trace_id FROM annotation_index WHERE annotation_id = ?", (annotation_id,)
        )
        row = rows[0] if rows else None
        if row is None:
            return None
        trace_id = str(row["trace_id"])
        head = self.evidence_head(trace_id)
        if head is None:
            return None
        for annotation in head.annotations:
            if annotation.annotation_id == annotation_id:
                return annotation, trace_id
        return None

    def annotation_job(self, annotation_id: str) -> str | None:
        rows = self._execute(
            "SELECT job_id FROM annotation_index WHERE annotation_id = ?", (annotation_id,)
        )
        row = rows[0] if rows else None
        return str(row["job_id"]) if row is not None and row["job_id"] else None

    # -- integrity -----------------------------------------------------------------------

    def verify(self) -> dict[str, Any]:
        """Re-verify every sealed file; return counts and the first problem per kind."""

        report: dict[str, Any] = {"jobs": 0, "evidence": 0, "sources": 0, "problems": []}
        for path in (self.root / "jobs").glob("*/job.json"):
            try:
                _read_sealed_json(path)
                report["jobs"] += 1
            except StoreCorruption as error:
                report["problems"].append(str(error))
        for path in (self.root / "traces").glob("*/evidence/*.json"):
            try:
                _read_sealed_json(path)
                report["evidence"] += 1
            except StoreCorruption as error:
                report["problems"].append(str(error))
        for path in (self.root / "traces").glob("*/source/*.json"):
            try:
                _read_sealed_json(path)
                report["sources"] += 1
            except StoreCorruption as error:
                report["problems"].append(str(error))
        report["ok"] = not report["problems"]
        return report

    def rebuild_index(self) -> dict[str, int]:
        """Recreate the SQLite indexes from sealed JSON authority."""

        self._jobs_db.close()
        self.catalog.close()
        for name in ("jobs.sqlite", "catalog.sqlite"):
            path = self.root / name
            if path.exists():
                path.unlink()
        self._jobs_db = self._connect()
        self.catalog = SqliteCatalogStore(self.root / "catalog.sqlite")
        counts = {"jobs": 0, "evidence": 0}
        for path in sorted((self.root / "jobs").glob("*/job.json")):
            job = build(AnnotationJobV1, _read_sealed_json(path))
            self._execute(
                "INSERT OR REPLACE INTO annotation_jobs (job_id, idempotency_key, trace_id, "
                "trace_digest, annotator_id, annotator_digest, mode, state, revision, created_at, "
                "updated_at, content_digest, bundle_digest, parent_job_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job.job_id,
                    job.idempotency_key,
                    job.request.source_trace_id,
                    job.request.source_trace_digest,
                    job.request.annotator_id,
                    job.request.annotator_digest,
                    str(job.request.mode),
                    str(job.state),
                    job.revision,
                    job.created_at,
                    job.updated_at,
                    job.content_digest,
                    job.bundle_digest,
                    job.request.parent_job_id,
                ),
            )
            counts["jobs"] += 1
        for trace_dir in sorted((self.root / "traces").glob("*")):
            pointer = trace_dir / "head.json"
            if not pointer.exists():
                continue
            head = json.loads(pointer.read_text(encoding="utf-8"))
            bundle = self.get_evidence(str(head["trace_id"]), str(head["bundle_digest"]))
            if bundle is None:
                raise StoreCorruption(f"evidence head {head} points at a missing bundle")
            self.catalog.index_evidence(bundle)
            self._index_annotations(bundle, job_id=None)
            counts["evidence"] += 1
        return counts


__all__ = ["AnnotationStore", "RevisionConflict", "StoreCorruption"]
