"""SQLite ``CatalogStore`` — a rebuildable structured-search projection.

The catalog holds compact rows, not a second copy of the payloads: each entity row
carries its identity, owner, order, digest, a few typed facts, and a selector back to
the authoritative record. Dropping the file and rebuilding from the bundle manifest
must produce the same rows, which is what makes it a projection rather than an
authority.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional

from ..canonical import canonical_text
from ..models.document import TraceDocumentV5
from ..models.evidence import TraceEvidenceBundleV5
from .projection import catalog_projection


CATALOG_SCHEMA_VERSION = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS catalog_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trace_documents (
    trace_id TEXT NOT NULL,
    trace_digest TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    trace_kind TEXT NOT NULL,
    capture_id TEXT NOT NULL,
    binding_digest TEXT NOT NULL,
    lifecycle_status TEXT NOT NULL,
    capture_status TEXT NOT NULL,
    started_at TEXT,
    ended_at TEXT,
    actor_count INTEGER NOT NULL,
    span_count INTEGER NOT NULL,
    event_count INTEGER NOT NULL,
    message_count INTEGER NOT NULL,
    artifact_count INTEGER NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    usage_provenance TEXT,
    task_id TEXT,
    run_id TEXT,
    correlation_id TEXT
);
CREATE TABLE IF NOT EXISTS trace_entities (
    trace_digest TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    owner_actor_id TEXT,
    owner_session_id TEXT,
    source_order INTEGER,
    occurred_at TEXT,
    content_digest TEXT,
    facts TEXT NOT NULL,
    PRIMARY KEY (trace_digest, entity_id, kind)
);
CREATE TABLE IF NOT EXISTS trace_relationships (
    trace_digest TEXT NOT NULL,
    source_entity_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    target_entity_id TEXT NOT NULL,
    source_order INTEGER
);
CREATE TABLE IF NOT EXISTS trace_aliases (
    trace_digest TEXT NOT NULL,
    namespace TEXT NOT NULL,
    value TEXT NOT NULL,
    target_id TEXT NOT NULL,
    target_kind TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence_records (
    trace_digest TEXT NOT NULL,
    bundle_id TEXT NOT NULL,
    record_kind TEXT NOT NULL,
    record_id TEXT NOT NULL,
    definition_id TEXT,
    subject_entity_id TEXT,
    grounding TEXT,
    value REAL,
    verdict TEXT,
    facts TEXT NOT NULL,
    PRIMARY KEY (bundle_id, record_kind, record_id)
);
CREATE INDEX IF NOT EXISTS idx_entities_kind ON trace_entities (trace_digest, kind);
CREATE INDEX IF NOT EXISTS idx_entities_actor ON trace_entities (trace_digest, owner_actor_id);
CREATE INDEX IF NOT EXISTS idx_relationships ON trace_relationships (trace_digest, relation);
CREATE INDEX IF NOT EXISTS idx_aliases ON trace_aliases (namespace, value);
CREATE INDEX IF NOT EXISTS idx_evidence_kind ON evidence_records (trace_digest, record_kind);
CREATE VIRTUAL TABLE IF NOT EXISTS trace_search USING fts5(
    trace_digest UNINDEXED, entity_id UNINDEXED, kind UNINDEXED, text
);
CREATE VIRTUAL TABLE IF NOT EXISTS evidence_search USING fts5(
    trace_digest UNINDEXED,
    bundle_id UNINDEXED,
    record_id UNINDEXED,
    kind UNINDEXED,
    text
);
"""


class SqliteCatalogStore:
    """SQLite-backed catalog. The file is disposable; the manifests are not."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(_SCHEMA)
        self._connection.execute(
            "INSERT OR REPLACE INTO catalog_meta (key, value) VALUES (?, ?)",
            ("schema_version", str(CATALOG_SCHEMA_VERSION)),
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def reset(self) -> None:
        for table in (
            "trace_documents",
            "trace_entities",
            "trace_relationships",
            "trace_aliases",
            "evidence_records",
            "trace_search",
            "evidence_search",
        ):
            self._connection.execute(f"DELETE FROM {table}")
        self._connection.commit()

    # -- indexing ----------------------------------------------------------------

    def index_trace(self, document: TraceDocumentV5) -> None:
        if not document.content_digest:
            raise ValueError("only sealed trace documents can be indexed")
        digest = document.content_digest
        projected = catalog_projection(document)
        projected_document = projected["documents"][0]
        self._connection.execute("DELETE FROM trace_documents WHERE trace_digest = ?", (digest,))
        for table in ("trace_entities", "trace_relationships", "trace_aliases"):
            self._connection.execute(f"DELETE FROM {table} WHERE trace_digest = ?", (digest,))
        self._connection.execute("DELETE FROM trace_search WHERE trace_digest = ?", (digest,))
        self._connection.execute(
            """
            INSERT INTO trace_documents VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                projected_document["trace_id"],
                projected_document["trace_digest"],
                projected_document["schema_version"],
                projected_document["trace_kind"],
                projected_document["capture_id"],
                projected_document["binding_digest"],
                projected_document["lifecycle_status"],
                projected_document["capture_status"],
                projected_document["started_at"],
                projected_document["ended_at"],
                projected_document["actor_count"],
                projected_document["span_count"],
                projected_document["event_count"],
                projected_document["message_count"],
                projected_document["artifact_count"],
                projected_document["prompt_tokens"],
                projected_document["completion_tokens"],
                projected_document["usage_provenance"],
                projected_document["task_id"],
                projected_document["run_id"],
                projected_document["correlation_id"],
            ),
        )
        rows: list[tuple[Any, ...]] = []
        edges: list[tuple[Any, ...]] = []
        for actor in document.actors:
            rows.append(
                (
                    digest,
                    actor.actor_id,
                    "actor",
                    actor.actor_id,
                    None,
                    None,
                    None,
                    actor.content_digest,
                    json.dumps(
                        {"kind": str(actor.kind), "name": actor.display_name, "role": actor.role},
                        sort_keys=True,
                    ),
                )
            )
        for session in document.sessions:
            rows.append(
                (
                    digest,
                    session.session_id,
                    "session",
                    session.actor_id,
                    session.session_id,
                    None,
                    session.started_at,
                    session.content_digest,
                    json.dumps(
                        {
                            "status": str(session.status),
                            "coverage": session.coverage.to_dict(),
                        },
                        sort_keys=True,
                    ),
                )
            )
            edges.append((digest, session.actor_id, "owns_session", session.session_id, None))
        for index, span in enumerate(document.spans):
            rows.append(
                (
                    digest,
                    span.span_id,
                    "span",
                    span.actor_id,
                    span.session_id,
                    index,
                    span.started_at,
                    span.content_digest,
                    json.dumps(
                        {
                            "span_kind": str(span.span_kind),
                            "status": str(span.status),
                            "detail": span.detail,
                            "usage": span.usage.to_dict() if span.usage else None,
                        },
                        sort_keys=True,
                    ),
                )
            )
            if span.parent_span_id:
                edges.append((digest, span.parent_span_id, "parent_of", span.span_id, index))
            for message_id in span.output_message_ids:
                edges.append((digest, span.span_id, "produced_message", message_id, index))
        for index, event in enumerate(document.events):
            rows.append(
                (
                    digest,
                    event.event_id,
                    "event",
                    event.actor_id,
                    event.session_id,
                    event.order.chronological_sequence or index,
                    event.occurred_at,
                    event.content_digest,
                    json.dumps(
                        {"event_type": str(event.event_type), "payload": event.payload},
                        sort_keys=True,
                    ),
                )
            )
            for parent in event.caused_by_event_ids:
                edges.append((digest, parent, "caused", event.event_id, index))
        for index, message in enumerate(document.messages):
            rows.append(
                (
                    digest,
                    message.message_id,
                    "message",
                    message.sender_actor_id,
                    message.session_id,
                    index,
                    message.occurred_at,
                    message.content_digest,
                    json.dumps(
                        {
                            "role": str(message.role),
                            "part_types": [str(part.type) for part in message.parts],
                            "text_preview": message.text()[:512],
                        },
                        sort_keys=True,
                    ),
                )
            )
        for index, artifact in enumerate(document.artifacts):
            rows.append(
                (
                    digest,
                    artifact.artifact_id,
                    "artifact",
                    None,
                    None,
                    index,
                    artifact.observed_at,
                    artifact.digest,
                    json.dumps(
                        {
                            "role": str(artifact.role),
                            "media_type": artifact.media_type,
                            "size_bytes": artifact.size_bytes,
                            "logical_name": artifact.logical_name,
                            "uri": artifact.uri,
                        },
                        sort_keys=True,
                    ),
                )
            )
        rows = [
            (
                row["trace_digest"],
                row["entity_id"],
                row["kind"],
                row["owner_actor_id"],
                row["owner_session_id"],
                row["source_order"],
                row["occurred_at"],
                row["content_digest"],
                row["facts"],
            )
            for row in projected["entities"]
        ]
        edges = [
            (
                row["trace_digest"],
                row["source_entity_id"],
                row["relation"],
                row["target_entity_id"],
                row["source_order"],
            )
            for row in projected["relationships"]
        ]
        self._connection.executemany(
            "INSERT OR REPLACE INTO trace_entities VALUES (?,?,?,?,?,?,?,?,?)", rows
        )
        self._connection.executemany(
            "INSERT INTO trace_search (trace_digest, entity_id, kind, text) VALUES (?,?,?,?)",
            ((row[0], row[1], row[2], row[8]) for row in rows),
        )
        self._connection.executemany("INSERT INTO trace_relationships VALUES (?,?,?,?,?)", edges)
        self._connection.executemany(
            "INSERT INTO trace_aliases VALUES (?,?,?,?,?)",
            [
                (
                    row["trace_digest"],
                    row["namespace"],
                    row["value"],
                    row["target_id"],
                    row["target_kind"],
                )
                for row in projected["aliases"]
            ],
        )
        self._connection.commit()

    def index_evidence(self, bundle: TraceEvidenceBundleV5) -> None:
        digest = bundle.trace_ref.content_digest
        projected = catalog_projection(bundle)
        self._connection.execute(
            "DELETE FROM evidence_records WHERE bundle_id = ?", (bundle.bundle_id,)
        )
        self._connection.execute(
            "DELETE FROM evidence_search WHERE bundle_id = ?", (bundle.bundle_id,)
        )
        rows: list[tuple[Any, ...]] = []
        for annotation in bundle.annotations:
            rows.append(
                (
                    digest,
                    bundle.bundle_id,
                    "annotation",
                    annotation.annotation_id,
                    annotation.annotator_id,
                    annotation.target.entity_id,
                    str(annotation.grounding),
                    annotation.confidence,
                    ",".join(annotation.labels),
                    canonical_text(annotation.payload),
                )
            )
        for result in bundle.verifier_results:
            rows.append(
                (
                    digest,
                    bundle.bundle_id,
                    "verifier_result",
                    result.verifier_result_id,
                    result.verifier_id,
                    result.subject.entity_id,
                    str(result.grounding),
                    result.score,
                    result.verdict,
                    canonical_text(
                        {
                            "execution_status": str(result.execution_status),
                            "verification_status": str(result.verification_status),
                            "criteria": [item.to_dict() for item in result.criterion_results],
                        }
                    ),
                )
            )
        for record in bundle.reward_records:
            rows.append(
                (
                    digest,
                    bundle.bundle_id,
                    "reward_record",
                    record.reward_record_id,
                    record.reward_id,
                    record.subject.entity_id,
                    str(record.grounding),
                    record.value,
                    record.provenance,
                    canonical_text({"components": record.components, "position": record.position}),
                )
            )
        for aggregation in bundle.reward_aggregations:
            rows.append(
                (
                    digest,
                    bundle.bundle_id,
                    "reward_aggregation",
                    aggregation.aggregation_id,
                    aggregation.reward_id,
                    None,
                    None,
                    aggregation.value,
                    aggregation.grouping,
                    canonical_text({"inputs": list(aggregation.input_reward_record_ids)}),
                )
            )
        for evaluation in bundle.evaluation_results:
            rows.append(
                (
                    digest,
                    bundle.bundle_id,
                    "evaluation_result",
                    evaluation.evaluation_id,
                    evaluation.task_id,
                    evaluation.subject.entity_id,
                    None,
                    evaluation.aggregate_score,
                    str(evaluation.execution_status),
                    canonical_text(evaluation.objective_metrics),
                )
            )
        for verdict in bundle.benchmark_verdicts:
            rows.append(
                (
                    digest,
                    bundle.bundle_id,
                    "benchmark_verdict",
                    verdict.verdict_id,
                    verdict.benchmark_authority,
                    None,
                    None,
                    verdict.threshold,
                    verdict.decision,
                    canonical_text({"failure_reasons": list(verdict.failure_reasons)}),
                )
            )
        rows = [
            (
                row["trace_digest"],
                row["bundle_id"],
                row["record_kind"],
                row["record_id"],
                row["definition_id"],
                row["subject_entity_id"],
                row["grounding"],
                row["value"],
                row["verdict"],
                row["facts"],
            )
            for row in projected["evidence"]
        ]
        self._connection.executemany(
            "INSERT OR REPLACE INTO evidence_records VALUES (?,?,?,?,?,?,?,?,?,?)", rows
        )
        self._connection.executemany(
            "INSERT INTO evidence_search "
            "(trace_digest, bundle_id, record_id, kind, text) VALUES (?,?,?,?,?)",
            (
                (
                    row["trace_digest"],
                    row["bundle_id"],
                    row["record_id"],
                    row["record_kind"],
                    canonical_text(
                        {
                            "record_id": row["record_id"],
                            "definition_id": row["definition_id"],
                            "verdict": row["verdict"],
                            "facts": json.loads(row["facts"]),
                        }
                    ),
                )
                for row in projected["evidence"]
            ),
        )
        self._connection.commit()

    # -- queries -----------------------------------------------------------------

    def traces(self) -> Iterable[dict[str, Any]]:
        cursor = self._connection.execute("SELECT * FROM trace_documents ORDER BY started_at")
        return [dict(row) for row in cursor.fetchall()]

    def search(
        self, query: str, *, trace_digest: str | None = None, limit: int = 100
    ) -> Iterable[dict[str, Any]]:
        sql = (
            "SELECT trace_digest, entity_id, kind, "
            "snippet(trace_search, 3, '[', ']', '…', 16) AS snippet "
            "FROM trace_search WHERE trace_search MATCH ?"
        )
        params: list[Any] = [query]
        if trace_digest:
            sql += " AND trace_digest = ?"
            params.append(trace_digest)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        return [dict(row) for row in self._connection.execute(sql, params).fetchall()]

    def search_evidence(
        self,
        query: str,
        *,
        trace_digest: Optional[str] = None,
        record_kind: Optional[str] = None,
        limit: int = 100,
    ) -> Iterable[dict[str, Any]]:
        if int(limit) <= 0:
            raise ValueError("search limit must be positive")
        sql = (
            "SELECT DISTINCT trace_digest, record_id, kind, "
            "snippet(evidence_search, 4, '[', ']', '…', 16) AS snippet "
            "FROM evidence_search WHERE evidence_search MATCH ?"
        )
        params: list[Any] = [query]
        if trace_digest is not None:
            sql += " AND trace_digest = ?"
            params.append(trace_digest)
        if record_kind is not None:
            sql += " AND kind = ?"
            params.append(record_kind)
        sql += " ORDER BY rank LIMIT ?"
        params.append(int(limit))
        return [
            dict(row)
            for row in self._connection.execute(sql, params).fetchall()
        ]

    def query_traces(
        self,
        *,
        query: str | None = None,
        trace_id: str | None = None,
        trace_digest: str | None = None,
        task_id: str | None = None,
        run_id: str | None = None,
        correlation_id: str | None = None,
        actor_id: str | None = None,
        session_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        event_kind: str | None = None,
        span_kind: str | None = None,
        criterion_id: str | None = None,
        judgment_id: str | None = None,
        annotation_id: str | None = None,
        annotator_id: Optional[str] = None,
        annotation_type: Optional[str] = None,
        annotation_label: Optional[str] = None,
        annotation_status: Optional[str] = None,
        annotation_review_state: Optional[str] = None,
        annotation_confidence_min: Optional[float] = None,
        annotation_confidence_max: Optional[float] = None,
        reward_id: str | None = None,
        workflow_address: str | None = None,
        started_after: str | None = None,
        started_before: str | None = None,
        completeness: str | None = None,
        visibility: str | None = None,
        digest: str | None = None,
        reward_min: float | None = None,
        reward_max: float | None = None,
        limit: int = 100,
    ) -> Iterable[dict[str, Any]]:
        """Return trace rows satisfying an exact conjunction of local filters.

        Immutable trace/evidence bodies remain authoritative; this method uses only
        their disposable catalog projection. Every entity/evidence predicate is an
        independent ``EXISTS`` clause, so compound filters require all cited facts
        to occur in the same trace without accidentally requiring one row to carry
        unrelated actor, event, and reward fields.
        """

        clauses: list[str] = []
        params: list[Any] = []

        def exact(column: str, value: str | None) -> None:
            if value is not None:
                clauses.append(f"d.{column} = ?")
                params.append(value)

        exact("trace_id", trace_id)
        exact("trace_digest", trace_digest)
        exact("task_id", task_id)
        exact("run_id", run_id)
        exact("correlation_id", correlation_id)
        exact("capture_status", completeness)
        if query:
            clauses.append(
                "d.trace_digest IN ("
                "SELECT trace_digest FROM trace_search WHERE trace_search MATCH ? "
                "UNION "
                "SELECT trace_digest FROM evidence_search WHERE evidence_search MATCH ?"
                ")"
            )
            params.extend((query, query))
        if actor_id:
            clauses.append(
                "EXISTS (SELECT 1 FROM trace_entities e "
                "WHERE e.trace_digest = d.trace_digest AND e.owner_actor_id = ?)"
            )
            params.append(actor_id)
        if session_id:
            clauses.append(
                "EXISTS (SELECT 1 FROM trace_entities e "
                "WHERE e.trace_digest = d.trace_digest AND e.owner_session_id = ?)"
            )
            params.append(session_id)
        if provider:
            clauses.append(
                "EXISTS (SELECT 1 FROM trace_entities e "
                "WHERE e.trace_digest = d.trace_digest AND "
                "json_extract(e.facts, '$.provider') = ?)"
            )
            params.append(provider)
        if model:
            clauses.append(
                "EXISTS (SELECT 1 FROM trace_entities e "
                "WHERE e.trace_digest = d.trace_digest AND "
                "json_extract(e.facts, '$.model') = ?)"
            )
            params.append(model)
        if event_kind:
            clauses.append(
                "EXISTS (SELECT 1 FROM trace_entities e "
                "WHERE e.trace_digest = d.trace_digest AND e.kind = 'event' "
                "AND json_extract(e.facts, '$.event_type') = ?)"
            )
            params.append(event_kind)
        if span_kind:
            clauses.append(
                "EXISTS (SELECT 1 FROM trace_entities e "
                "WHERE e.trace_digest = d.trace_digest AND e.kind = 'span' "
                "AND json_extract(e.facts, '$.span_kind') = ?)"
            )
            params.append(span_kind)
        if criterion_id:
            clauses.append(
                "EXISTS (SELECT 1 FROM evidence_records er "
                "WHERE er.trace_digest = d.trace_digest AND ("
                "(er.record_kind = 'judgment' AND er.definition_id = ?) OR "
                "(er.record_kind = 'verifier_result' AND EXISTS ("
                "SELECT 1 FROM json_each("
                "json_extract(er.facts, '$.criterion_results')) cr "
                "WHERE json_extract(cr.value, '$.criterion_id') = ?))))"
            )
            params.extend((criterion_id, criterion_id))
        if judgment_id:
            clauses.append(
                "EXISTS (SELECT 1 FROM evidence_records er "
                "WHERE er.trace_digest = d.trace_digest "
                "AND er.record_kind = 'judgment' AND er.record_id = ?)"
            )
            params.append(judgment_id)
        if annotation_id:
            clauses.append(
                "EXISTS (SELECT 1 FROM evidence_records er "
                "WHERE er.trace_digest = d.trace_digest "
                "AND er.record_kind = 'annotation' AND er.record_id = ?)"
            )
            params.append(annotation_id)
        annotation_filters: list[str] = [
            "er.trace_digest = d.trace_digest",
            "er.record_kind = 'annotation'",
        ]
        annotation_params: list[Any] = []
        if annotator_id is not None:
            annotation_filters.append("er.definition_id = ?")
            annotation_params.append(annotator_id)
        if annotation_type is not None:
            annotation_filters.append(
                "json_extract(er.facts, '$.annotation_type') = ?"
            )
            annotation_params.append(annotation_type)
        if annotation_status is not None:
            annotation_filters.append("json_extract(er.facts, '$.status') = ?")
            annotation_params.append(annotation_status)
        if annotation_review_state is not None:
            annotation_filters.append(
                "json_extract(er.facts, '$.review_state') = ?"
            )
            annotation_params.append(annotation_review_state)
        if annotation_confidence_min is not None:
            annotation_filters.append("er.value >= ?")
            annotation_params.append(float(annotation_confidence_min))
        if annotation_confidence_max is not None:
            annotation_filters.append("er.value <= ?")
            annotation_params.append(float(annotation_confidence_max))
        if annotation_label is not None:
            annotation_filters.append(
                "EXISTS (SELECT 1 FROM json_each("
                "json_extract(er.facts, '$.labels')) labels "
                "WHERE labels.value = ?)"
            )
            annotation_params.append(annotation_label)
        if len(annotation_filters) > 2:
            clauses.append(
                "EXISTS (SELECT 1 FROM evidence_records er WHERE "
                + " AND ".join(annotation_filters)
                + ")"
            )
            params.extend(annotation_params)
        if reward_id or reward_min is not None or reward_max is not None:
            reward_predicates = [
                "er.trace_digest = d.trace_digest",
                "er.record_kind IN ('reward_record', 'reward_aggregation')",
            ]
            reward_params: list[Any] = []
            if reward_id:
                reward_predicates.append("er.definition_id = ?")
                reward_params.append(reward_id)
            if reward_min is not None:
                reward_predicates.append("er.value >= ?")
                reward_params.append(float(reward_min))
            if reward_max is not None:
                reward_predicates.append("er.value <= ?")
                reward_params.append(float(reward_max))
            clauses.append(
                "EXISTS (SELECT 1 FROM evidence_records er WHERE "
                + " AND ".join(reward_predicates)
                + ")"
            )
            params.extend(reward_params)
        if workflow_address:
            clauses.append(
                "EXISTS (SELECT 1 FROM trace_aliases a "
                "WHERE a.trace_digest = d.trace_digest "
                "AND a.namespace = 'workflow_address' AND a.value = ?)"
            )
            params.append(workflow_address)
        if started_after:
            clauses.append("julianday(d.started_at) >= julianday(?)")
            params.append(started_after)
        if started_before:
            clauses.append("julianday(d.started_at) <= julianday(?)")
            params.append(started_before)
        if visibility:
            clauses.append(
                "EXISTS (SELECT 1 FROM trace_entities e "
                "WHERE e.trace_digest = d.trace_digest "
                "AND json_extract(e.facts, '$.trace_visibility') = ?)"
            )
            params.append(visibility)
        if digest:
            clauses.append(
                "(d.trace_digest = ? OR EXISTS ("
                "SELECT 1 FROM trace_entities e "
                "WHERE e.trace_digest = d.trace_digest AND e.content_digest = ?"
                "))"
            )
            params.extend((digest, digest))
        if int(limit) <= 0:
            raise ValueError("search limit must be positive")
        if (
            reward_min is not None
            and reward_max is not None
            and float(reward_min) > float(reward_max)
        ):
            raise ValueError("reward_min must not exceed reward_max")
        if (
            annotation_confidence_min is not None
            and annotation_confidence_max is not None
            and float(annotation_confidence_min)
            > float(annotation_confidence_max)
        ):
            raise ValueError(
                "annotation_confidence_min must not exceed annotation_confidence_max"
            )
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        return [
            dict(row)
            for row in self._connection.execute(
                f"SELECT d.* FROM trace_documents d {where} "
                "ORDER BY d.started_at, d.trace_digest LIMIT ?",
                params,
            ).fetchall()
        ]

    def entities(
        self,
        *,
        trace_id: str | None = None,
        kind: str | None = None,
        limit: int = 100,
    ) -> Iterable[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if trace_id:
            clauses.append(
                "trace_digest IN (SELECT trace_digest FROM trace_documents WHERE trace_id = ?)"
            )
            params.append(trace_id)
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        cursor = self._connection.execute(
            f"SELECT * FROM trace_entities {where} ORDER BY source_order, entity_id LIMIT ?",
            params,
        )
        return [dict(row) for row in cursor.fetchall()]

    def relationships(self, *, trace_id: str | None = None) -> Iterable[dict[str, Any]]:
        if trace_id:
            cursor = self._connection.execute(
                """
                SELECT * FROM trace_relationships
                WHERE trace_digest IN (
                    SELECT trace_digest FROM trace_documents WHERE trace_id = ?
                )
                ORDER BY source_order
                """,
                (trace_id,),
            )
        else:
            cursor = self._connection.execute(
                "SELECT * FROM trace_relationships ORDER BY source_order"
            )
        return [dict(row) for row in cursor.fetchall()]

    def aliases(self, *, namespace: str | None = None) -> Iterable[dict[str, Any]]:
        if namespace:
            cursor = self._connection.execute(
                "SELECT * FROM trace_aliases WHERE namespace = ?", (namespace,)
            )
        else:
            cursor = self._connection.execute("SELECT * FROM trace_aliases")
        return [dict(row) for row in cursor.fetchall()]

    def evidence(
        self,
        *,
        trace_digest: str | None = None,
        record_kind: str | None = None,
    ) -> Iterable[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if trace_digest:
            clauses.append("trace_digest = ?")
            params.append(trace_digest)
        if record_kind:
            clauses.append("record_kind = ?")
            params.append(record_kind)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cursor = self._connection.execute(
            f"SELECT * FROM evidence_records {where} ORDER BY record_kind, record_id", params
        )
        return [dict(row) for row in cursor.fetchall()]

    def annotation_facets(
        self,
        facet: str,
        *,
        trace_digest: Optional[str] = None,
        annotator_id: Optional[str] = None,
        annotation_type: Optional[str] = None,
        include_superseded: bool = False,
        limit: int = 100,
    ) -> Iterable[dict[str, Any]]:
        if int(limit) <= 0:
            raise ValueError("facet limit must be positive")
        clauses: list[str] = []
        params: list[Any] = []
        if trace_digest is not None:
            clauses.append("a.trace_digest = ?")
            params.append(trace_digest)
        if annotator_id is not None:
            clauses.append("a.definition_id = ?")
            params.append(annotator_id)
        if annotation_type is not None:
            clauses.append("json_extract(a.facts, '$.annotation_type') = ?")
            params.append(annotation_type)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        annotation_records = (
            "WITH annotation_records AS ("
            "SELECT DISTINCT trace_digest, record_id, definition_id, value, facts "
            "FROM evidence_records WHERE record_kind = 'annotation'"
        )
        if include_superseded:
            annotation_rows = annotation_records + (
                "), annotations AS ("
                "SELECT * FROM annotation_records) "
            )
        else:
            annotation_rows = annotation_records + (
                "), annotations AS ("
                "SELECT current.* FROM annotation_records current "
                "WHERE NOT EXISTS ("
                "SELECT 1 FROM annotation_records successor "
                "WHERE json_extract(successor.facts, '$.supersedes_id') "
                "= current.record_id)) "
            )
        if facet == "label":
            sql = (
                annotation_rows
                + "SELECT labels.value AS value, COUNT(*) AS count, "
                "AVG(a.value) AS mean_confidence "
                "FROM annotations a, "
                "json_each(json_extract(a.facts, '$.labels')) labels "
                f"{where} "
                "GROUP BY labels.value ORDER BY count DESC, value LIMIT ?"
            )
        else:
            expressions = {
                "annotator": "a.definition_id",
                "annotation_type": "json_extract(a.facts, '$.annotation_type')",
                "status": "json_extract(a.facts, '$.status')",
                "review_state": "json_extract(a.facts, '$.review_state')",
                "producer_kind": "json_extract(a.facts, '$.producer.kind')",
                "target_kind": "json_extract(a.facts, '$.target_kind')",
            }
            expression = expressions.get(facet)
            if expression is None:
                raise ValueError(f"unsupported annotation facet: {facet}")
            sql = (
                annotation_rows
                + f"SELECT {expression} AS value, COUNT(*) AS count, "
                "AVG(a.value) AS mean_confidence "
                f"FROM annotations a {where} "
                f"GROUP BY {expression} ORDER BY count DESC, value LIMIT ?"
            )
        params.append(int(limit))
        return [
            dict(row)
            for row in self._connection.execute(sql, params).fetchall()
        ]


__all__ = ["CATALOG_SCHEMA_VERSION", "SqliteCatalogStore"]
