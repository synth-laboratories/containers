"""Job-scoped research queries over a disposable, versioned V5 projection cache.

The host supplies trusted archive paths and existing job/trial membership. The
agent query accepts typed facts only, never paths or SQL. Results are immutable
values which Workshop stores in its existing query_snapshots table.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .canonical import canonical_text, utc_now
from .projections.inspector import load_bundle
from .store.bundle import LocalTraceBundle
from .store.projection import catalog_projection

SCHEMA = "synth.trace-query.v2"
RESULT_SCHEMA = "synth.trace-query-result.v2"
MAX_RESULTS = 100_000
PAGE_SIZE = 200
CACHE_VERSION = 3
FIELDS = {
    "jobId", "trialId", "candidateId", "checkpointId", "stage", "taskId", "seed", "repeat",
    "environment", "environmentVersion", "definitionDigest", "rewardVersion", "rewardSemantics", "scenario", "promptRevision", "protocolRevision",
    "harnessRevision", "model", "effort", "status", "valid", "reward", "rewardId", "units",
    "captureStatus", "traceDigest", "actorId", "sessionId", "kind", "eventType", "action",
    "tool", "itemId", "annotationId", "annotatorId", "annotatorVersion", "label", "score",
    "confidence", "reviewState", "annotationState", "analysisState", "current", "evidenceDigest", "emission", "text",
}
EPISODE_FIELDS = FIELDS - {
    "actorId", "sessionId", "kind", "eventType", "action", "tool", "itemId", "annotationId",
    "annotatorId", "annotatorVersion", "label", "score", "confidence", "reviewState",
    "annotationState", "current", "evidenceDigest", "emission", "text",
}


@contextmanager
def retained_records(path: Path):
    """Read verified portable bundles or the original sealed standalone V5."""
    import zipfile
    if zipfile.is_zipfile(path):
        with tempfile.TemporaryDirectory(prefix="trace-query-") as staging:
            bundle = LocalTraceBundle.extract_archive(path, Path(staging) / "bundle")
            yield load_bundle(bundle.root)
    else:
        from .inspection import inspect_trace_input
        from .validation.rehydrate import rehydrate_trace
        from .projections.inspector import InspectedBundle
        inspection = inspect_trace_input(path)
        if not inspection.trusted or inspection.input_kind != "standalone_trace":
            raise ValueError("unverified standalone trace")
        yield [InspectedBundle(rehydrate_trace(json.loads(path.read_text())), None)]


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_text(value).encode()).hexdigest()


def validate_query(query: dict) -> dict:
    if not isinstance(query, dict):
        raise ValueError("query must be an object")
    unknown = set(query) - {"schemaVersion", "evalJobIds", "includeChildEvals", "grain", "where", "entityWhere", "annotationWhere", "rewardWhere", "relation", "groupBy", "aggregate", "limit"}
    if unknown:
        raise ValueError(f"unknown query fields: {sorted(unknown)}")
    if query.get("schemaVersion") != SCHEMA:
        raise ValueError(f"schemaVersion must be {SCHEMA}")
    ids = query.get("evalJobIds")
    if not isinstance(ids, list) or not ids or len(ids) > 100 or any(not isinstance(x, str) or not x.strip() for x in ids):
        raise ValueError("evalJobIds requires 1..100 existing job IDs")
    if "includeChildEvals" in query and not isinstance(query["includeChildEvals"], bool):
        raise ValueError("includeChildEvals must be a boolean")
    if query.get("grain", "episodes") not in {"episodes", "entities", "annotations", "rewards"}:
        raise ValueError("unsupported grain")
    for key in ("where", "entityWhere", "annotationWhere", "rewardWhere"):
        filters = query.get(key, [])
        if not isinstance(filters, list) or len(filters) > 32:
            raise ValueError(f"{key} must contain at most 32 predicates")
        for f in filters:
            if not isinstance(f, dict) or set(f) - {"field", "op", "value"}:
                raise ValueError("predicate accepts field, op, value only")
            if f.get("field") not in FIELDS or f.get("op", "eq") not in {"eq", "ne", "in", "gte", "lte", "contains", "missing"}:
                raise ValueError("unsupported predicate")
            if f.get("op") == "in" and (not isinstance(f.get("value"), list) or len(f["value"]) > 100):
                raise ValueError("in requires at most 100 values")
            if f.get("op") in {"gte", "lte"} and (isinstance(f.get("value"), bool) or not isinstance(f.get("value"), (int, float)) or not math.isfinite(f["value"])):
                raise ValueError("range predicates require finite numbers")
    own = {
        "episodes": set(),
        "entities": {"actorId","sessionId","kind","eventType","action","tool","itemId","text"},
        "annotations": {"annotationId","annotatorId","annotatorVersion","label","score","confidence","reviewState","annotationState","current","evidenceDigest","itemId"},
        "rewards": {"emission","current","evidenceDigest","itemId","actorId","sessionId"},
    }
    allowed = EPISODE_FIELDS | own[query.get("grain", "episodes")]
    if any(f["field"] not in allowed for f in query.get("where",[])):
        raise ValueError("where field is unavailable at this grain; use a typed exists filter")
    group = query.get("groupBy", [])
    if not isinstance(group, list) or len(group) > 8 or any(x not in allowed for x in group):
        raise ValueError("invalid groupBy")
    if query.get("aggregate") not in {None, "count", "reward", "paired_reward"}:
        raise ValueError("aggregate must be count or reward")
    if query.get("aggregate") in {"reward", "paired_reward"} and query.get("grain", "episodes") != "episodes":
        raise ValueError("reward aggregates require episode grain; joins must not multiply rewards")
    if query.get("aggregate") == "paired_reward" and len(set(ids)) != 2:
        raise ValueError("paired_reward requires exactly two evalJobIds")
    if query.get("aggregate") == "paired_reward" and group:
        raise ValueError("paired_reward returns aligned episode pairs; groupBy is unsupported")
    relation = query.get("relation")
    if relation is not None and relation not in {"repeated_failed_action", "recorded_link", "annotation_target"}:
        raise ValueError("unsupported relation; interpretation requires an annotation")
    limit = query.get("limit", PAGE_SIZE)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= PAGE_SIZE:
        raise ValueError(f"page limit must be 1..{PAGE_SIZE}")
    return query


def matches(row: dict, filters: list) -> bool:
    for f in filters:
        actual, expected, op = row.get(f["field"]), f.get("value"), f.get("op", "eq")
        if op == "missing":
            ok = actual is None
        elif actual is None:
            ok = False
        elif op == "eq":
            ok = actual == expected
        elif op == "ne":
            ok = actual != expected
        elif op == "in":
            ok = actual in expected
        elif op == "contains":
            ok = expected in actual if isinstance(actual, (str, list)) and isinstance(expected, str) else False
        elif op == "gte":
            ok = isinstance(actual, (float, int)) and not isinstance(actual, bool) and actual >= expected
        else:
            ok = isinstance(actual, (float, int)) and not isinstance(actual, bool) and actual <= expected
        if not ok:
            return False
    return True


class ResearchIndex:
    """Rebuildable per-host cache. No provider calls, writes to bundles, or DB authority."""
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS research_cache_v1 (
                cache_key TEXT PRIMARY KEY, archive_path TEXT, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS research_rows_v1 (
                cache_key TEXT NOT NULL, trace_digest TEXT NOT NULL, grain TEXT NOT NULL,
                item_id TEXT NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY(cache_key,trace_digest,grain,item_id));
            CREATE INDEX IF NOT EXISTS research_trace_v1 ON research_rows_v1(cache_key,trace_digest,grain);
        """)

    def close(self):
        self.db.close()

    def index_records(self, key: str, records: list, path: str | None = None) -> dict:
        traces = {}
        with self.db:
            self.db.execute("DELETE FROM research_rows_v1 WHERE cache_key=?", (key,))
            for inspected in records:
                t, e = inspected.trace, inspected.evidence
                td = t.content_digest
                head = e.content_digest if e else None
                traces[td] = {"traceId": t.trace_id, "traceDigest": td, "evidenceDigest": head, "captureStatus": str(t.completeness.capture_status)}
                projection = catalog_projection(t)
                links = {}
                for link in projection["relationships"]:
                    links.setdefault(link["source_entity_id"], []).append(link)
                for entity in projection["entities"]:
                    f = json.loads(entity["facts"])
                    payload = f.get("payload") or {}
                    action = payload.get("action") if isinstance(payload, dict) else None
                    if isinstance(action, dict):
                        action = action.get("name") or action.get("type")
                    row = {"itemId": entity["entity_id"], "kind": entity["kind"],
                           "actorId": entity["owner_actor_id"], "sessionId": entity["owner_session_id"],
                           "sourceOrder": entity["source_order"],
                           "recordedOrder": f.get("actor_sequence") if f.get("actor_sequence") is not None else f.get("chronological_sequence"), "occurredAt": entity["occurred_at"],
                           "eventType": f.get("event_type"), "status": f.get("status"),
                           "action": action, "tool": f.get("tool_name") or (payload.get("tool_name") if isinstance(payload, dict) else None),
                           "facts": f, "links": links.get(entity["entity_id"], []),
                           "text": t.message(entity["entity_id"]).text() if entity["kind"] == "message" else f.get("text"),
                           "selector": {"schema_version":"synth.trace-selector.v1","trace_id":t.trace_id,"trace_digest":td,"kind": entity["kind"], "entity_id": entity["entity_id"]}}
                    if entity["kind"] == "error":
                        row["selector"] = {"schema_version":"synth.trace-selector.v1","trace_id":t.trace_id,"trace_digest":td,"kind":"trace","json_pointer":f"/errors/{entity['source_order']}"}
                    self._put(key, td, "entities", row)
                if e:
                    definitions = {d.content_digest: d for d in e.reward_definitions}
                    superseded = {a.supersedes_id for a in e.annotations if a.supersedes_id}
                    for a in e.annotations:
                        row = {"itemId": a.annotation_id, "annotationId": a.annotation_id,
                               "annotatorId": a.annotator_id, "annotatorVersion": a.annotator_version,
                               "annotatorDigest": a.annotator_digest, "label": list(a.labels),
                               "score": a.payload.get("score"), "confidence": a.confidence,
                               "reviewState": str(a.review_state) if a.review_state else None,
                               "annotationState": str(a.status) if a.status else None,
                               "current": a.annotation_id not in superseded,
                               "evidenceDigest": head, "selector": a.target.to_dict(),
                               "evidence": [x.to_dict() for x in a.evidence], "annotation": a.to_dict()}
                        self._put(key, td, "annotations", row)
                    for r in e.reward_records:
                        d = definitions.get(r.reward_digest)
                        row = {"itemId": r.reward_record_id, "rewardId": r.reward_id,
                               "reward": r.value, "units": d.units if d else None,
                               "emission": str(d.emission) if d else None,
                               "rewardVersion": r.reward_version, "definitionDigest": r.reward_digest,
                               "valid": r.validity == "valid", "current": str(r.state) == "current",
                               "actorId": r.actor_id, "sessionId": r.session_id,
                               "evidenceDigest": head, "selector": r.subject.to_dict(), "record": r.to_dict(),
                               "definition": d.to_dict() if d else None}
                        self._put(key, td, "rewards", row)
            self.db.execute("INSERT OR REPLACE INTO research_cache_v1 VALUES (?,?,?)", (key, path, canonical_text(traces)))
        return traces

    def _put(self, key, td, grain, row):
        self.db.execute("INSERT INTO research_rows_v1 VALUES (?,?,?,?,?)", (key, td, grain, row["itemId"], canonical_text(row)))

    def archive(self, path: Path) -> tuple[str, dict]:
        # CAS archive identity is verified on every import. Stat is only the lookup
        # key for our disposable cache; a changed file invalidates the projection.
        # Bind cached projections to archive bytes, not only mutable mtime/size.
        with path.open("rb") as stream:
            archive_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        key = digest([str(path.resolve()), archive_digest, CACHE_VERSION])
        existing = self.db.execute("SELECT payload FROM research_cache_v1 WHERE cache_key=?", (key,)).fetchone()
        if existing:
            return key, json.loads(existing[0])
        with retained_records(path) as records:
            return key, self.index_records(key, records, str(path))

    def rows(self, key: str, td: str, grain: str) -> list[dict]:
        return [json.loads(r[0]) for r in self.db.execute(
            "SELECT payload FROM research_rows_v1 WHERE cache_key=? AND trace_digest=? AND grain=? ORDER BY item_id",
            (key, td, grain))]

    def execute(self, query: dict, episodes: list[dict], annotation_rows: list[dict] | None = None) -> dict:
        query = copy.deepcopy(validate_query(query))
        results, coverage, inputs = [], [], []
        seen = set()
        for source in episodes:
            if source.get("jobId") not in query["evalJobIds"] and not source.get("includedChild"):
                continue
            identity = (source["jobId"], source["trialId"])
            if identity in seen:
                raise ValueError(f"ambiguous duplicate trial: {identity}")
            seen.add(identity)
            row = {k: v for k, v in source.items() if k not in {"archivePath", "cacheKey", "includedChild"}}
            row["episodeId"] = digest(identity)
            row["rewardSemantics"] = row.get("definitionDigest") or "unknown:" + row["jobId"]
            td = row.get("traceDigest")
            key, traces = None, {}
            if source.get("cacheKey"):
                key = source["cacheKey"]
                cached = self.db.execute("SELECT payload FROM research_cache_v1 WHERE cache_key=?", (key,)).fetchone()
                traces = json.loads(cached[0]) if cached else {}
            elif source.get("archivePath"):
                try:
                    key, traces = self.archive(Path(source["archivePath"]))
                except (OSError, ValueError):
                    row["traceUnavailableReason"] = "archive_missing_or_invalid"
            if td and td not in traces:
                row["traceAvailability"] = "unavailable"
            elif td:
                row.update(traces[td])
                row["traceAvailability"] = "available"
            else:
                row["traceAvailability"] = "not_produced"
            coverage.append({"jobId": row["jobId"], "trialId": row["trialId"], "status": row.get("status"), "traceAvailability": row["traceAvailability"], "captureStatus": row.get("captureStatus")})
            input_row=row.copy()
            inputs.append(input_row)
            entities = self.rows(key, td, "entities") if key and td in traces else []
            annotations = self.rows(key, td, "annotations") if key and td in traces else []
            rewards = self.rows(key, td, "rewards") if key and td in traces else []
            # A scalar episode score may carry its definition in sealed V5
            # evidence instead of the eval-job row. Adopt only a unique current,
            # valid trace-wide record that exactly agrees with the stored score.
            # Never infer semantics from an event reward or change that score.
            bind_episode_reward_definition(row, rewards)
            input_row.update({field: row[field] for field in
                ("rewardId", "rewardVersion", "definitionDigest", "units", "rewardSemantics")
                if field in row})
            # Host-projected annotation revisions are explicit query inputs. Never
            # silently combine two revisions of the same annotation.
            extra = [a for a in annotation_rows or [] if a.get("traceDigest") == td]
            annotations_by_id = {a["annotationId"]: a for a in annotations}
            annotations_by_id.update({a["annotationId"]: a for a in extra})
            annotations = list(annotations_by_id.values())
            row["analysisState"] = source.get("analysisState") or "not_requested"
            if annotations and row["analysisState"] == "not_requested":
                row["analysisState"] = "findings_available"
            input_row["analysisState"]=row["analysisState"]
            if td in traces:
                row["selector"] = {"schema_version":"synth.trace-selector.v1", "trace_id":row["traceId"], "trace_digest":td, "kind":"trace"}
            if query.get("annotationWhere"):
                annotations = [a for a in annotations if matches({**row, **a}, query["annotationWhere"])]
                if not annotations:
                    continue
            if query.get("entityWhere"):
                entities = [e for e in entities if matches({**row, **e}, query["entityWhere"])]
                if not entities:
                    continue
            if query.get("rewardWhere"):
                rewards = [r for r in rewards if matches({**row, **r}, query["rewardWhere"])]
                if not rewards:
                    continue
            relation = query.get("relation")
            if relation == "annotation_target":
                targets = {a.get("selector", {}).get("entity_id") for a in annotations}
                entities = [e for e in entities if e["itemId"] in targets]
                if not entities:
                    continue
                entity_ids = {e["itemId"] for e in entities}
                annotations = [a for a in annotations if a.get("selector", {}).get("entity_id") in entity_ids]
            elif relation:
                entities = related_entities(entities, self.rows(key, td, "entities") if key else [], relation)
                if not entities:
                    continue
            grain = query.get("grain", "episodes")
            selected = [row] if grain == "episodes" else [{**row, **item} for item in {"entities": entities, "annotations": annotations, "rewards": rewards}[grain]]
            results.extend(compact_result(r) for r in selected if matches(r, query.get("where", [])))
            if len(results) > MAX_RESULTS:
                raise ValueError("query exceeds 100000 results; narrow job IDs or filters (no partial aggregate returned)")
        results.sort(key=lambda r: (r["jobId"], r["trialId"], r.get("itemId", "")))
        if query.get("aggregate") == "paired_reward":
            results = paired_rewards(results, query["evalJobIds"])
        elif query.get("aggregate"):
            results = aggregate(results, query.get("groupBy", []), query["aggregate"])
        provenance = {"episodes": inputs, "coverage": coverage, "annotations": copy.deepcopy(annotation_rows or [])}
        identity_query = {k: v for k, v in query.items() if k != "limit"}
        rd = digest([SCHEMA, identity_query, provenance, results])
        return {"schemaVersion": RESULT_SCHEMA, "snapshotId": "trace_query_" + rd[7:39],
                "domain": "traces", "querySchemaVersion": SCHEMA, "queryAst": query,
                "resultIds": [digest(r) for r in results], "resultCount": len(results),
                "facets": {"rows": results, "provenance": provenance, **({"relationshipEvidence": {"basis": "recorded_only", "absenceMeaning": "unknown", "interpretation": "No recorded relationship does not prove a message was ignored or an action was not repeated."}} if query.get("relation") else {})}, "resultDigest": rd,
                "queriedAt": utc_now(), "truncated": False}


def related_entities(selected, all_entities, relation):
    if relation == "recorded_link":
        return [r for r in selected if r.get("links")]
    ordered = sorted([r for r in all_entities if r.get("kind") == "event" and r.get("action") is not None],
                     key=lambda r: (r.get("actorId") or "", r.get("sessionId") or "", r.get("recordedOrder") if r.get("recordedOrder") is not None else -1, r["itemId"]))
    following = {}
    for a, b in zip(ordered, ordered[1:]):
        if not a.get("actorId") or not a.get("sessionId"):
            continue
        if (a.get("actorId"), a.get("sessionId")) != (b.get("actorId"), b.get("sessionId")):
            continue
        if a.get("status") == "error" and a.get("action") == b.get("action") and a.get("recordedOrder") is not None and b.get("recordedOrder") is not None and b["recordedOrder"] > a["recordedOrder"]:
            following[a["itemId"]] = b["selector"]
    return [{**r, "relatedSelector": following[r["itemId"]]} for r in selected if r["itemId"] in following]


def aggregate(rows, fields, operation):
    groups = {}
    # Reward definition/units are always grouping keys for reward means. Unknown
    # semantics remain separated by environment/version instead of merged away.
    keys = list(dict.fromkeys(fields + (["environment", "environmentVersion", "rewardId", "rewardVersion", "definitionDigest", "rewardSemantics", "units"] if operation == "reward" else [])))
    for row in rows:
        values = {k: row.get(k) for k in keys}
        group = groups.setdefault(canonical_text(values), {**values, "count": 0, "episodeIds": set(), "values": []})
        group["count"] += 1
        eid = row["episodeId"]
        if eid not in group["episodeIds"]:
            reward = row.get("reward") if row.get("valid") is not False else None
            if isinstance(reward, (int, float)) and not isinstance(reward, bool) and math.isfinite(reward):
                group["values"].append(reward)
            group["episodeIds"].add(eid)
    output = []
    for key in sorted(groups):
        g = groups[key]
        values, ids = g.pop("values"), sorted(g.pop("episodeIds"))
        g.update(episodeCount=len(ids), episodeIds=ids)
        if operation == "reward":
            g.update(measuredCount=len(values), missingCount=len(ids)-len(values), rewardMean=sum(values)/len(values) if values else None, rewardMin=min(values) if values else None, rewardMax=max(values) if values else None)
        output.append(g)
    return output


def run_request(request: dict, cache_path: Path) -> dict:
    if request.get("operation") == "source":
        return read_source(request)
    if set(request) - {"query", "episodes", "annotations"}:
        raise ValueError("unexpected research request fields")
    index = ResearchIndex(cache_path)
    try:
        return index.execute(request["query"], request["episodes"], request.get("annotations"))
    finally:
        index.close()


def compact_result(row):
    # Queries return facts and exact citations, never an entire long trace body.
    out = dict(row)
    for key in ("facts", "annotation", "record", "payload", "definition", "text"):
        if key in out:
            value = out.pop(key)
            encoded = value if isinstance(value, str) else canonical_text(value)
            out[key + "Preview"] = encoded[:2000]
            if len(encoded) > 2000:
                out["previewTruncated"] = True
    return out


def bind_episode_reward_definition(row, rewards):
    if row.get("definitionDigest") or not isinstance(row.get("reward"), (int, float)) or isinstance(row.get("reward"), bool):
        return
    fields = ("rewardId", "rewardVersion", "definitionDigest", "units")
    candidates = {
        canonical_text({field: reward.get(field) for field in fields})
        for reward in rewards
        if reward.get("current") and reward.get("valid") and reward.get("definitionDigest")
        and reward.get("selector", {}).get("kind") == "trace"
        and isinstance(reward.get("reward"), (int, float)) and not isinstance(reward.get("reward"), bool)
        and reward["reward"] == row["reward"]
    }
    if len(candidates) == 1:
        row.update(json.loads(next(iter(candidates))))
        row["rewardSemantics"] = row["definitionDigest"]


def paired_rewards(rows, jobs):
    keys = ["environment", "environmentVersion", "taskId", "seed", "repeat", "rewardId", "rewardVersion", "definitionDigest", "units"]
    pairs = {}
    unmatched = []
    for row in rows:
        identity = {k:row.get(k) for k in keys}
        # Some environments have a fixed world and explicit repeat identities,
        # not randomized seeds. Require the task and at least one declared key.
        if not row.get("definitionDigest") or not row.get("environment"):
            unmatched.append({**identity,"matchStatus":"unknown_reward_semantics","episodeIds":[row["episodeId"]]})
            continue
        if not row.get("environmentVersion"):
            unmatched.append({**identity,"matchStatus":"unknown_environment_version","episodeIds":[row["episodeId"]]})
            continue
        if row.get("taskId") is None or (row.get("seed") is None and row.get("repeat") is None):
            unmatched.append({**identity, "matchStatus":"missing_match_key", "episodeIds":[row["episodeId"]]})
            continue
        pair = pairs.setdefault(canonical_text(identity), {**identity, "arms":{}})
        pair["arms"].setdefault(row["jobId"], []).append(row)
    result = unmatched
    for key in sorted(pairs):
        pair = pairs[key]
        arms = pair.pop("arms")
        a, b = arms.get(jobs[0], []), arms.get(jobs[1], [])
        state = "matched" if len(a) == len(b) == 1 else ("ambiguous" if len(a)>1 or len(b)>1 else "unmatched")
        values = [(v[0].get("reward") if len(v)==1 and v[0].get("valid") is not False else None) for v in (a,b)]
        numeric = all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in values)
        result.append({**pair,"matchStatus":state,"jobIds":jobs,"episodeIds":[r["episodeId"] for v in (a,b) for r in v],
                       "rewards":values,"rewardDelta":values[1]-values[0] if state=="matched" and numeric else None})
    return result


def read_source(request):
    """Host-resolved immutable archive and selector; callers cannot supply paths."""
    from .models.selectors import TraceSelectorV1, TextRangeV1, resolve_selector
    selector = dict(request["selector"])
    if isinstance(selector.get("range"), dict):
        selector["range"] = TextRangeV1(**selector["range"])
    selector = TraceSelectorV1(**selector)
    offset, limit = request.get("offset",0), request.get("limit",16000)
    if isinstance(offset,bool) or not isinstance(offset,int) or offset<0 or isinstance(limit,bool) or not isinstance(limit,int) or not 1<=limit<=64000:
        raise ValueError("source offset/limit invalid")
    with retained_records(Path(request["archivePath"])) as records:
        document = next((r.trace for r in records if r.trace.content_digest==selector.trace_digest), None)
        if document is None:
            return {"resolved":False,"reason":"trace_digest_not_in_bundle"}
        resolution = resolve_selector(document, selector).to_dict()
        if resolution["resolved"] and resolution.get("resolved_text") is None:
            from dataclasses import replace
            resolution["resolved_text"] = resolve_selector(document, replace(selector,json_pointer="")).resolved_text
            resolution["representation"] = "canonical_json"
        text = resolution.get("resolved_text") or ""
        resolution["resolved_text"] = text[offset:offset+limit]
        resolution["textLength"] = len(text)
        resolution["offset"] = offset
        resolution["nextOffset"] = offset+limit if offset+limit<len(text) else None
        resolution["textDigest"] = digest(text)
        return resolution
