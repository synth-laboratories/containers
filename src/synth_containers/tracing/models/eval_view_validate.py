"""Structural validation for ``eval_trace_view_v1/v2`` documents and ``eval_index_v1``.

Every ``eval_trace_view_v1`` document is a valid v2 document with the new blocks
absent, so v1 documents are checked against the v1 requirements only. Errors are
typed (``code`` + JSON ``path``); there are no bare-string errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from synth_containers.serde import JsonDataclassMixin

from .eval_view import (
    CHECK_STATUSES, COST_BASES, EVAL_INDEX_SCHEMA_VERSION, EVAL_SOURCE_SCHEMA,
    EVAL_VIEW_SCHEMA_VERSION, EVAL_VIEW_V1_SCHEMA_VERSION, EVALUATOR_KINDS,
    EVALUATOR_VERDICTS, FLAG_SEVERITIES, INTEGRITY_STATUSES, MEDIA_KINDS,
    MEDIA_PROVENANCE, MEDIA_STATUSES, RENDERERS, RUN_STATUSES, STEP_KINDS,
)


@dataclass(frozen=True, slots=True)
class EvalViewError(JsonDataclassMixin):
    code: str
    path: str
    message: str
    document: str = "eval_trace_view"

    def __str__(self) -> str:
        return f"{self.code} at {self.path}: {self.message}"


class _Ctx:
    def __init__(self, document: str) -> None:
        self.document = document
        self.errors: list[EvalViewError] = []

    def err(self, code: str, path: str, message: str) -> None:
        self.errors.append(EvalViewError(code, path, message, self.document))

    def obj(self, parent: Any, key: str, path: str, *, required: bool = True) -> dict | None:
        if not isinstance(parent, dict) or key not in parent:
            if required:
                self.err("missing_field", f"{path}.{key}", f"required object `{key}` is absent")
            return None
        value = parent[key]
        if not isinstance(value, dict):
            self.err("wrong_type", f"{path}.{key}", f"`{key}` must be an object")
            return None
        return value

    def need(self, parent: dict, key: str, path: str, types: tuple | type, *, nullable: bool = False) -> Any:
        if key not in parent:
            self.err("missing_field", f"{path}.{key}", f"required field `{key}` is absent")
            return None
        value = parent[key]
        if value is None:
            if not nullable:
                self.err("null_not_allowed", f"{path}.{key}", f"`{key}` must not be null")
            return None
        self.check(value, f"{path}.{key}", types)
        return value

    def opt(self, parent: dict, key: str, path: str, types: tuple | type) -> Any:
        if key in parent and parent[key] is not None:
            return self.check(parent[key], f"{path}.{key}", types)
        if key in parent:
            self.err("null_not_allowed", f"{path}.{key}", f"`{key}` is absent evidence and must be omitted, not null")
        return None

    def check(self, value: Any, path: str, types: tuple | type) -> Any:
        ts = types if isinstance(types, tuple) else (types,)
        ok = isinstance(value, ts) and not (isinstance(value, bool) and bool not in ts)
        if not ok:
            self.err("wrong_type", path, f"expected {'/'.join(t.__name__ for t in ts)}, got {type(value).__name__}")
            return None
        return value

    def enum(self, value: Any, path: str, allowed: tuple[str, ...], code: str = "bad_enum") -> None:
        if isinstance(value, str) and value not in allowed:
            self.err(code, path, f"{value!r} is not one of {list(allowed)}")


_NUM = (int, float)


def _strip_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_strip_nulls(v) for v in value]
    return value


def validate_eval_view(doc: Any) -> list[EvalViewError]:
    """Validate an ``eval_trace_view_v1`` or ``_v2`` document; empty list means valid."""

    ctx = _Ctx("eval_trace_view")
    if not isinstance(doc, dict):
        ctx.err("wrong_type", "$", "document must be an object")
        return ctx.errors
    schema = doc.get("schema")
    if schema not in (EVAL_VIEW_SCHEMA_VERSION, EVAL_VIEW_V1_SCHEMA_VERSION):
        ctx.err("bad_schema", "$.schema", f"expected {EVAL_VIEW_V1_SCHEMA_VERSION} or {EVAL_VIEW_SCHEMA_VERSION}, got {schema!r}")
        return ctx.errors
    v2 = schema == EVAL_VIEW_SCHEMA_VERSION
    if not v2:
        doc = _strip_nulls(doc)  # v1 producers emitted explicit nulls for absent fields
    ctx.need(doc, "source_schema", "$", str)
    if v2 and doc.get("source_schema") not in (None, EVAL_SOURCE_SCHEMA):
        ctx.err("bad_enum", "$.source_schema", f"v2 requires {EVAL_SOURCE_SCHEMA!r}")
    ctx.need(doc, "trace_id", "$", str)
    if v2:
        ctx.need(doc, "run_id", "$", str)
    else:
        ctx.opt(doc, "run_id", "$", str)

    task = ctx.obj(doc, "task", "$")
    if task is not None:
        for key in ("id", "name", "family"):
            ctx.need(task, key, "$.task", str)
        ctx.opt(task, "prompt", "$.task", str)

    run = ctx.obj(doc, "run", "$")
    if run is not None:
        ctx.need(run, "model", "$.run", str)
        status = ctx.need(run, "status", "$.run", str)
        if v2:
            ctx.enum(status, "$.run.status", RUN_STATUSES)
        for key in ("provider", "effort"):
            if key in run and run[key] is not None:
                ctx.check(run[key], f"$.run.{key}", str)
        for key in ("seed", "duration_ms"):
            if key in run:
                ctx.check(run[key], f"$.run.{key}", int)
        usage = ctx.obj(run, "usage", "$.run", required=False)
        if usage is not None:
            for key in ("calls", "input_tokens", "output_tokens", "cache_read_tokens"):
                ctx.opt(usage, key, "$.run.usage", int)
        timing = run.get("timing")
        if timing is not None and ctx.check(timing, "$.run.timing", list) is not None:
            for i, item in enumerate(timing):
                p = f"$.run.timing[{i}]"
                if ctx.check(item, p, dict) is not None:
                    ctx.need(item, "phase", p, str)
                    ctx.need(item, "duration_ms", p, _NUM)

    cost = ctx.obj(doc, "cost", "$", required=v2)
    if cost is not None:
        if "usd" not in cost:
            ctx.err("missing_field", "$.cost.usd", "`usd` must be present (number or null)")
        elif cost["usd"] is not None:
            ctx.check(cost["usd"], "$.cost.usd", _NUM)
        if "basis" not in cost or cost["basis"] is None:
            ctx.err("cost_missing_basis", "$.cost.basis", "a v2 cost block must declare its basis")
        elif ctx.check(cost["basis"], "$.cost.basis", str) is not None:
            ctx.enum(cost["basis"], "$.cost.basis", COST_BASES)
        ctx.need(cost, "source", "$.cost", str)
        pricing = ctx.obj(cost, "pricing", "$.cost", required=False)
        if pricing is not None:
            ctx.need(pricing, "model", "$.cost.pricing", str)
            ctx.need(pricing, "rates_per_token", "$.cost.pricing", dict)

    integrity = ctx.obj(doc, "integrity", "$")
    if integrity is not None:
        st = ctx.need(integrity, "status", "$.integrity", str)
        if v2:
            ctx.enum(st, "$.integrity.status", INTEGRITY_STATUSES)
        ctx.need(integrity, "content_digest", "$.integrity", str)
        ctx.need(integrity, "source", "$.integrity", str)
        ctx.opt(integrity, "capture_status", "$.integrity", str)

    ctx.opt(doc, "system_prompt", "$", str)
    media_ids = _media(ctx, doc)
    _steps(ctx, doc, v2, media_ids)
    _evaluator(ctx, doc)
    _environment(ctx, doc, media_ids)
    arts = doc.get("artifacts")
    if arts is not None and ctx.check(arts, "$.artifacts", list) is not None:
        for i, item in enumerate(arts):
            p = f"$.artifacts[{i}]"
            if ctx.check(item, p, dict) is not None:
                for key in ("id", "name", "media_type"):
                    ctx.need(item, key, p, str)
                if "media_id" in item and media_ids is not None and item["media_id"] not in media_ids:
                    ctx.err("unknown_media_ref", f"{p}.media_id", f"{item['media_id']!r} is not in media[]")
    if "reproduce" in doc:
        ctx.obj(doc, "reproduce", "$")
    if "ext" in doc:
        ctx.obj(doc, "ext", "$")
    return ctx.errors


def _media(ctx: _Ctx, doc: dict) -> set[str] | None:
    media = doc.get("media")
    if media is None:
        return None
    if ctx.check(media, "$.media", list) is None:
        return None
    ids: set[str] = set()
    for i, item in enumerate(media):
        p = f"$.media[{i}]"
        if ctx.check(item, p, dict) is None:
            continue
        mid = ctx.need(item, "id", p, str)
        if isinstance(mid, str):
            ids.add(mid)
        kind = ctx.need(item, "kind", p, str)
        ctx.enum(kind, f"{p}.kind", MEDIA_KINDS)
        ctx.need(item, "format", p, str)
        prov = item.get("provenance")
        if prov is None:
            ctx.err("media_missing_provenance", f"{p}.provenance", "every media item must declare provenance native|derived|none")
        elif prov not in MEDIA_PROVENANCE:
            ctx.err("media_bad_provenance", f"{p}.provenance", f"{prov!r} is not one of {list(MEDIA_PROVENANCE)}")
        for key, t in (("source", str), ("sha256", str), ("bytes", int)):
            ctx.opt(item, key, p, t)
        loc = ctx.obj(item, "location", p)
        if loc is not None:
            has_path, has_pack = "path" in loc, "pack" in loc
            if has_path == has_pack:
                ctx.err("bad_media_location", f"{p}.location", "location needs exactly one of `path` or `pack`")
            elif has_path:
                ctx.check(loc["path"], f"{p}.location.path", str)
            else:
                ctx.check(loc["pack"], f"{p}.location.pack", str)
                ctx.opt(loc, "key", f"{p}.location", str)
                ctx.opt(loc, "offset", f"{p}.location", int)
                ctx.opt(loc, "length", f"{p}.location", int)
                if ("offset" in loc) != ("length" in loc):
                    ctx.err("bad_media_location", f"{p}.location", "`offset` and `length` must be given together")
    return ids


def _steps(ctx: _Ctx, doc: dict, v2: bool, media_ids: set[str] | None) -> None:
    steps = ctx.need(doc, "steps", "$", list)
    if steps is None:
        return
    for i, step in enumerate(steps):
        p = f"$.steps[{i}]"
        if ctx.check(step, p, dict) is None:
            continue
        ctx.need(step, "id", p, str)
        ctx.need(step, "index", p, int)
        if v2:
            kind = ctx.need(step, "kind", p, str)
            ctx.enum(kind, f"{p}.kind", STEP_KINDS)
            ctx.need(step, "timestamp", p, str)
            ctx.need(step, "elapsed_ms", p, _NUM)
        else:
            if "kind" in step:
                ctx.enum(step["kind"], f"{p}.kind", STEP_KINDS)
        content = ctx.obj(step, "content", p)
        if content is not None:
            for key in ("reasoning", "message", "observation"):
                if key in content:
                    val = ctx.check(content[key], f"{p}.content.{key}", str)
                    if val == "":
                        ctx.err("empty_string", f"{p}.content.{key}", "absent evidence must be omitted, not an empty string")
        tokens = ctx.obj(step, "tokens", p, required=False)
        if tokens is not None:
            for key in ("input", "output", "cache_read"):
                ctx.opt(tokens, key, f"{p}.tokens", int)
        calls = step.get("tool_calls")
        if calls is not None and ctx.check(calls, f"{p}.tool_calls", list) is not None:
            for j, call in enumerate(calls):
                cp = f"{p}.tool_calls[{j}]"
                if ctx.check(call, cp, dict) is not None:
                    ctx.need(call, "id", cp, str)
                    ctx.need(call, "name", cp, str)
        action = ctx.obj(step, "action", p, required=False)
        if action is not None:
            _action(ctx, action, f"{p}.action", v2)
        for key in ("reward",):
            if key in step:
                ctx.check(step[key], f"{p}.{key}", _NUM)
        if "achievements" in step:
            ach = ctx.check(step["achievements"], f"{p}.achievements", list)
            for j, a in enumerate(ach or []):
                ctx.check(a, f"{p}.achievements[{j}]", str)
        refs = step.get("media_refs")
        if refs is not None and ctx.check(refs, f"{p}.media_refs", list) is not None:
            for j, ref in enumerate(refs):
                if media_ids is None or ref not in media_ids:
                    ctx.err("unknown_media_ref", f"{p}.media_refs[{j}]", f"{ref!r} is not in media[]")


def _action(ctx: _Ctx, action: dict, p: str, v2: bool) -> None:
    planned = action.get("planned")
    planned_names: set[str] | None = None
    if planned is not None and ctx.check(planned, f"{p}.planned", list) is not None:
        planned_names = set()
        for j, name in enumerate(planned):
            if ctx.check(name, f"{p}.planned[{j}]", str) is not None:
                planned_names.add(name)
    executed = action.get("executed")
    if executed is not None and ctx.check(executed, f"{p}.executed", list) is not None:
        for j, item in enumerate(executed):
            ep = f"{p}.executed[{j}]"
            if ctx.check(item, ep, dict) is None:
                continue
            name = ctx.need(item, "name", ep, str)
            if "turn" in item:
                ctx.check(item["turn"], f"{ep}.turn", int)
            if v2 and planned_names is not None and isinstance(name, str) and name not in planned_names:
                ctx.err("executed_not_planned", f"{ep}.name", f"executed action {name!r} is not in planned")
    rejected = action.get("rejected")
    if rejected is not None and ctx.check(rejected, f"{p}.rejected", list) is not None:
        for j, item in enumerate(rejected):
            rp = f"{p}.rejected[{j}]"
            if ctx.check(item, rp, dict) is not None:
                ctx.need(item, "name", rp, str)
                ctx.opt(item, "reason", rp, str)


def _evaluator(ctx: _Ctx, doc: dict) -> None:
    ev = ctx.obj(doc, "evaluator", "$", required=False)
    if ev is None:
        return
    p = "$.evaluator"
    ctx.enum(ctx.need(ev, "kind", p, str), f"{p}.kind", EVALUATOR_KINDS)
    ctx.enum(ctx.need(ev, "verdict", p, str), f"{p}.verdict", EVALUATOR_VERDICTS)
    for key in ("score", "reward"):
        ctx.opt(ev, key, p, _NUM)
    ctx.opt(ev, "output_ref", p, str)
    checks = ctx.need(ev, "checks", p, list)
    for i, c in enumerate(checks or []):
        cp = f"{p}.checks[{i}]"
        if ctx.check(c, cp, dict) is not None:
            ctx.need(c, "id", cp, str)
            ctx.need(c, "name", cp, str)
            ctx.enum(ctx.need(c, "status", cp, str), f"{cp}.status", CHECK_STATUSES)
            ctx.opt(c, "duration_ms", cp, _NUM)
    flags = ctx.need(ev, "flags", p, list)
    for i, f in enumerate(flags or []):
        fp = f"{p}.flags[{i}]"
        if ctx.check(f, fp, dict) is not None:
            ctx.need(f, "code", fp, str)
            ctx.enum(ctx.need(f, "severity", fp, str), f"{fp}.severity", FLAG_SEVERITIES)
            ctx.need(f, "message", fp, str)


def _environment(ctx: _Ctx, doc: dict, media_ids: set[str] | None) -> None:
    env = ctx.obj(doc, "environment", "$", required=False)
    if env is None:
        return
    p = "$.environment"
    ctx.enum(ctx.need(env, "renderer", p, str), f"{p}.renderer", RENDERERS)
    ctx.need(env, "version", p, int)
    frames = ctx.need(env, "frames", p, list)
    for i, fr in enumerate(frames or []):
        fp = f"{p}.frames[{i}]"
        if ctx.check(fr, fp, dict) is None:
            continue
        ctx.need(fr, "turn", fp, int)
        ctx.opt(fr, "step_id", fp, str)
        mid = fr.get("media_id")
        if mid is not None and (media_ids is None or mid not in media_ids):
            ctx.err("unknown_media_ref", f"{fp}.media_id", f"{mid!r} is not in media[]")


# --------------------------------------------------------------------------- index

_ROW_REQUIRED = (
    "id", "task", "model", "provider", "effort", "seed", "protocol", "status", "score",
    "reward", "success", "steps", "calls", "duration_ms", "tokens", "cost", "media_status", "flags",
)


def validate_eval_index(doc: Any) -> list[EvalViewError]:
    """Validate an ``eval_index_v1`` document; empty list means valid."""

    ctx = _Ctx("eval_index")
    if not isinstance(doc, dict):
        ctx.err("wrong_type", "$", "document must be an object")
        return ctx.errors
    if doc.get("schema") != EVAL_INDEX_SCHEMA_VERSION:
        ctx.err("bad_schema", "$.schema", f"expected {EVAL_INDEX_SCHEMA_VERSION}, got {doc.get('schema')!r}")
        return ctx.errors
    for key in ("benchmark_id", "environment", "published_at", "revision"):
        ctx.need(doc, key, "$", str)
    runs = ctx.need(doc, "runs", "$", list)
    seen: set[str] = set()
    for i, row in enumerate(runs or []):
        p = f"$.runs[{i}]"
        if ctx.check(row, p, dict) is None:
            continue
        for key in _ROW_REQUIRED:
            if key not in row:
                ctx.err("missing_field", f"{p}.{key}", f"required row field `{key}` is absent")
        rid = row.get("id")
        if isinstance(rid, str):
            if rid in seen:
                ctx.err("duplicate_id", f"{p}.id", f"run id {rid!r} appears twice")
            seen.add(rid)
        elif "id" in row:
            ctx.check(rid, f"{p}.id", str)
        task = row.get("task")
        if "task" in row and ctx.check(task, f"{p}.task", dict) is not None:
            for key in ("id", "name", "family"):
                ctx.need(task, key, f"{p}.task", str)
        for key in ("model", "protocol"):
            if key in row:
                ctx.check(row[key], f"{p}.{key}", str)
        for key in ("provider", "effort"):
            if row.get(key) is not None:
                ctx.check(row[key], f"{p}.{key}", str)
        if "status" in row:
            ctx.enum(row["status"], f"{p}.status", RUN_STATUSES)
        for key in ("score", "reward"):
            if row.get(key) is not None:
                ctx.check(row[key], f"{p}.{key}", _NUM)
        if row.get("success") is not None:
            ctx.check(row["success"], f"{p}.success", bool)
        for key in ("seed", "steps", "calls"):
            if key in row:
                ctx.check(row[key], f"{p}.{key}", int)
        if row.get("duration_ms") is not None:
            ctx.check(row["duration_ms"], f"{p}.duration_ms", _NUM)
        tokens = row.get("tokens")
        if "tokens" in row and ctx.check(tokens, f"{p}.tokens", dict) is not None:
            for key in ("input", "output", "cache_read"):
                if key not in tokens:
                    ctx.err("missing_field", f"{p}.tokens.{key}", "token counts must be present (null means unknown)")
                elif tokens[key] is not None:
                    ctx.check(tokens[key], f"{p}.tokens.{key}", int)
        cost = row.get("cost")
        if "cost" in row and ctx.check(cost, f"{p}.cost", dict) is not None:
            if "usd" not in cost:
                ctx.err("missing_field", f"{p}.cost.usd", "`usd` must be present (number or null)")
            elif cost["usd"] is not None:
                ctx.check(cost["usd"], f"{p}.cost.usd", _NUM)
            if cost.get("basis") is None:
                ctx.err("cost_missing_basis", f"{p}.cost.basis", "index cost must declare its basis")
            else:
                ctx.enum(cost["basis"], f"{p}.cost.basis", COST_BASES)
        if "media_status" in row:
            ctx.enum(row["media_status"], f"{p}.media_status", MEDIA_STATUSES)
        flags = row.get("flags")
        if "flags" in row and ctx.check(flags, f"{p}.flags", list) is not None:
            for j, f in enumerate(flags):
                ctx.check(f, f"{p}.flags[{j}]", str)
    return ctx.errors


__all__ = ["EvalViewError", "validate_eval_index", "validate_eval_view"]
