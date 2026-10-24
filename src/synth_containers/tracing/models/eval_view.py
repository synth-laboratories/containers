"""Typed contract for the eval trace viewer: ``eval_trace_view_v2`` and ``eval_index_v1``.

Rule carried from the Workshop projector: nothing is inferred from another field.
A field with no evidence is absent from the serialized document (``None`` is dropped),
never zero and never an empty string. The only nullable fields are those the contract
names as nullable (``cost.usd``, index ``provider``/``effort`` and unknown token counts).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

from synth_containers.serde import JsonDataclassMixin, jsonable


EVAL_VIEW_SCHEMA_VERSION = "eval_trace_view_v2"
EVAL_VIEW_V1_SCHEMA_VERSION = "eval_trace_view_v1"
EVAL_INDEX_SCHEMA_VERSION = "eval_index_v1"
EVAL_SOURCE_SCHEMA = "synth.trace.v5"

RUN_STATUSES = ("completed", "timed_out", "failed", "interrupted")
STEP_KINDS = ("system", "user", "policy_call", "tool_result", "environment", "evaluator", "note")
INTEGRITY_STATUSES = ("sealed", "provisional")
EVALUATOR_KINDS = ("verifier", "judge", "reward", "achievements")
EVALUATOR_VERDICTS = ("pass", "fail", "score", "unknown")
CHECK_STATUSES = ("passed", "failed", "skipped", "error")
FLAG_SEVERITIES = ("info", "warn", "error")
FLAG_CODES = (
    "frames_missing", "frames_derived", "frames_partial", "cost_estimated", "cost_repriced",
    "capture_interrupted", "timed_out", "tokens_unknown", "legacy_view_v1",
)
COST_BASES = ("measured", "estimated", "repriced", "unknown")
RENDERERS = ("craftax", "dungeongrid", "terminal", "browser", "none")
MEDIA_KINDS = ("image", "video", "cast", "file")
MEDIA_PROVENANCE = ("native", "derived", "none")
MEDIA_STATUSES = ("native", "partial", "none")

# Fields whose explicit ``null`` is meaningful in the contract and therefore kept.
_NULLABLE = frozenset({"usd"})


def _clean(value: Any, *, keep_null: frozenset[str] = _NULLABLE) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        nullable = getattr(value, "_nullable", frozenset()) | keep_null
        out: dict[str, Any] = {}
        for item in fields(value):
            raw = getattr(value, item.name)
            if raw is None:
                if item.name in nullable:
                    out[item.name] = None
                continue
            out[item.name] = _clean(raw, keep_null=keep_null)
        return out
    if isinstance(value, dict):
        return {str(k): _clean(v, keep_null=keep_null) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v, keep_null=keep_null) for v in value]
    return jsonable(value)


class _View(JsonDataclassMixin):
    """Serializes without ``None`` values (absent evidence stays absent)."""

    def to_dict(self) -> dict[str, Any]:
        return _clean(self)


# --------------------------------------------------------------------------- view


@dataclass(frozen=True, slots=True)
class EvalTaskV2(_View):
    id: str
    name: str
    family: str
    prompt: str | None = None


@dataclass(frozen=True, slots=True)
class EvalUsageV2(_View):
    calls: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class EvalTimingV2(_View):
    phase: str
    duration_ms: int


@dataclass(frozen=True, slots=True)
class EvalRunV2(_View):
    model: str
    status: str
    effort: str | None = None
    provider: str | None = None
    seed: int | None = None
    started_at: str | None = None
    duration_ms: int | None = None
    usage: EvalUsageV2 | None = None
    timing: tuple[EvalTimingV2, ...] | None = None


@dataclass(frozen=True, slots=True)
class EvalPricingV2(_View):
    model: str
    rates_per_token: dict[str, float]
    url: str | None = None
    as_of: str | None = None


@dataclass(frozen=True, slots=True)
class EvalCostV2(_View):
    """``usd`` may be an explicit null; ``basis`` is mandatory in v2 documents."""

    usd: float | None
    basis: str
    source: str
    pricing: EvalPricingV2 | None = None


@dataclass(frozen=True, slots=True)
class EvalIntegrityV2(_View):
    status: str
    content_digest: str
    source: str
    capture_status: str | None = None


@dataclass(frozen=True, slots=True)
class EvalStepTokensV2(_View):
    input: int | None = None
    output: int | None = None
    cache_read: int | None = None


@dataclass(frozen=True, slots=True)
class EvalStepContentV2(_View):
    reasoning: str | None = None
    message: str | None = None
    observation: str | None = None


@dataclass(frozen=True, slots=True)
class EvalToolCallV2(_View):
    id: str
    name: str
    arguments: dict[str, Any] | None = None
    result_ref: str | None = None
    result: str | None = None


@dataclass(frozen=True, slots=True)
class EvalExecutedActionV2(_View):
    name: str
    turn: int | None = None


@dataclass(frozen=True, slots=True)
class EvalRejectedActionV2(_View):
    name: str
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class EvalActionV2(_View):
    """``executed`` is authoritative; ``planned`` and ``rejected`` are kept separate."""

    planned: tuple[str, ...] | None = None
    executed: tuple[EvalExecutedActionV2, ...] | None = None
    rejected: tuple[EvalRejectedActionV2, ...] | None = None


@dataclass(frozen=True, slots=True)
class EvalStepV2(_View):
    id: str
    index: int
    kind: str
    content: EvalStepContentV2
    timestamp: str | None = None
    elapsed_ms: int | None = None
    actor: str | None = None
    turn_start: int | None = None
    turn_end: int | None = None
    tokens: EvalStepTokensV2 | None = None
    tool_calls: tuple[EvalToolCallV2, ...] | None = None
    action: EvalActionV2 | None = None
    reward: float | None = None
    achievements: tuple[str, ...] | None = None
    state_delta: dict[str, Any] | None = None
    media_refs: tuple[str, ...] | None = None
    raw: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class EvalCheckV2(_View):
    id: str
    name: str
    status: str
    duration_ms: int | None = None
    detail_ref: str | None = None


@dataclass(frozen=True, slots=True)
class EvalFlagV2(_View):
    code: str
    severity: str
    message: str


@dataclass(frozen=True, slots=True)
class EvalEvaluatorV2(_View):
    kind: str
    verdict: str
    checks: tuple[EvalCheckV2, ...] = ()
    flags: tuple[EvalFlagV2, ...] = ()
    score: float | None = None
    reward: float | None = None
    output_ref: str | None = None


@dataclass(frozen=True, slots=True)
class EvalFrameV2(_View):
    turn: int
    step_id: str | None = None
    media_id: str | None = None
    state: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class EvalEnvironmentV2(_View):
    renderer: str
    version: int = 1
    frames: tuple[EvalFrameV2, ...] = ()
    summary: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class EvalMediaLocationV2(_View):
    """Either ``path``, or a ``pack`` reference (``key`` and/or ``offset``+``length``)."""

    path: str | None = None
    pack: str | None = None
    key: str | None = None
    offset: int | None = None
    length: int | None = None


@dataclass(frozen=True, slots=True)
class EvalMediaV2(_View):
    id: str
    kind: str
    format: str
    provenance: str
    location: EvalMediaLocationV2
    source: str | None = None
    sha256: str | None = None
    bytes: int | None = None


@dataclass(frozen=True, slots=True)
class EvalArtifactV2(_View):
    id: str
    name: str
    media_type: str
    bytes: int | None = None
    media_id: str | None = None


@dataclass(frozen=True, slots=True)
class EvalReproduceV2(_View):
    command: str | None = None
    config_digest: str | None = None
    container_image_digest: str | None = None


@dataclass(frozen=True, slots=True)
class EvalTraceViewV2(_View):
    trace_id: str
    run_id: str
    task: EvalTaskV2
    run: EvalRunV2
    cost: EvalCostV2
    integrity: EvalIntegrityV2
    steps: tuple[EvalStepV2, ...]
    system_prompt: str | None = None
    evaluator: EvalEvaluatorV2 | None = None
    environment: EvalEnvironmentV2 | None = None
    media: tuple[EvalMediaV2, ...] | None = None
    artifacts: tuple[EvalArtifactV2, ...] | None = None
    reproduce: EvalReproduceV2 | None = None
    ext: dict[str, Any] | None = None
    schema: str = EVAL_VIEW_SCHEMA_VERSION
    source_schema: str = EVAL_SOURCE_SCHEMA


# --------------------------------------------------------------------------- index


@dataclass(frozen=True, slots=True)
class EvalIndexTaskV1(_View):
    id: str
    name: str
    family: str


@dataclass(frozen=True, slots=True)
class EvalIndexTokensV1(_View):
    """Zeros are allowed; ``None`` means unknown and is serialized as null."""

    _nullable = frozenset({"input", "output", "cache_read"})
    input: int | None = None
    output: int | None = None
    cache_read: int | None = None


@dataclass(frozen=True, slots=True)
class EvalIndexCostV1(_View):
    usd: float | None
    basis: str


@dataclass(frozen=True, slots=True)
class EvalIndexRowV1(_View):
    _nullable = frozenset({"provider", "effort", "score", "reward", "success", "duration_ms"})
    id: str
    task: EvalIndexTaskV1
    model: str
    provider: str | None
    effort: str | None
    seed: int
    protocol: str
    status: str
    score: float | None
    reward: float | None
    success: bool | None
    steps: int
    calls: int
    duration_ms: int | None
    tokens: EvalIndexTokensV1
    cost: EvalIndexCostV1
    media_status: str
    flags: tuple[str, ...] = ()
    ext: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class EvalIndexV1(_View):
    benchmark_id: str
    environment: str
    published_at: str
    revision: str
    runs: tuple[EvalIndexRowV1, ...]
    cohorts: tuple[dict[str, Any], ...] | None = None
    schema: str = EVAL_INDEX_SCHEMA_VERSION


__all__ = [
    "CHECK_STATUSES",
    "COST_BASES",
    "EVALUATOR_KINDS",
    "EVALUATOR_VERDICTS",
    "EVAL_INDEX_SCHEMA_VERSION",
    "EVAL_SOURCE_SCHEMA",
    "EVAL_VIEW_SCHEMA_VERSION",
    "EVAL_VIEW_V1_SCHEMA_VERSION",
    "EvalActionV2",
    "EvalArtifactV2",
    "EvalCheckV2",
    "EvalCostV2",
    "EvalEnvironmentV2",
    "EvalEvaluatorV2",
    "EvalExecutedActionV2",
    "EvalFlagV2",
    "EvalFrameV2",
    "EvalIndexCostV1",
    "EvalIndexRowV1",
    "EvalIndexTaskV1",
    "EvalIndexTokensV1",
    "EvalIndexV1",
    "EvalIntegrityV2",
    "EvalMediaLocationV2",
    "EvalMediaV2",
    "EvalPricingV2",
    "EvalRejectedActionV2",
    "EvalReproduceV2",
    "EvalRunV2",
    "EvalStepContentV2",
    "EvalStepTokensV2",
    "EvalStepV2",
    "EvalTaskV2",
    "EvalTimingV2",
    "EvalToolCallV2",
    "EvalTraceViewV2",
    "EvalUsageV2",
    "FLAG_CODES",
    "FLAG_SEVERITIES",
    "INTEGRITY_STATUSES",
    "MEDIA_KINDS",
    "MEDIA_PROVENANCE",
    "MEDIA_STATUSES",
    "RENDERERS",
    "RUN_STATUSES",
    "STEP_KINDS"
]
