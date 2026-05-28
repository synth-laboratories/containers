from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from synth_containers.serde import JsonDataclassMixin, jsonable


RUBRIC_SCHEMA_VERSION = 1
RUBRIC_SCHEMA_VERSION_NAME = "synth_rubric_v1"
VERIFIER_RESULT_SCHEMA_VERSION_NAME = "synth_verifier_result_v1"


def _clamp_score(value: Any, *, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = float(default)
    return max(0.0, min(1.0, score))


def _tuple_of_strings(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in list(value or []) if str(item).strip())


@dataclass(slots=True, frozen=True)
class RubricScaleV1(JsonDataclassMixin):
    min_score: float = 0.0
    max_score: float = 1.0
    pass_threshold: float = 0.7
    labels: tuple[str, ...] = ("fail", "partial", "pass")
    schema_version: str = RUBRIC_SCHEMA_VERSION_NAME


@dataclass(slots=True, frozen=True)
class RubricCriterionV1(JsonDataclassMixin):
    criterion_id: str
    name: str
    description: str
    weight: float = 1.0
    guidance: tuple[str, ...] = ()
    evidence_expectations: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = RUBRIC_SCHEMA_VERSION_NAME


@dataclass(slots=True, frozen=True)
class RubricDefinitionV1(JsonDataclassMixin):
    rubric_id: str
    name: str
    task_family: str
    criteria: tuple[RubricCriterionV1, ...]
    scale: RubricScaleV1 = field(default_factory=RubricScaleV1)
    scoring_instructions: str = ""
    version: str = "v1"
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = RUBRIC_SCHEMA_VERSION_NAME
    rubric_schema_version: int = RUBRIC_SCHEMA_VERSION


@dataclass(slots=True, frozen=True)
class TraceEvidenceRefV1(JsonDataclassMixin):
    trace_id: str = ""
    rollout_id: str = ""
    span_id: str = ""
    call_index: int | None = None
    message_index: int | None = None
    message_role: str = ""
    part_index: int | None = None
    tool_call_id: str = ""
    tool_name: str = ""
    turn_index: int | None = None
    event_index: int | None = None
    path: str = ""
    quote: str = ""
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = RUBRIC_SCHEMA_VERSION_NAME


@dataclass(slots=True, frozen=True)
class CriterionVerdictV1(JsonDataclassMixin):
    criterion_id: str
    score: float
    passed: bool
    rationale: str = ""
    failure_modes: tuple[str, ...] = ()
    evidence_refs: tuple[TraceEvidenceRefV1, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = VERIFIER_RESULT_SCHEMA_VERSION_NAME


@dataclass(slots=True, frozen=True)
class VerifierResultV1(JsonDataclassMixin):
    rubric_id: str
    score: float
    passed: bool
    verdict: str
    rationale: str = ""
    failure_modes: tuple[str, ...] = ()
    criterion_verdicts: tuple[CriterionVerdictV1, ...] = ()
    evidence_refs: tuple[TraceEvidenceRefV1, ...] = ()
    verifier_id: str = ""
    source: str = "goex_agent"
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = VERIFIER_RESULT_SCHEMA_VERSION_NAME
    verifier_result_schema_version: int = RUBRIC_SCHEMA_VERSION


def openenv_react_base_v1() -> RubricDefinitionV1:
    return RubricDefinitionV1(
        rubric_id="openenv_react_base_v1",
        name="OpenEnv ReAct rollout quality",
        task_family="openenv_react",
        criteria=(
            RubricCriterionV1(
                criterion_id="state_grounding",
                name="State grounding",
                description="The agent grounds decisions in the current observation, inventory, task state, and recent outcomes.",
                weight=1.0,
                guidance=(
                    "Credit concrete use of visible state and recent tool results.",
                    "Penalize actions that ignore salient resources, hazards, prerequisites, or failed prior actions.",
                ),
                evidence_expectations=("Cite observation or tool-result spans that support the judgment.",),
            ),
            RubricCriterionV1(
                criterion_id="valid_react_control",
                name="Valid ReAct control",
                description="The agent emits valid actions/tool calls and adapts after each observation.",
                weight=1.0,
                guidance=(
                    "Credit parseable tool calls, action validity, and observation-conditioned next steps.",
                    "Penalize invalid parses, repeated failed calls, or action loops without new evidence.",
                ),
                evidence_expectations=("Cite assistant tool-call spans and any tool/result errors.",),
            ),
            RubricCriterionV1(
                criterion_id="progress_strategy",
                name="Progress strategy",
                description="The rollout pursues durable task progress rather than local dithering or premature stopping.",
                weight=1.25,
                guidance=(
                    "Credit resource acquisition, prerequisite completion, navigation toward reachable goals, and recovery from stalls.",
                    "Penalize plateauing when the trace shows actionable alternatives.",
                ),
                evidence_expectations=("Cite reward, achievement, action-tail, or no-progress evidence.",),
            ),
            RubricCriterionV1(
                criterion_id="evidence_bounded_verdict",
                name="Evidence-bounded verdict",
                description="The verifier judgment itself is supported only by real trace, rollout, and task evidence.",
                weight=0.75,
                guidance=(
                    "Credit judgments with trace evidence references.",
                    "Penalize unsupported claims, proxy-only reasoning, or invented environment facts.",
                ),
                evidence_expectations=("Every substantive failure mode should include a trace or rollout evidence ref.",),
            ),
        ),
        scale=RubricScaleV1(min_score=0.0, max_score=1.0, pass_threshold=0.7),
        scoring_instructions=(
            "Score each criterion from 0 to 1 using only real rollout evidence. "
            "Aggregate by criterion weight. Cite V4 trace spans/messages/tool calls when available."
        ),
        metadata={"standard": True, "intended_trace_schema": "synth_rollout_trace_v4"},
    )


def aggregate_criterion_score(
    rubric: RubricDefinitionV1,
    criterion_verdicts: tuple[CriterionVerdictV1, ...] | list[CriterionVerdictV1],
) -> float:
    verdict_by_id = {item.criterion_id: item for item in criterion_verdicts}
    numerator = 0.0
    denominator = 0.0
    for criterion in rubric.criteria:
        verdict = verdict_by_id.get(criterion.criterion_id)
        if verdict is None:
            continue
        weight = max(0.0, float(criterion.weight))
        numerator += _clamp_score(verdict.score) * weight
        denominator += weight
    if denominator <= 0.0:
        return 0.0
    return _clamp_score(numerator / denominator)


def trace_evidence_ref_from_mapping(payload: Mapping[str, Any]) -> TraceEvidenceRefV1:
    return TraceEvidenceRefV1(
        trace_id=str(payload.get("trace_id") or ""),
        rollout_id=str(payload.get("rollout_id") or ""),
        span_id=str(payload.get("span_id") or ""),
        call_index=(
            int(payload["call_index"])
            if payload.get("call_index") not in (None, "")
            else None
        ),
        message_index=(
            int(payload["message_index"])
            if payload.get("message_index") not in (None, "")
            else None
        ),
        message_role=str(payload.get("message_role") or ""),
        part_index=(
            int(payload["part_index"])
            if payload.get("part_index") not in (None, "")
            else None
        ),
        tool_call_id=str(payload.get("tool_call_id") or ""),
        tool_name=str(payload.get("tool_name") or ""),
        turn_index=(
            int(payload["turn_index"])
            if payload.get("turn_index") not in (None, "")
            else None
        ),
        event_index=(
            int(payload["event_index"])
            if payload.get("event_index") not in (None, "")
            else None
        ),
        path=str(payload.get("path") or ""),
        quote=str(payload.get("quote") or ""),
        summary=str(payload.get("summary") or ""),
        metadata=dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), Mapping) else {},
    )


def criterion_verdict_from_mapping(payload: Mapping[str, Any], *, pass_threshold: float) -> CriterionVerdictV1:
    score = _clamp_score(payload.get("score"))
    evidence_payloads = payload.get("evidence_refs")
    if not isinstance(evidence_payloads, list):
        evidence_payloads = []
    failure_modes = payload.get("failure_modes")
    return CriterionVerdictV1(
        criterion_id=str(payload.get("criterion_id") or payload.get("id") or ""),
        score=score,
        passed=bool(payload.get("passed")) if payload.get("passed") is not None else score >= float(pass_threshold),
        rationale=str(payload.get("rationale") or payload.get("reasoning") or ""),
        failure_modes=_tuple_of_strings(failure_modes),
        evidence_refs=tuple(
            trace_evidence_ref_from_mapping(item)
            for item in evidence_payloads
            if isinstance(item, Mapping)
        ),
        metadata=dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), Mapping) else {},
    )


def verifier_result_from_mapping(
    payload: Mapping[str, Any],
    *,
    rubric: RubricDefinitionV1,
    fallback_score: float | None = None,
    source: str = "goex_agent",
    model: str = "",
) -> VerifierResultV1:
    criterion_payloads = payload.get("criterion_verdicts")
    if criterion_payloads is None:
        criterion_payloads = payload.get("criteria")
    if not isinstance(criterion_payloads, list):
        criterion_payloads = []
    criterion_verdicts = tuple(
        item
        for item in (
            criterion_verdict_from_mapping(entry, pass_threshold=rubric.scale.pass_threshold)
            for entry in criterion_payloads
            if isinstance(entry, Mapping)
        )
        if item.criterion_id
    )
    score_value = payload.get("score")
    if score_value is None:
        score_value = payload.get("verifier_score")
    if score_value is None and criterion_verdicts:
        score_value = aggregate_criterion_score(rubric, criterion_verdicts)
    if score_value is None:
        score_value = fallback_score
    score = _clamp_score(score_value)
    evidence_payloads = payload.get("evidence_refs")
    if not isinstance(evidence_payloads, list):
        evidence_payloads = []
    failure_modes = payload.get("failure_modes")
    passed = bool(payload.get("passed")) if payload.get("passed") is not None else score >= float(rubric.scale.pass_threshold)
    return VerifierResultV1(
        verifier_id=str(payload.get("verifier_id") or payload.get("id") or ""),
        rubric_id=str(payload.get("rubric_id") or rubric.rubric_id),
        score=score,
        passed=passed,
        verdict=str(payload.get("verdict") or ("pass" if passed else "fail")),
        rationale=str(payload.get("rationale") or payload.get("reasoning") or ""),
        failure_modes=_tuple_of_strings(failure_modes),
        criterion_verdicts=criterion_verdicts,
        evidence_refs=tuple(
            trace_evidence_ref_from_mapping(item)
            for item in evidence_payloads
            if isinstance(item, Mapping)
        ),
        source=str(payload.get("source") or source),
        model=str(payload.get("model") or model),
        metadata=dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), Mapping) else {},
    )


__all__ = [
    "RUBRIC_SCHEMA_VERSION",
    "RUBRIC_SCHEMA_VERSION_NAME",
    "VERIFIER_RESULT_SCHEMA_VERSION_NAME",
    "CriterionVerdictV1",
    "RubricCriterionV1",
    "RubricDefinitionV1",
    "RubricScaleV1",
    "TraceEvidenceRefV1",
    "VerifierResultV1",
    "aggregate_criterion_score",
    "criterion_verdict_from_mapping",
    "jsonable",
    "openenv_react_base_v1",
    "trace_evidence_ref_from_mapping",
    "verifier_result_from_mapping",
]
