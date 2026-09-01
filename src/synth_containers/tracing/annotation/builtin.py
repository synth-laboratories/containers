"""Domain-neutral deterministic annotators every Trace V5 can run.

These extract facts any trace carries: whether tool calls were well-formed and
answered, and how each environment step ended. Domain semantics (Craftax
beliefs, milestones, rubrics) live in ``evals``; nothing here knows about them.
"""

from __future__ import annotations

import json
from typing import Any

from ..models.document import TraceDocumentV5
from ..models.messages import PartType
from ..models.selectors import SelectorKind
from ..models.spans import SpanKind, SpanStatus
from ..models.standards import (
    AnnotationOutputContractV1,
    AnnotationPayloadFieldV1,
    AnnotationPayloadSchemaV1,
    AnnotationTaskKind,
    AnnotationTaxonV1,
    AnnotatorGroundingRequirement,
    ConfidenceSemantics,
    ProducerKind,
    TraceAnnotatorDefinitionV1,
    UnavailableEvidenceBehavior,
)
from .definitions import AnnotatorProgramV1, DefinitionRegistry, ProgramContext, RunnerKind
from .proposal import empty_proposal


TOOL_CALL_INTEGRITY_ID = "synth.deterministic.tool_call_integrity"
ENVIRONMENT_STEP_STATUS_ID = "synth.deterministic.environment_step_status"


def tool_call_integrity_definition() -> TraceAnnotatorDefinitionV1:
    return TraceAnnotatorDefinitionV1(
        annotator_id=TOOL_CALL_INTEGRITY_ID,
        name="Tool call integrity",
        purpose="Label each assistant tool call as well-formed and answered, or not.",
        taxonomy=(
            "tool_call.valid_json",
            "tool_call.malformed_arguments",
            "tool_call.answered",
            "tool_call.unanswered",
            "tool_call.result_error",
        ),
        version="v1",
        required_subject_scope="part",
        grounding_requirement=AnnotatorGroundingRequirement.EXACT_SELECTOR,
        minimum_evidence=1,
        unavailable_evidence_behavior=UnavailableEvidenceBehavior.ABSTAIN,
        confidence_semantics=ConfidenceSemantics.DETERMINISTIC,
        output_contract=AnnotationOutputContractV1(
            task_kind=AnnotationTaskKind.CLASSIFY,
            annotation_types=("tool_call",),
            taxonomy=(
                AnnotationTaxonV1(label="tool_call.valid_json", description="arguments parse as JSON"),
                AnnotationTaxonV1(label="tool_call.malformed_arguments", description="arguments do not parse"),
                AnnotationTaxonV1(label="tool_call.answered", description="a tool result part exists"),
                AnnotationTaxonV1(label="tool_call.unanswered", description="no tool result part exists"),
                AnnotationTaxonV1(label="tool_call.result_error", description="the tool result is an error"),
            ),
            payload_schema=AnnotationPayloadSchemaV1(
                schema_id="synth.tool_call_integrity.payload",
                version="v1",
                fields=(
                    AnnotationPayloadFieldV1(field_name="tool_name", value_kind="string", required=False),
                    AnnotationPayloadFieldV1(field_name="tool_call_id", value_kind="string", required=True),
                ),
            ),
            allowed_producer_kinds=(ProducerKind.DETERMINISTIC, ProducerKind.COMPOSITE),
        ),
    ).sealed()


def tool_call_integrity_program() -> AnnotatorProgramV1:
    return AnnotatorProgramV1(
        program_id=f"{TOOL_CALL_INTEGRITY_ID}.program",
        runner_kind=RunnerKind.DETERMINISTIC,
        version="v1",
        program_ref="synth_containers.tracing.annotation.builtin:tool_call_integrity",
    ).sealed()


def tool_call_integrity(document: TraceDocumentV5, context: ProgramContext) -> dict[str, Any]:
    proposal = empty_proposal(trace_id=document.trace_id, trace_digest=document.content_digest)
    results: dict[str, tuple[str, str, bool | None]] = {}
    for message in document.messages:
        for part in message.parts:
            if str(part.type) == PartType.TOOL_RESULT and part.tool_call_id:
                results.setdefault(part.tool_call_id, (message.message_id, part.part_id, part.is_error))
    for message in document.messages:
        for part in message.parts:
            if str(part.type) != PartType.TOOL_CALL or not part.tool_call_id:
                continue
            labels: list[str] = []
            try:
                json.loads(part.arguments_json or "")
                labels.append("tool_call.valid_json")
            except ValueError:
                labels.append("tool_call.malformed_arguments")
            evidence = [
                {"kind": SelectorKind.PART.value, "entity_id": message.message_id, "part_id": part.part_id}
            ]
            answer = results.get(part.tool_call_id)
            if answer is None:
                labels.append("tool_call.unanswered")
            else:
                labels.append("tool_call.answered")
                if answer[2]:
                    labels.append("tool_call.result_error")
                evidence.append({"kind": SelectorKind.PART.value, "entity_id": answer[0], "part_id": answer[1]})
            proposal["findings"].append(
                {
                    "target": {"kind": SelectorKind.PART.value, "entity_id": message.message_id, "part_id": part.part_id},
                    "annotation_type": "tool_call",
                    "labels": labels,
                    "payload": {"tool_call_id": part.tool_call_id, **({"tool_name": part.tool_name} if part.tool_name else {})},
                    "confidence": 1.0,
                    "rationale": "deterministic structural check",
                    "evidence": evidence,
                }
            )
    return proposal


def environment_step_status_definition() -> TraceAnnotatorDefinitionV1:
    return TraceAnnotatorDefinitionV1(
        annotator_id=ENVIRONMENT_STEP_STATUS_ID,
        name="Environment step status",
        purpose="Label each environment_step span by its recorded outcome.",
        taxonomy=(
            "environment_step.ok",
            "environment_step.error",
            "environment_step.no_effect",
            "environment_step.blocked",
        ),
        version="v1",
        required_subject_scope="span",
        grounding_requirement=AnnotatorGroundingRequirement.EXACT_SELECTOR,
        minimum_evidence=1,
        unavailable_evidence_behavior=UnavailableEvidenceBehavior.ABSTAIN,
        confidence_semantics=ConfidenceSemantics.DETERMINISTIC,
        output_contract=AnnotationOutputContractV1(
            task_kind=AnnotationTaskKind.CLASSIFY,
            annotation_types=("environment_step",),
            taxonomy=(
                AnnotationTaxonV1(label="environment_step.ok", description="span status ok"),
                AnnotationTaxonV1(label="environment_step.error", description="span status error"),
                AnnotationTaxonV1(label="environment_step.no_effect", description="the engine reported a no-op transition"),
                AnnotationTaxonV1(label="environment_step.blocked", description="the engine reported a blocked transition"),
            ),
            payload_schema=AnnotationPayloadSchemaV1(
                schema_id="synth.environment_step_status.payload",
                version="v1",
                fields=(
                    AnnotationPayloadFieldV1(field_name="action", value_kind="string", required=False),
                    AnnotationPayloadFieldV1(field_name="transition", value_kind="string", required=False),
                    AnnotationPayloadFieldV1(field_name="reason", value_kind="string", required=False),
                    AnnotationPayloadFieldV1(field_name="step_index", value_kind="integer", required=False),
                ),
            ),
            allowed_producer_kinds=(ProducerKind.DETERMINISTIC, ProducerKind.COMPOSITE),
        ),
    ).sealed()


def environment_step_status_program() -> AnnotatorProgramV1:
    return AnnotatorProgramV1(
        program_id=f"{ENVIRONMENT_STEP_STATUS_ID}.program",
        runner_kind=RunnerKind.DETERMINISTIC,
        version="v1",
        program_ref="synth_containers.tracing.annotation.builtin:environment_step_status",
    ).sealed()


def environment_step_status(document: TraceDocumentV5, context: ProgramContext) -> dict[str, Any]:
    proposal = empty_proposal(trace_id=document.trace_id, trace_digest=document.content_digest)
    for span in document.spans:
        if str(span.span_kind) != SpanKind.ENVIRONMENT_STEP:
            continue
        labels = ["environment_step.ok" if str(span.status) == SpanStatus.OK else "environment_step.error"]
        transition = str(span.detail.get("transition") or "")
        if transition == "noop":
            labels.append("environment_step.no_effect")
        elif transition.startswith("blocked"):
            labels.append("environment_step.blocked")
        payload: dict[str, Any] = {}
        for key in ("action", "transition", "reason"):
            value = span.detail.get(key)
            if isinstance(value, str) and value:
                payload[key] = value
        step_index = span.detail.get("step_index")
        if isinstance(step_index, int) and not isinstance(step_index, bool):
            payload["step_index"] = step_index
        proposal["findings"].append(
            {
                "target": {"kind": SelectorKind.SPAN.value, "entity_id": span.span_id},
                "annotation_type": "environment_step",
                "labels": labels,
                "payload": payload,
                "confidence": 1.0,
                "rationale": "deterministic span status read",
                "evidence": [{"kind": SelectorKind.SPAN.value, "entity_id": span.span_id}],
            }
        )
    return proposal


def register_builtin_annotators(registry: DefinitionRegistry) -> None:
    registry.register(
        tool_call_integrity_definition(),
        tool_call_integrity_program(),
        domain="generic",
        deterministic_program=tool_call_integrity,
    )
    registry.register(
        environment_step_status_definition(),
        environment_step_status_program(),
        domain="generic",
        deterministic_program=environment_step_status,
    )


__all__ = [
    "ENVIRONMENT_STEP_STATUS_ID",
    "TOOL_CALL_INTEGRITY_ID",
    "environment_step_status",
    "environment_step_status_definition",
    "environment_step_status_program",
    "register_builtin_annotators",
    "tool_call_integrity",
    "tool_call_integrity_definition",
    "tool_call_integrity_program",
]
