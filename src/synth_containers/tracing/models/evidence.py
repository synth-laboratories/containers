"""``TraceEvidenceBundleV5`` — append-only derived evidence about a sealed trace."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from synth_containers.serde import JsonDataclassMixin

from ..canonical import content_digest
from .artifacts import ArtifactRefV5
from .identity import EVIDENCE_BUNDLE_SCHEMA_VERSION
from .standards import (
    AnnotationV1,
    BenchmarkVerdictV1,
    CriterionDefinitionV1,
    EvaluationResultV1,
    ReceiptV1,
    RewardAggregationV1,
    RewardDefinitionV1,
    RewardRecordV1,
    RubricDefinitionV2,
    TraceAnnotatorDefinitionV1,
    VerifierDefinitionV1,
    VerifierResultV2,
)


@dataclass(frozen=True, slots=True)
class TraceRefV5(JsonDataclassMixin):
    trace_id: str
    content_digest: str
    schema_version: str = "synth.trace.v5"


@dataclass(frozen=True, slots=True)
class TraceEvidenceBundleV5(JsonDataclassMixin):
    """Derived records about one sealed trace. Appending never changes the trace digest."""

    bundle_id: str
    trace_ref: TraceRefV5
    created_at: str
    criteria: tuple[CriterionDefinitionV1, ...] = ()
    rubrics: tuple[RubricDefinitionV2, ...] = ()
    verifier_definitions: tuple[VerifierDefinitionV1, ...] = ()
    annotator_definitions: tuple[TraceAnnotatorDefinitionV1, ...] = ()
    reward_definitions: tuple[RewardDefinitionV1, ...] = ()
    annotations: tuple[AnnotationV1, ...] = ()
    verifier_results: tuple[VerifierResultV2, ...] = ()
    reward_records: tuple[RewardRecordV1, ...] = ()
    reward_aggregations: tuple[RewardAggregationV1, ...] = ()
    evaluation_results: tuple[EvaluationResultV1, ...] = ()
    benchmark_verdicts: tuple[BenchmarkVerdictV1, ...] = ()
    receipts: tuple[ReceiptV1, ...] = ()
    artifacts: tuple[ArtifactRefV5, ...] = ()
    schema_version: str = EVIDENCE_BUNDLE_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    content_digest: str = ""

    def sealed(self) -> "TraceEvidenceBundleV5":
        return replace(self, content_digest=content_digest(self))

    def selectors(self) -> tuple[Any, ...]:
        """Every selector this bundle cites, for validation against the sealed trace."""

        found: list[Any] = []
        for annotation in self.annotations:
            found.append(annotation.target)
            found.extend(annotation.evidence)
        for result in self.verifier_results:
            found.append(result.subject)
            found.extend(result.evidence)
            for judgment in result.judgments:
                if judgment.subject is not None:
                    found.append(judgment.subject)
                found.extend(judgment.evidence)
                if judgment.adjudication is not None:
                    found.extend(judgment.adjudication.evidence)
        for record in self.reward_records:
            found.append(record.subject)
            found.extend(record.evidence)
        for evaluation in self.evaluation_results:
            found.append(evaluation.subject)
        return tuple(found)


__all__ = ["TraceEvidenceBundleV5", "TraceRefV5"]
