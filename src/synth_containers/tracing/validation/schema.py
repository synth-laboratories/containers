"""Generate JSON Schema artifacts from the V5 dataclass contracts.

The schemas are generated, not hand-maintained, so they cannot drift from the types
the library actually writes. They describe the canonical form: ``None`` values are
dropped on write, so optional fields are simply absent rather than null.
"""

from __future__ import annotations

import types
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from typing import Any, Union, get_args, get_origin, get_type_hints

from ..capture.binding import TraceCaptureBindingV1
from ..capture.coverage import CaptureCoverageReceiptV1
from ..capture.envelope import RawCaptureEnvelopeV1
from ..capture.spool import TraceSegmentManifestV1
from ..models.capture_data import CapturedBodyRefV1
from ..models.tokens import TokenCaptureV5, TokenSequenceRefV1
from ..store.bundle import (
    BundleManifestPointerV1,
    BundleManifestV1,
    BundleObjectRefV1,
)
from ..models.document import TraceDocumentV5
from ..models.evidence import TraceEvidenceBundleV5
from ..models.projection import ProjectionManifestV1
from ..models.selectors import TraceSelectorV1
from ..models.standards import (
    AnnotationDerivationV1,
    AnnotationEvidenceGapsV1,
    AnnotationInspectionV1,
    AnnotationOutputContractV1,
    AnnotationPayloadFieldV1,
    AnnotationPayloadSchemaV1,
    AnnotationTaxonV1,
    AnnotationV1,
    BenchmarkVerdictV1,
    CriterionDefinitionV1,
    EvaluationResultV1,
    JudgmentAdjudicationV1,
    JudgmentV1,
    ReceiptV1,
    RewardAggregationV1,
    RewardDefinitionV1,
    RewardRecordV1,
    RubricDefinitionV2,
    TraceAnnotatorDefinitionV1,
    UnavailableAnnotationEvidenceV1,
    VerifierDefinitionV1,
    VerifierResultV2,
)
from .validator import ValidationReceiptV1


SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

PUBLIC_CONTRACTS: tuple[type, ...] = (
    TraceCaptureBindingV1,
    RawCaptureEnvelopeV1,
    TraceSegmentManifestV1,
    CapturedBodyRefV1,
    CaptureCoverageReceiptV1,
    TraceDocumentV5,
    TraceEvidenceBundleV5,
    TokenCaptureV5,
    TokenSequenceRefV1,
    BundleManifestV1,
    BundleManifestPointerV1,
    BundleObjectRefV1,
    TraceSelectorV1,
    CriterionDefinitionV1,
    RubricDefinitionV2,
    JudgmentV1,
    JudgmentAdjudicationV1,
    VerifierDefinitionV1,
    VerifierResultV2,
    TraceAnnotatorDefinitionV1,
    AnnotationTaxonV1,
    AnnotationPayloadFieldV1,
    AnnotationPayloadSchemaV1,
    AnnotationOutputContractV1,
    UnavailableAnnotationEvidenceV1,
    AnnotationEvidenceGapsV1,
    AnnotationInspectionV1,
    AnnotationDerivationV1,
    AnnotationV1,
    EvaluationResultV1,
    BenchmarkVerdictV1,
    ReceiptV1,
    RewardDefinitionV1,
    RewardRecordV1,
    RewardAggregationV1,
    ProjectionManifestV1,
    ValidationReceiptV1,
)


def json_schema(record_type: type) -> dict[str, Any]:
    """Build a self-contained JSON Schema for one contract."""

    defs: dict[str, Any] = {}
    root = _schema_for(record_type, defs)
    schema: dict[str, Any] = {
        "$schema": SCHEMA_DIALECT,
        "title": record_type.__name__,
        **root,
    }
    if defs:
        schema["$defs"] = defs
    return schema


def all_schemas() -> dict[str, dict[str, Any]]:
    return {record_type.__name__: json_schema(record_type) for record_type in PUBLIC_CONTRACTS}


def _schema_for(record_type: type, defs: dict[str, Any]) -> dict[str, Any]:
    hints = get_type_hints(record_type)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for field_info in fields(record_type):
        properties[field_info.name] = _annotation_schema(hints[field_info.name], defs)
        if (
            field_info.default is MISSING and field_info.default_factory is MISSING  # type: ignore[misc]
            and not _allows_none(hints[field_info.name])
        ):
            required.append(field_info.name)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def _allows_none(annotation: Any) -> bool:
    origin = get_origin(annotation)
    return origin in (Union, types.UnionType) and type(None) in get_args(annotation)


def _annotation_schema(annotation: Any, defs: dict[str, Any]) -> dict[str, Any]:
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        options = [item for item in get_args(annotation) if item is not type(None)]
        schemas = [_annotation_schema(item, defs) for item in options]
        unique: list[dict[str, Any]] = []
        for item in schemas:
            if item not in unique:
                unique.append(item)
        return unique[0] if len(unique) == 1 else {"anyOf": unique}
    if origin in (tuple, list):
        args = get_args(annotation)
        element = args[0] if args else Any
        return {"type": "array", "items": _annotation_schema(element, defs)}
    if origin is dict:
        return {"type": "object"}
    if is_dataclass(annotation):
        name = annotation.__name__
        if name not in defs:
            defs[name] = {}
            defs[name] = _schema_for(annotation, defs)
        return {"$ref": f"#/$defs/{name}"}
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return {"type": "string", "enum": [member.value for member in annotation]}
    return _primitive_schema(annotation)


def _primitive_schema(annotation: Any) -> dict[str, Any]:
    if annotation is str:
        return {"type": "string"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    return {}


__all__ = ["PUBLIC_CONTRACTS", "SCHEMA_DIALECT", "all_schemas", "json_schema"]
