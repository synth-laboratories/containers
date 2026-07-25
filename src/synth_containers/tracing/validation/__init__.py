"""Validation: invariants, generated JSON Schemas, and rehydration."""

from .rehydrate import (
    RehydrationError,
    evidence_bundle_from_payload,
    rehydrate_trace,
    trace_document_from_payload,
)
from .schema import PUBLIC_CONTRACTS, all_schemas, json_schema
from .validator import (
    Severity,
    ValidationFindingV1,
    ValidationReceiptV1,
    validate,
    validate_evidence,
    validate_trace,
)

__all__ = [
    "PUBLIC_CONTRACTS",
    "RehydrationError",
    "Severity",
    "ValidationFindingV1",
    "ValidationReceiptV1",
    "all_schemas",
    "evidence_bundle_from_payload",
    "json_schema",
    "rehydrate_trace",
    "trace_document_from_payload",
    "validate",
    "validate_evidence",
    "validate_trace",
]
