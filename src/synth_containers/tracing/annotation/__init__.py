"""Agentic and deterministic annotation over sealed Trace V5 documents.

Ownership: this package owns how annotations become trustworthy evidence —
job lifecycle, idempotency, bounded read-only inspection, selector validation,
sealing, persistence, receipts, consensus lineage. Domain semantics (what a
Craftax belief error *is*) live in ``evals``; presentation lives in Workshop.
"""

from .broker import (
    DenyAllBroker,
    FlakyBroker,
    LocalReservationBroker,
    PaidComputeBroker,
    PaidComputeReservationV1,
    ReservationBindingV1,
    ReservationError,
    usd_to_micros,
)
from .builtin import register_builtin_annotators
from .campaign import AnnotationCampaign, AnnotatorPlan, CampaignEstimate, CampaignPlan, CampaignRun, plan_from_refs
from .codex_app_server import CodexAppServerRunner, ScriptedAppServer, StdioAppServerTransport
from .consensus import AgreementReportV1, adjudication_annotation, agreement, consensus_annotation
from .definitions import AnnotatorProgramV1, DefinitionRegistry, ProgramContext, RegisteredAnnotator, RunnerKind
from .evidence_check import validate_appended_evidence
from .execution_trace import ExecutionCapture, build_execution_trace
from .fixtures import build_craftax_compaction_trace, build_craftax_smoke_trace
from .ledger import PaidLedger, PaidLedgerEntryV1
from .model_api import CompletionResult, ModelApiRunner, render_trace_digest
from .scheduler import AnnotationScheduler, SchedulerStats, ThroughputLimits
from .sources import bundle_trace_loader, bundle_trace_refs, chain_loaders
from .jobs import (
    AnnotationEstimateV1,
    AnnotationJobErrorCode,
    AnnotationJobErrorV1,
    AnnotationJobLimitsV1,
    AnnotationJobMode,
    AnnotationJobRequestV1,
    AnnotationJobState,
    AnnotationJobUsageV1,
    AnnotationJobV1,
    idempotency_key,
)
from .operations import AnnotationOperations, OPERATION_DESCRIPTORS
from .persistence import AnnotationStore, RevisionConflict, StoreCorruption
from .pricing import PRICE_TABLE_ENV, ModelPrice, PriceTable, PriceTableError
from .proposal import PROPOSAL_JSON_SCHEMA, PROPOSAL_SCHEMA_VERSION, check_proposal_shape, empty_proposal
from .service import AnnotationService, AnnotationServiceError, DeterministicRunner, RunContext, RunOutcome
from .tools import TOOL_NAMES, TOOL_SPECS, TraceInspectionTools, ToolLimitExceeded, tool_contract_digest
from .trace_index import IndexedTraceDocument, SealedTraceCache, SealedTraceIndex
from .validation import ProposalValidationResult, ProposalValidator
from .worker import AnnotationWorker

__all__ = [
    "OPERATION_DESCRIPTORS",
    "PRICE_TABLE_ENV",
    "PROPOSAL_JSON_SCHEMA",
    "PROPOSAL_SCHEMA_VERSION",
    "TOOL_NAMES",
    "TOOL_SPECS",
    "AgreementReportV1",
    "AnnotationEstimateV1",
    "AnnotationJobErrorCode",
    "AnnotationJobErrorV1",
    "AnnotationJobLimitsV1",
    "AnnotationJobMode",
    "AnnotationJobRequestV1",
    "AnnotationJobState",
    "AnnotationJobUsageV1",
    "AnnotationJobV1",
    "AnnotationCampaign",
    "AnnotationScheduler",
    "AnnotatorPlan",
    "CampaignEstimate",
    "CampaignPlan",
    "CampaignRun",
    "CompletionResult",
    "IndexedTraceDocument",
    "ModelApiRunner",
    "ModelPrice",
    "PriceTable",
    "PriceTableError",
    "SchedulerStats",
    "ThroughputLimits",
    "AnnotationOperations",
    "AnnotationService",
    "AnnotationServiceError",
    "AnnotationStore",
    "AnnotationWorker",
    "AnnotatorProgramV1",
    "CodexAppServerRunner",
    "DefinitionRegistry",
    "DenyAllBroker",
    "FlakyBroker",
    "PaidLedger",
    "PaidLedgerEntryV1",
    "DeterministicRunner",
    "ExecutionCapture",
    "LocalReservationBroker",
    "PaidComputeBroker",
    "PaidComputeReservationV1",
    "ProgramContext",
    "ProposalValidationResult",
    "ProposalValidator",
    "RegisteredAnnotator",
    "ReservationBindingV1",
    "ReservationError",
    "RevisionConflict",
    "RunContext",
    "RunOutcome",
    "RunnerKind",
    "ScriptedAppServer",
    "SealedTraceCache",
    "SealedTraceIndex",
    "StdioAppServerTransport",
    "StoreCorruption",
    "ToolLimitExceeded",
    "TraceInspectionTools",
    "adjudication_annotation",
    "agreement",
    "build_craftax_compaction_trace",
    "bundle_trace_loader",
    "bundle_trace_refs",
    "chain_loaders",
    "build_craftax_smoke_trace",
    "build_execution_trace",
    "check_proposal_shape",
    "consensus_annotation",
    "empty_proposal",
    "idempotency_key",
    "plan_from_refs",
    "render_trace_digest",
    "register_builtin_annotators",
    "tool_contract_digest",
    "usd_to_micros",
    "validate_appended_evidence",
]
