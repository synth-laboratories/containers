"""Annotation job contracts: what a job asked for, and where it is in its lifecycle.

A job binds one sealed Trace V5 digest to one sealed annotator definition (and,
optionally, one sealed rubric), one model/effort, one tool contract, and one
runner version. Everything that determines execution is content-addressed so an
identical request is recognised and served from the local store instead of
starting another paid task.

States::

    prepared -> running -> validating -> sealed
                        \\-> abstained
                        \\-> failed
                        \\-> cancelled

State transitions are appended as new sealed job revisions; a job record is never
edited in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from synth_containers.serde import JsonDataclassMixin

from ..canonical import canonical_bytes, content_digest, record_id, seal_record, utc_now


ANNOTATION_JOB_SCHEMA_VERSION = "synth.annotation-job.v1"
ANNOTATION_JOB_REQUEST_SCHEMA_VERSION = "synth.annotation-job-request.v1"
ANNOTATION_ESTIMATE_SCHEMA_VERSION = "synth.annotation-estimate.v1"
RUNNER_VERSION = "synth.annotation-runner@1"


class AnnotationJobState(StrEnum):
    PREPARED = "prepared"
    RUNNING = "running"
    VALIDATING = "validating"
    SEALED = "sealed"
    ABSTAINED = "abstained"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES: frozenset[AnnotationJobState] = frozenset(
    {
        AnnotationJobState.SEALED,
        AnnotationJobState.ABSTAINED,
        AnnotationJobState.FAILED,
        AnnotationJobState.CANCELLED,
    }
)

_ALLOWED_TRANSITIONS: dict[AnnotationJobState, frozenset[AnnotationJobState]] = {
    AnnotationJobState.PREPARED: frozenset(
        {AnnotationJobState.RUNNING, AnnotationJobState.FAILED, AnnotationJobState.CANCELLED}
    ),
    AnnotationJobState.RUNNING: frozenset(
        {
            AnnotationJobState.VALIDATING,
            AnnotationJobState.FAILED,
            AnnotationJobState.CANCELLED,
        }
    ),
    AnnotationJobState.VALIDATING: frozenset(
        {
            AnnotationJobState.SEALED,
            AnnotationJobState.ABSTAINED,
            AnnotationJobState.FAILED,
            AnnotationJobState.CANCELLED,
        }
    ),
    AnnotationJobState.SEALED: frozenset(),
    AnnotationJobState.ABSTAINED: frozenset(),
    AnnotationJobState.FAILED: frozenset(),
    AnnotationJobState.CANCELLED: frozenset(),
}


class AnnotationJobMode(StrEnum):
    ANNOTATE = "annotate"
    VERIFY = "verify"
    ADJUDICATE = "adjudicate"


class AnnotationJobErrorCode(StrEnum):
    """Typed, surfaceable failure reasons. Free text lives in ``message``."""

    SOURCE_TRACE_UNAVAILABLE = "source_trace_unavailable"
    SOURCE_DIGEST_MISMATCH = "source_digest_mismatch"
    DEFINITION_UNKNOWN = "definition_unknown"
    DEFINITION_DIGEST_MISMATCH = "definition_digest_mismatch"
    RUBRIC_REQUIRED = "rubric_required"
    RUNNER_UNAVAILABLE = "runner_unavailable"
    RESERVATION_REQUIRED = "reservation_required"
    RESERVATION_REJECTED = "reservation_rejected"
    TOOL_LIMIT_EXCEEDED = "tool_limit_exceeded"
    TOKEN_LIMIT_EXCEEDED = "token_limit_exceeded"
    COST_LIMIT_EXCEEDED = "cost_limit_exceeded"
    TIMEOUT = "timeout"
    TRANSPORT_DISCONNECTED = "transport_disconnected"
    NO_STRUCTURED_OUTPUT = "no_structured_output"
    MALFORMED_OUTPUT = "malformed_output"
    UNSUPPORTED_FINDING = "unsupported_finding"
    EVIDENCE_INVALID = "evidence_invalid"
    REVISION_CONFLICT = "revision_conflict"
    STORE_CORRUPT = "store_corrupt"
    CANCELLED = "cancelled"
    INTERNAL = "internal"


@dataclass(frozen=True, slots=True)
class AnnotationJobLimitsV1(JsonDataclassMixin):
    """Hard bounds the runner enforces. Exceeding any of them fails the job."""

    max_tool_calls: int = 200
    max_tool_response_bytes: int = 32_000
    max_total_tool_bytes: int = 2_000_000
    max_total_tokens: int | None = 400_000
    max_cost_usd: float | None = None
    timeout_seconds: float = 900.0


@dataclass(frozen=True, slots=True)
class AnnotationJobRequestV1(JsonDataclassMixin):
    """Everything that determines what an annotation job will do.

    Mutable presentation values (display names, file paths, timestamps) are
    deliberately absent; they never belong in the identity of a result.
    """

    source_trace_id: str
    source_trace_digest: str
    annotator_id: str
    annotator_digest: str
    model: str | None = None
    reasoning_effort: str | None = None
    mode: AnnotationJobMode | str = AnnotationJobMode.ANNOTATE
    rubric_id: str | None = None
    rubric_digest: str | None = None
    allowed_projection_ids: tuple[str, ...] = ()
    allowed_projection_digests: tuple[str, ...] = ()
    target_selector_ids: tuple[str, ...] = ()
    repeat_index: int = 0
    parent_job_id: str | None = None
    source_annotation_ids: tuple[str, ...] = ()
    limits: AnnotationJobLimitsV1 = field(default_factory=AnnotationJobLimitsV1)
    runner_version: str = RUNNER_VERSION
    schema_version: str = ANNOTATION_JOB_REQUEST_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AnnotationJobErrorV1(JsonDataclassMixin):
    code: AnnotationJobErrorCode | str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AnnotationJobUsageV1(JsonDataclassMixin):
    """What the annotator consumed; ``cost_usd`` is null when nobody billed it."""

    tool_calls: int = 0
    tool_bytes: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    cost_status: str = "unavailable"
    wall_time_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class AnnotationJobV1(JsonDataclassMixin):
    """One sealed revision of a job record."""

    job_id: str
    request: AnnotationJobRequestV1
    idempotency_key: str
    state: AnnotationJobState | str
    created_at: str
    updated_at: str
    revision: int = 1
    program_digest: str | None = None
    tool_contract_digest: str | None = None
    workspace_manifest_digest: str | None = None
    receipt_ids: tuple[str, ...] = ()
    bundle_id: str | None = None
    bundle_digest: str | None = None
    prior_bundle_digest: str | None = None
    annotation_ids: tuple[str, ...] = ()
    verifier_result_ids: tuple[str, ...] = ()
    applied_count: int = 0
    abstained_count: int = 0
    rejected_count: int = 0
    execution_trace_id: str | None = None
    execution_trace_digest: str | None = None
    usage: AnnotationJobUsageV1 = field(default_factory=AnnotationJobUsageV1)
    cached_from_job_id: str | None = None
    error: AnnotationJobErrorV1 | None = None
    reservation_id: str | None = None
    schema_version: str = ANNOTATION_JOB_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    content_digest: str = ""

    def sealed(self) -> "AnnotationJobV1":
        return seal_record(self)

    @property
    def terminal(self) -> bool:
        return AnnotationJobState(str(self.state)) in TERMINAL_STATES

    def transition(self, state: AnnotationJobState | str, **changes: Any) -> "AnnotationJobV1":
        """Return the next sealed revision or raise if the transition is illegal."""

        current = AnnotationJobState(str(self.state))
        target = AnnotationJobState(str(state))
        if target not in _ALLOWED_TRANSITIONS[current]:
            raise ValueError(f"illegal annotation job transition {current} -> {target}")
        return replace(
            self,
            state=target,
            revision=self.revision + 1,
            updated_at=utc_now(),
            content_digest="",
            **changes,
        ).sealed()


@dataclass(frozen=True, slots=True)
class AnnotationEstimateV1(JsonDataclassMixin):
    """A compact approval summary. Never a promise of the exact bill."""

    idempotency_key: str
    cached: bool
    cached_job_id: str | None
    paid: bool
    runner_kind: str
    model: str | None
    reasoning_effort: str | None
    max_tool_calls: int
    max_total_tokens: int | None
    max_cost_usd: float | None
    repeat_index: int
    requires_reservation: bool
    resolved_model: str | None = None
    resolved_reasoning_effort: str | None = None
    schema_version: str = ANNOTATION_ESTIMATE_SCHEMA_VERSION
    notes: tuple[str, ...] = ()


def idempotency_key(
    request: AnnotationJobRequestV1,
    *,
    program_digest: str | None,
    tool_contract_digest: str | None,
    runner_version: str | None = None,
) -> str:
    """Content address of everything that determines execution.

    Intentionally excludes ``metadata`` and ``limits.timeout_seconds``: neither
    changes what an annotator can conclude about a sealed trace. ``model`` and
    ``reasoning_effort`` must already be *resolved* (never ``None`` for a paid
    runner) so a changed default can never serve an older model's output.
    """

    identity = {
        "source_trace_digest": request.source_trace_digest,
        "annotator_digest": request.annotator_digest,
        "rubric_digest": request.rubric_digest,
        "program_digest": program_digest,
        "model": request.model,
        "reasoning_effort": request.reasoning_effort,
        "mode": str(request.mode),
        "tool_contract_digest": tool_contract_digest,
        "allowed_projection_digests": sorted(request.allowed_projection_digests),
        "target_selector_ids": sorted(request.target_selector_ids),
        "runner_version": request.runner_version,
        "runner_implementation": runner_version,
        "repeat_index": request.repeat_index,
        "parent_job_id": request.parent_job_id,
        "source_annotation_ids": sorted(request.source_annotation_ids),
        "limits": {
            "max_tool_calls": request.limits.max_tool_calls,
            "max_tool_response_bytes": request.limits.max_tool_response_bytes,
            "max_total_tool_bytes": request.limits.max_total_tool_bytes,
            "max_total_tokens": request.limits.max_total_tokens,
        },
    }
    return content_digest(identity)


def new_job(
    request: AnnotationJobRequestV1,
    *,
    key: str,
    program_digest: str | None,
    tool_contract_digest: str | None,
    reservation_id: str | None = None,
) -> AnnotationJobV1:
    now = utc_now()
    job_id = record_id(
        "ajob",
        kind="annotation_job",
        scope=(request.source_trace_id,),
        key={"idempotency_key": key, "created_at": now},
    )
    return AnnotationJobV1(
        job_id=job_id,
        request=request,
        idempotency_key=key,
        state=AnnotationJobState.PREPARED,
        created_at=now,
        updated_at=now,
        program_digest=program_digest,
        tool_contract_digest=tool_contract_digest,
        reservation_id=reservation_id,
    ).sealed()


def request_digest(request: AnnotationJobRequestV1) -> str:
    return content_digest(canonical_bytes(request))


__all__ = [
    "ANNOTATION_ESTIMATE_SCHEMA_VERSION",
    "ANNOTATION_JOB_REQUEST_SCHEMA_VERSION",
    "ANNOTATION_JOB_SCHEMA_VERSION",
    "RUNNER_VERSION",
    "TERMINAL_STATES",
    "AnnotationEstimateV1",
    "AnnotationJobErrorCode",
    "AnnotationJobErrorV1",
    "AnnotationJobLimitsV1",
    "AnnotationJobMode",
    "AnnotationJobRequestV1",
    "AnnotationJobState",
    "AnnotationJobUsageV1",
    "AnnotationJobV1",
    "idempotency_key",
    "new_job",
    "request_digest",
]
