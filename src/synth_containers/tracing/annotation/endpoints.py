"""Versioned HTTP contract for in-container annotators.

Workshop and MCP shims proxy this surface. Paths, methods, and paid flags here
are the source of truth; ``OPERATION_DESCRIPTORS`` names the agent operations
that map onto them 1:1. Hidden chain-of-thought is never a stream event.
"""

from __future__ import annotations

from typing import Any


ANNOTATION_API_SCHEMA = "synth.container.annotation-api.v1"
ANNOTATION_STREAM_SCHEMA = "synth.annotation.stream.v1"

ANNOTATION_EVENT_KINDS: tuple[str, ...] = (
    "annotation.prepared",
    "annotation.running",
    "annotation.tool",
    "annotation.validating",
    "annotation.finding",
    "annotation.sealed",
    "annotation.abstained",
    "annotation.failed",
    "annotation.cancelled",
    "annotation.cached",
)

# Core router (``build_annotation_router``) plus extras ``mount_annotation`` adds.
ANNOTATION_ENDPOINTS: tuple[dict[str, Any], ...] = (
    {
        "name": "annotation_catalog",
        "method": "GET",
        "path": "/annotation/catalog",
        "operation": None,
        "paid": False,
        "surface": "core",
        "description": "This contract: operations, event kinds, runner kinds, routes.",
    },
    {
        "name": "annotation_operations",
        "method": "GET",
        "path": "/annotation/operations",
        "operation": None,
        "paid": False,
        "surface": "core",
        "description": "Agent operation descriptors (read-only / paid / arguments).",
    },
    {
        "name": "annotation_scheduler",
        "method": "GET",
        "path": "/annotation/scheduler",
        "operation": None,
        "paid": False,
        "surface": "core",
        "description": "Throughput snapshot: queued, running, per-class caps.",
    },
    {
        "name": "annotation_list_definitions",
        "method": "GET",
        "path": "/traces/{trace_id}/annotation-definitions",
        "operation": "annotation_list_definitions",
        "paid": False,
        "surface": "core",
        "description": "Annotators and rubrics compatible with the sealed trace.",
    },
    {
        "name": "annotation_estimate",
        "method": "POST",
        "path": "/traces/{trace_id}/annotation-estimates",
        "operation": "annotation_estimate",
        "paid": False,
        "surface": "core",
        "description": "Idempotency key, cache hit, resolved model/effort/runner, reservation need.",
    },
    {
        "name": "annotation_start",
        "method": "POST",
        "path": "/traces/{trace_id}/annotation-jobs",
        "operation": "annotation_start",
        "paid": True,
        "status": 202,
        "surface": "core",
        "description": (
            "Enqueue one job. Body.request must bear model, reasoning_effort, and runner_kind "
            "for paid annotators. Returns stream poll/SSE URLs; poll or subscribe, do not wait on POST."
        ),
    },
    {
        "name": "annotation_get",
        "method": "GET",
        "path": "/annotation-jobs/{job_id}",
        "operation": "annotation_get",
        "paid": False,
        "surface": "core",
        "description": "Job record, receipts, sealed output ids.",
    },
    {
        "name": "annotation_events",
        "method": "GET",
        "path": "/annotation-jobs/{job_id}/events",
        "operation": "annotation_events",
        "paid": False,
        "stream": "poll",
        "surface": "core",
        "description": "Sequence-cursor page of annotation lifecycle events.",
    },
    {
        "name": "annotation_stream",
        "method": "GET",
        "path": "/annotation-jobs/{job_id}/stream",
        "operation": None,
        "paid": False,
        "stream": "sse",
        "surface": "core",
        "description": "SSE of the same events. Last-Event-ID resumes. No hidden CoT.",
    },
    {
        "name": "annotation_cancel",
        "method": "POST",
        "path": "/annotation-jobs/{job_id}/cancel",
        "operation": "annotation_cancel",
        "paid": False,
        "surface": "core",
        "description": "Cancel prepared or running. Sealed results stay.",
    },
    {
        "name": "annotation_list",
        "method": "GET",
        "path": "/traces/{trace_id}/annotations",
        "operation": "annotation_list",
        "paid": False,
        "surface": "core",
        "description": "Current evidence-head annotations. Labels are not reward.",
    },
    {
        "name": "annotation_get_evidence",
        "method": "GET",
        "path": "/annotations/{annotation_id}",
        "operation": "annotation_get_evidence",
        "paid": False,
        "surface": "core",
        "description": "Resolve selectors against the sealed trace; return cited text.",
    },
    {
        "name": "evidence_bundles",
        "method": "GET",
        "path": "/traces/{trace_id}/evidence-bundles",
        "operation": None,
        "paid": False,
        "surface": "core",
        "description": "Sealed evidence bundle revisions for a trace, including bounded verifier_results.",
    },
    {
        "name": "evidence_head",
        "method": "GET",
        "path": "/traces/{trace_id}/evidence-head",
        "operation": None,
        "paid": False,
        "surface": "core",
        "description": "Current evidence-head revision: counts plus bounded verifier_results. Missing verifier evidence is not a zero score.",
    },
    {
        "name": "evidence_bundle",
        "method": "GET",
        "path": "/traces/{trace_id}/evidence-bundles/{bundle_digest}",
        "operation": None,
        "paid": False,
        "surface": "core",
        "description": "One sealed evidence-bundle revision by content digest.",
    },
    {
        "name": "verification_start",
        "method": "POST",
        "path": "/traces/{trace_id}/verification-jobs",
        "operation": "verification_start",
        "paid": True,
        "status": 202,
        "surface": "core",
        "description": "Enqueue rubric verification. Seals VerifierResultV2; never rewrites reward_signal.",
    },
    {
        "name": "verification_get",
        "method": "GET",
        "path": "/verification-jobs/{job_id}",
        "operation": "verification_get",
        "paid": False,
        "surface": "core",
        "description": "Verification job and sealed verifier result id.",
    },
    {
        "name": "verification_events",
        "method": "GET",
        "path": "/verification-jobs/{job_id}/events",
        "operation": "annotation_events",
        "paid": False,
        "stream": "poll",
        "surface": "core",
        "description": "Same event log as the underlying annotation job.",
    },
    {
        "name": "verification_stream",
        "method": "GET",
        "path": "/verification-jobs/{job_id}/stream",
        "operation": None,
        "paid": False,
        "stream": "sse",
        "surface": "core",
        "description": "SSE alias for the verification job.",
    },
    {
        "name": "annotation_review",
        "method": "POST",
        "path": "/annotations/{annotation_id}/reviews",
        "operation": "annotation_review",
        "paid": False,
        "surface": "core",
        "description": "Accept/reject/dispute/flag. Appends a superseding revision.",
    },
    {
        "name": "annotation_consensus",
        "method": "POST",
        "path": "/traces/{trace_id}/annotation-consensus",
        "operation": "annotation_consensus",
        "paid": False,
        "surface": "core",
        "description": "Inter-annotator agreement and majority consensus records.",
    },
    {
        "name": "annotation_traces",
        "method": "GET",
        "path": "/annotation/traces",
        "operation": None,
        "paid": False,
        "surface": "container",
        "description": "Sealed Trace V5 bundles this container can annotate.",
    },
    {
        "name": "annotation_status",
        "method": "GET",
        "path": "/annotation/status",
        "operation": None,
        "paid": False,
        "surface": "container",
        "description": "Registered annotators, post-rollout plans, scheduler, broker.",
    },
    {
        "name": "annotation_reservations",
        "method": "GET",
        "path": "/annotation/reservations",
        "operation": None,
        "paid": False,
        "surface": "container",
        "description": "Broker reconciliation outbox for the host to pull.",
    },
    {
        "name": "annotation_pricing",
        "method": "GET",
        "path": "/annotation/pricing",
        "operation": None,
        "paid": False,
        "surface": "container",
        "description": "Priced models per runner. Unpriced paid models are refused.",
    },
    {
        "name": "annotation_campaigns",
        "method": "POST",
        "path": "/annotation/campaigns",
        "operation": None,
        "paid": True,
        "surface": "container",
        "description": "Estimate or start a campaign over sealed traces.",
    },
)


def annotation_stream_descriptor(job_id: str) -> dict[str, Any]:
    return {
        "schema": ANNOTATION_STREAM_SCHEMA,
        "id": f"annotation:{job_id}",
        "transports": {
            "poll": {"url": f"/annotation-jobs/{job_id}/events"},
            "sse": {"url": f"/annotation-jobs/{job_id}/stream"},
            "websocket": None,
        },
        "cursor": {"kind": "sequence", "producer_kind": "annotation"},
        "event_kinds": list(ANNOTATION_EVENT_KINDS),
        "auth": {"mode": "none"},
        "retention": "run",
        "hidden_cot": False,
        "note": "Tool events name the inspection tool and arguments (selectors/ids), never model chain-of-thought.",
    }


def annotation_api_catalog(*, operations: list[dict[str, Any]] | None = None, guidance: str = "") -> dict[str, Any]:
    return {
        "schema": ANNOTATION_API_SCHEMA,
        "stream_schema": ANNOTATION_STREAM_SCHEMA,
        "event_kinds": list(ANNOTATION_EVENT_KINDS),
        "runner_kinds": ("deterministic", "model_api", "codex_app_server", "jesterky"),
        "request_must_bear": ("model", "reasoning_effort", "runner_kind"),
        "rewrites_reward_signal": False,
        "hidden_cot": False,
        "guidance": guidance,
        "operations": operations or [],
        "endpoints": [dict(item) for item in ANNOTATION_ENDPOINTS],
    }


__all__ = [
    "ANNOTATION_API_SCHEMA",
    "ANNOTATION_ENDPOINTS",
    "ANNOTATION_EVENT_KINDS",
    "ANNOTATION_STREAM_SCHEMA",
    "annotation_api_catalog",
    "annotation_stream_descriptor",
]
