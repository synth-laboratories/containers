"""Incremental and post-seal reducers for ``synth.trace-visual.v1``."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..capture.envelope import RawCaptureEnvelopeV1
from ..models.document import TraceDocumentV5
from ..models.evidence import TraceEvidenceBundleV5
from ..models.selectors import SelectorKind, TraceSelectorV1, selector_for
from ..models.visual import (
    TraceVisualItemV1,
    TraceVisualLaneV1,
    TraceVisualProjectionV1,
    TraceVisualState,
)


def visual_from_raw(
    envelopes: Iterable[RawCaptureEnvelopeV1],
    *,
    trace_id: str | None = None,
    run_id: str | None = None,
    task_id: str | None = None,
    visibility_ceiling: str = "private",
) -> TraceVisualProjectionV1:
    """Reduce exact raw facts without claiming sealed V5 identity."""

    records = tuple(sorted(envelopes, key=lambda item: item.ordinal))
    capture_ids = {item.capture_id for item in records}
    if len(capture_ids) != 1:
        raise ValueError("a visual projection must contain exactly one capture")
    capture_id = next(iter(capture_ids))
    lane_pairs = {
        (item.actor_id, item.session_id)
        for item in records
        if _visible(_raw_visibility(item), visibility_ceiling)
    }
    lanes = tuple(
        TraceVisualLaneV1(
            lane_id=_lane_id(actor_id, session_id),
            actor_id=actor_id,
            session_id=session_id,
            display_name=actor_id,
            actor_kind="unknown",
            visibility="private",
        )
        for actor_id, session_id in sorted(lane_pairs)
    )
    items = tuple(
        TraceVisualItemV1(
            item_id=item.envelope_id,
            kind=str(item.record_type),
            occurred_at=item.occurred_at,
            title=_raw_title(item),
            actor_id=item.actor_id,
            session_id=item.session_id,
            lane_id=_lane_id(item.actor_id, item.session_id),
            sequence=item.ordinal,
            source_envelope_id=item.envelope_id,
            source_ordinal=item.ordinal,
            source_digest=item.content_digest,
            visibility=_raw_visibility(item),
            detail=dict(item.payload),
        )
        for item in records
        if _visible(_raw_visibility(item), visibility_ceiling)
    )
    omitted = len(records) - len(items)
    return TraceVisualProjectionV1(
        capture_id=capture_id,
        trace_id=trace_id,
        run_id=run_id,
        task_id=task_id,
        state=TraceVisualState.PROVISIONAL,
        high_water_ordinal=records[-1].ordinal if records else -1,
        lanes=lanes,
        items=items,
        visibility_ceiling=visibility_ceiling,
        summary={
            "record_count": len(records),
            "visible_item_count": len(items),
            "lane_count": len(lanes),
        },
        losses=((f"visibility_filtered:{omitted}",) if omitted else ()),
    ).sealed()


def visual_from_sealed(
    document: TraceDocumentV5,
    evidence: TraceEvidenceBundleV5 | None = None,
    *,
    visibility_ceiling: str = "private",
) -> TraceVisualProjectionV1:
    """Build the shared visual packet from sealed trace and evidence authority."""

    if not document.content_digest:
        raise ValueError("sealed visual projection requires a sealed trace")
    actor_by_id = {item.actor_id: item for item in document.actors}
    active_session_ids = {
        *(item.session_id for item in document.events),
        *(item.session_id for item in document.spans),
        *(item.session_id for item in document.messages),
    }
    if not active_session_ids:
        active_session_ids = {item.session_id for item in document.sessions}
    lanes = tuple(
        TraceVisualLaneV1(
            lane_id=_lane_id(session.actor_id, session.session_id),
            actor_id=session.actor_id,
            session_id=session.session_id,
            display_name=(
                actor_by_id[session.actor_id].display_name
                if session.actor_id in actor_by_id
                else session.actor_id
            ),
            actor_kind=(
                str(actor_by_id[session.actor_id].kind)
                if session.actor_id in actor_by_id
                else "unknown"
            ),
            role=(actor_by_id[session.actor_id].role if session.actor_id in actor_by_id else ""),
            parent_actor_id=(
                actor_by_id[session.actor_id].parent_actor_id
                if session.actor_id in actor_by_id
                else None
            ),
            visibility=(
                str(actor_by_id[session.actor_id].visibility)
                if session.actor_id in actor_by_id
                else "private"
            ),
            detail={
                "status": str(session.status),
                "coverage": session.coverage.to_dict(),
            },
        )
        for session in document.sessions
        if session.session_id in active_session_ids
        if _visible(
            str(actor_by_id[session.actor_id].visibility)
            if session.actor_id in actor_by_id
            else "private",
            visibility_ceiling,
        )
    )
    items: list[TraceVisualItemV1] = []
    for event in document.events:
        visibility = _entity_visibility(
            actor_by_id,
            event.actor_id,
        )
        if not _visible(visibility, visibility_ceiling):
            continue
        items.append(
            TraceVisualItemV1(
                item_id=event.event_id,
                kind=str(event.event_type),
                occurred_at=event.occurred_at,
                title=str(event.event_type),
                actor_id=event.actor_id,
                session_id=event.session_id,
                lane_id=_lane_id(event.actor_id, event.session_id),
                sequence=event.order.chronological_sequence,
                status=str(event.status),
                source_envelope_id=event.raw_source_ref,
                source_selector=selector_for(
                    document,
                    kind=SelectorKind.EVENT,
                    entity_id=event.event_id,
                ),
                source_digest=event.content_digest,
                visibility=visibility,
                detail=dict(event.payload),
            )
        )
    for span in document.spans:
        visibility = _entity_visibility(actor_by_id, span.actor_id)
        if not _visible(visibility, visibility_ceiling):
            continue
        sequence = min(
            (
                item.order.chronological_sequence
                for item in document.events
                if item.span_id == span.span_id and item.order.chronological_sequence is not None
            ),
            default=None,
        )
        items.append(
            TraceVisualItemV1(
                item_id=span.span_id,
                kind=f"span.{span.span_kind}",
                occurred_at=span.started_at,
                title=str(span.span_kind),
                actor_id=span.actor_id,
                session_id=span.session_id,
                lane_id=_lane_id(span.actor_id, span.session_id),
                sequence=sequence,
                status=str(span.status),
                source_selector=selector_for(
                    document,
                    kind=SelectorKind.SPAN,
                    entity_id=span.span_id,
                ),
                source_digest=span.content_digest,
                visibility=visibility,
                detail={
                    **span.detail,
                    "ended_at": span.ended_at,
                    "input_message_ids": list(span.input_message_ids),
                    "output_message_ids": list(span.output_message_ids),
                    "artifact_ids": list(span.artifact_ids),
                    "usage": span.usage.to_dict() if span.usage is not None else None,
                },
            )
        )
    _append_coordination_items(items, document, visibility_ceiling)
    if evidence is not None:
        if evidence.trace_ref.content_digest != document.content_digest:
            raise ValueError("evidence visual source does not match the sealed trace")
        _append_evidence_items(items, evidence, visibility_ceiling)
    items.sort(
        key=lambda item: (
            item.sequence is None,
            item.sequence if item.sequence is not None else 0,
            item.occurred_at,
            item.item_id,
        )
    )
    omitted_actor_count = len(document.sessions) - len(lanes)
    summary: dict[str, Any] = {
        "actor_count": len(document.actors),
        "session_count": len(document.sessions),
        "event_count": len(document.events),
        "span_count": len(document.spans),
        "artifact_count": len(document.artifacts),
        "visual_item_count": len(items),
    }
    craftax = document.extensions.get("craftax")
    if isinstance(craftax, dict):
        summary["craftax"] = craftax
    return TraceVisualProjectionV1(
        capture_id=document.capture.capture_id,
        trace_id=document.trace_id,
        trace_digest=document.content_digest,
        run_id=document.identity.run_id,
        task_id=document.identity.task_id,
        state=TraceVisualState.SEALED,
        high_water_ordinal=document.capture.raw_record_count - 1,
        lanes=lanes,
        items=tuple(items),
        visibility_ceiling=visibility_ceiling,
        usage=document.usage.to_dict(),
        summary=summary,
        losses=(
            (f"visibility_filtered_sessions:{omitted_actor_count}",) if omitted_actor_count else ()
        ),
    ).sealed()


def _append_coordination_items(
    items: list[TraceVisualItemV1],
    document: TraceDocumentV5,
    visibility_ceiling: str,
) -> None:
    graph = document.coordination
    if graph is None:
        return
    actor_by_id = {item.actor_id: item for item in document.actors}
    if _visible("private", visibility_ceiling):
        for group in graph.actor_groups:
            items.append(
                _sealed_item(
                    document,
                    item_id=group.group_id,
                    kind="coordination.actor_group",
                    occurred_at=group.formed_at or document.lifecycle.started_at,
                    title=group.display_name,
                    sequence=group.formed_sequence,
                    selector_kind=SelectorKind.ACTOR_GROUP,
                    digest=group.content_digest,
                    detail={
                        "kind": str(group.kind),
                        "member_actor_ids": list(group.member_actor_ids),
                        "leader_actor_ids": list(group.leader_actor_ids),
                    },
                )
            )
        for edge in graph.interaction_edges:
            items.append(
                _sealed_item(
                    document,
                    item_id=edge.interaction_id,
                    kind=f"coordination.{edge.kind}",
                    occurred_at=edge.started_at,
                    title=str(edge.kind),
                    sequence=edge.started_sequence,
                    selector_kind=SelectorKind.INTERACTION,
                    digest=edge.content_digest,
                    status=str(edge.status),
                    detail={
                        "source": edge.source.to_dict(),
                        "target": edge.target.to_dict(),
                        "message_ids": list(edge.carried_message_ids),
                        "artifact_ids": list(edge.carried_artifact_ids),
                    },
                )
            )
    for epoch in graph.context_epochs:
        visibility = _entity_visibility(actor_by_id, epoch.actor_id)
        if not _visible(visibility, visibility_ceiling):
            continue
        item = _sealed_item(
            document,
            item_id=epoch.context_epoch_id,
            kind="coordination.context_epoch",
            occurred_at=epoch.started_at,
            title="context epoch",
            sequence=epoch.started_sequence,
            selector_kind=SelectorKind.CONTEXT_EPOCH,
            digest=epoch.content_digest,
            actor_id=epoch.actor_id,
            session_id=epoch.session_id,
            visibility=visibility,
            detail={
                "message_ids": list(epoch.model_visible_message_ids),
                "span_ids": list(epoch.model_call_span_ids),
                "losses": list(epoch.losses),
            },
        )
        items.append(item)
    if _visible("private", visibility_ceiling):
        for turn in graph.joint_turns:
            items.append(
                _sealed_item(
                    document,
                    item_id=turn.joint_turn_id,
                    kind="coordination.joint_turn",
                    occurred_at=turn.started_at,
                    title=f"joint turn {turn.environment_step}",
                    sequence=turn.started_sequence,
                    selector_kind=SelectorKind.JOINT_TURN,
                    digest=turn.content_digest,
                    status=str(turn.status),
                    detail={
                        "environment_actor_id": turn.environment_actor_id,
                        "participants": [item.to_dict() for item in turn.participants],
                        "shared_transition_event_ids": list(turn.shared_transition_event_ids),
                        "shared_reward_event_ids": list(turn.shared_reward_event_ids),
                    },
                )
            )


def _append_evidence_items(
    items: list[TraceVisualItemV1],
    evidence: TraceEvidenceBundleV5,
    visibility_ceiling: str,
) -> None:
    start = len(items)
    for result in evidence.verifier_results:
        items.append(
            _evidence_item(
                item_id=result.verifier_result_id,
                kind="evidence.verifier_result",
                occurred_at=result.produced_at,
                title=result.verdict or result.verifier_id,
                status=str(result.verification_status),
                selector=result.subject,
                digest=result.content_digest,
                detail={
                    "verifier_id": result.verifier_id,
                    "score": result.score,
                    "passed": result.passed,
                    "grounding": str(result.grounding),
                },
            )
        )
        for judgment in result.judgments:
            items.append(
                _evidence_item(
                    item_id=judgment.judgment_id
                    or (f"{result.verifier_result_id}:{judgment.criterion_id}"),
                    kind="evidence.judgment",
                    occurred_at=judgment.produced_at or result.produced_at,
                    title=judgment.verdict,
                    status=str(judgment.status or ""),
                    selector=judgment.subject or result.subject,
                    digest=judgment.content_digest,
                    detail={
                        "criterion_id": judgment.criterion_id,
                        "score": judgment.score,
                        "passed": judgment.passed,
                        "rationale": judgment.rationale,
                    },
                )
            )
    for annotation in evidence.annotations:
        items.append(
            _evidence_item(
                item_id=annotation.annotation_id,
                kind="evidence.annotation",
                occurred_at=annotation.created_at,
                title=annotation.annotation_type,
                status=str(annotation.status or ""),
                selector=annotation.target,
                digest=annotation.content_digest,
                visibility=str(annotation.visibility),
                detail={
                    "labels": list(annotation.labels),
                    "confidence": annotation.confidence,
                    "review_state": str(annotation.review_state or ""),
                },
            )
        )
    for reward in evidence.reward_records:
        items.append(
            _evidence_item(
                item_id=reward.reward_record_id,
                kind="evidence.reward",
                occurred_at=reward.produced_at,
                title=reward.reward_id,
                status=reward.validity,
                selector=reward.subject,
                digest=reward.content_digest,
                detail={
                    "value": reward.value,
                    "actor_id": reward.actor_id,
                    "session_id": reward.session_id,
                    "components": reward.components,
                },
            )
        )
    for evaluation in evidence.evaluation_results:
        items.append(
            _evidence_item(
                item_id=evaluation.evaluation_id,
                kind="evidence.evaluation",
                occurred_at=evaluation.produced_at,
                title=evaluation.benchmark or evaluation.suite or "evaluation",
                status=str(evaluation.execution_status),
                selector=evaluation.subject,
                digest=evaluation.content_digest,
                detail={
                    "aggregate_score": evaluation.aggregate_score,
                    "metrics": evaluation.objective_metrics,
                    "task_id": evaluation.task_id,
                },
            )
        )
    for verdict in evidence.benchmark_verdicts:
        items.append(
            TraceVisualItemV1(
                item_id=verdict.verdict_id,
                kind="evidence.benchmark_verdict",
                occurred_at=verdict.produced_at,
                title=verdict.decision,
                status=verdict.decision,
                source_digest=verdict.content_digest,
                detail={
                    "authority": verdict.benchmark_authority,
                    "threshold": verdict.threshold,
                    "failure_reasons": list(verdict.failure_reasons),
                },
            )
        )
    items[start:] = [
        item for item in items[start:] if _visible(item.visibility, visibility_ceiling)
    ]


def _sealed_item(
    document: TraceDocumentV5,
    *,
    item_id: str,
    kind: str,
    occurred_at: str,
    title: str,
    sequence: int | None,
    selector_kind: SelectorKind,
    digest: str,
    status: str = "",
    actor_id: str | None = None,
    session_id: str | None = None,
    visibility: str = "private",
    detail: dict[str, Any] | None = None,
) -> TraceVisualItemV1:
    return TraceVisualItemV1(
        item_id=item_id,
        kind=kind,
        occurred_at=occurred_at,
        title=title,
        sequence=sequence,
        actor_id=actor_id,
        session_id=session_id,
        lane_id=(_lane_id(actor_id, session_id) if actor_id and session_id else None),
        status=status,
        source_selector=selector_for(
            document,
            kind=selector_kind,
            entity_id=item_id,
        ),
        source_digest=digest,
        visibility=visibility,
        detail=dict(detail or {}),
    )


def _evidence_item(
    *,
    item_id: str,
    kind: str,
    occurred_at: str,
    title: str,
    status: str,
    selector: TraceSelectorV1,
    digest: str | None,
    visibility: str = "private",
    detail: dict[str, Any],
) -> TraceVisualItemV1:
    return TraceVisualItemV1(
        item_id=item_id,
        kind=kind,
        occurred_at=occurred_at,
        title=title,
        status=status,
        source_selector=selector,
        source_digest=digest,
        visibility=visibility,
        detail=detail,
    )


def _raw_title(envelope: RawCaptureEnvelopeV1) -> str:
    for key in ("event_type", "logical_name", "role", "status"):
        value = envelope.payload.get(key)
        if value:
            return str(value)
    return str(envelope.record_type)


def _raw_visibility(envelope: RawCaptureEnvelopeV1) -> str:
    return str(envelope.payload.get("visibility") or "private")


def _entity_visibility(actor_by_id, actor_id: str) -> str:
    actor = actor_by_id.get(actor_id)
    return str(actor.visibility) if actor is not None else "private"


def _visible(visibility: str, ceiling: str) -> bool:
    ranks = {"public": 0, "operator": 1, "private": 2}
    if visibility not in ranks or ceiling not in ranks:
        raise ValueError("visual visibility must be public, operator, or private")
    return ranks[visibility] <= ranks[ceiling]


def _lane_id(actor_id: str, session_id: str) -> str:
    return f"{actor_id}:{session_id}"


__all__ = ["visual_from_raw", "visual_from_sealed"]
