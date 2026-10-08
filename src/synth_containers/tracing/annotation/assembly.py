"""Fail-closed validation for a derived annotation assembly graph."""

from __future__ import annotations

from ..models.assembly import (
    ASSEMBLY_SCHEMA_VERSION,
    AnnotationAssemblyGraphV1,
    AssemblyKind,
    AssemblyRelation,
)
from ..models.document import TraceDocumentV5
from ..models.evidence import TraceEvidenceBundleV5
from ..models.selectors import SelectorKind, resolve_selector
from ..models.spans import SpanKind


def validate_assembly_graph(
    graph: AnnotationAssemblyGraphV1,
    document: TraceDocumentV5,
    evidence: TraceEvidenceBundleV5,
) -> None:
    """Reject dangling, cross-lane, unordered, or unsupported assembly links."""

    if graph.schema_version != ASSEMBLY_SCHEMA_VERSION:
        raise ValueError("unsupported assembly schema")
    if (graph.trace_ref.trace_id, graph.trace_ref.content_digest) != (
        document.trace_id, document.content_digest
    ):
        raise ValueError("assembly source trace mismatch")
    if graph.source_evidence_digest != evidence.content_digest:
        raise ValueError("assembly source evidence mismatch")
    if (evidence.trace_ref.trace_id, evidence.trace_ref.content_digest) != (
        document.trace_id, document.content_digest
    ):
        raise ValueError("evidence belongs to another trace")

    annotations = {item.annotation_id: item for item in evidence.annotations}
    nodes = {item.node_id: item for item in graph.nodes}
    if len(nodes) != len(graph.nodes):
        raise ValueError("duplicate assembly node id")
    calls: dict[str, tuple[str, int]] = {}
    for span in document.spans:
        if str(span.span_kind) != SpanKind.MODEL_CALL:
            continue
        index = span.detail.get("call_index")
        if isinstance(index, int) and not isinstance(index, bool):
            calls[span.span_id] = (span.session_id, index)

    intervals: dict[str, tuple[int, int]] = {}
    for node in graph.nodes:
        if str(node.kind) not in {item.value for item in AssemblyKind}:
            raise ValueError(f"unknown assembly kind: {node.kind}")
        annotation = annotations.get(node.annotation_id)
        if annotation is None or str(annotation.status or "applied") != "applied":
            raise ValueError(f"missing applied source annotation: {node.annotation_id}")
        if annotation.annotation_type != str(node.kind):
            raise ValueError(f"assembly annotation type mismatch: {node.node_id}")
        if node.label not in annotation.labels:
            raise ValueError(f"assembly label absent from annotation: {node.node_id}")
        if len(node.member_annotation_ids) != len(set(node.member_annotation_ids)):
            raise ValueError(f"duplicate member annotation link: {node.node_id}")
        if any(member_id not in annotations for member_id in node.member_annotation_ids):
            raise ValueError(f"missing member annotation link: {node.node_id}")
        if str(annotation.target.kind) != SelectorKind.SESSION or annotation.target.entity_id != node.session_id:
            raise ValueError(f"assembly annotation session mismatch: {node.node_id}")
        if not node.call_span_ids or len(node.call_span_ids) != len(set(node.call_span_ids)):
            raise ValueError(f"empty or duplicate call membership: {node.node_id}")
        if str(node.kind) == AssemblyKind.OPTION_INSTANCE and len(node.call_span_ids) < 2:
            raise ValueError(f"option must span multiple model calls: {node.node_id}")
        members = [calls.get(span_id) for span_id in node.call_span_ids]
        if any(item is None or item[0] != node.session_id for item in members):
            raise ValueError(f"assembly references missing or cross-session model call: {node.node_id}")
        indices = [item[1] for item in members if item is not None]
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError(f"assembly model calls must be ordered and contiguous: {node.node_id}")
        if (annotation.payload.get("start_call"), annotation.payload.get("end_call")) != (
            indices[0], indices[-1]
        ):
            raise ValueError(f"assembly call range disagrees with annotation: {node.node_id}")
        for member_id in node.member_annotation_ids:
            member = annotations[member_id]
            message = document.message(member.target.entity_id or "") if str(member.target.kind) == SelectorKind.MESSAGE else None
            if message is None or message.session_id != node.session_id or message.produced_by_span_id not in node.call_span_ids:
                raise ValueError(f"assembly member annotation is outside model calls: {node.node_id}")
        intervals[node.node_id] = (indices[0], indices[-1])
        if (str(node.start.kind), node.start.entity_id) != (SelectorKind.SPAN, node.call_span_ids[0]):
            raise ValueError(f"start selector does not match first call: {node.node_id}")
        if (str(node.end.kind), node.end.entity_id) != (SelectorKind.SPAN, node.call_span_ids[-1]):
            raise ValueError(f"end selector does not match last call: {node.node_id}")
        for selector in (node.start, node.end, *node.evidence):
            if not resolve_selector(document, selector).resolved:
                raise ValueError(f"unresolved assembly selector: {node.node_id}")

    seen_edges: set[tuple[str, str, str]] = set()
    contains: dict[str, set[str]] = {}
    for edge in graph.edges:
        relation = str(edge.relation)
        key = (edge.source_node_id, edge.target_node_id, relation)
        if relation not in {item.value for item in AssemblyRelation}:
            raise ValueError(f"unknown assembly relation: {relation}")
        if key in seen_edges or edge.source_node_id == edge.target_node_id:
            raise ValueError("duplicate or self assembly edge")
        seen_edges.add(key)
        source = nodes.get(edge.source_node_id)
        target = nodes.get(edge.target_node_id)
        if source is None or target is None or source.session_id != target.session_id:
            raise ValueError("dangling or cross-session assembly edge")
        if relation == AssemblyRelation.CONTAINS:
            if str(source.kind) != AssemblyKind.OPTION_INSTANCE or str(target.kind) != AssemblyKind.SKILL_ATTEMPT:
                raise ValueError("contains must link option to skill")
            if not set(target.call_span_ids).issubset(source.call_span_ids):
                raise ValueError("contained node extends beyond parent calls")
            contains.setdefault(source.node_id, set()).add(target.node_id)
        if relation in {AssemblyRelation.PRECEDES, AssemblyRelation.INTERRUPTS, AssemblyRelation.RESUMES}:
            if str(source.kind) != str(target.kind) or intervals[source.node_id][1] >= intervals[target.node_id][0]:
                raise ValueError(f"{relation} edge contradicts kind or call order")

    def visit(node_id: str, path: set[str]) -> None:
        if node_id in path:
            raise ValueError("assembly containment cycle")
        for child in contains.get(node_id, ()):
            visit(child, path | {node_id})

    for node_id in contains:
        visit(node_id, set())


__all__ = ["validate_assembly_graph"]
