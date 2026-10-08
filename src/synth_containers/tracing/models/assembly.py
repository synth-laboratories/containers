"""Typed, source-anchored hierarchy over immutable trace annotations.

Assemblies are derived evidence: they do not insert virtual spans into the
execution trace or change the trace's content digest.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from synth_containers.serde import JsonDataclassMixin

from ..canonical import content_digest
from .evidence import TraceRefV5
from .selectors import TraceSelectorV1


ASSEMBLY_SCHEMA_VERSION = "synth.annotation-assembly.v1"


class AssemblyKind(StrEnum):
    SKILL_ATTEMPT = "skill_attempt"
    OPTION_INSTANCE = "option_instance"


class AssemblyRelation(StrEnum):
    CONTAINS = "contains"
    PRECEDES = "precedes"
    INTERRUPTS = "interrupts"
    RESUMES = "resumes"


@dataclass(frozen=True, slots=True)
class AssemblyNodeV1(JsonDataclassMixin):
    """A temporally extended skill or option, supported by one annotation."""

    node_id: str
    kind: AssemblyKind | str
    label: str
    annotation_id: str
    session_id: str
    call_span_ids: tuple[str, ...]
    start: TraceSelectorV1
    end: TraceSelectorV1
    evidence: tuple[TraceSelectorV1, ...] = ()
    member_annotation_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AssemblyEdgeV1(JsonDataclassMixin):
    source_node_id: str
    target_node_id: str
    relation: AssemblyRelation | str


@dataclass(frozen=True, slots=True)
class AnnotationAssemblyGraphV1(JsonDataclassMixin):
    """Content-addressed graph whose nodes cite source annotations and calls."""

    graph_id: str
    trace_ref: TraceRefV5
    source_evidence_digest: str
    created_at: str
    nodes: tuple[AssemblyNodeV1, ...]
    edges: tuple[AssemblyEdgeV1, ...] = ()
    schema_version: str = ASSEMBLY_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    content_digest: str = ""

    def sealed(self) -> "AnnotationAssemblyGraphV1":
        return replace(self, content_digest=content_digest(self))


__all__ = [
    "ASSEMBLY_SCHEMA_VERSION", "AssemblyKind", "AssemblyRelation",
    "AssemblyNodeV1", "AssemblyEdgeV1", "AnnotationAssemblyGraphV1",
]
