"""Registry of sealed annotator definitions and the programs that execute them.

A *definition* (``TraceAnnotatorDefinitionV1``) says what an annotator may claim.
A *program* says how it is executed: a deterministic Python callable, or a prompt
run by a Codex app-server task with a bounded tool contract. Both are sealed and
their digests are part of every job's idempotency key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Optional

from synth_containers.serde import JsonDataclassMixin

from ..canonical import content_digest, seal_record
from ..models.document import TraceDocumentV5
from ..models.standards import RubricDefinitionV2, TraceAnnotatorDefinitionV1


ANNOTATOR_PROGRAM_SCHEMA_VERSION = "synth.annotator-program.v1"


class RunnerKind(StrEnum):
    """How an annotator executes. Each kind is also a scheduling class."""

    DETERMINISTIC = "deterministic"  # in-process Python; CPU-bound; free
    MODEL_API = "model_api"  # one structured completion, no tools; paid; I/O-bound
    CODEX_APP_SERVER = "codex_app_server"  # agentic task with trace tools; paid; I/O-bound
    JESTERKY = "jesterky"  # jesterky map/reduce swarm of actors; paid; I/O-bound


DeterministicProgram = Callable[[TraceDocumentV5, "ProgramContext"], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ProgramContext:
    """What a deterministic program may see besides the sealed trace."""

    definition: TraceAnnotatorDefinitionV1
    rubric: RubricDefinitionV2 | None
    parameters: dict[str, Any]
    source_annotations: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class AnnotatorProgramV1(JsonDataclassMixin):
    """Sealed description of how an annotator runs.

    ``prompt`` is the full instruction text handed to a Codex app-server task.
    ``program_ref`` names a deterministic callable for auditability; the callable
    itself is registered separately because code is not content-addressed here.
    """

    program_id: str
    runner_kind: RunnerKind | str
    version: str = "v1"
    prompt: str = ""
    program_ref: str | None = None
    tool_names: tuple[str, ...] = ()
    output_schema_id: str = "synth.annotation-proposal.v1"
    parameters: dict[str, Any] = field(default_factory=dict)
    paid: bool = False
    schema_version: str = ANNOTATOR_PROGRAM_SCHEMA_VERSION
    content_digest: str = ""

    def sealed(self) -> "AnnotatorProgramV1":
        return seal_record(self)


@dataclass(frozen=True, slots=True)
class RegisteredAnnotator:
    definition: TraceAnnotatorDefinitionV1
    program: AnnotatorProgramV1
    rubric: RubricDefinitionV2 | None = None
    domain: str = ""
    deterministic_program: DeterministicProgram | None = None
    description: str = ""
    requires_rubric: bool = False

    @property
    def annotator_id(self) -> str:
        return self.definition.annotator_id

    @property
    def paid(self) -> bool:
        return bool(self.program.paid)

    def compatible_with(self, trace_schema: str) -> bool:
        return trace_schema in self.definition.supported_trace_schemas


class DefinitionRegistry:
    """In-process catalogue of annotators a service is willing to run."""

    def __init__(self) -> None:
        self._by_id: dict[str, RegisteredAnnotator] = {}
        self._rubrics: dict[str, RubricDefinitionV2] = {}

    def register(
        self,
        definition: TraceAnnotatorDefinitionV1,
        program: AnnotatorProgramV1,
        *,
        rubric: RubricDefinitionV2 | None = None,
        domain: str = "",
        deterministic_program: DeterministicProgram | None = None,
        description: str = "",
        requires_rubric: bool = False,
    ) -> RegisteredAnnotator:
        if not definition.content_digest or content_digest(definition) != definition.content_digest:
            raise ValueError("annotator definition must be sealed before registration")
        if not program.content_digest or content_digest(program) != program.content_digest:
            raise ValueError("annotator program must be sealed before registration")
        if str(program.runner_kind) == RunnerKind.DETERMINISTIC and deterministic_program is None:
            raise ValueError("deterministic programs need a callable")
        if str(program.runner_kind) in {RunnerKind.CODEX_APP_SERVER, RunnerKind.MODEL_API, RunnerKind.JESTERKY} and not program.prompt.strip():
            raise ValueError(f"{program.runner_kind} programs need a prompt")
        if definition.annotator_id in self._by_id:
            raise ValueError(f"annotator already registered: {definition.annotator_id}")
        if rubric is not None:
            if not rubric.content_digest or content_digest(rubric) != rubric.content_digest:
                raise ValueError("rubric must be sealed before registration")
            self._rubrics.setdefault(rubric.rubric_id, rubric)
        entry = RegisteredAnnotator(
            definition=definition,
            program=program,
            rubric=rubric,
            domain=domain,
            deterministic_program=deterministic_program,
            description=description or definition.purpose,
            requires_rubric=requires_rubric,
        )
        self._by_id[definition.annotator_id] = entry
        return entry

    def register_rubric(self, rubric: RubricDefinitionV2) -> None:
        if not rubric.content_digest or content_digest(rubric) != rubric.content_digest:
            raise ValueError("rubric must be sealed before registration")
        self._rubrics[rubric.rubric_id] = rubric

    def get(self, annotator_id: str) -> Optional[RegisteredAnnotator]:
        return self._by_id.get(annotator_id)

    def require(self, annotator_id: str, *, digest: str | None = None) -> RegisteredAnnotator:
        entry = self._by_id.get(annotator_id)
        if entry is None:
            raise KeyError(annotator_id)
        if digest is not None and entry.definition.content_digest != digest:
            raise ValueError(
                f"annotator {annotator_id} digest {entry.definition.content_digest} "
                f"does not match requested {digest}"
            )
        return entry

    def rubric(self, rubric_id: str, *, digest: str | None = None) -> RubricDefinitionV2 | None:
        rubric = self._rubrics.get(rubric_id)
        if rubric is None:
            return None
        if digest is not None and rubric.content_digest != digest:
            raise ValueError(f"rubric {rubric_id} digest does not match requested {digest}")
        return rubric

    def rubrics(self) -> tuple[RubricDefinitionV2, ...]:
        return tuple(self._rubrics[key] for key in sorted(self._rubrics))

    def list(
        self,
        *,
        trace_schema: str | None = None,
        domain: str | None = None,
    ) -> tuple[RegisteredAnnotator, ...]:
        found: list[RegisteredAnnotator] = []
        for key in sorted(self._by_id):
            entry = self._by_id[key]
            if trace_schema is not None and not entry.compatible_with(trace_schema):
                continue
            if domain is not None and entry.domain != domain:
                continue
            found.append(entry)
        return tuple(found)

    def describe(self, entry: RegisteredAnnotator) -> dict[str, Any]:
        return {
            "annotator_id": entry.annotator_id,
            "annotator_version": entry.definition.version,
            "annotator_digest": entry.definition.content_digest,
            "name": entry.definition.name,
            "purpose": entry.definition.purpose,
            "domain": entry.domain,
            "runner_kind": str(entry.program.runner_kind),
            "paid": entry.paid,
            "program_digest": entry.program.content_digest,
            "program_version": entry.program.version,
            "taxonomy": list(entry.definition.taxonomy),
            "supported_trace_schemas": list(entry.definition.supported_trace_schemas),
            "required_subject_scope": entry.definition.required_subject_scope,
            "grounding_requirement": str(entry.definition.grounding_requirement),
            "minimum_evidence": entry.definition.minimum_evidence,
            "unavailable_evidence_behavior": str(
                entry.definition.unavailable_evidence_behavior
            ),
            "rubric_id": entry.rubric.rubric_id if entry.rubric else None,
            "rubric_digest": entry.rubric.content_digest if entry.rubric else None,
            "requires_rubric": entry.requires_rubric,
            "tool_names": list(entry.program.tool_names),
            "model": entry.definition.model,
        }


__all__ = [
    "ANNOTATOR_PROGRAM_SCHEMA_VERSION",
    "AnnotatorProgramV1",
    "DefinitionRegistry",
    "DeterministicProgram",
    "ProgramContext",
    "RegisteredAnnotator",
    "RunnerKind",
]
