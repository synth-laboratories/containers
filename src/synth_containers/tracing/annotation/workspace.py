"""The small, read-only manifest an annotator task is given instead of a repo.

The workspace never contains the trace body. It names the sealed source, the
definitions in force, the tools the task may call, the schema its answer must
satisfy, and nothing else. Files are written once and made read-only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from synth_containers.serde import JsonDataclassMixin, jsonable

from ..canonical import readable_json, seal_record, utc_now
from ..models.projection import ProjectionManifestV1
from ..models.standards import RubricDefinitionV2, TraceAnnotatorDefinitionV1
from .definitions import AnnotatorProgramV1
from .jobs import AnnotationJobLimitsV1, AnnotationJobRequestV1
from .proposal import PROPOSAL_SCHEMA_VERSION, STRICT_PROPOSAL_JSON_SCHEMA


WORKSPACE_MANIFEST_SCHEMA_VERSION = "synth.annotation-workspace.v1"


@dataclass(frozen=True, slots=True)
class WorkspaceProjectionV1(JsonDataclassMixin):
    projection_id: str
    projection_digest: str
    manifest_digest: str
    format: str
    losses: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class AnnotationWorkspaceManifestV1(JsonDataclassMixin):
    job_id: str
    source_trace_id: str
    source_trace_digest: str
    source_trace_schema: str
    annotator_id: str
    annotator_version: str
    annotator_digest: str
    program_id: str
    program_digest: str
    runner_kind: str
    tool_contract_digest: str
    tool_names: tuple[str, ...]
    output_schema_id: str
    mode: str
    created_at: str
    rubric_id: str | None = None
    rubric_digest: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    projections: tuple[WorkspaceProjectionV1, ...] = ()
    limits: AnnotationJobLimitsV1 = field(default_factory=AnnotationJobLimitsV1)
    source_annotation_ids: tuple[str, ...] = ()
    scope_session_ids: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    reasoning_policy: str = "not_captured"
    schema_version: str = WORKSPACE_MANIFEST_SCHEMA_VERSION
    content_digest: str = ""

    def sealed(self) -> "AnnotationWorkspaceManifestV1":
        return seal_record(self)


def build_workspace_manifest(
    *,
    job_id: str,
    request: AnnotationJobRequestV1,
    trace_schema: str,
    definition: TraceAnnotatorDefinitionV1,
    program: AnnotatorProgramV1,
    tool_contract_digest: str,
    tool_names: tuple[str, ...],
    rubric: RubricDefinitionV2 | None,
    projections: tuple[ProjectionManifestV1, ...] = (),
) -> AnnotationWorkspaceManifestV1:
    return AnnotationWorkspaceManifestV1(
        job_id=job_id,
        source_trace_id=request.source_trace_id,
        source_trace_digest=request.source_trace_digest,
        source_trace_schema=trace_schema,
        annotator_id=definition.annotator_id,
        annotator_version=definition.version,
        annotator_digest=definition.content_digest,
        program_id=program.program_id,
        program_digest=program.content_digest,
        runner_kind=str(program.runner_kind),
        tool_contract_digest=tool_contract_digest,
        tool_names=tuple(tool_names),
        output_schema_id=PROPOSAL_SCHEMA_VERSION,
        mode=str(request.mode),
        created_at=utc_now(),
        rubric_id=rubric.rubric_id if rubric else None,
        rubric_digest=rubric.content_digest if rubric else None,
        model=request.model,
        reasoning_effort=request.reasoning_effort,
        projections=tuple(
            WorkspaceProjectionV1(
                projection_id=item.projection_id,
                projection_digest=item.target_digest or "",
                manifest_digest=item.content_digest,
                format=item.format,
                losses=tuple(jsonable(loss) for loss in item.losses),
            )
            for item in projections
        ),
        limits=request.limits,
        source_annotation_ids=tuple(request.source_annotation_ids),
        scope_session_ids=tuple(item.split(":", 1)[1] for item in request.target_selector_ids if item.startswith("session:")),
        files=(
            "manifest.json",
            "INSTRUCTIONS.md",
            "annotator_definition.json",
            "output_schema.json",
            "tool_contract.json",
        )
        + (("rubric.json",) if rubric else ()),
    ).sealed()


def render_instructions(
    manifest: AnnotationWorkspaceManifestV1,
    *,
    program: AnnotatorProgramV1,
    definition: TraceAnnotatorDefinitionV1,
    rubric: RubricDefinitionV2 | None,
    source_annotations: tuple[dict[str, Any], ...] = (),
) -> str:
    """The exact text a Codex task receives. Sealed via the program digest + manifest digest."""

    lines: list[str] = []
    lines.append(f"# Trace V5 annotation task — {definition.name}")
    lines.append("")
    lines.append(program.prompt.strip())
    lines.append("")
    lines.append("## Non-negotiable rules")
    lines.append("")
    lines.append(
        "- You are inspecting one immutable, sealed Trace V5. You cannot change it, and "
        "nothing you say changes the recorded run, its reward, or its engine state."
    )
    lines.append(
        "- Use only the trace inspection tools you were given. There is no shell, no "
        "network, and no file access. Do not attempt them."
    )
    lines.append(
        "- Every applied finding must cite evidence selectors you verified with "
        "`trace_resolve_selector`. Uncited or unresolvable findings are discarded."
    )
    lines.append(
        f"- Cite at least {definition.minimum_evidence} distinct evidence selector(s) per "
        "finding. Prefer `part` or `message` selectors with a `quote` for text, `span` "
        "selectors for environment steps, and `event` selectors for engine facts."
    )
    lines.append(
        "- A `quote` must be a verbatim substring of the entity text exactly as the tools "
        "render it (`trace_get_message` / `trace_get_event` / `trace_resolve_selector`): copy "
        "it character for character, including spacing and punctuation. Quote short scalar "
        "phrases — a sentence fragment, one `key: value` pair as rendered — and never re-type, "
        "reflow, or summarise JSON (`{\"actions\": …}` retyped from memory is a `quote_mismatch`). "
        "If you are not sure of the exact text, fetch the entity with `trace_get_message` / "
        "`trace_get_event` and copy from it; a selector without a `quote` is always valid when "
        "the whole part or event is the evidence."
    )
    lines.append(
        "- If the evidence you need is absent, emit an abstention with a typed reason. "
        "Never guess, never complete a finding from memory or from summaries."
    )
    lines.append(
        "- `rationale` is a concise, retained explanation (under 2000 characters). Do not "
        "include hidden or step-by-step reasoning; it is neither requested nor stored."
    )
    lines.append("- Engine facts (reward, achievements, actions, inventory, termination) are authoritative; annotations never override them.")
    lines.append("- Labels must come from the taxonomy below; unknown labels invalidate the finding.")
    lines.append("")
    lines.append("## Taxonomy")
    lines.append("")
    contract = definition.output_contract
    if contract is not None:
        lines.append(f"Annotation types: {', '.join(contract.annotation_types)}")
        lines.append("")
        for taxon in contract.taxonomy:
            parent = f" (child of {taxon.parent_label})" if taxon.parent_label else ""
            lines.append(f"- `{taxon.label}`{parent}: {taxon.description}")
        if contract.payload_schema is not None:
            lines.append("")
            lines.append("Payload fields:")
            for item in contract.payload_schema.fields:
                required = "required" if item.required else "optional"
                allowed = (
                    f" one of {list(item.allowed_values)}" if item.allowed_values else ""
                )
                lines.append(
                    f"- `{item.field_name}` ({item.value_kind}, {required}){allowed}: {item.description}"
                )
    else:
        for label in definition.taxonomy:
            lines.append(f"- `{label}`")
    if rubric is not None:
        lines.append("")
        lines.append(f"## Rubric — {rubric.name} ({rubric.rubric_id} {rubric.version})")
        lines.append("")
        if rubric.scoring_instructions:
            lines.append(rubric.scoring_instructions.strip())
            lines.append("")
        for criterion in rubric.criteria:
            lines.append(
                f"### {criterion.criterion_id} — {criterion.name} "
                f"[{criterion.min_score:g}..{criterion.max_score:g}, weight {criterion.weight:g}]"
            )
            lines.append(criterion.requirement.strip())
            if criterion.evaluator_instructions:
                lines.append("")
                lines.append(criterion.evaluator_instructions.strip())
            if criterion.failure_modes:
                lines.append("")
                lines.append("Failure modes: " + "; ".join(criterion.failure_modes))
            lines.append("")
    if source_annotations:
        lines.append("## Source annotations to adjudicate")
        lines.append("")
        lines.append(
            "These are immutable records produced by independent annotators. Cite their "
            "`annotation_id` values in `source_annotation_ids`; never restate them as your own."
        )
        lines.append("")
        lines.append("```json")
        lines.append(readable_json(list(source_annotations)))
        lines.append("```")
        lines.append("")
    if manifest.scope_session_ids:
        lines.append("## Scope")
        lines.append("")
        lines.append(
            "Annotate ONLY these sessions (lanes); pass `session_id` to `trace_list_entities` and "
            "ignore entities from other sessions: " + ", ".join(f"`{item}`" for item in manifest.scope_session_ids)
        )
        lines.append("")
    lines.append("## Output")
    lines.append("")
    if manifest.mode == "verify":
        lines.append(
            "This is a **verification** job. Populate the `judgments` array (one object per "
            "rubric criterion). `findings` and `abstentions` may be empty. Do not emit "
            "classification findings in place of judgments — that seals an empty verifier result."
        )
        lines.append("")
    elif manifest.mode == "adjudicate":
        lines.append(
            "This is an **adjudication** job. Each finding must name `source_annotation_ids` "
            "from the source list above."
        )
        lines.append("")
    lines.append(
        f"Reply with a single JSON object conforming to `{manifest.output_schema_id}` "
        "(see output_schema.json). Set `source_trace_id` and `source_trace_digest` to:"
    )
    lines.append("")
    lines.append(f"- source_trace_id: `{manifest.source_trace_id}`")
    lines.append(f"- source_trace_digest: `{manifest.source_trace_digest}`")
    lines.append("")
    lines.append(
        "Strict output format: every field is present (use null for unknown optionals); "
        "`payload_json` is the payload object JSON-encoded as a string; `rationale` stays under 2000 characters."
    )
    lines.append("")
    lines.append("## Limits")
    lines.append("")
    lines.append(f"- tool calls: {manifest.limits.max_tool_calls}")
    lines.append(f"- bytes per tool response: {manifest.limits.max_tool_response_bytes}")
    lines.append(f"- cumulative tool bytes: {manifest.limits.max_total_tool_bytes}")
    if manifest.limits.max_total_tokens is not None:
        lines.append(f"- total tokens: {manifest.limits.max_total_tokens}")
    lines.append("")
    return "\n".join(lines) + "\n"


def materialize_workspace(
    root: Path,
    manifest: AnnotationWorkspaceManifestV1,
    *,
    instructions: str,
    definition: TraceAnnotatorDefinitionV1,
    rubric: RubricDefinitionV2 | None,
    tool_specs: list[dict[str, Any]],
) -> Path:
    """Write the workspace once; files become read-only, the directory stays listable."""

    root.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {
        "manifest.json": readable_json(manifest),
        "INSTRUCTIONS.md": instructions,
        "annotator_definition.json": readable_json(definition),
        "output_schema.json": readable_json(STRICT_PROPOSAL_JSON_SCHEMA),
        "tool_contract.json": readable_json(
            {"tool_contract_digest": manifest.tool_contract_digest, "tools": tool_specs}
        ),
    }
    if rubric is not None:
        files["rubric.json"] = readable_json(rubric)
    for name, text in files.items():
        path = root / name
        if path.exists():
            path.chmod(0o644)
        path.write_text(text, encoding="utf-8")
        path.chmod(0o444)
    os.chmod(root, 0o555)
    return root


def unlock_workspace(root: Path) -> None:
    if root.exists():
        os.chmod(root, 0o755)


__all__ = [
    "WORKSPACE_MANIFEST_SCHEMA_VERSION",
    "AnnotationWorkspaceManifestV1",
    "WorkspaceProjectionV1",
    "build_workspace_manifest",
    "materialize_workspace",
    "render_instructions",
    "unlock_workspace",
]
