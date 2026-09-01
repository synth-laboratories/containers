"""A container-sealed Craftax rollout promotes to the lane-typed document under its own trace id."""

from __future__ import annotations

from pathlib import Path

from synth_containers.tracing.adapters.craftax_container import (
    container_rollout_to_native_events,
    is_container_craftax_rollout,
    promote_container_rollout,
)
from synth_containers.tracing.annotation import (
    AnnotationService,
    AnnotationStore,
    DefinitionRegistry,
    bundle_trace_loader,
    bundle_trace_refs,
    register_builtin_annotators,
)
from synth_containers.tracing.annotation.builtin import ENVIRONMENT_STEP_STATUS_ID
from synth_containers.tracing.canonical import content_digest
from synth_containers.tracing.store.bundle import LocalTraceBundle
from synth_containers.tracing.validation.rehydrate import trace_document_from_payload
from synth_containers.tracing.validation.validator import Severity, validate_evidence, validate_trace

FIXTURE = Path(__file__).parent / "fixtures" / "craftax" / "container_rollout_uniform.trace-v5.zip"


def _container_document(tmp_path: Path):
    bundle = LocalTraceBundle.extract_archive(FIXTURE, tmp_path / "bundle", require_self_contained=False)
    entry = bundle.read_manifest()["traces"][0]
    return trace_document_from_payload(bundle.read_trace(entry["trace_digest"])), tmp_path / "bundle"


def test_container_rollout_promotes_under_its_own_trace_id(tmp_path: Path) -> None:
    document, _ = _container_document(tmp_path)
    assert is_container_craftax_rollout(document) and not document.spans
    native = container_rollout_to_native_events(document)
    kinds = [event["payload"].get("kind") for event in native["events"]]
    assert kinds.count("policy.call") == 20 and kinds.count("action_applied") == 20
    promoted = promote_container_rollout(document)
    assert promoted.trace_id == document.trace_id and promoted.content_digest != document.content_digest
    assert content_digest(promoted) == promoted.content_digest
    assert len(promoted.spans_of_kind("model_call")) == 20 and len(promoted.spans_of_kind("environment_step")) == 20
    assert promoted.provenance.extra["promoted_from_container_trace_digest"] == document.content_digest
    assert not [f for f in validate_trace(promoted) if str(f.severity) == Severity.ERROR]
    assert promote_container_rollout(promoted).content_digest == promoted.content_digest  # idempotent
    assert promote_container_rollout(document).content_digest == promoted.content_digest  # deterministic


def test_promoted_container_rollout_flows_through_the_bundle_loader(tmp_path: Path) -> None:
    document, bundle_dir = _container_document(tmp_path)
    refs = bundle_trace_refs(bundle_dir, promote=promote_container_rollout)
    assert refs[0]["id"] == document.trace_id and refs[0]["sealed_digest"] == document.content_digest
    registry = DefinitionRegistry()
    register_builtin_annotators(registry)
    service = AnnotationService(store=AnnotationStore(tmp_path / "store"), registry=registry, trace_loader=bundle_trace_loader(bundle_dir, promote=promote_container_rollout))
    promoted = service.resolve_trace(refs[0]["id"], refs[0]["digest"])
    job = service.submit_and_run(service.request_for(promoted, ENVIRONMENT_STEP_STATUS_ID))
    assert str(job.state) == "sealed" and job.applied_count == 20
    head = service.evidence_head(document.trace_id)
    assert head.trace_ref.content_digest == refs[0]["digest"]
    assert not [f for f in validate_evidence(promoted, head)[0] if str(f.severity) == Severity.ERROR]


def test_llm_rollout_promotion_keeps_model_reasoning_tool_call_and_usage(tmp_path: Path) -> None:
    from synth_containers.tracing.adapters.craftax_container import policy_call_records

    fixture = Path(__file__).parent / "fixtures" / "craftax" / "container_rollout_llm.trace-v5.zip"
    bundle = LocalTraceBundle.extract_archive(fixture, tmp_path / "llm", require_self_contained=False)
    entry = bundle.read_manifest()["traces"][0]
    document = trace_document_from_payload(bundle.read_trace(entry["trace_digest"]))
    records = policy_call_records(document)
    assert set(records) == {1, 2} and records[1]["tool_arguments"].startswith('{"actions"') and records[1]["prompt_tokens"] == 632
    promoted = promote_container_rollout(document)
    assert promoted.provenance.model == "google/gemini-2.5-flash-lite" and promoted.provenance.extra["policy"]["reasoning_effort"] == "low"
    calls = promoted.spans_of_kind("model_call")
    assert len(calls) == 2 and calls[0].detail["model"] == "google/gemini-2.5-flash-lite" and calls[0].usage.prompt_tokens == 632
    reply = promoted.message(calls[0].output_message_ids[0])
    kinds = [str(p.type) for p in reply.parts]
    assert kinds[:2] == ["reasoning", "tool_call"] and "text" in kinds
    assert reply.parts[1].tool_name == "choose_actions" and '"make_wood_pickaxe"' in reply.parts[1].arguments_json
    assert reply.parts[0].text.startswith("**Planning Next Steps**")
    assert reply.text().startswith("THOUGHT: **Planning Next Steps**")  # text rendering keeps THOUGHT/ACTIONS tooling working
    assert not [f for f in validate_trace(promoted) if str(f.severity) == Severity.ERROR]
    assert promote_container_rollout(document).content_digest == promoted.content_digest


def test_choose_actions_tool_call_gets_a_synthetic_result_from_the_executed_steps(tmp_path: Path) -> None:
    """The environment steps that execute a ``choose_actions`` batch are its tool result."""

    import json

    from synth_containers.tracing.adapters.craftax_container import policy_call_steps
    from synth_containers.tracing.annotation.jobs import AnnotationJobLimitsV1
    from synth_containers.tracing.annotation.tools import TraceInspectionTools
    from synth_containers.tracing.models.selectors import SelectorKind, TraceSelectorV1, resolve_selector

    fixture = Path(__file__).parent / "fixtures" / "craftax" / "container_rollout_llm.trace-v5.zip"
    bundle = LocalTraceBundle.extract_archive(fixture, tmp_path / "llm", require_self_contained=False)
    entry = bundle.read_manifest()["traces"][0]
    document = trace_document_from_payload(bundle.read_trace(entry["trace_digest"]))
    batches = policy_call_steps(document)
    assert [(len(b["planned"]), len(b["steps"]), b["terminal"]) for b in batches.values()] == [(20, 20, None), (20, 16, "death")]
    promoted = promote_container_rollout(document)
    reply = promoted.message(promoted.spans_of_kind("model_call")[0].output_message_ids[0])
    call = next(p for p in reply.parts if str(p.type) == "tool_call")
    tool_message = promoted.message(reply.metadata["tool_result_message_id"])
    assert str(tool_message.role) == "tool" and tool_message.predecessor_message_ids == (reply.message_id,)
    assert tool_message.session_id == reply.session_id and tool_message.occurred_at > reply.occurred_at
    result = tool_message.parts[0]
    assert str(result.type) == "tool_result" and result.tool_call_id == call.tool_call_id and result.tool_name == "choose_actions"
    assert result.text == json.dumps(result.structured, sort_keys=True, separators=(",", ":"))  # quotes resolve against what tools render
    assert result.structured["planned"] == 20 and result.structured["executed"] == 20 and result.structured["terminal"] is None
    assert result.structured["steps"][0] == {"action": "make_wood_pickaxe", "reason": "needs_crafting_table", "span_id": promoted.spans_of_kind("environment_step")[0].span_id, "step_index": 1, "transition": "noop"}
    assert result.structured["outcomes"] == {"blocked": 1, "harvest": 1, "move": 9, "noop": 9}
    assert set(tool_message.metadata["step_span_ids"]) <= {s.span_id for s in promoted.spans_of_kind("environment_step")}
    # the second call emitted plain content (no tool call), so it gets no synthetic result
    second = promoted.message(promoted.spans_of_kind("model_call")[1].output_message_ids[0])
    # The second reply wrote its plan as content JSON instead of calling the tool;
    # it is promoted as the same choose_actions call (provenance recorded) and gets
    # its own synthetic result from the steps that followed it.
    assert second.metadata["tool_call_provenance"] == "content_json"
    assert "tool_result_message_id" in second.metadata and sum(str(m.role) == "tool" for m in promoted.messages) == 2
    assert not [f for f in validate_trace(promoted) if str(f.severity) in (Severity.ERROR, Severity.WARNING)]
    tools = TraceInspectionTools(promoted, limits=AnnotationJobLimitsV1())
    paired = tools.trace_get_tool_call(call.tool_call_id)
    assert paired["call"]["part_id"] == call.part_id and paired["result"]["message_id"] == tool_message.message_id
    selector = TraceSelectorV1(trace_id=promoted.trace_id, trace_digest=promoted.content_digest, kind=SelectorKind.PART, entity_id=tool_message.message_id, part_id=result.part_id).with_quote('"reason":"needs_crafting_table"')
    assert resolve_selector(promoted, selector).resolved
    assert promote_container_rollout(document).content_digest == promoted.content_digest


def test_content_json_action_plan_is_promoted_as_the_choose_actions_tool_call() -> None:
    from synth_containers.tracing.adapters.craftax_container import _content_action_plan

    assert _content_action_plan('{"actions":["up","do"]}') == '{"actions":["up","do"]}'
    assert _content_action_plan('```json\n{"actions": ["do"]}\n```') == '{"actions":["do"]}'
    assert _content_action_plan("THOUGHT: go up\nACTIONS: up, do") is None
    assert _content_action_plan('{"actions": []}') is None
    assert _content_action_plan('{"actions": [1, 2]}') is None
    assert _content_action_plan(None) is None
