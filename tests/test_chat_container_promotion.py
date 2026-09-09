"""A container-sealed Banking77 LLM rollout promotes to messages + a model_call span."""

from __future__ import annotations

from pathlib import Path

from synth_containers.tracing.adapters.chat_container import is_container_chat_rollout, promote_chat_rollout, promote_container_rollout_any
from synth_containers.tracing.canonical import content_digest
from synth_containers.tracing.store.bundle import LocalTraceBundle
from synth_containers.tracing.validation.rehydrate import trace_document_from_payload
from synth_containers.tracing.validation.validator import Severity, validate_trace

FIXTURE = Path(__file__).parent / "fixtures" / "banking77_container_rollout_llm.trace-v5.zip"


def test_banking77_llm_rollout_promotes_to_messages_and_model_call(tmp_path: Path) -> None:
    bundle = LocalTraceBundle.extract_archive(FIXTURE, tmp_path / "b", require_self_contained=False)
    entry = bundle.read_manifest()["traces"][0]
    document = trace_document_from_payload(bundle.read_trace(entry["trace_digest"]))
    assert is_container_chat_rollout(document) and not document.messages
    promoted = promote_chat_rollout(document)
    assert promoted.trace_id == document.trace_id and content_digest(promoted) == promoted.content_digest
    roles = [str(m.role) for m in promoted.messages]
    assert roles == ["system", "user", "assistant"]
    assert promoted.messages[1].text().startswith("Customer query:") and promoted.messages[2].text() == "find_card"
    call = promoted.spans_of_kind("model_call")[0]
    assert call.input_message_ids == (promoted.messages[0].message_id, promoted.messages[1].message_id)
    assert call.output_message_ids == (promoted.messages[2].message_id,) and call.detail["label"] == "find_card"
    assert promoted.provenance.extra["promoted_from_container_trace_digest"] == document.content_digest
    assert len(promoted.events) == len(document.events)
    assert not [f for f in validate_trace(promoted) if str(f.severity) == Severity.ERROR]
    assert promote_chat_rollout(promoted).content_digest == promoted.content_digest
    assert promote_container_rollout_any(document).content_digest == promoted.content_digest


HEALTHBENCH_FIXTURE = Path(__file__).parent / "fixtures" / "healthbench2_container_rollout_llm.trace-v5.zip"


def test_healthbench2_llm_rollout_promotes_conversation_and_model_call(tmp_path: Path) -> None:
    """A real healthbench2 rollout (OpenRouter policy + rubric grader) seals `observation.messages`
    and `action.content`; promotion derives the conversation turns, one model_call span, and keeps
    every rubric.grade event verbatim."""

    bundle = LocalTraceBundle.extract_archive(HEALTHBENCH_FIXTURE, tmp_path / "hb", require_self_contained=False)
    entry = bundle.read_manifest()["traces"][0]
    document = trace_document_from_payload(bundle.read_trace(entry["trace_digest"]))
    assert is_container_chat_rollout(document) and not document.messages and not document.spans
    promoted = promote_chat_rollout(document, entry["trace_digest"])
    assert promoted.trace_id == document.trace_id and content_digest(promoted) == promoted.content_digest
    assert [str(m.role) for m in promoted.messages] == ["user", "assistant"]
    assert promoted.messages[0].text().startswith("I’m a 39 year old female.")
    assert promoted.messages[1].text().startswith("As the world's leading expert")
    call = promoted.spans_of_kind("model_call")[0]
    assert call.input_message_ids == (promoted.messages[0].message_id,)
    assert call.output_message_ids == (promoted.messages[1].message_id,)
    assert call.detail["policy"] == {"harness": "chat_completion", "config": "openrouter_gemini25_flash_lite"}
    kinds = [str(e.event_type) for e in promoted.events]
    assert len(promoted.events) == len(document.events) and kinds.count("rubric.grade") == 10 and "reward_signal" in kinds
    assert promoted.provenance.extra["promoted_from_container_trace_digest"] == entry["trace_digest"] == document.content_digest
    assert promoted.provenance.transformation_chain[-1] == "chat_container_promotion@1"
    assert not [f for f in validate_trace(promoted) if str(f.severity) == Severity.ERROR]
    assert promote_chat_rollout(promoted).content_digest == promoted.content_digest
    assert promote_container_rollout_any(document, entry["trace_digest"]).content_digest == promoted.content_digest
