"""Offline producer → sealed bundle/readback regressions; no provider calls."""

from dataclasses import replace
from itertools import permutations
import json
from pathlib import Path

import httpx
import pytest

from synth_containers.tracing.adapters.codex_jsonl import import_codex_jsonl
from synth_containers.tracing.adapters.atif import import_atif
from synth_containers.tracing.adapters.native import import_native_to_bundle
from synth_containers.tracing.adapters.openai_responses import (
    OpenAIResponsesAdapter,
    normalize_responses_response,
    usage_from_responses,
)
from synth_containers.tracing.capture.supervisor import CaptureSupervisor, SupervisorConfig
from synth_containers.tracing.canonical import canonical_bytes
from synth_containers.tracing.models.identity import TraceProvenanceV5
from synth_containers.tracing.models.spans import UsageV5
from synth_containers.tracing.projections.v4 import project_v4
from synth_containers.tracing.store.bundle import LocalTraceBundle


def test_partial_usage_never_becomes_a_complete_derived_total() -> None:
    left = UsageV5(provenance="partial", prompt_tokens=10, source_refs=("a",))
    right = UsageV5(provenance="observed_provider", prompt_tokens=5, source_refs=("b",))
    total = left.merged(right)
    assert total.provenance == "partial"
    assert total.prompt_tokens == 15
    assert total.cost_usd is None
    assert total.source_refs == ("a", "b")


@pytest.mark.parametrize("field", [
    "prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens",
    "cache_write_tokens", "total_tokens", "requests", "wall_time_seconds", "cost_usd",
])
def test_missing_usage_component_is_null_and_remains_partial(field: str) -> None:
    observations = [
        UsageV5(provenance="observed_provider", **{field: 0}),
        UsageV5(provenance="observed_provider", **{field: None}),
        UsageV5(provenance="observed_harness", **{field: 2}),
    ]
    for a, b, c in permutations(observations):
        left, right = a.merged(b).merged(c), a.merged(b.merged(c))
        assert left == right
        assert getattr(left, field) is None
        assert field in left.unavailable_fields
        assert left.provenance == "partial"


def test_explicit_unavailability_wins_over_a_supplied_number() -> None:
    left = UsageV5(provenance="observed_provider", cost_usd=1.0, unavailable_fields=("cost_usd",))
    assert left.merged(replace(left, unavailable_fields=())).cost_usd is None


def test_v4_projection_declares_nullable_usage_loss_even_for_observed_provider() -> None:
    trace = import_atif({
        "schema_version": "ATIF-v1.7", "trajectory_id": "nullable-usage",
        "agent": {"name": "fixture", "version": "1"},
        "steps": [{"step_id": 1, "source": "agent", "message": "fixture",
                   "metrics": {"prompt_tokens": 0, "completion_tokens": 2}}],
    })
    span = replace(trace.spans[0], usage=UsageV5(provenance="observed_provider", prompt_tokens=0),
                   content_digest="").sealed()
    trace = replace(trace, spans=(span,), content_digest="").sealed()
    before = canonical_bytes(trace)
    projection, manifest = project_v4(trace)
    assert projection.spans[0].response.usage.completion_tokens == 0
    assert any(loss.field_path == "spans[].usage" and "completion_tokens" in loss.reason
               for loss in manifest.losses)
    assert canonical_bytes(trace) == before


def test_observed_zero_and_provider_field_precedence_survive_responses_usage() -> None:
    usage = usage_from_responses({
        "input_tokens": 0, "prompt_tokens": 99,
        "output_tokens": 0, "completion_tokens": 88, "total_tokens": 0,
        "input_tokens_details": {"cache_write_tokens": 0, "cache_creation_tokens": 77},
    })
    assert usage.prompt_tokens == usage.completion_tokens == usage.total_tokens == 0
    assert usage.cache_write_tokens == 0
    assert usage.unavailable_fields == ()


@pytest.mark.parametrize("timestamp", ["2026-10-20T22:00:00Z", "2026-10-20T18:00:00.123456-04:00"])
def test_codex_timestamp_survives_sealing_and_bundle_readback(tmp_path: Path, timestamp: str) -> None:
    records = [
        {"timestamp": timestamp, "type": "item.completed",
         "item": {"id": "i1", "type": "agent_message", "text": "fixture"}},
        {"timestamp": "2026-10-20T17:00:00-04:00", "type": "turn.completed"},
    ]
    source = tmp_path / "codex.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    bundle = LocalTraceBundle(tmp_path / "bundle")
    receipt = import_native_to_bundle(source, source_format="codex_jsonl", bundle=bundle)
    document = bundle.read_trace(receipt["trace_digest"])
    events = [e for e in document["events"] if e["event_type"].startswith("codex.")]
    assert [e["occurred_at"] for e in events] == [timestamp, records[1]["timestamp"]]
    assert events[0]["order"]["chronological_sequence"] < events[1]["order"]["chronological_sequence"]
    assert events[0]["payload"]["_synth_import_time"]["basis"] == "producer"
    assert events[0]["payload"]["_synth_import_time"].get("canonical_fallback") is None
    again = import_native_to_bundle(source, source_format="codex_jsonl",
                                  bundle=LocalTraceBundle(tmp_path / "second-bundle"))
    assert again["trace_digest"] == receipt["trace_digest"]


@pytest.mark.parametrize("timestamp,reason", [(None, "absent"), (42, "invalid"),
    ("2026-10-20T22:00:00", "invalid"), ("not-a-time", "invalid")])
def test_codex_unknown_producer_time_is_explicit(tmp_path: Path, timestamp: object, reason: str) -> None:
    record = {"timestamp": timestamp, "type": "item.completed",
              "item": {"id": "i1", "type": "agent_message", "text": "fixture"}}
    raw = (json.dumps(record) + "\n").encode()
    imported = import_codex_jsonl(raw, target_id="s1")
    assert imported.events[0]["occurred_at"] is None
    source = tmp_path / "codex.jsonl"
    source.write_bytes(raw)
    bundle = LocalTraceBundle(tmp_path / "bundle")
    receipt = import_native_to_bundle(source, source_format="codex_jsonl", bundle=bundle)
    event = next(e for e in bundle.read_trace(receipt["trace_digest"])["events"]
                 if e["event_type"] == "codex.agent_message")
    basis = event["payload"]["_synth_import_time"]
    assert event["occurred_at"] == basis["canonical_fallback"] == "1970-01-01T00:00:00Z"
    assert basis.get("occurred_at") is None
    assert basis["basis"] == "unavailable"
    assert basis["reason"] == reason


def test_codex_nested_source_timestamp() -> None:
    raw = b'{"timestamp":"2026-10-20T22:00:00Z","msg":{"type":"task_complete"}}\n'
    assert import_codex_jsonl(raw, target_id="s1").events[0]["occurred_at"] == "2026-10-20T22:00:00Z"


@pytest.mark.parametrize("terminal", ["incomplete", "completed", "failed", "cancelled"])
@pytest.mark.parametrize("output", [True, False])
def test_responses_terminal_status_and_details_survive_stream_fallback(terminal: str, output: bool) -> None:
    body = {"id": "r1", "status": terminal,
            "incomplete_details": {"reason": "max_output_tokens"} if terminal == "incomplete" else None,
            "usage": {"input_tokens": 2, "output_tokens": 0, "total_tokens": 2}}
    if output:
        body["output"] = [{"type": "message", "role": "assistant",
                           "content": [{"type": "output_text", "text": "partial"}]}]
    unary = normalize_responses_response(body)
    assert unary.terminal_observed
    assert unary.terminal_status == terminal
    frames = [
        {"type": "response.output_text.delta", "delta": "partial"},
        {"type": f"response.{terminal}", "response": body},
    ]
    stream = OpenAIResponsesAdapter().new_stream()
    data = "".join(f"event: {f['type']}\ndata: {json.dumps(f)}\n\n" for f in frames).encode()
    for offset in range(0, len(data), 7):
        stream.feed(data[offset:offset + 7])
    result = stream.finish()
    assert result.terminal_observed
    assert result.terminal_status == terminal
    assert result.terminal_details == unary.terminal_details
    assert result.messages[0].parts[0].text == "partial"
    assert result.usage.completion_tokens == 0
    assert result.raw_events == frames
    assert not any("unsupported" in d for d in result.diagnostics)


def test_interrupted_responses_stream_does_not_invent_a_terminal_event() -> None:
    stream = OpenAIResponsesAdapter().new_stream()
    stream.feed(b'data: {"type":"response.output_text.delta","delta":"partial"}\n\n')
    result = stream.finish()
    assert not result.terminal_observed
    assert result.terminal_status is None
    assert result.usage.provenance == "unavailable"


@pytest.mark.parametrize("streaming", [True, False])
def test_incomplete_responses_is_truncated_after_capture_and_storage(tmp_path: Path, streaming: bool) -> None:
    body = {"id": "r1", "status": "incomplete", "output": [{"type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "partial"}]}],
            "incomplete_details": {"reason": "max_output_tokens"},
            "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}}

    def upstream(_: httpx.Request) -> httpx.Response:
        if streaming:
            frame = {"type": "response.incomplete", "response": body}
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  stream=httpx.ByteStream(
                                      f"event: response.incomplete\ndata: {json.dumps(frame)}\n\n".encode()))
        return httpx.Response(200, json=body)

    supervisor = CaptureSupervisor(SupervisorConfig(
        bundle_root=tmp_path / "bundle", trace_key={"test": "incomplete", "stream": streaming},
        upstream_base_url="https://api.openai.com/v1",
        provenance=TraceProvenanceV5(producer="offline-test", producer_version="1"),
    ))
    supervisor.proxy._client.close()
    supervisor.proxy._client = httpx.Client(transport=httpx.MockTransport(upstream), trust_env=False)
    supervisor.start_capture()
    try:
        response = httpx.post(f"{supervisor.openai_base_url}/responses",
                              json={"model": "fixture", "input": "fixture", "stream": streaming},
                              timeout=10, trust_env=False)
        assert response.status_code == 200
    finally:
        sealed = supervisor.finalize()
    document = LocalTraceBundle(tmp_path / "bundle").read_trace(sealed.document.content_digest)
    span = next(s for s in document["spans"] if s["span_kind"] == "model_call")
    assert span["status"] == "truncated"
    assert span["detail"]["provider_terminal_status"] == "incomplete"
    assert span["detail"]["provider_terminal_details"] == {"incomplete_details": {"reason": "max_output_tokens"}}
    assert span["usage"]["total_tokens"] == document["usage"]["total_tokens"] == 3
    event = next(e for e in document["events"] if e["event_type"] == "model_call.finished")
    assert event["status"] == "truncated"
    assert event["payload"]["provider_terminal_observed"] is True
