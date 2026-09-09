"""Codex app-server contract tests against the scripted in-process fake."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


from synth_containers.tracing.annotation import (
    AnnotationJobErrorCode,
    AnnotationJobLimitsV1,
    AnnotationJobState,
    AnnotationService,
    AnnotationStore,
    AnnotatorProgramV1,
    CodexAppServerRunner,
    DefinitionRegistry,
    LocalReservationBroker,
    ReservationBindingV1,
    RunnerKind,
    ScriptedAppServer,
    build_craftax_smoke_trace,
)
from synth_containers.tracing.annotation.proposal import PROPOSAL_SCHEMA_VERSION
from synth_containers.tracing.models.standards import (
    AnnotationOutputContractV1,
    AnnotationTaskKind,
    AnnotationTaxonV1,
    TraceAnnotatorDefinitionV1,
)
from synth_containers.tracing.validation.validator import Severity, validate_evidence, validate_trace


TAXONOMY = ("belief.contradicted", "belief.correct", "spatial.traversability")


def _errors(findings):
    return [item.code for item in findings if str(item.severity) == Severity.ERROR]


def _definition() -> TraceAnnotatorDefinitionV1:
    return TraceAnnotatorDefinitionV1(
        annotator_id="test.codex.belief",
        name="Belief annotator",
        purpose="beliefs contradicted by engine evidence",
        taxonomy=TAXONOMY,
        required_subject_scope="message",
        minimum_evidence=2,
        model="gpt-5.6-luna",
        output_contract=AnnotationOutputContractV1(
            task_kind=AnnotationTaskKind.CLASSIFY,
            annotation_types=("belief",),
            taxonomy=tuple(AnnotationTaxonV1(label=label) for label in TAXONOMY),
        ),
    ).sealed()


def _program(**overrides: Any) -> AnnotatorProgramV1:
    return AnnotatorProgramV1(
        program_id="test.codex.belief.program",
        runner_kind=RunnerKind.CODEX_APP_SERVER,
        prompt="Find beliefs in assistant replies that the observation contradicts.",
        paid=True,
        **overrides,
    ).sealed()


LIMITS = AnnotationJobLimitsV1(max_total_tokens=100_000)


class _Harness:
    """Service + broker; ``run`` issues one reservation and executes in-process."""

    def __init__(self, service, trace, fakes, broker):
        self.service, self.trace, self.fakes, self.broker = service, trace, fakes, broker

    def reserve(self, *, cap_usd_micros: int = 1_000_000, model: str | None = "gpt-5.6-luna") -> str:
        return self.broker.issue(cap_usd_micros=cap_usd_micros, binding=ReservationBindingV1(trace_digest=self.trace.content_digest, annotator_id="test.codex.belief", model=model, session_id="sess-test")).reservation_id

    def run(self, request):
        return self.service.submit_and_run(request, reservation_id=self.reserve(model=request.model or "gpt-5.6-luna"), session_id="sess-test")


def _service(tmp_path: Path, agent, **fake_kwargs: Any):
    registry = DefinitionRegistry()
    registry.register(_definition(), _program(), domain="craftax")
    fakes: list[ScriptedAppServer] = []

    def factory(cwd: Path) -> ScriptedAppServer:
        fake = ScriptedAppServer(agent, **fake_kwargs)
        fakes.append(fake)
        return fake

    runner = CodexAppServerRunner(factory, poll_seconds=0.01, default_effort="medium", proxy_enforces_reservation=True)
    broker = LocalReservationBroker(tmp_path / "broker")
    service = AnnotationService(store=AnnotationStore(tmp_path / "store"), registry=registry, runners={runner.kind: runner}, broker=broker)
    trace = build_craftax_smoke_trace()
    service.register_trace(trace)
    return _Harness(service, trace, fakes, broker)


def good_agent():
    manifest = yield ("trace_get_manifest", {})
    trace_id, digest = manifest["trace_id"], manifest["trace_digest"]
    replies = yield ("trace_list_entities", {"kind": "message", "role": "assistant", "limit": 5})
    first = replies["items"][0]
    reply = yield ("trace_get_message", {"message_id": first["message_id"]})
    prompt_id = reply["predecessor_message_ids"][0]
    prompt = yield ("trace_get_message", {"message_id": prompt_id})
    assert "front_tile: grass" in prompt["text"]
    check = yield ("trace_resolve_selector", {"selector": {"kind": "message", "entity_id": first["message_id"], "quote": "tree directly in front"}})
    assert check["resolved"]
    approval = yield ("__request_approval__", {"command": "cat /etc/passwd"})
    assert approval["decision"] == "decline"
    return json.dumps(
        {
            "schema_version": PROPOSAL_SCHEMA_VERSION,
            "source_trace_id": trace_id,
            "source_trace_digest": digest,
            "findings": [
                {
                    "target": {"kind": "message", "entity_id": first["message_id"]},
                    "annotation_type": "belief",
                    "labels": ["belief.contradicted", "spatial.traversability"],
                    "payload": {},
                    "confidence": 0.9,
                    "rationale": "claims a tree in front; observation says front_tile grass",
                    "evidence": [
                        {"kind": "message", "entity_id": first["message_id"], "quote": "tree directly in front"},
                        {"kind": "message", "entity_id": prompt_id, "quote": "front_tile: grass"},
                    ],
                }
            ],
            "abstentions": [],
            "judgments": [],
            "summary": "one contradicted belief",
        }
    )


def test_scripted_app_server_seals_grounded_annotation_with_execution_trace(tmp_path: Path) -> None:
    h = _service(tmp_path, good_agent, token_usage={"inputTokens": 1200, "outputTokens": 300, "cachedInputTokens": 0, "reasoningOutputTokens": 100, "totalTokens": 1500})
    request = h.service.request_for(h.trace, "test.codex.belief", model="gpt-5.6-luna", reasoning_effort="low", limits=AnnotationJobLimitsV1(max_total_tokens=100_000, max_cost_usd=0.5))
    job = h.run(request)
    assert str(job.state) == AnnotationJobState.SEALED, job.error
    assert job.applied_count == 1 and job.usage.total_tokens == 1500 and job.usage.tool_calls == 5
    assert job.execution_trace_id and job.execution_trace_digest
    head = h.service.evidence_head(h.trace.trace_id)
    assert not _errors(validate_evidence(h.trace, head)[0])
    annotation = head.annotations[0]
    assert str(annotation.author_kind) == "agentic" and annotation.producer.model == "gpt-5.6-luna"
    assert annotation.annotator_execution_trace_id == job.execution_trace_id
    assert annotation.labels == ("belief.contradicted", "spatial.traversability")
    execution = h.service.store.get_execution_trace(job.job_id)
    assert execution is not None and execution.content_digest == job.execution_trace_digest
    assert not _errors(validate_trace(execution))
    assert len(execution.spans_of_kind("tool_execution")) == 5
    assert execution.links[0].target_digest == h.trace.content_digest
    assert execution.provenance.extra["reasoning_policy"] == "not_captured"
    assert not any(str(part.type) == "reasoning" for message in execution.messages for part in message.parts)
    fake = h.fakes[0]
    assert fake.approval_decisions == ["decline"]
    thread_start = next(m for m in fake.sent if m.get("method") == "thread/start")["params"]
    assert thread_start["sandbox"] == "read-only" and thread_start["approvalPolicy"] == "never" and thread_start["ephemeral"] is True
    assert [tool["name"] for tool in thread_start["dynamicTools"]][0] == "trace_get_manifest"
    turn_start = next(m for m in fake.sent if m.get("method") == "turn/start")["params"]
    assert turn_start["outputSchema"]["properties"]["schema_version"]["enum"] == [PROPOSAL_SCHEMA_VERSION] and turn_start["effort"] == "low"
    assert turn_start["sandboxPolicy"] == {"type": "readOnly", "networkAccess": False}
    workspace = h.service.store.workspace_dir(job.job_id)
    assert (workspace / "INSTRUCTIONS.md").exists() and (workspace / "manifest.json").stat().st_mode & 0o222 == 0
    assert "hidden or step-by-step reasoning" in (workspace / "INSTRUCTIONS.md").read_text()
    receipt = h.service.store.receipts(job.job_id)[-1]
    assert receipt.detail["usage"]["cost_status"] == "unavailable" and receipt.detail["execution_trace_digest"] == job.execution_trace_digest


def test_cached_replay_never_starts_a_second_task(tmp_path: Path) -> None:
    h = _service(tmp_path, good_agent)
    request = h.service.request_for(h.trace, "test.codex.belief", model="gpt-5.6-luna", limits=LIMITS)
    first = h.run(request)
    assert str(first.state) == AnnotationJobState.SEALED
    second = h.service.submit_and_run(request)  # no reservation needed: nothing is spent
    assert second.job_id == first.job_id and len(h.fakes) == 1
    estimate = h.service.estimate(request)
    assert estimate.cached and not estimate.requires_reservation
    reservation = h.broker.get(first.reservation_id)
    assert reservation.claimed_by_job_id == first.job_id and reservation.outcome == "sealed" and reservation.reconciled_at
    assert h.service.store.ledger.get(first.job_id).stage == "acknowledged"


def malformed_agent():
    yield ("trace_get_manifest", {})
    return "this is not json {"


def unresolvable_agent():
    manifest = yield ("trace_get_manifest", {})
    return json.dumps(
        {
            "schema_version": PROPOSAL_SCHEMA_VERSION,
            "source_trace_id": manifest["trace_id"],
            "source_trace_digest": manifest["trace_digest"],
            "findings": [
                {
                    "target": {"kind": "message", "entity_id": "msg_invented"},
                    "annotation_type": "belief",
                    "labels": ["belief.correct"],
                    "evidence": [{"kind": "message", "entity_id": "msg_invented"}, {"kind": "trace"}],
                    "rationale": "I remember this",
                }
            ],
            "abstentions": [],
        }
    )


def looping_agent():
    while True:
        yield ("trace_get_manifest", {})


def test_malformed_final_message_fails_closed_but_keeps_execution_trace(tmp_path: Path) -> None:
    h = _service(tmp_path, malformed_agent)
    job = h.run(h.service.request_for(h.trace, "test.codex.belief", limits=LIMITS))
    assert str(job.state) == AnnotationJobState.FAILED and job.error.code == AnnotationJobErrorCode.MALFORMED_OUTPUT
    assert h.service.evidence_head(h.trace.trace_id) is None
    execution = h.service.store.get_execution_trace(job.job_id)
    assert execution is not None and execution.lifecycle.status == "failed"


def test_invented_target_becomes_rejected_abstention(tmp_path: Path) -> None:
    h = _service(tmp_path, unresolvable_agent)
    job = h.run(h.service.request_for(h.trace, "test.codex.belief", limits=LIMITS))
    # message-scoped annotator cannot abstain on an unresolvable message target, so the finding is rejected
    assert str(job.state) == AnnotationJobState.SEALED and job.applied_count == 0 and job.rejected_count == 1
    head = h.service.evidence_head(h.trace.trace_id)
    assert head is not None and head.annotations == ()
    receipt = h.service.store.receipts(job.job_id)[-1]
    assert receipt.detail["rejected"][0]["reason"] == "target_unresolved"


def test_no_final_message_is_typed(tmp_path: Path) -> None:
    h = _service(tmp_path, malformed_agent, omit_final_message=True)
    job = h.run(h.service.request_for(h.trace, "test.codex.belief", limits=LIMITS))
    assert job.error.code == AnnotationJobErrorCode.NO_STRUCTURED_OUTPUT


def test_failed_turn_is_typed(tmp_path: Path) -> None:
    h = _service(tmp_path, malformed_agent, final_status="failed")
    job = h.run(h.service.request_for(h.trace, "test.codex.belief", limits=LIMITS))
    assert str(job.state) == AnnotationJobState.FAILED and job.error.code == AnnotationJobErrorCode.INTERNAL
    assert "scripted failure" in job.error.message


def test_disconnect_is_typed(tmp_path: Path) -> None:
    h = _service(tmp_path, good_agent, disconnect_after_calls=2)
    job = h.run(h.service.request_for(h.trace, "test.codex.belief", limits=LIMITS))
    assert job.error.code == AnnotationJobErrorCode.TRANSPORT_DISCONNECTED


def test_hang_times_out_and_interrupts(tmp_path: Path) -> None:
    h = _service(tmp_path, good_agent, hang_after_calls=1)
    job = h.run(h.service.request_for(h.trace, "test.codex.belief", limits=AnnotationJobLimitsV1(max_total_tokens=100_000, timeout_seconds=0.3)))
    assert job.error.code == AnnotationJobErrorCode.TIMEOUT
    assert any(m.get("method") == "turn/interrupt" for m in h.fakes[0].sent)


def test_tool_call_limit_interrupts_and_fails(tmp_path: Path) -> None:
    h = _service(tmp_path, looping_agent)
    job = h.run(h.service.request_for(h.trace, "test.codex.belief", limits=AnnotationJobLimitsV1(max_total_tokens=100_000, max_tool_calls=4)))
    assert job.error.code == AnnotationJobErrorCode.TOOL_LIMIT_EXCEEDED
    assert h.fakes[0].interrupted and job.usage.tool_calls >= 5  # attempts past the limit are refused, not served


def test_token_limit_is_enforced(tmp_path: Path) -> None:
    h = _service(tmp_path, good_agent, token_usage={"inputTokens": 9000, "outputTokens": 100, "totalTokens": 9100})
    job = h.run(h.service.request_for(h.trace, "test.codex.belief", limits=AnnotationJobLimitsV1(max_total_tokens=5000)))
    assert job.error.code == AnnotationJobErrorCode.TOKEN_LIMIT_EXCEEDED


def test_thread_start_rejection_is_typed(tmp_path: Path) -> None:
    h = _service(tmp_path, good_agent, reject_thread_start=True)
    job = h.run(h.service.request_for(h.trace, "test.codex.belief", limits=LIMITS))
    assert job.error.code == AnnotationJobErrorCode.RUNNER_UNAVAILABLE


def test_tools_outside_the_contract_are_refused(tmp_path: Path) -> None:
    def sneaky_agent():
        manifest = yield ("trace_get_manifest", {})
        refused = yield ("shell_exec", {"command": "ls"})
        assert "error" in refused
        return json.dumps({"schema_version": PROPOSAL_SCHEMA_VERSION, "source_trace_id": manifest["trace_id"], "source_trace_digest": manifest["trace_digest"], "findings": [], "abstentions": []})

    h = _service(tmp_path, sneaky_agent)
    job = h.run(h.service.request_for(h.trace, "test.codex.belief", limits=LIMITS))
    assert str(job.state) == AnnotationJobState.SEALED and job.applied_count == 0
    execution = h.service.store.get_execution_trace(job.job_id)
    refused = [span for span in execution.spans_of_kind("tool_execution") if span.detail["tool"] == "shell_exec"]
    assert refused and refused[0].status == "error"


def test_cost_ceiling_becomes_a_token_ceiling_when_priced(tmp_path: Path) -> None:
    h = _service(tmp_path, good_agent, token_usage={"inputTokens": 40_000, "outputTokens": 2_000, "totalTokens": 42_000})
    h.service.runners["codex_app_server"].usd_per_million_tokens = 10.0  # $0.10 buys 10k tokens
    job = h.run(h.service.request_for(h.trace, "test.codex.belief", limits=AnnotationJobLimitsV1(max_total_tokens=1_000_000, max_cost_usd=0.1)))
    assert job.error.code == AnnotationJobErrorCode.TOKEN_LIMIT_EXCEEDED and "ceiling 10000" in job.error.message
    assert h.broker.get(job.reservation_id).outcome == "failed"


def test_isolated_codex_home_never_copies_host_config(tmp_path: Path, monkeypatch) -> None:
    from synth_containers.tracing.annotation import StdioAppServerTransport
    from synth_containers.tracing.annotation.codex_app_server import minimal_codex_config

    host = tmp_path / "host-codex"
    host.mkdir()
    (host / "auth.json").write_text('{"token": "x"}')
    (host / "config.toml").write_text('[mcp_servers.evil]\ncommand = "rm"\n')
    monkeypatch.setenv("CODEX_HOME", str(host))
    monkeypatch.setenv("OPENAI_API_KEY", "leak")
    transport = StdioAppServerTransport(command=("cat",), model="gpt-5.6-luna", reasoning_effort="low")
    transport.start()
    try:
        home = Path(transport.process.args[0] if False else transport._home.name)
        generated = (home / "config.toml").read_text()
        assert "evil" not in generated and 'model = "gpt-5.6-luna"' in generated and 'sandbox_mode = "read-only"' in generated
        assert (home / "auth.json").read_text() == '{"token": "x"}'
        assert oct((home / "auth.json").stat().st_mode & 0o777) == "0o600"
    finally:
        transport.close()
    assert "[mcp_servers]" in minimal_codex_config(model=None, reasoning_effort=None)


def test_stdio_transport_close_kills_the_whole_process_group(tmp_path) -> None:
    import os
    import sys
    import time

    from synth_containers.tracing.annotation.codex_app_server import StdioAppServerTransport

    # A wrapper that spawns a grandchild and ignores stdin EOF, like a stuck
    # ``node codex app-server`` whose native binary keeps running.
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(
        "import json, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print(json.dumps({'child': child.pid}), flush=True)\n"
        "time.sleep(60)\n"
    )
    transport = StdioAppServerTransport(command=(sys.executable, str(wrapper)), cwd=tmp_path)
    transport.start()
    assert transport.process is not None
    grandchild = None
    deadline = time.monotonic() + 10
    while grandchild is None and time.monotonic() < deadline:
        item = transport._queue.get(timeout=1)
        if isinstance(item, dict) and "child" in item:
            grandchild = int(item["child"])
    assert grandchild is not None, "wrapper did not report its child pid"
    wrapper_pid = transport.process.pid
    assert os.getpgid(wrapper_pid) == wrapper_pid  # own session/group
    transport.close()
    for pid in (wrapper_pid, grandchild):
        deadline = time.monotonic() + 5
        alive = True
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                alive = False
                break
            try:
                os.waitpid(pid, os.WNOHANG)  # reap if it is our zombie
            except ChildProcessError:
                pass
            time.sleep(0.05)
        assert not alive, f"pid {pid} survived close()"
