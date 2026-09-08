"""Jesterky swarm runner: request.runner_kind dispatch and proposal extraction."""

from __future__ import annotations

import pytest

from pathlib import Path
from typing import Any

from synth_containers.tracing.annotation import (
    AnnotationJobLimitsV1,
    AnnotationJobState,
    AnnotationService,
    AnnotationStore,
    AnnotatorProgramV1,
    DefinitionRegistry,
    JesterkyRunner,
    LocalReservationBroker,
    ReservationBindingV1,
    RunnerKind,
    extract_proposal,
    swarm_spec,
)
from synth_containers.tracing.annotation.proposal import PROPOSAL_SCHEMA_VERSION
from synth_containers.tracing.models.standards import (
    AnnotationOutputContractV1,
    AnnotationTaskKind,
    AnnotationTaxonV1,
    TraceAnnotatorDefinitionV1,
)


TAXONOMY = ("problem_model.matches_issue", "problem_model.wrong")


def _definition() -> TraceAnnotatorDefinitionV1:
    return TraceAnnotatorDefinitionV1(
        annotator_id="test.jesterky.problem_model",
        name="Problem model",
        purpose="diagnosis vs issue",
        taxonomy=TAXONOMY,
        required_subject_scope="trace",
        minimum_evidence=2,
        model="gpt-5.6-luna",
        output_contract=AnnotationOutputContractV1(
            task_kind=AnnotationTaskKind.CLASSIFY,
            annotation_types=("problem_model",),
            taxonomy=tuple(AnnotationTaxonV1(label=label) for label in TAXONOMY),
        ),
    ).sealed()


def _program() -> AnnotatorProgramV1:
    return AnnotatorProgramV1(
        program_id="test.jesterky.problem_model.program",
        runner_kind=RunnerKind.CODEX_APP_SERVER,
        prompt="Judge whether the diagnosis matches the issue.",
        paid=True,
        parameters={"default_effort": "medium"},
    ).sealed()


LIMITS = AnnotationJobLimitsV1(max_total_tokens=50_000, max_cost_usd=0.5)


def _proposal(context: Any) -> dict[str, Any]:
    listing = context.tools.call("trace_list_entities", {"kind": "event", "limit": 80})
    items = listing["items"]
    assert len(items) >= 2, items
    first, second = items[0], items[-1]
    return {
        "schema_version": PROPOSAL_SCHEMA_VERSION,
        "source_trace_id": context.document.trace_id,
        "source_trace_digest": context.document.content_digest,
        "findings": [
            {
                "target": {"kind": "trace", "entity_id": context.document.trace_id},
                "annotation_type": "problem_model",
                "labels": ["problem_model.matches_issue"],
                "payload": {},
                "confidence": 0.8,
                "rationale": "scripted jesterky swarm worker cited two events",
                "evidence": [
                    {"kind": "event", "entity_id": first["event_id"]},
                    {"kind": "event", "entity_id": second["event_id"]},
                ],
            }
        ],
        "abstentions": [],
        "judgments": [],
        "summary": "scripted jesterky swarm worker",
    }


def _service(tmp_path: Path) -> tuple[AnnotationService, Any]:
    from synth_containers.tracing.annotation.fixtures import build_craftax_smoke_trace

    registry = DefinitionRegistry()
    registry.register(_definition(), _program(), domain="test")
    runner = JesterkyRunner(proposal_factory=_proposal, proxy_enforces_reservation=True, default_effort="medium")
    broker = LocalReservationBroker(tmp_path / "broker")
    service = AnnotationService(
        store=AnnotationStore(tmp_path / "store"),
        registry=registry,
        runners={runner.kind: runner},
        broker=broker,
    )
    trace = build_craftax_smoke_trace()
    service.register_trace(trace)
    return service, trace


def test_swarm_spec_is_a_map_over_jobs() -> None:
    spec = swarm_spec(concurrency=4, prompt="annotate")
    assert spec["nodes"]["annotate_jobs"]["kind"] == "map"
    assert spec["nodes"]["annotate_jobs"]["concurrency"] == 4
    assert spec["nodes"]["annotate_jobs"]["body"]["actor"] == "trace_annotator"
    assert spec["host"]["roles"]["trace_annotator"]["prompt"] == "annotate"


def test_extract_proposal_from_recorded_outputs() -> None:
    proposal = {"schema_version": PROPOSAL_SCHEMA_VERSION, "findings": []}
    assert extract_proposal({"recorded": [{"outputs": {"proposal": proposal}}]}) == proposal
    assert extract_proposal({"recorded": [{"outputs": {"echo": "nope"}}]}) is None


def test_request_runner_kind_and_model_are_pinned_and_keyed(tmp_path: Path) -> None:
    service, trace = _service(tmp_path)
    request = service.request_for(
        trace,
        "test.jesterky.problem_model",
        model="gpt-5.6-luna",
        reasoning_effort="low",
        runner_kind=RunnerKind.JESTERKY,
        limits=LIMITS,
    )
    assert request.model == "gpt-5.6-luna"
    assert request.reasoning_effort == "low"
    assert request.runner_kind == RunnerKind.JESTERKY
    estimate = service.estimate(request)
    assert estimate.runner_kind == RunnerKind.JESTERKY
    assert estimate.resolved_model == "gpt-5.6-luna"
    assert any("jesterky swarm" in note for note in estimate.notes)
    defaulted = service.request_for(trace, "test.jesterky.problem_model", model="gpt-5.6-luna", limits=LIMITS)
    assert defaulted.runner_kind == RunnerKind.CODEX_APP_SERVER
    assert service.estimate(request).idempotency_key != service.estimate(defaulted).idempotency_key


def test_scripted_jesterky_seals_with_reservation(tmp_path: Path) -> None:
    service, trace = _service(tmp_path)
    request = service.request_for(
        trace,
        "test.jesterky.problem_model",
        model="gpt-5.6-luna",
        reasoning_effort="medium",
        runner_kind=RunnerKind.JESTERKY,
        limits=LIMITS,
    )
    reservation = service.broker.issue(
        cap_usd_micros=1_000_000,
        binding=ReservationBindingV1(
            trace_digest=trace.content_digest,
            annotator_id="test.jesterky.problem_model",
            model="gpt-5.6-luna",
            session_id="sess-jesterky",
        ),
    ).reservation_id
    job = service.submit_and_run(request, reservation_id=reservation, session_id="sess-jesterky")
    assert str(job.state) == AnnotationJobState.SEALED, job.error
    assert job.request.runner_kind == RunnerKind.JESTERKY
    assert job.request.model == "gpt-5.6-luna"
    assert job.applied_count == 1
    head = service.evidence_head(trace.trace_id)
    assert head is not None and head.annotations[0].producer.model == "gpt-5.6-luna"


def test_jesterky_cli_fake_writes_manifest(tmp_path: Path) -> None:
    script = tmp_path / "fake_jesterky.py"
    script.write_text(
        "\n".join(
            [
                "import json, sys",
                "from pathlib import Path",
                "argv = sys.argv[1:]",
                "out = Path(argv[argv.index('--out') + 1])",
                "args = json.loads(Path(argv[argv.index('--args-file') + 1]).read_text())",
                "job = args['jobs'][0]",
                "trace = json.loads(Path(job['path']).read_text())",
                "events = trace['events']",
                "first, second = events[0], events[-1]",
                "proposal = {",
                "    'schema_version': 'synth.annotation-proposal.v1',",
                "    'source_trace_id': job['trace_id'],",
                "    'source_trace_digest': job['trace_digest'],",
                "    'findings': [{",
                "        'target': {'kind': 'trace', 'entity_id': job['trace_id']},",
                "        'annotation_type': 'problem_model',",
                "        'labels': ['problem_model.matches_issue'],",
                "        'payload': {},",
                "        'confidence': 0.7,",
                "        'rationale': 'cli fake swarm worker',",
                "        'evidence': [",
                "            {'kind': 'event', 'entity_id': first['event_id']},",
                "            {'kind': 'event', 'entity_id': second['event_id']},",
                "        ],",
                "    }],",
                "    'abstentions': [],",
                "    'judgments': [],",
                "    'summary': 'cli',",
                "}",
                "out.write_text(json.dumps({'recorded': [{'outputs': {'proposal': proposal}}]}))",
            ]
        ),
        encoding="utf-8",
    )
    from synth_containers.tracing.annotation.fixtures import build_craftax_smoke_trace

    registry = DefinitionRegistry()
    registry.register(_definition(), _program(), domain="test")
    runner = JesterkyRunner(
        command=("python3", str(script)),
        actor="fake",
        proxy_enforces_reservation=True,
        default_effort="medium",
    )
    broker = LocalReservationBroker(tmp_path / "broker")
    service = AnnotationService(store=AnnotationStore(tmp_path / "store"), registry=registry, runners={runner.kind: runner}, broker=broker)
    trace = build_craftax_smoke_trace()
    service.register_trace(trace)
    request = service.request_for(
        trace,
        "test.jesterky.problem_model",
        model="gpt-5.6-luna",
        runner_kind=RunnerKind.JESTERKY,
        limits=LIMITS,
    )
    reservation = service.broker.issue(
        cap_usd_micros=1_000_000,
        binding=ReservationBindingV1(
            trace_digest=trace.content_digest,
            annotator_id="test.jesterky.problem_model",
            model="gpt-5.6-luna",
            session_id="sess-cli",
        ),
    ).reservation_id
    job = service.submit_and_run(request, reservation_id=reservation, session_id="sess-cli")
    assert str(job.state) == AnnotationJobState.SEALED, job.error
    assert job.applied_count == 1


def test_cannot_override_deterministic_to_jesterky(tmp_path: Path) -> None:
    from synth_containers.tracing.annotation import register_builtin_annotators

    registry = DefinitionRegistry()
    register_builtin_annotators(registry)
    service = AnnotationService(store=AnnotationStore(tmp_path / "store"), registry=registry)
    from synth_containers.tracing.annotation.fixtures import build_craftax_smoke_trace

    trace = build_craftax_smoke_trace()
    service.register_trace(trace)
    try:
        service.request_for(trace, "synth.deterministic.tool_call_integrity", runner_kind=RunnerKind.JESTERKY)
    except Exception as error:
        assert "deterministic" in str(error).lower() or "runner" in str(error).lower()
    else:
        raise AssertionError("expected runner override to fail")


def test_every_worker_is_collected_without_erasing_disagreement():
    from synth_containers.tracing.annotation.jesterky_runner import extract_proposals
    base={"schema_version":PROPOSAL_SCHEMA_VERSION,"source_trace_id":"t","source_trace_digest":"sha256:t","findings":[{"labels":["loop"],"evidence":[{"entity_id":"e"}]}],"summary":"first"}
    second={**base,"findings":[{"labels":["recovery"],"evidence":[{"entity_id":"e"}]}],"summary":"second"}
    manifest={"args":{"echo":base},"recorded":[{"outputs":{"first":base}},{"outputs":{"second":second}},{"outputs":{"duplicate":base}}]}
    assert len(extract_proposals(manifest))==3
    result=extract_proposal(manifest)
    assert len(result["findings"])==2
    assert result["summary"]=="first\nsecond"


def test_partitions_are_actor_scoped_bounded_and_stable():
    from types import SimpleNamespace
    from synth_containers.tracing.annotation.jesterky_runner import partition_jobs
    events=[SimpleNamespace(actor_id=f"actor-{i%4}",session_id=f"session-{i%4}",event_id=f"event-{i}") for i in range(10000)]
    document=SimpleNamespace(trace_id="t",content_digest="sha256:t",events=events)
    jobs=partition_jobs(document,Path("trace.json"),"annotator",None,None)
    assert len(jobs)==40
    assert jobs==partition_jobs(document,Path("trace.json"),"annotator",None,None)
    ids=[event for job in jobs for event in job['event_ids']]
    assert len(ids)==len(set(ids))==10000
    for job in jobs:
        assert len(job['event_ids'])<=250
        assert len({int(event.split('-')[1])%4 for event in job['event_ids']})==1


def test_luna_low_is_analysis_default_and_cli_requires_real_cost_enforcement():
    runner=JesterkyRunner()
    assert runner.default_model=='gpt-5.6-luna'
    assert runner.default_effort=='low'
    assert runner.cost_enforcement() is None


@pytest.mark.parametrize("journal_only", [False, True])
@pytest.mark.parametrize("retained_long", [False, True])
def test_retry_reuses_completed_workers_and_merges_every_output(tmp_path, monkeypatch, journal_only, retained_long):
    import json
    from types import SimpleNamespace
    import pytest
    from synth_containers.tracing.annotation.fixtures import build_craftax_smoke_trace
    import synth_containers.tracing.annotation.jesterky_runner as module
    trace=build_craftax_smoke_trace()
    if retained_long:
        import os
        from synth_containers.tracing.store.bundle import LocalTraceBundle
        from synth_containers.tracing.projections.inspector import load_bundle
        archive=os.environ.get('SYNTH_RESEARCH_LONG_ARCHIVE')
        if not archive:pytest.skip('opt-in retained 10,000-event four-actor archive')
        bundle=LocalTraceBundle.extract_archive(Path(archive),tmp_path/'source-bundle')
        trace=load_bundle(bundle.root)[0].trace
        assert len(trace.events)>=10000
        assert len({event.actor_id for event in trace.events if event.actor_id})>=4
    request=SimpleNamespace(annotator_id='test',metadata={'jesterky_window_events':250 if retained_long else 1},limits=AnnotationJobLimitsV1())
    context=SimpleNamespace(shard_cache_dir=tmp_path/'cache',workspace_dir=tmp_path/'workspace',document=trace,instructions_text='inspect',instructions_digest='sha256:instructions',entry=SimpleNamespace(program=SimpleNamespace(parameters={})),job=SimpleNamespace(job_id='job',request=request))
    seen=[]
    class Process:
        def __init__(self,argv,**kw):
            pending=json.loads(Path(argv[argv.index('--args-file')+1]).read_text())['jobs']
            seen.append([j['shard_id'] for j in pending])
            self.returncode=1 if len(seen)==1 else 0
            selected=pending[:1] if self.returncode else pending
            recorded=[]
            for i, job in enumerate(selected):
                proposal={'schema_version':PROPOSAL_SCHEMA_VERSION,'source_trace_id':trace.trace_id,'source_trace_digest':trace.content_digest,'findings':[],'abstentions':[],'judgments':[],'summary':job['shard_id']}
                recorded.append({'addr':{'node_path':[{'node':'annotate_jobs'},{'index':i}]},'outputs':{'proposal':proposal}})
            if journal_only and self.returncode:
                rows=[{'addr':{**r['addr'],'run_id':'ann-job'},'kind':{'kind':'actor_invoked'},'payload':{'outputs':r['outputs']}} for r in recorded]
                Path(argv[argv.index('--events-out')+1]).write_text('\n'.join(json.dumps(r) for r in rows)+'\n{')
            else:
                Path(argv[argv.index('--out')+1]).write_text(json.dumps({'recorded':recorded}))
        def communicate(self,timeout): return ('','')
    monkeypatch.setattr(module.subprocess,'Popen',Process)
    runner=JesterkyRunner()
    with pytest.raises(RuntimeError,match='successful proposals retained|without a manifest'):
        runner._run_cli(context,model='gpt-5.6-luna',effort='low',events=[])
    context.workspace_dir=tmp_path/'retry-workspace'
    context.job.job_id='retry-job'
    result=runner._run_cli(context,model='gpt-5.6-luna',effort='low',events=[])
    if retained_long:assert len(seen[0])>=40
    assert seen[0][0] not in seen[1]
    assert len(seen[1])==len(seen[0])-1
    assert len(result['summary'].splitlines())==len(seen[0])
    runner._run_cli(context,model='gpt-5.6-luna',effort='low',events=[])
    assert len(seen)==2  # completed receipt requires no third invocation


def test_inspection_mcp_uses_shared_bounded_tools():
    import json, urllib.request
    from synth_containers.tracing.annotation.jesterky_tools import serve_inspection_tools
    from synth_containers.tracing.annotation.tools import TraceInspectionTools
    from synth_containers.tracing.annotation.fixtures import build_craftax_smoke_trace
    tools = TraceInspectionTools(build_craftax_smoke_trace(), limits=AnnotationJobLimitsV1(max_tool_calls=1))
    with serve_inspection_tools(tools) as url:
        def call(method, params=None):
            req = urllib.request.Request(url, data=json.dumps({"jsonrpc":"2.0","id":1,"method":method,"params":params or {}}).encode(), headers={"Content-Type":"application/json"})
            return json.load(urllib.request.urlopen(req))["result"]
        assert call("initialize")["capabilities"] == {"tools":{}}
        assert any(t["name"] == "trace_resolve_selector" for t in call("tools/list")["tools"])
        assert not call("tools/call", {"name":"trace_get_manifest"}).get("isError")
        assert call("tools/call", {"name":"trace_get_manifest"})["isError"]
