"""eval_trace_view_v2 / eval_index_v1: validator, projector, ATIF export, CLI."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from synth_containers.tracing.cli import main as trace_cli_main
from synth_containers.tracing.models import (
    EVAL_INDEX_SCHEMA_VERSION,
    EVAL_VIEW_SCHEMA_VERSION,
    EvalEvaluatorV2,
    EvalFlagV2,
    EvalMediaLocationV2,
    EvalMediaV2,
    validate_eval_index,
    validate_eval_view,
)
from synth_containers.tracing.models.actors import ActorV5, SessionV5
from synth_containers.tracing.models.completeness import TraceCompletenessV5, TraceLifecycleV5
from synth_containers.tracing.models.document import TraceCaptureSummaryV5, TraceDocumentV5
from synth_containers.tracing.models.events import EventV5
from synth_containers.tracing.models.identity import TraceIdentityV5, TraceProvenanceV5
from synth_containers.tracing.models.messages import MessageNodeV5, MessagePartV5
from synth_containers.tracing.models.spans import SpanV5, UsageV5
from synth_containers.tracing.projections.eval_view import (
    project_eval_index_row,
    project_eval_view,
    to_atif,
)

T0 = "2026-01-01T00:00:00Z"


def v1_doc() -> dict:
    return {
        "schema": "eval_trace_view_v1",
        "source_schema": "trace_v5",
        "trace_id": "trc_1",
        "task": {"id": "t", "name": "Task", "family": "craftax"},
        "run": {"model": "m", "provider": None, "effort": "low", "seed": 1, "status": "completed",
                "duration_ms": 10, "cost_usd": 0.1, "usage": {"calls": 1, "input_tokens": 3, "output_tokens": 4}},
        "integrity": {"status": "sealed", "content_digest": "sha256:x", "source": "trace_v5"},
        "system_prompt": "hi",
        "steps": [{
            "id": "s1", "index": 0, "title": "Call 1", "turn_start": None,
            "tokens": {"input": 3, "output": 4},
            "content": {"message": "go"},
            "action": {"planned": ["LEFT"], "executed": [{"turn": 0, "name": "DOWN"}]},
            "reward": 1.0, "achievements": ["collect_wood"],
        }],
    }


def v2_doc() -> dict:
    return {
        "schema": EVAL_VIEW_SCHEMA_VERSION, "source_schema": "synth.trace.v5",
        "trace_id": "trc_1", "run_id": "r1",
        "task": {"id": "t", "name": "Task", "family": "craftax"},
        "run": {"model": "m", "status": "completed", "seed": 1},
        "cost": {"usd": None, "basis": "unknown", "source": "none"},
        "integrity": {"status": "sealed", "content_digest": "sha256:x", "source": "p"},
        "steps": [{
            "id": "s1", "index": 1, "kind": "policy_call", "timestamp": T0, "elapsed_ms": 0,
            "content": {"message": "go"},
            "action": {"planned": ["LEFT", "DOWN"], "executed": [{"turn": 0, "name": "DOWN"}]},
        }],
        "media": [{"id": "m1", "kind": "image", "format": "webp", "provenance": "native",
                   "location": {"pack": "packs/a.json", "key": "k"}}],
        "environment": {"renderer": "craftax", "version": 1, "frames": [{"turn": 0, "media_id": "m1"}]},
    }


def codes(errors) -> set[str]:
    return {e.code for e in errors}


def test_accepts_v1_and_v2_fixtures() -> None:
    assert validate_eval_view(v1_doc()) == []
    assert validate_eval_view(v2_doc()) == []


def test_v1_executed_outside_planned_still_valid() -> None:
    # v1 documents are valid v2 documents; the planned/executed rule is a v2 rule.
    assert validate_eval_view(v1_doc()) == []


def test_rejects_executed_not_planned() -> None:
    d = v2_doc()
    d["steps"][0]["action"]["executed"] = [{"turn": 0, "name": "JUMP"}]
    errors = validate_eval_view(d)
    assert codes(errors) == {"executed_not_planned"}
    assert errors[0].path == "$.steps[0].action.executed[0].name"


def test_executed_without_planned_is_fine() -> None:
    d = v2_doc()
    del d["steps"][0]["action"]["planned"]
    assert validate_eval_view(d) == []


def test_rejects_media_without_provenance_or_bad_provenance() -> None:
    d = v2_doc()
    del d["media"][0]["provenance"]
    assert "media_missing_provenance" in codes(validate_eval_view(d))
    d = v2_doc()
    d["media"][0]["provenance"] = "synthetic"
    assert "media_bad_provenance" in codes(validate_eval_view(d))


def test_rejects_v2_cost_without_basis() -> None:
    d = v2_doc()
    del d["cost"]["basis"]
    errors = validate_eval_view(d)
    assert "cost_missing_basis" in codes(errors)
    d = v2_doc()
    del d["cost"]
    assert "missing_field" in codes(validate_eval_view(d))


def test_media_location_variants() -> None:
    d = v2_doc()
    d["media"][0]["location"] = {"pack": "p", "offset": 0, "length": 5}
    assert validate_eval_view(d) == []
    d["media"][0]["location"] = {"pack": "p", "offset": 0}
    assert "bad_media_location" in codes(validate_eval_view(d))
    d["media"][0]["location"] = {"path": "a", "pack": "p"}
    assert "bad_media_location" in codes(validate_eval_view(d))


def row() -> dict:
    return {
        "id": "r1", "task": {"id": "t", "name": "T", "family": "f"}, "model": "m", "provider": None,
        "effort": None, "seed": 1, "protocol": "p", "status": "completed", "score": 1.0, "reward": 1.0,
        "success": True, "steps": 2, "calls": 1, "duration_ms": 5,
        "tokens": {"input": 0, "output": None, "cache_read": None},
        "cost": {"usd": None, "basis": "unknown"}, "media_status": "none", "flags": [],
    }


def index_doc() -> dict:
    return {"schema": EVAL_INDEX_SCHEMA_VERSION, "benchmark_id": "b", "environment": "craftax",
            "published_at": T0, "revision": "1", "runs": [row()]}


def test_index_validation() -> None:
    assert validate_eval_index(index_doc()) == []
    d = index_doc()
    del d["runs"][0]["media_status"]
    del d["runs"][0]["flags"]
    errors = validate_eval_index(d)
    assert {e.path for e in errors} == {"$.runs[0].media_status", "$.runs[0].flags"}
    assert codes(errors) == {"missing_field"}


def synthetic_v5() -> TraceDocumentV5:
    actor = ActorV5(actor_id="a1", kind="agent", display_name="a", model="gpt-x", provider="openai")
    session = SessionV5(session_id="s1", actor_id="a1", started_at=T0)
    sys_msg = MessageNodeV5(message_id="m0", role="system", sender_actor_id="a1", session_id="s1",
                            parts=(MessagePartV5(part_id="m0:0", type="text", text="be good"),))
    reply = MessageNodeV5(
        message_id="m1", role="assistant", sender_actor_id="a1", session_id="s1", occurred_at="2026-01-01T00:00:01Z",
        parts=(MessagePartV5(part_id="m1:0", type="reasoning", text="think"),
               MessagePartV5(part_id="m1:1", type="text", text="move"),
               MessagePartV5(part_id="m1:2", type="tool_call", tool_call_id="c1", tool_name="act",
                             arguments_json='{"a": 1}')))
    tool = MessageNodeV5(
        message_id="m2", role="tool", sender_actor_id="a1", session_id="s1", occurred_at="2026-01-01T00:00:02Z",
        parts=(MessagePartV5(part_id="m2:0", type="tool_result", tool_call_id="c1", text="ok"),))
    spans = (
        SpanV5(span_id="sp1", span_kind="model_call", actor_id="a1", session_id="s1",
               started_at="2026-01-01T00:00:01Z", input_message_ids=("m0",), output_message_ids=("m1",),
               usage=UsageV5(prompt_tokens=10, completion_tokens=5), detail={"call_index": 0, "reasoning_effort": "low"}),
        SpanV5(span_id="sp2", span_kind="model_call", actor_id="a1", session_id="s1",
               started_at="2026-01-01T00:00:03Z", detail={"call_index": 1}),
    )

    def ev(eid, etype, at, payload, **kw):
        return EventV5(event_id=eid, event_type=etype, actor_id="a1", session_id="s1", occurred_at=at,
                       payload=payload, **kw)

    events = (
        ev("e1", "environment.action_proposed", "2026-01-01T00:00:01Z", {"actions": ["LEFT", "DOWN"], "call_index": 0}),
        ev("e2", "environment.action_executed", "2026-01-01T00:00:02Z", {"action": "DOWN", "step_index": 0, "call_index": 0}),
        ev("e3", "environment.action_executed", "2026-01-01T00:00:04Z", {"kind": "action_applied", "action": "UP", "step_index": 1},
           span_id="sp2"),
        ev("e4", "environment.action_rejected", "2026-01-01T00:00:04Z", {"action": "FLY", "reason": "invalid", "call_index": 1}),
        ev("e5", "environment.reward", "2026-01-01T00:00:04Z", {"kind": "achievement_unlocked", "payload": {"achievement": "wood"}, "call_index": 1}),
        ev("e6", "environment.action_executed", "2026-01-01T00:00:05Z", {"action": "NOOP", "step_index": 2}, span_id="envspan"),
    )
    return TraceDocumentV5(
        trace_id="trc_1", trace_kind="evaluation_attempt",
        identity=TraceIdentityV5(run_id="r1", task_id="t1", benchmark="craftax", seed=7),
        lifecycle=TraceLifecycleV5(status="completed", started_at=T0, ended_at="2026-01-01T00:00:10Z"),
        capture=TraceCaptureSummaryV5(capture_id="cap", binding_id="b", binding_digest="d", capture_profile="p",
                                      interception="none", mode="m"),
        provenance=TraceProvenanceV5(producer="test", producer_version="1", model="gpt-x", provider="openai"),
        completeness=TraceCompletenessV5(capture_status="complete", terminal_event_observed=True),
        actors=(actor,), sessions=(session,), messages=(sys_msg, reply, tool), spans=spans, events=events,
        usage=UsageV5(prompt_tokens=10, completion_tokens=5, requests=2),
    ).sealed()


def test_projector_round_trip_and_rules() -> None:
    media = [EvalMediaV2(id="m1", kind="image", format="webp", provenance="native",
                         location=EvalMediaLocationV2(path="a.webp"))]
    evaluator = EvalEvaluatorV2(kind="reward", verdict="pass", score=1.0, reward=1.0,
                                flags=(EvalFlagV2("cost_estimated", "info", "est"),))
    view = project_eval_view(synthetic_v5(), environment="craftax", media=media, evaluator=evaluator)
    doc = json.loads(json.dumps(view.to_dict()))
    assert validate_eval_view(doc) == []
    assert doc["schema"] == EVAL_VIEW_SCHEMA_VERSION
    assert doc["cost"] == {"usd": None, "basis": "unknown", "source": "none"}
    assert doc["system_prompt"] == "be good"
    assert doc["run"]["seed"] == 7 and doc["run"]["effort"] == "low" and doc["run"]["duration_ms"] == 10000
    steps = doc["steps"]
    assert [s["index"] for s in steps] == list(range(1, len(steps) + 1))
    call1 = next(s for s in steps if s["id"] == "sp1")
    assert call1["content"] == {"reasoning": "think", "message": "move"}
    assert call1["action"]["planned"] == ["LEFT", "DOWN"]
    assert call1["action"]["executed"] == [{"name": "DOWN", "turn": 0}]
    assert call1["tool_calls"][0]["arguments"] == {"a": 1} and call1["tool_calls"][0]["result_ref"] == "m2"
    assert call1["tokens"] == {"input": 10, "output": 5}
    call2 = next(s for s in steps if s["id"] == "sp2")
    assert call2["action"]["executed"] == [{"name": "UP", "turn": 1}]
    assert call2["action"]["rejected"] == [{"name": "FLY", "reason": "invalid"}]
    assert "planned" not in call2["action"]  # nothing inferred from executed
    assert call2["achievements"] == ["wood"]
    assert "tokens" not in call2 and "message" not in call2["content"]
    orphan = next(s for s in steps if s["kind"] == "environment")
    assert orphan["action"]["executed"] == [{"name": "NOOP", "turn": 2}]
    row = project_eval_index_row(view, protocol="p")
    assert row["success"] is True and row["flags"] == ["cost_estimated"] and row["media_status"] == "none"
    index = {"schema": EVAL_INDEX_SCHEMA_VERSION, "benchmark_id": "b", "environment": "craftax",
             "published_at": T0, "revision": "1", "runs": [row]}
    assert validate_eval_index(index) == []


def test_index_refuses_unknown_required_facts_without_losing_view() -> None:
    view = project_eval_view(synthetic_v5(), environment="none")
    missing_seed = replace(view, run=replace(view.run, seed=None))
    assert "seed" not in missing_seed.to_dict()["run"]
    with pytest.raises(ValueError, match="code=missing_seed"):
        project_eval_index_row(missing_seed, protocol="p")

    missing_calls = replace(view, run=replace(view.run, usage=None))
    assert "usage" not in missing_calls.to_dict()["run"]
    with pytest.raises(ValueError, match="code=missing_calls"):
        project_eval_index_row(missing_calls, protocol="p")


def test_index_preserves_observed_zero_seed_and_call_count() -> None:
    from synth_containers.tracing.models.eval_view import EvalUsageV2

    view = project_eval_view(synthetic_v5(), environment="none")
    view = replace(view, run=replace(view.run, seed=0, usage=EvalUsageV2(calls=0)))
    row = project_eval_index_row(view, protocol="p")
    assert row["seed"] == 0
    assert row["calls"] == 0


def test_atif_step_ids_sequential_and_paired() -> None:
    view = project_eval_view(synthetic_v5(), environment="none")
    atif = to_atif(view)
    assert atif["schema_version"] == "ATIF-v1.7"
    assert [s["step_id"] for s in atif["steps"]] == list(range(1, len(atif["steps"]) + 1))
    assert atif["steps"][0]["source"] == "system"
    agent = next(s for s in atif["steps"] if s.get("tool_calls"))
    assert agent["source"] == "agent"
    assert agent["tool_calls"][0] == {"tool_call_id": "c1", "function_name": "act", "arguments": {"a": 1}}
    assert agent["observation"]["results"] == [{"source_call_id": "c1", "content": "ok"}]
    assert agent["extra"]["reasoning"] == "think"
    assert agent["metrics"] == {"prompt_tokens": 10, "completion_tokens": 5}
    assert not any(s["extra"]["kind"] == "tool_result" for s in atif["steps"] if "extra" in s)
    assert {s["source"] for s in atif["steps"]} <= {"system", "user", "agent"}


def test_cli_validate_eval_view(tmp_path: Path, capsys) -> None:
    good = tmp_path / "good.json"
    good.write_text(json.dumps(v1_doc()))
    assert trace_cli_main(["validate-eval-view", str(good)]) == 0
    bad = copy.deepcopy(v2_doc())
    del bad["media"][0]["provenance"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    assert trace_cli_main(["validate-eval-view", str(path)]) == 1
    assert "media_missing_provenance" in capsys.readouterr().out


def test_cli_project_eval_view_from_bundle(tmp_path: Path) -> None:
    from synth_containers.tracing.adapters.atif import import_atif
    from synth_containers.tracing.adapters.native import write_imported_document
    from synth_containers.tracing.canonical import bytes_digest
    from synth_containers.tracing.store.bundle import LocalTraceBundle

    trace = import_atif({
        "schema_version": "ATIF-v1.7", "trajectory_id": "ev-demo", "agent": {"name": "a", "version": "1"},
        "steps": [{"step_id": 1, "timestamp": T0, "source": "user", "message": "hi"},
                  {"step_id": 2, "timestamp": "2026-01-01T00:00:01Z", "source": "agent", "message": "ok"}],
    })
    bundle = LocalTraceBundle(tmp_path / "b")
    write_imported_document(trace, source_digest=bytes_digest(b"x"), source_format="ATIF-v1.7", bundle=bundle)
    out = tmp_path / "view.json"
    assert trace_cli_main(["project", "eval-view", str(tmp_path / "b"), "--environment", "none", "--output", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert validate_eval_view(doc) == []
    assert doc["environment"]["renderer"] == "none"
