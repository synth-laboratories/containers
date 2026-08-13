"""Tests for container-facing rollout annotations route."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from synth_containers.annotations import (
    ROLLOUT_ANNOTATIONS_SCHEMA,
    RolloutAnnotation,
    annotation_list,
    derive_annotations_from_execution,
    make_annotation,
)
from synth_containers.http_adapter import create_reference_app
from synth_containers.nouns import ExecutionRecord, Outcome
from synth_containers.ontology import OutcomeKind
from synth_containers.reference_runtime import ReferenceManagedRuntime


def test_derive_annotations_from_execution_covers_core_kinds() -> None:
    execution = ExecutionRecord(
        execution_id="roll_test",
        trace_correlation_id="corr_test",
        status="completed",
        success_status="success",
        created_at="2026-08-03T00:00:00Z",
        updated_at="2026-08-03T00:01:00Z",
        outcome=Outcome(kind=OutcomeKind.REWARD, reward=2.0, passed=True),
        summary={
            "achievements": {"collect_drink": True, "collect_sapling": False},
            "generated_tokens": 400,
            "reasoning_tokens": 0,
            "llm_calls": 9,
            "actions": 40,
            "invalid_actions": 2,
            "effective_noops": 18,
            "termination_reason": "died",
            "outcome_reward": 1.0,
            "terminal_inventory": {
                "health": -1.0,
                "energy": 9,
                "drink": 8,
                "wood": 2,
                "stone": 1,
                "iron": 0,
                "tools": {"pickaxe": 1, "sword": 0, "bow": 0},
            },
            "survival": {
                "health_start": 9.0,
                "health_end": -1.0,
                "health_min": -1.0,
                "died": True,
                "death_reason": "death",
                "death_source": "zombie",
            },
            "achievement_unlocks": {
                "unique_achievements": 1.0,
                "unlocked": ["collect_drink"],
                "first_unlock_step": {"collect_drink": 14},
            },
            "progress": {
                "deepest_floor": 1,
                "final_floor": 1,
                "boss_progress": 2,
                "boss_progress_max": 2,
                "floor_state": {"monsters_killed": 3},
            },
        },
        usage={"prompt_tokens": 1200, "completion_tokens": 400},
        metadata={},
    )
    bundle = derive_annotations_from_execution(execution)
    assert bundle.schema == ROLLOUT_ANNOTATIONS_SCHEMA
    assert bundle.rollout_id == "roll_test"
    assert bundle.status == "ready"
    assert bundle.count == len(bundle.annotations)
    kinds = {row.kind for row in bundle.annotations}
    assert kinds >= {
        "achievement",
        "termination",
        "token_stats",
        "action_stats",
        "terminal_inventory",
        "survival",
        "achievement_unlocks",
        "progress",
        "outcome",
    }
    achievement = next(row for row in bundle.annotations if row.kind == "achievement")
    assert achievement.payload["unique_achievements"] == 1.0
    assert "collect_drink" in achievement.labels
    assert achievement.payload["first_unlock_step"]["collect_drink"] == 14
    action = next(row for row in bundle.annotations if row.kind == "action_stats")
    assert action.payload["actions_per_llm"] == pytest.approx(40 / 9)
    assert action.payload["invalid_frac"] == pytest.approx(2 / 40)
    assert action.payload["noop_frac"] == pytest.approx(18 / 40)
    survival = next(row for row in bundle.annotations if row.kind == "survival")
    assert survival.payload["died"] is True
    assert survival.payload["death_source"] == "zombie"
    progress = next(row for row in bundle.annotations if row.kind == "progress")
    assert progress.payload["deepest_floor"] == 1


def test_extract_env_progress_from_craftax_transitions() -> None:
    from synth_containers.annotations import extract_env_progress

    events = [
        {
            "event_type": "environment.transition",
            "data": {
                "step": 1,
                "new_achievements": [],
                "observation_after": {
                    "observation": {
                        "inventory": {"health": 9.0, "energy": 9, "drink": 9, "boss_progress": 0},
                        "player": {"level": 0},
                        "floor_state": {"monsters_killed": 0},
                    }
                },
            },
        },
        {
            "event_type": "environment.transition",
            "data": {
                "step": 10,
                "new_achievements": ["collect_drink"],
                "native_events": [
                    {
                        "kind": "combat",
                        "transition": "mob_attack",
                        "payload": {"entity": {"kind": "skeleton"}},
                    },
                    {"kind": "death", "transition": "death", "payload": {"reason": "death"}},
                ],
                "observation_after": {
                    "observation": {
                        "inventory": {
                            "health": -1.0,
                            "energy": 8,
                            "drink": 7,
                            "wood": 3,
                            "pickaxe": 1,
                            "sword": 0,
                            "bow": 0,
                            "boss_progress": 1,
                        },
                        "player": {"level": 2, "pos": [4, 5]},
                        "floor_state": {"monsters_killed": 4, "chests_opened": 1},
                    }
                },
            },
        },
    ]
    progress = extract_env_progress(events)
    assert progress["terminal_inventory"]["health"] == -1.0
    assert progress["terminal_inventory"]["wood"] == 3
    assert progress["survival"]["died"] is True
    assert progress["survival"]["death_source"] == "skeleton"
    assert progress["achievement_unlocks"]["first_unlock_step"]["collect_drink"] == 10
    assert progress["progress"]["deepest_floor"] == 2
    assert progress["progress"]["boss_progress"] == 1



def test_extract_behavior_diagnostics_from_turns_and_events() -> None:
    from synth_containers.annotations import extract_behavior_diagnostics

    turns = [
        {
            "actions": ["noop", "noop", "noop", "noop", "noop"],
            "invalid_parse": False,
            "prompt_continuity": {
                "kind": "initial",
                "rendered_tokens": 1000,
            },
        },
        {
            "actions": ["do", "place_table"],
            "invalid_parse": True,
            "prompt_continuity": {
                "kind": "fork",
                "rendered_tokens": 350,
            },
        },
    ]
    events = [
        {
            "event_type": "environment.transition",
            "data": {"noop_reasons": ["missing_potion"], "invalid_codes": []},
        },
        {
            "event_type": "agent.context_compacted",
            "data": {"dropped_item_count": 5, "retained_item_count": 3},
        },
    ]
    behavior = extract_behavior_diagnostics(
        events,
        turns,
        summary={"llm_calls": 2, "invalid_parse_turn_count": 1},
    )
    assert behavior["action_histogram"]["counts"]["noop"] == 5
    assert behavior["batch_size_hist"]["counts"]["5"] == 1
    assert behavior["transition_diagnostics"]["noop_reasons"]["missing_potion"] == 1
    assert behavior["parse_stats"]["invalid_parse_rate"] == 0.5
    assert behavior["prompt_continuity"]["counts"]["fork"] == 1
    assert behavior["compaction"]["compaction_count"] == 1
    assert behavior["compaction"]["prompt_tokens_delta_mean"] == -650


def test_annotations_route_default_derivation() -> None:
    runtime = ReferenceManagedRuntime.counter_default(target=2)
    app = create_reference_app(runtime)
    client = TestClient(app)
    submitted = client.post(
        "/rollouts",
        json={
            "trace_correlation_id": "ann_default",
            "submission_mode": "sync",
            "env": {"config": {"actions": ["increment", "increment", "stop"]}},
        },
    )
    assert submitted.status_code == 200
    rollout_id = submitted.json()["rollout_id"]
    response = client.get(f"/rollouts/{rollout_id}/annotations")
    assert response.status_code == 200
    payload = response.json()
    assert payload["schema"] == ROLLOUT_ANNOTATIONS_SCHEMA
    assert payload["rollout_id"] == rollout_id
    assert payload["count"] == len(payload["annotations"])
    assert payload["status"] in {"ready", "unavailable"}
    assert any(row["kind"] == "outcome" for row in payload["annotations"])


def test_rollout_telemetry_advertises_and_streams_sse() -> None:
    client = TestClient(create_reference_app(ReferenceManagedRuntime.counter_default(target=1)))
    submitted = client.post("/rollouts", json={
        "submission_mode": "sync",
        "env": {"config": {"actions": ["increment", "stop"]}},
        "telemetry": {"enabled": True, "transport": "sse", "poll_interval_ms": 100},
    })
    assert submitted.status_code == 200
    stream = submitted.json()["stream"]
    assert stream["schema"] == "synth.rollout.stream.v1"
    with client.stream("GET", stream["transports"]["sse"]["url"]) as response:
        body = "".join(response.iter_text())
    assert response.status_code == 200
    assert "event: eval.run.terminal" in body
    assert "synth.trace-stream-event.v1" in body


def test_rollout_telemetry_websocket_uses_same_event_schema() -> None:
    client = TestClient(create_reference_app(ReferenceManagedRuntime.counter_default(target=1)))
    submitted = client.post("/rollouts", json={
        "submission_mode": "sync",
        "env": {"config": {"actions": ["increment", "stop"]}},
        "telemetry": {"enabled": True, "transport": "websocket", "poll_interval_ms": 100},
    })
    with client.websocket_connect(submitted.json()["stream"]["transports"]["websocket"]["url"]) as socket:
        event = socket.receive_json()
        while event.get("kind") != "eval.run.terminal":
            event = socket.receive_json()
    assert event["schema"] == "synth.trace-stream-event.v1"
    assert event["kind"] == "eval.run.terminal"


def test_annotations_route_custom_runtime_override() -> None:
    class CustomRuntime(ReferenceManagedRuntime):
        async def get_rollout_annotations(self, rollout_id: str) -> Any:
            return annotation_list(
                rollout_id,
                [
                    make_annotation(
                        kind="teacher_action",
                        rollout_id=rollout_id,
                        source="teacher",
                        labels=["craftax_interact"],
                        payload={"actions": ["noop", "do", "sleep"]},
                        ok=True,
                    )
                ],
                status="ready",
                trace_correlation_id="custom",
            )

    runtime = CustomRuntime.counter_default(target=1)
    app = create_reference_app(runtime)
    client = TestClient(app)
    submitted = client.post(
        "/rollouts",
        json={
            "trace_correlation_id": "ann_custom",
            "submission_mode": "sync",
            "env": {"config": {"actions": ["increment", "stop"]}},
        },
    )
    assert submitted.status_code == 200
    rollout_id = submitted.json()["rollout_id"]
    response = client.get(f"/rollouts/{rollout_id}/annotations")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["annotations"][0]["kind"] == "teacher_action"
    assert payload["annotations"][0]["source"] == "teacher"


def test_annotations_route_404_unknown_rollout() -> None:
    runtime = ReferenceManagedRuntime.counter_default()
    client = TestClient(create_reference_app(runtime))
    response = client.get("/rollouts/does-not-exist/annotations")
    assert response.status_code == 404
    assert "unknown_rollout" in response.json()["detail"]


def test_route_hints_include_annotations() -> None:
    runtime = ReferenceManagedRuntime.counter_default()
    hints = runtime.metadata().capabilities.route_hints.to_dict()
    assert "/rollouts/{rollout_id}/annotations" in hints["annotation_routes"]


def test_make_annotation_round_trip_dict() -> None:
    row = make_annotation(
        kind="token_stats",
        rollout_id="r1",
        payload={"generated_tokens": 10},
        ok=True,
    )
    assert isinstance(row, RolloutAnnotation)
    payload = row.to_dict()
    assert payload["kind"] == "token_stats"
    assert payload["payload"]["generated_tokens"] == 10
