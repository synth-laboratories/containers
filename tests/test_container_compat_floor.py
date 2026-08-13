"""C0–C2 honesty floor: OpenEnv checkpoints, missing≠0, durable sequence log, /reward."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from synth_containers.compat.openenv import openenv_capability_surface
from synth_containers.contracts import CheckpointResumeContract
from synth_containers.http_adapter import create_reference_app
from synth_containers.ontology import CheckpointSemantics, ExecutionProfile
from synth_containers.recovery import derive_run_recovery_projection
from synth_containers.reference_runtime import ReferenceManagedRuntime
from synth_containers.rubrics.v1 import _clamp_score, openenv_react_base_v1, verifier_result_from_mapping


def test_openenv_does_not_claim_checkpoints_by_default() -> None:
    surface = openenv_capability_surface()
    assert surface.checkpoint_support is False
    assert surface.true_environment_snapshot is False
    assert surface.checkpoint_semantics == CheckpointSemantics.NONE
    assert ExecutionProfile.CHECKPOINTABLE_LONG_HORIZON_ENVIRONMENT not in surface.profiles


def test_clamp_score_missing_stays_missing() -> None:
    assert _clamp_score(None) is None
    assert _clamp_score("") is None
    assert _clamp_score("not-a-number") is None
    assert _clamp_score(0.4) == 0.4


def test_verifier_absent_score_stays_absent() -> None:
    rubric = openenv_react_base_v1()
    result = verifier_result_from_mapping({}, rubric=rubric)
    assert result.score is None
    assert result.passed is None
    assert result.verdict == "absent"


def test_recovery_does_not_treat_missing_reward_as_zero() -> None:
    projection = derive_run_recovery_projection(
        run_id="run_missing",
        profile="test",
        checkpoint_resume=CheckpointResumeContract(),
        run_outcome="failed",
        run_phase="failed",
        checkpoints=[
            {"checkpoint_id": "ckpt_a", "resume_eligible": True},
            {"checkpoint_id": "ckpt_b", "resume_eligible": True, "reward": 1.5},
        ],
    )
    assert projection.highest_quality_recovery_point is not None
    assert projection.highest_quality_recovery_point.checkpoint_id == "ckpt_b"
    assert projection.highest_quality_recovery_point.reward == 1.5


def _client() -> TestClient:
    return TestClient(create_reference_app(ReferenceManagedRuntime.counter_default(target=1)))


def test_auto_transport_refused_when_telemetry_enabled() -> None:
    response = _client().post(
        "/rollouts",
        json={
            "submission_mode": "sync",
            "env": {"config": {"actions": ["increment", "stop"]}},
            "telemetry": {"enabled": True, "transport": "auto"},
        },
    )
    assert response.status_code == 422


def test_create_rollout_echoes_stream_descriptor_and_poll() -> None:
    client = _client()
    submitted = client.post(
        "/rollouts",
        json={
            "submission_mode": "sync",
            "env": {"config": {"actions": ["increment", "stop"]}},
            "telemetry": {"enabled": True, "transport": "sse", "poll_interval_ms": 100},
        },
    )
    assert submitted.status_code == 200
    stream = submitted.json()["stream"]
    rollout_id = submitted.json()["rollout_id"]
    assert stream["id"] == f"stream:{rollout_id}"
    assert stream["cursor"]["kind"] == "sequence"
    assert stream["transports"]["poll"]["url"] == f"/rollouts/{rollout_id}/events"
    assert stream["transports"]["sse"]["url"] == f"/rollouts/{rollout_id}/stream"
    assert stream["transports"]["websocket"] is None
    assert stream["reward"]["url"] == f"/rollouts/{rollout_id}/reward"
    assert stream["retention"] == "run"

    polled = client.get(stream["transports"]["poll"]["url"], params={"after": 0})
    assert polled.status_code == 200
    body = polled.json()
    kinds = [row["kind"] for row in body["events"]]
    assert "stream.subscribed" in kinds
    assert body["cursor"]["kind"] == "sequence"
    assert any(row.get("sequence") == 1 for row in body["events"] if not row.get("control"))


def test_poll_and_sse_share_sequence_and_reconnect() -> None:
    client = _client()
    submitted = client.post(
        "/rollouts",
        json={
            "submission_mode": "sync",
            "env": {"config": {"actions": ["increment", "stop"]}},
            "telemetry": {"enabled": True, "transport": "sse", "poll_interval_ms": 100},
        },
    )
    stream = submitted.json()["stream"]
    first = client.get(stream["transports"]["poll"]["url"], params={"after": 0}).json()
    sequences = [row["sequence"] for row in first["events"] if row.get("sequence") is not None]
    assert sequences
    high = max(sequences)
    with client.stream("GET", stream["transports"]["sse"]["url"], headers={"Last-Event-ID": str(high)}) as response:
        body = "".join(response.iter_text())
    assert response.status_code == 200
    # Reconnect after high_water should not mint a new semantic sequence.
    assert f"id: {high + 1}" not in body or "event: snapshot" not in body.split(f"id: {high + 1}")[-1]


def test_reward_get_absent_then_post() -> None:
    client = _client()
    submitted = client.post(
        "/rollouts",
        json={
            "submission_mode": "sync",
            "env": {"config": {"actions": ["increment", "stop"]}},
        },
    )
    rollout_id = submitted.json()["rollout_id"]
    absent = client.get("/reward", params={"rollout_id": rollout_id})
    assert absent.status_code == 200
    assert absent.json()["status"] == "absent"
    assert absent.json()["reward"] is None
    scored = client.post("/reward", json={"rollout_id": rollout_id, "mode": "terminal"})
    assert scored.status_code == 200
    assert scored.json()["reward"] is not None
    again = client.get(f"/rollouts/{rollout_id}/reward")
    assert again.json()["execution_id"] == scored.json()["execution_id"]


def test_reference_app_persists_trace_stream_schema_before_publish(tmp_path: Path) -> None:
    client = TestClient(
        create_reference_app(
            ReferenceManagedRuntime.counter_default(target=1),
            storage_root=tmp_path,
        )
    )
    submitted = client.post(
        "/rollouts",
        json={
            "rollout_id": "reference-durable",
            "submission_mode": "sync",
            "env": {"config": {"actions": ["increment", "stop"]}},
            "telemetry": {"enabled": True, "transport": "sse"},
        },
    )
    assert submitted.status_code == 200, submitted.text
    events = client.get("/rollouts/reference-durable/events", params={"after": 0}).json()[
        "events"
    ]
    assert events
    assert {row["schema"] for row in events} == {"synth.trace-stream-event.v1"}
    journal = next((tmp_path / "event_logs").glob("*.jsonl"))
    persisted = [json.loads(line) for line in journal.read_text().splitlines()]
    assert len(persisted) >= len(events)
    assert persisted[-1]["record"] == "closed"


def test_reference_prepare_binding_mismatch_refuses_before_rollout(tmp_path: Path) -> None:
    client = TestClient(
        create_reference_app(
            ReferenceManagedRuntime.counter_default(target=1),
            storage_root=tmp_path,
        )
    )
    prepared = client.post(
        "/rollouts/prepare",
        json={
            "rollout_id": "reference-poll-only",
            "telemetry": {"enabled": True, "transport": "poll", "retention": "run"},
        },
    )
    assert prepared.status_code == 200, prepared.text
    assert prepared.json()["stream"]["transports"]["sse"] is None
    assert client.get("/rollouts/reference-poll-only/stream").status_code == 404
    mismatch = client.post(
        "/rollouts",
        json={
            "rollout_id": "reference-poll-only",
            "submission_mode": "sync",
            "env": {"config": {"actions": ["increment", "stop"]}},
            "telemetry": {"enabled": True, "transport": "sse", "retention": "run"},
        },
    )
    assert mismatch.status_code == 409
    assert mismatch.json()["detail"]["error"] == "stream_binding_mismatch"
    prepared_status = client.get("/rollouts/reference-poll-only")
    assert prepared_status.status_code == 200
    assert prepared_status.json()["status"] == "prepared"
    assert prepared_status.json()["started"] is False

    invalid = client.post(
        "/rollouts/prepare",
        json={
            "rollout_id": "../../escape",
            "telemetry": {"enabled": True, "transport": "sse"},
        },
    )
    assert invalid.status_code == 422


def test_reference_prepare_ack_precedes_first_semantic_event(tmp_path: Path) -> None:
    client = TestClient(
        create_reference_app(
            ReferenceManagedRuntime.counter_default(target=1),
            storage_root=tmp_path,
        )
    )
    prepared = client.post(
        "/rollouts/prepare",
        json={
            "rollout_id": "reference-ready-first",
            "telemetry": {"enabled": True, "transport": "sse", "retention": "run"},
        },
    )
    assert prepared.status_code == 200, prepared.text
    stream = prepared.json()["stream"]
    assert "poll_url" not in stream
    assert "sse_url" not in stream
    assert "websocket_url" not in stream
    assert "stream.id" not in stream
    before = client.get(stream["transports"]["poll"]["url"], params={"after": 0}).json()["events"]
    assert [row["kind"] for row in before] == ["stream.subscribed"]
    assert before[0]["control"] is True
    assert "sequence" not in before[0]

    started = client.post(
        "/rollouts",
        json={
            "rollout_id": "reference-ready-first",
            "submission_mode": "sync",
            "env": {"config": {"actions": ["increment", "stop"]}},
            "telemetry": {"enabled": True, "transport": "sse", "retention": "run"},
        },
    )
    assert started.status_code == 200, started.text
    assert started.json()["rollout_id"] == "reference-ready-first"
    assert started.json()["stream"] == stream
    after = client.get(stream["transports"]["poll"]["url"], params={"after": 0}).json()["events"]
    semantic = [row for row in after if not row["control"]]
    assert semantic[0]["sequence"] == 1
    assert semantic[0]["kind"] == "eval.run.terminal"
    duplicate = client.post(
        "/rollouts",
        json={
            "rollout_id": "reference-ready-first",
            "submission_mode": "sync",
            "env": {"config": {"actions": ["increment", "stop"]}},
            "telemetry": {"enabled": True, "transport": "sse", "retention": "run"},
        },
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["replayed"] is True
    assert duplicate.json()["rollout_id"] == "reference-ready-first"

    changed = client.post(
        "/rollouts",
        json={
            "rollout_id": "reference-ready-first",
            "submission_mode": "sync",
            "env": {"config": {"actions": ["stop"]}},
            "telemetry": {"enabled": True, "transport": "sse", "retention": "run"},
        },
    )
    assert changed.status_code == 409
    assert changed.json()["detail"]["error"] == "rollout_identity_conflict"


def test_reference_openapi_advertises_only_canonical_rollouts_route() -> None:
    client = TestClient(create_reference_app(ReferenceManagedRuntime.counter_default(target=1)))
    paths = client.get("/openapi.json").json()["paths"]
    assert "/rollouts" in paths
    assert "/rollout" not in paths
