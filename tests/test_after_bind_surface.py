"""AFTER bind surface: one eval stream for Craftax 10-lane, Harbor, dig.bench.

Not A1 (paid Luna), A2 (in-app GameBench), or A8 (public token). Docker daemon
and DIGBENCH_API_TOKEN are not required.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app
from synth_containers.platform.targets import TARGETS


TELEMETRY = {"enabled": True, "transport": "sse"}
_CRAFTAX_KINDS = {
    "frame",
    "observation",
    "action",
    "reward_signal",
    "env.episode.opened",
    "env.episode.closed",
}


def _prepare(client: TestClient, rollout_id: str) -> dict:
    prepared = client.post(
        "/rollouts/prepare",
        json={"rollout_id": rollout_id, "telemetry": TELEMETRY},
    )
    assert prepared.status_code == 200, prepared.text
    return prepared.json()


def _subscribe(client: TestClient, stream: dict) -> list[dict]:
    poll = client.get(stream["transports"]["poll"]["url"], params={"after": 0})
    assert poll.status_code == 200, poll.text
    events = poll.json().get("events") or []
    kinds = [row.get("kind") for row in events]
    assert "stream.subscribed" in kinds
    assert all(row.get("control") for row in events)
    assert all(row.get("sequence") is None for row in events)
    return events


def test_info_classifies_families_and_advertises_policy_refs() -> None:
    harbor = TestClient(create_compat_app("harbor_public")).get("/info").json()
    docker = TestClient(create_compat_app("harbor_docker")).get("/info").json()
    mock = TestClient(create_compat_app("digbench_mock")).get("/info").json()
    relay = TestClient(create_compat_app("digbench_public")).get("/info").json()
    echo = TestClient(create_compat_app("openenv_echo")).get("/info").json()

    assert echo["runtime_family"] == "openenv"
    assert echo["environment_ref"] == "env:echo"

    assert harbor["runtime_family"] == docker["runtime_family"] == "harbor"
    assert harbor["live_frames"] == docker["live_frames"] == "unsupported"
    assert harbor["environment_ref"] == "env:harbor_sandbox"
    assert docker["environment_ref"] == "env:harbor_docker"
    harbor_pins = {(row["harness"], row["config"]) for row in harbor["policy_refs"]}
    docker_pins = {(row["harness"], row["config"]) for row in docker["policy_refs"]}
    assert harbor_pins == docker_pins == {("harbor_fused", "luna_med"), ("harbor_fused", "sol_med")}

    assert mock["runtime_family"] == relay["runtime_family"] == "digbench"
    assert mock["live_frames"] == relay["live_frames"] == "unsupported"
    assert mock["environment_ref"] == "env:digbench_mock"
    assert relay["environment_ref"] == "env:digbench_relay"
    mock_pins = {(row["harness"], row["config"]) for row in mock["policy_refs"]}
    relay_pins = {(row["harness"], row["config"]) for row in relay["policy_refs"]}
    assert mock_pins == relay_pins == {
        ("react_legal_actions", "react_legal_actions"),
        ("codex", "agentic_codex"),
    }


def test_health_names_runtime_family() -> None:
    harbor = TestClient(create_compat_app("harbor_public")).get("/health").json()
    assert harbor["runtime_family"] == "harbor"
    assert harbor["environment_ref"] == "env:harbor_sandbox"
    assert harbor["target"] == "harbor_public"


def test_prepare_get_returns_prepared_before_start() -> None:
    client = TestClient(create_compat_app("harbor_public"))
    prepared = _prepare(client, "roll_prepared_status")
    stream = prepared["stream"]
    _subscribe(client, stream)
    status = client.get("/rollouts/roll_prepared_status")
    assert status.status_code == 200, status.text
    body = status.json()
    assert body["status"] == "prepared"
    assert body["started"] is False
    assert body["terminated"] is False
    assert body["stream"]["id"] == stream["id"]


def test_auto_transport_refused_on_authoritative_prepare() -> None:
    client = TestClient(create_compat_app("harbor_public"))
    refused = client.post(
        "/rollouts/prepare",
        json={"rollout_id": "roll_auto", "telemetry": {"enabled": True, "transport": "auto"}},
    )
    assert refused.status_code == 422, refused.text


def test_openenv_fifth_lease_is_typed_429() -> None:
    client = TestClient(create_compat_app("openenv_echo"))
    leases = TARGETS["openenv_echo"].scale_leases
    assert leases == 4
    held: list[str] = []
    try:
        for index in range(leases):
            started = client.post(
                "/rollouts",
                json={
                    "telemetry": TELEMETRY,
                    "submission_mode": "async",
                    "task_instance_id": f"seed:{index}",
                    "policy_ref": {"harness": "gym_loop", "config": "echo"},
                },
            )
            assert started.status_code == 200, started.text
            held.append(started.json()["rollout_id"])
        fifth = client.post(
            "/rollouts",
            json={
                "telemetry": TELEMETRY,
                "submission_mode": "async",
                "task_instance_id": "seed:10",
                "policy_ref": {"harness": "gym_loop", "config": "echo"},
            },
        )
        assert fifth.status_code == 429, fifth.text
        body = fifth.json()
        assert body["affordance"] == "scale_leases"
        assert body["scale_leases"] == 4
        assert "error" in body
    finally:
        for rollout_id in held:
            client.post(f"/rollouts/{rollout_id}/complete")


def test_harbor_fixture_subscribe_before_trial_and_no_frames() -> None:
    client = TestClient(create_compat_app("harbor_public"))
    prepared = _prepare(client, "roll_harbor_bind")
    _subscribe(client, prepared["stream"])
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": "roll_harbor_bind",
            "telemetry": TELEMETRY,
            "policy_ref": {"harness": "harbor_fused", "config": "luna_med"},
        },
    )
    assert started.status_code == 200, started.text
    events = client.get("/rollouts/roll_harbor_bind/events", params={"after": 0}).json()["events"]
    kinds = [row["kind"] for row in events if not row.get("control")]
    assert kinds[0] == "trace.opened"
    assert "trial.planned" in kinds
    assert "span.agent.opened" in kinds
    assert "span.verifier.opened" in kinds
    assert not _CRAFTAX_KINDS.intersection(kinds)
    native = next(row["payload"].get("reward.txt") for row in events if row["kind"] == "verifier")
    scored = client.post("/reward", json={"rollout_id": "roll_harbor_bind", "mode": "terminal"}).json()
    assert native == scored.get("reward") == 1.0


def test_harbor_docker_info_is_distinct_from_fixture() -> None:
    from synth_containers.platform.targets import PR_TARGETS

    info = TestClient(create_compat_app("harbor_docker")).get("/info").json()
    assert info["runtime_family"] == "harbor"
    assert info["environment_ref"] == "env:harbor_docker"
    assert info["live_frames"] == "unsupported"
    assert info["reward_authority"] == "trusted_scorer"
    assert "harbor_docker" not in PR_TARGETS


def test_digbench_mock_both_harnesses_share_one_eval_stream() -> None:
    client = TestClient(create_compat_app("digbench_mock"))
    prepared = _prepare(client, "roll_digbench_basic")
    _subscribe(client, prepared["stream"])
    status = client.get("/rollouts/roll_digbench_basic")
    assert status.json()["status"] == "prepared"
    basic = client.post(
        "/rollouts",
        json={
            "rollout_id": "roll_digbench_basic",
            "telemetry": TELEMETRY,
            "policy_ref": {"harness": "react_legal_actions", "config": "react_legal_actions"},
        },
    )
    assert basic.status_code == 200, basic.text
    basic_events = client.get("/rollouts/roll_digbench_basic/events", params={"after": 0}).json()["events"]
    basic_kinds = [row["kind"] for row in basic_events if not row.get("control")]
    basic_opened = next(row for row in basic_events if row["kind"] == "trace.opened")
    assert basic_opened["payload"]["policy_ref"] == {
        "harness": "react_legal_actions",
        "config": "react_legal_actions",
    }
    mutating = [kind for kind in basic_kinds if kind != "trace.opened"]
    assert mutating[0] == "start_session"
    assert "frame" not in basic_kinds
    assert "span.mcp.opened" not in basic_kinds
    basic_action = next(row for row in basic_events if row["kind"] == "action")
    assert basic_action["payload"]["action_authority"] == "harness_stub"
    blob = str(basic_events)
    assert "DIGBENCH_API_TOKEN" not in blob
    assert "Bearer " not in blob
    scored = client.post("/reward", json={"rollout_id": "roll_digbench_basic", "mode": "terminal"}).json()
    assert scored.get("reward") == 1.0

    agentic = client.post(
        "/rollouts",
        json={
            "rollout_id": "roll_digbench_agentic",
            "telemetry": TELEMETRY,
            "policy_ref": {"harness": "codex", "config": "agentic_codex"},
        },
    )
    assert agentic.status_code == 200, agentic.text
    assert agentic.json()["stream"]["id"] != basic.json()["stream"]["id"]
    agentic_events = client.get("/rollouts/roll_digbench_agentic/events", params={"after": 0}).json()["events"]
    agentic_kinds = [row["kind"] for row in agentic_events if not row.get("control")]
    agentic_opened = next(row for row in agentic_events if row["kind"] == "trace.opened")
    assert agentic_opened["payload"]["policy_ref"] == {
        "harness": "codex",
        "config": "agentic_codex",
    }
    assert "span.mcp.opened" in agentic_kinds
    assert "span.mcp.closed" in agentic_kinds
    assert "frame" not in agentic_kinds
    opened = agentic_kinds.index("span.mcp.opened")
    action = agentic_kinds.index("action")
    closed = agentic_kinds.index("span.mcp.closed")
    assert opened < action < closed
    agentic_mcp = next(row for row in agentic_events if row["kind"] == "span.mcp.opened")
    assert agentic_mcp["payload"]["evidence_class"] == "simulated"
    agentic_action = next(row for row in agentic_events if row["kind"] == "action")
    assert agentic_action["payload"]["action_authority"] == "harness_stub"
    assert agentic.json()["stream"]["id"].startswith("stream:")
    assert "optimizer" not in str(agentic_events).lower()


def test_digbench_public_without_token_allows_subscribe_before_start_session(
    monkeypatch,
) -> None:
    monkeypatch.delenv("DIGBENCH_API_TOKEN", raising=False)
    client = TestClient(create_compat_app("digbench_public"))
    prepared = _prepare(client, "roll_digbench_no_token")
    _subscribe(client, prepared["stream"])
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": "roll_digbench_no_token",
            "telemetry": TELEMETRY,
            "policy_ref": {"harness": "react_legal_actions", "config": "react_legal_actions"},
        },
    )
    assert started.status_code == 200, started.text
    events = client.get("/rollouts/roll_digbench_no_token/events", params={"after": 0}).json()["events"]
    kinds = [row["kind"] for row in events]
    assert "start_session" not in kinds
    status = next(row for row in events if row["kind"] == "status")
    assert status["payload"]["reason"] == "credential_missing"
    scored = client.post("/reward", json={"rollout_id": "roll_digbench_no_token", "mode": "terminal"})
    assert scored.status_code in {200, 202, 409}
    assert scored.json().get("reward") is None
    assert "DIGBENCH_API_TOKEN" not in str(events)
