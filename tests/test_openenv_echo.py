"""OpenEnv Echo wrap: gym EnvironmentService, not a Harbor fold, not an image.

See: workshop/docs/container_compat.md C6 / §10.0 / §10.7
"""

from __future__ import annotations

import tempfile
from fastapi.testclient import TestClient

from synth_containers.compat.openenv import openenv_capability_surface
from synth_containers.platform import create_compat_app
from synth_containers.platform.echo_world import EchoWorld, prompt_for_seed
from synth_containers.platform.state import PolicyConfig


TELEMETRY = {"enabled": True, "transport": "sse", "retention": "run"}
TARGET = "openenv_echo"
POLICY = {"harness": "gym_loop", "config": "luna_med"}


def _client() -> TestClient:
    return TestClient(create_compat_app(TARGET, storage_root=tempfile.mkdtemp(prefix="test_openenv_echo-")))


def _prepare_start(client: TestClient, *, rollout_id: str, body: dict) -> dict:
    prepared = client.post(
        "/rollouts/prepare",
        json={"rollout_id": rollout_id, "telemetry": TELEMETRY},
    )
    assert prepared.status_code == 200, prepared.text
    stream = prepared.json()["stream"]
    before = client.get(stream["transports"]["poll"]["url"], params={"after": 0}).json()
    kinds = [row["kind"] for row in before["events"]]
    assert "stream.subscribed" in kinds
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": rollout_id,
            "telemetry": TELEMETRY,
            "slot": "stream",
            **body,
        },
    )
    assert started.status_code == 200, started.text
    return started.json()


def _events(client: TestClient, started: dict) -> list[dict]:
    return client.get(started["stream"]["transports"]["poll"]["url"], params={"after": 0}).json()[
        "events"
    ]


def test_echo_world_reset_and_step() -> None:
    world = EchoWorld()
    opened = world.reset(3)
    assert opened.done is False
    assert opened.reward is None
    assert opened.observation["text"] == prompt_for_seed(3) == "echo-3"
    assert opened.valid_action == opened.observation["text"]
    hit = world.step(opened.valid_action)
    assert hit.reward == 1.0
    assert hit.done is True
    miss = EchoWorld()
    miss.reset(3)
    wrong = miss.step("nope")
    assert wrong.reward == 0.0
    assert wrong.done is True
    assert not hasattr(EchoWorld, "snapshot")
    assert not hasattr(EchoWorld, "restore")
    assert not hasattr(EchoWorld, "fork")


def test_info_does_not_claim_checkpoint_or_frames() -> None:
    info = _client().get("/info").json()
    assert info["environment_ref"] == "env:echo"
    assert info["true_checkpoint"] == "unsupported"
    assert info["live_frames"] == "unsupported"
    assert "openenv" in info["adapter_chain"]
    assert info["runtime_family"] == "openenv"
    assert info["reward_authority"] == "environment"
    surface = openenv_capability_surface()
    assert surface.checkpoint_support is False
    assert surface.true_environment_snapshot is False


def test_matching_action_is_env_reward_one() -> None:
    client = _client()
    started = _prepare_start(
        client,
        rollout_id="echo_match",
        body={"task_instance_id": "seed:0", "policy_ref": POLICY},
    )
    events = _events(client, started)
    kinds = [row["kind"] for row in events]
    assert "frame" not in kinds
    assert {"observation", "action", "reward_signal", "status"} <= set(kinds)
    obs = next(row for row in events if row["kind"] == "observation")
    action = next(row for row in events if row["kind"] == "action")
    signal = next(row for row in events if row["kind"] == "reward_signal")
    assert obs["payload"]["text"] == prompt_for_seed(0)
    assert obs["payload"]["obs"]["text"] == prompt_for_seed(0)
    assert action["payload"]["action"] == prompt_for_seed(0)
    assert signal["payload"]["value"] == 1.0
    assert signal["payload"]["authority"] == "environment"
    scored = client.post("/reward", json={"rollout_id": "echo_match", "mode": "terminal"}).json()
    assert scored["reward"] == 1.0
    assert scored["status"] == "scored"
    node = scored["node_results"][0]
    assert node["authority"] == "environment"
    assert node["kind"] != "script"


def test_wrong_action_is_honest_zero() -> None:
    client = _client()
    # Echo does not advertise bind_policy_config (policy is the caller).
    client.app.state.platform.policy_configs["forced_wrong"] = PolicyConfig(
        config_id="forced_wrong",
        harness="gym_loop",
        config={"forced_action": "not-the-prompt"},
    )
    started = _prepare_start(
        client,
        rollout_id="echo_miss",
        body={
            "task_instance_id": "seed:0",
            "policy_ref": {"harness": "gym_loop", "config": "forced_wrong"},
        },
    )
    events = _events(client, started)
    signal = next(row for row in events if row["kind"] == "reward_signal")
    assert signal["payload"]["value"] == 0.0
    assert signal["payload"]["value"] is not None
    scored = client.post("/reward", json={"rollout_id": "echo_miss", "mode": "terminal"}).json()
    assert scored["reward"] == 0.0
    assert scored["status"] == "scored"
    assert scored["node_results"][0]["authority"] == "environment"
    assert scored["node_results"][0]["kind"] != "script"


def test_omit_reward_stays_null() -> None:
    client = _client()
    started = _prepare_start(
        client,
        rollout_id="echo_omit",
        body={
            "task_instance_id": "seed:2",
            "omit_reward": True,
            "policy_ref": POLICY,
        },
    )
    events = _events(client, started)
    signal = next(row for row in events if row["kind"] == "reward_signal")
    assert signal["payload"]["value"] is None
    scored = client.post("/reward", json={"rollout_id": "echo_omit", "mode": "terminal"}).json()
    assert scored["reward"] is None
    assert scored["status"] == "absent"
    assert scored["reward"] != 0
    assert scored["reward"] != 0.0


def test_log_has_no_frames() -> None:
    client = _client()
    started = _prepare_start(
        client,
        rollout_id="echo_no_frames",
        body={"task_instance_id": "seed:1", "policy_ref": POLICY},
    )
    events = _events(client, started)
    kinds = [row["kind"] for row in events]
    assert "frame" not in kinds
    assert "artifact.declared" not in kinds
    payloads = [row.get("payload") or {} for row in events]
    assert not any("ascii" in item or "grid" in item for item in payloads)
    assert "env.episode.opened" in kinds
    assert "env.episode.closed" in kinds
    assert kinds[-2:] == ["capture.high_water", "capture.closed"]
