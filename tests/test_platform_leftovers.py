from __future__ import annotations

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app
from synth_containers.platform.http_requests import parse_create_rollout
from synth_containers.platform.runtimes.harbor import atif_is_projection, project_harbor_atif
from synth_containers.platform.reducer import assert_honest_projection
from synth_containers.platform.state import CompatPlatform, _seed_from_task_instance_id
from synth_containers.platform.targets import TARGETS


def test_seed_from_task_instance_id_rules() -> None:
    assert _seed_from_task_instance_id(None) == 0
    assert _seed_from_task_instance_id("seed:7") == 7
    assert _seed_from_task_instance_id("occ:3") == 3
    assert _seed_from_task_instance_id("seed:ws") == 0
    assert _seed_from_task_instance_id("seed:occ:overflow") == 0
    with pytest.raises(ValueError, match="not an integer"):
        _seed_from_task_instance_id("seed:12abc")


def test_metadata_names_runtime_family() -> None:
    harbor = TestClient(create_compat_app("harbor_public")).get("/info").json()
    craftax = TestClient(create_compat_app("craftax_engine")).get("/info").json()
    banking77 = TestClient(create_compat_app("banking77_classify")).get("/info").json()
    assert harbor["runtime_family"] == "harbor"
    assert harbor["live_frames"] == "unsupported"
    assert harbor["adapter_chain"] == ["harbor"]
    assert {row["config"] for row in harbor["policy_refs"]} == {"luna_med", "sol_med"}
    assert craftax["runtime_family"] == "craftax"
    assert craftax["environment_ref"] == "env:craftax_fixture"
    assert craftax["max_episode_steps"] == 8
    assert "harbor" not in craftax.get("adapter_chain", [])
    react = TestClient(create_compat_app("craftax_react")).get("/info").json()
    assert react["environment_ref"] == "env:craftax_gold"
    assert react["max_episode_steps"] == 120
    assert banking77["runtime_family"] == "banking77"
    assert banking77["adapter_chain"] == []
    assert banking77["live_frames"] == "unsupported"


def test_start_rollout_reads_create_rollout_request() -> None:
    platform = CompatPlatform(TARGETS["craftax_engine"])
    req = parse_create_rollout(
        {
            "telemetry": {"enabled": True, "transport": "sse"},
            "task_instance_id": "seed:4",
            "omit_reward": True,
            "outcome": "demo",
            "evaluation_plan_ref": "eval:craftax.env_sum",
            "world_ref": "world:craftax",
            "submission_mode": "sync",
            "policy_ref": {"harness": "react", "config": "luna_med"},
        }
    )
    body = platform.start_rollout(req)
    pin = platform.pins[body["rollout_id"]]
    assert pin.seed == 4
    assert pin.task_instance_id == "seed:4"
    assert pin.omit_reward is True
    assert pin.outcome == "demo"
    assert pin.policy_ref["harness"] == "react"
    assert pin.policy_ref["config"] == "luna_med"
    assert body["terminated"] is True
    assert body["status"] == "completed"


def test_start_rollout_refuses_silent_policy_pin() -> None:
    client = TestClient(create_compat_app("craftax_engine"))
    missing = client.post("/rollouts", json={"telemetry": {"enabled": True, "transport": "sse"}})
    assert missing.status_code == 422
    assert "policy_ref.harness" in missing.json()["detail"]
    no_config = client.post(
        "/rollouts",
        json={
            "telemetry": {"enabled": True, "transport": "sse"},
            "policy_ref": {"harness": "react"},
        },
    )
    assert no_config.status_code == 422
    assert "policy_ref.config" in no_config.json()["detail"]


def test_craftax_policy_failure_closes_and_seals_the_stream(monkeypatch, tmp_path) -> None:
    def fail_after_open(*, log, planner, **_kwargs):
        log.append("policy.session.opened", dict(planner.metadata()))
        log.append("span.policy.opened", {"harness": "react"})
        raise RuntimeError("provider secret must never enter the event log")

    monkeypatch.setattr(
        "synth_containers.platform.runtimes.craftax.run_episode",
        fail_after_open,
    )
    platform = CompatPlatform(TARGETS["craftax_react"], storage_root=tmp_path)
    body = platform.start_rollout(
        parse_create_rollout(
            {
                "task_instance_id": "seed:0",
                "policy_ref": {"harness": "react", "config": "luna_med"},
            }
        )
    )

    assert body["status"] == "failed"
    assert body["terminated"] is True
    log = platform.logs[body["rollout_id"]]
    events = [item.to_dict() for item in log.after(0)]
    assert [item["kind"] for item in events][-2:] == ["capture.high_water", "capture.closed"]
    status = next(item for item in events if item["kind"] == "status")
    assert status["payload"] == {
        "status": "failed",
        "reason": "policy_error",
        "error_type": "RuntimeError",
    }
    assert "provider secret" not in str(events)
    assert log.closed is True
    assert body["rollout_id"] in platform.seals


def test_prepare_and_start_retries_replay_one_rollout_identity() -> None:
    client = TestClient(create_compat_app("craftax_engine"))
    prepared_body = {
        "rollout_id": "roll_retry_safe",
        "telemetry": {"enabled": True, "transport": "sse"},
    }
    first_prepare = client.post("/rollouts/prepare", json=prepared_body)
    retry_prepare = client.post("/rollouts/prepare", json=prepared_body)
    assert first_prepare.status_code == retry_prepare.status_code == 200
    assert retry_prepare.json()["replayed"] is True
    assert retry_prepare.json()["stream"] == first_prepare.json()["stream"]

    start_body = {
        **prepared_body,
        "task_instance_id": "seed:0",
        "policy_ref": {"harness": "react", "config": "luna_med"},
    }
    first_start = client.post("/rollouts", json=start_body)
    retry_start = client.post("/rollouts", json=start_body)
    assert first_start.status_code == retry_start.status_code == 200
    assert retry_start.json()["replayed"] is True
    assert retry_start.json()["rollout_id"] == "roll_retry_safe"

    status = client.get("/rollouts/roll_retry_safe")
    assert status.status_code == 200
    assert status.json()["started"] is True
    assert status.json()["terminated"] is True


def test_start_retry_refuses_changed_identity() -> None:
    client = TestClient(create_compat_app("craftax_engine"))
    original = {
        "rollout_id": "roll_identity_locked",
        "task_instance_id": "seed:1",
        "policy_ref": {"harness": "react", "config": "luna_med"},
        "telemetry": {"enabled": True, "transport": "sse"},
    }
    assert client.post("/rollouts", json=original).status_code == 200
    changed = {**original, "task_instance_id": "seed:2"}
    conflict = client.post("/rollouts", json=changed)
    assert conflict.status_code == 409
    assert conflict.json()["error"] == "rollout_identity_conflict"

    changed_code = {
        **original,
        "policy_ref": {"harness": "react", "config": "luna_med", "code": "changed"},
    }
    assert client.post("/rollouts", json=changed_code).status_code == 409

    changed_transport = {
        **original,
        "telemetry": {"enabled": True, "transport": "poll"},
    }
    assert client.post("/rollouts", json=changed_transport).status_code == 409


def test_harbor_atif_is_projection_of_the_log() -> None:
    client = TestClient(create_compat_app("harbor_public"))
    started = client.post(
        "/rollouts",
        json={
            "telemetry": {"enabled": True, "transport": "sse"},
            "policy_ref": {"harness": "harbor_fused", "config": "luna_med"},
        },
    ).json()
    rid = started["rollout_id"]
    events = client.get(f"/rollouts/{rid}/events", params={"after": 0}).json()["events"]
    scored = client.post("/reward", json={"rollout_id": rid})
    body = scored.json()
    node = next(item for item in body["node_results"] if item["node_id"] == "reward.txt")
    atif = project_harbor_atif(events)
    assert atif["reward.txt"] == node["value"]
    original = deepcopy(events)
    atif["reward.txt"] = 0.0
    atif["verifier"] = {"reward.txt": 0.0}
    assert project_harbor_atif(events)["reward.txt"] == node["value"]
    assert events == original
    assert atif_is_projection(events)


def test_headless_visual_consumer_craftax_vs_harbor() -> None:
    import importlib.util
    import tempfile
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "examples" / "headless_visual_consumer.py"
    spec = importlib.util.spec_from_file_location("headless_visual_consumer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        craftax = module.consume("craftax_engine", tmp)
        harbor = module.consume("harbor_public", tmp)
        digbench = module.consume("digbench_mock", tmp)
    assert craftax["ready"] is True
    assert harbor["ready"] is True
    assert digbench["ready"] is True
    assert craftax["slot"] == harbor["slot"] == digbench["slot"] == "stream"
    assert craftax["projection"]["has_live_frames"] is True
    assert craftax["projection"]["has_reward_txt"] is False
    assert harbor["projection"]["has_live_frames"] is False
    assert harbor["projection"]["has_reward_txt"] is True
    assert digbench["projection"]["has_live_frames"] is False
    assert digbench["projection"]["has_reward_txt"] is False
    assert not assert_honest_projection(craftax["projection"])
    assert not assert_honest_projection(harbor["projection"])
    assert not assert_honest_projection(digbench["projection"])


def test_optimizer_child_eval_refs_and_occupancy() -> None:
    client = TestClient(create_compat_app("harbor_public"))
    refs = []
    for index in range(2):
        started = client.post(
            "/rollouts",
            json={
                "telemetry": {"enabled": True, "transport": "sse"},
                "submission_mode": "async",
                "task_instance_id": f"seed:{index}",
                "policy_ref": {"harness": "harbor_fused", "config": "luna_med"},
            },
        )
        assert started.status_code == 200, started.text
        body = started.json()
        stream = body["stream"]
        refs.append(
            {
                "id": body["rollout_id"],
                "stream_id": stream["id"],
                "reward_url": stream["reward"]["url"],
            }
        )
    busy = client.post(
        "/rollouts",
        json={
            "telemetry": {"enabled": True, "transport": "sse"},
            "submission_mode": "async",
            "task_instance_id": "seed:2",
            "policy_ref": {"harness": "harbor_fused", "config": "luna_med"},
        },
    )
    assert busy.status_code == 429
    assert busy.json()["affordance"] == "scale_leases"
    assert refs[0]["id"] != refs[1]["id"]
    assert refs[0]["stream_id"] != refs[1]["stream_id"]


def test_c7_w06_trace_survives_world_stop() -> None:
    client = TestClient(create_compat_app("craftax_engine"))
    started = client.post(
        "/rollouts",
        json={
            "telemetry": {"enabled": True, "transport": "sse"},
            "slot": "stream",
            "policy_ref": {"harness": "react", "config": "luna_med"},
        },
    )
    assert started.status_code == 200, started.text
    rid = started.json()["rollout_id"]
    before = client.get(f"/rollouts/{rid}/trace")
    assert before.status_code == 200, before.text
    seal = before.json()
    stopped = client.post("/world/stop")
    assert stopped.status_code == 200, stopped.text
    after = client.get(f"/rollouts/{rid}/trace")
    assert after.status_code == 200, after.text
    again = after.json()
    assert again["trace_id"] == seal["trace_id"] == rid
    assert again["content_digest"] == seal["content_digest"]
    live = client.get(f"/rollouts/{rid}/events", params={"after": 0})
    assert live.status_code in {200, 404}


def test_policy_restart_is_independent_and_unproven_environment_restart_fails_closed() -> None:
    client = TestClient(create_compat_app("craftax_engine"))
    policy = client.post("/policy/restart")
    assert policy.status_code == 200, policy.text
    assert policy.json()["policy_generation"] == 2
    assert policy.json()["engine_generation"] == 1

    environment = client.post("/world/restart")
    assert environment.status_code == 409, environment.text
    assert environment.json()["error"] == "environment_restart_unsupported"
    assert environment.json()["policy_generation"] == 2


def test_craftax_ten_seeds_distinct_rewards_field() -> None:
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "examples" / "craftax_ten_seeds.py"
    spec = importlib.util.spec_from_file_location("craftax_ten_seeds", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    result = module.run()
    board = result["leaderboard"]
    assert len(board) == 10
    ids = [row["rollout_id"] for row in board]
    assert len(set(ids)) == 10
    assert all(row["subscribed_before_start"] is True for row in board)
    assert all(row["environment_ref"] == "env:craftax_fixture" for row in board)
    for row in board:
        if row.get("status") == "absent":
            assert row["reward"] is None
        else:
            assert row["reward"] is None or isinstance(row["reward"], (int, float))


def test_a_raising_rollout_does_not_wedge_later_policy_binds(monkeypatch, tmp_path) -> None:
    """A rollout that raises must terminalize its pin.

    `register_policy_config` refuses while any pin is started and not terminal.
    An exception raised before the runtime could record an outcome — a
    malformed policy config, for instance — used to leave the pin pinned
    forever, so every later bind returned 409 and the only recovery was
    restarting the container.
    """

    def explode(*_args, **_kwargs):
        raise RuntimeError("policy config is malformed")

    monkeypatch.setattr(
        "synth_containers.platform.runtimes.craftax.CraftaxRuntime.simulate",
        explode,
    )
    platform = CompatPlatform(TARGETS["craftax_react"], storage_root=tmp_path)
    request = parse_create_rollout(
        {
            "task_instance_id": "seed:0",
            "policy_ref": {"harness": "react", "config": "luna_med"},
        }
    )

    # The error still reaches the caller — it is not swallowed.
    with pytest.raises(RuntimeError, match="policy config is malformed"):
        platform.start_rollout(request)

    pin = next(iter(platform.pins.values()))
    assert pin.terminal is True
    assert pin.status == "failed"

    # ...and the container still accepts work.
    result = platform.register_policy_config(
        "next_policy", {"harness": "react", "config": {"model": "m"}}
    )
    assert result.get("error") != "in_flight", result
    assert result["config_id"] == "next_policy"
