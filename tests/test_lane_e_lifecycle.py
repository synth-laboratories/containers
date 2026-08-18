"""Lane E: identity, capacity, owner-scoped cleanup, crash recovery, trace contract."""

from __future__ import annotations

from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app
from synth_containers.platform.http_requests import parse_create_rollout
from synth_containers.platform.state import CompatPlatform
from synth_containers.platform.targets import TARGETS

_TELEMETRY = {"enabled": True, "transport": "sse"}
_POLICY = {"harness": "react", "config": "luna_med"}


def _async_hold(owner: str, seed: int, rollout_id: str) -> dict:
    return {
        "rollout_id": rollout_id,
        "telemetry": _TELEMETRY,
        "submission_mode": "async",
        "execution": "on_complete",
        "task_instance_id": f"seed:{seed}",
        "policy_ref": _POLICY,
        "metadata": {"owner_id": owner, "owner_kind": "workshop_instance"},
    }


def test_health_reports_identity_and_truthful_capacity(tmp_path) -> None:
    app = create_compat_app(
        "craftax_engine",
        storage_root=tmp_path,
        runtime_config={"instance_id": "containers-lane-e"},
    )
    client = TestClient(app)
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["identity"]["instance_id"] == "containers-lane-e"
    assert health["identity"]["target_id"] == "craftax_engine"
    assert health["capacity"]["declared"] == 10
    assert health["capacity"]["active"] == 0
    assert health["capacity"]["reserved"] == 0
    started = client.post("/rollouts", json=_async_hold("ws-a", 0, "roll_e_cap_a"))
    assert started.status_code == 202, started.text
    health = client.get("/health").json()
    assert health["capacity"]["reserved"] == 1
    assert health["capacity"]["active"] == 0
    assert health["capacity"]["by_owner"]["ws-a"]["reserved"] == 1
    meta = client.get("/metadata").json()
    assert meta["identity"]["instance_id"] == "containers-lane-e"
    assert meta["capacity"]["declared"] == 10
    client.post("/rollouts/roll_e_cap_a/complete")


def test_concurrent_owners_and_owner_scoped_cleanup(tmp_path) -> None:
    client = TestClient(create_compat_app("craftax_engine", storage_root=tmp_path))
    first = client.post("/rollouts", json=_async_hold("owner-a", 0, "roll_e_a"))
    second = client.post("/rollouts", json=_async_hold("owner-b", 1, "roll_e_b"))
    assert first.status_code == 202, first.text
    assert second.status_code == 202, second.text
    assert first.json()["owner_id"] == "owner-a"
    assert second.json()["owner_id"] == "owner-b"
    refused = client.post(
        "/rollouts/roll_e_b/cancel",
        json={"owner_id": "owner-a"},
    )
    assert refused.status_code == 403, refused.text
    assert refused.json()["error"] == "owner_mismatch"
    still_b = client.get("/rollouts/roll_e_b").json()
    assert still_b["terminated"] is False
    assert still_b["owner_id"] == "owner-b"
    cleaned = client.post("/cleanup", json={"owner_id": "owner-a"})
    assert cleaned.status_code == 200, cleaned.text
    body = cleaned.json()
    assert "roll_e_a" in body["cancelled"]
    assert "roll_e_b" not in body["cancelled"]
    assert client.get("/rollouts/roll_e_a").json()["status"] == "cancelled"
    assert client.get("/rollouts/roll_e_b").json()["terminated"] is False
    client.post("/rollouts/roll_e_b/complete")


def test_crash_recovery_does_not_touch_unrelated_owners(tmp_path) -> None:
    first = create_compat_app(
        "craftax_engine",
        storage_root=tmp_path,
        runtime_config={"instance_id": "inst-1"},
    )
    client = TestClient(first)
    held = client.post("/rollouts", json=_async_hold("owner-live", 0, "roll_e_orphan"))
    assert held.status_code == 202, held.text
    completed = client.post(
        "/rollouts",
        json={
            "rollout_id": "roll_e_done",
            "telemetry": _TELEMETRY,
            "policy_ref": _POLICY,
            "task_instance_id": "seed:1",
            "metadata": {"owner_id": "owner-done"},
        },
    )
    assert completed.status_code == 200, completed.text
    assert completed.json()["terminated"] is True
    recovered = create_compat_app(
        "craftax_engine",
        storage_root=tmp_path,
        runtime_config={"instance_id": "inst-2"},
    )
    orphan = recovered.state.platform.pins["roll_e_orphan"]
    done = recovered.state.platform.pins["roll_e_done"]
    assert orphan.status == "crashed"
    assert orphan.owner_id == "owner-live"
    assert orphan.terminal is True
    assert done.status == "completed"
    assert done.owner_id == "owner-done"
    health = TestClient(recovered).get("/health").json()
    assert any(row["rollout_id"] == "roll_e_orphan" for row in health["crash_signals"])
    assert not any(row["rollout_id"] == "roll_e_done" for row in health["crash_signals"])


def test_trace_bundle_absent_is_honest_lite_seal_fallback(tmp_path) -> None:
    client = TestClient(create_compat_app("craftax_engine", storage_root=tmp_path))
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": "roll_e_trace",
            "telemetry": _TELEMETRY,
            "policy_ref": _POLICY,
            "task_instance_id": "seed:2",
        },
    )
    assert started.status_code == 200, started.text
    reference = started.json()["trace"]
    assert reference["url"] == "/rollouts/roll_e_trace/trace"
    assert reference["bundle_url"] == "/rollouts/roll_e_trace/trace/bundle"
    assert reference["kind"] == "lite_seal"
    assert reference["inspectable"] is False
    seal = client.get("/rollouts/roll_e_trace/trace")
    assert seal.status_code == 200, seal.text
    assert seal.json()["kind"] == "lite_seal"
    assert seal.json()["inspectable"] is False
    missing = client.get("/rollouts/roll_e_trace/trace/bundle")
    assert missing.status_code == 404, missing.text
    assert missing.json()["inspectable"] is False
    assert missing.json()["fallback"] == "/rollouts/roll_e_trace/trace"


def test_trace_bundle_served_when_capture_supervisor_archive_exists(tmp_path) -> None:
    client = TestClient(create_compat_app("craftax_engine", storage_root=tmp_path))
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": "roll_e_bundle",
            "telemetry": _TELEMETRY,
            "policy_ref": _POLICY,
            "task_instance_id": "seed:3",
        },
    )
    assert started.status_code == 200, started.text
    archive = tmp_path / "seals" / "roll_e_bundle.trace-bundle.zip"
    archive.write_bytes(b"PK\x03\x04fake-trace-v5-bundle")
    fetched = client.get("/rollouts/roll_e_bundle/trace/bundle")
    assert fetched.status_code == 200, fetched.text
    assert fetched.content.startswith(b"PK")
    assert fetched.headers["content-type"].startswith("application/zip")
    reference = client.get("/rollouts/roll_e_bundle").json()["trace"]
    assert reference["kind"] == "trace_v5_bundle"
    assert reference["inspectable"] is True


def test_policy_config_registration_is_idempotent_and_conflict_safe() -> None:
    platform = CompatPlatform(TARGETS["craftax_engine"])
    first = platform.register_policy_config(
        "checkpoint-e",
        {"harness": "react", "config": {"model": "checkpoint-e"}},
    )
    replay = platform.register_policy_config(
        "checkpoint-e",
        {"harness": "react", "config": {"model": "checkpoint-e"}},
    )
    assert first["config_id"] == "checkpoint-e"
    assert replay.get("replayed") is True
    conflict = platform.register_policy_config(
        "checkpoint-e",
        {"harness": "react", "config": {"model": "different"}},
    )
    assert conflict["status_code"] == 409
    assert conflict["error"] == "policy_config_conflict"


def test_usage_totals_match_emitted_events() -> None:
    platform = CompatPlatform(TARGETS["craftax_engine"])
    body = platform.start_rollout(
        parse_create_rollout(
            {
                "telemetry": _TELEMETRY,
                "policy_ref": _POLICY,
                "task_instance_id": "seed:4",
            }
        )
    )
    pin = platform.pins[body["rollout_id"]]
    log = platform.logs[body["rollout_id"]]
    opened = sum(1 for item in log.after(0) if item.kind == "span.policy.opened")
    reconciled = next(item for item in log.after(0) if item.kind == "usage.reconciled")
    assert pin.usage["llm_calls"] == opened == reconciled.payload["llm_calls"]
    assert pin.usage["llm_call_events"] == opened
    assert pin.usage["prompt_tokens"] is None
