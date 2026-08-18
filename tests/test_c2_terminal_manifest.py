"""C-2: write-once terminal execution manifest and contract digests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app
from synth_containers.platform.http_requests import parse_create_rollout
from synth_containers.platform.state import CompatPlatform, _canonical_sha256
from synth_containers.platform.targets import TARGETS

_POLICY = {"harness": "react", "config": "luna_med"}
_START = {
    "telemetry": {"enabled": True, "transport": "sse"},
    "policy_ref": _POLICY,
    "task_instance_id": "seed:0",
}


def test_terminal_manifest_is_self_digested_and_write_once(tmp_path) -> None:
    client = TestClient(create_compat_app("craftax_engine", storage_root=tmp_path))
    started = client.post("/rollouts", json={**_START, "rollout_id": "roll_c2_manifest"})
    assert started.status_code == 200, started.text
    body = started.json()
    assert body["config_digest"].startswith("sha256:")
    assert body["capability_digest"].startswith("sha256:")

    fetched = client.get("/rollouts/roll_c2_manifest/manifest")
    assert fetched.status_code == 200, fetched.text
    manifest = fetched.json()
    assert manifest["rollout_id"] == "roll_c2_manifest"
    assert manifest["status"] == "completed"
    assert manifest["digests"]["config_digest"] == body["config_digest"]
    assert manifest["digests"]["capability_digest"] == body["capability_digest"]
    assert manifest["trace_digest"] == body["trace"]["content_digest"]
    assert manifest["timestamps"]["completed_at"]
    assert "event_kind_counts" in manifest["taxonomy"]
    digest_body = {key: value for key, value in manifest.items() if key != "content_digest"}
    assert manifest["content_digest"] == _canonical_sha256(digest_body)

    path = tmp_path / "seals" / "roll_c2_manifest.manifest.json"
    assert path.is_file()
    original = path.read_text(encoding="utf-8")
    platform = client.app.state.platform
    pin = platform.pins["roll_c2_manifest"]
    pin.status = "tampered"
    platform._write_execution_manifest(pin, platform.logs["roll_c2_manifest"])
    assert path.read_text(encoding="utf-8") == original
    again = client.get("/rollouts/roll_c2_manifest/manifest").json()
    assert again["status"] == "completed"
    assert again["content_digest"] == manifest["content_digest"]


def test_seal_pin_carries_config_and_capability_digests() -> None:
    platform = CompatPlatform(TARGETS["craftax_engine"])
    body = platform.start_rollout(parse_create_rollout(_START))
    seal = platform.seals[body["rollout_id"]]
    assert seal["pin"]["config_digest"] == body["config_digest"]
    assert seal["pin"]["capability_digest"] == body["capability_digest"]
    missing = platform.get_execution_manifest("roll_absent")
    assert missing["status_code"] == 404
