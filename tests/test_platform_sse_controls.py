from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app


def test_prepared_sse_replays_control_once_before_first_semantic_event(tmp_path) -> None:
    """A prepared rollout must not flood its unsequenced control record."""

    app = create_compat_app("openenv_echo", storage_root=tmp_path)
    client = TestClient(app)
    rollout_id = "prepared-sse-control"
    prepared = client.post(
        "/rollouts/prepare",
        json={
            "rollout_id": rollout_id,
            "telemetry": {"enabled": True, "transport": "sse"},
        },
    )
    assert prepared.status_code == 200, prepared.text

    # Close without starting so the stream has only its unsequenced control
    # record and the response terminates deterministically in this test.
    app.state.platform.logs[rollout_id].mark_closed()
    stream_url = prepared.json()["stream"]["transports"]["sse"]["url"]
    with client.stream("GET", stream_url) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert body.count("event: stream.subscribed") == 1
