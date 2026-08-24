from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app


def test_craftax_explicitly_advertises_workshop_live_eval_operations(tmp_path) -> None:
    client = TestClient(create_compat_app("craftax_engine", storage_root=tmp_path))
    payload = client.get("/info").json()
    capabilities = payload["capabilities"]

    assert capabilities["protocol"] == "synth.container.live-eval.v1"
    assert capabilities["operations"] == {
        "rollouts.prepare": True,
        "rollouts.start_prepared": True,
        "rollouts.get": True,
        "rollouts.poll": True,
        "reward.get": True,
        "trace_v5.capture": True,
    }
    assert capabilities["policy_refs"] == payload["policy_refs"]


def test_registered_policy_metadata_is_projected_without_credentials(tmp_path) -> None:
    client = TestClient(create_compat_app("craftax_engine", storage_root=tmp_path))
    response = client.post(
        "/policy-configs",
        json={
            "config_id": "luna_low",
            "config": {
                "model": "gpt-5.6-luna",
                "effort": "low",
                "api_key_env": "DO_NOT_PROJECT_THIS",
            },
        },
    )
    assert response.status_code == 200

    refs = client.get("/info").json()["capabilities"]["policy_refs"]
    # Dynamic configs are usable but are not silently added to the target's
    # allow-list. Projection enriches only explicitly advertised policy refs.
    assert all("api_key_env" not in row for row in refs)
