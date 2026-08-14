from __future__ import annotations

from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app
from synth_containers.tasks import DatasetSplit


def test_craftax_search_family_has_frozen_disjoint_heldout_split() -> None:
    client = TestClient(create_compat_app("craftax_react"))
    manifest = client.get("/dataset").json()
    assert manifest["dataset_id"] == "craftax_singleplayer_search_v1"
    assert manifest["is_reference_world"] is True
    assert manifest["objective"]["primary_metric"] == "craftax_env_sum"
    train = set(manifest["splits"]["train"]["seeds"])
    heldout = set(manifest["splits"]["heldout"]["seeds"])
    assert train.isdisjoint(heldout)
    assert manifest["splits"]["heldout"]["frozen"] is True

    response = client.post("/dataset/rows", json={"split": "heldout", "seeds": [501, 520]})
    assert response.status_code == 200
    rows = response.json()["rows"]
    assert [row["seed"] for row in rows] == [501, 520]
    assert all(row["example"]["is_reference_world"] is True for row in rows)
    assert all(row["task_family"] == "craftax_singleplayer_search_v1" for row in rows)


def test_rogue_search_family_publishes_graded_objective() -> None:
    client = TestClient(create_compat_app("rogue_react"))
    manifest = client.get("/dataset").json()
    objective = manifest["objective"]
    assert manifest["dataset_id"] == "rogue_singleplayer_search_v1"
    assert objective["primary_metric"] == "rogue_graded_progress"
    assert objective["source_field"].endswith("synth_shaped_reward")
    assert objective["components"]["depth_reached"] == 100.0
    assert set(manifest["splits"]["train"]["seeds"]).isdisjoint(
        manifest["splits"]["heldout"]["seeds"]
    )


def test_dataset_rejects_seed_outside_frozen_split() -> None:
    client = TestClient(create_compat_app("rogue_react"))
    response = client.post("/dataset/rows", json={"split": "heldout", "seeds": [1]})
    assert response.status_code == 422
    assert "not members" in response.json()["detail"]


def test_heldout_is_a_first_class_dataset_split() -> None:
    assert DatasetSplit.parse("heldout") is DatasetSplit.HELDOUT
