"""Generic GEPA vendor parity and authoring-SDK route honesty."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from synth_containers import vendored_gepa_contract as vendored
from synth_containers.prompt_programs import gepa_optimizer_contract
from synth_containers.sdk import Container


OWNED_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "contracts"
    / "fixtures"
    / "gepa"
    / "gepa-contract-owned.json"
)


def _owned() -> dict[str, Any]:
    return json.loads(OWNED_FIXTURE.read_text(encoding="utf-8"))


def _contract_at_canonical_path(payload: dict[str, Any]) -> dict[str, Any]:
    node: Any = payload
    for key in vendored.GEPA_CONTRACT_METADATA_PATH:
        assert isinstance(node, dict), f"missing {key!r} on the canonical contract path"
        node = node.get(key)
    assert isinstance(node, dict), "no GEPA contract at the canonical metadata path"
    return node


def _assert_contract_shape(contract: dict[str, Any]) -> None:
    assert contract["version"] == vendored.GEPA_OPTIMIZER_CONTRACT_VERSION
    for route_field in vendored.GEPA_CONTRACT_REQUIRED_ROUTES:
        value = contract.get(route_field)
        assert isinstance(value, str) and value.startswith("/")


def test_vendored_constants_match_owned_values() -> None:
    owned = _owned()
    assert vendored.GEPA_OPTIMIZER_CONTRACT_VERSION == owned["GEPA_OPTIMIZER_CONTRACT_VERSION"]
    assert list(vendored.GEPA_CONTRACT_REQUIRED_ROUTES) == owned["GEPA_CONTRACT_REQUIRED_ROUTES"]
    assert list(vendored.GEPA_CONTRACT_OPTIONAL_ROUTES) == owned["GEPA_CONTRACT_OPTIONAL_ROUTES"]
    assert list(vendored.GEPA_CONTRACT_METADATA_PATH) == owned["GEPA_CONTRACT_METADATA_PATH"]


def test_prompt_program_contract_uses_vendor_values() -> None:
    _assert_contract_shape(gepa_optimizer_contract())


def test_sdk_advertises_only_a_complete_served_gepa_surface() -> None:
    container = Container("gepa-conformant")

    @container.program
    def _program() -> dict[str, Any]:
        return {"version": "prompt_program.v1", "program_id": "demo.v1", "modules": []}

    @container.taskset
    def _taskset() -> dict[str, Any]:
        return {"taskset_id": "demo.v1", "splits": {"train": 1}}

    @container.taskset_tasks
    def _taskset_tasks(payload: dict[str, Any]) -> dict[str, Any]:
        return {"tasks": [{"task_id": "seed:0", "seed": 0}], "metadata": {}}

    @container.rollout
    def _rollout(payload: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    app = container.fastapi()
    client = TestClient(app)
    contract = _contract_at_canonical_path(client.get("/metadata").json())
    _assert_contract_shape(contract)
    assert client.get(contract["program_route"]).status_code == 200
    assert client.get(contract["taskset_route"]).status_code == 200
    assert client.post(
        contract["taskset_tasks_route"], json={"task_ids": ["seed:0"]}
    ).status_code == 200
    assert contract["rollout_route"] in {getattr(route, "path", None) for route in app.routes}


def test_sdk_omits_gepa_for_partial_route_surface() -> None:
    container = Container("gepa-incomplete")

    @container.rollout
    def _rollout(payload: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    payload = TestClient(container.fastapi()).get("/metadata").json()
    assert "gepa" not in payload.get("metadata", {}).get("optimizer_contracts", {})
    assert "gepa" not in payload.get("optimizer_contracts", {})
    assert "gepa" not in payload["capabilities"].get("optimizer_contracts", {})
