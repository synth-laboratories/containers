"""The vendored GEPA sub-contract: parity with the owned values, honest routes.

Optimizers own the GEPA sub-contract (boundary B2);
``synth_containers/vendored_gepa_contract.py`` is a vendor copy checked here
against the committed owned values in
``contracts/fixtures/gepa/gepa-contract-owned.json``.  The conformance tests
assert the producers advertise exactly what the owned contract requires and
actually serve every route they declare (the drift the optimizers corpus test
caught: banking77 omitted ``taskset_tasks_route`` and 404'd ``/program`` and
``/taskset`` while advertising them).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from synth_containers import vendored_gepa_contract as vendored
from synth_containers.platform import create_compat_app
from synth_containers.prompt_programs import gepa_optimizer_contract
from synth_containers.sdk import Container

REPO_ROOT = Path(__file__).resolve().parents[1]
OWNED_FIXTURE = REPO_ROOT / "contracts" / "fixtures" / "gepa" / "gepa-contract-owned.json"


def _owned() -> dict[str, Any]:
    return json.loads(OWNED_FIXTURE.read_text(encoding="utf-8"))


def test_vendored_constants_match_the_owned_values() -> None:
    owned = _owned()
    assert vendored.GEPA_OPTIMIZER_CONTRACT_VERSION == owned["GEPA_OPTIMIZER_CONTRACT_VERSION"]
    assert list(vendored.GEPA_CONTRACT_REQUIRED_ROUTES) == owned["GEPA_CONTRACT_REQUIRED_ROUTES"]
    assert list(vendored.GEPA_CONTRACT_OPTIONAL_ROUTES) == owned["GEPA_CONTRACT_OPTIONAL_ROUTES"]
    assert list(vendored.GEPA_CONTRACT_METADATA_PATH) == owned["GEPA_CONTRACT_METADATA_PATH"]


def test_prompt_programs_mints_from_the_vendored_constant() -> None:
    contract = gepa_optimizer_contract()
    assert contract["version"] == vendored.GEPA_OPTIMIZER_CONTRACT_VERSION
    _assert_contract_shape(contract)


def _contract_at_canonical_path(payload: dict[str, Any]) -> dict[str, Any]:
    node: Any = payload
    for key in vendored.GEPA_CONTRACT_METADATA_PATH:
        assert isinstance(node, dict), f"missing {key!r} on the canonical contract path"
        node = node.get(key)
    assert isinstance(node, dict), "no gepa contract at the canonical metadata path"
    return node


def _assert_contract_shape(contract: dict[str, Any]) -> None:
    assert contract["version"] == vendored.GEPA_OPTIMIZER_CONTRACT_VERSION
    for route_field in vendored.GEPA_CONTRACT_REQUIRED_ROUTES:
        value = contract.get(route_field)
        assert isinstance(value, str) and value.startswith("/"), (
            f"{route_field} must be an absolute path, got {value!r}"
        )


def _assert_routes_served(client: TestClient, app: Any, contract: dict[str, Any]) -> None:
    """Every declared required route must exist (404 = advertised but not served)."""

    assert client.get(contract["program_route"]).status_code == 200
    assert client.get(contract["taskset_route"]).status_code == 200
    tasks = client.post(contract["taskset_tasks_route"], json={"task_ids": ["seed:0"]})
    assert tasks.status_code == 200
    (task,) = tasks.json()["tasks"]
    assert task["seed"] == 0
    # The rollout route is proven by registration, not by POSTing: a probe
    # body could start a real rollout on a machine with credentials present.
    registered = {getattr(route, "path", None) for route in app.routes}
    assert contract["rollout_route"] in registered


@pytest.mark.parametrize("target", ["banking77_classify", "healthbench_chat"])
def test_platform_gepa_families_serve_every_declared_route(target: str, tmp_path) -> None:
    app = create_compat_app(target, storage_root=tmp_path / target)
    client = TestClient(app)
    payload = client.get("/metadata").json()
    contract = _contract_at_canonical_path(payload)
    _assert_contract_shape(contract)
    _assert_routes_served(client, app, contract)


def test_banking77_program_is_derived_from_the_classify_seed(tmp_path) -> None:
    from synth_containers.platform.banking77_world import CLASSIFY_SYSTEM

    client = TestClient(create_compat_app("banking77_classify", storage_root=tmp_path / "b"))
    program = client.get("/program").json()
    assert program["program_id"] == "banking77.classify.v1"
    assert program["seed_candidate"] == {"system_prompt": CLASSIFY_SYSTEM}
    (module,) = program["modules"]
    assert module["mutable"] is True
    assert module["content"] == CLASSIFY_SYSTEM
    taskset = client.get("/taskset").json()
    assert taskset["taskset_id"] == "banking77.classify.v1"
    assert taskset["splits"]["train"] > 0
    assert taskset["splits"]["heldout"] > 0


def test_sdk_container_contract_conforms_when_routes_are_registered() -> None:
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
    payload = client.get("/metadata").json()
    contract = _contract_at_canonical_path(payload)
    _assert_contract_shape(contract)
    assert client.get(contract["program_route"]).status_code == 200
    assert client.get(contract["taskset_route"]).status_code == 200
    assert (
        client.post(contract["taskset_tasks_route"], json={"task_ids": ["seed:0"]}).status_code
        == 200
    )
    registered = {getattr(route, "path", None) for route in app.routes}
    assert contract["rollout_route"] in registered


def test_sdk_container_does_not_advertise_gepa_for_partial_route_surface() -> None:
    container = Container("gepa-incomplete")

    @container.rollout
    def _rollout(payload: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    payload = TestClient(container.fastapi()).get("/metadata").json()
    assert "gepa" not in payload.get("metadata", {}).get("optimizer_contracts", {})
    assert "gepa" not in payload.get("optimizer_contracts", {})
    assert "gepa" not in payload["capabilities"].get("optimizer_contracts", {})
