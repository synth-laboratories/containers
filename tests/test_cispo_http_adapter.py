"""CISPO advertisement and route dispatch on the reference app.

The adapter's job here is narrow and worth pinning exactly. It advertises
``optimizer_contracts.cispo`` beside ``gepa`` for a runtime that declares
CISPO and for no other; it serves the capability document itself, because the
document is derived from the runtime's own declaration rather than from a
handler; and it dispatches the other sixteen routes to two ports it does not
implement, answering a typed 501 until streams B and C supply them.

A 501 rather than a 404 is the point: the route *is* declared. Route presence
and behavior support are separate questions, and an executor must be able to
read the whole shape it would talk to before deciding whether to spend.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from synth_containers.cispo_contract import (
    CISPO_OPTIMIZER_CONTRACT_VERSION,
    CISPO_ROUTE_PREFIX,
    CispoNotImplementedError,
    cispo_declared_routes,
)
from synth_containers.http_adapter import create_reference_app
from synth_containers.reference_runtime import ReferenceManagedRuntime

from tests.test_cispo_contract import (
    DECLARED_KEYS,
    minimal_declaration,
    minimal_surface,
)


class _DeclaringRuntime(ReferenceManagedRuntime):
    """The reference runtime plus a CISPO declaration and nothing else.

    It supplies no CISPO ports, which is the state streams B and C start from.
    The three GEPA methods are here so the two advertisements can be seen
    coexisting rather than one replacing the other.
    """

    def cispo_declaration(self) -> Any:
        return minimal_declaration()

    def metadata(self) -> Any:
        value = super().metadata()
        value.capabilities = minimal_surface()
        return value

    def program(self) -> dict[str, Any]:
        return {"program_id": "program-1"}

    def taskset_info(self) -> dict[str, Any]:
        return {"taskset_id": "taskset-small"}

    def taskset_tasks(self, request: dict[str, Any]) -> dict[str, Any]:
        return {"tasks": [], "echo": dict(request)}


def _declaring_client() -> TestClient:
    runtime = _DeclaringRuntime.counter_default(target=1)
    return TestClient(create_reference_app(runtime), raise_server_exceptions=False)


def _plain_client() -> TestClient:
    return TestClient(
        create_reference_app(ReferenceManagedRuntime.counter_default(target=1)),
        raise_server_exceptions=False,
    )


# --------------------------------------------------------------------------- #
# Advertisement
# --------------------------------------------------------------------------- #


def test_metadata_advertises_cispo_beside_gepa() -> None:
    payload = _declaring_client().get("/metadata").json()
    contracts = payload["metadata"]["optimizer_contracts"]
    assert "gepa" in contracts, "the existing advertisement must survive"
    block = contracts["cispo"]
    assert block["version"] == CISPO_OPTIMIZER_CONTRACT_VERSION
    for name in DECLARED_KEYS:
        assert block[name].startswith("/"), name


def test_a_runtime_that_declares_no_cispo_advertises_none() -> None:
    """Discovery, not decoration: nothing to hash means nothing to promise."""

    metadata = _plain_client().get("/metadata").json()["metadata"]
    contracts = metadata.get("optimizer_contracts") or {}
    assert "cispo" not in contracts


# --------------------------------------------------------------------------- #
# The capability document over HTTP
# --------------------------------------------------------------------------- #


def test_capabilities_route_serves_the_hashed_document() -> None:
    routes = cispo_declared_routes()
    response = _declaring_client().get(routes["capabilities_route"])
    assert response.status_code == 200
    document = response.json()
    assert document["schema_version"] == "cispo.capabilities.v1"
    assert document["capability_hash"].startswith("sha256:")
    assert document["container_id"] == "container-small"
    # The route table the document names is the one actually mounted.
    assert document["routes"] == dict(sorted(routes.items()))
    assert document["lifecycle"]["lease"]["heartbeat_route"] == routes["rollout_renew_route"]


def test_the_capability_document_is_the_top_level_body() -> None:
    """The optimizer feeds the response body straight to ``from_payload``; a
    wrapper object would fail its schema check."""

    body = _declaring_client().get(cispo_declared_routes()["capabilities_route"]).json()
    assert "capabilities" not in body
    assert set(body) >= {"schema_version", "capability_hash", "topology", "reward"}


# --------------------------------------------------------------------------- #
# Dispatch to the unimplemented ports
# --------------------------------------------------------------------------- #


def _call(client: TestClient, name: str) -> Any:
    routes = cispo_declared_routes()
    route = routes[name].replace("{rollout_id}", "rollout-1").replace(
        "{topology_id}", "topology-small"
    )
    method = {
        "handshake_route": "POST",
        "taskset_tasks_route": "POST",
        "policy_bind_route": "POST",
        "policy_set_bind_route": "POST",
        "rollout_route": "POST",
        "rollout_renew_route": "POST",
        "rollout_finalize_route": "POST",
        "rollout_terminate_route": "POST",
    }.get(name, "GET")
    if method == "POST":
        return client.post(route, json={})
    if name == "reward_route":
        return client.get(route, params={"rollout_id": "rollout-1"})
    return client.get(route)


def test_every_declared_route_but_capabilities_answers_a_typed_501() -> None:
    client = _declaring_client()
    for name in DECLARED_KEYS:
        if name == "capabilities_route":
            continue
        response = _call(client, name)
        assert response.status_code == 501, f"{name} answered {response.status_code}"
        detail = response.json()["detail"]
        assert detail["error"] == "cispo_port_not_implemented"
        assert detail["port"] in {"CispoAdmissionPort", "CispoRolloutPort"}
        assert detail["operation"].startswith("cispo_")
        assert detail["contract_version"] == CISPO_OPTIMIZER_CONTRACT_VERSION


def test_health_and_rollout_go_to_the_two_different_ports() -> None:
    client = _declaring_client()
    assert _call(client, "health_route").json()["detail"]["port"] == "CispoAdmissionPort"
    assert _call(client, "rollout_route").json()["detail"]["port"] == "CispoRolloutPort"


def test_the_legacy_surface_is_untouched() -> None:
    """Additive means additive: the pre-existing paths keep their semantics."""

    client = _declaring_client()
    assert client.get("/health").json()["status"] == "ok"
    assert client.get(f"{CISPO_ROUTE_PREFIX}/health").status_code == 501
    assert client.get("/rollouts/unknown").status_code == 404


def test_a_supplied_port_is_dispatched_to() -> None:
    """The signatures streams B and C implement, exercised end to end."""

    class _Admission:
        def cispo_health(self) -> dict[str, Any]:
            return {
                "status": "ok",
                "container_version": "0.0.1",
                "container_image_digest": "sha256:image-small",
            }

        def cispo_topology(self, topology_id: str) -> dict[str, Any]:
            return {"topology_id": topology_id}

        def cispo_handshake(self, request: dict[str, Any]) -> dict[str, Any]:
            return {"handshake_id": "hs-1", "echo": dict(request)}

    class _Rollouts:
        async def cispo_submit_rollout(self, request: dict[str, Any]) -> dict[str, Any]:
            return {"rollout_id": "rollout-1", "echo": dict(request)}

        def cispo_rollout_events(
            self, rollout_id: str, *, cursor: str | None = None, limit: int | None = None
        ) -> dict[str, Any]:
            return {"rollout_id": rollout_id, "cursor": cursor, "limit": limit, "events": []}

        def cispo_reward(self, request: dict[str, Any]) -> dict[str, Any]:
            raise CispoNotImplementedError("CispoRolloutPort", "cispo_reward")

    class _Wired(_DeclaringRuntime):
        def cispo_admission(self) -> Any:
            return _Admission()

        def cispo_rollouts(self) -> Any:
            return _Rollouts()

    client = TestClient(
        create_reference_app(_Wired.counter_default(target=1)), raise_server_exceptions=False
    )
    routes = cispo_declared_routes()

    assert client.get(routes["health_route"]).json()["container_version"] == "0.0.1"
    assert (
        client.get(routes["topology_route"].format(topology_id="topology-small")).json()[
            "topology_id"
        ]
        == "topology-small"
    )
    handshake = client.post(routes["handshake_route"], json={"run_id": "run-1"})
    assert handshake.json()["echo"] == {"run_id": "run-1"}

    # An async port method is awaited, and submission answers 202.
    submitted = client.post(routes["rollout_route"], json={"idempotency_key": "k"})
    assert submitted.status_code == 202
    assert submitted.json()["rollout_id"] == "rollout-1"

    # The cursor and limit reach the port as keyword arguments.
    events = client.get(
        routes["rollout_events_route"].format(rollout_id="rollout-1"),
        params={"cursor": "c-7", "limit": 5},
    )
    assert events.json() == {
        "rollout_id": "rollout-1",
        "cursor": "c-7",
        "limit": 5,
        "events": [],
    }

    # An operation the port explicitly does not serve still answers 501.
    refused = client.get(routes["reward_route"], params={"rollout_id": "rollout-1"})
    assert refused.status_code == 501
    assert refused.json()["detail"]["operation"] == "cispo_reward"


def test_a_runtime_missing_a_mandatory_capability_serves_no_document() -> None:
    """The document refuses to exist rather than defaulting a claim."""

    class _NoTrace(_DeclaringRuntime):
        def metadata(self) -> Any:
            value = ReferenceManagedRuntime.metadata(self)
            value.capabilities = minimal_surface(trace_support=False)
            return value

    client = TestClient(
        create_reference_app(_NoTrace.counter_default(target=1)), raise_server_exceptions=False
    )
    response = client.get(cispo_declared_routes()["capabilities_route"])
    assert response.status_code == 501
    detail = response.json()["detail"]
    assert detail["error"] == "cispo_capability_not_declared"
    assert "trace_v5" in detail["reason"]
