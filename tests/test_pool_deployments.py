import asyncio
import json

import httpx
import pytest
from synth_containers.pools import PoolClient, PoolClientError

PROJECT = "00000000-0000-0000-0000-000000000001"


def test_mutation_pins_revision_and_never_retries_uncertain_delivery():
    sent = []

    def handle(request):
        sent.append(request)
        raise httpx.ReadTimeout("uncertain", request=request)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle), base_url="https://fixture"
        ) as transport:
            client = PoolClient(api_key="fixture", client=transport, max_retries=3)
            with pytest.raises(PoolClientError):
                await client.mutate_deployment(
                    "pool",
                    "task",
                    project_id=PROJECT,
                    operation="update",
                    idempotency_key="stable",
                    expected_revision="revision",
                    payload={"image_ref": "pinned"},
                )

    asyncio.run(run())
    assert len(sent) == 1
    assert sent[0].url.path == "/v1/pools/pool/deployments/task/operations"
    assert json.loads(sent[0].content)["expected_revision"] == "revision"


def test_delete_requires_revision_before_network_io():
    async def run():
        async with PoolClient(api_key="fixture") as client:
            with pytest.raises(PoolClientError, match="expected_revision"):
                await client.mutate_deployment(
                    "pool", "task", project_id=PROJECT, operation="delete", idempotency_key="stable"
                )

    asyncio.run(run())


@pytest.mark.parametrize("coordinate", ["..", ".", "other/task", "", "x" * 257])
def test_coordinates_cannot_change_route(coordinate):
    with pytest.raises(PoolClientError):
        PoolClient._deployment_coordinate(coordinate)
