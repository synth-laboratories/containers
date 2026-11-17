"""Lost replies/restarts never repeat native work; controls retain custody."""

import asyncio
from pathlib import Path

import httpx
import pytest

from synth_containers.rollout_journal import RolloutJournalError
from synth_containers.service_contracts import ServiceDeclaration
from synth_containers.service_runtime import (
    DurableServiceRuntime,
    OperationObservation,
    OperationRequest,
    create_service_app,
)


def declaration():
    return ServiceDeclaration.model_validate(
        {
            "interfaces": [
                {"name": "jobs", "protocol": "counter.http", "version": "1", "port": 8000}
            ],
            "execution": {
                "protocol": "counter.http",
                "version": "1",
                "interface": "jobs",
                "concurrency": 2,
            },
        }
    )


class Counter:
    def __init__(self):
        self.operations = {}
        self.submissions = 0
        self.lose_response = False

    async def submit(self, operation_id, payload):
        self.submissions += 1
        value = OperationObservation(operation_id, "running")
        self.operations[operation_id] = value
        if self.lose_response:
            raise TimeoutError()
        return value

    async def recover(self, operation_id):
        return self.operations.get(operation_id)

    async def observe(self, native_id):
        return self.operations[native_id]

    async def cancel(self, native_id):
        self.operations[native_id] = OperationObservation(
            native_id, "cancelled", cleanup="unverified"
        )


def runtime(tmp_path: Path, driver):
    return DurableServiceRuntime(
        runtime_id="counter", declaration=declaration(), directory=tmp_path, driver=driver
    )


@pytest.mark.asyncio
async def test_restart_after_lost_reply_reconciles_original_identity(tmp_path):
    driver = Counter()
    driver.lose_response = True
    first = runtime(tmp_path, driver)
    request = OperationRequest(operation_id="one", input={"value": 1})
    accepted = await first.submit(request)
    await asyncio.gather(*first.tasks.values(), return_exceptions=True)
    assert first.journal.get("one")["status"] == "in_doubt"
    await first.close()
    second = runtime(tmp_path, driver)
    replay = await second.submit(request)
    assert replay["origin_instance"] == accepted["origin_instance"]
    assert (await second.read("one"))["record"]["native_id"] == "one"
    assert driver.submissions == 1
    with pytest.raises(RolloutJournalError, match="identity_conflict"):
        await second.submit(OperationRequest(operation_id="one", input={"value": 2}))
    await second.close()


@pytest.mark.asyncio
async def test_cancel_requires_cleanup_and_bound_http_identity(tmp_path):
    driver = Counter()
    rt = runtime(tmp_path, driver)
    app = create_service_app(rt, api_key="x" * 32)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://service"
    ) as client:
        assert (await client.get("/.well-known/synth-service")).status_code == 401
        client.headers["Authorization"] = "Bearer " + "x" * 32
        assert (await client.get("/.well-known/synth-service")).json()[
            "services"
        ] == declaration().model_dump(mode="json")
        response = await client.post("/operations", json={"operation_id": "one", "input": {}})
        assert response.status_code == 202
        await asyncio.gather(*rt.tasks.values())
        binding = {key: response.json()[key] for key in ("origin_instance", "input_sha256")}
        assert (
            await client.get("/operations/one", params={**binding, "origin_instance": "wrong"})
        ).status_code == 409
        assert (await client.post("/operations/one/cancel", params=binding)).status_code == 200
        assert (await client.get("/operations/one", params=binding)).json()["record"][
            "status"
        ] != "cancelled"
        driver.operations["one"] = OperationObservation("one", "cancelled", cleanup="confirmed")
        assert (await client.get("/operations/one", params=binding)).json()["record"][
            "status"
        ] == "cancelled"
    await rt.close()


def test_single_writer_and_state_mount_contract(tmp_path):
    first = runtime(tmp_path, Counter())
    with pytest.raises(RolloutJournalError, match="owner_conflict"):
        runtime(tmp_path, Counter())
    first.journal.close()
    with pytest.raises(ValueError, match="overlap"):
        ServiceDeclaration.model_validate(
            {
                **declaration().model_dump(),
                "state": [
                    {"name": "a", "mount_path": "/state"},
                    {"name": "b", "mount_path": "/state/nested"},
                ],
            }
        )
