"""Actual HTTP compatibility termination cannot invent an origin stop receipt."""
import asyncio

import httpx

from synth_containers.capabilities import RuntimeMetadata, TaskCatalog, TaskInfo
from synth_containers.compat.base import ThinCompatRuntime
from synth_containers.http_adapter import create_reference_app
from synth_containers.nouns import ExecutionRecord, TaskDefinition


def test_unsupported_termination_preserves_origin_work_and_late_completion(tmp_path):
    async def exercise():
        entered, release = asyncio.Event(), asyncio.Event()
        effects = []
        execution = ExecutionRecord(execution_id="owned-origin", trace_correlation_id="owned-trace",
                                    status="running", success_status="running")

        async def origin():
            entered.set()
            await release.wait()
            effects.append("committed")
            execution.status = "completed"
            execution.success_status = "success"

        runtime = ThinCompatRuntime(metadata=RuntimeMetadata(runtime_id="thin", name="thin"),
                                    task_info=TaskInfo(task=TaskDefinition(task_id="task", task_name="task")),
                                    task_catalog=TaskCatalog(catalog_id="tasks"),
                                    rollout_handler=lambda request: execution)
        await runtime.submit_rollout({"rollout_id": execution.execution_id})
        task = asyncio.create_task(origin())
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_reference_app(runtime, storage_root=tmp_path / "events")), base_url="http://owned-origin") as client:
                await entered.wait()
                response = await client.post("/rollouts/owned-origin/terminate", json={"reason": "test"})
                assert response.status_code == 200
                assert response.json()["status"] == "in_doubt"
                assert response.json()["metadata"]["origin_outcome"] == "unverified"
                assert response.json()["metadata"]["resource_cleanup"] == "unverified"
                assert not task.done() and effects == []
                release.set()
                await task
                assert effects == ["committed"]
                for state in ("completed", "failed", "cancelled", "terminated", "stopped"):
                    execution.status = state
                    execution.metadata = {"immutable": True}
                    response = await client.post("/rollouts/owned-origin/terminate", json={})
                    assert response.status_code == 200 and response.json()["status"] == state
                    assert execution.metadata == {"immutable": True}
        finally:
            release.set()
            await task
    asyncio.run(exercise())
