"""Real Harbor REST/MCP and native scheduler, with an explicit resource authority fixture."""

import asyncio
import base64
import io
import tarfile
from pathlib import Path

import httpx
import pytest

from synth_containers.harbor_serve import create_managed_harbor_app


class Authority:
    def __init__(self):
        self.prepared = {}
        self.calls = []
        self.finished = set()

    async def call(self, action, payload):
        self.calls.append((action, payload))
        if action == "prepare":
            self.prepared[payload["native_id"]] = payload
        if action in {"finish", "recover"}:
            self.finished.add(payload["native_id"])
        if action == "observe":
            return {"cleanup": "confirmed" if payload["native_id"] in self.finished else "pending"}
        if action == "sandbox":
            operation = payload["action"]
            if operation == "exec":
                return {"return_code": 0, "stdout": "", "stderr": ""}
            if operation == "download":
                stream = io.BytesIO()
                with tarfile.open(fileobj=stream, mode="w") as archive:
                    data = b"1.0\n"
                    entry = tarfile.TarInfo("reward.txt")
                    entry.size = len(data)
                    archive.addfile(entry, io.BytesIO(data))
                return {"archive": base64.b64encode(stream.getvalue()).decode()}
        return {"admitted": True, "cleanup": "confirmed"}


def task(root: Path):
    root.mkdir()
    (root / "instruction.md").write_text("Complete the task.")
    (root / "task.toml").write_text("""[environment]
docker_image = "registry.example/task@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
cpus = 1
memory_mb = 2048
storage_mb = 8192
[agent]
timeout_sec = 30
[verifier]
timeout_sec = 30
""")
    (root / "tests").mkdir()
    (root / "tests/test.sh").write_text("#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n")
    (root / "solution").mkdir()
    (root / "solution/solve.sh").write_text("#!/bin/sh\ntrue\n")
    return root


@pytest.mark.asyncio
async def test_native_submit_same_job_controls_and_synth_projection(tmp_path):
    root = task(tmp_path / "task")
    authority = Authority()
    app = create_managed_harbor_app(
        data_dir=tmp_path / "data",
        api_key="x" * 32,
        authority=authority,
        task_roots=(root,),
        concurrency=2,
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app),
            base_url="http://test",
            headers={"Authorization": "Bearer " + "x" * 32},
        ) as client,
    ):
        config = {
            "job_name": "native",
            "tasks": [{"path": str(root)}],
            "agents": [{"name": "oracle"}],
            "n_concurrent_trials": 1,
            "retry": {"max_retries": 0},
        }
        response = await client.post(
            "/jobs/submit", json={"config": config}, headers={"Idempotency-Key": "native-one"}
        )
        assert response.status_code == 200, response.text
        job_id = response.json()["job_id"]
        replay = await client.post(
            "/jobs/submit", json={"config": config}, headers={"Idempotency-Key": "native-one"}
        )
        assert replay.status_code == 409, replay.text
        assert replay.json()["job_id"] == job_id
        for _ in range(100):
            result = await client.get(f"/jobs/{job_id}")
            assert result.status_code == 200, result.text
            if result.json()["job"]["state"] in {"finished", "cancelled"}:
                break
            await asyncio.sleep(0.02)
        assert result.json()["job"]["state"] == "finished", result.text
        assert result.json()["job"]["n_errors"] == 0, result.text
        assert len(authority.prepared) == 1
        assert any(action == "sandbox" for action, _ in authority.calls)
        metadata = await client.get("/synth/.well-known/synth-service")
        assert metadata.status_code == 200
        assert {item["name"] for item in metadata.json()["services"]["interfaces"]} == {
            "harbor",
            "harbor-mcp",
            "synth",
        }
        # The native Streamable HTTP endpoint is present and authenticated.
        mcp = await client.post(
            "/mcp",
            headers={"Accept": "application/json, text/event-stream"},
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1"},
                },
            },
        )
        assert mcp.status_code == 200, mcp.text


@pytest.mark.asyncio
async def test_native_imports_and_task_paths_fail_before_admission(tmp_path):
    root = task(tmp_path / "task")
    authority = Authority()
    app = create_managed_harbor_app(
        data_dir=tmp_path / "data", api_key="x" * 32, authority=authority, task_roots=(root,)
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app),
            base_url="http://test",
            headers={"Authorization": "Bearer " + "x" * 32},
        ) as client,
    ):
        response = await client.post(
            "/jobs/submit",
            json={
                "config": {
                    "tasks": [{"path": str(root.parent)}],
                    "agents": [{"name": "oracle"}],
                }
            },
        )
        assert response.status_code == 422, response.text
        assert not authority.calls
