"""Core-owned lifecycle, immutable input custody and cleanup without paid I/O."""

import asyncio
import copy
import hashlib
import json

import httpx
import pytest

from synth_containers.rhodes_api_contract import APIContractError
from synth_containers.rhodes_core_harbor_e2e import CoreHarborRun, launch_request, upload_task
from tests.test_rhodes_deepswe_e2e import body, events, inputs


@pytest.mark.parametrize("lost_ack", [False, True])
@pytest.mark.parametrize("valid_result", [True, False])
def test_owned_lifecycle_reconciles_original_launch_and_cleans_up_even_for_bad_results(
    tmp_path, lost_ack, valid_result
):
    args = inputs(tmp_path)
    args.budget, args.wall_seconds = "10", 60
    output = tmp_path / "owned"
    output.mkdir()
    seen, uploads = [], []
    state = {"closed": False, "launches": 0}

    def core_respond(request):
        seen.append(
            (
                request.method,
                request.url.path,
                request.headers.get("Idempotency-Key"),
                request.content,
            )
        )
        path = request.url.path
        if path == "/v1/uploads":
            declaration = json.loads(request.content)
            state["declaration"] = declaration
            return httpx.Response(
                201,
                json={
                    "upload_id": "upload-owned",
                    "put": {
                        "url": "https://storage.invalid/owned",
                        "headers": {"Content-Type": "application/json"},
                    },
                },
            )
        if path.endswith("/finalize"):
            declaration = state["declaration"]
            return httpx.Response(
                200,
                json={
                    "file": {
                        "file_id": "file-owned",
                        "revision_id": "rev-owned",
                        "digest_sha256": declaration["digest_sha256"],
                        "size_bytes": declaration["size_bytes"],
                    }
                },
            )
        if path == "/v1/runs":
            state["launches"] += 1
            if lost_ack and state["launches"] == 1:
                raise httpx.ReadTimeout("lost acknowledgement")
            return httpx.Response(202, json={"run_id": "run-owned"})
        if path.endswith("/cancel"):
            state["closed"] = True
            return httpx.Response(200, json={})
        if path.endswith("/report"):
            return httpx.Response(
                200,
                json={
                    "reward": {"rollouts": [{"rollout_id": "rollout-owned"}]},
                    "settlement": {"status": "settled"},
                },
            )
        if path == "/v1/runs/run-owned":
            return httpx.Response(
                200,
                json={
                    key: state["closed"]
                    for key in ("execution_closed", "accounting_final", "cleanup_complete")
                },
            )
        raise AssertionError(path)

    def storage_respond(request):
        assert "authorization" not in request.headers and "cookie" not in request.headers
        uploads.append(request.content)
        return httpx.Response(200)

    class Rhodes:
        async def negotiate_api(self, *, required_capabilities):
            assert required_capabilities == ["core-owned.harbor.execute"]
            return {"api": {"version": "2.0.0"}}

        async def get_rollout(self, rid):
            assert rid == "rollout-owned"
            result = copy.deepcopy(body())
            result["pool_id"] = "pool-owned"
            if not valid_result:
                result["result"]["metrics"]["details"]["agent_exit_code"] = 1
            return result

        async def event_page(self, rid, *, after_sequence):
            assert after_sequence == 0
            return {"events": events(), "has_more": False}

        async def _request(self, method, path):
            assert (method, path) == ("GET", "/pools/pool-owned/archive")
            return {"status": "archived", "archive": {"state": "archived"}}

    class Observer:
        async def snapshots(self):
            return set()

        async def remaining_owners(self, owners):
            assert len(owners) == 2
            return []

    async def exercise():
        async with (
            httpx.AsyncClient(
                base_url="https://core.invalid",
                transport=httpx.MockTransport(core_respond),
                headers={"Authorization": "Bearer core-fixture"},
            ) as core,
            httpx.AsyncClient(
                transport=httpx.MockTransport(storage_respond),
                headers={"Authorization": "Bearer must-strip"},
                cookies={"private": "fixture"},
            ) as storage,
        ):
            return await CoreHarborRun(core, Rhodes(), storage, Observer(), args, output).execute()

    report = asyncio.run(exercise())
    assert report["passed"] is valid_result
    assert state["closed"] is True
    assert report["checks"]["provider_sandboxes_absent"] is True
    assert len(uploads) == 1
    assert hashlib.sha256(uploads[0]).hexdigest() == state["declaration"]["digest_sha256"]
    declaration = json.loads(uploads[0])
    assert declaration["harbor_agent"] == {
        "name": "codex",
        "model_name": "gpt-6-luna",
        "kwargs": {"reasoning_effort": "low"},
    }
    launches = [r for r in seen if r[1] == "/v1/runs"]
    assert len(launches) == (2 if lost_ack else 1)
    assert len({r[2:] for r in launches}) == 1
    assert not any(path.startswith("/v1/pools") for _, path, _, _ in seen)


def test_incompatible_host_refuses_before_upload_inventory_or_launch(tmp_path):
    args = inputs(tmp_path)
    output = tmp_path / "refused"
    output.mkdir()

    class Rhodes:
        async def negotiate_api(self, **kwargs):
            raise APIContractError("unsupported", code="api_capability_unsupported")

    report = asyncio.run(CoreHarborRun(None, Rhodes(), None, None, args, output).execute())
    assert report["passed"] is False
    assert report["errors"] == [{"stage": "preflight", "type": "APIContractError"}]
    assert report["checks"]["parent_not_created"] is True


def test_launch_retains_single_rollout_cap_and_exact_uploaded_revision():
    file = {"file_id": "file", "revision_id": "revision", "digest_sha256": "a" * 64}
    request = launch_request(file, task_id="task", seed=9, budget="12.123456", wall_seconds=600)
    assert request["inputs"]["files"] == [file]
    assert request["resources"][0]["limits"] == {"rollouts": 1, "spend_micros": 12123456}
    assert request["resources"][0]["selection"] == {}


def test_upload_rejects_a_redirect_target_without_transferring_bytes():
    def respond(request):
        return httpx.Response(
            201, json={"upload_id": "u", "put": {"url": "http://untrusted.invalid", "headers": {}}}
        )

    async def exercise():
        async with httpx.AsyncClient(
            base_url="https://core.invalid", transport=httpx.MockTransport(respond)
        ) as core:
            with pytest.raises(ValueError):
                await upload_task(core, None, b"task", key="original", name="task.json")

    asyncio.run(exercise())
