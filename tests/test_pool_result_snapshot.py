import asyncio
import hashlib
import json
from uuid import NAMESPACE_URL, uuid5

import httpx
import pytest

from synth_containers.pools import PoolClient, PoolClientError


def fixture():
    result = {"schema_version": "synth.eval-result.v1", "org_id": "org",
              "pool_id": "pool", "rollout_id": "rollout", "status": "failed", "score": None}
    content = json.dumps(result).encode()
    digest = hashlib.sha256(content).hexdigest()
    receipt = {"status": "committed", "digest_sha256": digest, "size_bytes": len(content),
               "publication_id": str(uuid5(NAMESPACE_URL, f"rhodes-result:org:rollout:{digest}"))}
    return result, content, receipt


def run(*, content=None, receipt_change=None, mutate_owner=False, redirect=False):
    result, original, receipt = fixture()
    receipt.update(receipt_change or {})
    calls = []

    def handle(request):
        calls.append(request)
        if request.url.path == "/v1/rollouts/rollout":
            return httpx.Response(200, json={"pool_id": "different" if mutate_owner else "pool",
                "metadata": {"result_publication": receipt}})
        assert request.headers["Authorization"] == "Bearer fixture-key"
        assert request.url.path == f"/artifacts/v1/publications/{receipt['publication_id']}/assets/result.json"
        if redirect:
            return httpx.Response(302, headers={"Location": "https://other.invalid/secret"})
        return httpx.Response(200, content=original if content is None else content)

    async def read():
        async with httpx.AsyncClient(base_url="https://fixture",transport=httpx.MockTransport(handle)) as transport:
            client = PoolClient(api_key="fixture-key",client=transport)
            return await client.get_result_snapshot("rollout")

    return asyncio.run(read()), calls, result


def test_download_verifies_failed_result_and_keeps_null_reward():
    value, calls, expected = run()
    assert value == expected and value["score"] is None
    assert len(calls) == 2


@pytest.mark.parametrize("options", [
    {"receipt_change": {"status": "pending"}},
    {"receipt_change": {"size_bytes": 1024 * 1024 + 1}},
    {"receipt_change": {"digest_sha256": "x" * 64}},
    {"content": b"{}"},
    {"content": b"x" * 1024},
    {"mutate_owner": True},
    {"redirect": True},
])
def test_invalid_custody_never_becomes_a_result(options):
    with pytest.raises(PoolClientError):
        run(**options)
