import asyncio
import json
from functools import wraps

import httpx
import pytest

from synth_containers.rhodes_build_operations import (
    ArtifactProviderMismatch,
    BuildOperationRefused,
    BuildOperationRequest,
    ExecutorNotPermitted,
    ExecutorUnavailable,
    ManifestMismatch,
    RegistryImage,
    SourceArtifact,
    SourceForbidden,
    SourceInvalid,
    SourceMissing,
)
from synth_containers.rhodes_client import RhodesClient, RhodesRequestError

D = "sha256:" + "a" * 64


def run_async(function):
    @wraps(function)
    def run(*a, **k):
        return asyncio.run(function(*a, **k))

    return run


def operation(**over):
    base = {
        "schema_version": "synth.rhodes-environment-build.v1",
        "build_operation_id": "11111111-1111-1111-1111-111111111111",
        "operation_id": "op-1",
        "kind": "build",
        "environment": "craftax",
        "source": {"kind": "source_artifact", "artifact_id": "art-1", "digest": D},
        "manifest_digest": D,
        "state": "pending",
        "attempt": 1,
        "failure_history": [],
        "created_at": "2026-10-28T00:00:00+00:00",
        "updated_at": "2026-10-28T00:00:00+00:00",
    }
    return {**base, **over}


async def client_for(handler):
    transport = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://rhodes.invalid"
    )
    return RhodesClient(api_key="fixture", client=transport), transport


REQUEST = BuildOperationRequest("op-1", "craftax", SourceArtifact("art-1", D), D)


@run_async
async def test_create_build_sends_contract_and_reports_reuse():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(201 if len(seen) == 1 else 200, json={
            "operation": operation(), "reused": len(seen) > 1})

    client, http = await client_for(handler)
    async with http:
        first = await client.create_build_operation(REQUEST)
        again = await client.create_build_operation(REQUEST)
    assert seen[0].method == "POST" and seen[0].url.path == "/v1/environment_build_operations"
    assert json.loads(seen[0].content) == {
        "schema_version": "synth.rhodes-environment-build.v1",
        "operation_id": "op-1",
        "environment": "craftax",
        "source": {"kind": "source_artifact", "artifact_id": "art-1", "digest": D},
        "manifest_digest": D,
    }
    assert REQUEST.operation_kind == "build"
    assert (first.reused, again.reused) == (False, True)
    assert first.state == "pending" and not first.terminal


@run_async
async def test_import_request_and_succeeded_operation():
    body = operation(
        kind="import", state="succeeded", resolved_digest=D, artifact_kind="oci_image",
        release_id="rel-1", failure_history=[
            {"attempt": 1, "code": "rhodes.build.timeout", "retryable": True,
             "observed_at": "t"}],
    )
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(201, json={"operation": body, "reused": False})

    req = BuildOperationRequest("op-2", "craftax", RegistryImage("ghcr.io/x/y@" + D, D), D)
    client, http = await client_for(handler)
    async with http:
        op = await client.create_build_operation(req)
    assert req.operation_kind == "import"
    assert json.loads(seen[0].content)["source"]["expected_digest"] == D
    assert op.terminal and not op.retryable
    assert (op.artifact_kind, op.release_id, op.resolved_digest) == ("oci_image", "rel-1", D)
    assert op.failure_history[0].code == "rhodes.build.timeout"


@run_async
async def test_get_list_retry_paths():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, dict(request.url.params)))
        if request.url.path.endswith("/build_operations"):
            return httpx.Response(200, json={"items": [operation(), operation(state="failed")]})
        if request.url.path.endswith("/retry"):
            return httpx.Response(200, json={"operation": operation(attempt=2)})
        return httpx.Response(200, json=operation(state="running"))

    client, http = await client_for(handler)
    async with http:
        got = await client.get_build_operation("b0")
        items = await client.list_build_operations("craftax", limit=5)
        retried = await client.retry_build_operation("b1")
    assert seen == [
        ("GET", "/v1/environment_build_operations/b0", {}),
        ("GET", "/v1/environment_definitions/craftax/build_operations", {"limit": "5"}),
        ("POST", "/v1/environment_build_operations/b1/retry", {}),
    ]
    assert got.state == "running"
    assert [i.retryable for i in items] == [False, True]
    assert retried.attempt == 2


@run_async
@pytest.mark.parametrize(
    ("code", "cls", "status"),
    [
        ("executor_unavailable", ExecutorUnavailable, 409),
        ("executor_not_permitted", ExecutorNotPermitted, 409),
        ("artifact_provider_mismatch", ArtifactProviderMismatch, 409),
        ("source_missing", SourceMissing, 404),
        ("manifest_mismatch", ManifestMismatch, 409),
        ("source_forbidden", SourceForbidden, 403),
        ("source_invalid", SourceInvalid, 422),
        ("operation_conflict", BuildOperationRefused, 409),
    ],
)
async def test_typed_refusals_carry_backend_code(code, cls, status):
    full = "rhodes.build." + code

    def handler(request):
        return httpx.Response(status, json={"detail": {"code": full}})

    client, http = await client_for(handler)
    async with http:
        with pytest.raises(cls) as caught:
            await client.create_build_operation(REQUEST)
    assert type(caught.value) is cls
    assert caught.value.code == full and caught.value.status_code == status
    assert isinstance(caught.value, RhodesRequestError)


@run_async
async def test_unrelated_refusal_stays_plain_and_get_refusal_typed():
    def handler(request):
        if request.url.path.endswith("b2"):
            return httpx.Response(404, json={"detail": {"code": "rhodes.build.not_found"}})
        return httpx.Response(403, json={"detail": {"code": "auth.denied"}})

    client, http = await client_for(handler)
    async with http:
        with pytest.raises(RhodesRequestError) as plain:
            await client.get_build_operation("b1")
        with pytest.raises(BuildOperationRefused) as typed:
            await client.get_build_operation("b2")
    assert not isinstance(plain.value, BuildOperationRefused)
    assert typed.value.reason == "not_found"


def test_request_validation_refuses_before_network():
    with pytest.raises(ValueError):
        BuildOperationRequest("op", "Bad Env", SourceArtifact("a", D), D).to_json()
    with pytest.raises(ValueError):
        BuildOperationRequest("op", "env", SourceArtifact("a", "sha256:zz"), D).to_json()
