"""Wire-level SDK negotiation; fixtures contain no provider credentials."""

import asyncio
import copy

import httpx
import pytest

from synth_containers.rhodes_api_contract import APIContractError, compatible_version
from synth_containers.rhodes_client import RhodesClient


def profile():
    return {
        "schema": "synth.api-discovery.v1",
        "service": "rhodes-api",
        "authority": "synth-backend.core",
        "api": {"name": "synth.rhodes", "version": "2.0.0", "supported_versions": ["2.0.0"]},
        "capabilities": {"customer.pool.create": True, "customer.rollout.create": True},
    }


@pytest.mark.parametrize("version", ["2.0.0", "2.0.1", "2.12.34", "2.0.0+build.123"])
def test_stable_compatible_semver(version):
    assert compatible_version(version)


@pytest.mark.parametrize(
    "version",
    [
        "1.99.99",
        "3.0.0",
        "2.0.0-rc.1",
        "2.00.0",
        "02.0.0",
        "2.0",
        "v2.0.0",
        "2.0.0+",
        "2.0.0+bad_thing",
        "2.0.0\n",
        True,
        None,
        "2." + "0" * 130 + ".0",
    ],
)
def test_unsupported_or_malformed_semver(version):
    assert not compatible_version(version)


def test_discovery_strips_credentials_and_pins_future_requests():
    seen = []
    body = profile()
    body["future_optional_field"] = {"anything": True}

    def respond(request):
        seen.append(request)
        if request.url.path == "/.well-known/synth-api":
            return httpx.Response(200, json=body, headers={"Synth-API-Version": "2.0.0"})
        return httpx.Response(200, json={})

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond),
            base_url="https://rhodes.invalid",
            auth=("fixture", "fixture"),
            headers={"Authorization": "Bearer fixture", "X-API-Key": "fixture"},
            cookies={"private": "fixture"},
        ) as transport:
            client = RhodesClient(api_key="fixture", client=transport)
            assert (
                await client.negotiate_api(required_capabilities=["customer.pool.create"]) == body
            )
            await client._request("POST", "/pools", payload={})

    asyncio.run(exercise())
    assert len(seen) == 2
    assert seen[0].url.path == "/.well-known/synth-api"
    for header in ("authorization", "x-api-key", "cookie"):
        assert header not in seen[0].headers
    assert all(request.headers["Synth-API-Major"] == "2" for request in seen)
    assert "authorization" in seen[1].headers


@pytest.mark.parametrize(
    "case",
    [
        "legacy",
        "redirect",
        "html",
        "oversize",
        "invalid_json",
        "duplicate",
        "list",
        "major1",
        "major3",
        "prerelease",
        "mismatch",
        "no_header",
        "service",
        "authority",
        "missing_capability",
        "false_capability",
        "string_capability",
        "bad_supported",
    ],
)
def test_failures_stop_after_one_read_and_never_pin_or_mutate(case):
    body = copy.deepcopy(profile())
    headers = {"Synth-API-Version": "2.0.0"}
    status = 200
    content = None
    if case == "legacy":
        status = 404
    elif case == "redirect":
        status, headers["location"] = 307, "https://other.invalid"
    elif case == "html":
        headers["content-type"] = "text/html"
    elif case == "oversize":
        content = b" " * 65537
    elif case == "invalid_json":
        content = b"{"
    elif case == "duplicate":
        content = b'{"api":{},"api":{}}'
    elif case == "list":
        body = []
    elif case in {"major1", "major3", "prerelease"}:
        body["api"]["version"] = {"major1": "1.0.0", "major3": "3.0.0", "prerelease": "2.0.0-rc.1"}[
            case
        ]
    elif case == "mismatch":
        headers["Synth-API-Version"] = "2.0.1"
    elif case == "no_header":
        headers = {}
    elif case in {"service", "authority"}:
        body[case] = "other"
    elif case == "missing_capability":
        body["capabilities"] = {}
    elif case == "false_capability":
        body["capabilities"]["customer.pool.create"] = False
    elif case == "string_capability":
        body["capabilities"]["customer.pool.create"] = "true"
    elif case == "bad_supported":
        body["api"]["supported_versions"] = ["1.0.0"]
    seen = []

    def respond(request):
        seen.append(request)
        if content is not None:
            headers["content-type"] = "application/json"
            return httpx.Response(status, content=content, headers=headers)
        return httpx.Response(status, json=body, headers=headers)

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond), base_url="https://rhodes.invalid"
        ) as transport:
            client = RhodesClient(api_key="fixture", client=transport)
            with pytest.raises(APIContractError):
                await client.negotiate_api(required_capabilities=["customer.pool.create"])
            assert "Synth-API-Major" not in transport.headers

    asyncio.run(exercise())
    assert len(seen) == 1 and seen[0].method == "GET"


@pytest.mark.parametrize("version", ["2.0.1", "2.5.0", "2.0.0+sha.123"])
def test_additive_versions_pass_the_full_handshake(version):
    body = profile()
    body["api"].update(version=version, supported_versions=[version])

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json=body,
                    headers={"Synth-API-Version": version},
                )
            ),
            base_url="https://rhodes.invalid",
        ) as transport:
            assert (await RhodesClient(api_key="fixture", client=transport).negotiate_api())["api"][
                "version"
            ] == version

    asyncio.run(exercise())
