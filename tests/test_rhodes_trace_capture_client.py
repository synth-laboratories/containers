import asyncio
import json

import httpx
import pytest

from synth_containers.pools import PoolClientError
from synth_containers.rhodes_client import (
    RhodesClient,
    RhodesTraceCapturePolicy,
    RhodesTraceSaveRequest,
)


def test_submit_opt_in_preserves_original_request_and_replay_identity():
    async def check():
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={"rollout_id": "original-rollout"})

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond), base_url="https://rhodes.invalid"
        ) as transport:
            client = RhodesClient(api_key="fixture", client=transport)
            payload = {"idempotency_key": "original-key", "seed": 7}
            assert (
                await client.submit("pool", payload, trace_capture=RhodesTraceCapturePolicy())
                == "original-rollout"
            )
            assert payload == {"idempotency_key": "original-key", "seed": 7}
            body = json.loads(requests[0].content)
            assert body["idempotency_key"] == "original-key"
            assert body["trace_capture"] == {
                "schema_version": "synth.rhodes-trace-capture.v1",
                "mode": "when_advertised",
            }
            with pytest.raises(ValueError, match="only once"):
                await client.submit(
                    "pool",
                    {**payload, "trace_capture": {}},
                    trace_capture=RhodesTraceCapturePolicy(),
                )
            assert len(requests) == 1

    asyncio.run(check())


def test_capability_discovery_preserves_unsupported_and_malformed_states():
    async def check():
        replies = iter(
            [
                {},
                {
                    "trace_capture": {
                        "schema_version": "synth.rhodes-trace-capture.v1",
                        "modes": ["when_advertised"],
                        "profiles": ["observed_local_synth_http"],
                        "producer_capability": "not_negotiated",
                        "required_capture_supported": False,
                        "project_retention_supported": False,
                    }
                },
                {"trace_capture": {"modes": ["when_advertised"]}},
                {"trace_capture": None},
            ]
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=next(replies))),
            base_url="https://rhodes.invalid",
        ) as transport:
            client = RhodesClient(api_key="fixture", client=transport)
            assert await client.trace_capture_capabilities() is None
            capabilities = await client.trace_capture_capabilities()
            assert capabilities.modes == ("when_advertised",)
            assert capabilities.producer_capability == "not_negotiated"
            assert capabilities.project_retention_supported is False
            with pytest.raises(PoolClientError, match="malformed"):
                await client.trace_capture_capabilities()
            with pytest.raises(PoolClientError, match="malformed"):
                await client.trace_capture_capabilities()

    asyncio.run(check())


@pytest.mark.parametrize("seconds", [59, 86401, True, 1.5, "60", None])
def test_trace_save_request_rejects_invalid_expiry(seconds):
    with pytest.raises(ValueError):
        RhodesTraceSaveRequest(valid_for_seconds=seconds)


def test_trace_save_request_rejects_wrong_destination():
    with pytest.raises(ValueError):
        RhodesTraceSaveRequest(valid_for_seconds=60, destination="other")


def test_configured_trace_save_serializes_exactly_with_capture_and_refuses_conflicts():
    async def check():
        calls = []

        def respond(request):
            calls.append(request)
            return httpx.Response(200, json={"rollout_id": "rollout-1"})

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond), base_url="https://rhodes.invalid"
        ) as transport:
            client = RhodesClient(api_key="fixture", client=transport)
            original = {"idempotency_key": "stable", "seed": 7}
            for seconds in (60, 86400):
                assert (
                    await client.submit(
                        "pool",
                        original,
                        trace_capture=RhodesTraceCapturePolicy(),
                        trace_save=RhodesTraceSaveRequest(valid_for_seconds=seconds),
                    )
                    == "rollout-1"
                )
                assert json.loads(calls[-1].content) == {
                    **original,
                    "trace_capture": {
                        "schema_version": "synth.rhodes-trace-capture.v1",
                        "mode": "when_advertised",
                    },
                    "trace_save": {"destination": "project", "valid_for_seconds": seconds},
                }
            assert original == {"idempotency_key": "stable", "seed": 7}
            raw_capture = RhodesTraceCapturePolicy().to_payload()
            await client.submit(
                "pool",
                {**original, "trace_capture": raw_capture},
                trace_save=RhodesTraceSaveRequest(60),
            )
            assert json.loads(calls[-1].content)["trace_capture"] == raw_capture
            with pytest.raises(ValueError, match="requires"):
                await client.submit("pool", original, trace_save=RhodesTraceSaveRequest(60))
            with pytest.raises(ValueError, match="requires"):
                await client.submit(
                    "pool", {**original, "trace_capture": {}}, trace_save=RhodesTraceSaveRequest(60)
                )
            with pytest.raises(ValueError, match="only once"):
                await client.submit(
                    "pool",
                    {**original, "trace_save": {}},
                    trace_capture=RhodesTraceCapturePolicy(),
                    trace_save=RhodesTraceSaveRequest(60),
                )
            with pytest.raises(TypeError, match="RhodesTraceSaveRequest"):
                await client.submit(
                    "pool", original, trace_capture=RhodesTraceCapturePolicy(), trace_save={}
                )
            assert len(calls) == 3

    asyncio.run(check())


def test_configured_save_status_exposes_observed_captures_without_implying_completeness():
    async def check():
        value = {
            "rollout_id": "rollout-1",
            "project_id": "project-1",
            "delegation_id": "delegation-1",
            "status": "retained",
            "delegation_status": "expired",
            "expires_at": "2026-10-31T00:00:00Z",
            "revoked_at": None,
            "captures": [
                {
                    "association_id": "mna_" + "a" * 64,
                    "hold_id": "hold",
                    "execution_epoch": "epoch-1",
                    "publication_id": "publication",
                }
            ],
            "captures_truncated": False,
            "attempts": [
                {
                    "execution_epoch": "epoch-1",
                    "state": "attached",
                    "created_at": "2026-10-31T00:00:00Z",
                    "released_at": None,
                    "release_reason": None,
                    "release_decision_ref": None,
                }
            ],
            "attempts_truncated": False,
            "capture_coverage": "observed_only",
        }
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: httpx.Response(200, json=value)),
            base_url="https://rhodes.invalid",
        ) as transport:
            client = RhodesClient(api_key="fixture", client=transport)
            status = await client.configured_trace_save_status("rollout-1")
            assert status.status == "retained" and status.delegation_status == "expired"
            assert status.captures[0].association_id == "mna_" + "a" * 64
            assert (
                status.attempts[0].state == "attached"
                and status.capture_coverage == "observed_only"
            )
            with pytest.raises(PoolClientError, match="malformed"):
                await client.configured_trace_save_status("other")
            value.update(status="revoked", delegation_status=None, captures=[])
            assert (await client.configured_trace_save_status("rollout-1")).status == "revoked"
            value["capture_coverage"] = "complete"
            with pytest.raises(PoolClientError, match="malformed"):
                await client.configured_trace_save_status("rollout-1")
            with pytest.raises(ValueError):
                await client.configured_trace_save_status("../other")

    asyncio.run(check())
