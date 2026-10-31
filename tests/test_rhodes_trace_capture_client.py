import asyncio
import json

import httpx
import pytest

from synth_containers.pools import PoolClientError
from synth_containers.rhodes_client import RhodesClient, RhodesTraceCapturePolicy


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
