"""Committed event replay and safe transport behavior through the public client."""

import asyncio

import httpx
import pytest

from synth_containers.pools import PoolClient, PoolClientError


def test_replay_drains_every_page_and_resumes_from_last_applied_event():
    cursors = []

    def handler(request):
        cursor = int(request.url.params["after_sequence"])
        cursors.append(cursor)
        assert request.url.params["format"] == "json"
        sequence = cursor + 1
        return httpx.Response(
            200,
            json={
                "rollout_id": "r1",
                "events": [{"rollout_id": "r1", "sequence": sequence}],
                "next_sequence": sequence,
                "has_more": sequence < 4,
                "status": "completed",
            },
        )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://fixture"
        ) as transport:
            client = PoolClient(api_key="fixture", client=transport)
            events = [event async for event in client.watch_events("r1", after_sequence=1, limit=1)]
            assert [event["sequence"] for event in events] == [2, 3, 4]

    asyncio.run(run())
    assert cursors == [1, 2, 3]


@pytest.mark.parametrize(
    "change",
    [
        {"events": [{"rollout_id": "r1", "sequence": 2}]},
        {"events": [{"rollout_id": "other", "sequence": 1}]},
        {"next_sequence": 100},
        {"has_more": "true"},
    ],
)
def test_bad_replay_cannot_advance_cursor(change):
    page = {
        "rollout_id": "r1",
        "events": [{"rollout_id": "r1", "sequence": 1}],
        "next_sequence": 1,
        "has_more": False,
        "status": "completed",
    } | change

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=page)),
            base_url="https://fixture",
        ) as transport:
            client = PoolClient(api_key="fixture", client=transport)
            with pytest.raises(PoolClientError):
                await client.events("r1")

    asyncio.run(run())


def test_ambiguous_mutation_is_not_automatically_repeated():
    requests = []

    def handler(request):
        requests.append(request)
        raise httpx.ReadTimeout("response lost", request=request)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://fixture"
        ) as transport:
            client = PoolClient(api_key="fixture", client=transport, max_retries=3)
            with pytest.raises(PoolClientError):
                await client.submit("p1", {"seed": 1})

    asyncio.run(run())
    assert len(requests) == 1
