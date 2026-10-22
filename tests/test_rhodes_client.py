import json
import asyncio
from functools import wraps

import httpx
import pytest

from synth_containers.rhodes_client import RhodesClient, RhodesRequestError, _operator_frames
from synth_containers.pools import PoolClientError, PoolRolloutTimeout


def run_async(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return run


@run_async
@pytest.mark.parametrize("phase", ["request", "retry", "poll"])
async def test_poll_deadline_bounds_entire_observation_without_cancel(phase):
    requests = []
    request_cancelled = asyncio.Event()

    async def respond(request):
        requests.append(request)
        if phase == "request":
            try:
                await asyncio.sleep(10)
            finally:
                request_cancelled.set()
        if phase == "retry":
            return httpx.Response(429, headers={"Retry-After": "10"}, json={})
        return httpx.Response(200, json={"status": "running"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="https://rhodes.invalid"
    ) as transport:
        client = RhodesClient(api_key="fixture", client=transport)
        # The outer guard fails the test if a request, retry or poll interval
        # outlives the client's much smaller observation deadline.
        async with asyncio.timeout(2):
            with pytest.raises(PoolRolloutTimeout, match="not cancelled"):
                await client.wait_for("original-run", timeout_seconds=0.02, poll_interval_seconds=10)
    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert requests[0].url.path == "/v1/rollouts/original-run"
    assert request_cancelled.is_set() == (phase == "request")


@run_async
@pytest.mark.parametrize("parameter", ["timeout_seconds", "poll_interval_seconds"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), 86401])
async def test_poll_invalid_timing_refuses_before_network(parameter, value):
    def unexpected(request):
        pytest.fail("invalid observation timing must not perform a request")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(unexpected), base_url="https://rhodes.invalid"
    ) as transport:
        client = RhodesClient(api_key="fixture", client=transport)
        with pytest.raises(ValueError, match=parameter):
            await client.wait_for("run", **{parameter: value})


@run_async
async def test_poll_preserves_transport_timeout_and_external_cancellation():
    entered = asyncio.Event()

    async def respond(request):
        if not entered.is_set():
            entered.set()
            raise TimeoutError("transport-owned timeout")
        await asyncio.sleep(10)
        pytest.fail("cancelled request completed")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="https://rhodes.invalid"
    ) as transport:
        client = RhodesClient(api_key="fixture", client=transport)
        with pytest.raises(TimeoutError, match="transport-owned timeout"):
            await client.wait_for("run", timeout_seconds=1)
        task = asyncio.create_task(client.wait_for("run", timeout_seconds=1))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@run_async
async def test_poll_terminal_result_retains_original_payload():
    payload = {"status": "completed", "rollout_id": "run", "reward": 0, "cleanup_pending": True}
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
        base_url="https://rhodes.invalid",
    ) as transport:
        result = await RhodesClient(api_key="fixture", client=transport).wait_for("run")
    assert result.rollout_id == "run"
    assert result.status == "completed"
    assert result.payload == payload


@run_async
async def test_observed_capacity_is_explicit_and_keeps_exact_task_and_operation():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={"ready": True}
            if request.method == "GET"
            else {"lease": {"lease_id": "original", "container_id": "observed"}, "reused": False},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="https://rhodes.invalid"
    ) as transport:
        client = RhodesClient(api_key="fixture", client=transport)
        await client.task_readiness("pool", "task/one")
        lease = await client.acquire_observed_lease(
            "pool", "task/one", image_kind="craftax_eval", idempotency_key="exact-operation"
        )
    assert lease == {"lease_id": "original", "container_id": "observed", "reused": False}
    assert requests[0].url.raw_path == b"/v1/pools/pool/tasks/task%2Fone/readiness"
    assert requests[1].url.raw_path == b"/v1/pools/pool/tasks/task%2Fone/capacity_leases"
    payload = json.loads(requests[1].content)
    assert payload == {
        "schema_version": "synth.rhodes-observed-capacity.v1",
        "image_kind": "craftax_eval",
        "idempotency_key": "exact-operation",
        "project_id": None,
        "ttl_seconds": 900,
        "slots": 1,
        "reset": "fresh_instance",
    }


@run_async
async def test_registry_and_lease_verbs_preserve_original_coordinates():
    seen = []

    def respond(request):
        seen.append(
            (
                request.method,
                request.url.path,
                json.loads(request.content) if request.content else None,
            )
        )
        if "/container_leases" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "lease": {
                        "lease_id": "lease-original",
                        "status": "released" if request.url.path.endswith("/release") else "active",
                    },
                    "reused": True,
                },
            )
        return httpx.Response(200, json={"release_id": "original", "lease_id": "lease-original"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="https://rhodes.invalid"
    ) as transport:
        client = RhodesClient(api_key="fixture", client=transport)
        assert (await client.import_environment({"descriptor": {"digest": "frozen"}}))[
            "release_id"
        ] == "original"
        await client.acquire_lease(
            project_id="project",
            image_kind="craftax_eval",
            substrate="docker",
            idempotency_key="same-key",
        )
        await client.renew_lease("lease-original")
        assert (await client.release_lease("lease-original"))["status"] == "released"
    assert seen[0] == ("POST", "/v1/environment_releases", {"descriptor": {"digest": "frozen"}})
    assert seen[1][2]["idempotency_key"] == "same-key"
    assert seen[-1][1] == "/v1/container_leases/lease-original/release"


@run_async
async def test_definition_cas_transport_retry_keeps_original_operation_and_revision():
    seen = []

    def respond(request):
        seen.append(request)
        if len(seen) == 1:
            raise httpx.ReadError("response lost", request=request)
        return httpx.Response(200, json={"receipt": {"revision": 1}, "reused": True})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="https://rhodes.invalid"
    ) as transport:
        client = RhodesClient(api_key="fixture", client=transport, retry_backoff_seconds=0)
        result = await client.bind_definition(
            "craftax", "release-1", expected_revision=0, operation_id="same-operation"
        )
        await client.archive_definition("craftax", expected_revision=1, operation_id="archive")
        await client.definition_operation("same-operation")
        await client.definition_history("craftax", after_revision=1, limit=10)
        await client.list_definitions(after_name="banking77")
    assert result["reused"] is True
    assert seen[0].content == seen[1].content
    assert json.loads(seen[1].content)["expected_revision"] == 0
    assert json.loads(seen[1].content)["operation_id"] == "same-operation"
    assert seen[2].url.path.endswith("/craftax/archive")
    assert seen[3].url.path.endswith("/same-operation")
    assert seen[4].url.params["after_revision"] == "1"
    assert seen[5].url.params["after_name"] == "banking77"


@run_async
async def test_follow_reconnects_at_delivered_event_not_page_high_water():
    cursors = []
    pages = 0

    def respond(request):
        nonlocal pages
        if request.url.params.get("format") == "sse":
            cursors.append(request.headers["Last-Event-ID"])
            sequence = len(cursors)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=f'id: {sequence}\ndata: {{"sequence":{sequence},"rollout_id":"run"}}\n\n',
            )
        pages += 1
        return httpx.Response(
            200, json={"status": "completed", "high_water": 2, "cleanup_pending": pages == 1}
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="https://rhodes.invalid"
    ) as transport:
        client = RhodesClient(api_key="fixture", client=transport)
        events = [event async for event in client.follow_events("run", timeout_seconds=2)]
    assert cursors == ["0", "1"]
    assert [event["sequence"] for event in events] == [1, 2]


@run_async
async def test_timeout_detaches_and_never_sends_cancel():
    calls = []

    def respond(request):
        calls.append(request.method)
        if request.url.params.get("format") == "sse":
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=b": heartbeat\n\n"
            )
        return httpx.Response(200, json={"status": "running", "high_water": 0})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="https://rhodes.invalid"
    ) as transport:
        client = RhodesClient(api_key="fixture", client=transport)
        with pytest.raises(PoolRolloutTimeout):
            async for _ in client.follow_events("run", timeout_seconds=0.02):
                pass
    assert calls and set(calls) == {"GET"}


@run_async
@pytest.mark.parametrize(
    "content",
    [
        b'id: 1\ndata: {"sequence":1}',
        b"x" * (1024 * 1024 + 1),
        b"id: invalid\ndata: {}\n\n",
        b'id: 1\ndata: {"sequence":1,"sequence":1}\n\n',
        b'id: 1\ndata: {"sequence":1,"value":NaN}\n\n',
        b'id: 1\ndata: {"sequence":1,"value":1e999}\n\n',
    ],
)
async def test_corrupt_or_incomplete_sse_refuses(content):
    async def chunks():
        for i in range(0, len(content), 7):
            yield content[i : i + 7]

    with pytest.raises(PoolClientError):
        _ = [value async for value in _operator_frames(chunks())]


@run_async
async def test_chunk_boundaries_and_heartbeats_do_not_invent_events():
    async def chunks():
        for value in [
            b": hi\n\nid: 1\ndat",
            b'a: {"sequence":1}\n',
            b'\nid: 2\ndata: {"sequence":2}\n\n',
        ]:
            yield value

    assert [identifier async for identifier, event in _operator_frames(chunks())] == [1, 2]


@run_async
async def test_retryable_refusal_retains_code_and_retry_after(monkeypatch):
    attempts = []
    delays = []

    async def sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr("synth_containers.pools.asyncio.sleep", sleep)

    def respond(request):
        attempts.append(request)
        return httpx.Response(
            429,
            headers={"Retry-After": "17"},
            json={"detail": {"code": "rhodes.capacity_exhausted"}},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="https://rhodes.invalid"
    ) as transport:
        client = RhodesClient(
            api_key="fixture", client=transport, max_retries=1, retry_backoff_seconds=0
        )
        with pytest.raises(RhodesRequestError) as caught:
            await client.get_environment("release")
    assert len(attempts) == 2
    assert delays == [17.0]
    assert (caught.value.status_code, caught.value.code, caught.value.retry_after) == (
        429,
        "rhodes.capacity_exhausted",
        "17",
    )


@run_async
async def test_unidentified_mutation_transport_failure_is_not_retried():
    attempts = []

    def respond(request):
        attempts.append(request)
        raise httpx.ReadError("response lost", request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="https://rhodes.invalid"
    ) as transport:
        client = RhodesClient(
            api_key="fixture", client=transport, max_retries=3, retry_backoff_seconds=0
        )
        with pytest.raises(PoolClientError):
            await client.renew_lease("lease")
    assert len(attempts) == 1


@run_async
async def test_batch_validates_all_keys_before_io_and_uses_service_wire():
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"items": [{"id": "run-a"}, {"id": "run-b"}]})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="https://rhodes.invalid"
    ) as transport:
        client = RhodesClient(api_key="fixture", client=transport)
        with pytest.raises(ValueError):
            await client.submit_batch("pool", [{"idempotency_key": "a"}, {}])
        assert not requests
        assert await client.submit_batch(
            "pool", [{"idempotency_key": "a"}, {"idempotency_key": "b"}]
        ) == ["run-a", "run-b"]
    assert requests == [
        {
            "requests": [
                {"idempotency_key": "a", "pool_id": "pool"},
                {"idempotency_key": "b", "pool_id": "pool"},
            ]
        }
    ]


@run_async
@pytest.mark.parametrize("storage_status", [200, 412])
async def test_source_upload_uses_separate_create_only_transport_and_finalizes(
    tmp_path, monkeypatch, storage_status
):
    import hashlib
    import synth_containers.rhodes_client as module

    path = tmp_path / "source.tar"
    path.write_bytes(b"frozen source")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    calls = []

    def backend(request):
        declaration = json.loads(request.content)
        calls.append((request.url.path, declaration))
        if request.url.path.endswith("/prepare"):
            return httpx.Response(
                200,
                json={
                    "publication_id": "publication",
                    "upload_targets": [
                        {
                            "logical_path": "source.tar",
                            "digest_sha256": digest,
                            "upload_url": "https://storage.invalid/upload",
                            "required_headers": {"If-None-Match": "*"},
                        }
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "publication_id": "publication",
                "status": "committed",
                "manifest_digest": "a" * 64,
            },
        )

    def storage(request):
        assert request.url.host == "storage.invalid"
        assert "authorization" not in request.headers
        assert request.headers["If-None-Match"] == "*"
        assert request.content == b"frozen source"
        return httpx.Response(storage_status)

    original = httpx.AsyncClient
    async with original(
        transport=httpx.MockTransport(backend),
        base_url="https://rhodes.invalid",
        headers={"Authorization": "Bearer fixture"},
    ) as transport:
        monkeypatch.setattr(
            module.httpx,
            "AsyncClient",
            lambda **kwargs: original(transport=httpx.MockTransport(storage), **kwargs),
        )
        client = RhodesClient(api_key="fixture", client=transport)
        source = await client.upload_environment_source(path, operation_id="same-operation")
    assert source["digest"] == "sha256:" + digest
    assert source["manifest_digest"] == "sha256:" + "a" * 64
    assert calls[0][1] == calls[1][1]
    assert calls[1][0] == "/v1/environment_sources/publication/finalize"


@run_async
async def test_optional_capacity_lease_is_forwarded_with_original_submission_identity():
    seen = []

    def respond(request):
        seen.append(request)
        if len(seen) == 1:
            raise httpx.ReadError("accepted response lost", request=request)
        return httpx.Response(200, json={"rollout_id": "original-rollout"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="https://rhodes.invalid"
    ) as transport:
        client = RhodesClient(api_key="fixture", client=transport, retry_backoff_seconds=0)
        result = await client.submit(
            "pool-one",
            {
                "idempotency_key": "evaluation-17",
                "capacity_lease_id": "observed-lease",
                "task_id": "craftax",
                "seed": 17,
            },
        )
    assert result == "original-rollout"
    assert seen[0].content == seen[1].content
    assert json.loads(seen[1].content)["capacity_lease_id"] == "observed-lease"
    assert json.loads(seen[1].content)["idempotency_key"] == "evaluation-17"


@run_async
@pytest.mark.parametrize(
    "response", [{}, {"lease": {}, "reused": False}, {"lease": {"lease_id": "r"}, "reused": "yes"}]
)
async def test_observed_capacity_refuses_malformed_authority_response(response):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response)),
        base_url="https://rhodes.invalid",
    ) as transport:
        client = RhodesClient(api_key="fixture", client=transport)
        with pytest.raises(PoolClientError, match="malformed lease"):
            await client.acquire_observed_lease(
                "pool", "task", image_kind="craftax_eval", idempotency_key="same"
            )


@pytest.mark.parametrize("status", [429, 503])
@run_async
async def test_long_retry_after_surfaces_without_early_retry_or_sleep(monkeypatch, status):
    attempts = []

    def respond(request):
        attempts.append(request)
        return httpx.Response(status, headers={"Retry-After": "120"},
                              json={"detail": {"code": "rhodes.capacity_exhausted"}})

    async def sleep(_):
        raise AssertionError("Long backpressure belongs to the caller queue")

    monkeypatch.setattr("synth_containers.pools.asyncio.sleep", sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond),
                                base_url="https://rhodes.invalid") as transport:
        client = RhodesClient(api_key="fixture", client=transport, max_retries=3)
        with pytest.raises(RhodesRequestError) as caught:
            await client.get_environment("release")
    assert len(attempts) == 1
    assert caught.value.retry_after == "120" and caught.value.status_code == status


def test_retry_after_date_and_malformed_values():
    from datetime import datetime, timezone
    from synth_containers.rhodes_client import _retry_after_seconds

    now = datetime(2026, 10, 22, 0, 0, 0, tzinfo=timezone.utc)
    assert _retry_after_seconds("Thu, 22 Oct 2026 00:00:17 GMT", now=now) == 17
    assert _retry_after_seconds("Thu, 22 Oct 2026 00:00:00 GMT", now=now) == 0
    assert _retry_after_seconds("Wed, 21 Oct 2026 23:59:59 GMT", now=now) == 0
    assert _retry_after_seconds("0", now=now) == 0
    for value in ("-1", "nan", "inf", "1.5", "invalid", "9" * 129,
                  "Thu, 22 Oct 2026 00:00:17"):
        assert _retry_after_seconds(value, now=now) is None


@pytest.mark.parametrize("status", [429, 503])
@run_async
async def test_backpressure_does_not_make_unidentified_mutation_replay_safe(monkeypatch, status):
    attempts = []

    def respond(request):
        attempts.append(request)
        return httpx.Response(status, headers={"Retry-After": "2"}, json={})

    async def sleep(_):
        raise AssertionError("Unidentified mutation must not retry")

    monkeypatch.setattr("synth_containers.pools.asyncio.sleep", sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond),
                                base_url="https://rhodes.invalid") as transport:
        client = RhodesClient(api_key="fixture", client=transport, max_retries=3)
        with pytest.raises(RhodesRequestError):
            await client.create_task("pool", {"task_id": "task"})
    assert len(attempts) == 1


@run_async
async def test_retry_delay_honors_date_and_local_backoff(monkeypatch):
    from datetime import datetime, timezone
    from synth_containers import rhodes_client

    class Clock:
        @staticmethod
        def now(zone):
            assert zone == timezone.utc
            return datetime(2026, 10, 22, tzinfo=zone)

    monkeypatch.setattr(rhodes_client, "datetime", Clock)
    client = RhodesClient(api_key="fixture", backend_url="https://rhodes.invalid", retry_backoff_seconds=2)
    date_response = httpx.Response(503, headers={"Retry-After": "Thu, 22 Oct 2026 00:00:17 GMT"})
    assert client._retry_delay_seconds(date_response, 0) == 17
    assert client._retry_delay_seconds(httpx.Response(429, headers={"Retry-After": "0"}), 0) == 2
    assert client._retry_delay_seconds(httpx.Response(429, headers={"Retry-After": "nan"}), 1) == 4
    assert client._retry_delay_seconds(httpx.Response(429, headers={"Retry-After": "9" * 100}), 0) is None
    await client.aclose()
