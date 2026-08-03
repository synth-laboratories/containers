"""Regression tests for the public container-pool client."""

from __future__ import annotations

import asyncio
import hashlib
import io
import tarfile
from pathlib import Path
from typing import Any, Mapping

import httpx
import pytest

from synth_containers.pools import (
    PoolClient,
    PoolClientError,
    PoolState,
    PoolStateError,
    RolloutOutcome,
    pack_build_context,
    pool_state_admits_rollouts,
    pool_state_is_provisioning,
    pool_state_is_terminal,
    validate_pool_transition,
)


def test_pool_state_contract() -> None:
    assert pool_state_is_provisioning(PoolState.BUILDING)
    assert pool_state_admits_rollouts(PoolState.ACTIVE)
    assert not pool_state_admits_rollouts(PoolState.PAUSED)
    assert pool_state_is_terminal(PoolState.ARCHIVED)

    validate_pool_transition(PoolState.CREATED, PoolState.BUILDING)
    validate_pool_transition(PoolState.FAILED, PoolState.BUILDING)
    validate_pool_transition(PoolState.ACTIVE, PoolState.ARCHIVED)

    with pytest.raises(PoolStateError, match="active -> building"):
        validate_pool_transition(PoolState.ACTIVE, PoolState.BUILDING)
    with pytest.raises(PoolStateError, match="unknown pool state"):
        validate_pool_transition("invented", PoolState.ACTIVE)


def test_pack_build_context_is_deterministic_and_preserves_executable_bit(
    tmp_path: Path,
) -> None:
    script = tmp_path / "entrypoint.sh"
    script.write_text("#!/bin/sh\nexec true\n")
    script.chmod(0o755)
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    excluded = tmp_path / "target"
    excluded.mkdir()
    (excluded / "large.bin").write_bytes(b"ignored")

    first = pack_build_context(tmp_path)
    script.touch()
    second = pack_build_context(tmp_path)

    assert first == second
    assert first[4:8] == b"\x00\x00\x00\x00"
    with tarfile.open(fileobj=io.BytesIO(first), mode="r:gz") as archive:
        names = archive.getnames()
        assert names == sorted(names)
        assert "target" not in names
        assert "target/large.bin" not in names
        assert archive.getmember("entrypoint.sh").mode == 0o755


class _BoundedClient(PoolClient):
    def __init__(self) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(500, request=request))
        super().__init__(
            api_key="test",
            client=httpx.AsyncClient(base_url="https://example.test", transport=transport),
        )
        self.active = 0
        self.peak = 0
        self.payloads: dict[str, Mapping[str, Any]] = {}

    async def submit(self, pool_id: str, payload: Mapping[str, Any]) -> str:
        del pool_id
        self.active += 1
        self.peak = max(self.peak, self.active)
        rollout_id = f"rollout-{payload['seed']}"
        self.payloads[rollout_id] = payload
        if payload["seed"] == 2:
            self.active -= 1
            raise PoolClientError("seed refused")
        return rollout_id

    async def wait_for(
        self,
        rollout_id: str,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float,
    ) -> RolloutOutcome:
        del timeout_seconds, poll_interval_seconds
        await asyncio.sleep(0.01)
        self.active -= 1
        return RolloutOutcome(
            rollout_id=rollout_id,
            status="completed",
            payload={"result": {"seed": self.payloads[rollout_id]["seed"]}},
        )


def test_run_many_bounds_concurrency_and_keeps_failures_in_order() -> None:
    async def exercise() -> tuple[_BoundedClient, list[RolloutOutcome | PoolClientError]]:
        client = _BoundedClient()
        outcomes = await client.run_many(
            "pool-1",
            ({"seed": seed} for seed in range(5)),
            max_in_flight=2,
        )
        await client._client.aclose()
        return client, outcomes

    client, outcomes = asyncio.run(exercise())

    assert client.peak == 2
    assert [outcome.rollout_id for outcome in outcomes if isinstance(outcome, RolloutOutcome)] == [
        "rollout-0",
        "rollout-1",
        "rollout-3",
        "rollout-4",
    ]
    assert isinstance(outcomes[2], PoolClientError)


def test_preflight_reports_optional_missing_probes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, json={"status": "ok"}, request=request)
        if request.url.path.endswith("/info"):
            return httpx.Response(200, json={"runtime": "craftax"}, request=request)
        return httpx.Response(404, json={"detail": "missing"}, request=request)

    async def exercise() -> dict[str, Any]:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            base_url="https://example.test", transport=transport
        ) as transport_client:
            client = PoolClient(api_key="test", client=transport_client)
            return await client.preflight("pool-1")

    result = asyncio.run(exercise())
    assert result["health"] == {"status": "ok"}
    assert result["info"] == {"runtime": "craftax"}
    assert result["missing"] == ["metadata", "task_info"]


def test_pause_names_the_backend_state_machine_gap() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "not found"}, request=request)

    async def exercise() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            base_url="https://example.test", transport=transport
        ) as transport_client:
            client = PoolClient(api_key="test", client=transport_client)
            await client.pause_pool("pool-1")

    with pytest.raises(PoolStateError, match="ContainerPool.status"):
        asyncio.run(exercise())


def test_create_image_release_refuses_backend_hash_mismatch() -> None:
    archive = b"deterministic-build-context"
    unexpected = hashlib.sha256(b"different").hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"id": "release-1", "source_content_hash": unexpected},
            request=request,
        )

    async def exercise() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            base_url="https://example.test", transport=transport
        ) as transport_client:
            client = PoolClient(api_key="test", client=transport_client)
            await client.create_image_release("pool-1", archive=archive)

    with pytest.raises(PoolClientError, match="hash mismatch"):
        asyncio.run(exercise())
