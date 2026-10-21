"""Rhodes registry, lease and committed event lifecycle over PoolClient.

See docs/specs/rhodes-client-v1.md. No provider fallback or client-side compute
is performed. Disconnect and timeout detach; cancel is an explicit operation.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any
from urllib.parse import quote

from .pools import PoolClient, PoolClientError, PoolRolloutTimeout, TERMINAL_STATUSES, _rollout_id
import httpx


class RhodesRequestError(PoolClientError):
    """Preserve the service refusal and backpressure without retrying effects."""

    def __init__(self, status_code: int, code: str, retry_after: str | None):
        self.status_code = status_code
        self.code = code
        self.retry_after = retry_after
        super().__init__(f"Rhodes request refused: HTTP {status_code} ({code})")


def _http_refusal(error: httpx.HTTPStatusError) -> RhodesRequestError:
    code = "rhodes.http_error"
    try:
        body = error.response.json()
        detail = body.get("detail", body) if isinstance(body, dict) else {}
        candidate = detail.get("code") if isinstance(detail, dict) else None
        if isinstance(candidate, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", candidate):
            code = candidate
    except ValueError:
        pass  # Non-JSON error bodies still retain the HTTP refusal and header.
    return RhodesRequestError(
        error.response.status_code, code, error.response.headers.get("Retry-After")
    )


class RhodesClient(PoolClient):
    async def _request(
        self, method: str, path: str, *, payload=None, params=None, optional=False, replay_safe=None
    ) -> dict[str, Any]:
        # No mutation retry without an existing service replay identity.
        safe = (
            method.upper() in {"GET", "DELETE"}
            or (method.upper() == "POST" and path == "/environment_releases")
            or bool(payload and payload.get("idempotency_key"))
        )
        try:
            return await super()._request(
                method,
                path,
                payload=payload,
                params=params,
                optional=optional,
                replay_safe=safe if replay_safe is None else replay_safe,
            )
        except PoolClientError as error:
            cause = error.__cause__
            for _ in range(8):
                if isinstance(cause, httpx.HTTPStatusError):
                    raise _http_refusal(cause) from error
                cause = cause.__cause__ if cause is not None else None
            raise

    async def submit(self, pool_id: str, payload: Mapping[str, Any]) -> str:
        if not isinstance(payload.get("idempotency_key"), str) or not payload["idempotency_key"]:
            raise ValueError("Rhodes submission requires a stable idempotency_key")
        return await super().submit(pool_id, payload)

    async def submit_batch(self, pool_id: str, payloads: Sequence[Mapping[str, Any]]) -> list[str]:
        if len(payloads) > 1000:
            raise ValueError("Rhodes batches support at most 1000 rollouts")
        if any(
            not isinstance(item.get("idempotency_key"), str) or not item["idempotency_key"]
            for item in payloads
        ):
            raise ValueError("Every Rhodes batch item requires a stable idempotency_key")
        # The batch transport itself has no replay receipt. Individual keys
        # allow callers to reconcile partial admission before resubmitting.
        if not payloads:
            return []
        body = await self._request(
            "POST",
            "/rollouts/batch",
            payload={
                "requests": [{**dict(item), "pool_id": pool_id} for item in payloads],
            },
            replay_safe=False,
        )
        items = body.get("items")
        if not isinstance(items, list) or len(items) != len(payloads):
            raise PoolClientError(
                "Rhodes batch response is incomplete; reconcile using original idempotency keys"
            )
        ids = [_rollout_id(item) if isinstance(item, Mapping) else "" for item in items]
        if not all(ids):
            raise PoolClientError("Rhodes batch response lacks original rollout identities")
        return ids

    async def import_environment(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        """Register a content-addressed version from retained Artifact Platform input."""
        return await self._request("POST", "/environment_releases", payload=manifest)

    async def list_environments(self, *, after: str = "", limit: int = 50) -> dict[str, Any]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        return await self._request(
            "GET", "/environment_releases", params={"after": after, "limit": limit}
        )

    async def get_environment(self, release_id: str) -> dict[str, Any]:
        return await self._request("GET", "/environment_releases/" + quote(release_id, safe=""))

    async def archive_environment(self, release_id: str) -> dict[str, Any]:
        return await self._request("DELETE", "/environment_releases/" + quote(release_id, safe=""))

    async def environment_compatibility(self, release_id: str, *, provider: str) -> dict[str, Any]:
        return await self._request(
            "GET",
            "/environment_releases/" + quote(release_id, safe="") + "/compatibility",
            params={"provider": provider},
        )

    async def acquire_lease(
        self,
        *,
        project_id: str | None = None,
        image_kind: str,
        substrate: str,
        idempotency_key: str,
        ttl_seconds: int = 900,
    ) -> dict[str, Any]:
        """Acquire the server lease; inspect readiness independently before work."""
        if not idempotency_key:
            raise ValueError("A stable lease idempotency_key is required")
        return await self._request(
            "POST",
            "/container_leases",
            payload={
                "project_id": project_id,
                "image_kind": image_kind,
                "substrate": substrate,
                "idempotency_key": idempotency_key,
                "ttl_seconds": ttl_seconds,
            },
        )

    async def get_lease(self, lease_id: str) -> dict[str, Any]:
        return await self._request("GET", "/container_leases/" + quote(lease_id, safe=""))

    async def renew_lease(self, lease_id: str, *, ttl_seconds: int = 900) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/container_leases/" + quote(lease_id, safe="") + "/renew",
            payload={"ttl_seconds": ttl_seconds},
        )

    async def release_lease(self, lease_id: str) -> dict[str, Any]:
        return await self._request(
            "POST", "/container_leases/" + quote(lease_id, safe="") + "/release"
        )

    async def event_page(
        self, rollout_id: str, *, after_sequence: int = 0, limit: int = 200
    ) -> dict[str, Any]:
        if type(after_sequence) is not int or after_sequence < 0:
            raise ValueError("after_sequence must be nonnegative")
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        return await self._request(
            "GET",
            "/rollouts/" + quote(rollout_id, safe="") + "/events",
            params={"format": "json", "after_sequence": after_sequence, "limit": limit},
        )

    async def follow_events(
        self, rollout_id: str, *, after_sequence: int = 0, timeout_seconds: float = 1800
    ) -> AsyncIterator[dict[str, Any]]:
        """Follow bounded SSE windows, reconnecting from delivered committed IDs.

        A page high-water mark is never an acknowledgment. An unknown gap or
        malformed frame refuses; consumers retain each event's sequence for
        later attach. No new run, allowance or cancellation is issued here.
        """
        if type(after_sequence) is not int or after_sequence < 0:
            raise ValueError("after_sequence must be nonnegative")
        if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 86400:
            raise ValueError("timeout_seconds must be positive and at most one day")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        cursor = after_sequence
        path = "/v1/rollouts/" + quote(rollout_id, safe="") + "/events"
        try:
            async with asyncio.timeout(timeout_seconds):
                while loop.time() < deadline:
                    async with self._client.stream(
                        "GET",
                        path,
                        params={
                            "format": "sse",
                            "follow": "true",
                            "follow_seconds": 30,
                        },
                        headers={"Last-Event-ID": str(cursor)},
                    ) as response:
                        response.raise_for_status()
                        if not response.headers.get("content-type", "").startswith(
                            "text/event-stream"
                        ):
                            raise PoolClientError("Rhodes event follow requires text/event-stream")
                        async for identifier, event in _operator_frames(response.aiter_bytes()):
                            sequence = event.get("sequence")
                            if (
                                type(sequence) is not int
                                or sequence != identifier
                                or sequence != cursor + 1
                            ):
                                raise PoolClientError(
                                    "Rhodes committed event sequence gap or mismatched ID"
                                )
                            if event.get("rollout_id", rollout_id) != rollout_id:
                                raise PoolClientError("Rhodes event belongs to a different rollout")
                            cursor = sequence
                            yield event
                    page = await self.event_page(rollout_id, after_sequence=cursor)
                    pending = any(
                        page.get(key, False)
                        for key in ("publication_pending", "cleanup_pending", "inference_pending")
                    )
                    if (
                        page.get("status") in TERMINAL_STATUSES
                        and not pending
                        and page.get("high_water") == cursor
                    ):
                        return
                    # A proxy may close an idle connection immediately. Bound
                    # reconnect frequency without inventing a new effect retry.
                    await asyncio.sleep(min(0.25, max(0, deadline - loop.time())))
                raise PoolRolloutTimeout(
                    f"Rhodes event follow timed out at sequence {cursor}; execution remains attached to its original ID"
                )
        except TimeoutError as error:
            raise PoolRolloutTimeout(
                f"Rhodes event follow timed out at sequence {cursor}; execution remains attached to its original ID"
            ) from error
        except httpx.HTTPStatusError as error:
            raise _http_refusal(error) from error


async def _operator_frames(
    chunks: AsyncIterator[bytes],
) -> AsyncIterator[tuple[int, dict[str, Any]]]:
    pending = bytearray()
    maximum_frame_bytes = 1024 * 1024
    async for chunk in chunks:
        # Parse incrementally even when a transport combines many small frames.
        for offset in range(0, len(chunk), 65536):
            search_from = max(0, len(pending) - 3)
            pending.extend(chunk[offset : offset + 65536])
            while True:
                separators = [
                    (pending.find(marker, search_from), marker) for marker in (b"\n\n", b"\r\n\r\n")
                ]
                positions = [(index, marker) for index, marker in separators if index >= 0]
                if not positions:
                    break
                index, marker = min(positions, key=lambda item: item[0])
                frame = pending[:index]
                pending = pending[index + len(marker) :]
                search_from = 0
                if len(frame) > maximum_frame_bytes:
                    raise PoolClientError("Rhodes event frame exceeds 1 MiB")
                identifiers = []
                data = []
                for line in frame.splitlines():
                    if line.startswith(b"id:"):
                        identifiers.append(line[3:].strip())
                    elif line.startswith(b"data:"):
                        data.append(line[5:].strip())
                if not data and not identifiers:
                    continue
                try:
                    if len(identifiers) != 1 or len(data) != 1 or not identifiers[0].isdigit():
                        raise ValueError("Expected exact operator event frame")
                    event = json.loads(data[0])
                    if not isinstance(event, dict):
                        raise ValueError("Expected operator event object")
                    yield int(identifiers[0]), event
                except (ValueError, UnicodeError) as error:
                    raise PoolClientError("Malformed Rhodes operator SSE event") from error
            if len(pending) > maximum_frame_bytes:
                raise PoolClientError("Rhodes event frame exceeds 1 MiB")
    if pending.strip():
        # EOF cannot turn an unterminated event into committed evidence.
        raise PoolClientError("Rhodes event stream ended with an incomplete frame")
