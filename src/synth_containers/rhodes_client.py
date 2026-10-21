"""Rhodes registry, lease and committed event lifecycle over PoolClient.

See docs/specs/rhodes-client-v1.md. No provider fallback or client-side compute
is performed. Disconnect and timeout detach; cancel is an explicit operation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any
from pathlib import Path
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

    async def prepare_environment_source(self, declaration: Mapping[str, Any]) -> dict[str, Any]:
        return await self._request("POST", "/environment_sources/prepare", payload=declaration)

    async def finalize_environment_source(
        self, publication_id: str, declaration: Mapping[str, Any]
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/environment_sources/" + quote(publication_id, safe="") + "/finalize",
            payload=declaration,
        )

    async def upload_environment_source(
        self,
        path: str | Path,
        *,
        operation_id: str,
        logical_path: str = "source.tar",
        media_type: str = "application/x-tar",
    ) -> dict[str, Any]:
        """Upload a bounded snapshot through Artifact Platform, returning its reference.

        Storage PUTs use a separate client without the Synth bearer token.
        Finalize verifies content even when a create-only target already exists.
        Retrying uses the same operation ID and never starts a build or rollout.
        """
        if not operation_id or len(operation_id) > 256:
            raise ValueError("A stable bounded upload operation_id is required")

        def snapshot():
            with Path(path).open("rb") as stream:
                content = stream.read(64 * 1024 * 1024 + 1)
            if not 0 < len(content) <= 64 * 1024 * 1024:
                raise ValueError("Environment source must be between 1 byte and 64 MiB")
            return content

        content = await asyncio.to_thread(snapshot)
        digest = hashlib.sha256(content).hexdigest()
        declaration = {
            "schema_version": "synth.rhodes-source-upload.v1",
            "operation_id": operation_id,
            "source": {
                "logical_path": logical_path,
                "digest_sha256": digest,
                "size_bytes": len(content),
                "media_type": media_type,
            },
        }
        prepared = await self.prepare_environment_source(declaration)
        targets = prepared.get("upload_targets")
        publication_id = prepared.get("publication_id")
        if (
            not isinstance(publication_id, str)
            or not publication_id
            or not isinstance(targets, list)
            or len(targets) > 1
        ):
            raise PoolClientError("Rhodes source preparation returned invalid coordinates")
        for target in targets:
            if (
                not isinstance(target, dict)
                or target.get("digest_sha256") != digest
                or target.get("logical_path") != logical_path
            ):
                raise PoolClientError("Rhodes upload target does not match the source declaration")
            url = httpx.URL(target.get("upload_url", ""))
            headers = target.get("required_headers")
            if (
                (
                    url.scheme != "https"
                    and not (url.scheme == "http" and url.host in {"127.0.0.1", "::1", "localhost"})
                )
                or url.username
                or url.password
                or url.fragment
            ):
                raise PoolClientError("Rhodes source upload requires a protected storage URL")
            if not isinstance(headers, dict) or any(
                not isinstance(k, str) or not isinstance(v, str) for k, v in headers.items()
            ):
                raise PoolClientError("Rhodes upload headers are malformed")
            normalized = {k.lower(): v for k, v in headers.items()}
            if (
                normalized.get("if-none-match") != "*"
                or {"authorization", "cookie", "proxy-authorization"} & normalized.keys()
            ):
                raise PoolClientError("Rhodes upload requires create-only storage headers")
            try:
                async with httpx.AsyncClient(
                    trust_env=False, follow_redirects=False, timeout=60
                ) as storage:
                    response = await storage.put(url, headers=headers, content=content)
                    if response.status_code != 412:
                        response.raise_for_status()
            except httpx.HTTPError:
                # Signed storage URLs can contain credentials; omit them from
                # the refusal and retain the original publication for retry.
                raise PoolClientError(
                    "Rhodes source transfer failed; retry the original operation"
                ) from None
        finalized = await self.finalize_environment_source(publication_id, declaration)
        manifest_digest = finalized.get("manifest_digest")
        if (
            finalized.get("status") != "committed"
            or finalized.get("publication_id") != publication_id
            or not isinstance(manifest_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", manifest_digest)
        ):
            raise PoolClientError("Rhodes source publication did not commit its original identity")
        return {
            "publication_id": publication_id,
            "manifest_digest": "sha256:" + manifest_digest,
            "logical_path": logical_path,
            "digest": "sha256:" + digest,
            "size_bytes": len(content),
        }

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

    async def bind_definition(
        self, name: str, release_id: str, *, expected_revision: int, operation_id: str
    ) -> dict[str, Any]:
        """Create at revision0 or atomically bind a new immutable release.

        Reuse the operation ID and original body after uncertain transport loss.
        Replaying returns that operation's receipt even after later revisions.
        """
        if not operation_id:
            raise ValueError("A stable definition operation_id is required")
        return await self._request(
            "PUT",
            "/environment_definitions/" + quote(name, safe=""),
            payload={
                "schema_version": "synth.rhodes-definition-bind.v1",
                "release_id": release_id,
                "expected_revision": expected_revision,
                "operation_id": operation_id,
            },
            replay_safe=True,
        )

    async def archive_definition(
        self, name: str, *, expected_revision: int, operation_id: str
    ) -> dict[str, Any]:
        """Archive discoverability; retain immutable releases and history."""
        if not operation_id:
            raise ValueError("A stable definition operation_id is required")
        return await self._request(
            "POST",
            "/environment_definitions/" + quote(name, safe="") + "/archive",
            payload={
                "schema_version": "synth.rhodes-definition-archive.v1",
                "expected_revision": expected_revision,
                "operation_id": operation_id,
            },
            replay_safe=True,
        )

    async def get_definition(self, name: str) -> dict[str, Any]:
        return await self._request("GET", "/environment_definitions/" + quote(name, safe=""))

    async def list_definitions(
        self, *, after_name: str | None = None, include_archived: bool = False, limit: int = 100
    ) -> dict[str, Any]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        return await self._request(
            "GET",
            "/environment_definitions",
            params={
                "include_archived": str(include_archived).lower(),
                "limit": limit,
                **({"after_name": after_name} if after_name is not None else {}),
            },
        )

    async def definition_history(
        self, name: str, *, after_revision: int = 0, limit: int = 100
    ) -> dict[str, Any]:
        return await self._request(
            "GET",
            "/environment_definitions/" + quote(name, safe="") + "/history",
            params={"after_revision": after_revision, "limit": limit},
        )

    async def definition_operation(self, operation_id: str) -> dict[str, Any]:
        return await self._request(
            "GET", "/environment_definition_operations/" + quote(operation_id, safe="")
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

    async def task_readiness(self, pool_id: str, task_id: str) -> dict[str, Any]:
        """Read provider-observed readiness; this does not reserve capacity."""
        return await self._request(
            "GET",
            "/pools/"
            + quote(pool_id, safe="")
            + "/tasks/"
            + quote(task_id, safe="")
            + "/readiness",
        )

    async def acquire_observed_lease(
        self,
        pool_id: str,
        task_id: str,
        *,
        image_kind: str,
        idempotency_key: str,
        project_id: str | None = None,
        ttl_seconds: int = 900,
    ) -> dict[str, Any]:
        """Reserve one observed fresh-instance slot on a supported profile.

        Reuse/reset and hosted observation support are explicit server decisions.
        Close releases this borrowed slot; its pool retains provider custody.
        """
        if not idempotency_key:
            raise ValueError("A stable lease idempotency_key is required")
        return await self._request(
            "POST",
            "/pools/"
            + quote(pool_id, safe="")
            + "/tasks/"
            + quote(task_id, safe="")
            + "/capacity_leases",
            payload={
                "schema_version": "synth.rhodes-observed-capacity.v1",
                "image_kind": image_kind,
                "idempotency_key": idempotency_key,
                "project_id": project_id,
                "ttl_seconds": ttl_seconds,
                "slots": 1,
                "reset": "fresh_instance",
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
                    event = json.loads(
                        data[0],
                        object_pairs_hook=_unique_event_fields,
                        parse_constant=_invalid_number,
                        parse_float=_finite_number,
                    )
                    if not isinstance(event, dict):
                        raise ValueError("Expected operator event object")
                    yield int(identifiers[0]), event
                except (ValueError, UnicodeError, RecursionError) as error:
                    raise PoolClientError("Malformed Rhodes operator SSE event") from error
            if len(pending) > maximum_frame_bytes:
                raise PoolClientError("Rhodes event frame exceeds 1 MiB")
    if pending.strip():
        # EOF cannot turn an unterminated event into committed evidence.
        raise PoolClientError("Rhodes event stream ended with an incomplete frame")


def _unique_event_fields(pairs):
    value = {}
    for name, content in pairs:
        if name in value:
            raise ValueError("Duplicate operator event field")
        value[name] = content
    return value


def _invalid_number(value):
    raise ValueError("Nonfinite operator event number")


def _finite_number(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Nonfinite operator event number")
    return parsed
