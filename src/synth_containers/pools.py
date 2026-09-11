"""Client for driving container-pool rollouts from outside the platform.

`HTTPContainerClient` speaks the container contract directly, against a single
container's base URL. This module speaks the *platform* contract: it targets a
deployed pool through the backend's `/v1/pools*` routes, so an eval running on a
laptop can fan work out across managed compute.

Why this lives in `synth-containers` rather than `synth-ai`: this package
already owns the public Synth container contract, already models rollout
submission and completion payloads, and is already installed wherever container
work happens. `synth-ai` is the research-product SDK; pool discovery belongs
there, but driving rollouts does not.

**Fan out server-side, never client-side.** It is tempting to resolve each
container's URL from the pool and post rollouts straight at it. Do not. Requests
that skip the platform skip admission, resource clamping, usage attribution, and
receipts — every guarantee the pool exists to provide. Every method here submits
through the backend and polls for completion, which is slower per call and
correct.

Lifecycle coverage, as of 2026-08-03:

===============  ====================================================
create           ``create_pool`` — ``POST /v1/pools``
read             ``list_pools`` / ``get_pool`` / ``metrics``
update           ``update_pool`` (PATCH) / ``replace_pool`` (PUT)
delete           ``delete_pool`` — archives; see the method note
metadata         ``pool_metadata``, free-form and uninterpreted
tasks            list / create / update / delete
image releases   create / register / list / get / bind / delete
pause, resume    ``pause_pool`` / ``resume_pool``
===============  ====================================================

``pause_pool`` and ``resume_pool`` exist and target ``POST /v1/pools/{id}/pause``
and ``/resume``. Backends predating the lifecycle migration still receive a
typed `PoolStateError` instead of a bare 404.
"""

from __future__ import annotations

import asyncio
import base64
import fnmatch
import hashlib
import io
import json
import logging
import math
import os
import tarfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote
from uuid import UUID, NAMESPACE_URL, uuid5
from typing import AsyncIterator, Any, Iterable, Mapping, Sequence

import httpx

from .ontology import (
    ContainerComputeProvider,
    ContainerHarnessSubtype,
    ContainerSourceKind,
    ContainerSubtype,
)

logger = logging.getLogger(__name__)

DEFAULT_BACKEND_URL = "https://api.usesynth.ai"
POOL_API_PREFIX = "/v1"

# `ContainerPoolRolloutStatus` in the backend. A rollout that reaches one of
# these will not change again, so polling stops.
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
SUCCESS_STATUSES = frozenset({"completed"})

DEFAULT_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_POLL_TIMEOUT_SECONDS = 1800.0
DEFAULT_MAX_IN_FLIGHT = 8

HARBOR_CONTAINER_SUBTYPE = ContainerSubtype.HARBOR.value
HARBOR_TASK_CONFIG = "task.toml"
HARBOR_ENVIRONMENT_DOCKERFILE = "environment/Dockerfile"


class PoolState:
    """Lifecycle states for a container pool.

    A pool is not usable the moment it is created. Source has to be uploaded,
    an image has to be built or pulled, and a provider sandbox has to warm
    before a rollout can be admitted. Today every one of those phases is
    invisible: `ContainerPool.status` is constrained to ``('active','archived')``
    by a CHECK constraint, so a pool reads "active" while its image is still
    building, and a rollout submitted then fails for no stated reason.

    These are the states that already happen in reality. Naming them is what
    lets an operator tell *provisioning* apart from *broken*::

        created ──► uploading ──► building ──► warming ──► active
           │            │             │           │          │  ▲
           │            │             │           │          │  │ resume
           │            ▼             ▼           ▼          ▼  │
           │          failed ◄────────┴───────────┘        paused
           │            │                                     │
           │            │ (rebind a new revision)              │
           │            └──────────► building                  │
           │                                                   │
           ▼                                                   ▼
        archived ◄─────────────────────────────────────────────┘

    Rules the diagram encodes:

    - ``active`` is the only state that admits new rollouts.
    - ``paused`` stops admission; rollouts already in flight drain. Watch
      ``metrics()["active_rollouts"]`` to see the drain finish.
    - ``failed`` is recoverable, but only by binding a new image revision —
      which re-enters ``building``. It is not a dead end and not a retry loop.
    - ``archived`` is terminal and releases provider resources. Nothing leaves.
    - Any non-terminal state may be archived directly; abandoning a pool
      mid-build must not require driving it to ``active`` first.

    All states are implemented by the backend lifecycle migration. Older
    deployments may still report only ``active`` and ``archived``; the
    predicates remain backward compatible with those responses.
    """

    CREATED = "created"
    UPLOADING = "uploading"
    BUILDING = "building"
    WARMING = "warming"
    ACTIVE = "active"
    PAUSED = "paused"
    FAILED = "failed"
    ARCHIVED = "archived"


#: States recognized by the backend and public client.
IMPLEMENTED_POOL_STATES = frozenset(
    {
        PoolState.CREATED,
        PoolState.UPLOADING,
        PoolState.BUILDING,
        PoolState.WARMING,
        PoolState.ACTIVE,
        PoolState.PAUSED,
        PoolState.FAILED,
        PoolState.ARCHIVED,
    }
)

#: A pool in one of these is on its way to `active` without operator action.
PROVISIONING_POOL_STATES = frozenset(
    {
        PoolState.CREATED,
        PoolState.UPLOADING,
        PoolState.BUILDING,
        PoolState.WARMING,
    }
)

#: Nothing leaves these.
TERMINAL_POOL_STATES = frozenset({PoolState.ARCHIVED})

#: Allowed transitions. Archival is added to every non-terminal state below.
POOL_STATE_TRANSITIONS: dict[str, frozenset[str]] = {
    PoolState.CREATED: frozenset({PoolState.UPLOADING, PoolState.BUILDING, PoolState.FAILED}),
    PoolState.UPLOADING: frozenset({PoolState.BUILDING, PoolState.FAILED}),
    PoolState.BUILDING: frozenset({PoolState.WARMING, PoolState.FAILED}),
    PoolState.WARMING: frozenset({PoolState.ACTIVE, PoolState.FAILED}),
    PoolState.ACTIVE: frozenset({PoolState.PAUSED, PoolState.FAILED}),
    PoolState.PAUSED: frozenset({PoolState.ACTIVE, PoolState.FAILED}),
    # Recovery is always by rebinding a revision, never an in-place retry.
    PoolState.FAILED: frozenset({PoolState.BUILDING}),
    PoolState.ARCHIVED: frozenset(),
}

POOL_STATE_TRANSITIONS = {
    state: targets if state in TERMINAL_POOL_STATES else targets | {PoolState.ARCHIVED}
    for state, targets in POOL_STATE_TRANSITIONS.items()
}


def pool_state_is_terminal(state: str) -> bool:
    return _normalize_state(state) in TERMINAL_POOL_STATES


def pool_state_is_provisioning(state: str) -> bool:
    """True while a pool is still coming up on its own."""

    return _normalize_state(state) in PROVISIONING_POOL_STATES


def pool_state_admits_rollouts(state: str) -> bool:
    """Only `active` admits new work. Pausing is how you stop it."""

    return _normalize_state(state) == PoolState.ACTIVE


def validate_pool_transition(current: str, target: str) -> None:
    """Raise `PoolStateError` if `current -> target` is not allowed."""

    source, destination = _normalize_state(current), _normalize_state(target)
    allowed = POOL_STATE_TRANSITIONS.get(source)
    if allowed is None:
        raise PoolStateError(f"unknown pool state {current!r}")
    if destination not in allowed:
        raise PoolStateError(
            f"pool cannot move {source} -> {destination}; "
            f"allowed: {sorted(allowed) or 'none (terminal)'}"
        )


def _normalize_state(state: str) -> str:
    return str(state or "").strip().lower()


class PoolClientError(RuntimeError):
    """A pool request failed, or a rollout did not reach a terminal state."""


class PoolStateError(PoolClientError):
    """A pool lifecycle transition is not permitted."""


class PoolRolloutTimeout(PoolClientError):
    """A rollout was still non-terminal when its deadline elapsed."""


class HarborPoolSupportError(PoolClientError):
    """A Harbor task requires pool behavior the platform cannot honor yet."""


@dataclass(frozen=True, slots=True)
class HarborTaskBundle:
    """Pool-facing projection of a Harbor task directory.

    Harbor is a container subtype and task/evaluation format, not a runtime.
    The underlying runtime remains an ordinary image or Docker build context.
    The backend translates setup, agent, and verifier phases onto that runtime.
    Requirements the pool cannot faithfully represent are retained and rejected
    before upload, never silently ignored.
    """

    task_directory: Path
    build_context_root: Path
    schema_version: str
    task_name: str
    task_config_path: str
    dockerfile_path: str
    cpu_cores: float | None
    memory_mb: int | None
    storage_mb: int | None
    allow_internet: bool | None
    gpu_count: int | None
    gpu_types: tuple[str, ...]
    build_timeout_seconds: float | None
    agent_timeout_seconds: float | None
    verifier_timeout_seconds: float | None
    verifier_environment_mode: str
    has_tpu_config: bool

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        *,
        build_context_root: str | Path | None = None,
    ) -> "HarborTaskBundle":
        base = Path(root).resolve()
        context = Path(build_context_root).resolve() if build_context_root else base
        try:
            base.relative_to(context)
        except ValueError as error:
            raise HarborPoolSupportError(
                f"Harbor task directory {base} is outside build context {context}"
            ) from error
        task_config = base / HARBOR_TASK_CONFIG
        if not task_config.is_file():
            raise HarborPoolSupportError(f"Harbor task is missing {task_config}")
        try:
            payload = tomllib.loads(task_config.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise HarborPoolSupportError(
                f"Harbor task config is not readable TOML: {task_config}: {error}"
            ) from error

        task = _mapping(payload.get("task"))
        environment = _mapping(payload.get("environment"))
        agent = _mapping(payload.get("agent"))
        verifier = _mapping(payload.get("verifier"))
        dockerfile = base / HARBOR_ENVIRONMENT_DOCKERFILE
        if not dockerfile.is_file():
            raise HarborPoolSupportError(
                "Harbor pool source-build currently requires "
                f"{HARBOR_ENVIRONMENT_DOCKERFILE}; docker_image-only tasks are not yet supported"
            )

        return cls(
            task_directory=base,
            build_context_root=context,
            schema_version=str(payload.get("schema_version") or "").strip(),
            task_name=str(task.get("name") or base.name).strip() or base.name,
            task_config_path=task_config.relative_to(context).as_posix(),
            dockerfile_path=dockerfile.relative_to(context).as_posix(),
            cpu_cores=_optional_float(environment.get("cpus"), field_name="environment.cpus"),
            memory_mb=_optional_int(
                environment.get("memory_mb"), field_name="environment.memory_mb"
            ),
            storage_mb=_optional_int(
                environment.get("storage_mb"), field_name="environment.storage_mb"
            ),
            allow_internet=_optional_bool(
                environment.get("allow_internet"), field_name="environment.allow_internet"
            ),
            gpu_count=_optional_int(environment.get("gpus"), field_name="environment.gpus"),
            gpu_types=_string_tuple(environment.get("gpu_types")),
            build_timeout_seconds=_optional_float(
                environment.get("build_timeout_sec"),
                field_name="environment.build_timeout_sec",
            ),
            agent_timeout_seconds=_optional_float(
                agent.get("timeout_sec"), field_name="agent.timeout_sec"
            ),
            verifier_timeout_seconds=_optional_float(
                verifier.get("timeout_sec"), field_name="verifier.timeout_sec"
            ),
            verifier_environment_mode=str(verifier.get("environment_mode") or "shared")
            .strip()
            .lower(),
            has_tpu_config=bool(_mapping(environment.get("tpu"))),
        )

    def unsupported_requirements(self) -> tuple[str, ...]:
        """Return requirements that a pool must refuse rather than ignore."""

        unsupported: list[str] = []
        if self.allow_internet is False:
            unsupported.append("environment.allow_internet=false (egress isolation)")
        if self.storage_mb not in {None, 0}:
            unsupported.append("environment.storage_mb")
        if self.gpu_count not in {None, 0} or self.gpu_types:
            unsupported.append("environment.gpus/gpu_types")
        if self.has_tpu_config:
            unsupported.append("environment.tpu")
        if self.verifier_environment_mode != "shared":
            unsupported.append("verifier.environment_mode=separate")
        return tuple(unsupported)

    def require_supported(self) -> None:
        unsupported = self.unsupported_requirements()
        if unsupported:
            raise HarborPoolSupportError(
                "Harbor task cannot be translated faithfully by the current pool backend; "
                f"unsupported requirements: {', '.join(unsupported)}"
            )

    def pool_limits(self) -> dict[str, Any]:
        """Map only resource dimensions with equivalent pool semantics."""

        limits: dict[str, Any] = {}
        if self.cpu_cores is not None:
            limits["cpu_cores"] = self.cpu_cores
        if self.memory_mb is not None:
            limits["memory_mb"] = self.memory_mb
        return limits

    def release_metadata(
        self,
        *,
        container_harness_subtype: ContainerHarnessSubtype | str | None = None,
    ) -> dict[str, Any]:
        """Metadata needed by the backend's native three-phase translator."""

        metadata: dict[str, Any] = {
            "container_subtype": HARBOR_CONTAINER_SUBTYPE,
            "task_format": "harbor",
            "harbor_schema_version": self.schema_version,
            "harbor_task_name": self.task_name,
            "harbor_task_config_path": self.task_config_path,
            "harbor_dockerfile_path": self.dockerfile_path,
            "harbor_allow_internet": self.allow_internet,
            "harbor_phase_timeouts_seconds": {
                "build": self.build_timeout_seconds,
                "agent": self.agent_timeout_seconds,
                "verifier": self.verifier_timeout_seconds,
            },
            "harbor_verifier_environment_mode": self.verifier_environment_mode,
        }
        if container_harness_subtype is not None:
            metadata["container_harness_subtype"] = ContainerHarnessSubtype.parse(
                container_harness_subtype
            ).value
        return metadata


@dataclass(slots=True)
class RolloutOutcome:
    """One rollout's terminal state, as the platform reported it."""

    rollout_id: str
    status: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status in SUCCESS_STATUSES

    @property
    def result(self) -> Mapping[str, Any]:
        value = self.payload.get("result")
        return value if isinstance(value, Mapping) else {}


class PoolClient:
    """Submit and collect container-pool rollouts through the backend."""

    def __init__(
        self,
        *,
        api_key: str,
        backend_url: str = DEFAULT_BACKEND_URL,
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        key = str(api_key or "").strip()
        if not key:
            raise PoolClientError("api_key is required")
        self.backend_url = str(backend_url or DEFAULT_BACKEND_URL).rstrip("/")
        self._api_key = key
        self._max_retries = max(0, int(max_retries))
        self._retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self.backend_url,
            timeout=httpx.Timeout(float(timeout_seconds)),
            headers={"Authorization": f"Bearer {key}"},
        )

    @classmethod
    def from_env(cls, **kwargs: Any) -> "PoolClient":
        """Build from `SYNTH_API_KEY` and `SYNTH_BACKEND_URL`."""

        api_key = os.environ.get("SYNTH_API_KEY", "")
        backend_url = os.environ.get("SYNTH_BACKEND_URL") or DEFAULT_BACKEND_URL
        return cls(api_key=api_key, backend_url=backend_url, **kwargs)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "PoolClient":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()

    # -- transport ---------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
        optional: bool = False,
    ) -> dict[str, Any]:
        url = f"{POOL_API_PREFIX}{path}"
        last_error: Exception | None = None
        retries = self._max_retries if method.upper() in {"GET", "HEAD"} else 0
        for attempt in range(retries + 1):
            try:
                response = await self._client.request(
                    method.upper(),
                    url,
                    json=dict(payload) if payload is not None else None,
                    params=dict(params) if params else None,
                )
                if optional and response.status_code == 404:
                    return {}
                response.raise_for_status()
                if not str(response.text or "").strip():
                    return {}
                body = response.json()
                return body if isinstance(body, dict) else {"value": body}
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if optional and status == 404:
                    return {}
                detail = str(exc.response.text or "").strip()
                message = f"{status} {exc.response.reason_phrase}"
                if detail:
                    message = f"{message}: {detail[:1000]}"
                # A refusal is the platform's considered answer — an admission
                # denial, a clamped resource, a bad payload. Retrying it just
                # burns quota and hides the reason, so surface it immediately.
                if 400 <= status < 500 and status not in {408, 429}:
                    raise PoolClientError(f"{method.upper()} {url} refused: {message}") from exc
                last_error = PoolClientError(message)
            except (httpx.RequestError, ValueError) as exc:
                last_error = exc
            if attempt < retries:
                await asyncio.sleep(self._retry_backoff_seconds * (attempt + 1))
        raise PoolClientError(
            f"pool request failed {method.upper()} {url}: {last_error}"
        ) from last_error

    # -- lifecycle ---------------------------------------------------------

    async def list_pools(self) -> list[dict[str, Any]]:
        body = await self._request("GET", "/pools")
        pools = body.get("pools")
        if isinstance(pools, list):
            return [item for item in pools if isinstance(item, dict)]
        return [item for item in body.get("value", []) if isinstance(item, dict)]

    async def get_pool(self, pool_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/pools/{pool_id}")

    async def create_pool(
        self,
        *,
        name: str,
        backend: str,
        pool_type: str | None = None,
        template: str | None = None,
        capacity: int | None = None,
        concurrency: int | None = None,
        pool_config: Mapping[str, Any] | None = None,
        pool_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a pool. `pool_metadata` is free-form and round-trips intact.

        Put anything an operator needs to recognise this pool later in
        ``pool_metadata`` — image digest, source commit, owning experiment. The
        platform does not interpret it, and it is the only field that survives
        without a schema change.
        """

        payload: dict[str, Any] = {"name": name, "backend": backend}
        for key, value in (
            ("pool_type", pool_type),
            ("template", template),
            ("capacity", capacity),
            ("concurrency", concurrency),
            ("pool_config", dict(pool_config) if pool_config else None),
            ("pool_metadata", dict(pool_metadata) if pool_metadata else None),
        ):
            if value is not None:
                payload[key] = value
        return await self._request("POST", "/pools", payload=payload)

    async def update_pool(self, pool_id: str, **fields: Any) -> dict[str, Any]:
        """Partially update a pool (PATCH). Unset fields are left alone."""

        if not fields:
            raise PoolClientError("update_pool requires at least one field")
        return await self._request("PATCH", f"/pools/{pool_id}", payload=fields)

    async def replace_pool(self, pool_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Full replace (PUT). Prefer `update_pool` unless you mean to reset."""

        return await self._request("PUT", f"/pools/{pool_id}", payload=payload)

    async def delete_pool(self, pool_id: str) -> dict[str, Any]:
        """Delete a pool.

        The model carries `status IN ('active','archived')` and an
        `archived_at`, so this is expected to archive rather than hard-delete.
        Confirm against a live backend before relying on either reading.
        """

        return await self._request("DELETE", f"/pools/{pool_id}")

    async def pause_pool(self, pool_id: str) -> dict[str, Any]:
        """Stop admitting new rollouts; let in-flight work drain.

        Against a pre-lifecycle backend this raises `PoolStateError` naming the
        missing capability rather than surfacing a bare 404.
        """

        return await self._transition(pool_id, "pause")

    async def resume_pool(self, pool_id: str) -> dict[str, Any]:
        """Return a paused pool to `active`. See `pause_pool`."""

        return await self._transition(pool_id, "resume")

    async def _transition(self, pool_id: str, verb: str) -> dict[str, Any]:
        body = await self._request("POST", f"/pools/{pool_id}/{verb}", optional=True)
        if not body:
            raise PoolStateError(
                f"pool {verb} is not available on this backend: "
                f"the /{verb} lifecycle route is missing. "
                "Upgrade the backend container-pool lifecycle migration."
            )
        return body

    async def wait_until_active(
        self,
        pool_id: str,
        *,
        timeout_seconds: float = 900.0,
        poll_interval_seconds: float = 3.0,
    ) -> dict[str, Any]:
        """Poll a pool through provisioning until it admits rollouts.

        Raises as soon as the pool reports `failed`, rather than waiting out
        the timeout — a build that failed will not un-fail.
        """

        deadline = asyncio.get_running_loop().time() + max(0.0, float(timeout_seconds))
        interval = max(0.1, float(poll_interval_seconds))
        last_state = ""
        while True:
            pool = await self.get_pool(pool_id)
            last_state = _normalize_state(str(pool.get("status") or ""))
            if pool_state_admits_rollouts(last_state):
                return pool
            if last_state == PoolState.FAILED:
                raise PoolStateError(f"pool {pool_id} failed to provision")
            if last_state in TERMINAL_POOL_STATES:
                raise PoolStateError(f"pool {pool_id} is {last_state} and will not become active")
            if asyncio.get_running_loop().time() >= deadline:
                raise PoolRolloutTimeout(
                    f"pool {pool_id} still {last_state or 'unknown'} after {timeout_seconds}s"
                )
            await asyncio.sleep(interval)

    async def metrics(self, pool_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/pools/{pool_id}/metrics", optional=True)

    async def pool_urls(self, pool_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/pools/{pool_id}/urls", optional=True)

    async def list_rollouts(self, pool_id: str) -> list[dict[str, Any]]:
        body = await self._request("GET", f"/pools/{pool_id}/rollouts", optional=True)
        rollouts = body.get("rollouts")
        return (
            [item for item in rollouts if isinstance(item, dict)]
            if isinstance(rollouts, list)
            else []
        )

    # -- tasks -------------------------------------------------------------

    @staticmethod
    def _deployment_coordinate(value: str) -> str:
        if (
            not isinstance(value, str)
            or not 1 <= len(value) <= 256
            or value in {".", ".."}
            or "/" in value
        ):
            raise PoolClientError(
                "deployment coordinate must be a nonempty path segment of at most 256 characters"
            )
        return quote(value, safe="")

    async def get_deployment(
        self, pool_id: str, task_id: str, *, project_id: str
    ) -> dict[str, Any]:
        """Read the deployment revision and observed resource usage."""
        return await self._request(
            "GET",
            f"/pools/{self._deployment_coordinate(pool_id)}/deployments/{self._deployment_coordinate(task_id)}",
            params={"project_id": str(UUID(project_id))},
        )

    async def mutate_deployment(
        self,
        pool_id: str,
        task_id: str,
        *,
        project_id: str,
        operation: str,
        idempotency_key: str,
        payload: Mapping[str, Any] | None = None,
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        """Use the shared actor intent service; never repeat an ambiguous mutation.

        Read the operation receipt before deciding how to reconcile uncertainty.
        Update/delete require the exact revision returned by get_deployment.
        """
        if operation not in {"create", "update", "delete"}:
            raise PoolClientError("deployment operation must be create, update, or delete")
        if (
            not isinstance(idempotency_key, str)
            or not 1 <= len(idempotency_key) <= 256
            or idempotency_key.strip() != idempotency_key
        ):
            raise PoolClientError("deployment idempotency key must be 1-256 non-padded characters")
        if operation != "create" and not expected_revision:
            raise PoolClientError("update/delete require expected_revision")
        body = {
            "project_id": str(UUID(project_id)),
            "operation": operation,
            "idempotency_key": idempotency_key,
            "payload": dict(payload or {}),
            "expected_revision": expected_revision,
        }
        encoded = json.dumps(body, allow_nan=False).encode()
        if len(encoded) > 1024 * 1024:
            raise PoolClientError("deployment request exceeds 1 MiB")
        return await self._request(
            "POST",
            f"/pools/{self._deployment_coordinate(pool_id)}/deployments/{self._deployment_coordinate(task_id)}/operations",
            payload=body,
        )

    async def find_deployment_operation(
        self, pool_id: str, task_id: str, *, project_id: str, idempotency_key: str
    ) -> dict[str, Any]:
        """Recover the accepted intent after a lost mutation response."""
        return await self._request(
            "GET",
            f"/pools/{self._deployment_coordinate(pool_id)}/deployments/{self._deployment_coordinate(task_id)}/operations",
            params={"project_id": str(UUID(project_id)), "idempotency_key": idempotency_key},
        )

    async def get_deployment_operation(
        self, pool_id: str, operation_id: str, *, project_id: str
    ) -> dict[str, Any]:
        return await self._request(
            "GET",
            f"/pools/{self._deployment_coordinate(pool_id)}/deployment-operations/{self._deployment_coordinate(operation_id)}",
            params={"project_id": str(UUID(project_id))},
        )

    async def list_tasks(self, pool_id: str) -> list[dict[str, Any]]:
        body = await self._request("GET", f"/pools/{pool_id}/tasks", optional=True)
        tasks = body.get("tasks")
        return [item for item in tasks if isinstance(item, dict)] if isinstance(tasks, list) else []

    async def create_task(self, pool_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        return await self._request("POST", f"/pools/{pool_id}/tasks", payload=payload)

    async def update_task(self, pool_id: str, task_id: str, **fields: Any) -> dict[str, Any]:
        if not fields:
            raise PoolClientError("update_task requires at least one field")
        return await self._request("PATCH", f"/pools/{pool_id}/tasks/{task_id}", payload=fields)

    async def delete_task(self, pool_id: str, task_id: str) -> dict[str, Any]:
        return await self._request("DELETE", f"/pools/{pool_id}/tasks/{task_id}")

    # -- runtime image releases -------------------------------------------
    #
    # Pool images are NOT actor images. They live in
    # `container_pool_runtime_image_releases`, are scoped to a pool rather than
    # an org, and carry pool-only concepts: source kind, compute provider,
    # `dockerfile_path`, and `entrypoint`. The SMR actor image lane has its own
    # table, its own `RuntimeImageKind` vocabulary, its own registry env vars,
    # and no interface modes at all. Push/pull *mechanics* can be shared; the
    # namespaces, lifecycles, and vocabularies must not be.

    async def list_image_releases(self, pool_id: str) -> list[dict[str, Any]]:
        body = await self._request("GET", f"/pools/{pool_id}/runtime_image_releases", optional=True)
        releases = body.get("releases", body.get("items"))
        return (
            [item for item in releases if isinstance(item, dict)]
            if isinstance(releases, list)
            else []
        )

    async def get_image_release(self, pool_id: str, release_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/pools/{pool_id}/runtime_image_releases/{release_id}")

    async def create_image_release(
        self,
        pool_id: str,
        *,
        archive: bytes | str | Path,
        dockerfile_path: str = "Dockerfile",
        source_kind: ContainerSourceKind | str = ContainerSourceKind.DOCKER_CONTEXT,
        compute_provider: ContainerComputeProvider | str | None = None,
        name: str | None = None,
        entrypoint: str | None = None,
        env_vars: Mapping[str, str] | None = None,
        limits: Mapping[str, Any] | None = None,
        filename: str = "runtime-source.tar.gz",
        release_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Publish a build context and create a source-built release.

        `archive` is a tar or zip — the backend sniffs the format. Pass bytes,
        or a path to an archive on disk; `pack_build_context` produces one that
        is byte-identical for identical inputs.

        The context is uploaded to blob storage and the release records
        `source_storage_uri`, `source_content_hash`, and `source_size_bytes`.
        We verify the hash the backend computed against our own before
        returning: a context that silently differs from what we packed would
        produce an image nobody can reproduce, and it would not surface until
        the container behaved oddly at rollout time.
        """

        payload_bytes = _archive_bytes(archive)
        local_hash = hashlib.sha256(payload_bytes).hexdigest()
        normalized_source_kind = ContainerSourceKind.parse(source_kind).value
        normalized_compute_provider = (
            ContainerComputeProvider.parse(compute_provider).value
            if compute_provider is not None
            else None
        )
        payload: dict[str, Any] = {
            "source_kind": normalized_source_kind,
            "dockerfile_path": dockerfile_path,
            "archive_base64": base64.b64encode(payload_bytes).decode("ascii"),
            "filename": filename,
        }
        for key, value in (
            ("name", name),
            ("compute_provider", normalized_compute_provider),
            ("entrypoint", entrypoint),
            ("env_vars", dict(env_vars) if env_vars else None),
            ("limits", dict(limits) if limits else None),
            ("metadata", dict(release_metadata) if release_metadata else None),
        ):
            if value is not None:
                payload[key] = value

        release = await self._request(
            "POST", f"/pools/{pool_id}/runtime_image_releases", payload=payload
        )
        remote_hash = str(release.get("source_content_hash") or "").strip()
        if remote_hash and remote_hash != local_hash:
            raise PoolClientError(
                "uploaded build context hash mismatch: "
                f"sent {local_hash}, backend stored {remote_hash}"
            )
        return release

    async def create_harbor_task_release(
        self,
        pool_id: str,
        *,
        task_directory: str | Path,
        build_context_root: str | Path | None = None,
        compute_provider: ContainerComputeProvider | str,
        name: str | None = None,
        env_vars: Mapping[str, str] | None = None,
        container_harness_subtype: ContainerHarnessSubtype | str | None = None,
    ) -> dict[str, Any]:
        """Publish one Harbor-formatted task as a normal container release.

        The task is validated before upload. Harbor is sent only as a
        container subtype, never as a runtime or harness identity. Typed
        metadata tells the pool adapter to
        translate setup, agent, verifier, reward extraction, and artifact
        transfer rather than treating the context as an ordinary one-phase
        command job.
        """

        environment = HarborTaskBundle.from_directory(
            task_directory, build_context_root=build_context_root
        )
        environment.require_supported()
        archive = pack_build_context(environment.build_context_root)
        return await self.create_image_release(
            pool_id,
            archive=archive,
            dockerfile_path=environment.dockerfile_path,
            source_kind=ContainerSourceKind.DOCKER_CONTEXT,
            compute_provider=compute_provider,
            name=name or environment.task_name,
            env_vars=env_vars,
            limits=environment.pool_limits(),
            filename="harbor-task.tar.gz",
            release_metadata={
                **environment.release_metadata(container_harness_subtype=container_harness_subtype),
                "container_compute_provider": ContainerComputeProvider.parse(
                    compute_provider
                ).value,
            },
        )

    async def register_image_release(
        self,
        pool_id: str,
        *,
        image_ref: str,
        compute_provider: ContainerComputeProvider | str | None = None,
        name: str | None = None,
        entrypoint: str | None = None,
        env_vars: Mapping[str, str] | None = None,
        limits: Mapping[str, Any] | None = None,
        release_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a release against an image already in a registry.

        Prefer a digest-pinned ref (``host/repo@sha256:…``) over a tag. A tag is
        mutable, so a pool bound to one can silently change what it runs between
        rollouts — which makes a reward curve impossible to attribute. This
        warns rather than refuses, because tags are legitimate during
        development.
        """

        ref = str(image_ref or "").strip()
        if not ref:
            raise PoolClientError("image_ref is required")
        if "@sha256:" not in ref:
            logger.warning(
                "pool image_ref %r is not digest-pinned; the pool may silently "
                "change what it runs between rollouts",
                ref,
            )
        normalized_compute_provider = (
            ContainerComputeProvider.parse(compute_provider).value
            if compute_provider is not None
            else None
        )
        payload: dict[str, Any] = {
            "source_kind": ContainerSourceKind.IMAGE_REF.value,
            "image_ref": ref,
        }
        for key, value in (
            ("name", name),
            ("compute_provider", normalized_compute_provider),
            ("entrypoint", entrypoint),
            ("env_vars", dict(env_vars) if env_vars else None),
            ("limits", dict(limits) if limits else None),
            ("metadata", dict(release_metadata) if release_metadata else None),
        ):
            if value is not None:
                payload[key] = value
        return await self._request(
            "POST", f"/pools/{pool_id}/runtime_image_releases", payload=payload
        )

    async def delete_image_release(self, pool_id: str, release_id: str) -> dict[str, Any]:
        return await self._request(
            "DELETE", f"/pools/{pool_id}/runtime_image_releases/{release_id}"
        )

    async def bind_image_release(
        self, pool_id: str, release_id: str, *, expected_release_id: str | None = None
    ) -> dict[str, Any]:
        """Switch a release binding with optional compare-and-swap protection.

        An empty expected ID means unbound; None preserves legacy behavior.
        Binding is not proof that the target passed runtime readiness checks.
        """
        return await self._request(
            "POST",
            f"/pools/{pool_id}/runtime_image_releases/{release_id}/bind",
            params={"expected_release_id": expected_release_id}
            if expected_release_id is not None
            else None,
        )

    # -- container probes, proxied through the pool ------------------------

    async def health(self, pool_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/pools/{pool_id}/container/health")

    async def info(self, pool_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/pools/{pool_id}/container/info", optional=True)

    async def metadata(self, pool_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/pools/{pool_id}/container/metadata", optional=True)

    async def task_info(self, pool_id: str, *, seed: int | None = None) -> dict[str, Any]:
        params = {"seed": seed} if seed is not None else None
        return await self._request(
            "GET", f"/pools/{pool_id}/container/task_info", params=params, optional=True
        )

    async def preflight(self, pool_id: str) -> dict[str, Any]:
        """Probe all four contract endpoints before committing to a run.

        Returns what each probe answered, so a caller can fail with the specific
        missing endpoint rather than an opaque error on the first rollout.
        """

        health = await self.health(pool_id)
        info = await self.info(pool_id)
        metadata = await self.metadata(pool_id)
        task_info = await self.task_info(pool_id)
        return {
            "health": health,
            "info": info,
            "metadata": metadata,
            "task_info": task_info,
            "missing": [
                name
                for name, value in (
                    ("info", info),
                    ("metadata", metadata),
                    ("task_info", task_info),
                )
                if not value
            ],
        }

    # -- submission --------------------------------------------------------

    async def submit(self, pool_id: str, payload: Mapping[str, Any]) -> str:
        body = await self._request("POST", f"/pools/{pool_id}/rollouts", payload=payload)
        rollout_id = _rollout_id(body)
        if not rollout_id:
            raise PoolClientError(f"pool rollout response carried no id: {body}")
        return rollout_id

    async def submit_batch(self, pool_id: str, payloads: Sequence[Mapping[str, Any]]) -> list[str]:
        """Submit many rollouts in one request.

        The backend admits each item individually, so a batch that crosses an
        org's concurrency cap is partially admitted rather than wholly refused.
        Callers must not assume the returned list is the same length as
        ``payloads``.
        """

        if not payloads:
            return []
        body = await self._request(
            "POST",
            "/rollouts/batch",
            payload={"pool_id": pool_id, "rollouts": [dict(p) for p in payloads]},
        )
        rollouts = body.get("rollouts")
        if not isinstance(rollouts, list):
            raise PoolClientError(f"batch response carried no rollouts: {body}")
        ids = [_rollout_id(item) for item in rollouts if isinstance(item, dict)]
        return [value for value in ids if value]

    async def get_rollout(self, rollout_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/rollouts/{rollout_id}")

    async def get_result_snapshot(self, rollout_id: str) -> dict[str, Any]:
        """Read committed result custody and verify its exact bytes and owner.

        Execution completion alone does not imply publication. Reads use the
        authenticated Artifact Platform, never a target-supplied storage URL.
        """
        self._deployment_coordinate(rollout_id)
        rollout = await self.get_rollout(rollout_id)
        metadata = rollout.get("metadata") or {}
        receipt = metadata.get("result_publication") if isinstance(metadata, dict) else None
        if not isinstance(receipt, dict) or receipt.get("status") != "committed":
            raise PoolClientError("result publication is not committed")
        try:
            publication_id = str(UUID(receipt["publication_id"]))
            digest = receipt["digest_sha256"]
            size = receipt["size_bytes"]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
                or type(size) is not int
                or not 0 < size <= 1024 * 1024
            ):
                raise ValueError("invalid result declaration")
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise PoolClientError("invalid committed result receipt") from error
        chunks = bytearray()
        path = f"/artifacts/v1/publications/{publication_id}/assets/result.json"
        try:
            async with asyncio.timeout(60):
                async with self._client.stream(
                    "GET", path, headers={"Authorization": f"Bearer {self._api_key}"},
                    follow_redirects=False,
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                        if len(chunks) + len(chunk) > size:
                            raise PoolClientError("result bytes exceed committed size")
                        chunks.extend(chunk)
        except (httpx.HTTPError, TimeoutError) as error:
            raise PoolClientError("committed result read failed") from error
        if len(chunks) != size or hashlib.sha256(chunks).hexdigest() != digest:
            raise PoolClientError("committed result integrity check failed")
        try:
            result = json.loads(chunks)
            json.dumps(result, allow_nan=False)
        except (ValueError, TypeError) as error:
            raise PoolClientError("committed result is not finite JSON") from error
        if (
            not isinstance(result, dict)
            or result.get("schema_version") != "synth.eval-result.v1"
            or result.get("rollout_id") != rollout_id
            or result.get("pool_id") != rollout.get("pool_id")
            or publication_id != str(uuid5(
                NAMESPACE_URL, f"rhodes-result:{result.get('org_id')}:{rollout_id}:{digest}"
            ))
        ):
            raise PoolClientError("committed result owner or schema mismatch")
        return result

    async def usage(self, rollout_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/rollouts/{rollout_id}/usage", optional=True)

    async def events(
        self, rollout_id: str, *, after_sequence: int = 0, limit: int = 200
    ) -> dict[str, Any]:
        """Read one authorized committed page; missing history is an error."""
        if type(after_sequence) is not int or after_sequence < 0:
            raise ValueError("after_sequence must be a nonnegative integer")
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        page = await self._request(
            "GET",
            f"/rollouts/{rollout_id}/events",
            params={"format": "json", "after_sequence": after_sequence, "limit": limit},
        )
        rows = page.get("events")
        if not isinstance(rows, list) or len(rows) > limit:
            raise PoolClientError("invalid rollout event page")
        cursor = after_sequence
        for event in rows:
            if not isinstance(event, dict) or event.get("rollout_id") != rollout_id:
                raise PoolClientError("event rollout identity mismatch")
            sequence = event.get("sequence")
            if type(sequence) is not int or sequence != cursor + 1:
                raise PoolClientError("rollout event sequence gap; reconcile history")
            cursor = sequence
        if type(page.get("next_sequence")) is not int or page["next_sequence"] != cursor:
            raise PoolClientError("invalid rollout event cursor")
        if type(page.get("has_more")) is not bool or (page["has_more"] and not rows):
            raise PoolClientError("invalid rollout event continuation")
        if page.get("rollout_id") != rollout_id or not isinstance(page.get("status"), str):
            raise PoolClientError("invalid rollout event page identity/status")
        for field in ("cleanup_pending", "publication_pending"):
            if field in page and type(page[field]) is not bool:
                raise PoolClientError(f"invalid rollout {field}")
        return page

    async def watch_events(
        self,
        rollout_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 200,
        timeout_seconds: float = 300.0,
        poll_interval_seconds: float = 1.0,
    ) -> AsyncIterator[dict[str, Any]]:
        """Replay then poll with bounded memory; disconnect never cancels work.

        Persist the yielded sequence after applying an event. Reattach using that
        cursor. Terminal execution status ends observation only after its backlog
        drains and any declared cleanup obligation clears. Older servers without
        cleanup_pending retain scientific-terminal behavior. This does not claim
        artifact custody or settled cost.
        """
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if not math.isfinite(poll_interval_seconds) or poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be finite and positive")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        cursor = after_sequence
        while loop.time() < deadline:
            page = await self.events(rollout_id, after_sequence=cursor, limit=limit)
            for event in page["events"]:
                yield event
                cursor = event["sequence"]
            if page["has_more"]:
                continue
            if (page["status"] in TERMINAL_STATUSES
                and not page.get("cleanup_pending", False)
                and not page.get("publication_pending", False)):
                return
            await asyncio.sleep(min(poll_interval_seconds, max(0, deadline - loop.time())))
        raise PoolRolloutTimeout(
            f"event observation timed out for {rollout_id}; resume after {cursor}"
        )

    async def cancel(self, rollout_id: str) -> dict[str, Any]:
        return await self._request("POST", f"/rollouts/{rollout_id}/cancel")

    # -- collection --------------------------------------------------------

    async def wait_for(
        self,
        rollout_id: str,
        *,
        timeout_seconds: float = DEFAULT_POLL_TIMEOUT_SECONDS,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> RolloutOutcome:
        """Poll until the rollout is terminal, or raise `PoolRolloutTimeout`.

        Timing out does not cancel the rollout — it is still consuming the org's
        concurrency. Callers who want it stopped must call `cancel` themselves;
        silently cancelling here would discard work a caller may still want.
        """

        deadline = asyncio.get_running_loop().time() + max(0.0, float(timeout_seconds))
        interval = max(0.1, float(poll_interval_seconds))
        while True:
            body = await self.get_rollout(rollout_id)
            status = str(body.get("status") or "").strip().lower()
            if status in TERMINAL_STATUSES:
                return RolloutOutcome(rollout_id=rollout_id, status=status, payload=body)
            if asyncio.get_running_loop().time() >= deadline:
                raise PoolRolloutTimeout(
                    f"rollout {rollout_id} still {status or 'unknown'} "
                    f"after {timeout_seconds}s (not cancelled)"
                )
            await asyncio.sleep(interval)

    async def run_many(
        self,
        pool_id: str,
        payloads: Iterable[Mapping[str, Any]],
        *,
        max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
        timeout_seconds: float = DEFAULT_POLL_TIMEOUT_SECONDS,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> list[RolloutOutcome | PoolClientError]:
        """Run many rollouts against one pool, bounded, and collect the results.

        Submission is bounded by ``max_in_flight`` rather than fired all at
        once: an org has a finite concurrency allowance, and flooding it turns
        a normal eval into a wall of admission refusals.

        Failures are returned in place rather than raised, so one bad seed does
        not discard the rest of a sweep. Results are in submission order.
        """

        items = [dict(payload) for payload in payloads]
        if not items:
            return []
        semaphore = asyncio.Semaphore(max(1, int(max_in_flight)))

        async def _one(payload: Mapping[str, Any]) -> RolloutOutcome | PoolClientError:
            async with semaphore:
                try:
                    rollout_id = await self.submit(pool_id, payload)
                    return await self.wait_for(
                        rollout_id,
                        timeout_seconds=timeout_seconds,
                        poll_interval_seconds=poll_interval_seconds,
                    )
                except PoolClientError as error:
                    return error

        return list(await asyncio.gather(*(_one(item) for item in items)))


#: Never worth uploading, and `target/` alone is ~162 MB on the Craftax tree.
DEFAULT_BUILD_CONTEXT_EXCLUDES: tuple[str, ...] = (
    ".git",
    ".git/*",
    "target",
    "target/*",
    "node_modules",
    "node_modules/*",
    "__pycache__",
    "*/__pycache__/*",
    "*.pyc",
    ".venv",
    ".venv/*",
    ".DS_Store",
)


def pack_build_context(
    root: str | Path,
    *,
    exclude: Sequence[str] = DEFAULT_BUILD_CONTEXT_EXCLUDES,
    extra_exclude: Sequence[str] = (),
) -> bytes:
    """Pack a directory into a deterministic gzipped tar.

    Deterministic on purpose: identical inputs must produce identical bytes, so
    the content hash is a real identity. Entries are sorted, and mtime, uid,
    gid, uname, and gname are zeroed — otherwise a rebuild on another machine,
    or five minutes later, yields a different hash for the same source and the
    upload stops being idempotent.

    The executable bit is preserved, because build contexts contain scripts.
    """

    base = Path(root).resolve()
    if not base.is_dir():
        raise PoolClientError(f"build context is not a directory: {base}")
    patterns = tuple(exclude) + tuple(extra_exclude)

    paths: list[tuple[str, Path]] = []
    for path in base.rglob("*"):
        if not path.is_file() and not path.is_dir():
            continue  # skip sockets, fifos, dangling symlinks
        relative = path.relative_to(base).as_posix()
        if _is_excluded(relative, patterns):
            continue
        paths.append((relative, path))
    paths.sort(key=lambda item: item[0])

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", compresslevel=6) as tar:
        for relative, path in paths:
            info = tar.gettarinfo(str(path), arcname=relative)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o755 if (info.isdir() or info.mode & 0o111) else 0o644
            if info.isfile():
                with path.open("rb") as handle:
                    tar.addfile(info, handle)
            else:
                tar.addfile(info)
    return _strip_gzip_mtime(buffer.getvalue())


def _strip_gzip_mtime(payload: bytes) -> bytes:
    """Zero the gzip header's mtime field (bytes 4-8, RFC 1952)."""

    if len(payload) < 8 or payload[:2] != b"\x1f\x8b":
        return payload
    return payload[:4] + b"\x00\x00\x00\x00" + payload[8:]


def _is_excluded(relative: str, patterns: Sequence[str]) -> bool:
    for pattern in patterns:
        if fnmatch.fnmatch(relative, pattern):
            return True
        # A directory match must exclude everything beneath it.
        if relative.startswith(pattern.rstrip("/*") + "/"):
            return True
    return False


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_float(value: Any, *, field_name: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise HarborPoolSupportError(f"{field_name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise HarborPoolSupportError(f"{field_name} must be numeric") from error
    if number < 0:
        raise HarborPoolSupportError(f"{field_name} must be non-negative")
    return number


def _optional_int(value: Any, *, field_name: str) -> int | None:
    number = _optional_float(value, field_name=field_name)
    if number is None:
        return None
    if int(number) != number:
        raise HarborPoolSupportError(f"{field_name} must be an integer")
    return int(number)


def _optional_bool(value: Any, *, field_name: str) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    raise HarborPoolSupportError(f"{field_name} must be a boolean")


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise HarborPoolSupportError("environment.gpu_types must be a list")
    return tuple(text for item in value if (text := str(item).strip()))


def _archive_bytes(archive: bytes | str | Path) -> bytes:
    if isinstance(archive, bytes):
        return archive
    path = Path(archive)
    if not path.is_file():
        raise PoolClientError(f"archive is not a readable file: {path}")
    return path.read_bytes()


def _rollout_id(body: Mapping[str, Any]) -> str:
    for key in ("rollout_id", "id"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    rollout = body.get("rollout")
    if isinstance(rollout, Mapping):
        return _rollout_id(rollout)
    return ""


__all__ = [
    "DEFAULT_BACKEND_URL",
    "DEFAULT_BUILD_CONTEXT_EXCLUDES",
    "DEFAULT_MAX_IN_FLIGHT",
    "pack_build_context",
    "DEFAULT_POLL_INTERVAL_SECONDS",
    "DEFAULT_POLL_TIMEOUT_SECONDS",
    "HARBOR_CONTAINER_SUBTYPE",
    "HARBOR_TASK_CONFIG",
    "HarborPoolSupportError",
    "HarborTaskBundle",
    "IMPLEMENTED_POOL_STATES",
    "POOL_STATE_TRANSITIONS",
    "PROVISIONING_POOL_STATES",
    "PoolClient",
    "PoolClientError",
    "PoolState",
    "PoolStateError",
    "PoolRolloutTimeout",
    "RolloutOutcome",
    "SUCCESS_STATUSES",
    "TERMINAL_POOL_STATES",
    "TERMINAL_STATUSES",
    "pool_state_admits_rollouts",
    "pool_state_is_provisioning",
    "pool_state_is_terminal",
    "validate_pool_transition",
]
