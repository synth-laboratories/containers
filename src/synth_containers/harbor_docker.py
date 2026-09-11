"""Native Harbor Docker resource custody; optional and pinned to Harbor 0.22."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import closing
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import docker
from docker.errors import NotFound
from harbor.environments.docker.docker import DockerEnvironment

from .harbor_environment import is_pinned_harbor_image
from .harbor_resource_receipts import HarborResourceCleanupPending
from .operator_journal import OperatorJournal


class ObservedDockerEnvironment(DockerEnvironment):
    """Retain creation authority and require fresh absence after Harbor cleanup.

    Docker supplies CPU/memory controls, not a provider TTL or workspace quota.
    This extension does not claim those unsupported guarantees.
    """

    def __init__(self, *args, egress_control_image: str | None = None, **kwargs):
        if version("harbor") != "0.22.0":
            raise ValueError("Native Docker requires qualified Harbor 0.22.0")
        if args:
            raise ValueError("Native Docker requires named environment arguments")
        config = kwargs["task_env_config"]
        root = Path(kwargs["environment_dir"])
        timeout = config.build_timeout_sec
        if type(timeout) not in (int, float) or not 0 < timeout <= 300:
            raise ValueError(
                "Native Docker creation timeout must be positive and at most 300 seconds"
            )
        self._creation_timeout_seconds = timeout
        if egress_control_image is not None and not is_pinned_harbor_image(
            egress_control_image, "docker"
        ):
            raise ValueError("Native Docker egress image must be immutable")
        self._qualified_egress_image = egress_control_image
        if not is_pinned_harbor_image(config.docker_image, "docker"):
            raise ValueError("Native Docker requires an immutable image reference")
        if (root / "Dockerfile").exists() or any(root.glob("*compose*y*ml")):
            raise ValueError("Native Docker requires a prebuilt single-container task")
        if kwargs.get("keep_containers") or kwargs.get("extra_docker_compose"):
            raise ValueError("Native Docker requires deletion and no extra Compose services")
        self._resource_owner = "synth-harbor-" + uuid4().hex
        kwargs["session_id"] = self._resource_owner
        super().__init__(**kwargs)
        if self._uses_compose or self._is_windows_container:
            raise ValueError("Native Docker qualifies Linux single-container tasks only")
        if self._enable_egress_control and not self._qualified_egress_image:
            raise ValueError(
                "Native Docker egress control requires an explicitly prepared immutable sidecar image"
            )
        self._resource_root = Path(self.trial_paths.trial_dir)
        self._resource_journal = OperatorJournal(
            self._resource_root / "resource-events.jsonl",
            run_id=self._resource_owner,
            max_record_bytes=16384,
        )
        self._resource_create_attempted = False
        self._resource_handles: list[dict[str, str]] = []

    async def _ensure_egress_control_sidecar_image_built(self):
        # Image preparation belongs to the manager/build lane, not trial startup.
        if not self._qualified_egress_image:
            raise ValueError("Native Docker egress image was not prepared")

        def inspect_image():
            with closing(docker.from_env(timeout=5)) as client:
                client.images.get(self._qualified_egress_image)

        async with asyncio.timeout(10):
            await asyncio.to_thread(inspect_image)
        self._env_vars.egress_control_sidecar_image_name = self._qualified_egress_image

    def _resource_event(self, event: str, **fields):
        return self._resource_journal.append(
            lambda sequence: {
                "schema_version": "synth.harbor-resource-event.v1",
                "run_id": self._resource_owner,
                "seq": sequence,
                "event": event,
                "occurred_at": datetime.now(UTC).isoformat(),
                "provider": "docker",
                "project": self._resource_owner,
                **fields,
            }
        )

    def _discover(self) -> list[dict[str, str]]:
        """Read only; refuse ambiguous labels rather than widening cleanup."""
        with closing(docker.from_env(timeout=5)) as client:
            filters = {"label": "com.docker.compose.project=" + self._resource_owner}
            found = []
            for kind, manager in (
                ("container", client.containers),
                ("network", client.networks),
                ("volume", client.volumes),
            ):
                items = manager.list(
                    filters=filters, **({"all": True} if kind == "container" else {})
                )
                if len(items) > 16:
                    raise RuntimeError("Native Docker resource discovery exceeds bound")
                for item in items:
                    labels = (
                        item.attrs.get("Labels")
                        if kind != "container"
                        else item.attrs.get("Config", {}).get("Labels")
                    )
                    if (labels or {}).get("com.docker.compose.project") != self._resource_owner:
                        raise RuntimeError("Native Docker resource ownership mismatch")
                    found.append({"kind": kind, "id": item.name if kind == "volume" else item.id})
            return found

    async def _remember_resources(self):
        async with asyncio.timeout(20):
            found = await asyncio.to_thread(self._discover)
        for handle in found:
            if handle not in self._resource_handles:
                self._resource_handles.append(handle)
        if len(self._resource_handles) > 48:
            raise RuntimeError("Native Docker resource handle bound exceeded")
        self._resource_event("resource.handles_observed", handles=self._resource_handles)

    async def start(self, force_build: bool):
        if force_build or self._resource_create_attempted:
            raise ValueError("Native Docker permits one prebuilt creation attempt")
        self._resource_root.mkdir(parents=True, exist_ok=True)
        with (self._resource_root / "resource-create-claim.json").open("x") as claim:
            json.dump({"owner": self._resource_owner, "provider": "docker"}, claim)
            claim.flush()
            os.fsync(claim.fileno())
        directory = os.open(self._resource_root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        self._resource_create_attempted = True
        self._resource_event(
            "resource.create_requested",
            provider_ttl_supported=False,
            workspace_quota_supported=False,
        )
        try:
            async with asyncio.timeout(self._creation_timeout_seconds):
                await super().start(force_build=False)
            await self._remember_resources()
            if not any(h["kind"] == "container" for h in self._resource_handles):
                raise RuntimeError("Native Docker primary container was not observed")
        except BaseException as error:
            self._resource_event("resource.create_unconfirmed", error_type=type(error).__name__)
            raise
        self._resource_event("resource.created", handles=self._resource_handles)

    def _confirm_absence(self):
        with closing(docker.from_env(timeout=5)) as client:
            managers = {
                "container": client.containers,
                "network": client.networks,
                "volume": client.volumes,
            }
            for handle in self._resource_handles:
                try:
                    managers[handle["kind"]].get(handle["id"])
                except NotFound:
                    continue
                raise RuntimeError("Native Docker resource deletion is unconfirmed")
        if self._discover():
            raise RuntimeError("Native Docker project still owns resources")

    async def stop(self, delete: bool):
        if not delete:
            raise ValueError("Native Docker requires resource deletion")
        if not self._resource_create_attempted:
            return
        self._resource_event("resource.cleanup_requested", handles=self._resource_handles)
        try:
            # Persist even partially created resources before asking Harbor to
            # remove them. Its best-effort stop result is never absence proof.
            await self._remember_resources()
            async with asyncio.timeout(60):
                await super().stop(delete=True)
                if not any(h["kind"] == "container" for h in self._resource_handles):
                    raise RuntimeError("Native Docker ambiguous creation has no primary handle")
                await asyncio.to_thread(self._confirm_absence)
        except BaseException as error:
            self._resource_event(
                "resource.cleanup_pending",
                error_type=type(error).__name__,
                handles=self._resource_handles,
            )
            if isinstance(error, Exception):
                raise HarborResourceCleanupPending("Native Docker cleanup remains pending") from error
            raise
        self._resource_event("resource.cleanup_confirmed", handles=self._resource_handles)
