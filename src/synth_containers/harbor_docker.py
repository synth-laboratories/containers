"""Native Harbor Docker resource custody; optional and pinned to Harbor 0.22."""

from __future__ import annotations

import asyncio
import fcntl
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
from .native_limit_admission import require_native_limit_capabilities
from .operator_journal import OperatorJournal


class ObservedDockerEnvironment(DockerEnvironment):
    """Retain creation authority and require fresh absence after Harbor cleanup.

    Optional kernel tmpfs and storage-driver quotas scope workspace/writable-layer
    bytes. They do not quota bind mounts or provide a worker-independent TTL.
    """

    def __init__(
        self,
        *args,
        egress_control_image: str | None = None,
        workspace_tmpfs_bytes: int | None = None,
        writable_layer_bytes: int | None = None,
        required_limit_capabilities: object = (),
        **kwargs,
    ):
        require_native_limit_capabilities(required_limit_capabilities)
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
        self._writable_layer_bytes = writable_layer_bytes
        if writable_layer_bytes is not None and (
            type(writable_layer_bytes) is not int or not 1024**2 <= writable_layer_bytes <= 1024**4
        ):
            raise ValueError("writable_layer_bytes must be from 1 MiB through 1 TiB")
        self._workspace_tmpfs_bytes = workspace_tmpfs_bytes
        self._quota_compose_path = None
        if workspace_tmpfs_bytes is not None and (
            type(workspace_tmpfs_bytes) is not int or not 1 <= workspace_tmpfs_bytes <= 16 * 1024**3
        ):
            raise ValueError("workspace_tmpfs_bytes must be an integer from 1 through 16 GiB")
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
        self._resource_owner_lock = None
        self._resource_handles: list[dict[str, str]] = []

    @property
    def _docker_compose_paths(self):
        paths = super()._docker_compose_paths
        if self._quota_compose_path is not None:
            paths.append(self._quota_compose_path)
        return paths

    def _arm_workspace_quota(self):
        if self._workspace_tmpfs_bytes is None and self._writable_layer_bytes is None:
            return
        # Explicit opt-in: /workspace begins empty. Image content under this path
        # is obscured, never copied using an unbounded host staging directory.
        path = self._resource_root / "workspace-quota.compose.json"
        service = {"cap_drop": ["SYS_ADMIN"], "security_opt": ["no-new-privileges:true"]}
        if self._workspace_tmpfs_bytes is not None:
            service["tmpfs"] = [
                f"/workspace:rw,nosuid,nodev,size={self._workspace_tmpfs_bytes},mode=1777"
            ]
        if self._writable_layer_bytes is not None:
            # Docker rejects unsupported storage drivers/backing filesystems.
            # No fallback to an unlimited writable layer is permitted.
            service["storage_opt"] = {"size": str(self._writable_layer_bytes)}
        with path.open("x") as handle:
            json.dump({"services": {"main": service}}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        self._quota_compose_path = path
        self._resource_event(
            "resource.workspace_quota_armed",
            path="/workspace",
            bytes=self._workspace_tmpfs_bytes,
            mechanism="docker_tmpfs",
            scope="workspace_mount_only",
            initial_contents="empty",
        )

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

    def _confirm_workspace_quota(self):
        if self._workspace_tmpfs_bytes is None and self._writable_layer_bytes is None:
            return
        with closing(docker.from_env(timeout=5)) as client:
            primary = []
            for resource in self._resource_handles:
                if resource["kind"] != "container":
                    continue
                container = client.containers.get(resource["id"])
                labels = container.attrs.get("Config", {}).get("Labels", {})
                if labels.get("com.docker.compose.service") == "main":
                    primary.append(container)
            if len(primary) != 1:
                raise RuntimeError("Workspace quota requires one observed primary container")
            options = primary[0].attrs.get("HostConfig", {}).get("Tmpfs", {}).get("/workspace", "")
            if self._writable_layer_bytes is not None:
                storage = primary[0].attrs.get("HostConfig", {}).get("StorageOpt", {})
                if storage.get("size") != str(self._writable_layer_bytes):
                    raise RuntimeError("Provider did not confirm the writable-layer storage quota")
                self._resource_event(
                    "resource.writable_layer_quota_confirmed",
                    bytes=self._writable_layer_bytes,
                    mechanism="docker_storage_opt",
                )
            if self._workspace_tmpfs_bytes is None:
                return
            if f"size={self._workspace_tmpfs_bytes}" not in options.split(","):
                raise RuntimeError("Provider did not confirm the workspace tmpfs quota")
            for mount in primary[0].attrs.get("Mounts", []):
                destination = mount.get("Destination", "")
                if (
                    destination == "/workspace" or destination.startswith("/workspace/")
                ) and mount.get("Type") != "tmpfs":
                    raise RuntimeError("A writable mount bypasses the workspace tmpfs quota")
        self._resource_event(
            "resource.workspace_quota_confirmed",
            path="/workspace",
            bytes=self._workspace_tmpfs_bytes,
            mechanism="docker_tmpfs",
        )

    async def start(self, force_build: bool):
        if force_build or self._resource_create_attempted:
            raise ValueError("Native Docker permits one prebuilt creation attempt")
        async with asyncio.timeout(10):
            daemon_id = await asyncio.to_thread(self._daemon_identity)
        self._resource_root.mkdir(parents=True, exist_ok=True)
        with (self._resource_root / "resource-create-claim.json").open("x") as claim:
            json.dump(
                {
                    "owner": self._resource_owner,
                    "provider": "docker",
                    "recovery_protocol": "owner_lock.v1",
                    "daemon_id": daemon_id,
                },
                claim,
            )
            claim.flush()
            os.fsync(claim.fileno())
        directory = os.open(self._resource_root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        self._resource_create_attempted = True
        self._arm_workspace_quota()
        self._resource_owner_lock = (self._resource_root / "resource-owner.lock").open("x+")
        fcntl.flock(self._resource_owner_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self._resource_event(
            "resource.create_requested",
            creation_timeout_seconds=self._creation_timeout_seconds,
            provider_ttl_supported=False,
            workspace_quota_supported=self._workspace_tmpfs_bytes is not None,
            workspace_quota_scope="/workspace" if self._workspace_tmpfs_bytes else None,
        )
        try:
            async with asyncio.timeout(self._creation_timeout_seconds):
                await super().start(force_build=False)
            await self._remember_resources()
            async with asyncio.timeout(15):
                await asyncio.to_thread(self._confirm_workspace_quota)
            if not any(h["kind"] == "container" for h in self._resource_handles):
                raise RuntimeError("Native Docker primary container was not observed")
        except BaseException as error:
            self._resource_event("resource.create_unconfirmed", error_type=type(error).__name__)
            raise
        self._resource_event("resource.created", handles=self._resource_handles)

    @staticmethod
    def _daemon_identity():
        with closing(docker.from_env(timeout=5)) as client:
            identifier = client.info().get("ID")
        if not isinstance(identifier, str) or not 1 <= len(identifier) <= 255:
            raise ValueError("Native Docker daemon identity was not observed")
        return identifier

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
            self._resource_event("resource.cleanup_confirmed", handles=self._resource_handles)
        except BaseException as error:
            self._resource_event(
                "resource.cleanup_pending",
                error_type=type(error).__name__,
                handles=self._resource_handles,
            )
            if isinstance(error, Exception):
                raise HarborResourceCleanupPending(
                    "Native Docker cleanup remains pending"
                ) from error
            raise
        finally:
            if self._resource_owner_lock is not None:
                fcntl.flock(self._resource_owner_lock.fileno(), fcntl.LOCK_UN)
                self._resource_owner_lock.close()
                self._resource_owner_lock = None
