"""Bounded native Harbor Daytona environment extension.

See docs/native-harbor-daytona.md. This module is an optional, version-qualified
Harbor integration; importing the ordinary containers package does not load it.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from daytona.common.errors import DaytonaNotFoundError
from harbor.environments.daytona.environment import DaytonaEnvironment

from .harbor_resource_receipts import HarborResourceCleanupPending
from .harbor_results import finite_number
from .operator_journal import OperatorJournal


class BoundedDaytonaEnvironment(DaytonaEnvironment):
    """One pinned-image sandbox per trial, with finite provider lifetime."""

    def __init__(
        self,
        *args,
        resource_ttl_minutes: int = 5,
        maximum_cpu: int = 1,
        maximum_memory_gib: int = 1,
        maximum_disk_gib: int = 1,
        prepared_snapshot_artifact: str | None = None,
        expected_source_package_digest: str | None = None,
        expected_architecture: str = "amd64",
        **kwargs,
    ):
        if version("harbor") != "0.22.0" or version("daytona") != "0.210.0":
            raise ValueError("Native Daytona requires qualified Harbor 0.22.0 and Daytona 0.210.0")
        for name, value, ceiling in (
            ("resource_ttl_minutes", resource_ttl_minutes, 360),
            ("maximum_cpu", maximum_cpu, 64),
            ("maximum_memory_gib", maximum_memory_gib, 512),
            ("maximum_disk_gib", maximum_disk_gib, 1024),
        ):
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"{name} must be an integer between 1 and {ceiling}")
        if (
            kwargs.get("auto_snapshot")
            or kwargs.get("snapshot_template_name")
            or kwargs.get("extra_docker_compose")
        ):
            raise ValueError(
                "Bounded native Daytona requires a prebuilt image, without snapshots or Compose"
            )
        super().__init__(*args, **kwargs)
        if self._auto_snapshot or self._snapshot_template_name:
            raise ValueError("Bounded native Daytona cannot create or select snapshots")
        self._prepared_snapshot = None
        self._prepared_resource_shape = None
        if prepared_snapshot_artifact is not None:
            raw = Path(prepared_snapshot_artifact).read_bytes()
            if len(raw) > 16384:
                raise ValueError("Snapshot artifact exceeds bound")
            artifact = json.loads(raw)
            if (
                artifact.get("schema_version") != "synth.daytona-build-artifact.v1"
                or artifact.get("provider") != "daytona"
                or artifact.get("private_snapshot_confirmed") is not True
                or not expected_source_package_digest
                or artifact.get("source_package_digest") != expected_source_package_digest
                or artifact.get("architecture") != expected_architecture
                or not re.fullmatch(
                    r"synth-eval-build-[0-9a-f]{32}", artifact.get("snapshot_name", "")
                )
                or not isinstance(artifact.get("snapshot_id"), str)
                or not 1 <= len(artifact["snapshot_id"]) <= 255
            ):
                raise ValueError("Prepared snapshot binding mismatch")
            expiry = datetime.fromisoformat(artifact["expires_at"])
            if expiry.tzinfo is None or expiry <= datetime.now(UTC):
                raise ValueError("Prepared snapshot retention has expired")
            self._prepared_snapshot = artifact
        image = str(self.task_env_config.docker_image or "")
        if (
            self._prepared_snapshot is None
            and not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image)
        ) or self._compose_mode:
            raise ValueError("Bounded native Daytona requires a digest-pinned direct image")
        if self._dockerfile_path.exists():
            raise ValueError("Bounded native Daytona does not build task Dockerfiles")
        self._resource_ttl_minutes = resource_ttl_minutes
        self._resource_maxima = (maximum_cpu, maximum_memory_gib, maximum_disk_gib)
        self._resource_owner = "synth-harbor-" + uuid4().hex
        self._resource_create_attempted = False
        self._resource_root = Path(self.trial_paths.trial_dir)
        self._resource_journal = OperatorJournal(
            self._resource_root / "resource-events.jsonl",
            run_id=self._resource_owner,
            max_record_bytes=16384,
        )

    def _resource_event(self, event: str, **fields):
        return self._resource_journal.append(
            lambda sequence: {
                "schema_version": "synth.harbor-resource-event.v1",
                "run_id": self._resource_owner,
                "seq": sequence,
                "event": event,
                "occurred_at": datetime.now(UTC).isoformat(),
                **fields,
            }
        )

    async def start(self, force_build: bool):
        if force_build:
            raise ValueError("Bounded native Daytona cannot force image builds")
        return await super().start(force_build=False)

    async def _resolve_start_sandbox_params(self, daytona, resources, *, force_build):
        if self._prepared_snapshot is None:
            return await super()._resolve_start_sandbox_params(
                daytona, resources, force_build=force_build
            )
        if force_build or resources is None:
            raise ValueError("Prepared snapshot requires admitted resources and no build")
        artifact = self._prepared_snapshot
        if datetime.fromisoformat(artifact["expires_at"]) <= datetime.now(UTC):
            raise ValueError("Prepared snapshot expired before launch")
        async with asyncio.timeout(15):
            snapshot = await daytona.snapshot.get(artifact["snapshot_id"])
        state = getattr(snapshot, "state", "")
        if (
            getattr(snapshot, "id", None) != artifact["snapshot_id"]
            or getattr(snapshot, "name", None) != artifact["snapshot_name"]
            or getattr(snapshot, "general", None) is not False
            or str(getattr(state, "value", state)).lower() != "active"
        ):
            raise ValueError("Provider snapshot identity/state changed")
        requested = (resources.cpu, resources.memory, resources.disk)
        observed = tuple(getattr(snapshot, key, None) for key in ("cpu", "mem", "disk"))
        if observed != requested:
            raise ValueError("Prepared snapshot resource shape differs from admitted request")
        self._prepared_resource_shape = resources
        self._resource_event(
            "resource.snapshot_bound",
            snapshot_id=artifact["snapshot_id"],
            source_package_digest=artifact["source_package_digest"],
            context_digest=artifact["context_digest"],
            architecture=artifact["architecture"],
            image_digest_verified=False,
        )
        # The provider identity is checked afresh; it is never cast to an OCI digest.
        return self._snapshot_sandbox_params(artifact["snapshot_id"])

    async def _create_sandbox(self, params, daytona=None):
        if self._resource_create_attempted:
            raise RuntimeError("Native Daytona creation allowance exhausted")
        resources = getattr(params, "resources", None) or self._prepared_resource_shape
        if resources is None:
            raise ValueError("Native Daytona requires explicit CPU, memory and disk requests")
        requested = (resources.cpu, resources.memory, resources.disk)
        if (
            any(
                type(value) is not int or not 1 <= value <= maximum
                for value, maximum in zip(requested, self._resource_maxima)
            )
            or resources.gpu
        ):
            raise ValueError("Native Daytona resource request exceeds admitted bounds")
        timeout = self.task_env_config.build_timeout_sec
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not 0 < timeout <= 300
        ):
            raise ValueError(
                "Native Daytona creation timeout must be positive and at most 300 seconds"
            )
        self._resource_create_attempted = True
        self._resource_root.mkdir(parents=True, exist_ok=True)
        # The exclusive claim survives worker restart. A new process cannot
        # repeat an ambiguous create against the same trial directory.
        with (self._resource_root / "resource-create-claim.json").open("x") as claim:
            json.dump({"owner": self._resource_owner}, claim)
            claim.flush()
            os.fsync(claim.fileno())
        params.ttl_minutes = self._resource_ttl_minutes
        params.auto_stop_interval = min(5, self._resource_ttl_minutes)
        params.auto_delete_interval = 0
        self._user_labels["ai.synth.harbor.owner"] = self._resource_owner
        self._resource_event(
            "resource.create_requested",
            provider="daytona",
            ttl_minutes=self._resource_ttl_minutes,
            creation_timeout_seconds=timeout,
            cpu=requested[0],
            memory_gib=requested[1],
            disk_gib=requested[2],
        )
        try:
            # Retain Harbor's cancellation shield and handle capture, but not
            # its provider-create retry decorator. Version checks pin this seam.
            await DaytonaEnvironment._create_sandbox.__wrapped__(self, params, daytona)
        except BaseException as error:
            self._resource_event(
                "resource.create_unconfirmed",
                error_type=type(error).__name__,
                provider_id=self._sandbox.id if self._sandbox else None,
            )
            raise
        self._resource_event("resource.created", provider_id=self._sandbox.id)
        observed = tuple(
            finite_number(getattr(self._sandbox, field, None), field=field)
            for field in ("cpu", "memory", "disk")
        )
        deadline_text = getattr(self._sandbox, "auto_destroy_at", None)
        self._resource_event(
            "resource.allocation_observed",
            provider_id=self._sandbox.id,
            cpu=observed[0],
            memory_gib=observed[1],
            disk_gib=observed[2],
            auto_destroy_at=deadline_text,
        )
        if any(not 0 < value <= maximum for value, maximum in zip(observed, self._resource_maxima)):
            raise RuntimeError("Native Daytona observed allocation exceeds admitted bounds")
        if not isinstance(deadline_text, str):
            raise TypeError("Native Daytona provider did not confirm resource TTL")
        deadline = datetime.fromisoformat(deadline_text)
        if (
            deadline.tzinfo is None
            or not 0
            < (deadline - datetime.now(UTC)).total_seconds()
            <= self._resource_ttl_minutes * 60 + 5
        ):
            raise RuntimeError("Native Daytona observed resource TTL exceeds admitted bounds")

    async def stop(self, delete: bool):
        if not delete:
            raise ValueError("Bounded native Daytona requires sandbox deletion")
        if self._sandbox is None:
            return
        identifier = self._sandbox.id
        self._resource_event("resource.cleanup_requested", provider_id=identifier)
        try:
            async with asyncio.timeout(45):
                client = await self._client_manager.get_client()
                try:
                    await client.delete(self._sandbox, timeout=20, wait=True)
                except DaytonaNotFoundError:
                    pass
                try:
                    await client.get(identifier, request_timeout=10)
                except DaytonaNotFoundError:
                    pass
                else:
                    raise RuntimeError("Native Daytona deletion absence unconfirmed")
        except BaseException as error:
            self._resource_event(
                "resource.cleanup_pending", provider_id=identifier, error_type=type(error).__name__
            )
            if isinstance(error, Exception):
                raise HarborResourceCleanupPending(
                    "Native Daytona cleanup remains pending"
                ) from error
            raise
        self._resource_event("resource.cleanup_confirmed", provider_id=identifier)
        self._sandbox = None
        self._client_manager = None
