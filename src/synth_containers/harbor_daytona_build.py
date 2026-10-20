"""Bounded private Daytona snapshot preparation with durable local custody.

This is image preparation, separate from a sandbox rollout. Resulting sandbox
CPU/memory settings do not assert limits on Daytona's internal image builder.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from .harbor_image_context import HarborImageContext
from .operator_journal import OperatorJournal


@dataclass(frozen=True)
class SnapshotBuildLimits:
    context_seconds: int = 120
    create_seconds: int = 30
    build_seconds: int = 600
    cleanup_seconds: int = 60
    retention_seconds: int = 86400

    def __post_init__(self):
        for field, ceiling in (
            ("context_seconds", 300),
            ("create_seconds", 60),
            ("build_seconds", 1800),
            ("cleanup_seconds", 120),
            ("retention_seconds", 604800),
        ):
            value = getattr(self, field)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"{field} must be an integer from 1 through {ceiling}")


class SnapshotBuildError(RuntimeError):
    """A build is not ready; provider details remain outside operator events."""


class DaytonaSnapshotBuild:
    """One create per intent; failed/ambiguous builds retain recovery identity.

    The provider seam returns plain snapshots with id/name/state attributes and
    raises its typed not-found error through is_not_found(). No retry creates are
    permitted. No published registry identity is inferred from snapshot names.
    """

    def __init__(
        self,
        context: HarborImageContext,
        output: Path,
        provider: Any,
        *,
        limits: SnapshotBuildLimits | None = None,
        architecture: str = "amd64",
        required_capabilities: tuple[str, ...] = (),
    ):
        if architecture not in {"amd64", "arm64"}:
            raise ValueError("Snapshot architecture must be amd64 or arm64")
        available = frozenset(getattr(provider, "build_capabilities", ()))
        missing = set(required_capabilities) - available
        if missing:
            raise SnapshotBuildError(
                "Provider cannot enforce build capabilities: " + ", ".join(sorted(missing))
            )
        supported_architectures = getattr(provider, "build_architectures", ("amd64",))
        if architecture not in supported_architectures:
            raise SnapshotBuildError("Provider cannot bind requested build architecture")
        self.architecture = architecture
        self.required_capabilities = required_capabilities
        self.expires_at = datetime.now(UTC) + timedelta(
            seconds=(limits or SnapshotBuildLimits()).retention_seconds
        )
        self.context = context
        self.output = output
        self.provider = provider
        self.limits = limits or SnapshotBuildLimits()
        self.owner = "synth-eval-build-" + uuid4().hex
        self.identifier: str | None = None
        self.create_attempted = False
        self._lock_handle = None
        self.journal = OperatorJournal(
            output / "image-events.jsonl", run_id=self.owner, max_record_bytes=32768
        )

    def event(self, kind: str, **fields):
        return self.journal.append(
            lambda seq: {
                "schema_version": "synth.image-build-event.v1",
                "run_id": self.owner,
                "seq": seq,
                "event": kind,
                "provider": "daytona",
                "occurred_at": datetime.now(UTC).isoformat(),
                **fields,
            }
        )

    def _claim(self):
        self.context.verify()
        if self.output.expanduser().resolve().is_relative_to(self.context.root.resolve()):
            raise ValueError("Build custody cannot be stored inside its build context")
        self.output.mkdir(parents=True, exist_ok=False)
        with (self.output / "build-claim.json").open("x") as handle:
            json.dump(
                {
                    "owner": self.owner,
                    "context": self.context.as_dict(),
                    "limits": vars(self.limits),
                    "architecture": self.architecture,
                    "required_capabilities": list(self.required_capabilities),
                    "expires_at": self.expires_at.isoformat(),
                    "created_at": datetime.now(UTC).isoformat(),
                },
                handle,
                sort_keys=True,
            )
            handle.flush()
            os.fsync(handle.fileno())
        descriptor = os.open(self.output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.event(
            "image.build_requested",
            context=self.context.as_dict(),
            limits=vars(self.limits),
            builder_resource_limits_supported=False,
            provider_storage_expiry_supported=False,
        )

    def _remember(self, snapshot):
        identifier = getattr(snapshot, "id", None)
        if (
            getattr(snapshot, "name", None) != self.owner
            or getattr(snapshot, "general", None) is not False
            or not isinstance(identifier, str)
            or not 1 <= len(identifier) <= 255
        ):
            raise SnapshotBuildError("Snapshot ownership could not be established")
        if self.identifier is not None and self.identifier != identifier:
            raise SnapshotBuildError("Snapshot identity changed")
        self.identifier = identifier
        self.event("image.identity_observed", snapshot_id=identifier)

    async def build(self):
        self._claim()
        self._lock_handle = (self.output / "build-owner.lock").open("a+")
        fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            async with asyncio.timeout(self.limits.context_seconds):
                prepared = await self.provider.prepare_context(self.context)
            self.context.verify()
            self.event("image.context_prepared")
            self.create_attempted = True
            self.event("image.create_requested", snapshot_name=self.owner)
            async with asyncio.timeout(self.limits.create_seconds):
                snapshot = await self.provider.create(self.owner, prepared)
            self._remember(snapshot)
            async with asyncio.timeout(self.limits.build_seconds):
                previous = None
                while True:
                    self._remember_identity(snapshot)
                    observed_state = getattr(snapshot, "state", "")
                    state = str(getattr(observed_state, "value", observed_state)).lower()
                    if state != previous:
                        self.event("image.state_observed", snapshot_id=self.identifier, state=state)
                        previous = state
                    if state == "active":
                        artifact = self._artifact_receipt(snapshot)
                        with (self.output / "build-artifact.json").open("x") as handle:
                            json.dump(artifact, handle, sort_keys=True, allow_nan=False)
                            handle.flush()
                            os.fsync(handle.fileno())
                        self.event(
                            "image.build_completed",
                            snapshot_id=self.identifier,
                            artifact=self._artifact_receipt(snapshot),
                        )
                        return snapshot
                    if state in {"error", "build_failed", "build failed", "removing"}:
                        raise SnapshotBuildError("Snapshot build did not complete")
                    await asyncio.sleep(1)
                    snapshot = await self.provider.get(self.identifier)
        except BaseException as error:
            self.event(
                "image.build_unconfirmed",
                snapshot_id=self.identifier,
                error_type=type(error).__name__,
            )
            if self.create_attempted:
                outcomes = await asyncio.shield(
                    asyncio.gather(self.cleanup(), return_exceptions=True)
                )
                if isinstance(outcomes[0], BaseException):
                    # Cleanup already committed a pending event. Preserve the
                    # original error while retaining that independent outcome.
                    error.add_note(
                        "Snapshot cleanup remains pending: " + type(outcomes[0]).__name__
                    )
            raise
        finally:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            self._lock_handle.close()
            self._lock_handle = None

    def _remember_identity(self, snapshot):
        if (
            getattr(snapshot, "id", None) != self.identifier
            or getattr(snapshot, "name", None) != self.owner
            or getattr(snapshot, "general", None) is not False
        ):
            raise SnapshotBuildError("Snapshot identity changed while building")

    def _artifact_receipt(self, snapshot) -> dict:
        # Provider snapshot identity is not a digest of the image bytes. Do not
        # retain arbitrary registry references (which may contain credentials)
        # or promote a tagged provider ref through the pinned-image launch gate.
        self._remember_identity(snapshot)
        return {
            "schema_version": "synth.daytona-build-artifact.v1",
            "provider": "daytona",
            "snapshot_id": self.identifier,
            "snapshot_name": self.owner,
            "private_snapshot_confirmed": True,
            "source_package_digest": self.context.source_package_digest,
            "context_digest": self.context.context_digest,
            "image_digest": None,
            "image_digest_verified": False,
            "native_pinned_image_launch_eligible": False,
            "architecture": self.architecture,
            "architecture_enforcement": "provider_admission",
            "retention": "owner_reconciled_expiry",
            "expires_at": self.expires_at.isoformat(),
            "provider_storage_expiry_supported": False,
            "context_storage_release_confirmed": False,
        }

    async def cleanup(self) -> dict:
        self.event("image.cleanup_requested", snapshot_id=self.identifier)
        try:
            async with asyncio.timeout(self.limits.cleanup_seconds):
                if self.identifier is None:
                    try:
                        snapshot = await self.provider.get(self.owner)
                    except Exception as error:
                        if self.provider.is_not_found(error):
                            raise SnapshotBuildError(
                                "Ambiguous create has no observed snapshot identity"
                            ) from error
                        raise
                    self._remember(snapshot)
                try:
                    snapshot = await self.provider.get(self.identifier)
                except Exception as error:
                    if not self.provider.is_not_found(error):
                        raise
                else:
                    self._remember_identity(snapshot)
                    await self.provider.delete(self.identifier)
                while True:
                    try:
                        await self.provider.get(self.identifier)
                    except Exception as error:
                        if self.provider.is_not_found(error):
                            break
                        raise
                    await asyncio.sleep(1)
                try:
                    await self.provider.get(self.owner)
                except Exception as error:
                    if not self.provider.is_not_found(error):
                        raise
                else:
                    raise SnapshotBuildError("Snapshot name still resolves after deletion")
        except BaseException as error:
            self.event(
                "image.cleanup_pending",
                snapshot_id=self.identifier,
                error_type=type(error).__name__,
            )
            raise
        return self.event("image.cleanup_confirmed", snapshot_id=self.identifier)


async def recover_daytona_snapshot_build(
    output: Path, provider: Any, *, now: datetime | None = None
):
    """Reconcile expired failed custody; never take a live builder's lock.

    Completed snapshots remain artifacts until their owner requests cleanup.
    Recovery does not convert that retention policy into an implicit deletion.
    """
    import re
    from datetime import timedelta

    from .operator_journal import read_operator_events

    with (output / "build-claim.json").open("rb") as handle:
        encoded = handle.read(65537)
    if len(encoded) > 65536:
        raise SnapshotBuildError("Snapshot build claim exceeds bound")
    claim = json.loads(encoded)
    owner = claim.get("owner")
    if not isinstance(owner, str) or not re.fullmatch(r"synth-eval-build-[0-9a-f]{32}", owner):
        raise SnapshotBuildError("Snapshot build owner is invalid")
    limits = SnapshotBuildLimits(**claim["limits"])
    created_at = datetime.fromisoformat(claim["created_at"])
    observed_at = now or datetime.now(UTC)
    if created_at.tzinfo is None or observed_at.tzinfo is None:
        raise SnapshotBuildError("Snapshot recovery requires timezone-aware clocks")
    expires = created_at + timedelta(
        seconds=limits.context_seconds + limits.create_seconds + limits.build_seconds + 60
    )
    if observed_at < expires:
        raise SnapshotBuildError("Snapshot build custody has not expired")
    with (output / "build-owner.lock").open("r+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            page = read_operator_events(output / "image-events.jsonl", run_id=owner, limit=1000)
            if page["has_more"]:
                raise SnapshotBuildError("Snapshot build history exceeds recovery bound")
            rows = page["events"]
            if sum(row.get("event") == "image.create_requested" for row in rows) != 1:
                raise SnapshotBuildError("Snapshot recovery requires exactly one create intent")
            kinds = {row.get("event") for row in rows}
            if "image.build_completed" in kinds and "image.cleanup_requested" not in kinds:
                retained_until = claim.get("expires_at")
                if retained_until is None or observed_at < datetime.fromisoformat(retained_until):
                    raise SnapshotBuildError("Completed snapshot has not expired or been released")
            identifiers = [row["snapshot_id"] for row in rows if row.get("snapshot_id")]
            if any(
                not isinstance(value, str) or not 1 <= len(value) <= 255 for value in identifiers
            ):
                raise SnapshotBuildError("Snapshot recovery identity is invalid")
            known = set(identifiers)
            if len(known) > 1:
                raise SnapshotBuildError("Snapshot build has conflicting identities")
            context = claim["context"]
            operation = DaytonaSnapshotBuild(
                HarborImageContext(
                    Path(context["root"]),
                    context["source_package_digest"],
                    context["context_digest"],
                ),
                output,
                provider,
                limits=limits,
                architecture=claim.get("architecture", "amd64"),
                required_capabilities=tuple(claim.get("required_capabilities", ())),
            )
            if claim.get("expires_at"):
                operation.expires_at = datetime.fromisoformat(claim["expires_at"])
            operation.owner = owner
            operation.identifier = next(iter(known), None)
            operation.create_attempted = True
            operation.journal = OperatorJournal(
                output / "image-events.jsonl", run_id=owner, max_record_bytes=32768
            )
            return await operation.cleanup()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
