"""Cold restore custody; partial effects remain fenced until durable reconciliation.

See synth-tunnel/docs/SESSION_CHECKPOINTS.md. A lost origin reply never causes
another restore call. Origins must expose durable operation lookup separately.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from hashlib import sha256
from typing import Protocol
from uuid import UUID

from pydantic import Field, model_validator

from .contracts import (
    ApplicationCheckpoint,
    ArtifactReference,
    CheckpointFailure,
    Contract,
    ParticipantIdentity,
    ParticipantRestore,
    ParticipantSnapshot,
    SessionFence,
)
from .journal import FileCheckpointJournal
from .storage import ArtifactStore


class RestoreRequest(Contract):
    operation: UUID
    connector: UUID
    session: UUID
    expected_generation: int = Field(ge=1, le=9_007_199_254_740_991)
    artifact: ArtifactReference
    deadline_at: datetime


class RestoreResult(Contract):
    operation: UUID
    checkpoint: UUID
    checkpoint_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    fence: SessionFence
    receipts: tuple[ParticipantRestore, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def consistent_receipts(self):
        if len({item.participant_name for item in self.receipts}) != len(self.receipts):
            raise ValueError("duplicate_restore_participant")
        if any(
            item.operation != self.operation
            or item.checkpoint != self.checkpoint
            or item.checkpoint_sha256 != self.checkpoint_sha256
            for item in self.receipts
        ):
            raise ValueError("inconsistent_restore_receipts")
        return self


class RestoreParticipant(Protocol):
    identity: ParticipantIdentity

    async def restore(
        self,
        request: RestoreRequest,
        source: ParticipantSnapshot,
        payload: bytes,
        fence: SessionFence,
    ) -> ParticipantRestore: ...
    async def lookup_restore(
        self, request: RestoreRequest, source: ParticipantSnapshot
    ) -> ParticipantRestore | None: ...


class RestoreAuthority(Protocol):
    async def restart(
        self, request: RestoreRequest, manifest: ApplicationCheckpoint
    ) -> SessionFence: ...
    async def lookup_restart(
        self, request: RestoreRequest, manifest: ApplicationCheckpoint
    ) -> SessionFence | None: ...
    async def complete_restore(self, request: RestoreRequest, result: RestoreResult) -> None: ...
    async def lookup_completion(self, request: RestoreRequest, result: RestoreResult) -> bool: ...


class ApplicationRestoreCoordinator:
    def __init__(
        self, *, authority: RestoreAuthority, store: ArtifactStore, journal: FileCheckpointJournal
    ):
        self.authority = authority
        self.store = store
        self.journal = journal

    async def _manifest(
        self, request: RestoreRequest, participants: tuple[RestoreParticipant, ...]
    ):
        manifest = ApplicationCheckpoint.model_validate_json(
            await asyncio.to_thread(self.store.get, request.artifact)
        )
        if (
            manifest.fence.connector != request.connector
            or manifest.fence.session != request.session
            or manifest.fence.generation > request.expected_generation
        ):
            raise CheckpointFailure("restore_manifest_lineage")
        sources = {row.cut.participant.name: row for row in manifest.participants}
        if len(participants) != len(sources) or {row.identity.name for row in participants} != set(
            sources
        ):
            raise CheckpointFailure("restore_participant_set")
        for participant in participants:
            source = sources[participant.identity.name].cut.participant
            if (
                source.service_identity != participant.identity.service_identity
                or source.origin_instance == participant.identity.origin_instance
                or source.semantics != participant.identity.semantics
            ):
                raise CheckpointFailure("restore_participant_lineage")
        digest = sha256(
            json.dumps(
                {
                    "request": request.model_dump(mode="json"),
                    "participants": [row.identity.model_dump(mode="json") for row in participants],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return manifest, sources, digest

    def _fence(self, request: RestoreRequest, fence: SessionFence):
        if (
            fence.connector != request.connector
            or fence.session != request.session
            or fence.source_generation != request.expected_generation
        ):
            raise CheckpointFailure("restore_generation_lineage")

    def _receipt(
        self,
        request: RestoreRequest,
        manifest: ApplicationCheckpoint,
        source: ParticipantSnapshot,
        participant: RestoreParticipant,
        receipt: ParticipantRestore,
    ):
        if (
            receipt.operation != request.operation
            or receipt.checkpoint != manifest.checkpoint
            or receipt.checkpoint_sha256 != request.artifact.sha256
            or receipt.participant_name != participant.identity.name
            or receipt.service_identity != participant.identity.service_identity
            or receipt.source_origin_instance != source.cut.participant.origin_instance
            or receipt.restored_origin_instance != participant.identity.origin_instance
        ):
            raise CheckpointFailure("restore_receipt_lineage")

    async def restore(
        self, request: RestoreRequest, participants: tuple[RestoreParticipant, ...]
    ) -> RestoreResult:
        manifest, sources, digest = await self._manifest(request, participants)
        if request.deadline_at.tzinfo is None:
            raise CheckpointFailure("invalid_restore_deadline")
        previous = self.journal.begin(request.operation, digest, phase="restoring")
        if previous is not None:
            if previous["phase"] != "complete":
                raise CheckpointFailure("restore_custody_in_doubt")
            return await self._saved(
                request, ArtifactReference.model_validate(previous["artifact"])
            )
        result_artifact = None
        try:
            remaining_seconds = (request.deadline_at - datetime.now(timezone.utc)).total_seconds()
            if not 0 < remaining_seconds <= 600:
                raise CheckpointFailure("invalid_restore_deadline")
            async with asyncio.timeout(remaining_seconds):
                # Check every blob before the first origin effect; reads stay
                # serial and bounded instead of buffering sixteen large blobs.
                for source in sources.values():
                    await asyncio.to_thread(self.store.get, source.artifact)
                fence = await self.authority.restart(request, manifest)
                self._fence(request, fence)
                receipts = []
                for participant in participants:
                    source = sources[participant.identity.name]
                    payload = await asyncio.to_thread(self.store.get, source.artifact)
                    receipt = await participant.restore(request, source, payload, fence)
                    self._receipt(request, manifest, source, participant, receipt)
                    receipts.append(receipt)
                result = RestoreResult(
                    operation=request.operation,
                    checkpoint=manifest.checkpoint,
                    checkpoint_sha256=request.artifact.sha256,
                    fence=fence,
                    receipts=tuple(receipts),
                )
                result_artifact = await asyncio.to_thread(
                    self.store.put, result.model_dump_json().encode()
                )
                self.journal.record(
                    request.operation,
                    request_sha256=digest,
                    phase="sealing",
                    artifact=result_artifact,
                )
                await self.authority.complete_restore(request, result)
                self.journal.record(
                    request.operation,
                    request_sha256=digest,
                    phase="complete",
                    artifact=result_artifact,
                )
                return result
        except BaseException:
            self.journal.record(
                request.operation, request_sha256=digest, phase="in_doubt", artifact=result_artifact
            )
            raise

    async def _saved(self, request: RestoreRequest, artifact: ArtifactReference) -> RestoreResult:
        result = RestoreResult.model_validate_json(
            await asyncio.to_thread(self.store.get, artifact)
        )
        if (
            result.operation != request.operation
            or result.checkpoint_sha256 != request.artifact.sha256
        ):
            raise CheckpointFailure("restore_result_lineage")
        self._fence(request, result.fence)
        return result

    async def reconcile(
        self, request: RestoreRequest, participants: tuple[RestoreParticipant, ...]
    ) -> RestoreResult:
        """Observe durable origin facts; settle only fenced, idempotent control completion."""
        manifest, sources, digest = await self._manifest(request, participants)
        state = self.journal.lookup(request.operation)
        if state["request_sha256"] != digest:
            raise CheckpointFailure("restore_operation_conflict")
        artifact = (
            ArtifactReference.model_validate(state["artifact"]) if state["artifact"] else None
        )
        result = await self._saved(request, artifact) if artifact else None
        if result is not None:
            for participant, receipt in zip(participants, result.receipts, strict=True):
                self._receipt(
                    request, manifest, sources[participant.identity.name], participant, receipt
                )
            if state["phase"] == "complete" or await self.authority.lookup_completion(
                request, result
            ):
                self.journal.record(
                    request.operation, request_sha256=digest, phase="complete", artifact=artifact
                )
                return result
        # A fresh control fence may be needed after actor wake. Every origin
        # must independently prove its exact completed restore; no origin POST
        # is sent here, even when its reply was lost before journal settlement.
        fence = await self.authority.lookup_restart(request, manifest)
        if fence is None:
            raise CheckpointFailure("restore_custody_in_doubt")
        self._fence(request, fence)
        receipts = []
        for participant in participants:
            source = sources[participant.identity.name]
            receipt = await participant.lookup_restore(request, source)
            if receipt is None:
                raise CheckpointFailure("restore_custody_in_doubt")
            self._receipt(request, manifest, source, participant, receipt)
            if result is not None and receipt != result.receipts[len(receipts)]:
                raise CheckpointFailure("restore_receipt_conflict")
            receipts.append(receipt)
        result = RestoreResult(
            operation=request.operation,
            checkpoint=manifest.checkpoint,
            checkpoint_sha256=request.artifact.sha256,
            fence=fence,
            receipts=tuple(receipts),
        )
        artifact = await asyncio.to_thread(self.store.put, result.model_dump_json().encode())
        self.journal.record(
            request.operation, request_sha256=digest, phase="sealing", artifact=artifact
        )
        try:
            await self.authority.complete_restore(request, result)
            self.journal.record(
                request.operation, request_sha256=digest, phase="complete", artifact=artifact
            )
            return result
        except BaseException:
            self.journal.record(
                request.operation, request_sha256=digest, phase="in_doubt", artifact=artifact
            )
            raise
