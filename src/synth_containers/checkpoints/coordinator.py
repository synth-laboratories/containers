"""Joint participant cuts with durable custody and no automatic origin replay.

See synth-tunnel/docs/SESSION_CHECKPOINTS.md. Origins remain quiesced after a
failure; recovery observes the recorded operation instead of repeating effects.
"""

from __future__ import annotations
import asyncio
from datetime import datetime, timezone
from hashlib import sha256
import json
from time import monotonic
from typing import Protocol
from uuid import UUID
from pydantic import Field
from .contracts import (
    ApplicationCheckpoint,
    ArtifactReference,
    CheckpointFailure,
    CheckpointSemantics,
    Contract,
    ParticipantCut,
    ParticipantIdentity,
    ParticipantSnapshot,
    SessionFence,
)
from .storage import ArtifactStore
from .journal import FileCheckpointJournal


class CaptureRequest(Contract):
    operation: UUID
    checkpoint: UUID
    connector: UUID
    session: UUID
    expected_generation: int = Field(ge=1, le=9_007_199_254_740_991)
    semantics: CheckpointSemantics
    deadline_at: datetime


class CheckpointParticipant(Protocol):
    identity: ParticipantIdentity

    async def quiesce(
        self, operation: UUID, fence: SessionFence, deadline_at: datetime
    ) -> ParticipantCut: ...
    async def capture(self, cut: ParticipantCut, deadline_at: datetime) -> bytes: ...


class CheckpointAuthority(Protocol):
    async def pause(self, request: CaptureRequest) -> SessionFence: ...
    async def seal(
        self, checkpoint: ApplicationCheckpoint, artifact: ArtifactReference
    ) -> None: ...
    async def lookup_seal(
        self, checkpoint: ApplicationCheckpoint, artifact: ArtifactReference
    ) -> bool: ...


def capture_identity(
    request: CaptureRequest, participants: tuple[CheckpointParticipant, ...]
) -> str:
    return sha256(
        json.dumps(
            {
                "request": request.model_dump(mode="json"),
                "participants": [
                    participant.identity.model_dump(mode="json") for participant in participants
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


class ApplicationCheckpointCoordinator:
    def __init__(
        self,
        *,
        authority: CheckpointAuthority,
        store: ArtifactStore,
        journal: FileCheckpointJournal,
    ):
        self.authority = authority
        self.store = store
        self.journal = journal

    async def capture(
        self, request: CaptureRequest, participants: tuple[CheckpointParticipant, ...]
    ) -> tuple[ApplicationCheckpoint, ArtifactReference]:
        """Seal one joint cut; repeat completed capture only by reading its immutable result."""
        identities = tuple(participant.identity for participant in participants)
        if not 1 <= len(identities) <= 16 or len({identity.name for identity in identities}) != len(
            identities
        ):
            raise CheckpointFailure("invalid_checkpoint_participant_set")
        if request.deadline_at.tzinfo is None:
            raise CheckpointFailure("invalid_checkpoint_deadline")
        if request.semantics == CheckpointSemantics.TRUE_ENVIRONMENT_SNAPSHOT and any(
            identity.semantics != request.semantics for identity in identities
        ):
            raise CheckpointFailure("checkpoint_semantics_unsupported")
        digest = capture_identity(request, participants)
        previous = self.journal.begin(request.operation, digest)
        if previous:
            if previous["phase"] != "complete":
                raise CheckpointFailure("checkpoint_custody_in_doubt")
            artifact = ArtifactReference.model_validate(previous["artifact"])
            manifest = ApplicationCheckpoint.model_validate_json(
                await asyncio.to_thread(self.store.get, artifact)
            )
            return manifest, artifact
        remaining_seconds = (request.deadline_at - datetime.now(timezone.utc)).total_seconds()
        artifact = None
        try:
            if not 0 < remaining_seconds <= 600:
                raise CheckpointFailure("invalid_checkpoint_deadline")
            async with asyncio.timeout(remaining_seconds):
                fence = await self.authority.pause(request)
                # The supported v2 permit envelope is five seconds. Start after
                # the pause acknowledgement; the actor's wall timestamp cannot
                # be compared with this host's clock to establish quiescence.
                permit_deadline = monotonic() + 5.0
                if (
                    fence.connector != request.connector
                    or fence.session != request.session
                    or fence.source_generation != request.expected_generation
                    or fence.generation <= fence.source_generation
                ):
                    raise CheckpointFailure("checkpoint_pause_lineage")
                self.journal.record(request.operation, request_sha256=digest, phase="quiescing")
                cuts = []
                for participant in participants:
                    cut = await participant.quiesce(request.operation, fence, request.deadline_at)
                    if (
                        cut.operation != request.operation
                        or cut.participant != participant.identity
                        or cut.pending_operations
                        or cut.unresolved_operations
                    ):
                        raise CheckpointFailure("checkpoint_participant_not_quiescent")
                    cuts.append(cut)
                await asyncio.sleep(max(0, permit_deadline - monotonic()))
                self.journal.record(request.operation, request_sha256=digest, phase="capturing")
                snapshots = []
                for participant, cut in zip(participants, cuts, strict=True):
                    payload = await participant.capture(cut, request.deadline_at)
                    snapshot = await asyncio.to_thread(self.store.put, payload)
                    snapshots.append(ParticipantSnapshot(cut=cut, artifact=snapshot))
                manifest = ApplicationCheckpoint(
                    checkpoint=request.checkpoint,
                    operation=request.operation,
                    fence=fence,
                    semantics=request.semantics,
                    participants=tuple(snapshots),
                )
                artifact = await asyncio.to_thread(self.store.put, manifest.canonical_bytes())
                self.journal.record(
                    request.operation, request_sha256=digest, phase="sealing", artifact=artifact
                )
                await self.authority.seal(manifest, artifact)
                self.journal.record(
                    request.operation, request_sha256=digest, phase="complete", artifact=artifact
                )
                return manifest, artifact
        except BaseException:
            # Do not resume participants, clear custody, or infer stop from a
            # transport exception. The journal survives coordinator replacement.
            self.journal.record(
                request.operation, request_sha256=digest, phase="in_doubt", artifact=artifact
            )
            raise

    async def reconcile(
        self, request: CaptureRequest, participants: tuple[CheckpointParticipant, ...]
    ) -> tuple[ApplicationCheckpoint, ArtifactReference]:
        """Settle a lost seal reply by lookup; never repeat capture or origin work."""
        digest = capture_identity(request, participants)
        state = self.journal.lookup(request.operation)
        if state["request_sha256"] != digest:
            raise CheckpointFailure("checkpoint_operation_conflict")
        if state["artifact"] is None:
            raise CheckpointFailure("checkpoint_custody_in_doubt")
        artifact = ArtifactReference.model_validate(state["artifact"])
        manifest = ApplicationCheckpoint.model_validate_json(
            await asyncio.to_thread(self.store.get, artifact)
        )
        if (
            manifest.operation != request.operation
            or manifest.checkpoint != request.checkpoint
            or manifest.fence.connector != request.connector
            or manifest.fence.session != request.session
        ):
            raise CheckpointFailure("checkpoint_manifest_lineage")
        if state["phase"] != "complete" and not await self.authority.lookup_seal(
            manifest, artifact
        ):
            raise CheckpointFailure("checkpoint_custody_in_doubt")
        self.journal.record(
            request.operation, request_sha256=digest, phase="complete", artifact=artifact
        )
        return manifest, artifact
