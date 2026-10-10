"""Restore custody under lost replies, corruption and partial origin effects."""

from __future__ import annotations
import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4
import pytest
from synth_containers.checkpoints import (
    ApplicationCheckpoint,
    ApplicationRestoreCoordinator,
    CheckpointFailure,
    CheckpointSemantics,
    FileArtifactStore,
    FileCheckpointJournal,
    ParticipantCut,
    ParticipantIdentity,
    ParticipantRestore,
    RestoreRequest,
    SessionFence,
)
from synth_containers.checkpoints.contracts import ParticipantSnapshot


def fixture(tmp_path):
    store = FileArtifactStore(tmp_path / "blobs", namespace="owned-restore")
    source = ParticipantIdentity(
        name="rollout",
        service_identity="service",
        origin_instance=uuid4(),
        semantics=CheckpointSemantics.APPLICATION_COLD_RESTORE,
    )
    fence = SessionFence(
        connector=uuid4(), session=uuid4(), source_generation=1, generation=2, permit_barrier_ms=0
    )
    snapshot = ParticipantSnapshot(
        cut=ParticipantCut(
            operation=uuid4(),
            receipt=uuid4(),
            participant=source,
            pending_operations=0,
            unresolved_operations=0,
        ),
        artifact=store.put(b"completed-step-and-pending-mailboxes"),
    )
    manifest = ApplicationCheckpoint(
        checkpoint=uuid4(),
        operation=snapshot.cut.operation,
        fence=fence,
        semantics=source.semantics,
        participants=(snapshot,),
    )
    artifact = store.put(manifest.canonical_bytes())
    request = RestoreRequest(
        operation=uuid4(),
        connector=fence.connector,
        session=fence.session,
        expected_generation=2,
        artifact=artifact,
        deadline_at=datetime.now(timezone.utc) + timedelta(seconds=30),
    )

    class Participant:
        identity = source.model_copy(update={"origin_instance": uuid4()})
        calls = 0
        saved = None
        lose_reply = False

        async def restore(self, request, snapshot, payload, fence):
            self.calls += 1
            assert payload == b"completed-step-and-pending-mailboxes"
            self.saved = ParticipantRestore(
                operation=request.operation,
                checkpoint=manifest.checkpoint,
                checkpoint_sha256=artifact.sha256,
                participant_name=self.identity.name,
                service_identity=self.identity.service_identity,
                source_origin_instance=source.origin_instance,
                restored_origin_instance=self.identity.origin_instance,
                receipt=uuid4(),
            )
            if self.lose_reply:
                raise TimeoutError("injected_lost_origin_reply")
            return self.saved

        async def lookup_restore(self, request, source):
            return self.saved

    class Authority:
        calls = 0
        completed = None

        async def restart(self, request, manifest):
            self.calls += 1
            return fence.model_copy(update={"source_generation": 2, "generation": 3})

        async def lookup_restart(self, request, manifest):
            return (
                fence.model_copy(update={"source_generation": 2, "generation": 3})
                if self.calls
                else None
            )

        async def complete_restore(self, request, result):
            self.completed = result
            raise TimeoutError("injected_lost_completion_reply")

        async def lookup_completion(self, request, result):
            return self.completed == result

    participant, authority = Participant(), Authority()

    def coordinator():
        return ApplicationRestoreCoordinator(
            authority=authority, store=store, journal=FileCheckpointJournal(tmp_path / "journal")
        )

    return request, participant, authority, store, snapshot, coordinator


def test_lost_completion_reconciles_and_duplicate_restore_never_replays(tmp_path):
    async def run():
        request, participant, authority, store, snapshot, coordinator = fixture(tmp_path)
        with pytest.raises(TimeoutError, match="lost_completion"):
            await coordinator().restore(request, (participant,))
        with pytest.raises(CheckpointFailure, match="custody_in_doubt"):
            await coordinator().restore(request, (participant,))
        result = await coordinator().reconcile(request, (participant,))
        assert await coordinator().restore(request, (participant,)) == result
        assert participant.calls == authority.calls == 1
        assert (
            result.receipts[0].source_origin_instance != result.receipts[0].restored_origin_instance
        )
        assert store.get(snapshot.artifact) == b"completed-step-and-pending-mailboxes"

    asyncio.run(run())


def test_lost_origin_reply_stays_fenced_and_is_observed_without_replay(tmp_path):
    async def run():
        request, participant, authority, store, snapshot, coordinator = fixture(tmp_path)
        participant.lose_reply = True
        with pytest.raises(TimeoutError, match="lost_origin"):
            await coordinator().restore(request, (participant,))
        saved = participant.saved
        participant.saved = None
        with pytest.raises(CheckpointFailure, match="custody_in_doubt"):
            await coordinator().reconcile(request, (participant,))
        assert authority.completed is None
        participant.saved = saved
        with pytest.raises(TimeoutError, match="lost_completion"):
            await coordinator().reconcile(request, (participant,))
        result = await coordinator().reconcile(request, (participant,))
        assert result.receipts == (saved,)
        assert participant.calls == authority.calls == 1

    asyncio.run(run())


def test_corrupt_participant_is_refused_before_any_restart_effect(tmp_path):
    async def run():
        request, participant, authority, store, snapshot, coordinator = fixture(tmp_path)
        path = tmp_path / "blobs" / snapshot.artifact.object_key
        path.write_bytes(b"x" * snapshot.artifact.size_bytes)
        with pytest.raises(CheckpointFailure, match="digest_mismatch"):
            await coordinator().restore(request, (participant,))
        assert participant.calls == authority.calls == 0

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["restoring", "open"])
@pytest.mark.parametrize("mutation", [None, "fence", "receipt", "reference", "session", "operation"])
def test_http_restart_lookup_preserves_receipt_and_uses_live_completion_fence(tmp_path, phase, mutation):
    from synth_containers.checkpoints import ApplicationCheckpoint, CheckpointFailure, HttpSessionAuthority

    async def run():
        request, _, _, store, _, _ = fixture(tmp_path)
        manifest = ApplicationCheckpoint.model_validate_json(store.get(request.artifact))
        first = manifest.participants[0].cut
        reference = {
            "id": str(manifest.checkpoint), "connector": str(manifest.fence.connector),
            "session": str(manifest.fence.session), "source_generation": manifest.fence.source_generation,
            "service_identity": "service", "source_origin_instance": str(first.participant.origin_instance),
            "cut_receipt": str(first.receipt), "semantics": manifest.semantics.value,
            "artifact": request.artifact.model_dump(mode="json", exclude={"size_bytes"}),
        }
        result = {"generation": 3, "checkpoint": reference}
        restart = {"operation": str(request.operation), "source_generation": 2,
                   "generation": 3, "restore_generation": 5, "reference": dict(reference),
                   "completion": {"generation": 5} if phase == "open" else None}
        observed = {"session": str(request.session), "generation": 6 if phase == "open" else 5,
                    "lifecycle": {"phase": phase, "restart": restart}}
        if mutation == "fence":
            if phase == "open":
                restart["completion"]["generation"] = 7
            else:
                restart["restore_generation"] = 4
        elif mutation == "receipt":
            restart["generation"] = 4
        elif mutation == "reference":
            restart["reference"]["cut_receipt"] = str(uuid4())
        elif mutation == "session":
            observed["session"] = str(uuid4())
        elif mutation == "operation":
            restart["operation"] = str(uuid4())

        class Catalog:
            async def session_operation(self, connector, operation):
                return {"state": "applied", "result": result}
        authority = HttpSessionAuthority(catalog=Catalog(), gateway_url="http://gateway", service_identity="service")
        async def gateway(*args, **kwargs):
            return observed
        authority._gateway = gateway
        try:
            if mutation:
                with pytest.raises(CheckpointFailure, match="session_restart_lineage"):
                    await authority.lookup_restart(request, manifest)
            else:
                fence = await authority.lookup_restart(request, manifest)
                assert fence.generation == 5
                assert result["generation"] == restart["generation"] == 3
        finally:
            await authority.close()
    asyncio.run(run())
