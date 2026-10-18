"""A joint application cut must wait remote permits despite unrelated wall clocks."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4
import asyncio

import pytest
from synth_containers.checkpoints import (
    ApplicationCheckpointCoordinator, CaptureRequest, CheckpointSemantics,
    FileArtifactStore, FileCheckpointJournal, ParticipantCut, ParticipantIdentity,
    SessionFence,
)
from synth_containers.checkpoints import coordinator as module


@pytest.mark.parametrize("issuer_barrier", [0, 1_000_000_000_000_000])
def test_capture_waits_monotonic_permits_and_reuses_the_confirmed_cut(tmp_path, monkeypatch, issuer_barrier):
    clock = [100.0]
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(module, "monotonic", lambda: clock[0])
    # Keep the event loop's real clock and timeout machinery untouched.
    monkeypatch.setattr(module, "asyncio", SimpleNamespace(
        sleep=sleep, timeout=asyncio.timeout, to_thread=asyncio.to_thread))
    request = CaptureRequest(operation=uuid4(), checkpoint=uuid4(), connector=uuid4(),
                             session=uuid4(), expected_generation=1,
                             semantics=CheckpointSemantics.APPLICATION_COLD_RESTORE,
                             deadline_at=datetime.now(timezone.utc) + timedelta(seconds=30))

    class Authority:
        async def pause(self, request):
            return SessionFence(connector=request.connector, session=request.session,
                                source_generation=1, generation=2, permit_barrier_ms=issuer_barrier)

        async def seal(self, checkpoint, artifact):
            assert clock[0] == 105.0

    class Participant:
        captures = 0
        identity = ParticipantIdentity(name="rollout", service_identity="fixture",
                                       origin_instance=uuid4(), semantics=request.semantics)

        async def quiesce(self, operation, fence, deadline_at):
            clock[0] += 2
            return ParticipantCut(operation=operation, receipt=uuid4(), participant=self.identity,
                                  pending_operations=0, unresolved_operations=0)

        async def capture(self, cut, deadline_at):
            assert clock[0] == 105.0, "capture crossed an outstanding remote permit"
            self.captures += 1
            return b"completed rollout step and pending mailbox IDs"

    participant = Participant()
    journal = FileCheckpointJournal(tmp_path / "custody")
    coordinator = ApplicationCheckpointCoordinator(authority=Authority(),
                  store=FileArtifactStore(tmp_path / "blobs", namespace="fixture"), journal=journal)
    first = asyncio.run(coordinator.capture(request, (participant,)))
    assert asyncio.run(coordinator.capture(request, (participant,))) == first
    assert sleeps == [3.0] and participant.captures == 1
    assert journal.lookup(request.operation)["phase"] == "complete"
