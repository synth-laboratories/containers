"""Bound unresolved checkpoint custody; archive immutable completed receipts.

See docs/specs/tunnel-v2-connector.md. Archive is history, never replay permission.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
from uuid import UUID, uuid4

from .contracts import ArtifactReference, CheckpointFailure

MAX_PENDING_OPERATIONS = 4096


def _sync(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class FileCheckpointJournal:
    """One durable owner per operation; uncertain entries never leave active custody."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        for name in ("operations", "archive"):
            (self.root / name).mkdir(exist_ok=True, mode=0o700)
        _sync(self.root)

    @contextmanager
    def _admission(self):
        # One lock spans transitions and archive renames. Readers cannot observe
        # the rename gap, and a new process cannot replay an archived operation.
        with (self.root / "admission.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _active(self, operation: UUID) -> Path:
        return self.root / "operations" / str(operation)

    def _archived(self, operation: UUID) -> Path:
        return self.root / "archive" / operation.hex[:2] / str(operation)

    def _directory(self, operation: UUID) -> Path:
        active, archived = self._active(operation), self._archived(operation)
        if active.exists() and archived.exists():
            raise CheckpointFailure("checkpoint_custody_in_doubt")
        return archived if archived.exists() else active

    def _lookup(self, operation: UUID) -> dict:
        path = self._directory(operation) / "state.json"
        try:
            if path.stat().st_size > 4096:
                raise ValueError("journal capacity")
            state = json.loads(path.read_bytes())
            if set(state) != {"request_sha256", "phase", "artifact"}:
                raise ValueError("journal contract")
            return state
        except (OSError, ValueError):
            raise CheckpointFailure("checkpoint_custody_in_doubt") from None

    def lookup(self, operation: UUID) -> dict:
        with self._admission():
            return self._lookup(operation)

    def _archive_complete(self, operation: UUID, state: dict) -> None:
        if state["phase"] != "complete" or state["artifact"] is None:
            return
        active, archived = self._active(operation), self._archived(operation)
        if not active.exists():
            return
        archived.parent.mkdir(exist_ok=True, mode=0o700)
        _sync(archived.parent.parent)
        if archived.exists():
            raise CheckpointFailure("checkpoint_custody_in_doubt")
        os.rename(active, archived)
        _sync(active.parent)
        _sync(archived.parent)

    def begin(self, operation: UUID, request_sha256: str, *, phase: str = "pausing") -> dict | None:
        with self._admission():
            directory = self._directory(operation)
            if directory.exists():
                state = self._lookup(operation)
                if state["request_sha256"] != request_sha256:
                    raise CheckpointFailure("checkpoint_operation_conflict")
                self._archive_complete(operation, state)
                return state
            active = self.root / "operations"
            # Upgrade old journals conservatively, including a crash between
            # completed-state fsync and archive rename. Uncertain state stays put.
            entries = list(active.iterdir())
            if len(entries) >= MAX_PENDING_OPERATIONS:
                for entry in entries:
                    try:
                        saved = UUID(entry.name)
                    except ValueError:
                        raise CheckpointFailure("checkpoint_custody_in_doubt") from None
                    self._archive_complete(saved, self._lookup(saved))
            if sum(1 for _ in active.iterdir()) >= MAX_PENDING_OPERATIONS:
                raise CheckpointFailure("checkpoint_journal_capacity")
            if phase == "complete":
                raise CheckpointFailure("checkpoint_complete_requires_artifact")
            directory.mkdir(mode=0o700)
            self._record(operation, request_sha256=request_sha256, phase=phase, artifact=None)
            _sync(directory.parent)
            return None

    def record(self, operation: UUID, *, request_sha256: str, phase: str,
               artifact: ArtifactReference | None = None) -> None:
        with self._admission():
            self._record(operation, request_sha256=request_sha256, phase=phase, artifact=artifact)

    def _record(self, operation: UUID, *, request_sha256: str, phase: str,
                artifact: ArtifactReference | None) -> None:
        directory = self._directory(operation)
        if (directory / "state.json").exists():
            previous = self._lookup(operation)
            if previous["request_sha256"] != request_sha256:
                raise CheckpointFailure("checkpoint_operation_conflict")
            if previous["phase"] == "complete":
                if artifact is not None and previous["artifact"] != artifact.model_dump(mode="json"):
                    raise CheckpointFailure("checkpoint_operation_conflict")
                self._archive_complete(operation, previous)
                return
        if phase == "complete" and artifact is None:
            raise CheckpointFailure("checkpoint_complete_requires_artifact")
        state = {"request_sha256": request_sha256, "phase": phase,
                 "artifact": artifact.model_dump(mode="json") if artifact else None}
        temporary = directory / (".state-" + uuid4().hex)
        try:
            with temporary.open("xb") as stream:
                stream.write(json.dumps(state, sort_keys=True, separators=(",", ":")).encode())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, directory / "state.json")
            _sync(directory)
            self._archive_complete(operation, state)
        finally:
            # Completion may have renamed the containing directory.
            (self._directory(operation) / temporary.name).unlink(missing_ok=True)
