"""Local append-before-notify operator custody, preserving caller event schemas.

This is distinct from the exact rollout trace journal. A per-file OS lock makes
sequence allocation and fsync one operation across worker restarts/processes.
A partial/corrupt tail refuses append; recovery must preserve and reconcile it.
Linux and macOS are supported. Hosted custody belongs in Rhodes/Artifact Platform.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO


class JournalIntegrityError(ValueError):
    """History cannot safely accept another event without reconciliation."""


class OperatorJournal:
    def __init__(self, path: Path, *, run_id: str, max_record_bytes: int = 1024 * 1024) -> None:
        if type(max_record_bytes) is not int or max_record_bytes < 2:
            raise ValueError("max_record_bytes must be an integer of at least two")
        self.path = path
        self.run_id = run_id
        self.max_record_bytes = max_record_bytes

    def _last_sequence(self, handle: BinaryIO) -> int:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        if size == 0:
            return 0
        handle.seek(max(0, size - self.max_record_bytes - 1))
        tail = handle.read(self.max_record_bytes + 1)
        if not tail.endswith(b"\n"):
            raise JournalIntegrityError("incomplete operator journal tail; reconcile before append")
        line = tail[:-1].rsplit(b"\n", 1)[-1]
        if len(line) + 1 > self.max_record_bytes:
            raise JournalIntegrityError("operator journal record exceeds bound")
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError) as error:
            raise JournalIntegrityError("invalid operator journal tail") from error
        if not isinstance(row, dict) or row.get("run_id") != self.run_id:
            raise JournalIntegrityError("operator journal run identity mismatch")
        sequence = row.get("seq")
        if type(sequence) is not int or sequence < 1:
            raise JournalIntegrityError("invalid operator journal sequence")
        return sequence

    def append(self, make_event: Callable[[int], dict[str, Any]]) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # a+b does not truncate; flock protects allocation as well as the append.
        with self.path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                sequence = self._last_sequence(handle) + 1
                row = make_event(sequence)
                if row.get("run_id") != self.run_id or row.get("seq") != sequence:
                    raise JournalIntegrityError("event cannot override journal identity")
                data = (
                    json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
                ).encode()
                if len(data) > self.max_record_bytes:
                    raise JournalIntegrityError("operator event exceeds record bound")
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
                # Include directory-entry custody for a newly created journal.
                directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
                return row
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
