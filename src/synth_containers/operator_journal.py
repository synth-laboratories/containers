"""Local append-before-notify operator custody, preserving caller event schemas.

This is distinct from the exact rollout trace journal. A per-file OS lock makes
sequence allocation and fsync one operation across worker restarts/processes.
A partial/corrupt tail refuses append; recovery must preserve and reconcile it.
Linux and macOS are supported. Hosted custody belongs in Rhodes/Artifact Platform.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import time
from collections.abc import Callable, Iterator
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


def read_operator_events(
    path: Path,
    *,
    run_id: str | None = None,
    after_sequence: int = 0,
    limit: int = 100,
    max_journal_bytes: int = 64 * 1024 * 1024,
) -> dict[str, Any]:
    """Read a bounded, identity-checked replay page without altering custody.

    A shared nonblocking lock avoids reading an in-progress append. Corrupt or
    truncated history and a future cursor are explicit failures, never an empty
    success page. Large journals require an indexed/hosted reader.
    """
    if type(after_sequence) is not int or after_sequence < 0:
        raise ValueError("after_sequence must be a nonnegative integer")
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("journal page limit must be between 1 and 1000")
    if type(max_journal_bytes) is not int or not 1 <= max_journal_bytes <= 64 * 1024 * 1024:
        raise ValueError("journal read bound must be between one byte and 64 MiB")
    if run_id is not None and (not isinstance(run_id, str) or not run_id):
        raise ValueError("run_id must be a nonempty string")
    with path.open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        try:
            data = handle.read(max_journal_bytes + 1)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    if len(data) > max_journal_bytes:
        raise JournalIntegrityError("operator journal exceeds replay read bound")
    if data and not data.endswith(b"\n"):
        raise JournalIntegrityError("incomplete operator journal tail; reconcile before replay")
    selected = []
    high_water = 0
    identity = run_id
    for line in data.splitlines():
        if len(line) > 1024 * 1024:
            raise JournalIntegrityError("operator journal record exceeds replay bound")
        try:
            row = json.loads(line)
            json.dumps(row, allow_nan=False)
        except (ValueError, UnicodeError) as error:
            raise JournalIntegrityError("invalid operator journal record") from error
        if not isinstance(row, dict) or not isinstance(row.get("run_id"), str) or not row["run_id"]:
            raise JournalIntegrityError("operator journal record has no run identity")
        identity = identity or row["run_id"]
        if row["run_id"] != identity:
            raise JournalIntegrityError("operator journal run identity mismatch")
        sequence = row.get("seq")
        if type(sequence) is not int or sequence != high_water + 1:
            raise JournalIntegrityError("operator journal sequence is not contiguous")
        high_water = sequence
        if sequence > after_sequence and len(selected) < limit:
            selected.append(row)
    if after_sequence > high_water:
        raise JournalIntegrityError("operator journal cursor is ahead of durable history")
    next_sequence = selected[-1]["seq"] if selected else after_sequence
    return {
        "run_id": identity,
        "events": selected,
        "next_sequence": next_sequence,
        "high_water": high_water,
        "has_more": next_sequence < high_water,
    }


def follow_operator_events(
    path: Path,
    *,
    run_id: str | None = None,
    after_sequence: int = 0,
    limit: int = 100,
    timeout_seconds: float = 300,
    poll_interval_seconds: float = 0.25,
) -> Iterator[dict[str, Any]]:
    """Replay and follow committed events; viewer exit never changes the run.

    Run identity is pinned by the first nonempty page. Foreign history, truncation,
    corrupt custody and unavailable paths fail explicitly. Writer lock contention
    alone is retried within the observation deadline. Events retain their durable
    sequence so callers can resume after the last event they processed.
    """
    for value, name, maximum in (
        (timeout_seconds, "timeout_seconds", 86400),
        (poll_interval_seconds, "poll_interval_seconds", 60),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite positive number")  # noqa: TRY004 - CLI validation
        if not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError(f"{name} must be between zero (exclusive) and {maximum}")
    deadline = time.monotonic() + timeout_seconds
    identity = run_id
    cursor = after_sequence
    while time.monotonic() < deadline:
        try:
            page = read_operator_events(path, run_id=identity, after_sequence=cursor, limit=limit)
        except BlockingIOError:
            page = None
        if page is not None:
            identity = page["run_id"]
            for event in page["events"]:
                cursor = event["seq"]
                yield event
            if page["has_more"]:
                continue
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(poll_interval_seconds, remaining))
