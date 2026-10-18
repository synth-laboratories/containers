"""Origin operation identity and results survive process restart.

See docs/specs/tunnel-v2-connector.md. This journal is not an external-effect
transaction: interrupted callbacks remain uncertain and are never redispatched.
One local process owns a journal; persist its directory with the origin service.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import sqlite3
import time
from uuid import UUID, uuid4

from .rollout_lifecycle import TERMINAL_ROLLOUT_STATES
from .serde import JsonObject


class RolloutJournalError(RuntimeError):
    """A journal refusal happens before new callback dispatch."""


class RolloutJournal:
    MAX_UNRESOLVED = 4096
    MAX_RECORD_BYTES = 1024 * 1024

    def __init__(self, directory: Path, runtime_id: str) -> None:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock_fd = os.open(directory / "owner.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        self._db: sqlite3.Connection | None = None
        try:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RolloutJournalError("rollout_journal_owner_conflict") from error
            path = directory / "rollouts.sqlite3"
            fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            os.close(fd)
            self._db = sqlite3.connect(path, timeout=1)
            self._db.execute("PRAGMA journal_mode=DELETE")
            self._db.execute("PRAGMA synchronous=FULL")
            with self._db:
                self._db.execute("CREATE TABLE IF NOT EXISTS identity(runtime_id TEXT PRIMARY KEY)")
                rows = self._db.execute("SELECT runtime_id FROM identity LIMIT 2").fetchall()
                if len(rows) > 1 or (rows and rows[0][0] != runtime_id):
                    raise RolloutJournalError("rollout_journal_runtime_conflict")
                # Migrate the initial journal without changing existing receipts.
                columns = {row[1] for row in self._db.execute("PRAGMA table_info(identity)")}
                if "origin_instance" not in columns:
                    self._db.execute("ALTER TABLE identity ADD COLUMN origin_instance TEXT")
                self._db.execute("INSERT OR IGNORE INTO identity(runtime_id,origin_instance) VALUES(?,?)",
                                 (runtime_id, str(uuid4())))
                self._db.execute("UPDATE identity SET origin_instance=? WHERE runtime_id=? AND origin_instance IS NULL",
                                 (str(uuid4()), runtime_id))
                instance = self._db.execute("SELECT origin_instance FROM identity WHERE runtime_id=?", (runtime_id,)).fetchone()[0]
                try:
                    parsed = UUID(instance)
                except (ValueError, TypeError, AttributeError) as error:
                    raise RolloutJournalError("rollout_journal_identity_invalid") from error
                if parsed.int == 0:
                    raise RolloutJournalError("rollout_journal_identity_invalid")
                self.origin_instance = str(parsed)
                self._db.execute("CREATE TABLE IF NOT EXISTS rollouts(rollout_id TEXT PRIMARY KEY, digest TEXT NOT NULL, state TEXT NOT NULL, payload TEXT NOT NULL)")
                self._db.execute("CREATE INDEX IF NOT EXISTS unresolved_rollouts ON rollouts(state) WHERE state NOT IN ('completed','failed','cancelled','terminated','stopped')")
                interrupted = self._db.execute("SELECT rollout_id,payload FROM rollouts WHERE state IN ('running','cancelling')")
                for rollout_id, encoded in interrupted:
                    payload = json.loads(encoded)
                    payload.update(status="in_doubt", success_status="unknown",
                                   status_detail="origin_restarted_outcome_unverified",
                                   updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                    payload["metadata"] = {**payload.get("metadata", {}),
                                           "callback_task": "lost", "origin_outcome": "unverified",
                                           "resource_cleanup": "unverified"}
                    self._db.execute("UPDATE rollouts SET state='in_doubt',payload=? WHERE rollout_id=?",
                                     (self._encode(payload), rollout_id))
            # SQLite fsyncs commits; also persist the newly created directory entries.
            for parent in (directory, directory.parent):
                directory_fd = os.open(parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except BaseException:
            self.close()
            raise

    def _connection(self) -> sqlite3.Connection:
        if self._db is None:
            raise RolloutJournalError("rollout_journal_closed")
        return self._db

    @classmethod
    def _encode(cls, payload: JsonObject) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(encoded.encode()) > cls.MAX_RECORD_BYTES:
            raise RolloutJournalError("rollout_journal_record_limit")
        return encoded

    def get(self, rollout_id: str) -> JsonObject | None:
        row = self._connection().execute("SELECT payload FROM rollouts WHERE rollout_id=?", (rollout_id,)).fetchone()
        return None if row is None else json.loads(row[0])

    def receipt(self, rollout_id: str) -> JsonObject | None:
        """Bind a retained result to its durable origin and immutable input."""
        row = self._connection().execute("SELECT digest,payload FROM rollouts WHERE rollout_id=?", (rollout_id,)).fetchone()
        if row is None:
            return None
        return {"origin_instance": self.origin_instance, "input_sha256": row[0], "record": json.loads(row[1])}

    def admit(self, rollout_id: str, digest: str, payload: JsonObject) -> tuple[JsonObject, bool]:
        db = self._connection()
        with db:
            row = db.execute("SELECT digest,payload FROM rollouts WHERE rollout_id=?", (rollout_id,)).fetchone()
            if row is not None:
                if row[0] != digest:
                    raise RolloutJournalError("rollout_identity_conflict")
                return json.loads(row[1]), False
            count = db.execute("SELECT count(*) FROM rollouts WHERE state NOT IN ('completed','failed','cancelled','terminated','stopped')").fetchone()[0]
            if count >= self.MAX_UNRESOLVED:
                raise RolloutJournalError("rollout_journal_unresolved_limit")
            db.execute("INSERT INTO rollouts VALUES(?,?,?,?)",
                       (rollout_id, digest, payload["status"], self._encode(payload)))
        return payload, True

    def write(self, rollout_id: str, payload: JsonObject) -> None:
        db = self._connection()
        encoded = self._encode(payload)
        with db:
            row = db.execute("SELECT state,payload FROM rollouts WHERE rollout_id=?", (rollout_id,)).fetchone()
            if row is None:
                raise RolloutJournalError("rollout_journal_unknown_identity")
            if row[0] in TERMINAL_ROLLOUT_STATES:
                if json.loads(row[1]) != payload:
                    raise RolloutJournalError("rollout_journal_terminal_conflict")
                return
            if payload.get("rollout_id") != rollout_id:
                raise RolloutJournalError("rollout_identity_conflict")
            db.execute("UPDATE rollouts SET state=?,payload=? WHERE rollout_id=?",
                       (payload["status"], encoded, rollout_id))

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None
        if self._lock_fd >= 0:
            os.close(self._lock_fd)
            self._lock_fd = -1
