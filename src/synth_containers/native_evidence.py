"""Transfer a sanitized native operator journal into existing hosted custody.

The source remains immutable. A local append-and-fsync receipt advances only
after the backend acknowledges the submitted page; uncertain calls replay the
same producer identities. This is not a Trace V5 converter or result publisher.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
from typing import Any

from .operator_journal import JournalIntegrityError, OperatorJournal, read_operator_events
from .pools import PoolClient, PoolClientError


async def publish_native_journal(
    client: PoolClient,
    *,
    path: Path,
    run_id: str,
    pool_id: str,
    rollout_id: str,
    execution_epoch: str,
    producer_id: str,
    checkpoint: Path,
    complete: bool = False,
    max_pages: int = 100,
) -> dict[str, Any]:
    """Upload a bounded prefix, retaining identity-checked restart receipts.

    Only a sanitized summary journal is admissible. Complete must be requested
    after the producer has stopped appending. The worker epoch is provided by
    hosted ownership, never inferred or acquired through this function.
    """
    if type(max_pages) is not int or not 1 <= max_pages <= 10000:
        raise ValueError("max_pages must be between 1 and 10000")
    if path.resolve() == checkpoint.resolve():
        raise ValueError("custody checkpoint must not overwrite the source")
    identity = {
        "run_id": run_id,
        "backend_url": client.backend_url,
        "pool_id": pool_id,
        "rollout_id": rollout_id,
        "producer_id": producer_id,
    }
    binding = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    lock_path = checkpoint.with_name(checkpoint.name + ".lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            cursor = 0
            if checkpoint.exists():
                history = read_operator_events(checkpoint, run_id=binding, limit=1000)
                while history["has_more"]:
                    history = read_operator_events(
                        checkpoint,
                        run_id=binding,
                        after_sequence=history["next_sequence"],
                        limit=1000,
                    )
                if history["events"]:
                    last = history["events"][-1]
                    cursor = last["source_sequence"]
                    if type(cursor) is not int or cursor < 0:
                        raise JournalIntegrityError("invalid native upload checkpoint")
                    # Verify the acknowledged source prefix still has the same bytes.
                    if _prefix_digest(path, run_id, cursor) != last["source_sha256"]:
                        raise JournalIntegrityError("acknowledged native journal was rewritten")
            receipts = OperatorJournal(checkpoint, run_id=binding)
            for _ in range(max_pages):
                page = read_operator_events(path, run_id=run_id, after_sequence=cursor, limit=128)
                records: list[dict[str, Any]] = []
                base = {
                    "execution_epoch": execution_epoch,
                    "producer_id": producer_id,
                    "events": records,
                    "complete": False,
                }
                for row in page["events"]:
                    record = {"event_id": f"{row['seq']}", "sequence": row["seq"], "event": row}
                    records.append(record)
                    if (
                        len(
                            json.dumps(
                                base, sort_keys=True, separators=(",", ":"), allow_nan=False
                            ).encode()
                        )
                        > 262144
                    ):
                        records.pop()
                        if not records:
                            raise ValueError("native event exceeds hosted page bound")
                        break
                next_cursor = records[-1]["sequence"] if records else cursor
                terminal = complete and next_cursor == page["high_water"]
                if not records and not terminal:
                    return {"next_sequence": cursor, "complete": False, "has_more": False}
                digest = _prefix_digest(path, run_id, next_cursor)
                receipt = await client.publish_native_events(
                    pool_id,
                    rollout_id,
                    execution_epoch=execution_epoch,
                    producer_id=producer_id,
                    events=records,
                    complete=terminal,
                )
                if receipt["next_sequence"] > page["high_water"]:
                    raise PoolClientError("hosted producer cursor is ahead of local history")
                if digest != _prefix_digest(path, run_id, next_cursor):
                    raise JournalIntegrityError("native journal changed during upload")
                # Do not skip to a server cursor beyond this submitted prefix:
                # replay later pages to verify their content identities too.
                receipts.append(
                    lambda sequence, next_cursor=next_cursor, digest=digest, receipt=receipt: {
                        "run_id": binding,
                        "seq": sequence,
                        "source_sequence": next_cursor,
                        "source_sha256": digest,
                        "execution_epoch": execution_epoch,
                        "custody_scope": "native_journal",
                        "receipt": receipt,
                    }
                )
                cursor = next_cursor
                if cursor == page["high_water"]:
                    return {"next_sequence": cursor, "complete": terminal, "has_more": False}
            return {"next_sequence": cursor, "complete": False, "has_more": True}
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _prefix_digest(path: Path, run_id: str, sequence: int) -> str:
    digest = hashlib.sha256()
    cursor = 0
    while cursor < sequence:
        page = read_operator_events(
            path, run_id=run_id, after_sequence=cursor, limit=min(1000, sequence - cursor)
        )
        if not page["events"]:
            raise JournalIntegrityError("native checkpoint exceeds source history")
        for row in page["events"]:
            digest.update(
                (
                    json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
                ).encode()
            )
        cursor = page["next_sequence"]
    return digest.hexdigest()
