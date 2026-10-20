"""Bounded recovery of expired native Harbor trials using local custody."""

from __future__ import annotations

import asyncio
import fcntl
import json
import re
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Any

from daytona import ListSandboxesQuery
from daytona.common.errors import DaytonaNotFoundError

from .operator_journal import OperatorJournal, read_operator_events

OWNER_LABEL = "ai.synth.harbor.owner"


class HarborRecoveryError(ValueError):
    """Recovery cannot establish safe ownership or confirmed absence."""


def _recovery_state(trial_dir: Path, now: datetime) -> tuple[str, set[str]]:
    with (trial_dir / "resource-create-claim.json").open("rb") as handle:
        claim_bytes = handle.read(1025)
    if len(claim_bytes) > 1024:
        raise HarborRecoveryError("Native Harbor creation claim exceeds bound")
    claim = json.loads(claim_bytes)
    owner = claim.get("owner") if isinstance(claim, dict) else None
    if not isinstance(owner, str) or not re.fullmatch(r"synth-harbor-[0-9a-f]{32}", owner):
        raise HarborRecoveryError("Native Harbor creation claim has invalid owner")
    page = read_operator_events(trial_dir / "resource-events.jsonl", run_id=owner, limit=1000)
    if page["has_more"]:
        raise HarborRecoveryError("Native Harbor recovery event count exceeds bound")
    rows = page["events"]
    intents = [row for row in rows if row.get("event") == "resource.create_requested"]
    if len(intents) != 1 or intents[0].get("provider") != "daytona":
        raise HarborRecoveryError("Native Harbor recovery requires one Daytona creation intent")
    intent = intents[0]
    ttl = intent.get("ttl_minutes")
    # Older qualified journals bounded creation at 300 seconds but did not save
    # that allowance. Use the conservative ceiling, never a guessed short timeout.
    create_timeout = intent.get("creation_timeout_seconds", 300)
    if type(ttl) is not int or not 1 <= ttl <= 360:
        raise HarborRecoveryError("Native Harbor creation intent has invalid lifetime")
    if (
        isinstance(create_timeout, bool)
        or not isinstance(create_timeout, (int, float))
        or not 0 < create_timeout <= 300
    ):
        raise HarborRecoveryError("Native Harbor creation intent has invalid timeout")
    try:
        created_at = datetime.fromisoformat(intent["occurred_at"])
    except (KeyError, TypeError, ValueError) as error:
        raise HarborRecoveryError("Native Harbor creation intent has invalid timestamp") from error
    if created_at.tzinfo is None or now.tzinfo is None:
        raise HarborRecoveryError("Native Harbor recovery requires timezone-aware clocks")
    # Never take over within the provider creation allowance, resource lifetime,
    # or a one-minute clock margin. No force flag can waive this boundary.
    if now < created_at + timedelta(seconds=create_timeout + ttl * 60 + 60):
        raise HarborRecoveryError("Native Harbor resource lifetime has not expired for recovery")
    identifiers = {row["provider_id"] for row in rows if row.get("provider_id")}
    if len(identifiers) > 1 or any(not isinstance(value, str) for value in identifiers):
        raise HarborRecoveryError("Native Harbor creation cardinality is inconsistent")
    return owner, identifiers


async def reconcile_daytona_trial(
    trial_dir: Path, client: Any, *, now: datetime | None = None
) -> dict[str, Any]:
    """Recover only an expired, exactly-owned trial; never create or renew it.

    Local custody is required. An empty owner listing without any known resource
    ID is insufficient to prove that an ambiguous create never reached Daytona.
    Such trials retain an unconfirmed receipt and rely on provider TTL/custody
    reconciliation. Provider error text is never written to the journal.
    """
    if version("daytona") != "0.210.0":
        raise HarborRecoveryError("Native Daytona recovery requires Daytona 0.210.0")
    owner, known = _recovery_state(trial_dir, now or datetime.now(UTC))
    journal = OperatorJournal(
        trial_dir / "resource-events.jsonl", run_id=owner, max_record_bytes=16384
    )

    def record(event: str, **fields: Any) -> dict[str, Any]:
        return journal.append(
            lambda seq: {
                "schema_version": "synth.harbor-resource-event.v1",
                "run_id": owner,
                "seq": seq,
                "event": event,
                "occurred_at": datetime.now(UTC).isoformat(),
                **fields,
            }
        )

    async def owned_listing() -> list[Any]:
        found = []
        async for sandbox in client.list(
            ListSandboxesQuery(labels={OWNER_LABEL: owner}, limit=2), request_timeout=10
        ):
            if sandbox.labels.get(OWNER_LABEL) != owner:
                raise HarborRecoveryError("Native Daytona listing ownership mismatch")
            found.append(sandbox)
            if len(found) > 1:
                raise HarborRecoveryError("Native Daytona owner has multiple resources")
        return found

    with (trial_dir / "resource-recovery.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            record("resource.recovery_requested")
            async with asyncio.timeout(90):
                listed = await owned_listing()
                identifiers = known | {sandbox.id for sandbox in listed}
                if len(identifiers) != 1:
                    raise HarborRecoveryError(
                        "Native Daytona resource identity remains unconfirmed"
                    )
                identifier = next(iter(identifiers))
                # Persist discovered handles before issuing a deletion request.
                record("resource.recovery_identified", provider_id=identifier)
                try:
                    sandbox = await client.get(identifier, request_timeout=10)
                except DaytonaNotFoundError:
                    sandbox = None
                if sandbox is not None:
                    if sandbox.labels.get(OWNER_LABEL) != owner:
                        raise HarborRecoveryError("Native Daytona resource ownership mismatch")
                    record("resource.cleanup_requested", provider_id=identifier, source="recovery")
                    try:
                        await client.delete(sandbox, timeout=20, wait=True)
                    except DaytonaNotFoundError:
                        pass
                try:
                    await client.get(identifier, request_timeout=10)
                except DaytonaNotFoundError:
                    pass
                else:
                    raise HarborRecoveryError("Native Daytona deletion absence unconfirmed")
                if await owned_listing():
                    raise HarborRecoveryError(
                        "Native Daytona owner listing still contains resource"
                    )
                return record(
                    "resource.cleanup_confirmed", provider_id=identifier, source="recovery"
                )
        except BaseException as error:
            record("resource.cleanup_pending", source="recovery", error_type=type(error).__name__)
            raise
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
