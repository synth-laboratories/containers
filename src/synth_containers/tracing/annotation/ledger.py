"""Durable paid-compute ledger: one entry per paid job, written before the claim.

The entry is the *preparation intent* (so a crash after ``claim`` but before the
job record exists can be resumed with the same job id) and the *reconciliation
outbox* (so a terminal outcome is retried until the broker acknowledges it).
Every write is an atomic replace; nothing here is ever deleted.

Lifecycle of ``stage``::

    intent -> claimed -> prepared -> terminal -> acknowledged
                     \\-> abandoned (claim refused; no job)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from synth_containers.serde import JsonDataclassMixin

from ..canonical import utc_now
from ..validation.rehydrate import build
from .jobs import AnnotationJobRequestV1

LEDGER_SCHEMA_VERSION = "synth.paid-ledger-entry.v1"


@dataclass(frozen=True, slots=True)
class PaidLedgerEntryV1(JsonDataclassMixin):
    job_id: str
    reservation_id: str
    idempotency_key: str
    request: AnnotationJobRequestV1
    program_digest: str | None
    tool_contract_digest: str | None
    created_at: str
    session_id: str | None = None
    stage: str = "intent"
    cap_usd_micros: int | None = None
    outcome: str | None = None
    actual_cost_usd_micros: int | None = None
    terminal_at: str | None = None
    acknowledged_at: str | None = None
    reconcile_attempts: int = 0
    last_error: str | None = None
    last_attempt_at: str | None = None
    schema_version: str = LEDGER_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_reconciliation(self) -> bool:
        return self.stage == "terminal"

    @property
    def needs_recovery(self) -> bool:
        return self.stage in {"intent", "claimed"}


class PaidLedger:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, job_id: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in job_id)
        return self.root / f"{safe}.json"

    def write(self, entry: PaidLedgerEntryV1) -> PaidLedgerEntryV1:
        path = self._path(entry.job_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entry.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

        os.replace(tmp, path)
        return entry

    def get(self, job_id: str) -> PaidLedgerEntryV1 | None:
        path = self._path(job_id)
        if not path.exists():
            return None
        return build(PaidLedgerEntryV1, json.loads(path.read_text(encoding="utf-8")))

    def update(self, job_id: str, **changes: Any) -> PaidLedgerEntryV1:
        entry = self.get(job_id)
        if entry is None:
            raise KeyError(job_id)
        return self.write(replace(entry, **changes))

    def entries(self) -> tuple[PaidLedgerEntryV1, ...]:
        found = [build(PaidLedgerEntryV1, json.loads(path.read_text(encoding="utf-8"))) for path in sorted(self.root.glob("*.json"))]
        return tuple(sorted(found, key=lambda item: (item.created_at, item.job_id)))

    def pending_reconciliation(self) -> tuple[PaidLedgerEntryV1, ...]:
        return tuple(item for item in self.entries() if item.needs_reconciliation)

    def pending_recovery(self) -> tuple[PaidLedgerEntryV1, ...]:
        return tuple(item for item in self.entries() if item.needs_recovery)

    def mark_terminal(self, job_id: str, *, outcome: str, actual_cost_usd_micros: int | None) -> PaidLedgerEntryV1:
        entry = self.get(job_id)
        if entry is None:
            raise KeyError(job_id)
        if entry.stage in {"terminal", "acknowledged"}:
            if entry.outcome != outcome or entry.actual_cost_usd_micros != actual_cost_usd_micros:
                raise ValueError(
                    f"terminal outcome for {job_id} is immutable: "
                    f"{entry.outcome}/{entry.actual_cost_usd_micros} != "
                    f"{outcome}/{actual_cost_usd_micros}"
                )
            return entry
        return self.write(
            replace(
                entry,
                stage="terminal",
                outcome=outcome,
                actual_cost_usd_micros=actual_cost_usd_micros,
                terminal_at=utc_now(),
            )
        )

    def mark_acknowledged(self, job_id: str) -> PaidLedgerEntryV1:
        return self.update(job_id, stage="acknowledged", acknowledged_at=utc_now(), last_error=None)

    def record_attempt(self, job_id: str, error: str) -> PaidLedgerEntryV1:
        entry = self.get(job_id)
        if entry is None:
            raise KeyError(job_id)
        return self.write(replace(entry, reconcile_attempts=entry.reconcile_attempts + 1, last_error=error[:500], last_attempt_at=utc_now()))


__all__ = ["LEDGER_SCHEMA_VERSION", "PaidLedger", "PaidLedgerEntryV1"]
