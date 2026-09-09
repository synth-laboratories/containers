"""Container-side CISPO attempt lifecycle.

One accepted attempt reaches exactly one terminal result. This module owns the
machine that makes that true over the real runtime:

* **Idempotent submission.** A retry after a lost response resolves to the same
  logical attempt. The key is the caller's when it supplies one, and otherwise
  derives from the correlation slot (run, group, sample index, seed), because a
  slot is what a retry is retrying.
* **Two clocks, only one of which is a lease.** A heartbeat TTL keeps a grant
  alive; a straggler deadline fixed at grant decides when the attempt has run
  too long. Heartbeats never move the deadline, or a heartbeating straggler is
  immortal. The deadline covers the declared horizon times its dilation plus the
  quiescence and artifact-collection budgets plus the declared grace.
* **Quiescence before scoring.** ``finalize`` stops every policy-authored
  background process the runtime allowed the policy to create, and attests that
  nothing mutated the environment between the horizon and the scored read. A
  runtime that cannot quiesce returns a horizon-clipped state snapshot instead;
  that is a declared substitute, never a silent degradation.
* **An ordered event stream with a monotone resumable cursor**, which is the
  package's own :class:`~synth_containers.event_log.RolloutEventLog` rather than
  a second journal.

Correlation metadata is opaque. Every key the caller submits comes back byte for
byte, including keys this module has never heard of.

No task, harness, or environment name appears here.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from .event_log import RolloutEventLog, poll_payload, stream_descriptor, validate_rollout_id
from .platform.seal import seal_rollout_log
from .serde import JsonDataclassMixin

ATTEMPT_SCHEMA_VERSION = "cispo.rollout_attempt.v1"
LEASE_SCHEMA_VERSION = "cispo.rollout_lease.v1"
QUIESCENCE_SCHEMA_VERSION = "cispo.quiescence_attestation.v1"

#: Correlation keys the contract names. Unknown keys are preserved too; these
#: are simply the ones that get typed accessors.
DECLARED_CORRELATION_FIELDS: tuple[str, ...] = (
    "run_id",
    "group_id",
    "sample_index",
    "seed",
    "policy_revision",
    "agent_instance_id",
    "team_id",
    "policy_set_revision_id",
)

HORIZON_KINDS = frozenset({"wall_clock", "steps", "env_ticks"})
UNIT_HORIZON_KINDS = frozenset({"steps", "env_ticks"})

TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})

#: Which terminal result a status records. The contract names three kinds and
#: an attempt produces exactly one of them.
TERMINAL_KINDS: Mapping[str, str] = {
    "completed": "episode",
    "failed": "failure",
    "cancelled": "cancellation",
}

EVENT_ACCEPTED = "rollout.attempt.accepted"
EVENT_DUPLICATE = "rollout.attempt.duplicate_submission"
EVENT_STARTED = "rollout.attempt.started"
EVENT_LEASE_GRANTED = "rollout.lease.granted"
EVENT_LEASE_RENEWED = "rollout.lease.renewed"
EVENT_PROCESS_REGISTERED = "rollout.background_process.registered"
EVENT_PROCESS_QUIESCED = "rollout.background_process.quiesced"
EVENT_QUIESCENCE = "rollout.quiescence.attested"
EVENT_AWAITING_SCORE = "rollout.attempt.awaiting_score"
EVENT_FINALIZED = "rollout.attempt.finalized"
EVENT_TERMINAL = "rollout.attempt.terminal"


class LifecycleError(RuntimeError):
    """The attempt machine refused a transition. Never degrade this to a reward."""


class QuiescenceUnsupported(RuntimeError):
    """The runtime cannot stop what the policy started; clip to the horizon instead."""


def _digest(payload: Any, *, length: int = 32) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


# --------------------------------------------------------------------------- #
# Correlation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CorrelationV1(JsonDataclassMixin):
    """Opaque caller metadata, round-tripped untouched.

    The declared keys get accessors because leases and receipts read them. Every
    other key the caller sent is kept verbatim in ``extra`` and re-emitted with
    its original value, so a container that has never heard of a field cannot
    drop, rename, or coerce it.
    """

    values: dict[str, Any] = field(default_factory=dict)
    order: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "CorrelationV1":
        items = dict(payload or {})
        return cls(values=items, order=tuple(items.keys()))

    def to_dict(self) -> dict[str, Any]:
        """The exact object graph the caller submitted, in submission order."""

        return {key: self.values[key] for key in self.order}

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)

    @property
    def extra(self) -> dict[str, Any]:
        return {
            key: self.values[key]
            for key in self.order
            if key not in DECLARED_CORRELATION_FIELDS
        }

    @property
    def run_id(self) -> Any:
        return self.values.get("run_id")

    @property
    def group_id(self) -> Any:
        return self.values.get("group_id")

    @property
    def sample_index(self) -> Any:
        return self.values.get("sample_index")

    @property
    def seed(self) -> Any:
        return self.values.get("seed")

    @property
    def policy_revision(self) -> Any:
        return self.values.get("policy_revision")

    @property
    def agent_instance_id(self) -> Any:
        return self.values.get("agent_instance_id")

    @property
    def team_id(self) -> Any:
        return self.values.get("team_id")

    @property
    def policy_set_revision_id(self) -> Any:
        return self.values.get("policy_set_revision_id")

    @property
    def slot_key(self) -> str:
        """The identity a retry is retrying: run, group, sample index, seed."""

        return _digest(
            {
                "run_id": self.run_id,
                "group_id": self.group_id,
                "sample_index": self.sample_index,
                "seed": self.seed,
                "agent_instance_id": self.agent_instance_id,
            },
            length=24,
        )


# --------------------------------------------------------------------------- #
# Horizon and lease
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class HorizonSpecV1(JsonDataclassMixin):
    """A declared episode horizon. A unit horizon must declare its conversion."""

    horizon_kind: str
    value: float
    time_dilation: float = 1.0
    grace_seconds: float = 0.0
    seconds_per_unit: float | None = None
    quiescence_budget_seconds: float = 0.0
    artifact_budget_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.horizon_kind not in HORIZON_KINDS:
            raise LifecycleError(f"unknown horizon_kind {self.horizon_kind!r}")
        if self.value <= 0:
            raise LifecycleError("horizon value must be positive")
        if self.time_dilation <= 0:
            raise LifecycleError("time_dilation must be positive")
        if self.horizon_kind in UNIT_HORIZON_KINDS and not self.seconds_per_unit:
            # A steps or env_ticks horizon carries no duration. Defaulting to one
            # second per unit would silently declare a queue timeout nobody chose.
            raise LifecycleError(
                f"a {self.horizon_kind} horizon must declare seconds_per_unit"
            )

    @property
    def duration_seconds(self) -> float:
        """Wall-clock the declared horizon is worth, dilation included."""

        if self.horizon_kind == "wall_clock":
            return float(self.value) * float(self.time_dilation)
        return float(self.value) * float(self.seconds_per_unit or 0.0) * float(self.time_dilation)

    @property
    def straggler_budget_seconds(self) -> float:
        """Horizon plus the post-horizon work that is still inside the attempt."""

        return (
            self.duration_seconds
            + float(self.quiescence_budget_seconds)
            + float(self.artifact_budget_seconds)
            + float(self.grace_seconds)
        )


@dataclass(frozen=True, slots=True)
class LeaseGrantV1(JsonDataclassMixin):
    """One grant. ``expires_at`` moves on heartbeat; ``deadline_at`` never does."""

    lease_id: str
    rollout_id: str
    ttl_seconds: float
    granted_at: float
    expires_at: float
    deadline_at: float
    renewals: int = 0
    schema_version: str = LEASE_SCHEMA_VERSION

    def expired(self, now: float) -> bool:
        return now > self.expires_at

    def overdue(self, now: float) -> bool:
        """Past the straggler deadline fixed at grant, however many heartbeats."""

        return now > self.deadline_at

    def renewed(self, now: float) -> "LeaseGrantV1":
        if self.overdue(now):
            raise LifecycleError(f"lease {self.lease_id} is past its straggler deadline")
        if self.expired(now):
            raise LifecycleError(f"lease {self.lease_id} expired at {self.expires_at}")
        return replace(
            self,
            expires_at=now + self.ttl_seconds,
            renewals=self.renewals + 1,
        )


# --------------------------------------------------------------------------- #
# Attempt state
# --------------------------------------------------------------------------- #


class AttemptState(StrEnum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    #: The episode has stopped producing and the horizon has not been attested.
    #:
    #: Not a terminal state and not a scoring state: it is the answer to the one
    #: question a polling client asks, "is there anything left to wait for". An
    #: attempt that had no such answer could only ever report ``running``, and a
    #: client polling it would either spin forever or finalize an episode that
    #: was still going. Distinct from ``CispoRewardAuthority``'s deferred
    #: ``awaiting_score``, which is about a measure that is not yet *readable*
    #: after the horizon was attested and is capability-gated on its own.
    AWAITING_SCORE = "awaiting_score"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self.value in TERMINAL_STATUSES


@dataclass(frozen=True, slots=True)
class TerminalResultV1(JsonDataclassMixin):
    """The one terminal result an accepted attempt is allowed to reach."""

    rollout_id: str
    kind: str
    terminal_status: str
    reason: str = ""
    at_offset_seconds: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.terminal_status not in TERMINAL_STATUSES:
            raise LifecycleError(f"unknown terminal status {self.terminal_status!r}")
        if self.kind != TERMINAL_KINDS[self.terminal_status]:
            raise LifecycleError(
                f"terminal kind {self.kind!r} does not match status {self.terminal_status!r}"
            )


@dataclass(frozen=True, slots=True)
class QuiescenceAttestationV1(JsonDataclassMixin):
    """What the container can say about the interval before the scored read.

    Exactly one of ``quiesced`` and ``clipped`` is the reason this attestation
    exists. A runtime that stopped everything attests quiescence; one that cannot
    stop what the policy started clips its state to the horizon and says so.
    """

    rollout_id: str
    quiesced: bool
    clipped: bool
    horizon_offset_seconds: float
    attested_at_offset_seconds: float
    stopped_processes: tuple[str, ...] = ()
    residual_processes: tuple[str, ...] = ()
    snapshot_digest: str | None = None
    reason: str = ""
    schema_version: str = QUIESCENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.quiesced and not self.clipped:
            raise LifecycleError(
                "an attempt with neither a quiescence attestation nor a horizon-clipped "
                "snapshot may not be scored"
            )
        if self.clipped and not self.snapshot_digest:
            raise LifecycleError("a horizon-clipped attestation must name its snapshot")
        if self.quiesced and self.residual_processes:
            raise LifecycleError(
                "quiescence was attested while policy-authored processes are still live"
            )


@dataclass(frozen=True, slots=True)
class FinalizeOutcomeV1(JsonDataclassMixin):
    """What ``finalize`` hands to scoring, and nothing scoring may skip."""

    rollout_id: str
    attestation: QuiescenceAttestationV1
    terminal: TerminalResultV1
    scored_read_offset_seconds: float
    state_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AttemptRecordV1(JsonDataclassMixin):
    """The immutable public view of one accepted attempt."""

    rollout_id: str
    idempotency_key: str
    state: AttemptState
    correlation: CorrelationV1
    horizon: HorizonSpecV1
    lease: LeaseGrantV1
    accepted_at_offset_seconds: float
    stream_id: str
    terminal: TerminalResultV1 | None = None
    attestation: QuiescenceAttestationV1 | None = None
    replaced_attempt_id: str | None = None
    replacement_reason: str | None = None
    schema_version: str = ATTEMPT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        # slots=True replaces the class on Python 3.11; zero-argument super()
        # retains the original class cell and raises TypeError on this path.
        payload = JsonDataclassMixin.to_dict(self)
        # ``jsonable`` walks the dataclass; the correlation must leave exactly as
        # it arrived rather than as this module's storage shape.
        payload["correlation"] = self.correlation.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class SubmissionV1(JsonDataclassMixin):
    """A submitted attempt. ``duplicate`` says a retry resolved to an existing one."""

    attempt: AttemptRecordV1
    duplicate: bool = False


# --------------------------------------------------------------------------- #
# The runtime seam
# --------------------------------------------------------------------------- #


@runtime_checkable
class BackgroundProcess(Protocol):
    """A policy-authored process the container owns and kills at the horizon.

    ``synth_containers.platform.policy_process.IsolatedPolicyProcess`` satisfies
    this as written; so does any handle exposing ``close``, ``stop``, or
    ``terminate``.
    """

    def close(self) -> None: ...


@runtime_checkable
class RolloutRuntime(Protocol):
    """The real runtime under one attempt.

    ``start`` and ``poll`` are the async submission/poll pair; ``quiesce`` stops
    policy-authored work; ``horizon_snapshot`` is the declared substitute for a
    runtime that cannot. A runtime that raises :class:`QuiescenceUnsupported`
    from ``quiesce`` must return a snapshot.
    """

    def start(self, attempt: AttemptRecordV1, log: RolloutEventLog) -> None: ...

    def poll(self, attempt: AttemptRecordV1, log: RolloutEventLog) -> str | None: ...

    def quiesce(self, attempt: AttemptRecordV1) -> tuple[str, ...]: ...

    def horizon_snapshot(self, attempt: AttemptRecordV1) -> Mapping[str, Any]: ...

    def cancel(self, attempt: AttemptRecordV1, reason: str) -> None: ...


def _stop(handle: Any) -> None:
    for name in ("close", "stop", "terminate", "kill"):
        method = getattr(handle, name, None)
        if callable(method):
            method()
            return
    raise LifecycleError(f"background process {handle!r} exposes no way to stop it")


# --------------------------------------------------------------------------- #
# The lifecycle machine
# --------------------------------------------------------------------------- #


class CispoRolloutLifecycle:
    """Attempt lifecycle over one runtime, with an injected clock.

    Nothing here sleeps. ``clock`` is any monotone ``() -> float`` in seconds;
    the default is ``time.monotonic`` and tests supply their own.
    """

    def __init__(
        self,
        runtime: RolloutRuntime,
        *,
        lease_ttl_seconds: float = 300.0,
        lease_renewable: bool = True,
        max_concurrency: int = 8,
        clock: Callable[[], float] = time.monotonic,
        stream_transport: str = "sse",
    ) -> None:
        if lease_ttl_seconds <= 0:
            raise LifecycleError("lease_ttl_seconds must be positive")
        self._runtime = runtime
        self._lease_ttl_seconds = float(lease_ttl_seconds)
        self._lease_renewable = bool(lease_renewable)
        self._max_concurrency = int(max_concurrency)
        self._clock = clock
        self._stream_transport = stream_transport
        self._lock = threading.RLock()
        self._epoch = float(clock())
        self._attempts: dict[str, AttemptRecordV1] = {}
        self._by_key: dict[str, str] = {}
        self._logs: dict[str, RolloutEventLog] = {}
        self._processes: dict[str, list[tuple[str, Any]]] = {}
        self._quiesced_at: dict[str, float] = {}

    # -- clock ----------------------------------------------------------- #

    def offset(self) -> float:
        """Seconds since this lifecycle's epoch. The only time anything records."""

        return float(self._clock()) - self._epoch

    # -- introspection ---------------------------------------------------- #

    @property
    def advertised_concurrency(self) -> int:
        return self._max_concurrency

    @property
    def live_count(self) -> int:
        with self._lock:
            return sum(1 for item in self._attempts.values() if not item.state.terminal)

    def attempt(self, rollout_id: str) -> AttemptRecordV1:
        with self._lock:
            record = self._attempts.get(rollout_id)
        if record is None:
            raise LifecycleError(f"unknown attempt {rollout_id!r}")
        return record

    def log(self, rollout_id: str) -> RolloutEventLog:
        with self._lock:
            log = self._logs.get(rollout_id)
        if log is None:
            raise LifecycleError(f"unknown attempt {rollout_id!r}")
        return log

    def stream(self, rollout_id: str) -> dict[str, Any]:
        record = self.attempt(rollout_id)
        return stream_descriptor(
            rollout_id=record.rollout_id,
            stream_id=record.stream_id,
            bound_transport=self._stream_transport,
        )

    # -- submission ------------------------------------------------------- #

    def submit(
        self,
        *,
        rollout_id: str,
        horizon: HorizonSpecV1,
        correlation: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
        replaced_attempt_id: str | None = None,
        replacement_reason: str | None = None,
    ) -> SubmissionV1:
        """Accept an attempt, or resolve a retry to the one already accepted.

        A retry after a lost response carries the same idempotency key (or, with
        none supplied, the same correlation slot) and yields the same logical
        attempt: same rollout id, same lease, no second execution.
        """

        validate_rollout_id(rollout_id)
        slot = CorrelationV1.from_payload(correlation)
        key = str(idempotency_key or "").strip() or f"slot:{slot.slot_key}"
        with self._lock:
            existing_id = self._by_key.get(key)
            if existing_id is not None:
                existing = self._attempts[existing_id]
                log = self._logs[existing_id]
                if not log.closed:
                    log.append(
                        EVENT_DUPLICATE,
                        {
                            "rollout_id": existing.rollout_id,
                            "idempotency_key": key,
                            "resubmitted_rollout_id": rollout_id,
                            "state": existing.state.value,
                        },
                    )
                return SubmissionV1(attempt=existing, duplicate=True)
            if rollout_id in self._attempts:
                raise LifecycleError(
                    f"rollout_id {rollout_id!r} is already bound to idempotency key "
                    f"{self._attempts[rollout_id].idempotency_key!r}"
                )
            # Only admission may be refused. Executed work never is.
            live = sum(1 for item in self._attempts.values() if not item.state.terminal)
            if replaced_attempt_id is None and live >= self._max_concurrency:
                raise LifecycleError(
                    f"admission refused: {live} live attempts at concurrency "
                    f"{self._max_concurrency}"
                )

            now = self.offset()
            stream_id = f"stream_{_digest([rollout_id, key], length=16)}"
            log = RolloutEventLog(rollout_id=rollout_id, stream_id=stream_id)
            log.append_control(
                "stream.subscribed",
                log.subscribed_payload(),
            )
            lease = LeaseGrantV1(
                lease_id=f"lease_{_digest([rollout_id, key, now], length=16)}",
                rollout_id=rollout_id,
                ttl_seconds=self._lease_ttl_seconds,
                granted_at=now,
                expires_at=now + self._lease_ttl_seconds,
                deadline_at=now + horizon.straggler_budget_seconds,
            )
            record = AttemptRecordV1(
                rollout_id=rollout_id,
                idempotency_key=key,
                state=AttemptState.ACCEPTED,
                correlation=slot,
                horizon=horizon,
                lease=lease,
                accepted_at_offset_seconds=now,
                stream_id=stream_id,
                replaced_attempt_id=replaced_attempt_id,
                replacement_reason=replacement_reason,
            )
            self._attempts[rollout_id] = record
            self._by_key[key] = rollout_id
            self._logs[rollout_id] = log
            self._processes[rollout_id] = []
            log.append(
                EVENT_ACCEPTED,
                {
                    "rollout_id": rollout_id,
                    "idempotency_key": key,
                    "correlation": slot.to_dict(),
                    "horizon": horizon.to_dict(),
                    "replaced_attempt_id": replaced_attempt_id,
                    "replacement_reason": replacement_reason,
                },
            )
            log.append(EVENT_LEASE_GRANTED, lease.to_dict())
            return SubmissionV1(attempt=record, duplicate=False)

    # -- execution -------------------------------------------------------- #

    def start(self, rollout_id: str) -> AttemptRecordV1:
        with self._lock:
            record = self.attempt(rollout_id)
            if record.state is not AttemptState.ACCEPTED:
                raise LifecycleError(
                    f"attempt {rollout_id} cannot start from state {record.state.value}"
                )
            log = self._logs[rollout_id]
            record = replace(record, state=AttemptState.RUNNING)
            self._attempts[rollout_id] = record
            log.append(EVENT_STARTED, {"rollout_id": rollout_id, "at": self.offset()})
        self._runtime.start(record, self.log(rollout_id))
        return record

    def observe(self, rollout_id: str) -> AttemptRecordV1:
        """Read whether the episode is still producing. Settles nothing.

        This is what a polling client is actually asking, and it is deliberately
        weaker than :meth:`poll`: an attempt whose episode has stopped moves to
        ``awaiting_score``, which is not terminal, so the horizon is still
        attested by ``finalize`` and by nothing else. Reading state may not
        decide an attempt's one terminal result.
        """

        record = self.attempt(rollout_id)
        if record.state is not AttemptState.RUNNING:
            return record
        status = self._runtime.poll(record, self.log(rollout_id))
        if status != "completed":
            return record
        with self._lock:
            record = self.attempt(rollout_id)
            if record.state is not AttemptState.RUNNING:
                return record
            record = replace(record, state=AttemptState.AWAITING_SCORE)
            self._attempts[rollout_id] = record
            self._logs[rollout_id].append(
                EVENT_AWAITING_SCORE,
                {"rollout_id": rollout_id, "at": self.offset()},
            )
        return record

    def poll(self, rollout_id: str) -> AttemptRecordV1:
        """Ask the runtime whether the attempt has reached a terminal status."""

        record = self.attempt(rollout_id)
        if record.state.terminal:
            return record
        status = self._runtime.poll(record, self.log(rollout_id))
        if status is None:
            return record
        if status not in TERMINAL_STATUSES:
            raise LifecycleError(f"runtime returned non-terminal status {status!r}")
        if status == "completed":
            self.finalize(rollout_id)
            return self.attempt(rollout_id)
        return self._settle(rollout_id, terminal_status=status, reason="runtime")

    def register_background_process(
        self,
        rollout_id: str,
        handle: Any,
        *,
        process_id: str | None = None,
    ) -> str:
        """Declare a policy-authored process this attempt owns and must kill."""

        record = self.attempt(rollout_id)
        if record.state.terminal:
            raise LifecycleError(
                f"attempt {rollout_id} is terminal; it can adopt no further processes"
            )
        with self._lock:
            handles = self._processes.setdefault(rollout_id, [])
            name = process_id or f"proc_{len(handles)}"
            handles.append((name, handle))
        self.log(rollout_id).append(
            EVENT_PROCESS_REGISTERED,
            {"rollout_id": rollout_id, "process_id": name, "at": self.offset()},
        )
        return name

    # -- leases ----------------------------------------------------------- #

    def renew(self, rollout_id: str) -> LeaseGrantV1:
        """Heartbeat. Extends the grant; never moves the straggler deadline."""

        with self._lock:
            record = self.attempt(rollout_id)
            if record.state.terminal:
                raise LifecycleError(f"attempt {rollout_id} is terminal; nothing to renew")
            if not self._lease_renewable:
                raise LifecycleError(
                    f"attempt {rollout_id} holds a non-renewable lease shorter than its horizon"
                )
            lease = record.lease.renewed(self.offset())
            self._attempts[rollout_id] = replace(record, lease=lease)
            self._logs[rollout_id].append(EVENT_LEASE_RENEWED, lease.to_dict())
            return lease

    def overdue_attempts(self) -> tuple[str, ...]:
        """Attempts past the deadline fixed at grant, heartbeats notwithstanding."""

        now = self.offset()
        with self._lock:
            return tuple(
                rollout_id
                for rollout_id, record in self._attempts.items()
                if not record.state.terminal and record.lease.overdue(now)
            )

    # -- cancellation ----------------------------------------------------- #

    def cancel(self, rollout_id: str, *, reason: str = "terminate") -> AttemptRecordV1:
        record = self.attempt(rollout_id)
        if record.state.terminal:
            raise LifecycleError(
                f"attempt {rollout_id} already reached {record.state.value}; "
                "an accepted attempt has exactly one terminal result"
            )
        self._runtime.cancel(record, reason)
        self._quiesce_processes(rollout_id)
        return self._settle(rollout_id, terminal_status="cancelled", reason=reason)

    def fail(self, rollout_id: str, *, reason: str) -> AttemptRecordV1:
        self._quiesce_processes(rollout_id)
        return self._settle(rollout_id, terminal_status="failed", reason=reason)

    # -- finalize --------------------------------------------------------- #

    def finalize(self, rollout_id: str) -> FinalizeOutcomeV1:
        """Quiesce, attest, then settle. Scoring may not run before this returns.

        Post-horizon quiescence and artifact collection are inside the attempt's
        lease, not work done after the attempt is considered complete.
        """

        record = self.attempt(rollout_id)
        if record.state.terminal:
            raise LifecycleError(
                f"attempt {rollout_id} already reached {record.state.value}; "
                "an accepted attempt has exactly one terminal result"
            )
        with self._lock:
            record = replace(record, state=AttemptState.FINALIZING)
            self._attempts[rollout_id] = record
        log = self._logs[rollout_id]

        horizon_offset = record.accepted_at_offset_seconds + record.horizon.duration_seconds
        stopped = self._quiesce_processes(rollout_id)
        snapshot: dict[str, Any] = {}
        try:
            residual = tuple(self._runtime.quiesce(record))
        except QuiescenceUnsupported as exc:
            snapshot = dict(self._runtime.horizon_snapshot(record))
            attestation = QuiescenceAttestationV1(
                rollout_id=rollout_id,
                quiesced=False,
                clipped=True,
                horizon_offset_seconds=horizon_offset,
                attested_at_offset_seconds=self.offset(),
                stopped_processes=stopped,
                residual_processes=("runtime",),
                snapshot_digest=_digest(snapshot),
                reason=str(exc) or "runtime cannot quiesce; state clipped to the horizon",
            )
        else:
            if residual:
                snapshot = dict(self._runtime.horizon_snapshot(record))
                attestation = QuiescenceAttestationV1(
                    rollout_id=rollout_id,
                    quiesced=False,
                    clipped=True,
                    horizon_offset_seconds=horizon_offset,
                    attested_at_offset_seconds=self.offset(),
                    stopped_processes=stopped,
                    residual_processes=residual,
                    snapshot_digest=_digest(snapshot),
                    reason="policy-authored work outlived the horizon; state clipped",
                )
            else:
                attestation = QuiescenceAttestationV1(
                    rollout_id=rollout_id,
                    quiesced=True,
                    clipped=False,
                    horizon_offset_seconds=horizon_offset,
                    attested_at_offset_seconds=self.offset(),
                    stopped_processes=stopped,
                    reason="every policy-authored process stopped before the scored read",
                )
        log.append(EVENT_QUIESCENCE, attestation.to_dict())

        scored_read = self.offset()
        terminal = self._settle_locked_terminal(
            rollout_id,
            terminal_status="completed",
            reason="horizon",
            attestation=attestation,
        )
        log.append(
            EVENT_FINALIZED,
            {
                "rollout_id": rollout_id,
                "scored_read_offset_seconds": scored_read,
                "quiesced": attestation.quiesced,
                "clipped": attestation.clipped,
            },
        )
        return FinalizeOutcomeV1(
            rollout_id=rollout_id,
            attestation=attestation,
            terminal=terminal,
            scored_read_offset_seconds=scored_read,
            state_snapshot=snapshot,
        )

    def quiesced_at(self, rollout_id: str) -> float | None:
        """When the last policy-authored process stopped. ``None`` if none ran."""

        return self._quiesced_at.get(rollout_id)

    def _quiesce_processes(self, rollout_id: str) -> tuple[str, ...]:
        with self._lock:
            handles = list(self._processes.get(rollout_id) or ())
            self._processes[rollout_id] = []
        if not handles:
            return ()
        log = self._logs[rollout_id]
        stopped: list[str] = []
        for name, handle in handles:
            _stop(handle)
            stopped.append(name)
            log.append(
                EVENT_PROCESS_QUIESCED,
                {"rollout_id": rollout_id, "process_id": name, "at": self.offset()},
            )
        self._quiesced_at[rollout_id] = self.offset()
        return tuple(stopped)

    # -- terminal settlement ---------------------------------------------- #

    def _settle(
        self,
        rollout_id: str,
        *,
        terminal_status: str,
        reason: str,
    ) -> AttemptRecordV1:
        self._settle_locked_terminal(
            rollout_id, terminal_status=terminal_status, reason=reason, attestation=None
        )
        return self.attempt(rollout_id)

    def _settle_locked_terminal(
        self,
        rollout_id: str,
        *,
        terminal_status: str,
        reason: str,
        attestation: QuiescenceAttestationV1 | None,
    ) -> TerminalResultV1:
        with self._lock:
            record = self._attempts[rollout_id]
            if record.terminal is not None:
                raise LifecycleError(
                    f"attempt {rollout_id} already reached {record.terminal.terminal_status}; "
                    "an accepted attempt has exactly one terminal result"
                )
            terminal = TerminalResultV1(
                rollout_id=rollout_id,
                kind=TERMINAL_KINDS[terminal_status],
                terminal_status=terminal_status,
                reason=reason,
                at_offset_seconds=self.offset(),
            )
            self._attempts[rollout_id] = replace(
                record,
                state=AttemptState(terminal_status),
                terminal=terminal,
                attestation=attestation or record.attestation,
            )
            self._logs[rollout_id].append(EVENT_TERMINAL, terminal.to_dict())
            return terminal

    # -- events ----------------------------------------------------------- #

    def events(self, rollout_id: str, *, after: int = 0, limit: int = 1000) -> dict[str, Any]:
        """A page of the ordered stream, resumable from any cursor.

        The cursor is the log sequence: strictly increasing, gap-free, and
        ``cursor.next`` from one page is the ``after`` of the next.
        """

        return poll_payload(self.log(rollout_id), after=after, limit=limit)

    def replay(self, rollout_id: str, *, after: int = 0) -> tuple[dict[str, Any], ...]:
        log = self.log(rollout_id)
        return tuple(item.to_dict() for item in log.after(after))

    # -- release ---------------------------------------------------------- #

    def seal(self, rollout_id: str, *, pin: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Close the lifecycle stream and seal it with the platform's own sealer."""

        record = self.attempt(rollout_id)
        if not record.state.terminal:
            raise LifecycleError(f"attempt {rollout_id} is not terminal; nothing to seal")
        log = self.log(rollout_id)
        log.mark_closed()
        return seal_rollout_log(log, pin=dict(pin or {}))

# --------------------------------------------------------------------------- #
# The one coupling point to the route layer
# --------------------------------------------------------------------------- #


class CispoRolloutAdapter:
    """The stream-C implementation of :class:`~.cispo_contract.CispoRolloutPort`.

    Everything the HTTP layer needs is here and nowhere else, so the lifecycle
    machine above never learns what a request looks like. The trace and reward
    halves are supplied as callables rather than imported, which keeps the
    evidence and reward modules free of any dependency on this one.

    An operation this container does not serve raises
    :class:`~.cispo_contract.CispoNotImplementedError`, which the route layer
    renders as a typed 501 -- never as an empty success.
    """

    def __init__(
        self,
        lifecycle: CispoRolloutLifecycle,
        *,
        evidence_source: Callable[[str], Any] | None = None,
        reward_source: Callable[[], Any] | None = None,
        artifact_source: Callable[[str], Any] | None = None,
    ) -> None:
        self._lifecycle = lifecycle
        self._evidence_source = evidence_source
        self._reward_source = reward_source
        self._artifact_source = artifact_source

    @property
    def lifecycle(self) -> CispoRolloutLifecycle:
        return self._lifecycle

    # -- submission ------------------------------------------------------- #

    def cispo_submit_rollout(self, request: Mapping[str, Any]) -> dict[str, Any]:
        submission = self._lifecycle.submit(
            rollout_id=str(request["rollout_id"]),
            horizon=horizon_from_payload(request.get("horizon")),
            correlation=request.get("correlation"),
            idempotency_key=request.get("idempotency_key"),
            replaced_attempt_id=request.get("replaced_attempt_id"),
            replacement_reason=request.get("replacement_reason"),
        )
        attempt = submission.attempt
        return {
            "rollout_id": attempt.rollout_id,
            "state": attempt.state.value,
            "idempotency_key": attempt.idempotency_key,
            "duplicate": submission.duplicate,
            "lease_expires_at": attempt.lease.expires_at,
            "lease_deadline_at": attempt.lease.deadline_at,
            "correlation": attempt.correlation.to_dict(),
            "handshake_id": request.get("handshake_id"),
            "stream": self._lifecycle.stream(attempt.rollout_id),
        }

    # -- polling ---------------------------------------------------------- #

    def cispo_rollout_state(self, rollout_id: str) -> dict[str, Any]:
        """Cheap enough to poll: it computes no reward and seals no trace."""

        # Reading the runtime's own progress is what makes this route answerable:
        # it computes no reward, seals no trace, and settles no terminal result.
        attempt = self._lifecycle.observe(rollout_id)
        now = self._lifecycle.offset()
        return {
            "rollout_id": attempt.rollout_id,
            "state": attempt.state.value,
            "terminal": None if attempt.terminal is None else attempt.terminal.to_dict(),
            "lease_expires_at": attempt.lease.expires_at,
            "lease_deadline_at": attempt.lease.deadline_at,
            "lease_expired": attempt.lease.expired(now),
            "overdue": attempt.lease.overdue(now),
            "correlation": attempt.correlation.to_dict(),
            "instances": [
                {
                    "agent_instance_id": attempt.correlation.agent_instance_id,
                    "team_id": attempt.correlation.team_id,
                    "live": not attempt.state.terminal,
                    "last_event_cursor": str(self._lifecycle.log(rollout_id).high_water),
                }
            ],
        }

    def cispo_rollout_events(
        self,
        rollout_id: str,
        *,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        after = 0 if cursor is None or cursor == "" else int(cursor)
        page = self._lifecycle.events(rollout_id, after=after, limit=limit or 1000)
        rows = []
        for row in page["events"]:
            sequence = row.get("sequence")
            rows.append({**row, "cursor": None if sequence is None else str(sequence)})
        return {
            "rollout_id": rollout_id,
            "events": rows,
            "next_cursor": str(page["cursor"]["next"]),
            "high_water_cursor": str(page["cursor"]["high_water"]),
            "closed": page["cursor"]["closed"],
            "has_more": page["cursor"]["has_more"],
        }

    # -- lease ------------------------------------------------------------ #

    def cispo_renew_lease(
        self, rollout_id: str, request: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        lease = self._lifecycle.renew(rollout_id)
        return {
            "rollout_id": rollout_id,
            "lease_id": lease.lease_id,
            "lease_expires_at": lease.expires_at,
            "lease_deadline_at": lease.deadline_at,
            "renewals": lease.renewals,
            "handshake_id": (request or {}).get("handshake_id"),
        }

    # -- terminal --------------------------------------------------------- #

    def cispo_finalize_rollout(
        self, rollout_id: str, request: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        outcome = self._lifecycle.finalize(rollout_id)
        attempt = self._lifecycle.attempt(rollout_id)
        return {
            "rollout_id": rollout_id,
            "state": attempt.state.value,
            "terminal": outcome.terminal.to_dict(),
            "horizon": attempt.horizon.to_dict(),
            "horizon_applied_seconds": attempt.horizon.duration_seconds,
            "scored_read_offset_seconds": outcome.scored_read_offset_seconds,
            "quiescence": outcome.attestation.to_dict(),
            "state_snapshot": dict(outcome.state_snapshot),
            "correlation": attempt.correlation.to_dict(),
        }

    def cispo_terminate_rollout(
        self, rollout_id: str, request: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Cancel exactly once. A second call reports, it does not re-terminate."""

        attempt = self._lifecycle.attempt(rollout_id)
        if attempt.state.terminal:
            assert attempt.terminal is not None
            return {
                "rollout_id": rollout_id,
                "state": attempt.state.value,
                "terminal": attempt.terminal.to_dict(),
                "already_terminal": True,
            }
        reason = str((request or {}).get("reason") or "terminate")
        record = self._lifecycle.cancel(rollout_id, reason=reason)
        assert record.terminal is not None
        return {
            "rollout_id": rollout_id,
            "state": record.state.value,
            "terminal": record.terminal.to_dict(),
            "already_terminal": False,
        }

    # -- evidence and reward ---------------------------------------------- #

    def cispo_rollout_trace(self, rollout_id: str) -> dict[str, Any]:
        evidence = self._require(self._evidence_source, "cispo_rollout_trace")(rollout_id)
        if evidence is None:
            raise LifecycleError(f"attempt {rollout_id} has sealed no trace")
        payload = evidence.to_dict()
        payload["trace_digest"] = evidence.trace_digest
        payload["document"] = evidence.document.to_dict()
        # The trace is sealed by construction -- ``SealedEvidenceV1`` is what the
        # evidence source returns -- and a reader that cannot see the seal has to
        # guess whether the evidence is complete. Say it.
        payload["sealed"] = True
        payload["inline"] = True
        return payload

    def cispo_rollout_artifacts(self, rollout_id: str) -> dict[str, Any]:
        source = self._require(self._artifact_source, "cispo_rollout_artifacts")
        return {"rollout_id": rollout_id, "artifacts": list(source(rollout_id))}

    def cispo_reward(self, request: Mapping[str, Any]) -> dict[str, Any]:
        authority = self._require(self._reward_source, "cispo_reward")()
        rollout_id = str(request["rollout_id"])
        attempt = self._lifecycle.attempt(rollout_id)
        if attempt.attestation is None:
            raise LifecycleError(
                f"attempt {rollout_id} has not been finalized; scoring before the horizon "
                "attestation is what post-horizon reward drift looks like"
            )
        slot = None
        try:
            slot = authority.slot(rollout_id)
        except Exception:  # noqa: BLE001 - no slot yet is not an error
            slot = None
        if slot is not None and slot.receipt is not None:
            return slot.receipt.to_dict()
        receipt = authority.settle(
            rollout_id=rollout_id,
            trace_digest=str(request.get("trace_digest") or ""),
            terminal_status=(
                attempt.terminal.terminal_status if attempt.terminal else "completed"
            ),
            attestation=attempt.attestation,
            scored_at_offset_seconds=float(
                request.get("scored_at_offset_seconds") or self._lifecycle.offset()
            ),
            horizon_kind=attempt.horizon.horizon_kind,
            horizon_value=attempt.horizon.value,
            measures=request.get("measures"),
            measure=request.get("measure"),
            optimized_team_id=request.get("optimized_team_id"),
            metadata=request.get("metadata"),
            log=self._lifecycle.log(rollout_id),
        )
        return receipt.to_dict()

    # -- advertisement ---------------------------------------------------- #

    def advertised_capabilities(self) -> dict[str, Any]:
        return {
            "schema_version": ATTEMPT_SCHEMA_VERSION,
            "max_concurrency": self._lifecycle.advertised_concurrency,
            "idempotent_submission": True,
            "cancellation": True,
            "lease_renewal": True,
            "correlation_fields": list(DECLARED_CORRELATION_FIELDS),
            "correlation_passthrough": "verbatim",
            "trace": self._evidence_source is not None,
            "reward": self._reward_source is not None,
            "artifacts": self._artifact_source is not None,
        }

    @staticmethod
    def _require(handler: Any, operation: str) -> Any:
        if handler is None:
            from .cispo_contract import CispoNotImplementedError

            raise CispoNotImplementedError(
                "CispoRolloutPort",
                operation,
                "this container serves no source for that half of the attempt",
            )
        return handler


def horizon_from_payload(payload: Mapping[str, Any] | None) -> HorizonSpecV1:
    """Decode a declared horizon. A unit horizon without a conversion is refused."""

    values = dict(payload or {})
    return HorizonSpecV1(
        horizon_kind=str(values.get("horizon_kind") or "wall_clock"),
        value=float(values.get("value") or 0.0),
        time_dilation=float(values.get("time_dilation") or 1.0),
        grace_seconds=float(values.get("grace_seconds") or 0.0),
        seconds_per_unit=(
            None
            if values.get("seconds_per_unit") is None
            else float(values["seconds_per_unit"])
        ),
        quiescence_budget_seconds=float(values.get("quiescence_budget_seconds") or 0.0),
        artifact_budget_seconds=float(values.get("artifact_budget_seconds") or 0.0),
    )


__all__ = [
    "ATTEMPT_SCHEMA_VERSION",
    "DECLARED_CORRELATION_FIELDS",
    "HORIZON_KINDS",
    "TERMINAL_KINDS",
    "TERMINAL_STATUSES",
    "AttemptRecordV1",
    "AttemptState",
    "BackgroundProcess",
    "CispoRolloutAdapter",
    "CispoRolloutLifecycle",
    "CorrelationV1",
    "FinalizeOutcomeV1",
    "HorizonSpecV1",
    "LeaseGrantV1",
    "LifecycleError",
    "QuiescenceAttestationV1",
    "QuiescenceUnsupported",
    "RolloutRuntime",
    "SubmissionV1",
    "TerminalResultV1",
    "horizon_from_payload",
]
