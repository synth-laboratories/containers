"""The attempt lifecycle: one terminal result, a resumable cursor, quiescence first.

Every clock in here is injected. Nothing sleeps, and an hour-scale attempt costs
the same as a one-second one.
"""

from __future__ import annotations

from typing import Any

import pytest

from synth_containers.cispo_rollout import (
    EVENT_FINALIZED,
    EVENT_PROCESS_QUIESCED,
    EVENT_QUIESCENCE,
    AttemptRecordV1,
    AttemptState,
    CispoRolloutAdapter,
    CispoRolloutLifecycle,
    HorizonSpecV1,
    LifecycleError,
    QuiescenceUnsupported,
)
from synth_containers.event_log import RolloutEventLog


class Clock:
    """An injected monotone clock. The lifecycle reads it and never sleeps."""

    def __init__(self, start: float = 0.0) -> None:
        self.value = float(start)

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError("a monotone clock cannot go backwards")
        self.value += float(seconds)
        return self.value


class FakeRuntime:
    """A runtime that records what the lifecycle asked it to do."""

    def __init__(
        self,
        *,
        clock: Clock,
        quiesces: bool = True,
        residual: tuple[str, ...] = (),
        quiesce_cost_seconds: float = 0.5,
        terminal_after_polls: int | None = None,
    ) -> None:
        self.clock = clock
        self.quiesces = quiesces
        self.residual = residual
        self.quiesce_cost_seconds = quiesce_cost_seconds
        self.terminal_after_polls = terminal_after_polls
        self.starts: list[str] = []
        self.cancels: list[tuple[str, str]] = []
        self.snapshots = 0
        self._polls = 0

    def start(self, attempt: AttemptRecordV1, log: RolloutEventLog) -> None:
        self.starts.append(attempt.rollout_id)
        log.append("runtime.started", {"rollout_id": attempt.rollout_id})

    def poll(self, attempt: AttemptRecordV1, log: RolloutEventLog) -> str | None:
        self._polls += 1
        log.append("runtime.polled", {"poll": self._polls})
        if self.terminal_after_polls is None or self._polls < self.terminal_after_polls:
            return None
        return "completed"

    def quiesce(self, attempt: AttemptRecordV1) -> tuple[str, ...]:
        self.clock.advance(self.quiesce_cost_seconds)
        if not self.quiesces:
            raise QuiescenceUnsupported("this runtime cannot stop what the policy started")
        return self.residual

    def horizon_snapshot(self, attempt: AttemptRecordV1) -> dict[str, Any]:
        self.snapshots += 1
        return {"clipped_at": attempt.horizon.value, "cells": [1, 2, 3]}

    def cancel(self, attempt: AttemptRecordV1, reason: str) -> None:
        self.cancels.append((attempt.rollout_id, reason))


class RecordingProcess:
    """A policy-authored background process the container owns and must kill."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.stopped_at: float | None = None

    def close(self) -> None:
        self.stopped_at = self.clock.value


HORIZON = HorizonSpecV1(horizon_kind="wall_clock", value=60.0, grace_seconds=5.0)

CORRELATION = {
    "run_id": "run-a",
    "group_id": "group-a",
    "sample_index": 3,
    "seed": 11,
    "policy_revision": 17,
    "agent_instance_id": "instance-a",
    "team_id": "team-a",
    "policy_set_revision_id": "set-20",
}


def _lifecycle(**kwargs: Any) -> tuple[CispoRolloutLifecycle, FakeRuntime, Clock]:
    clock = Clock()
    runtime = FakeRuntime(clock=clock, **kwargs)
    lifecycle = CispoRolloutLifecycle(runtime, clock=clock, lease_ttl_seconds=300.0)
    return lifecycle, runtime, clock


# --------------------------------------------------------------------------- #


def test_idempotent_retry_yields_one_attempt() -> None:
    lifecycle, runtime, _clock = _lifecycle()
    first = lifecycle.submit(
        rollout_id="attempt-1", horizon=HORIZON, correlation=CORRELATION, idempotency_key="k-1"
    )
    # The caller never saw the response and retried under the same key.
    second = lifecycle.submit(
        rollout_id="attempt-1", horizon=HORIZON, correlation=CORRELATION, idempotency_key="k-1"
    )
    assert first.duplicate is False
    assert second.duplicate is True
    assert second.attempt.rollout_id == first.attempt.rollout_id
    assert second.attempt.lease.lease_id == first.attempt.lease.lease_id
    assert lifecycle.live_count == 1

    lifecycle.start("attempt-1")
    assert runtime.starts == ["attempt-1"], "a retry must not execute a second attempt"


def test_a_retry_without_a_key_resolves_by_correlation_slot() -> None:
    lifecycle, _runtime, _clock = _lifecycle()
    first = lifecycle.submit(rollout_id="attempt-1", horizon=HORIZON, correlation=CORRELATION)
    # A different rollout id for the same slot is still the same logical attempt.
    second = lifecycle.submit(rollout_id="attempt-2", horizon=HORIZON, correlation=CORRELATION)
    assert second.duplicate is True
    assert second.attempt.rollout_id == first.attempt.rollout_id == "attempt-1"


def test_exactly_one_terminal_result_per_accepted_attempt() -> None:
    lifecycle, _runtime, _clock = _lifecycle()
    lifecycle.submit(rollout_id="attempt-1", horizon=HORIZON, correlation=CORRELATION)
    lifecycle.start("attempt-1")
    outcome = lifecycle.finalize("attempt-1")
    assert outcome.terminal.kind == "episode"
    assert lifecycle.attempt("attempt-1").state is AttemptState.COMPLETED

    with pytest.raises(LifecycleError, match="exactly one terminal result"):
        lifecycle.finalize("attempt-1")
    with pytest.raises(LifecycleError, match="exactly one terminal result"):
        lifecycle.cancel("attempt-1", reason="too late")


def test_cancellation_is_a_terminal_result_of_its_own() -> None:
    lifecycle, runtime, _clock = _lifecycle()
    lifecycle.submit(rollout_id="attempt-1", horizon=HORIZON, correlation=CORRELATION)
    lifecycle.start("attempt-1")
    record = lifecycle.cancel("attempt-1", reason="drain")
    assert record.state is AttemptState.CANCELLED
    assert record.terminal is not None
    assert record.terminal.kind == "cancellation"
    assert record.terminal.reason == "drain"
    assert runtime.cancels == [("attempt-1", "drain")]


def test_event_cursor_is_monotone_and_resumable_from_any_point() -> None:
    lifecycle, _runtime, clock = _lifecycle()
    lifecycle.submit(rollout_id="attempt-1", horizon=HORIZON, correlation=CORRELATION)
    lifecycle.start("attempt-1")
    for _ in range(4):
        clock.advance(10.0)
        lifecycle.renew("attempt-1")
    lifecycle.finalize("attempt-1")

    whole = lifecycle.events("attempt-1", after=0)
    sequences = [row["sequence"] for row in whole["events"] if "sequence" in row]
    assert sequences == list(range(1, len(sequences) + 1))
    assert whole["cursor"]["high_water"] == sequences[-1]

    # Resuming from every cursor reassembles exactly the same stream.
    for resume in range(0, sequences[-1] + 1):
        page = lifecycle.events("attempt-1", after=resume)
        tail = [row["sequence"] for row in page["events"] if "sequence" in row]
        assert tail == [item for item in sequences if item > resume]
        assert page["cursor"]["after"] == resume

    # And paging through in one-event steps loses nothing.
    cursor = 0
    walked: list[int] = []
    while True:
        page = lifecycle.events("attempt-1", after=cursor, limit=1)
        rows = [row["sequence"] for row in page["events"] if "sequence" in row]
        if not rows:
            break
        walked.extend(rows)
        cursor = page["cursor"]["next"]
    assert walked == sequences


def test_lease_renews_across_an_hour_scale_attempt() -> None:
    """A ninety-minute episode under a five-minute TTL, on an injected clock."""

    clock = Clock()
    runtime = FakeRuntime(clock=clock)
    horizon = HorizonSpecV1(
        horizon_kind="wall_clock",
        value=1800.0,
        time_dilation=3.0,  # 5400 seconds of wall clock
        grace_seconds=120.0,
        quiescence_budget_seconds=60.0,
    )
    lifecycle = CispoRolloutLifecycle(runtime, clock=clock, lease_ttl_seconds=300.0)
    submission = lifecycle.submit(
        rollout_id="attempt-long", horizon=horizon, correlation=CORRELATION
    )
    granted_deadline = submission.attempt.lease.deadline_at
    assert granted_deadline == pytest.approx(5400.0 + 60.0 + 120.0)
    lifecycle.start("attempt-long")

    renewals = 0
    while clock.value < 5400.0:
        clock.advance(240.0)
        lease = lifecycle.renew("attempt-long")
        renewals += 1
        assert lease.expired(clock.value) is False
        # Heartbeats never move the deadline, or a heartbeating straggler is immortal.
        assert lease.deadline_at == granted_deadline
    assert renewals >= 22

    assert lifecycle.overdue_attempts() == ()
    outcome = lifecycle.finalize("attempt-long")
    assert outcome.terminal.terminal_status == "completed"


def test_a_lease_left_unrenewed_expires_and_cannot_be_renewed() -> None:
    lifecycle, _runtime, clock = _lifecycle()
    long_horizon = HorizonSpecV1(horizon_kind="wall_clock", value=3600.0, grace_seconds=60.0)
    lifecycle.submit(rollout_id="attempt-1", horizon=long_horizon, correlation=CORRELATION)
    # Inside the straggler deadline, but past the heartbeat TTL: the two clocks
    # fail for different reasons and say so.
    clock.advance(301.0)
    with pytest.raises(LifecycleError, match="expired"):
        lifecycle.renew("attempt-1")


def test_a_heartbeating_straggler_is_still_a_straggler() -> None:
    clock = Clock()
    runtime = FakeRuntime(clock=clock)
    lifecycle = CispoRolloutLifecycle(runtime, clock=clock, lease_ttl_seconds=300.0)
    lifecycle.submit(rollout_id="attempt-1", horizon=HORIZON, correlation=CORRELATION)
    for _ in range(2):
        clock.advance(30.0)
        lease = lifecycle.renew("attempt-1")
        assert lease.expired(clock.value) is False
    # Heartbeats kept the grant alive well past the deadline fixed at grant.
    clock.advance(30.0)
    assert lifecycle.overdue_attempts() == ("attempt-1",)
    with pytest.raises(LifecycleError, match="straggler deadline"):
        lifecycle.renew("attempt-1")


def test_a_unit_horizon_must_declare_its_conversion() -> None:
    with pytest.raises(LifecycleError, match="seconds_per_unit"):
        HorizonSpecV1(horizon_kind="steps", value=200.0)
    steps = HorizonSpecV1(horizon_kind="steps", value=200.0, seconds_per_unit=0.25)
    assert steps.duration_seconds == pytest.approx(50.0)


def test_finalize_quiesces_every_policy_authored_process_before_scoring() -> None:
    lifecycle, runtime, clock = _lifecycle()
    lifecycle.submit(rollout_id="attempt-1", horizon=HORIZON, correlation=CORRELATION)
    lifecycle.start("attempt-1")
    loops = [RecordingProcess(clock) for _ in range(3)]
    for index, loop in enumerate(loops):
        lifecycle.register_background_process("attempt-1", loop, process_id=f"loop-{index}")
    clock.advance(60.0)

    outcome = lifecycle.finalize("attempt-1")

    assert all(loop.stopped_at is not None for loop in loops)
    for loop in loops:
        assert loop.stopped_at is not None
        assert loop.stopped_at < outcome.scored_read_offset_seconds, (
            "a program still running at the scored read is extra reward, not evidence"
        )
    assert outcome.attestation.quiesced is True
    assert outcome.attestation.clipped is False
    assert outcome.attestation.stopped_processes == ("loop-0", "loop-1", "loop-2")
    assert runtime.snapshots == 0

    kinds = [row["kind"] for row in lifecycle.replay("attempt-1") if "sequence" in row]
    assert kinds.index(EVENT_PROCESS_QUIESCED) < kinds.index(EVENT_QUIESCENCE)
    assert kinds.index(EVENT_QUIESCENCE) < kinds.index(EVENT_FINALIZED)


def test_a_runtime_that_cannot_quiesce_returns_a_horizon_clipped_snapshot() -> None:
    lifecycle, runtime, clock = _lifecycle(quiesces=False)
    lifecycle.submit(rollout_id="attempt-1", horizon=HORIZON, correlation=CORRELATION)
    lifecycle.start("attempt-1")
    clock.advance(60.0)

    outcome = lifecycle.finalize("attempt-1")

    assert outcome.attestation.quiesced is False
    assert outcome.attestation.clipped is True
    assert outcome.attestation.snapshot_digest
    assert outcome.state_snapshot["clipped_at"] == HORIZON.value
    assert runtime.snapshots == 1


def test_policy_work_outliving_the_horizon_clips_rather_than_claiming_quiescence() -> None:
    lifecycle, _runtime, clock = _lifecycle(residual=("standing-loop",))
    lifecycle.submit(rollout_id="attempt-1", horizon=HORIZON, correlation=CORRELATION)
    lifecycle.start("attempt-1")
    clock.advance(60.0)
    outcome = lifecycle.finalize("attempt-1")
    assert outcome.attestation.quiesced is False
    assert outcome.attestation.clipped is True
    assert outcome.attestation.residual_processes == ("standing-loop",)


def test_correlation_round_trips_untouched() -> None:
    lifecycle, _runtime, _clock = _lifecycle()
    submitted = {
        **CORRELATION,
        # Keys this container has never heard of are still the caller's.
        "match_set_revision_id": "match-set-0007",
        "policy_set_revision": "set-20",
        "opaque": {"nested": [1, 2, {"deep": True}]},
        "zero": 0,
        "false": False,
        "empty": "",
    }
    submission = lifecycle.submit(
        rollout_id="attempt-1", horizon=HORIZON, correlation=submitted
    )
    record = submission.attempt
    assert record.correlation.to_dict() == submitted
    assert list(record.correlation.to_dict()) == list(submitted), "key order preserved"
    assert record.to_dict()["correlation"] == submitted

    # It survives the terminal boundary and the event stream too.
    lifecycle.start("attempt-1")
    lifecycle.finalize("attempt-1")
    assert lifecycle.attempt("attempt-1").correlation.to_dict() == submitted
    accepted = next(
        row
        for row in lifecycle.replay("attempt-1")
        if row["kind"] == "rollout.attempt.accepted"
    )
    assert accepted["payload"]["correlation"] == submitted

    # The typed accessors read the same values without owning them.
    assert record.correlation.sample_index == 3
    assert record.correlation.policy_set_revision_id == "set-20"
    assert record.correlation.extra["match_set_revision_id"] == "match-set-0007"


def test_admission_is_refused_but_executed_work_never_is() -> None:
    clock = Clock()
    runtime = FakeRuntime(clock=clock)
    lifecycle = CispoRolloutLifecycle(runtime, clock=clock, max_concurrency=2)
    lifecycle.submit(rollout_id="a-1", horizon=HORIZON, correlation={"sample_index": 1})
    lifecycle.submit(rollout_id="a-2", horizon=HORIZON, correlation={"sample_index": 2})
    with pytest.raises(LifecycleError, match="admission refused"):
        lifecycle.submit(rollout_id="a-3", horizon=HORIZON, correlation={"sample_index": 3})
    # A straggler replacement re-enters work that already left the pipeline and
    # therefore bypasses the admission bound.
    replacement = lifecycle.submit(
        rollout_id="a-3",
        horizon=HORIZON,
        correlation={"sample_index": 3},
        replaced_attempt_id="a-1",
        replacement_reason="straggler",
    )
    assert replacement.attempt.replaced_attempt_id == "a-1"


def test_polling_reaches_a_terminal_result_without_holding_a_request_open() -> None:
    lifecycle, _runtime, _clock = _lifecycle(terminal_after_polls=3)
    lifecycle.submit(rollout_id="attempt-1", horizon=HORIZON, correlation=CORRELATION)
    lifecycle.start("attempt-1")
    assert lifecycle.poll("attempt-1").state is AttemptState.RUNNING
    assert lifecycle.poll("attempt-1").state is AttemptState.RUNNING
    assert lifecycle.poll("attempt-1").state is AttemptState.COMPLETED
    assert lifecycle.poll("attempt-1").state is AttemptState.COMPLETED


def test_the_adapter_serves_the_declared_rollout_port() -> None:
    import inspect

    from synth_containers.cispo_contract import CispoNotImplementedError, CispoRolloutPort

    lifecycle, _runtime, _clock = _lifecycle()
    adapter = CispoRolloutAdapter(lifecycle)
    declared = [
        name
        for name, _ in inspect.getmembers(CispoRolloutPort, inspect.isfunction)
        if name.startswith("cispo_")
    ]
    assert declared, "the port declares something"
    for name in declared:
        assert callable(getattr(adapter, name)), name

    created = adapter.cispo_submit_rollout(
        {
            "rollout_id": "attempt-1",
            "handshake_id": "hs-1",
            "horizon": {"horizon_kind": "env_ticks", "value": 400, "seconds_per_unit": 0.1},
            "correlation": CORRELATION,
            "idempotency_key": "k-1",
        }
    )
    assert created["duplicate"] is False
    assert created["correlation"] == CORRELATION
    assert created["lease_expires_at"] == 300.0
    again = adapter.cispo_submit_rollout(
        {
            "rollout_id": "attempt-1",
            "horizon": {"horizon_kind": "env_ticks", "value": 400, "seconds_per_unit": 0.1},
            "correlation": CORRELATION,
            "idempotency_key": "k-1",
        }
    )
    assert again["duplicate"] is True

    lifecycle.start("attempt-1")
    state = adapter.cispo_rollout_state("attempt-1")
    assert state["state"] == "running"
    assert state["instances"][0]["agent_instance_id"] == "instance-a"

    page = adapter.cispo_rollout_events("attempt-1", cursor=None, limit=2)
    assert page["next_cursor"] == "2"
    tail = adapter.cispo_rollout_events("attempt-1", cursor=page["next_cursor"])
    assert all(int(row["cursor"]) > 2 for row in tail["events"] if row.get("cursor"))

    assert adapter.cispo_renew_lease("attempt-1", {"handshake_id": "hs-1"})["renewals"] == 1
    finalized = adapter.cispo_finalize_rollout("attempt-1", {})
    assert finalized["quiescence"]["quiesced"] is True
    assert finalized["horizon_applied_seconds"] == pytest.approx(40.0)

    # Terminating an already-terminal attempt reports; it never re-terminates.
    terminated = adapter.cispo_terminate_rollout("attempt-1", {"reason": "drain"})
    assert terminated["already_terminal"] is True
    assert terminated["terminal"]["kind"] == "episode"

    # A half this container does not serve is a typed 501, never an empty success.
    with pytest.raises(CispoNotImplementedError):
        adapter.cispo_rollout_artifacts("attempt-1")
    with pytest.raises(CispoNotImplementedError):
        adapter.cispo_reward({"rollout_id": "attempt-1"})

    assert adapter.advertised_capabilities()["idempotent_submission"] is True
    assert adapter.advertised_capabilities()["artifacts"] is False


def test_the_adapter_refuses_to_score_before_the_horizon_attestation() -> None:
    from synth_containers.cispo_reward import CispoRewardAuthority

    lifecycle, _runtime, _clock = _lifecycle()
    authority = CispoRewardAuthority(evaluation_plan_id="plan.container.v3")
    adapter = CispoRolloutAdapter(lifecycle, reward_source=lambda: authority)
    lifecycle.submit(rollout_id="attempt-1", horizon=HORIZON, correlation=CORRELATION)
    lifecycle.start("attempt-1")
    with pytest.raises(LifecycleError, match="has not been finalized"):
        adapter.cispo_reward({"rollout_id": "attempt-1", "measure": 1.0})


def test_the_lifecycle_stream_seals_with_the_platform_sealer() -> None:
    from synth_containers.platform.seal import validate_rollout_seal

    lifecycle, _runtime, _clock = _lifecycle()
    lifecycle.submit(rollout_id="attempt-1", horizon=HORIZON, correlation=CORRELATION)
    lifecycle.start("attempt-1")
    lifecycle.finalize("attempt-1")
    seal = lifecycle.seal("attempt-1", pin={"group_id": "group-a"})
    validate_rollout_seal(seal)
    assert seal["rollout_id"] == "attempt-1"
    assert seal["pin"] == {"group_id": "group-a"}
