"""The reward receipt: bound to a sealed trace, absent is not zero, deferred is declared."""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest

from synth_containers.cispo_reward import (
    ABSENT_CHANNEL,
    CispoRewardAdapter,
    CispoRewardAuthority,
    HorizonEvidenceV1,
    RewardError,
    ScoringState,
    resolve_ranks,
)
from synth_containers.cispo_rollout import QuiescenceAttestationV1
from synth_containers.event_log import RolloutEventLog
from synth_containers.platform.reward import REWARD_CLOSED, REWARD_OPENED, REWARD_SIGNAL

TRACE_DIGEST = "sha256:" + "a" * 64

QUIESCED = QuiescenceAttestationV1(
    rollout_id="attempt-1",
    quiesced=True,
    clipped=False,
    horizon_offset_seconds=60.0,
    attested_at_offset_seconds=60.5,
    stopped_processes=("loop-0",),
)

CLIPPED = QuiescenceAttestationV1(
    rollout_id="attempt-1",
    quiesced=False,
    clipped=True,
    horizon_offset_seconds=60.0,
    attested_at_offset_seconds=60.5,
    snapshot_digest="sha256:snapshot",
    reason="this runtime cannot stop what the policy started",
)


def _authority(**kwargs: Any) -> CispoRewardAuthority:
    kwargs.setdefault("evaluation_plan_id", "plan.container.v3")
    return CispoRewardAuthority(**kwargs)


def _settle(authority: CispoRewardAuthority, **kwargs: Any) -> Any:
    payload: dict[str, Any] = {
        "rollout_id": "attempt-1",
        "trace_digest": TRACE_DIGEST,
        "terminal_status": "completed",
        "attestation": QUIESCED,
        "scored_at_offset_seconds": 61.0,
        "horizon_kind": "wall_clock",
        "horizon_value": 60.0,
        "measure": 0.75,
    }
    payload.update(kwargs)
    return authority.settle(**payload)


# --------------------------------------------------------------------------- #


def test_a_reward_is_bound_to_the_rollout_and_the_sealed_trace_digest() -> None:
    receipt = _settle(_authority())
    receipt.validate(episode_trace_digest=TRACE_DIGEST)
    assert receipt.rollout_id == "attempt-1"
    assert receipt.trace_digest == TRACE_DIGEST
    assert receipt.evaluation_plan_id == "plan.container.v3"
    assert receipt.value() == 0.75

    with pytest.raises(RewardError, match="does not match its episode"):
        receipt.validate(episode_trace_digest="sha256:" + "b" * 64)


def test_a_reward_without_a_trace_digest_is_refused_at_the_source() -> None:
    with pytest.raises(RewardError, match="no sealed trace digest"):
        _settle(_authority(), trace_digest="")


def test_an_attestation_for_another_attempt_is_refused() -> None:
    other = QuiescenceAttestationV1(
        rollout_id="attempt-2",
        quiesced=True,
        clipped=False,
        horizon_offset_seconds=60.0,
        attested_at_offset_seconds=60.5,
    )
    with pytest.raises(RewardError, match="names 'attempt-2'"):
        _settle(_authority(), attestation=other)


def test_zero_is_a_measure_and_absent_is_not() -> None:
    zero = _settle(_authority(), measure=0.0)
    zero.validate()
    assert zero.absent is False
    assert zero.value() == 0.0
    assert zero.channels[0].measure == 0.0

    absent = _settle(_authority(), measure=None)
    assert absent.absent is True
    assert absent.channels == ()
    assert absent.optimized_channel == ABSENT_CHANNEL
    assert "missing_evidence" in absent.metadata
    with pytest.raises(RewardError, match="absent is not zero"):
        absent.validate()

    # And the two are distinguishable on the wire, not only in Python.
    assert zero.to_dict()["channels"] == [
        {"channel_id": "score", "team_id": None, "measure": 0.0, "rank": None}
    ]
    assert absent.to_dict()["channels"] == []


def test_final_scoring_is_idempotent() -> None:
    authority = _authority()
    first = _settle(authority)
    # A second read of the same episode must not produce a second reward.
    second = _settle(authority, measure=0.99)
    assert second is first
    assert second.value() == 0.75
    assert authority.state("attempt-1") is ScoringState.COMPLETED


def test_deferred_scoring_reaches_completed() -> None:
    authority = _authority(deferred_scoring=True)
    assert authority.state("attempt-1") is ScoringState.RUNNING
    authority.open("attempt-1")
    assert authority.state("attempt-1") is ScoringState.RUNNING

    slot = authority.defer("attempt-1", reason="the verifier runs against a retained workspace")
    assert slot.state is ScoringState.AWAITING_SCORE
    assert authority.state("attempt-1") is ScoringState.AWAITING_SCORE

    adapter = CispoRewardAdapter(authority)
    pending = adapter.reward_get("attempt-1")
    assert pending["reward"] is None
    assert pending["scoring_state"] == "awaiting_score"

    receipt = _settle(authority, measure=1.0)
    assert authority.state("attempt-1") is ScoringState.COMPLETED
    assert receipt.scoring_state is ScoringState.COMPLETED
    receipt.validate(episode_trace_digest=TRACE_DIGEST)
    assert receipt.metadata["deferred_scoring"] is True
    assert adapter.reward_get("attempt-1")["scoring_state"] == "completed"


def test_deferred_scoring_is_refused_when_it_was_never_advertised() -> None:
    authority = _authority()
    with pytest.raises(RewardError, match="never advertised as a capability"):
        authority.defer("attempt-1", reason="the verifier is slow")


def test_an_unscored_receipt_is_not_a_reward() -> None:
    receipt = _settle(_authority())
    unscored = type(receipt)(
        **{
            **{
                field: getattr(receipt, field)
                for field in (
                    "reward_id",
                    "rollout_id",
                    "trace_digest",
                    "channels",
                    "optimized_channel",
                    "terminal_status",
                    "evaluation_plan_id",
                    "horizon",
                    "reward_relation",
                    "metadata",
                )
            },
            "scoring_state": ScoringState.AWAITING_SCORE,
        }
    )
    with pytest.raises(RewardError, match="is not a reward"):
        unscored.validate()


# --------------------------------------------------------------------------- #
# Horizon evidence
# --------------------------------------------------------------------------- #


def test_horizon_evidence_is_the_attestations_and_not_the_scorers() -> None:
    receipt = _settle(_authority())
    assert receipt.horizon is not None
    assert receipt.horizon.quiescence_attested is True
    assert receipt.horizon.clipped is False
    assert receipt.horizon.scored_at_offset_seconds == 61.0
    assert receipt.horizon.horizon_value == 60.0
    receipt.horizon.validate()

    clipped = _settle(_authority(), attestation=CLIPPED)
    assert clipped.horizon is not None
    assert clipped.horizon.quiescence_attested is False
    assert clipped.horizon.clipped is True
    clipped.horizon.validate()


def test_a_reward_with_neither_attestation_nor_clipping_is_refused() -> None:
    evidence = HorizonEvidenceV1(
        horizon_kind="wall_clock",
        horizon_value=60.0,
        scored_at_offset_seconds=61.0,
        clipped=False,
        quiescence_attested=False,
    )
    with pytest.raises(RewardError, match="neither a quiescence attestation nor"):
        evidence.validate()


def test_credited_settlement_never_exceeds_the_declared_window() -> None:
    authority = _authority(settlement_window_seconds=2.0)
    # Scored ten seconds past the horizon under a two-second declared window.
    receipt = _settle(authority, scored_at_offset_seconds=70.0)
    assert receipt.horizon is not None
    assert receipt.horizon.settlement_window_seconds == 2.0
    assert receipt.horizon.credited_settlement_seconds == 2.0
    receipt.horizon.validate()

    over = HorizonEvidenceV1(
        horizon_kind="wall_clock",
        horizon_value=60.0,
        scored_at_offset_seconds=70.0,
        clipped=False,
        quiescence_attested=True,
        settlement_window_seconds=2.0,
        credited_settlement_seconds=9.0,
    )
    with pytest.raises(RewardError, match="beyond its declared"):
        over.validate()


# --------------------------------------------------------------------------- #
# Competitive relations
# --------------------------------------------------------------------------- #


def test_competitive_channels_carry_the_measure_and_the_resolved_rank() -> None:
    authority = _authority(reward_relation="competitive_rank")
    receipt = _settle(
        authority,
        measure=None,
        measures={"team-a": 4.0, "team-b": 9.0, "team-c": 4.0},
        optimized_team_id="team-a",
    )
    receipt.validate(episode_trace_digest=TRACE_DIGEST)
    assert receipt.optimized_channel == "score::team-a"
    ranks = {channel.team_id: channel.rank for channel in receipt.channels}
    assert ranks == {"team-b": 1, "team-a": 2, "team-c": 2}, "a declared tie policy, not a guess"
    assert receipt.value_for_team("team-b") == 9.0
    assert receipt.metadata["tie_policy"] == "shared_rank"


def test_dense_and_stable_tie_policies_are_declared_choices() -> None:
    measures = {"team-a": 4.0, "team-b": 9.0, "team-c": 4.0}
    shared = {c.team_id: c.rank for c in resolve_ranks(measures, tie_policy="shared_rank")}
    dense = {c.team_id: c.rank for c in resolve_ranks(measures, tie_policy="dense_rank")}
    stable = {c.team_id: c.rank for c in resolve_ranks(measures, tie_policy="stable_order")}
    assert shared == {"team-b": 1, "team-a": 2, "team-c": 2}
    assert dense == {"team-b": 1, "team-a": 2, "team-c": 2}
    assert stable == {"team-b": 1, "team-a": 2, "team-c": 3}
    with pytest.raises(RewardError, match="unknown tie policy"):
        resolve_ranks(measures, tie_policy="whatever_ties")


def test_a_competitive_channel_without_a_rank_is_refused() -> None:
    authority = _authority(reward_relation="competitive_margin")
    receipt = _settle(authority, measure=None, measures={"team-a": 1.0, "team-b": 2.0})
    receipt.validate()
    unranked = type(receipt)(
        reward_id=receipt.reward_id,
        rollout_id=receipt.rollout_id,
        trace_digest=receipt.trace_digest,
        channels=tuple(
            type(channel)(
                channel_id=channel.channel_id,
                team_id=channel.team_id,
                measure=channel.measure,
                rank=None,
            )
            for channel in receipt.channels
        ),
        optimized_channel=receipt.optimized_channel,
        terminal_status="completed",
        evaluation_plan_id=receipt.evaluation_plan_id,
        reward_relation="competitive_margin",
    )
    with pytest.raises(RewardError, match="resolved no rank"):
        unranked.validate()


def test_a_team_measure_must_be_unambiguous() -> None:
    receipt = _settle(
        _authority(reward_relation="cooperative"),
        measure=None,
        measures={"team-a": 1.0, "team-b": 2.0},
    )
    assert receipt.value_for_team("team-a") == 1.0
    with pytest.raises(RewardError, match="no channel for team"):
        receipt.value_for_team("team-z")


def test_the_optimized_channel_must_be_one_the_receipt_carries() -> None:
    with pytest.raises(RewardError, match="no channel belongs to the optimized team"):
        _settle(
            _authority(reward_relation="competitive_rank"),
            measure=None,
            measures={"team-a": 1.0},
            optimized_team_id="team-z",
        )


# --------------------------------------------------------------------------- #
# The platform's own reward stream
# --------------------------------------------------------------------------- #


def test_scoring_writes_the_platforms_reward_events_and_never_coerces_absent_to_zero() -> None:
    log = RolloutEventLog(rollout_id="attempt-1", stream_id="stream-1")
    authority = _authority()
    authority.open("attempt-1", log=log)
    _settle(authority, measure=0.0, log=log)
    kinds = [item.kind for item in log.after(0)]
    assert REWARD_OPENED in kinds
    assert kinds.count(REWARD_SIGNAL) == 1
    assert REWARD_CLOSED in kinds
    signal = next(item for item in log.after(0) if item.kind == REWARD_SIGNAL)
    assert signal.payload["value"] == 0.0
    assert signal.payload["trace_digest"] == TRACE_DIGEST

    absent_log = RolloutEventLog(rollout_id="attempt-2", stream_id="stream-2")
    absent_authority = _authority()
    absent_authority.settle(
        rollout_id="attempt-2",
        trace_digest=TRACE_DIGEST,
        terminal_status="completed",
        attestation=QuiescenceAttestationV1(
            rollout_id="attempt-2",
            quiesced=True,
            clipped=False,
            horizon_offset_seconds=60.0,
            attested_at_offset_seconds=60.5,
        ),
        scored_at_offset_seconds=61.0,
        horizon_kind="wall_clock",
        horizon_value=60.0,
        measure=None,
        log=absent_log,
    )
    absent_signal = next(item for item in absent_log.after(0) if item.kind == REWARD_SIGNAL)
    assert absent_signal.payload["value"] is None, "missing stays null; never coerce to 0"


def test_the_evaluation_plan_identity_is_stable_and_classified() -> None:
    from synth_containers.platform.reward_plan import PlanOutcome

    scored = _settle(_authority(evaluation_plan_id="plan.container.v3"))
    assert scored.plan_outcome is PlanOutcome.SCORED
    gated = _settle(_authority(evaluation_plan_id="plan.container.gated"))
    assert gated.plan_outcome is PlanOutcome.GATED


def test_the_adapter_scores_and_reads_back() -> None:
    adapter = CispoRewardAdapter(_authority())
    payload = adapter.reward(
        {
            "rollout_id": "attempt-1",
            "trace_digest": TRACE_DIGEST,
            "terminal_status": "completed",
            "scored_at_offset_seconds": 61.0,
            "horizon_kind": "wall_clock",
            "horizon_value": 60.0,
            "measure": 0.5,
        },
        attestation=QUIESCED,
    )
    assert payload["trace_digest"] == TRACE_DIGEST
    assert payload["horizon"]["quiescence_attested"] is True
    assert adapter.reward_get("attempt-1") == payload
    assert adapter.advertised_capabilities()["absent_is_not_zero"] is True


# --------------------------------------------------------------------------- #
# The optimizer's own validators
# --------------------------------------------------------------------------- #


def _optimizer_records() -> Any:
    """Import the training plane's record contract if it is reachable from here."""

    root = os.environ.get("SYNTH_OPTIMIZERS_SRC")
    if root and root not in sys.path:
        sys.path.insert(0, root)
    try:
        from synth_optimizers.contracts import rl_records
    except ImportError:
        return None
    return rl_records


def test_the_receipt_passes_the_optimizer_reward_validators() -> None:
    receipt = _settle(
        _authority(reward_relation="competitive_rank", settlement_window_seconds=2.0),
        measure=None,
        measures={"team-a": 4.0, "team-b": 9.0},
        optimized_team_id="team-a",
        scored_at_offset_seconds=61.0,
    )
    payload = receipt.to_dict()
    records = _optimizer_records()

    if records is None:
        # synth-optimizers is not importable in this worktree, so the same
        # properties are asserted directly against the emitted payload.
        receipt.validate(episode_trace_digest=TRACE_DIGEST)
        assert payload["rollout_id"] and payload["trace_digest"]
        assert payload["terminal_status"] in {"completed", "failed", "cancelled"}
        assert payload["optimized_channel"] in {
            row["channel_id"] for row in payload["channels"]
        }
        assert payload["evaluation_plan_id"]
        for row in payload["channels"]:
            assert row["measure"] == row["measure"]  # finite, not NaN
            assert row["rank"] is not None
        horizon = payload["horizon"]
        assert horizon["quiescence_attested"] or horizon["clipped"]
        assert (
            horizon["credited_settlement_seconds"] <= horizon["settlement_window_seconds"]
        )
        return

    horizon = payload["horizon"]
    theirs = records.RewardRecord(
        reward_id=payload["reward_id"],
        rollout_id=payload["rollout_id"],
        trace_digest=payload["trace_digest"],
        channels=tuple(
            records.RewardChannel(
                channel_id=row["channel_id"],
                team_id=row["team_id"],
                measure=row["measure"],
                rank=row["rank"],
            )
            for row in payload["channels"]
        ),
        optimized_channel=payload["optimized_channel"],
        terminal_status=payload["terminal_status"],
        evaluation_plan_id=payload["evaluation_plan_id"],
        horizon=records.HorizonEvidence(
            horizon_kind=horizon["horizon_kind"],
            horizon_value=horizon["horizon_value"],
            scored_at_offset_seconds=horizon["scored_at_offset_seconds"],
            clipped=horizon["clipped"],
            quiescence_attested=horizon["quiescence_attested"],
            settlement_window_seconds=horizon["settlement_window_seconds"],
            credited_settlement_seconds=horizon["credited_settlement_seconds"],
        ),
        metadata=payload["metadata"],
    )
    theirs.validate(episode_trace_digest=TRACE_DIGEST)
    assert theirs.value() == 4.0
    assert theirs.value_for_team("team-b") == 9.0


# --------------------------------------------------------------------------- #
# The three streams meeting
# --------------------------------------------------------------------------- #


def test_an_attempt_finalizes_seals_and_scores_in_that_order() -> None:
    from tests.test_cispo_evidence import _tool_loop
    from tests.test_cispo_rollout import CORRELATION, HORIZON, Clock, FakeRuntime, RecordingProcess

    from synth_containers.cispo_rollout import CispoRolloutLifecycle

    clock = Clock()
    lifecycle = CispoRolloutLifecycle(FakeRuntime(clock=clock), clock=clock)
    lifecycle.submit(rollout_id="attempt-1", horizon=HORIZON, correlation=CORRELATION)
    lifecycle.start("attempt-1")
    loop = RecordingProcess(clock)
    lifecycle.register_background_process("attempt-1", loop, process_id="loop-0")
    clock.advance(60.0)

    outcome = lifecycle.finalize("attempt-1")
    evidence = _tool_loop().seal(correlation=CORRELATION)
    evidence.validate()

    authority = _authority()
    receipt = authority.settle(
        rollout_id="attempt-1",
        trace_digest=evidence.trace_digest,
        terminal_status=outcome.terminal.terminal_status,
        attestation=outcome.attestation,
        scored_at_offset_seconds=outcome.scored_read_offset_seconds,
        horizon_kind=HORIZON.horizon_kind,
        horizon_value=HORIZON.value,
        measure=0.5,
        log=lifecycle.log("attempt-1"),
    )
    receipt.validate(episode_trace_digest=evidence.episodes[0].trace_digest)
    assert loop.stopped_at is not None
    assert loop.stopped_at < receipt.horizon.scored_at_offset_seconds
    assert receipt.horizon.quiescence_attested is True


def test_the_rollout_port_serves_trace_and_reward_from_one_attempt() -> None:
    from tests.test_cispo_evidence import _tool_loop
    from tests.test_cispo_rollout import CORRELATION, HORIZON, Clock, FakeRuntime

    from synth_containers.cispo_rollout import CispoRolloutAdapter, CispoRolloutLifecycle

    clock = Clock()
    lifecycle = CispoRolloutLifecycle(FakeRuntime(clock=clock), clock=clock)
    authority = _authority()
    sealed: dict[str, Any] = {}
    adapter = CispoRolloutAdapter(
        lifecycle,
        evidence_source=sealed.get,
        reward_source=lambda: authority,
        artifact_source=lambda rollout_id: [
            {
                "artifact_id": "art-0",
                "content_digest": "sha256:artifact",
                "byte_size": 12,
                "media_type": "application/json",
                "fetch": f"/rollouts/{rollout_id}/artifacts/art-0",
            }
        ],
    )
    adapter.cispo_submit_rollout(
        {
            "rollout_id": "attempt-1",
            "horizon": {"horizon_kind": "wall_clock", "value": 60.0, "grace_seconds": 5.0},
            "correlation": CORRELATION,
        }
    )
    lifecycle.start("attempt-1")
    clock.advance(60.0)
    finalized = adapter.cispo_finalize_rollout("attempt-1", {})
    sealed["attempt-1"] = _tool_loop().seal(correlation=CORRELATION)

    trace = adapter.cispo_rollout_trace("attempt-1")
    assert trace["trace_digest"] == sealed["attempt-1"].trace_digest
    assert trace["document"]["content_digest"] == trace["trace_digest"]

    receipt = adapter.cispo_reward(
        {
            "rollout_id": "attempt-1",
            "trace_digest": trace["trace_digest"],
            "measure": 0.25,
            "scored_at_offset_seconds": finalized["scored_read_offset_seconds"],
        }
    )
    assert receipt["trace_digest"] == trace["trace_digest"]
    assert receipt["horizon"]["quiescence_attested"] is True
    # A second POST scores nothing new.
    assert adapter.cispo_reward({"rollout_id": "attempt-1", "measure": 9.0}) == receipt
    assert adapter.cispo_rollout_artifacts("attempt-1")["artifacts"][0]["artifact_id"] == "art-0"
    assert HORIZON.horizon_kind == "wall_clock"
