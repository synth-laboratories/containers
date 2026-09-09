"""Container-side CISPO reward authority: the receipt and its horizon evidence.

A reward is a claim about an attempt, so it is bound to the exact rollout id and
to the sealed trace digest that attempt produced. Everything else follows from
that binding:

* **Zero is distinguishable from absent.** A measure of ``0.0`` is a channel
  that scored zero. An absent reward carries no channel at all, names the
  missing evidence, and refuses validation. Missing evidence is a terminal
  failure, never a zero-reward trajectory.
* **Per-team channels plus the resolved rank** for a competitive relation, with
  a declared tie policy, and the optimized channel named in the receipt.
* **A stable evaluation-plan identity**, so two reads of the same episode
  cannot silently come from two plans.
* **Horizon evidence**: the horizon, the actual time of the scored read, whether
  clipping was applied, which post-horizon settlement was credited, and the
  quiescence attestation. Environment activity after the horizon never reaches
  the reward.
* **Deferred scoring as a declared capability**: ``running → awaiting_score →
  completed``, never inferred and never open-ended. Final scoring is idempotent.

The reward-calculation event stream is the package's own
:class:`~synth_containers.platform.reward.RewardStreamer`, and the plan-outcome
classifier is :func:`~synth_containers.platform.reward_plan.classify_plan_outcome`;
this module adds the receipt, not a second event vocabulary.

No task, harness, or environment name appears here.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .cispo_rollout import QuiescenceAttestationV1
from .event_log import RolloutEventLog
from .platform.reward import RewardStreamer
from .platform.reward_plan import PlanOutcome, classify_plan_outcome
from .serde import JsonDataclassMixin

REWARD_RECEIPT_SCHEMA_VERSION = "cispo.reward_record.v1"

TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})

REWARD_RELATIONS = frozenset(
    {"cooperative", "competitive_rank", "competitive_margin", "mixed"}
)
COMPETITIVE_RELATIONS = frozenset({"competitive_rank", "competitive_margin"})

TIE_POLICIES = frozenset({"shared_rank", "dense_rank", "stable_order"})

#: The one channel id an absent reward carries in its ``optimized_channel``.
#: It names no channel, which is what makes ``validate`` refuse it.
ABSENT_CHANNEL = "absent"


class RewardError(ValueError):
    """The reward receipt was incomplete, unbound, or internally inconsistent."""


class ScoringState(StrEnum):
    """Where scoring is. ``awaiting_score`` exists only where it was declared."""

    RUNNING = "running"
    AWAITING_SCORE = "awaiting_score"
    COMPLETED = "completed"


# --------------------------------------------------------------------------- #
# Channels
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RewardChannelV1(JsonDataclassMixin):
    """One team's measure. Absolute and rank are both recorded."""

    channel_id: str
    team_id: str | None
    measure: float
    rank: int | None = None

    def __post_init__(self) -> None:
        if math.isnan(self.measure) or math.isinf(self.measure):
            raise RewardError(f"reward channel {self.channel_id} measure is not finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "team_id": self.team_id,
            "measure": self.measure,
            "rank": self.rank,
        }


def resolve_ranks(
    measures: Mapping[str, float],
    *,
    tie_policy: str = "shared_rank",
    channel_prefix: str = "score",
) -> tuple[RewardChannelV1, ...]:
    """Resolve a competitive ordering into per-team channels with ranks.

    The ordering is declared, and so is the tie policy: ``shared_rank`` gives
    tied teams the same rank and skips the next, ``dense_rank`` gives them the
    same rank and does not skip, and ``stable_order`` refuses to call it a tie
    and breaks by team id.
    """

    if tie_policy not in TIE_POLICIES:
        raise RewardError(f"unknown tie policy {tie_policy!r}")
    for team_id, measure in measures.items():
        if math.isnan(measure) or math.isinf(measure):
            raise RewardError(f"team {team_id} measure is not finite")
    ordered = sorted(measures.items(), key=lambda item: (-item[1], item[0]))
    channels: list[RewardChannelV1] = []
    rank = 0
    dense = 0
    previous: float | None = None
    for position, (team_id, measure) in enumerate(ordered, start=1):
        if tie_policy == "stable_order" or previous is None or measure != previous:
            rank = position
            dense += 1
        previous = measure
        channels.append(
            RewardChannelV1(
                channel_id=f"{channel_prefix}::{team_id}",
                team_id=team_id,
                measure=measure,
                rank=dense if tie_policy == "dense_rank" else rank,
            )
        )
    return tuple(channels)


# --------------------------------------------------------------------------- #
# Horizon evidence
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class HorizonEvidenceV1(JsonDataclassMixin):
    """When the reward was read, and whether the environment was still moving."""

    horizon_kind: str
    horizon_value: float
    scored_at_offset_seconds: float
    clipped: bool
    quiescence_attested: bool
    settlement_window_seconds: float = 0.0
    credited_settlement_seconds: float = 0.0

    def validate(self) -> None:
        if not self.quiescence_attested and not self.clipped:
            raise RewardError(
                "reward has neither a quiescence attestation nor a horizon-clipped snapshot"
            )
        if self.credited_settlement_seconds > self.settlement_window_seconds:
            raise RewardError(
                f"reward credited {self.credited_settlement_seconds}s of settlement beyond "
                f"its declared {self.settlement_window_seconds}s window"
            )
        if self.credited_settlement_seconds < 0 or self.scored_at_offset_seconds < 0:
            raise RewardError("settlement and scored-read offsets must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon_kind": self.horizon_kind,
            "horizon_value": self.horizon_value,
            "scored_at_offset_seconds": self.scored_at_offset_seconds,
            "clipped": self.clipped,
            "quiescence_attested": self.quiescence_attested,
            "settlement_window_seconds": self.settlement_window_seconds,
            "credited_settlement_seconds": self.credited_settlement_seconds,
        }

    @classmethod
    def from_attestation(
        cls,
        attestation: QuiescenceAttestationV1,
        *,
        horizon_kind: str,
        horizon_value: float,
        scored_at_offset_seconds: float,
        settlement_window_seconds: float = 0.0,
    ) -> "HorizonEvidenceV1":
        """Horizon evidence is the attestation's, never the scorer's own opinion."""

        post_horizon = max(
            0.0, float(scored_at_offset_seconds) - float(attestation.horizon_offset_seconds)
        )
        return cls(
            horizon_kind=horizon_kind,
            horizon_value=horizon_value,
            scored_at_offset_seconds=scored_at_offset_seconds,
            clipped=attestation.clipped,
            quiescence_attested=attestation.quiesced,
            settlement_window_seconds=settlement_window_seconds,
            credited_settlement_seconds=min(post_horizon, settlement_window_seconds),
        )


# --------------------------------------------------------------------------- #
# The receipt
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RewardReceiptV1(JsonDataclassMixin):
    """Container-authoritative reward, bound to the rollout and trace digest."""

    reward_id: str
    rollout_id: str
    trace_digest: str
    channels: tuple[RewardChannelV1, ...]
    optimized_channel: str
    terminal_status: str
    evaluation_plan_id: str
    horizon: HorizonEvidenceV1 | None = None
    reward_relation: str = "cooperative"
    scoring_state: ScoringState = ScoringState.COMPLETED
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = REWARD_RECEIPT_SCHEMA_VERSION

    @property
    def absent(self) -> bool:
        """No channel at all. A zero measure is a channel; this is not one."""

        return not self.channels

    @property
    def plan_outcome(self) -> PlanOutcome:
        return classify_plan_outcome(self.evaluation_plan_id)

    def validate(self, *, episode_trace_digest: str | None = None) -> None:
        if not self.rollout_id.strip():
            raise RewardError(f"reward {self.reward_id} names no rollout")
        if self.terminal_status not in TERMINAL_STATUSES:
            raise RewardError(
                f"reward {self.reward_id} claims non-terminal status {self.terminal_status!r}"
            )
        if self.scoring_state is not ScoringState.COMPLETED:
            raise RewardError(
                f"reward {self.reward_id} is {self.scoring_state.value}; an unscored receipt "
                "is not a reward"
            )
        if not self.channels:
            raise RewardError(
                f"reward {self.reward_id} carries no channel; absent is not zero"
            )
        if not self.trace_digest:
            raise RewardError(f"reward {self.reward_id} is not bound to a trace digest")
        if episode_trace_digest is not None and episode_trace_digest != self.trace_digest:
            raise RewardError(
                f"reward {self.reward_id} trace digest does not match its episode"
            )
        if self.optimized_channel not in {channel.channel_id for channel in self.channels}:
            raise RewardError(
                f"reward {self.reward_id} optimizes channel {self.optimized_channel!r} "
                "which it does not carry"
            )
        if self.reward_relation in COMPETITIVE_RELATIONS:
            for channel in self.channels:
                if channel.rank is None:
                    raise RewardError(
                        f"reward {self.reward_id} declares {self.reward_relation} but channel "
                        f"{channel.channel_id} resolved no rank"
                    )
        if self.horizon is not None:
            self.horizon.validate()

    def value(self, channel_id: str | None = None) -> float:
        wanted = channel_id or self.optimized_channel
        for channel in self.channels:
            if channel.channel_id == wanted:
                return channel.measure
        raise RewardError(f"reward {self.reward_id} has no channel {wanted!r}")

    def channel_for(self, team_id: str) -> RewardChannelV1:
        """The one channel belonging to a team. Ambiguity is an error."""

        matches = [channel for channel in self.channels if channel.team_id == team_id]
        if not matches:
            raise RewardError(f"reward {self.reward_id} has no channel for team {team_id!r}")
        if len(matches) > 1:
            raise RewardError(
                f"reward {self.reward_id} carries {len(matches)} channels for team "
                f"{team_id!r}; a team's measure must be unambiguous"
            )
        return matches[0]

    def value_for_team(self, team_id: str) -> float:
        return self.channel_for(team_id).measure

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward_id": self.reward_id,
            "rollout_id": self.rollout_id,
            "trace_digest": self.trace_digest,
            "optimized_channel": self.optimized_channel,
            "terminal_status": self.terminal_status,
            "evaluation_plan_id": self.evaluation_plan_id,
            "reward_relation": self.reward_relation,
            "scoring_state": self.scoring_state.value,
            "metadata": dict(self.metadata),
            "channels": [channel.to_dict() for channel in self.channels],
            "horizon": None if self.horizon is None else self.horizon.to_dict(),
            "schema_version": self.schema_version,
        }


# --------------------------------------------------------------------------- #
# The authority
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ScoringSlotV1(JsonDataclassMixin):
    """What the authority knows about one attempt's scoring, before and after."""

    rollout_id: str
    state: ScoringState
    evaluation_plan_id: str
    trace_digest: str = ""
    receipt: RewardReceiptV1 | None = None
    deferred_reason: str | None = None


class CispoRewardAuthority:
    """Scores an attempt exactly once, only after finalize attested the horizon.

    ``deferred_scoring`` is a declared capability. Where it is not declared,
    :meth:`defer` refuses rather than inventing an ``awaiting_score`` state the
    caller never agreed to.
    """

    def __init__(
        self,
        *,
        evaluation_plan_id: str,
        reward_relation: str = "cooperative",
        deferred_scoring: bool = False,
        settlement_window_seconds: float = 0.0,
        tie_policy: str = "shared_rank",
    ) -> None:
        if reward_relation not in REWARD_RELATIONS:
            raise RewardError(f"unknown reward relation {reward_relation!r}")
        if tie_policy not in TIE_POLICIES:
            raise RewardError(f"unknown tie policy {tie_policy!r}")
        if settlement_window_seconds < 0:
            raise RewardError("settlement window must be non-negative")
        self._evaluation_plan_id = evaluation_plan_id
        self._reward_relation = reward_relation
        self._deferred_scoring = bool(deferred_scoring)
        self._settlement_window_seconds = float(settlement_window_seconds)
        self._tie_policy = tie_policy
        self._lock = threading.RLock()
        self._slots: dict[str, ScoringSlotV1] = {}

    @property
    def deferred_scoring(self) -> bool:
        return self._deferred_scoring

    @property
    def evaluation_plan_id(self) -> str:
        return self._evaluation_plan_id

    def state(self, rollout_id: str) -> ScoringState:
        with self._lock:
            slot = self._slots.get(rollout_id)
        return slot.state if slot else ScoringState.RUNNING

    def slot(self, rollout_id: str) -> ScoringSlotV1:
        with self._lock:
            slot = self._slots.get(rollout_id)
        if slot is None:
            raise RewardError(f"no scoring slot for {rollout_id!r}")
        return slot

    # -- lifecycle -------------------------------------------------------- #

    def open(self, rollout_id: str, *, log: RolloutEventLog | None = None) -> ScoringSlotV1:
        with self._lock:
            existing = self._slots.get(rollout_id)
            if existing is not None:
                return existing
            slot = ScoringSlotV1(
                rollout_id=rollout_id,
                state=ScoringState.RUNNING,
                evaluation_plan_id=self._evaluation_plan_id,
            )
            self._slots[rollout_id] = slot
        if log is not None:
            RewardStreamer.code(
                log, authority="container", kind=self._reward_relation, plan_ref=self._evaluation_plan_id
            ).opened({"scoring_state": ScoringState.RUNNING.value})
        return slot

    def defer(self, rollout_id: str, *, reason: str) -> ScoringSlotV1:
        """Declare that the measure is not yet readable. Capability-gated."""

        if not self._deferred_scoring:
            raise RewardError(
                f"attempt {rollout_id} cannot await a score: deferred scoring was never "
                "advertised as a capability"
            )
        with self._lock:
            slot = self._slots.get(rollout_id) or self.open(rollout_id)
            if slot.state is ScoringState.COMPLETED:
                raise RewardError(f"attempt {rollout_id} is already scored")
            slot = ScoringSlotV1(
                rollout_id=rollout_id,
                state=ScoringState.AWAITING_SCORE,
                evaluation_plan_id=self._evaluation_plan_id,
                deferred_reason=reason,
            )
            self._slots[rollout_id] = slot
            return slot

    def settle(
        self,
        *,
        rollout_id: str,
        trace_digest: str,
        terminal_status: str,
        attestation: QuiescenceAttestationV1,
        scored_at_offset_seconds: float,
        horizon_kind: str,
        horizon_value: float,
        measures: Mapping[str, float] | None = None,
        measure: float | None = None,
        optimized_team_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        log: RolloutEventLog | None = None,
    ) -> RewardReceiptV1:
        """Score once, bound to the sealed trace. Repeat calls return the same receipt.

        ``measures`` is the per-team form and ``measure`` the single-channel one;
        supplying neither is an absent reward, which is not a zero.
        """

        if not trace_digest:
            raise RewardError(
                f"attempt {rollout_id} has no sealed trace digest to bind a reward to"
            )
        if attestation.rollout_id != rollout_id:
            raise RewardError(
                f"quiescence attestation names {attestation.rollout_id!r}, not {rollout_id!r}"
            )
        with self._lock:
            slot = self._slots.get(rollout_id)
            if slot is not None and slot.receipt is not None:
                # Final scoring is idempotent; a second read must not produce a
                # second reward.
                return slot.receipt

        horizon = HorizonEvidenceV1.from_attestation(
            attestation,
            horizon_kind=horizon_kind,
            horizon_value=horizon_value,
            scored_at_offset_seconds=scored_at_offset_seconds,
            settlement_window_seconds=self._settlement_window_seconds,
        )
        channels = self._channels(measures=measures, measure=measure)
        optimized = self._optimized(channels, optimized_team_id=optimized_team_id)
        payload = dict(metadata or {})
        payload.setdefault("reward_relation", self._reward_relation)
        payload.setdefault("tie_policy", self._tie_policy)
        if self._deferred_scoring:
            payload.setdefault("deferred_scoring", True)
        if not channels:
            payload.setdefault(
                "missing_evidence", "the evaluation plan produced no measure for this attempt"
            )
        receipt = RewardReceiptV1(
            reward_id=f"reward_{rollout_id}",
            rollout_id=rollout_id,
            trace_digest=trace_digest,
            channels=channels,
            optimized_channel=optimized,
            terminal_status=terminal_status,
            evaluation_plan_id=self._evaluation_plan_id,
            horizon=horizon,
            reward_relation=self._reward_relation,
            scoring_state=ScoringState.COMPLETED,
            metadata=payload,
        )
        with self._lock:
            self._slots[rollout_id] = ScoringSlotV1(
                rollout_id=rollout_id,
                state=ScoringState.COMPLETED,
                evaluation_plan_id=self._evaluation_plan_id,
                trace_digest=trace_digest,
                receipt=receipt,
            )
        if log is not None:
            streamer = RewardStreamer.code(
                log,
                authority="container",
                kind=self._reward_relation,
                plan_ref=self._evaluation_plan_id,
            )
            streamer.opened({"scoring_state": ScoringState.COMPLETED.value})
            # Missing stays null; never coerce to 0.
            streamer.signal(
                value=None if receipt.absent else receipt.value(),
                extra={"trace_digest": trace_digest, "optimized_channel": optimized},
            )
            streamer.closed()
        return receipt

    def _channels(
        self,
        *,
        measures: Mapping[str, float] | None,
        measure: float | None,
    ) -> tuple[RewardChannelV1, ...]:
        if measures:
            if self._reward_relation in COMPETITIVE_RELATIONS:
                return resolve_ranks(measures, tie_policy=self._tie_policy)
            return tuple(
                RewardChannelV1(
                    channel_id=f"score::{team_id}", team_id=team_id, measure=float(value)
                )
                for team_id, value in sorted(measures.items())
            )
        if measure is None:
            return ()
        return (RewardChannelV1(channel_id="score", team_id=None, measure=float(measure)),)

    def _optimized(
        self,
        channels: Sequence[RewardChannelV1],
        *,
        optimized_team_id: str | None,
    ) -> str:
        if not channels:
            return ABSENT_CHANNEL
        if optimized_team_id is not None:
            for channel in channels:
                if channel.team_id == optimized_team_id:
                    return channel.channel_id
            raise RewardError(
                f"no channel belongs to the optimized team {optimized_team_id!r}"
            )
        return channels[0].channel_id


# --------------------------------------------------------------------------- #
# The one coupling point to the route layer
# --------------------------------------------------------------------------- #


class CispoRewardAdapter:
    """Route-shaped facade over :class:`CispoRewardAuthority`.

    Method names match the declared route table: ``reward`` scores, ``reward_get``
    reads the receipt back.
    """

    def __init__(self, authority: CispoRewardAuthority) -> None:
        self._authority = authority

    @property
    def authority(self) -> CispoRewardAuthority:
        return self._authority

    def reward(self, body: Mapping[str, Any], *, attestation: QuiescenceAttestationV1) -> dict[str, Any]:
        receipt = self._authority.settle(
            rollout_id=str(body["rollout_id"]),
            trace_digest=str(body.get("trace_digest") or ""),
            terminal_status=str(body.get("terminal_status") or "completed"),
            attestation=attestation,
            scored_at_offset_seconds=float(body.get("scored_at_offset_seconds") or 0.0),
            horizon_kind=str(body.get("horizon_kind") or "wall_clock"),
            horizon_value=float(body.get("horizon_value") or 0.0),
            measures=body.get("measures"),
            measure=body.get("measure"),
            optimized_team_id=body.get("optimized_team_id"),
            metadata=body.get("metadata"),
        )
        return receipt.to_dict()

    def reward_get(self, rollout_id: str) -> dict[str, Any]:
        slot = self._authority.slot(rollout_id)
        if slot.receipt is None:
            return {
                "rollout_id": rollout_id,
                "scoring_state": slot.state.value,
                "evaluation_plan_id": slot.evaluation_plan_id,
                "deferred_reason": slot.deferred_reason,
                "reward": None,
            }
        return slot.receipt.to_dict()

    def advertised_capabilities(self) -> dict[str, Any]:
        return {
            "schema_version": REWARD_RECEIPT_SCHEMA_VERSION,
            "evaluation_plan_id": self._authority.evaluation_plan_id,
            "deferred_scoring": self._authority.deferred_scoring,
            "absent_is_not_zero": True,
            "bound_to_trace_digest": True,
        }


__all__ = [
    "ABSENT_CHANNEL",
    "COMPETITIVE_RELATIONS",
    "REWARD_RECEIPT_SCHEMA_VERSION",
    "REWARD_RELATIONS",
    "TIE_POLICIES",
    "CispoRewardAdapter",
    "CispoRewardAuthority",
    "HorizonEvidenceV1",
    "RewardChannelV1",
    "RewardError",
    "RewardReceiptV1",
    "ScoringSlotV1",
    "ScoringState",
    "resolve_ranks",
]
