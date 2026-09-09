"""Container reward-calculation standard and live event streamer.

Two calculator families, one sealed ``reward_signal``:

* **code** — deterministic program: environment sum, trusted script
  (``reward.txt`` / held-out gate), or status. Banking77 label match, Craftax
  engine, Harbor verifier file.
* **verifier** — isolated / LLM / physician rubric, HealthBench-style: one
  ``rubric.grade`` per criterion, then an aggregate ``reward_signal``.

Annotations and rubric verification never rewrite this signal. Missing stays
null; never coerce to 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from ..event_log import RolloutEventLog
from .targets import RewardCalculatorFamily


REWARD_API_SCHEMA = "synth.container.reward-api.v1"
REWARD_STREAM_SCHEMA = "synth.reward.stream.v1"

REWARD_OPENED = "reward.calculation.opened"
REWARD_CLOSED = "reward.calculation.closed"
REWARD_FAILED = "reward.calculation.failed"
RUBRIC_GRADE = "rubric.grade"
REWARD_SIGNAL = "reward_signal"
EVALUATOR_OPENED = "span.evaluator.opened"
EVALUATOR_CLOSED = "span.evaluator.closed"

REWARD_EVENT_KINDS_CODE: tuple[str, ...] = (
    REWARD_OPENED,
    REWARD_SIGNAL,
    REWARD_CLOSED,
    REWARD_FAILED,
)

REWARD_EVENT_KINDS_VERIFIER: tuple[str, ...] = (
    REWARD_OPENED,
    EVALUATOR_OPENED,
    RUBRIC_GRADE,
    EVALUATOR_CLOSED,
    REWARD_SIGNAL,
    REWARD_CLOSED,
    REWARD_FAILED,
)

REWARD_STREAM_KINDS: frozenset[str] = frozenset(REWARD_EVENT_KINDS_VERIFIER)


def event_kinds_for(calculator: RewardCalculatorFamily | str) -> tuple[str, ...]:
    family = RewardCalculatorFamily(str(calculator))
    if family == RewardCalculatorFamily.VERIFIER:
        return REWARD_EVENT_KINDS_VERIFIER
    return REWARD_EVENT_KINDS_CODE


def reward_api_catalog(
    *,
    calculator: RewardCalculatorFamily | str = RewardCalculatorFamily.CODE,
    authority: str = "environment",
    aggregation: str = "env_sum",
    live: bool = False,
) -> dict[str, Any]:
    """Advertised HTTP + stream contract. Mounted on ``GET /reward/catalog``."""

    family = RewardCalculatorFamily(str(calculator))
    return {
        "schema": REWARD_API_SCHEMA,
        "calculator": family.value,
        "authority": authority,
        "aggregation": aggregation,
        "live": live,
        "immutable_after_seal": True,
        "rewritten_by_annotations": False,
        "missing_is_null": True,
        "event_kinds": list(event_kinds_for(family)),
        "guidance": (
            "Code calculators emit reward.calculation.opened, one or more reward_signal "
            "events, then closed. Verifier calculators (HealthBench) emit span.evaluator "
            "and one rubric.grade per criterion before the aggregate reward_signal. "
            "POST /reward reads the sealed signals; annotations never change them."
        ),
        "endpoints": [
            {
                "name": "reward_catalog",
                "method": "GET",
                "path": "/reward/catalog",
                "paid": False,
                "description": "This contract: calculator family, authority, event kinds, routes.",
            },
            {
                "name": "reward_get",
                "method": "GET",
                "path": "/reward",
                "paid": False,
                "description": "Latest scored receipt for ?rollout_id=.",
            },
            {
                "name": "reward_get_path",
                "method": "GET",
                "path": "/rollouts/{rollout_id}/reward",
                "paid": False,
                "description": "Same receipt on the rollout path.",
            },
            {
                "name": "reward_compute",
                "method": "POST",
                "path": "/reward",
                "paid": False,
                "description": "Score from sealed reward_signal events. Does not re-run a verifier.",
            },
            {
                "name": "reward_combine",
                "method": "POST",
                "path": "/reward/combine",
                "paid": False,
                "description": "Product combiner over named bases; missing stays absent.",
            },
            {
                "name": "reward_events",
                "method": "GET",
                "path": "/rollouts/{rollout_id}/reward/events",
                "paid": False,
                "stream": "poll",
                "description": "Poll reward-calculation events (sequence cursor, Last-Event-ID compatible).",
            },
            {
                "name": "reward_stream",
                "method": "GET",
                "path": "/rollouts/{rollout_id}/reward/stream",
                "paid": False,
                "stream": "sse",
                "description": "SSE of the same events. Hidden CoT is never included.",
            },
            {
                "name": "evaluation_get",
                "method": "GET",
                "path": "/evaluations/{evaluation_id}",
                "paid": False,
                "description": "Blocking script/gate evaluation record when the target uses one.",
            },
            {
                "name": "evaluation_events",
                "method": "GET",
                "path": "/evaluations/{evaluation_id}/events",
                "paid": False,
                "description": "Started/completed events for a blocking evaluation id.",
            },
        ],
    }


@dataclass
class RewardStreamer:
    """Write the standard reward-calculation events onto a rollout log."""

    log: RolloutEventLog
    calculator: str
    authority: str
    kind: str
    plan_ref: str | None = None
    _opened: bool = field(default=False, init=False, repr=False)
    _done: bool = field(default=False, init=False, repr=False)

    @classmethod
    def code(
        cls,
        log: RolloutEventLog,
        *,
        authority: str = "environment",
        kind: str = "env_sum",
        plan_ref: str | None = None,
    ) -> "RewardStreamer":
        return cls(log, calculator=RewardCalculatorFamily.CODE.value, authority=authority, kind=kind, plan_ref=plan_ref)

    @classmethod
    def verifier(
        cls,
        log: RolloutEventLog,
        *,
        authority: str = "verifier",
        kind: str = "rubric",
        plan_ref: str | None = None,
    ) -> "RewardStreamer":
        return cls(
            log,
            calculator=RewardCalculatorFamily.VERIFIER.value,
            authority=authority,
            kind=kind,
            plan_ref=plan_ref,
        )

    def opened(self, extra: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "calculator": self.calculator,
            "authority": self.authority,
            "kind": self.kind,
            "rewritten_by_annotations": False,
        }
        if self.plan_ref:
            payload["plan_ref"] = self.plan_ref
        if extra:
            payload.update(extra)
        self.log.append(REWARD_OPENED, payload)
        self._opened = True

    def evaluator_opened(self, payload: dict[str, Any]) -> None:
        self.log.append(EVALUATOR_OPENED, dict(payload))

    def grade(self, item: dict[str, Any]) -> None:
        self.log.append(RUBRIC_GRADE, dict(item))

    def evaluator_closed(self, payload: dict[str, Any]) -> None:
        self.log.append(EVALUATOR_CLOSED, dict(payload))

    def signal(self, *, value: float | None, extra: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "value": value,
            "authority": self.authority,
            "kind": self.kind,
            "calculator": self.calculator,
        }
        if extra:
            payload.update(extra)
        self.log.append(REWARD_SIGNAL, payload)

    def failed(self, reason: str, extra: dict[str, Any] | None = None) -> None:
        if self._done:
            return
        if not self._opened:
            self.opened()
        payload: dict[str, Any] = {
            "reason": reason,
            "calculator": self.calculator,
            "authority": self.authority,
        }
        if extra:
            payload.update(extra)
        self.log.append(REWARD_FAILED, payload)
        self.closed(status="failed")

    def closed(self, status: str = "completed") -> None:
        if not self._opened or self._done:
            return
        self.log.append(
            REWARD_CLOSED,
            {
                "status": status,
                "calculator": self.calculator,
                "authority": self.authority,
                "rewritten_by_annotations": False,
            },
        )
        self._done = True


def filter_reward_events(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in events if item.get("control") or item.get("kind") in REWARD_STREAM_KINDS]


__all__ = [
    "EVALUATOR_CLOSED",
    "EVALUATOR_OPENED",
    "REWARD_API_SCHEMA",
    "REWARD_CLOSED",
    "REWARD_EVENT_KINDS_CODE",
    "REWARD_EVENT_KINDS_VERIFIER",
    "REWARD_FAILED",
    "REWARD_OPENED",
    "REWARD_SIGNAL",
    "REWARD_STREAM_KINDS",
    "REWARD_STREAM_SCHEMA",
    "RUBRIC_GRADE",
    "RewardCalculatorFamily",
    "RewardStreamer",
    "event_kinds_for",
    "filter_reward_events",
    "reward_api_catalog",
]
