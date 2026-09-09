"""Generic single-turn classification runtime for chat/lab images.

One rollout = one observation, one policy call, one label, one reward. Banking77
and RedlineBench differ only in the dataset and the label space, so those are
constructor arguments and the rest of the contract (event shape, failure
sealing, reward authority) lives here once.

The policy half comes from ``policies.build_planner``: ``single_call`` for the
cheap floor, ``responses_react`` for the Responses API, ``codex_agentic`` for an
agent with a workspace. ``dataset_gold`` is the oracle lane — never graded.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .event_log import RolloutEventLog
from .platform.state import CompatPlatform, RolloutPin
from .policies import build_planner

__all__ = ["ClassifyRow", "ClassifyRuntime", "normalize_label"]

_EMPTY_USAGE: dict[str, Any] = {
    "prompt_tokens": None,
    "completion_tokens": None,
    "total_tokens": None,
    "calls": 0,
}
_PUNCT = re.compile(r"[^a-z0-9]+")


def normalize_label(value: str | None) -> str:
    return _PUNCT.sub("_", str(value or "").strip().lower()).strip("_")


@dataclass(frozen=True, slots=True)
class ClassifyRow:
    """One dataset item. ``label`` is gold and never enters the observation."""

    id: str
    text: str
    label: str
    metadata: Mapping[str, Any] = None  # type: ignore[assignment]

    def observation(self, *, labels: Sequence[str], seed: int, split: str) -> dict[str, Any]:
        return {
            "id": self.id,
            "seed": seed,
            "split": split,
            "text": self.text,
            "observation_text": self.text,
            "valid_actions": list(labels),
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class ClassifyRuntime:
    """``TargetSpec.runtime`` for a one-call classification image."""

    environment_ref: str
    load_row: Callable[[str, int], ClassifyRow | None]
    labels: Callable[[str], Sequence[str]]
    split_for: Callable[[str | None], str]
    system_prompt: str = ""
    reward_kind: str = "classification_accuracy"

    def simulate(self, platform: CompatPlatform, pin: RolloutPin, log: RolloutEventLog) -> None:
        if platform.spec.environment_ref != self.environment_ref:
            raise ValueError(f"unknown_environment:{platform.spec.environment_ref}")
        platform.step_calls += 1
        split = self.split_for(pin.world_ref)
        seed = int(pin.seed or 0)
        row = self.load_row(split, seed)
        log.append(
            "env.episode.opened", {"seed": seed, "split": split, "world_ref": pin.world_ref}
        )
        if row is None:
            self._fail(pin, log, reason="unknown_task_instance", detail=f"seed {seed} not in {split}")
            return

        labels = list(self.labels(split))
        observation = row.observation(labels=labels, seed=seed, split=split)
        log.append("observation", {key: value for key, value in observation.items()})

        harness = str(pin.policy_ref.get("harness") or "").strip()
        if not harness:
            self._fail(pin, log, reason="policy_ref_required", detail="harness missing")
            return
        log.append(
            "span.policy.opened",
            {"harness": harness, "config": pin.policy_ref.get("config")},
        )
        try:
            predicted, usage = self._act(platform, pin, harness, observation, row, labels)
        except Exception as exc:  # noqa: BLE001 — secret-free code onto the stream
            log.append(
                "span.policy.closed",
                {"status": "failed", "error_type": type(exc).__name__, "error_code": str(exc)[:160]},
            )
            self._fail(pin, log, reason="policy_error", detail=type(exc).__name__)
            return

        if isinstance(predicted, str) and not predicted.strip():
            predicted = None
        log.append("action", {"label": predicted, "text": predicted})
        log.append("span.policy.closed", {"status": "completed" if predicted else "empty"})

        if pin.omit_reward or predicted is None:
            value: float | None = None
        else:
            gold = normalize_label(row.label)
            value = 1.0 if gold and normalize_label(predicted) == gold else 0.0
        log.append(
            "reward_signal",
            {"value": value, "authority": "environment", "kind": self.reward_kind},
        )
        pin.reward_signals = [value]
        pin.usage = dict(usage)
        pin.status = "completed"
        pin.terminal = True
        log.append("env.episode.closed", {"status": "completed", "split": split, "seed": seed})
        log.append("status", {"status": "completed"})
        self._seal(log)

    # ---------------------------------------------------------------- internal

    def _act(
        self,
        platform: CompatPlatform,
        pin: RolloutPin,
        harness: str,
        observation: dict[str, Any],
        row: ClassifyRow,
        labels: Sequence[str],
    ) -> tuple[str | None, dict[str, Any]]:
        if harness == "dataset_gold":
            # The oracle lane. Useful for wiring proofs; never a graded result.
            return row.label, dict(_EMPTY_USAGE)

        config_id = str(pin.policy_ref.get("config") or "").strip()
        if not config_id:
            raise ValueError("simulate requires policy_ref.config; start must not default a model")
        policy = platform.policy_configs.get(config_id)
        config = dict(policy.config) if policy is not None else {}
        forced = config.get("forced_label")
        if isinstance(forced, str) and forced.strip():
            return forced.strip(), dict(_EMPTY_USAGE)
        config.setdefault("horizon", 1)
        config.setdefault("plan_max", 1)
        config.setdefault("objective", self.system_prompt)
        config.setdefault("system_prompt", self.system_prompt)

        planner = build_planner(harness, config_id=config_id, config=config)
        try:
            actions = planner.plan(observation)
        finally:
            closer = getattr(planner, "close", None)
            if callable(closer):
                closer()
        label = actions[0] if actions else None
        if label is not None and label not in labels:
            label = None
        return label, dict(planner.usage())

    def _fail(
        self, pin: RolloutPin, log: RolloutEventLog, *, reason: str, detail: str
    ) -> None:
        pin.reward_signals = [None]
        pin.usage = dict(_EMPTY_USAGE)
        pin.status = "failed"
        pin.terminal = True
        log.append("env.episode.closed", {"status": "failed", "reason": reason})
        log.append("status", {"status": "failed", "reason": reason, "detail": detail})
        self._seal(log)

    def _seal(self, log: RolloutEventLog) -> None:
        high_water = log.high_water
        log.append("capture.high_water", {"high_water": high_water})
        log.append("capture.closed", {"high_water": high_water})
        log.mark_closed()
