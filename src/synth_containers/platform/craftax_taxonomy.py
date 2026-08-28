"""Craftax action taxonomy and usage accounting.

Ported into the containers platform so emerald is not the canonical runtime.
Classification is around each world.step (GOLD.step on the gold adapter):

- syntactically_invalid: token is not in the action vocabulary
- infeasible: legal token but not currently valid (missing prerequisite)
- effective_noop: accepted by the environment with no state change
- executed: accepted and the world changed

Outcomes are the more specific labels Workshop already expects:
moved / blocked / harvested / crafted / refused_missing_prerequisite / no_effect.

LLM calls are counted by call identity (plan() / span.policy.opened), never by
whether the provider happened to attach a usage object. Missing usage serializes
as null + usage_status, never numeric zero.
"""

from __future__ import annotations

from typing import Any, Mapping

from .craftax_world import ACTIONS

ACTION_CLASSES = (
    "syntactically_invalid",
    "infeasible",
    "effective_noop",
    "executed",
)
ACTION_OUTCOMES = (
    "moved",
    "blocked",
    "harvested",
    "crafted",
    "refused_missing_prerequisite",
    "no_effect",
)
COMPLETION_KINDS = (
    "natural_completion",
    "truncated",
    "infra_complete",
)
GOLD_URL_CONFIG_KEY = "gold_base_url"
DEFAULT_GOLD_BASE_URL = "http://127.0.0.1:8098"

_MOVE = frozenset({"north", "south", "east", "west", "up", "down", "left", "right"})
_HARVEST = frozenset({"do", "interact", "harvest", "collect"})
_CRAFT = frozenset({"craft", "make", "smelt", "forge"})


def classify_action(
    *,
    action: str,
    valid_actions: list[str] | None,
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
    vocabulary: tuple[str, ...] = ACTIONS,
) -> dict[str, str]:
    """Classify one attempted action around a world.step."""
    name = str(action or "").strip().lower()
    vocab = {str(item).strip().lower() for item in vocabulary}
    legal = {str(item).strip().lower() for item in (valid_actions or [])}
    if name not in vocab and name not in legal:
        return {"class": "syntactically_invalid", "outcome": "no_effect"}
    if legal and name not in legal:
        return {"class": "infeasible", "outcome": "refused_missing_prerequisite"}

    prior = _snapshot(before)
    next_state = _snapshot(after)
    moved = prior.get("position") != next_state.get("position") and None not in (
        prior.get("position"),
        next_state.get("position"),
    )
    harvested = _numeric_increase(prior.get("wood"), next_state.get("wood")) or _new_labels(
        prior.get("achievements"), next_state.get("achievements"), "harvest"
    )
    crafted = _new_labels(prior.get("achievements"), next_state.get("achievements"), "craft")
    changed = prior != next_state

    if name in _MOVE:
        if moved:
            return {"class": "executed", "outcome": "moved"}
        return {"class": "effective_noop", "outcome": "blocked"}
    if name in _HARVEST:
        if harvested:
            return {"class": "executed", "outcome": "harvested"}
        return {"class": "infeasible", "outcome": "refused_missing_prerequisite"}
    if name in _CRAFT:
        if crafted:
            return {"class": "executed", "outcome": "crafted"}
        return {"class": "infeasible", "outcome": "refused_missing_prerequisite"}
    if changed:
        return {"class": "executed", "outcome": "no_effect" if not harvested and not crafted else "harvested"}
    return {"class": "effective_noop", "outcome": "no_effect"}


def classify_completion(
    *,
    terminated: bool,
    truncated: bool,
    env_steps: int,
    max_steps: int,
    infra_failed: bool = False,
) -> str:
    if infra_failed:
        return "infra_complete"
    if truncated or (env_steps >= max_steps and not terminated):
        return "truncated"
    if terminated:
        return "natural_completion"
    return "truncated"


def usage_from_call_identity(
    *,
    calls: int,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    cost_usd: float | None = None,
    provider_emitted_usage: bool | None = None,
    kind: str = "model",
) -> dict[str, Any]:
    """Count calls by identity. Omitted provider usage stays null, never 0."""
    call_count = int(calls)
    if kind != "model" or call_count == 0:
        status = "not_applicable"
    elif provider_emitted_usage is False or (
        prompt_tokens is None and completion_tokens is None and total_tokens is None
    ):
        status = "provider_omitted"
    elif None in (prompt_tokens, completion_tokens, total_tokens):
        status = "partial"
    else:
        status = "reported"
    return {
        "calls": call_count,
        "llm_calls": call_count,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "usage_status": status,
    }


def _snapshot(observation: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(observation, Mapping):
        return {}
    x = observation.get("x")
    y = observation.get("y")
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        public = observation.get("public") if isinstance(observation.get("public"), dict) else {}
        x = public.get("x", x)
        y = public.get("y", y)
    wood = observation.get("wood")
    if not isinstance(wood, (int, float)):
        inventory = observation.get("inventory")
        if isinstance(inventory, dict):
            wood = inventory.get("wood")
    achievements = observation.get("achievements")
    if not isinstance(achievements, (list, dict, set)):
        public = observation.get("public") if isinstance(observation.get("public"), dict) else {}
        achievements = public.get("achievements")
    return {
        "position": (int(x), int(y)) if isinstance(x, (int, float)) and isinstance(y, (int, float)) else None,
        "wood": wood if isinstance(wood, (int, float)) else None,
        "achievements": _achievement_set(achievements),
        "energy": observation.get("energy"),
    }


def _achievement_set(value: Any) -> frozenset[str]:
    labels: set[str] = set()
    if isinstance(value, dict):
        for name, enabled in value.items():
            if enabled and str(name).strip():
                labels.add(str(name).strip().lower())
    elif isinstance(value, (list, set, tuple)):
        for item in value:
            if str(item).strip():
                labels.add(str(item).strip().lower())
    return frozenset(labels)


def _numeric_increase(before: Any, after: Any) -> bool:
    return isinstance(before, (int, float)) and isinstance(after, (int, float)) and after > before


def _new_labels(before: Any, after: Any, needle: str) -> bool:
    prior = before if isinstance(before, frozenset) else _achievement_set(before)
    nxt = after if isinstance(after, frozenset) else _achievement_set(after)
    return any(needle in label for label in (nxt - prior))
