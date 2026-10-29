"""Terminal reward nodes for a rollout pin, and the product combiner.

Pure functions of a pin's recorded evidence: no platform state is read or written.
Split out of ``state.py`` (file-size cap) with behavior unchanged.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from .state_records import RolloutPin


class RewardNodesOutcome(NamedTuple):
    """``(reward, node_results, status, reason)`` for a pin's terminal reward."""

    reward: float | None
    nodes: list[dict[str, Any]]
    status: str
    reason: str | None


def reward_nodes(pin: RolloutPin, *, script_node: str) -> RewardNodesOutcome:
    """Score ``pin`` from its hill-climb nodes or its reward kind.

    ``script_node`` is the target spec's script node id (``reward.txt`` or a gate).
    """

    return RewardNodesOutcome(*_reward_nodes(pin, script_node))


def _reward_nodes(
    pin: RolloutPin, script_node: str
) -> tuple[float | None, list[dict[str, Any]], str, str | None]:
    if pin.hillclimb_nodes:
        nodes = [node.to_dict() for node in pin.hillclimb_nodes]
        gates = [node for node in pin.hillclimb_nodes if node.kind == "gate"]
        if any(node.status == "gated" or node.value is None for node in gates):
            return None, nodes, "gated", "heldout_gate"
        gate_value = gates[0].value if gates else None
        return float(gate_value) if gate_value is not None else None, nodes, "scored", None
    kind = pin.reward_kind
    if kind in {"env_sum", "verifier"}:
        authority = "verifier" if kind == "verifier" else "environment"
        node_kind = "verifier_reward" if kind == "verifier" else "env_reward"
        node_id = "verifier" if kind == "verifier" else "env_sum"
        if any(item is None for item in pin.reward_signals):
            return None, [
                {
                    "node_id": node_id,
                    "kind": node_kind,
                    "authority": authority,
                    "status": "skipped",
                    "value": None,
                }
            ], "absent", "missing_reward_signal"
        total = float(
            sum(float(item) for item in pin.reward_signals if item is not None)
        )
        return total, [
            {
                "node_id": node_id,
                "kind": node_kind,
                "authority": authority,
                "status": "scored",
                "value": total,
            }
        ], "scored", None
    if kind == "script":
        node_id = script_node
        value = pin.native_script_reward
        return value, [
            {
                "node_id": node_id,
                "kind": "script" if node_id == "reward.txt" else "gate",
                "authority": "trusted_scorer",
                "status": "scored" if value is not None else "skipped",
                "value": value,
            }
        ], "scored" if value is not None else "absent", None
    if kind == "env_status":
        value = pin.native_script_reward
        return value, [
            {
                "node_id": "env_status",
                "kind": "env_reward",
                "authority": "environment",
                "status": "scored" if value is not None else "skipped",
                "value": value,
            }
        ], "scored" if value is not None else "absent", None
    return None, [], "absent", "unknown_plan"


def product_combiner(bases: dict[str, float | None], required: list[str]) -> dict[str, Any]:
    """Multiply the required bases; absent when any required basis is missing."""

    values = []
    for name in required:
        item = bases.get(name)
        if item is None:
            return {"status": "absent", "reward": None, "reasons": [f"missing_basis:{name}"]}
        values.append(float(item))
    product = 1.0
    for item in values:
        product *= item
    return {"status": "scored", "reward": product}


__all__ = ["RewardNodesOutcome", "product_combiner", "reward_nodes"]
