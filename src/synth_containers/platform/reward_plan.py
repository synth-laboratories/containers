"""Quarantined evaluation-plan outcome classifier.

Substring checks on `evaluation_plan_ref` live here and nowhere else.
"""

from __future__ import annotations

from enum import StrEnum


class PlanOutcome(StrEnum):
    SCORED = "scored"
    GATED = "gated"
    REFUSED = "refused"


def classify_plan_outcome(plan_ref: str) -> PlanOutcome:
    """Classify an evaluation_plan_ref into a plan outcome.

    Inputs: evaluation_plan_ref str.
    Returns: PlanOutcome = scored | gated | refused.

    This is heuristic until producers emit structured plan kinds.
    Long-term replacement: typed EvaluationPlan.kind on the spec.
    """
    if "gated" in plan_ref or plan_ref.endswith(".gated"):
        return PlanOutcome.GATED
    if "refused" in plan_ref:
        return PlanOutcome.REFUSED
    return PlanOutcome.SCORED
