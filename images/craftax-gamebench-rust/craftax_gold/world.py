"""Craftax's only game-specific code: the gold engine's task payload.

Everything else — the HTTP relay, the episode loop, the policy ladder, the
checkpoint/frame plumbing — is `synth_containers.gold_*`, shared with Rogue and
DungeonGrid.
"""

from __future__ import annotations

from typing import Any

ENVIRONMENT_REF = "env:craftax_gold"
URL_ENV = "SYNTH_CRAFTAX_URL"
MAX_STEPS_ENV = "SYNTH_CRAFTAX_MAX_STEPS"
ENGINE = "craftax"

TASK_SCHEMA = "gamebench.task.craftax.v1"
TASK_ID = "synth_containers_craftax_react"

# The engine's own default world is a dev-sized board. Naming the world here is
# the difference between a real Craftax run and a toy one, so it is pinned.
DEFAULT_WORLD = "policy_dev_small"
# Real Craftax runs homeostasis: the gold engine's `_update_intrinsics` docks
# food at 25 steps, drink at 20, energy at 30. With it off those counters never
# tick, so `eat_cow`, `collect_drink` and `wake_up` become opportunities rather
# than pressures -- and a Crafter geometric mean computed that way is not
# comparable to a published Craftax number.
#
# `shared/task_resolve.py::_merge_rules` already defaults `homeostasis: True`.
# The only thing that ever turned it off was naming this base, and the two rule
# files are otherwise byte-identical.
DEFAULT_RULES = "symbolic_survival"
READOUT_PROFILE = "symbolic_compact"


def task_payload(seed: int, max_steps: int, *, world: str | None = None) -> dict[str, Any]:
    return {
        "schema": TASK_SCHEMA,
        "task_id": TASK_ID,
        "scenario_id": f"seed-{int(seed)}",
        "max_steps": int(max_steps),
        "world": {
            "use_default": world or DEFAULT_WORLD,
            "seed": int(seed),
            "max_steps": int(max_steps),
        },
        "rules": {"base": DEFAULT_RULES},
        "readouts": {"profile": READOUT_PROFILE},
    }


def task_payload_default(seed: int, max_steps: int) -> dict[str, Any]:
    """Full-board Craftax for the nanohorizon contest target."""

    return task_payload(seed, max_steps, world="craftax_default")
