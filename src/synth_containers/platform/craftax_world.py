"""Deterministic in-process Craftax-shaped world for engine / code-policy CI.

Not the rust gold binary. Same nouns: seed instance, legal actions, per-step
RewardSignal, ascii frame digest. Optional gold HTTP is a separate adapter.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

ACTIONS = ("noop", "sleep", "do", "north", "south", "east", "west")


def _digest(payload: Any) -> str:
    import json

    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class StepResult:
    observation: dict[str, Any]
    reward: float | None
    done: bool
    valid_actions: list[str]
    ascii_map: str
    frame_digest: str
    env_steps: int
    frame_url: str | None = None
    frame_bytes: bytes | None = None


@dataclass
class CraftaxWorld:
    """Tiny survival grid. `do` collects wood once (+0.5). Movement is free."""

    max_steps: int = 8
    width: int = 5
    height: int = 5
    seed: int = 0
    x: int = 2
    y: int = 2
    wood: int = 0
    energy: int = 3
    env_steps: int = 0
    collected: bool = False
    tree: tuple[int, int] = (3, 2)
    rng_draw: int = 0

    def reset(self, seed: int, *, max_steps: int | None = None) -> StepResult:
        self.seed = int(seed)
        self.max_steps = int(max_steps or self.max_steps)
        self.x, self.y = 2, 2
        self.wood = 0
        self.energy = 3
        self.env_steps = 0
        self.collected = False
        self.tree = ((seed * 3) % self.width, (seed * 5 + 1) % self.height)
        if self.tree == (2, 2):
            self.tree = (3, 2)
        self.rng_draw = seed
        return self._result(reward=0.0, done=False)

    def step(self, action: str) -> StepResult:
        self.env_steps += 1
        name = str(action or "noop")
        if name not in ACTIONS:
            name = "noop"
        reward = 0.0
        if name == "north":
            self.y = max(0, self.y - 1)
        elif name == "south":
            self.y = min(self.height - 1, self.y + 1)
        elif name == "west":
            self.x = max(0, self.x - 1)
        elif name == "east":
            self.x = min(self.width - 1, self.x + 1)
        elif name == "sleep":
            self.energy = min(3, self.energy + 1)
        elif name == "do" and (self.x, self.y) == self.tree and not self.collected:
            self.wood += 1
            self.collected = True
            reward = 0.5
        done = self.env_steps >= self.max_steps
        return self._result(reward=reward, done=done)

    def _ascii(self) -> str:
        rows = []
        for y in range(self.height):
            cells = []
            for x in range(self.width):
                if (x, y) == (self.x, self.y):
                    cells.append("P")
                elif (x, y) == self.tree and not self.collected:
                    cells.append("T")
                else:
                    cells.append(".")
            rows.append("".join(cells))
        return "\n".join(rows)

    def _result(self, *, reward: float | None, done: bool) -> StepResult:
        ascii_map = self._ascii()
        observation = {
            "seed": self.seed,
            "x": self.x,
            "y": self.y,
            "wood": self.wood,
            "energy": self.energy,
            "tree": list(self.tree),
            "ascii": ascii_map,
            "valid_actions": list(ACTIONS),
            "env_steps": self.env_steps,
        }
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            valid_actions=list(ACTIONS),
            ascii_map=ascii_map,
            frame_digest=_digest({"seed": self.seed, "step": self.env_steps, "ascii": ascii_map}),
            env_steps=self.env_steps,
        )
