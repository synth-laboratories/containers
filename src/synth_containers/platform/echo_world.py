"""Echo-shaped gym world. One reset, one step. Not a Harbor fold. Not a snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ECHO_ENVIRONMENT = "env:echo"


def prompt_for_seed(seed: int) -> str:
    return f"echo-{int(seed)}"


@dataclass
class EchoStep:
    observation: dict[str, Any]
    reward: float | None
    done: bool
    env_steps: int
    valid_action: str = ""


@dataclass
class EchoWorld:
    """Prompt is `echo-{seed}`. Matching action scores 1.0; mismatch is honest 0.0."""

    prompt: str = ""
    env_steps: int = 0
    seed: int = 0

    def reset(self, seed: int) -> EchoStep:
        self.seed = int(seed)
        self.env_steps = 0
        self.prompt = prompt_for_seed(self.seed)
        return EchoStep(
            observation={"text": self.prompt, "seed": self.seed},
            reward=None,
            done=False,
            env_steps=0,
            valid_action=self.prompt,
        )

    def step(self, action: str) -> EchoStep:
        self.env_steps += 1
        matched = str(action) == self.prompt
        return EchoStep(
            observation={"text": self.prompt, "seed": self.seed, "echoed": str(action)},
            reward=1.0 if matched else 0.0,
            done=True,
            env_steps=self.env_steps,
            valid_action=self.prompt,
        )
