"""GameBench adapter for clean-room ALFWorld ``game.tw-pddl`` artifacts."""
from __future__ import annotations

from pathlib import Path

from .core import Agent, MatchResult, State
from .pddl_runtime import PddlGame


class ArtifactAlfWorldGame:
    """A standalone text-only ALFWorld episode backed by one public artifact.

    Unlike :class:`AlfWorldGame`, this executes the actual generated PDDL
    problem and exposes the official numbered text command surface.
    """
    RULES = "Use commands exactly as listed. Satisfy the task stated in the observation."

    def __init__(self, gamefile: str | Path):
        self.gamefile = Path(gamefile)
        self.runtime = PddlGame.load(self.gamefile)
        self._state: State | None = None
        self._turn = 0

    def reset(self) -> State:
        _, observation = self.runtime.reset_text()
        self._turn = 0
        self._state = State(observation, 0.0, False, False, 0, self._info())
        return self._state

    def state(self) -> State:
        if self._state is None:
            raise RuntimeError("call reset() before state()")
        return self._state

    def available_actions(self) -> list[str]:
        if self._state is None:
            return []
        return [] if self._state.terminal else self.runtime.available_actions()

    def update(self, action: str) -> State:
        if self._state is None:
            raise RuntimeError("call reset() before update()")
        if self._state.terminal:
            return self._state
        before = self.runtime.state.won
        state, observation = self.runtime.step_text(action)
        self._turn += 1
        self._state = State(observation, 1.0 if state.won and not before else 0.0,
                            state.won, state.won, self._turn, self._info())
        return self._state

    def play(self, agent: Agent, max_turns: int = 50) -> MatchResult:
        state = self.reset(); actions: list[str] = []
        for _ in range(max_turns):
            if state.terminal:
                break
            action = agent.choose_action(self.RULES, state, self.available_actions())
            actions.append(action)
            state = self.update(action)
        return MatchResult(state, tuple(actions))

    def _info(self) -> dict[str, object]:
        return {"gamefile": str(self.gamefile), "admissible_commands": self.runtime.available_actions()}

