from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Sequence


class TimePenaltyMode(StrEnum):
    NONE = "none"
    CONSTANT = "constant"
    EXP = "exp"
    SQUARE = "square"
    LINEAR = "linear"
    ALWAYS = "always"


@dataclass(frozen=True, slots=True)
class ScoutRewardUpdate:
    reward: float
    scout_delta: int
    explored_tiles: int
    previous_explored_tiles: int
    dungeon_num: int
    dungeon_level: int
    time_penalty: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward": float(self.reward),
            "scout_delta": int(self.scout_delta),
            "explored_tiles": int(self.explored_tiles),
            "previous_explored_tiles": int(self.previous_explored_tiles),
            "dungeon_num": int(self.dungeon_num),
            "dungeon_level": int(self.dungeon_level),
            "time_penalty": float(self.time_penalty),
            "reward_source": "nle_scout",
        }


@dataclass(slots=True)
class NLEScoutRewardTracker:
    """Track NLE Scout reward from discovered glyph counts.

    The NLE Scout task rewards the increase in discovered map tiles per
    `(dungeon_num, dungeon_level)`. NLE represents undiscovered tiles as
    `nethack.GLYPH_CMAP_OFF`, so callers should pass that value as
    `unseen_glyph`.
    """

    unseen_glyph: int
    penalty_mode: TimePenaltyMode | str = TimePenaltyMode.NONE
    penalty_step: float = -0.01
    penalty_time: float = -0.0
    clamp_negative_delta: bool = False
    explored_by_level: dict[tuple[int, int], int] = field(default_factory=dict)
    frozen_steps: int = 0

    @classmethod
    def from_nethack_module(
        cls,
        nethack_module: Any,
        *,
        penalty_mode: TimePenaltyMode | str = TimePenaltyMode.NONE,
        penalty_step: float = -0.01,
        penalty_time: float = -0.0,
        clamp_negative_delta: bool = False,
    ) -> "NLEScoutRewardTracker":
        return cls(
            unseen_glyph=int(nethack_module.GLYPH_CMAP_OFF),
            penalty_mode=penalty_mode,
            penalty_step=penalty_step,
            penalty_time=penalty_time,
            clamp_negative_delta=clamp_negative_delta,
        )

    def reset(self) -> None:
        self.explored_by_level.clear()
        self.frozen_steps = 0

    def observed_tile_count(self, glyphs: Any) -> int:
        return _count_values_not_equal(glyphs, self.unseen_glyph)

    def update(
        self,
        *,
        glyphs: Any,
        dungeon_num: int,
        dungeon_level: int,
        previous_time: int | None = None,
        current_time: int | None = None,
    ) -> ScoutRewardUpdate:
        key = (int(dungeon_num), int(dungeon_level))
        explored = self.observed_tile_count(glyphs)
        previous = int(self.explored_by_level.get(key, 0))
        scout_delta = int(explored - previous)
        if self.clamp_negative_delta and scout_delta < 0:
            scout_delta = 0
        self.explored_by_level[key] = int(explored)
        time_penalty = self._time_penalty(previous_time=previous_time, current_time=current_time)
        reward = float(scout_delta) + float(time_penalty)
        return ScoutRewardUpdate(
            reward=reward,
            scout_delta=scout_delta,
            explored_tiles=explored,
            previous_explored_tiles=previous,
            dungeon_num=key[0],
            dungeon_level=key[1],
            time_penalty=time_penalty,
        )

    def update_from_observation(
        self,
        observation: dict[str, Any],
        *,
        previous_observation: dict[str, Any] | None = None,
        blstats_dungeon_num_index: int = 23,
        blstats_dungeon_level_index: int = 24,
        blstats_time_index: int = 20,
    ) -> ScoutRewardUpdate:
        glyphs = observation["glyphs"]
        blstats = observation["blstats"]
        previous_time = None
        current_time = None
        if previous_observation is not None:
            previous_time = int(previous_observation["blstats"][blstats_time_index])
            current_time = int(blstats[blstats_time_index])
        return self.update(
            glyphs=glyphs,
            dungeon_num=int(blstats[blstats_dungeon_num_index]),
            dungeon_level=int(blstats[blstats_dungeon_level_index]),
            previous_time=previous_time,
            current_time=current_time,
        )

    def _time_penalty(self, *, previous_time: int | None, current_time: int | None) -> float:
        mode = _parse_time_penalty_mode(self.penalty_mode)
        if mode is TimePenaltyMode.NONE or previous_time is None or current_time is None:
            return 0.0

        old_time = int(previous_time)
        new_time = int(current_time)
        if old_time == new_time:
            self.frozen_steps += 1
        else:
            self.frozen_steps = 0

        penalty = 0.0
        if mode is TimePenaltyMode.CONSTANT:
            if self.frozen_steps > 0:
                penalty += float(self.penalty_step)
        elif mode is TimePenaltyMode.EXP:
            penalty += (2**self.frozen_steps) * float(self.penalty_step)
        elif mode is TimePenaltyMode.SQUARE:
            penalty += (self.frozen_steps**2) * float(self.penalty_step)
        elif mode is TimePenaltyMode.LINEAR:
            penalty += self.frozen_steps * float(self.penalty_step)
        elif mode is TimePenaltyMode.ALWAYS:
            penalty += float(self.penalty_step)
        penalty += (new_time - old_time) * float(self.penalty_time)
        return float(penalty)


def _count_values_not_equal(values: Any, target: int) -> int:
    if hasattr(values, "__ne__") and hasattr(values, "sum"):
        result = (values != target).sum()
        if hasattr(result, "item"):
            return int(result.item())
        return int(result)
    return sum(1 for value in _flatten(values) if int(value) != int(target))


def _flatten(values: Any) -> Iterable[Any]:
    if isinstance(values, (str, bytes)):
        yield values
        return
    if isinstance(values, Sequence):
        for item in values:
            yield from _flatten(item)
        return
    yield values


def _parse_time_penalty_mode(value: TimePenaltyMode | str) -> TimePenaltyMode:
    if isinstance(value, TimePenaltyMode):
        return value
    return TimePenaltyMode(str(value).strip().lower())
