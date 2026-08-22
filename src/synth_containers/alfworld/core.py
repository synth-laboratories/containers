from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from random import Random
from typing import Protocol


class TaskFamily(StrEnum):
    PICK_AND_PLACE = "pick_and_place"
    EXAMINE_IN_LIGHT = "examine_in_light"
    CLEAN_AND_PLACE = "clean_and_place"
    HEAT_AND_PLACE = "heat_and_place"
    COOL_AND_PLACE = "cool_and_place"
    PICK_TWO_AND_PLACE = "pick_two_and_place"


@dataclass(frozen=True)
class State:
    observation: str
    reward: float
    terminal: bool
    won: bool
    turn: int
    info: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MatchResult:
    final_state: State
    actions: tuple[str, ...]


class Agent(Protocol):
    def choose_action(self, rules: str, state: State, actions: list[str]) -> str: ...


TASK_TEXT = {
    TaskFamily.PICK_AND_PLACE: "put the {obj} in the {target}",
    TaskFamily.EXAMINE_IN_LIGHT: "examine the {obj} with the desklamp",
    TaskFamily.CLEAN_AND_PLACE: "clean the {obj} and put it in the {target}",
    TaskFamily.HEAT_AND_PLACE: "heat the {obj} and put it in the {target}",
    TaskFamily.COOL_AND_PLACE: "cool the {obj} and put it in the {target}",
    TaskFamily.PICK_TWO_AND_PLACE: "put the {obj} and the {obj2} in the {target}",
}


class AlfWorldGame:
    """A deterministic, symbolic ALFWorld task with ALFWorld command wording."""

    RULES = "Use commands exactly as listed. Find objects, prepare them if required, then satisfy the task."

    def __init__(self, family: TaskFamily, obj: str = "apple", target: str = "fridge", obj2: str = "mug"):
        self.family, self.obj, self.target, self.obj2 = family, obj, target, obj2
        self.locations = {obj: "countertop", obj2: "table", "desklamp": "desk", "sinkbasin": "sink", "microwave": "counter", "fridge": "counter"}
        self.inventory: set[str] = set()
        self.prepared: set[str] = set()
        self.current = "middle of a room"
        self.turn = 0
        self.done = False
        self.won = False
        self.history: list[str] = []

    @classmethod
    def seeded(cls, family: TaskFamily, seed: int) -> "AlfWorldGame":
        rng = Random(seed)
        objects = ["apple", "mug", "soapbottle", "tomato", "plate"]
        targets = ["fridge", "cabinet", "drawer", "countertop"]
        obj = rng.choice(objects)
        obj2 = rng.choice([x for x in objects if x != obj])
        return cls(family, obj, rng.choice(targets), obj2)

    @property
    def task(self) -> str:
        return TASK_TEXT[self.family].format(obj=self.obj, target=self.target, obj2=self.obj2)

    def reset(self) -> State:
        self.__init__(self.family, self.obj, self.target, self.obj2)
        return self.state()

    def state(self, reward: float = 0.0, observation: str | None = None) -> State:
        return State(observation or self._describe(), reward, self.done, self.won, self.turn,
                     {"task": self.task, "inventory": sorted(self.inventory), "family": self.family.value})

    def _describe(self) -> str:
        if self.done:
            return "You won!" if self.won else "Episode finished."
        visible = [name for name, place in self.locations.items() if place == self.current]
        contents = ", ".join(visible) if visible else "nothing useful"
        return f"You are at the {self.current}. You see {contents}. Your task is to: {self.task}."

    def available_actions(self) -> list[str]:
        if self.done:
            return []
        actions = [f"go to {place}" for place in sorted(set(self.locations.values()))]
        actions += ["inventory", "look"]
        for name, place in self.locations.items():
            if place == self.current and name not in {"desklamp", "sinkbasin", "microwave", "fridge"}:
                actions.append(f"take {name} from {place}")
        for item in sorted(self.inventory):
            actions += [f"put {item} in {self.target}", f"put {item} in sinkbasin", f"put {item} in microwave", f"put {item} in fridge"]
            actions += [f"clean {item} with sinkbasin", f"heat {item} with microwave", f"cool {item} with fridge"]
        if self.current == self.locations["desklamp"] and self.obj in self.inventory:
            actions.append("use desklamp")
        return list(dict.fromkeys(actions))

    def update(self, action: str) -> State:
        if self.done:
            return self.state(0.0, "Episode already finished.")
        self.turn += 1
        action = action.strip().lower()
        self.history.append(action)
        if action == "look":
            return self.state(0.0)
        if action == "inventory":
            return self.state(0.0, "You are carrying: " + (", ".join(sorted(self.inventory)) or "nothing") + ".")
        if action.startswith("go to "):
            place = action[6:]
            if place in self.locations.values():
                self.current = place
                return self.state(0.0, f"You arrive at the {place}.")
        for item, place in list(self.locations.items()):
            if action == f"take {item} from {place}" and place == self.current and item not in {"desklamp", "sinkbasin", "microwave", "fridge"}:
                self.inventory.add(item); self.locations[item] = "inventory"
                return self.state(0.0, f"You pick up the {item}.")
        for verb, device, needed in (("clean", "sinkbasin", TaskFamily.CLEAN_AND_PLACE), ("heat", "microwave", TaskFamily.HEAT_AND_PLACE), ("cool", "fridge", TaskFamily.COOL_AND_PLACE)):
            if action == f"{verb} {self.obj} with {device}" and self.obj in self.inventory:
                self.prepared.add(verb)
                return self.state(0.0, f"You {verb} the {self.obj}.")
        if action == "use desklamp" and self.family == TaskFamily.EXAMINE_IN_LIGHT and self.obj in self.inventory:
            return self._finish()
        if action.startswith("put ") and " in " in action:
            item, destination = action[4:].split(" in ", 1)
            if item in self.inventory:
                self.inventory.remove(item); self.locations[item] = destination
                return self._finish() if self._is_complete() else self.state(0.0, f"You put the {item} in the {destination}.")
        return self.state(-0.01, "Nothing happens. That command is not valid here.")

    def _is_complete(self) -> bool:
        if self.family == TaskFamily.PICK_TWO_AND_PLACE:
            return self.locations.get(self.obj) == self.target and self.locations.get(self.obj2) == self.target
        preparation = {TaskFamily.CLEAN_AND_PLACE: "clean", TaskFamily.HEAT_AND_PLACE: "heat", TaskFamily.COOL_AND_PLACE: "cool"}
        return self.locations.get(self.obj) == self.target and (self.family not in preparation or preparation[self.family] in self.prepared)

    def _finish(self) -> State:
        self.done = self.won = True
        return self.state(1.0, "You won!")

    def play(self, agent: Agent, max_turns: int = 50) -> MatchResult:
        state, actions = self.reset(), []
        for _ in range(max_turns):
            if state.terminal: break
            action = agent.choose_action(self.RULES, state, self.available_actions())
            actions.append(action); state = self.update(action)
        return MatchResult(state, tuple(actions))

