"""Clean-room executor for ALFWorld's text-only ``game.tw-pddl`` artifacts.

This module deliberately has no dependency on ``alfworld`` or ``textworld``.
It reads the public JSON game artifact, evaluates the actions in ALFWorld's
``alfred.pddl`` domain, and exposes canonical (un-demangled) state facts for
differential testing.  Rendering remains deliberately separate from semantics.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .grammar import parse_grammar, render_macro


Fact = tuple[str, ...]


def _sexprs(text: str) -> list[object]:
    text = re.sub(r";[^\n]*", "", text)
    toks = re.findall(r"[()]|[^()\s]+", text)
    root: list[object] = []; stack = [root]
    for tok in toks:
        if tok == "(": child: list[object] = []; stack[-1].append(child); stack.append(child)
        elif tok == ")": stack.pop()
        else: stack[-1].append(tok)
    if len(stack) != 1: raise ValueError("unbalanced PDDL")
    return root


def _section(problem: str, key: str) -> list[object]:
    def walk(node: object):
        if isinstance(node, list):
            if node and node[0] == key: return node
            for child in node:
                found = walk(child)
                if found is not None: return found
        return None
    for item in _sexprs(problem):
        found = walk(item)
        if found is not None: return found
    raise ValueError(f"missing {key}")


def _atoms(node: object) -> set[Fact]:
    if not isinstance(node, list): return set()
    if node and isinstance(node[0], str) and node[0] not in {"and", "or", "exists", "not", ":init", ":goal"} and all(isinstance(x, str) for x in node):
        return {tuple(node)}
    return set().union(*(_atoms(child) for child in node[1:])) if len(node) > 1 else set()


def demangle(raw: str) -> str:
    return (raw.replace("_bar_", "|").replace("_minus_", "-").replace("_plus_", "+")
               .replace("_dot_", ".").replace("_comma_", ","))


@dataclass(frozen=True)
class PddlState:
    facts: frozenset[Fact]
    score: int
    won: bool


class PddlGame:
    """Executable state model of one official ``game.tw-pddl`` file.

    Commands use raw PDDL ids. Presentation adapters are responsible for the
    numbered natural-language aliases used by ALFWorld/TextWorld.
    """
    def __init__(self, artifact: dict[str, object], *, render: bool = True):
        self.artifact = artifact
        problem = str(artifact["pddl_problem"])
        self.initial = frozenset(_atoms(_section(problem, ":init")))
        self.goal = _section(problem, ":goal")[1]
        self.constants_by_type = self._typed_constants(problem)
        self.names = self._demangled_names(problem, self.initial)
        self.grammar = parse_grammar(str(artifact.get("grammar", ""))) if render else {}
        # TextWorld's generated PDDL keeps an object-at-location fact after a
        # same-location transfer.  The look grammar exposes exactly those
        # transfer artifacts, but not cross-location moves.
        self._floor_overrides: set[str] = set()
        self._taken_locations: dict[str, str] = {}
        self._taken_receptacles: dict[str, str] = {}
        self.state = PddlState(self.initial, 0, False)

    @classmethod
    def load(cls, path: str | Path) -> "PddlGame": return cls(json.loads(Path(path).read_text()))

    def reset(self) -> PddlState:
        self._floor_overrides.clear()
        self._taken_locations.clear()
        self._taken_receptacles.clear()
        self.state = PddlState(self.initial, 0, False); return self.state

    def reset_text(self) -> tuple[PddlState, str]:
        state = self.reset()
        match = re.search(r'"task"\s*:\s*\[\s*\{\s*"rhs"\s*:\s*"((?:\\.|[^"\\])*)"', str(self.artifact["grammar"]), re.S)
        task = bytes(match.group(1), "utf8").decode("unicode_escape") if match else "Your task is to: UNKNOWN GOAL"
        look = render_macro(self.grammar, "look.feedback", set(state.facts), {"a": "agent1"}, self.names) if "look.feedback" in self.grammar else self._render_special("look")
        return state, f"-= Welcome to TextWorld, ALFRED! =-\n\n{look}\n\n{task}"

    def _has(self, *fact: str) -> bool: return fact in self.state.facts
    def _replace(self, add: set[Fact] = set(), remove: set[Fact] = set(), cost: int = 0) -> PddlState:
        facts = (set(self.state.facts) - remove) | add
        won = self._goal_holds(self.goal, facts, {})
        self.state = PddlState(frozenset(facts), self.state.score + cost, won)
        return self.state

    @staticmethod
    def _typed_constants(problem: str) -> dict[str, tuple[str, ...]]:
        tokens = _section(problem, ":objects")[1:]
        out: dict[str, list[str]] = {}; pending: list[str] = []; i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == "-" and i + 1 < len(tokens):
                out.setdefault(str(tokens[i + 1]), []).extend(str(x) for x in pending)
                pending = []; i += 2
            else: pending.append(str(token)); i += 1
        return {kind: tuple(values) for kind, values in out.items()}

    @staticmethod
    def _demangled_names(problem: str, facts: frozenset[Fact]) -> dict[str, str]:
        """Match ALFWorld's ``Demangler(..., shuffle=False)`` numbering."""
        objects = _section(problem, ":objects")
        constants = {arg for fact in facts for arg in fact[1:]}
        ids = sorted(x for x in objects[1:] if isinstance(x, str) and x in constants)
        counts: dict[str, int] = {}
        bases: dict[str, str] = {}
        for raw in ids:
            if "_bar_" not in raw: continue
            base = raw.split("_bar_", 1)[0] + ("basin" if "basin" in raw.lower() else "")
            bases[raw] = base; counts[base] = counts.get(base, 0) + 1
        next_id = {name: count for name, count in counts.items()}
        out = {raw: demangle(raw) for raw in ids if raw not in bases}
        for raw in ids:
            if raw in bases:
                base = bases[raw]; out[raw] = f"{base} {next_id[base]}"; next_id[base] -= 1
        return out

    def resolve(self, phrase: str) -> str | None:
        phrase = phrase.strip().lower()
        matches = [raw for raw, shown in self.names.items() if shown.lower() == phrase]
        return matches[0] if len(matches) == 1 else None

    def available_actions(self) -> list[str]:
        """Return legal public commands for the current artifact state.

        This is the dependency-free counterpart of TextWorld's admissible
        command list.  It is derived from the same PDDL preconditions used by
        :meth:`step`, so it stays valid after every state transition.
        """
        facts = self.state.facts
        agent = next(x[1] for x in facts if x[0] == "atLocation")
        location = next(x[2] for x in facts if len(x) == 3 and x[0] == "atLocation" and x[1] == agent)
        here = [x[1] for x in facts if x[0] == "receptacleAtLocation" and x[2] == location]
        held = next((x[2] for x in facts if x[0] == "holds" and x[1] == agent), None)
        actions = ["look", "inventory", "help"]
        actions.extend(f"go to {self.names[r].lower()}" for r in self.names
                       if any(x[0] == "receptacleAtLocation" and x[1] == r for x in facts))
        for r in here:
            shown = self.names[r].lower()
            if ("openable", r) in facts:
                if ("opened", r) in facts: actions.append(f"close {shown}")
                else: actions.append(f"open {shown}")
            actions.append(f"examine {shown}")
        visible = [x[1] for x in facts if x[0] == "inReceptacle" and x[2] in here]
        actions.extend(f"examine {self.names[o].lower()}" for o in visible)
        if held:
            actions.append(f"examine {self.names[held].lower()}")
            for r in here:
                types = [x[2] for x in facts if x[0] == "objectType" and x[1] == held]
                rtypes = [x[2] for x in facts if x[0] == "receptacleType" and x[1] == r]
                if any(("canContain", rt, ot) in facts for rt in rtypes for ot in types):
                    if ("openable", r) not in facts or ("opened", r) in facts:
                        actions.append(f"move {self.names[held].lower()} to {self.names[r].lower()}")
            for r, kind, verb in ((r, "SinkBasinType", "clean") for r in here):
                if ("cleanable", held) in facts and ("receptacleType", r, kind) in facts: actions.append(f"{verb} {self.names[held].lower()} with {self.names[r].lower()}")
            for r, kind, verb in ((r, "MicrowaveType", "heat") for r in here):
                if ("heatable", held) in facts and ("receptacleType", r, kind) in facts: actions.append(f"{verb} {self.names[held].lower()} with {self.names[r].lower()}")
            for r, kind, verb in ((r, "FridgeType", "cool") for r in here):
                if ("coolable", held) in facts and ("receptacleType", r, kind) in facts: actions.append(f"{verb} {self.names[held].lower()} with {self.names[r].lower()}")
        else:
            for o in visible:
                r = next(x[2] for x in facts if x[0] == "inReceptacle" and x[1] == o)
                if ("pickupable", o) in facts and (("openable", r) not in facts or ("opened", r) in facts):
                    actions.append(f"take {self.names[o].lower()} from {self.names[r].lower()}")
        return list(dict.fromkeys(actions))

    def step_text(self, command: str, *, render: bool = True) -> tuple[PddlState, str]:
        """Execute the official natural-language command surface.

        The returned feedback covers all successful public action templates;
        unsupported/invalid-command wording remains a differential-test target.
        """
        text = command.strip().lower()
        if text in {"look", "inventory", "help"}:
            state = self.step(text)
            return state, self._render_special(text) if render else ""
        patterns = [(r"^go to (.+)$", "go to {0}"), (r"^(open|close|examine|use) (.+)$", "{0} {1}"),
                    (r"^take (.+) from (.+)$", "take {0} from {1}"), (r"^move (.+) to (.+)$", "move {0} to {1}"),
                    (r"^(clean|heat|cool|slice) (.+) with (.+)$", "{0} {1} with {2}")]
        for pattern, raw_format in patterns:
            m = re.match(pattern, text)
            if not m: continue
            vals = list(m.groups()); resolved = []
            for value in vals:
                if value in {"open", "close", "examine", "use", "clean", "heat", "cool", "slice"}: resolved.append(value)
                else:
                    entity = self.resolve(value)
                    if entity is None: return self.state, "Nothing happens."
                    resolved.append(entity)
            before = self.state; state = self.step(raw_format.format(*resolved))
            verb = vals[0] if vals[0] in {"open", "close", "examine", "use", "clean", "heat", "cool", "slice"} else text.split()[0]
            return state, self._feedback(verb, resolved, state != before or (verb == "examine" and self._can_examine(resolved[1]))) if render else ""
        return self.state, "Nothing happens."

    def _feedback(self, verb: str, args: list[str], valid: bool) -> str:
        if not valid: return "Nothing happens."
        action = {"go": "GotoLocation.feedback", "open": "OpenObject.feedback", "close": "CloseObject.feedback",
                  "use": "toggleObject.feedback", "take": "PickupObject.feedback",
                  "move": "PutObject.feedback", "clean": "cleanObject.feedback", "heat": "heatObject.feedback",
                  "cool": "coolObject.feedback", "slice": "sliceObject.feedback"}.get(verb)
        if verb == "examine":
            action = "examineReceptacle.feedback" if any(x[0] == "receptacleAtLocation" and x[1] == args[1] for x in self.state.facts) else "examineObject.feedback"
        if action and action in self.grammar:
            bindings = {"a": next(x[1] for x in self.state.facts if x[0] == "atLocation")}
            if verb == "go": bindings["r"] = args[0]
            elif verb in {"open", "close"}: bindings["r"] = args[1]
            elif verb == "examine":
                bindings["r" if any(x[0] == "receptacleAtLocation" and x[1] == args[1] for x in self.state.facts) else "o"] = args[1]
            elif verb == "use": bindings["o"] = args[1]
            elif verb in {"take", "move"}: bindings.update(o=args[0], r=args[1])
            elif verb in {"clean", "heat", "cool"}: bindings.update(o=args[1], r=args[2])
            elif verb == "slice": bindings.update(co=args[1], ko=args[2])
            return render_macro(self.grammar, action, set(self.state.facts), bindings, self.names)
        n = lambda raw: self.names[raw].lower()
        if verb == "go": return f"You arrive at {n(args[0])}. {self._describe_receptacle(args[0])}"
        if verb == "open": return f"You open the {n(args[1])}. {self._describe_receptacle(args[1])}"
        if verb == "close": return f"You close the {n(args[1])}."
        if verb == "examine": return self._describe_receptacle(args[1]) if any(x[0] == "receptacleAtLocation" and x[1] == args[1] for x in self.state.facts) else self._describe_object(args[1])
        if verb == "use": return f"You turn on the {n(args[1])}."
        if verb == "take": return f"You pick up the {n(args[0])} from the {n(args[1])}."
        if verb == "move": return f"You move the {n(args[0])} to the {n(args[1])}."
        if verb in {"clean", "heat", "cool"}: return f"You {verb} the {n(args[1])} using the {n(args[2])}."
        if verb == "slice": return f"You sliced the {n(args[1])} with the {n(args[2])}."
        return "Nothing happens."

    def _can_examine(self, item: str) -> bool:
        facts = self.state.facts; agent = next(x[1] for x in facts if x[0] == "atLocation")
        location = next(x[2] for x in facts if len(x) == 3 and x[0] == "atLocation" and x[1] == agent)
        return (("holds", agent, item) in facts or any(x[0] == "receptacleAtLocation" and x[1] == item and x[2] == location for x in facts) or any(x[0] == "inReceptacle" and x[1] == item and any(y == ("receptacleAtLocation", x[2], location) for y in facts) for x in facts))

    def _describe_receptacle(self, receptacle: str) -> str:
        name = self.names[receptacle].lower()
        if ("openable", receptacle) in self.state.facts and ("opened", receptacle) not in self.state.facts:
            return f"The {name} is closed."
        contents = sorted(x[1] for x in self.state.facts if len(x) == 3 and x[0] == "inReceptacle" and x[2] == receptacle)
        things = self._join([self._indefinite(self.names[x].lower()) for x in contents])
        return f"The {name} is open. In it, you see {things}." if ("openable", receptacle) in self.state.facts else f"On the {name}, you see {things}."

    def _describe_object(self, obj: str) -> str:
        name = self.names[obj].lower(); f = self.state.facts
        if ("isReceptacleObject", obj) in f:
            contents = sorted(x[1] for x in f if len(x) == 3 and x[0] == "inReceptacleObject" and x[2] == obj)
            return f"This is a normal {name}. In it, you see {self._join([self._indefinite(self.names[x].lower()) for x in contents])}."
        if ("isClean", obj) in f and ("isHot", obj) in f: return f"This is a hot and clean {name}."
        if ("isClean", obj) in f: return f"This is a clean {name}."
        if ("isHot", obj) in f: return f"This is a hot {name}."
        if ("isCool", obj) in f: return f"This is a cold {name}."
        return f"There's nothing special about {name}."

    def _render_special(self, command: str) -> str:
        # TextWorld performs the look action before feedback; the grammar's
        # checked-location branch intentionally suppresses the room overview.
        if command == "look":
            agent = next(x[1] for x in self.state.facts if x[0] == "atLocation")
            location = next(x[2] for x in self.state.facts if x[0] == "atLocation" and x[1] == agent)
            def entity_order(raw: str) -> tuple[str, int]:
                stem, _, suffix = self.names[raw].lower().rpartition(" ")
                return stem, -(int(suffix) if suffix.isdigit() else 0)
            current = sorted((x[1] for x in self.state.facts if x[0] == "receptacleAtLocation" and x[2] == location), key=entity_order)
            if not current:
                return "You are in the middle of a room. Looking quickly around you, you see nothing."
            floor = sorted((x[1] for x in self.state.facts if x[0] == "objectAtLocation" and x[2] == location
                            and x[1] in self._floor_overrides), key=entity_order)
            return f"You are facing the {self._join([self.names[x].lower() for x in current])}. Next to it, you see {self._join([self._indefinite(self.names[x].lower()) for x in floor])}."
        key = f"{command}.feedback"
        if key in self.grammar:
            facts = set(self.state.facts)
            agent = next(x[1] for x in facts if x[0] == "atLocation")
            return render_macro(self.grammar, key, facts, {"a": agent}, self.names)
        if command == "inventory":
            held = sorted(x[2] for x in self.state.facts if x[:2] == ("holds", "agent1"))
            return "You are not carrying anything." if not held else f"You are carrying: {self._join([self._indefinite(self.names[x].lower()) for x in held])}."
        if command == "look":
            agent = next(x[1] for x in self.state.facts if x[0] == "atLocation")
            location = next(x[2] for x in self.state.facts if len(x) == 3 and x[0] == "atLocation" and x[1] == agent)
            current = sorted(x[1] for x in self.state.facts if len(x) == 3 and x[0] == "receptacleAtLocation" and x[2] == location)
            if current:
                floor = sorted(x[1] for x in self.state.facts if len(x) == 3 and x[0] == "objectAtLocation" and x[2] == location and not any(y == ("inReceptacle", x[1], r) for y in self.state.facts for r in current))
                return f"You are facing the {self._join([self.names[x].lower() for x in current])}. Next to it, you see {self._join([self._indefinite(self.names[x].lower()) for x in floor])}."
            if ("checked", location) in self.state.facts:
                return "You are in the middle of a room. Looking quickly around you, you see nothing."
            rooms = sorted((x[1] for x in self.state.facts if len(x) == 3 and x[0] == "receptacleAtLocation"), key=lambda raw: (self.names[raw].split()[0].lower(), raw))
            return "You are in the middle of a room. Looking quickly around you, you see " + self._join([self._indefinite(self.names[x].lower()) for x in rooms]) + "."
        if command == "help": return "Available commands:\n  look:                             look around your current location\n  inventory:                        check your current inventory\n  go to (receptacle):               move to a receptacle\n  open (receptacle):                open a receptacle\n  close (receptacle):               close a receptacle\n  take (object) from (receptacle):  take an object from a receptacle\n  move (object) to (receptacle):  place an object in or on a receptacle\n  examine (something):              examine a receptacle or an object\n  use (object):                     use an object\n  heat (object) with (receptacle):  heat an object using a receptacle\n  clean (object) with (receptacle): clean an object using a receptacle\n  cool (object) with (receptacle):  cool an object using a receptacle\n  slice (object) with (object):     slice an object using a sharp object\n"
        return "Nothing happens."

    @staticmethod
    def _indefinite(name: str) -> str: return "a " + name

    @staticmethod
    def _join(items: list[str]) -> str:
        if not items: return "nothing"
        if len(items) == 1: return items[0]
        if len(items) == 2: return f"{items[0]}, and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    def _goal_holds(self, node: object, facts: set[Fact], env: dict[str, str]) -> bool:
        if isinstance(node, str): return False
        if not node: return True
        head = node[0]
        if head == "and": return all(self._goal_holds(x, facts, env) for x in node[1:])
        if head == "exists":
            # PDDL game's object typing only narrows the domain; checking all constants is equivalent.
            declaration = node[1]
            names = [(declaration[i], declaration[i + 2]) for i in range(len(declaration) - 2) if declaration[i].startswith("?") and declaration[i + 1] == "-"]
            def recur(i: int, scope: dict[str, str]) -> bool:
                if i == len(names): return self._goal_holds(node[2], facts, scope)
                name, kind = names[i]
                return any(recur(i + 1, {**scope, name: value}) for value in self.constants_by_type.get(kind, ()))
            return recur(0, env)
        if head == "not": return not self._goal_holds(node[1], facts, env)
        if head == "=":
            if len(node) != 3:
                return False
            return env.get(node[1], node[1]) == env.get(node[2], node[2])
        return tuple([head] + [env.get(x, x) for x in node[1:]]) in facts

    def goal_progress(self) -> float:
        """Return the best satisfied fraction of the declared PDDL goal."""
        return self._goal_progress(self.goal, set(self.state.facts), {})

    def _goal_progress(self, node: object, facts: set[Fact], env: dict[str, str]) -> float:
        if isinstance(node, str): return 0.0
        if not node: return 1.0
        head = node[0]
        if head == "and":
            parts = [self._goal_progress(child, facts, env) for child in node[1:]]
            return sum(parts) / len(parts) if parts else 1.0
        if head == "exists":
            declaration = node[1]
            names = [(declaration[i], declaration[i + 2]) for i in range(len(declaration) - 2)
                     if declaration[i].startswith("?") and declaration[i + 1] == "-"]
            def recur(i: int, scope: dict[str, str]) -> float:
                if i == len(names): return self._goal_progress(node[2], facts, scope)
                name, kind = names[i]
                return max((recur(i + 1, {**scope, name: value})
                            for value in self.constants_by_type.get(kind, ())), default=0.0)
            return recur(0, env)
        if head == "not": return 1.0 - self._goal_progress(node[1], facts, env)
        if head == "=":
            return float(len(node) == 3 and env.get(node[1], node[1]) == env.get(node[2], node[2]))
        atom = tuple([head] + [env.get(x, x) for x in node[1:]])
        return float(atom in facts)

    def step(self, command: str) -> PddlState:
        """Apply canonical command syntax: ``verb arg ...``; invalid actions are no-ops."""
        p = command.split()
        if not p: return self.state
        p[0] = p[0].lower()
        facts = self.state.facts
        agents = [x[1] for x in facts if x[0] == "atLocation"]
        if not agents: raise ValueError("game has no agent location")
        agent = agents[0]; location = next(x[2] for x in facts if len(x) == 3 and x[0] == "atLocation" and x[1] == agent)
        if p[0] in {"look", "help", "inventory"}: return self._replace({("checked", agent if p[0] != "look" else location)})
        if p[:2] == ["go", "to"] and len(p) == 3:
            recep = p[2]
            dest = next((x[2] for x in facts if x[0] == "receptacleAtLocation" and x[1] == recep), None)
            return self._replace({("atLocation", agent, dest)}, {("atLocation", agent, location)}, 1) if dest else self.state
        if p[0] in {"open", "close"} and len(p) == 2:
            r = p[1]; here = ("receptacleAtLocation", r, location) in facts
            if here and ("openable", r) in facts and ((p[0] == "open") != (("opened", r) in facts)):
                return self._replace({("opened", r)} if p[0] == "open" else set(), set() if p[0] == "open" else {("opened", r)}, 1)
            return self.state
        if p[0] == "examine" and len(p) == 2:
            item = p[1]
            if self._can_examine(item):
                return self._replace({("checked", item)})
            return self.state
        if p[0] == "use" and len(p) == 2:
            o = p[1]
            receptacle = next((x[2] for x in facts if len(x) == 3 and x[0] == "inReceptacle" and x[1] == o and ("receptacleAtLocation", x[2], location) in facts), None)
            if receptacle and ("toggleable", o) in facts:
                was_on = ("isOn", o) in facts
                return self._replace({("isToggled", o)} | (set() if was_on else {("isOn", o)}), {("isOn", o)} if was_on else set(), 5)
            return self.state
        if p[0] == "take" and len(p) == 4 and p[2] == "from":
            o, r = p[1], p[3]
            ok = all(x in facts for x in [("pickupable", o), ("inReceptacle", o, r), ("receptacleAtLocation", r, location)]) and not any(x[0] == "holds" and x[1] == agent for x in facts)
            if ok and (("openable", r) not in facts or ("opened", r) in facts):
                self._taken_locations[o] = location
                self._taken_receptacles[o] = r
                return self._replace({("holds", agent, o), ("holdsAny", agent)}, {("inReceptacle", o, r), ("objectAtLocation", o, location)}, 1)
            return self.state
        if p[0] == "move" and len(p) == 4 and p[2] == "to":
            o, r = p[1], p[3]
            types = [x[2] for x in facts if len(x) == 3 and x[0] == "objectType" and x[1] == o]
            rtypes = [x[2] for x in facts if x[0] == "receptacleType" and x[1] == r]
            ok = ("holds", agent, o) in facts and ("receptacleAtLocation", r, location) in facts and any(("canContain", rt, ot) in facts for ot in types for rt in rtypes)
            if ok and (("openable", r) not in facts or ("opened", r) in facts):
                if (self._taken_locations.pop(o, location) == location
                        and self._taken_receptacles.pop(o, r) != r):
                    self._floor_overrides.add(o)
                return self._replace({("inReceptacle", o, r), ("objectAtLocation", o, location)}, {("holds", agent, o), ("holdsAny", agent)}, 1)
            return self.state
        if p[0] in {"clean", "heat", "cool"} and len(p) == 4 and p[2] == "with":
            verb, o, r = p[0], p[1], p[3]; required = {"clean": ("cleanable", "SinkBasinType", "isClean"), "heat": ("heatable", "MicrowaveType", "isHot"), "cool": ("coolable", "FridgeType", "isCool")}[verb]
            if (required[0], o) in facts and ("holds", agent, o) in facts and ("receptacleAtLocation", r, location) in facts and ("receptacleType", r, required[1]) in facts:
                add, remove = {(required[2], o)}, ({("isCool", o)} if verb == "heat" else {("isHot", o)} if verb == "cool" else set())
                return self._replace(add, remove, 5)
        if p[0] == "slice" and len(p) == 4 and p[2] == "with":
            o, knife = p[1], p[3]
            if ("sliceable", o) in facts and ("holds", agent, knife) in facts and ("objectAtLocation", o, location) in facts and any(("objectType", knife, kind) in facts for kind in ("KnifeType", "ButterKnifeType")):
                return self._replace({("isSliced", o)}, cost=5)
        return self.state

