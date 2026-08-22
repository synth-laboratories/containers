"""Evaluator for the TextWorld grammar dialect embedded in ALFWorld games."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

Fact = tuple[str, ...]


@dataclass(frozen=True)
class Rule:
    rhs: str
    condition: str | None = None


def parse_grammar(source: str) -> dict[str, list[Rule]]:
    """Extract all JSON grammar blocks from a `.tw-pddl` grammar string."""
    rules: dict[str, list[Rule]] = {}
    for raw in re.findall(r'grammar\s*::\s*"""\s*(\{.*?\})\s*"""', source, re.S):
        for name, choices in json.loads(raw).items():
            rules[name] = [Rule(choice["rhs"], choice.get("condition")) for choice in choices]
    return rules


def _atom(source: str) -> tuple[str, list[str]] | None:
    match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*)\((.*?)\)", source.strip())
    if not match:
        return None
    return match.group(1).lower(), [x.strip().split(":", 1)[0] for x in match.group(2).split(",") if x.strip()]


def _value(source: str, bindings: dict[str, str]) -> str:
    return bindings.get(source.split(":", 1)[0], source.split(":", 1)[0])


def condition_holds(condition: str | None, facts: set[Fact], bindings: dict[str, str]) -> bool:
    """Evaluate the conjunctive ALFWorld grammar conditions for fixed bindings."""
    if not condition: return True
    # A rule condition may introduce variables (the top-level ``look`` rule
    # does).  In that case TextWorld treats it as an existential query.
    if any(_value(arg, bindings) == arg.split(":", 1)[0]
           for source in condition.split(" & ")
           for atom in [_atom(source[4:] if source.startswith("not_") else source)]
           if atom is not None for arg in atom[1]):
        return bool(query_bindings(condition, facts, bindings))
    for source in condition.split(" & "):
        negated = source.startswith("not_")
        atom = _atom(source[4:] if negated else source)
        if atom is None: return False
        predicate, args = atom
        values = tuple(_value(arg, bindings) for arg in args)
        present = any(fact[0].lower() == predicate and tuple(fact[1:]) == values for fact in facts)
        if present == negated: return False
    return True


def query_bindings(query: str, facts: set[Fact], initial: dict[str, str] | None = None) -> list[dict[str, str]]:
    """Evaluate a conjunctive TextWorld predicate query by unifying facts.

    Variables use the grammar's `x:type` spelling.  Negated atoms filter a
    complete binding, mirroring the query form used by list comprehensions.
    """
    bindings = [dict(initial or {})]
    for source in query.split(" & "):
        negated = source.startswith("not_"); source = source[4:] if negated else source
        atom = _atom(source)
        if atom is None: return []
        predicate, args = atom
        next_bindings: list[dict[str, str]] = []
        for bound in bindings:
            matches = []
            for fact in facts:
                if fact[0].lower() != predicate or len(fact) - 1 != len(args): continue
                candidate = dict(bound)
                for variable, value in zip(args, fact[1:]):
                    if variable in candidate and candidate[variable] != value: break
                    candidate[variable] = value
                else: matches.append(candidate)
            if negated:
                if not matches: next_bindings.append(bound)
            else: next_bindings.extend(matches)
        bindings = next_bindings
    return bindings


def textworld_list(items: list[str]) -> str:
    """Match the grammar's `list_separator` and `list_last_separator` output."""
    if not items: return "nothing"
    if len(items) == 1: return items[0]
    if len(items) == 2: return f"{items[0]}, and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _indefinite(name: str) -> str:
    # The shipped ALFWorld entity infos use the literal article ``a`` even
    # before vowel-leading object ids (e.g. "a alarmclock").
    return "a"


def _call(source: str) -> tuple[str, list[str]]:
    source = source.strip()
    if "(" not in source: return source, []
    name, tail = source.split("(", 1)
    return name, [x.strip() for x in tail[:-1].split(",") if x.strip()]


def _key(rules: dict[str, list[Rule]], call: str) -> str:
    name, args = _call(call)
    return next((key for key in rules if _call(key)[0] == name and len(_call(key)[1]) == len(args)), call)


def _fields(text: str, bindings: dict[str, str], names: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        variable, field = match.groups(); raw = bindings.get(variable, variable)
        shown = names.get(raw, raw).lower()
        return shown if field == "name" else raw if field == "id" else _indefinite(shown)
    text = re.sub(r"\{([A-Za-z0-9_]+)\.name\s+or\s+\1\.id\}", lambda m: names.get(bindings.get(m.group(1), m.group(1)), bindings.get(m.group(1), m.group(1))).lower(), text)
    return re.sub(r"\{([A-Za-z0-9_]+)\.(name|id|indefinite)\}", replace, text)


def render_rule(rules: dict[str, list[Rule]], name: str, facts: set[Fact], bindings: dict[str, str], names: dict[str, str]) -> str:
    """Render the first enabled grammar rule with direct field substitutions."""
    rule = next((rule for rule in rules[name] if condition_holds(rule.condition, facts, bindings)), None)
    if rule is None: raise KeyError(f"no enabled grammar rule for {name}")
    return _fields(rule.rhs, bindings, names)


def render_macro(rules: dict[str, list[Rule]], name: str, facts: set[Fact], bindings: dict[str, str], names: dict[str, str], depth: int = 0) -> str:
    """Render full ALFWorld grammar: parameter macros and fact-query lists."""
    if depth > 32: raise RecursionError("grammar macro recursion")
    key = _key(rules, name); called, actual = _call(name); _, formal = _call(key)
    local = dict(bindings)
    for dst, src in zip(formal, actual): local[dst] = _value(src, bindings)
    choice = next((rule for rule in rules[key] if condition_holds(rule.condition, facts, local)), None)
    if choice is None: raise KeyError(f"no enabled grammar rule for {key}")
    text = choice.rhs
    pattern = re.compile(r"\[\{(.*?)\s*\|\s*([^\[\]]+?)\}\]")
    def list_value(match: re.Match[str]) -> str:
        expression, query = match.groups(); values = []
        items = query_bindings(query.strip(), facts, local)
        def rank(item: dict[str, str]) -> tuple[object, ...]:
            # Query order is TextWorld's entity order.  The first variable
            # introduced by *this* list query is the list item; inherited
            # action bindings (e.g. ``r`` in examineReceptacle) are not it.
            raw = next((value for variable, value in item.items()
                        if variable not in local and variable not in {"a", "l"}), "")
            shown = names.get(raw, raw).lower()
            prefix, _, suffix = shown.rpartition(" ")
            return (prefix if suffix.isdigit() else shown, -(int(suffix) if suffix.isdigit() else 0), shown)
        for item in sorted(items, key=rank):
            expression = expression.strip()
            if expression.startswith("#") and expression.endswith("#"):
                values.append(render_macro(rules, expression[1:-1], facts, item, names, depth + 1))
            else:
                chunks = [x.strip() for x in expression.split(" + ")]
                out = ""
                for chunk in chunks:
                    if len(chunk) >= 2 and chunk[0] in "\"'" and chunk[-1] == chunk[0]: out += chunk[1:-1]
                    else: out += _fields("{" + chunk + "}", item, names)
                values.append(out)
        return textworld_list(values)
    text = pattern.sub(list_value, text)
    macro = re.compile(r"#([A-Za-z0-9_.-]+(?:\([^#]*?\))?)#")
    text = macro.sub(lambda m: render_macro(rules, m.group(1), facts, local, names, depth + 1), text)
    return _fields(text, local, names)

