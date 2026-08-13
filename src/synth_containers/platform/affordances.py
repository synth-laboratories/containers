"""Per-role affordance map. Default is unsupported. Bind fail-closed."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

LEVELS = ("native", "derived", "approximate", "unsupported")
LEVEL_RANK = {name: index for index, name in enumerate(LEVELS)}


def _level(value: Any) -> str:
    text = str(value or "unsupported").strip().lower()
    return text if text in LEVEL_RANK else "unsupported"


@dataclass
class AffordanceMap:
    """role -> affordance -> level. Unnamed affordances are unsupported."""

    by_role: dict[str, dict[str, str]] = field(default_factory=dict)

    def level(self, name: str, *, role: str = "environment") -> str:
        return _level(self.by_role.get(role, {}).get(name))

    def advertised(self) -> dict[str, dict[str, str]]:
        return {role: dict(items) for role, items in self.by_role.items()}

    def boolean(self, name: str, *, role: str = "environment") -> bool:
        return self.level(name, role=role) != "unsupported"

    def as_booleans(self, *, role: str = "environment") -> dict[str, bool]:
        return {name: self.boolean(name, role=role) for name in self.by_role.get(role, {})}


def bind_recipe(
    advertised: AffordanceMap,
    recipe: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return a refusal payload, or None if the recipe may start."""
    if not recipe:
        return None
    require = dict(recipe.get("require") or {})
    for name, wanted in require.items():
        role = "environment"
        affordance = str(name)
        if "." in affordance:
            role, affordance = affordance.split(".", 1)
        have = advertised.level(affordance, role=role)
        if isinstance(wanted, bool):
            need = "native" if wanted else "unsupported"
        else:
            need = _level(wanted)
        # Higher rank is weaker. Refuse when advertised is weaker than required.
        if LEVEL_RANK[have] > LEVEL_RANK[need]:
            return {
                "status": "refused",
                "affordance": affordance,
                "role": role,
                "advertised": have,
                "required": need,
            }
    return None
