"""Targeted affordances. Platform never names these in control flow.

See: backend/notes/specifications/tanha/references/synthstyle.md
     (general foundations, targeted affordances; one umbrella layer)
"""

from __future__ import annotations

from ..targets import TargetRuntimeKind, TargetSpec
from .banking77 import Banking77Runtime
from .craftax import CraftaxRuntime
from .digbench import DigbenchRuntime
from .harbor import HarborRuntime
from .healthbench import HealthBenchRuntime
from .openenv import OpenEnvRuntime

_BY_FAMILY = {
    TargetRuntimeKind.CRAFTAX: CraftaxRuntime(),
    TargetRuntimeKind.HARBOR: HarborRuntime(),
    TargetRuntimeKind.DIGBENCH: DigbenchRuntime(),
    TargetRuntimeKind.OPENENV: OpenEnvRuntime(),
    TargetRuntimeKind.BANKING77: Banking77Runtime(),
    TargetRuntimeKind.HEALTHBENCH: HealthBenchRuntime(),
}


def runtime_for(spec: TargetSpec):
    family = spec.runtime_family
    if family not in _BY_FAMILY:
        raise KeyError(f"unknown_runtime_family:{family}")
    return _BY_FAMILY[family]
