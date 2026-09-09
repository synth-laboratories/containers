"""Targeted affordances. Platform never names families in control flow.

See: backend/notes/specifications/tanha/references/synthstyle.md
     (general foundations, targeted affordances; one umbrella layer)
"""

from __future__ import annotations

from ..targets import TargetRuntimeKind, TargetSpec
from .digbench import DigbenchRuntime
from .harbor import HarborRuntime
from .openenv import OpenEnvRuntime

_BY_FAMILY = {
    TargetRuntimeKind.HARBOR: HarborRuntime(),
    TargetRuntimeKind.DIGBENCH: DigbenchRuntime(),
    TargetRuntimeKind.OPENENV: OpenEnvRuntime(),
}


def runtime_for(spec: TargetSpec):
    if spec.runtime is not None:
        return spec.runtime
    family = spec.runtime_family
    if family not in _BY_FAMILY:
        raise KeyError(f"unknown_runtime_family:{family}")
    return _BY_FAMILY[family]
