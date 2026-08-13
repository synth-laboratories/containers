"""TargetRuntime umbrella.

Callers on CompatPlatform say "run this target." They do not branch on
Craftax / Harbor / dig.bench / OpenEnv. Those variants live as children
under this protocol.

Today: CraftaxRuntime, HarborRuntime, DigbenchRuntime, OpenEnvRuntime,
Banking77Runtime.
Later: rust gold HTTP, Docker Harbor, live dig.bench relay plug in here
without rewriting pins, leases, logs, or /reward.
"""

from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

from ..event_log import RolloutEventLog

if TYPE_CHECKING:
    from .state import CompatPlatform, RolloutPin
    from .targets import TargetSpec


class TargetRuntime(Protocol):
    """One episode / trial against a pinned world. Mutates pin + log."""

    def simulate(
        self,
        platform: CompatPlatform,
        pin: RolloutPin,
        log: RolloutEventLog,
    ) -> None: ...


def runtime_for(spec: TargetSpec) -> TargetRuntime:
    from .runtimes import runtime_for as _runtime_for

    return _runtime_for(spec)
