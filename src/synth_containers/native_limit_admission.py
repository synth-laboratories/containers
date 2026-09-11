"""Fail-closed whole-attempt capability admission for native Harbor consumers.

Scoped tmpfs/storage-driver controls and bounded stdout collection do not enforce
all workspace/bind-mounted artifacts. Until an adapter supplies whole-attempt
actuator custody, it must not accept a corresponding required capability.
"""

from __future__ import annotations

from .limit_capabilities import (
    LimitCapability,
    LimitDimension,
    LimitEnforcement,
    unsupported_limit_capabilities,
)


def require_native_limit_capabilities(value: object = ()) -> list[dict[str, str | bool]]:
    if not isinstance(value, (tuple, list)) or len(value) > 32:
        raise ValueError("Native required limit capabilities must contain at most 32 entries")
    required = []
    for item in value:
        if isinstance(item, LimitCapability):
            required.append(item)
            continue
        if not isinstance(item, dict) or set(item) != {
            "dimension",
            "enforcement",
            "survives_supervisor_loss",
        }:
            raise ValueError("Native required limit capability has an invalid contract")
        required.append(
            LimitCapability(
                LimitDimension(item["dimension"]),
                LimitEnforcement(item["enforcement"]),
                item["survives_supervisor_loss"],
            )
        )
    # Empty is deliberate: phase timers and mount-specific actuators are described
    # by their own receipts, not promoted to a whole-attempt guarantee. In
    # particular WORKSPACE_BYTES includes writable bind mounts and output bytes
    # includes files outside process stdout; neither has a universal actuator.
    unsupported = unsupported_limit_capabilities(tuple(required), ())
    if unsupported:
        names = sorted(
            {item.dimension.value + ":" + item.enforcement.value for item in unsupported}
        )
        raise ValueError("Native whole-attempt limit actuator unavailable: " + ", ".join(names))
    return [item.to_payload() for item in required]
