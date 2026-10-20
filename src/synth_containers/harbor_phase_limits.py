"""Explicit native Harbor work and verification ceilings, frozen at staging."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class NativeHarborPhaseLimits:
    work_seconds: float
    verifier_seconds: float

    def __post_init__(self):
        for value in (self.work_seconds, self.verifier_seconds):
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 < value <= 86400
            ):
                raise ValueError(
                    "Native Harbor phase ceilings must be finite seconds from 0 through 86400, exclusive of zero"
                )

    def apply(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Narrow explicit source phase ceilings; never widen or infer missing ones."""
        source = {}
        resolved = {}
        for section, requested in (
            ("agent", self.work_seconds),
            ("verifier", self.verifier_seconds),
        ):
            value = manifest.get(section, {}).get("timeout_sec")
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "Native Harbor phase limit requires an explicit finite source timeout"
                )
            source[section] = value
            resolved[section] = min(value, requested)
        # Validate every phase before mutating the staged manifest.
        for section, value in resolved.items():
            manifest[section]["timeout_sec"] = value
        return {
            "source_seconds": source,
            "requested_seconds": {"agent": self.work_seconds, "verifier": self.verifier_seconds},
            "resolved_seconds": resolved,
            "enforcement": "harbor_phase_timeout",
            "clock": "phase_execution",
            "includes_queue_or_setup": False,
            "worker_failure_survival": False,
            "overall_deadline": None,
        }
