"""Shared observed rollout limits; substrate adapters own stop actuators.

This first slice covers elapsed work and retained output. Sampling is not a
filesystem quota, provider spend reservation, or watchdog surviving worker loss.
A stop decision remains sticky even if files are deleted or another observer
attaches. Persisting/recovering supervision is a separate runtime responsibility.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional


class RolloutLimitKind(StrEnum):
    WORK_TIME = "work_time"
    OUTPUT_BYTES = "output_bytes"


@dataclass(frozen=True, slots=True)
class RolloutLimits:
    timeout_seconds: float
    max_output_bytes: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.timeout_seconds, bool)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        if type(self.max_output_bytes) is not int or self.max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be a positive integer")


@dataclass(frozen=True, slots=True)
class RolloutLimitDecision:
    """Observed reason to stop; never a claim that termination succeeded."""

    kind: RolloutLimitKind
    threshold: float | int
    observed: float | int
    elapsed_seconds: float
    output_high_water_bytes: int

    def to_payload(self) -> dict[str, str | int | float]:
        """Payload for an existing lifecycle event, not a new wire protocol."""
        return {
            "limit_kind": self.kind.value,
            "threshold": self.threshold,
            "observed": self.observed,
            "elapsed_seconds": self.elapsed_seconds,
            "output_high_water_bytes": self.output_high_water_bytes,
            "enforcement": "observed_threshold",
        }


class RolloutLimitSupervisor:
    """One execution owner's monotonic, sticky stop-decision state.

    Construct immediately before starting work. Poll from the owning execution
    loop, independently of trace/UI activity. This object is deliberately not a
    restart checkpoint: a recovered execution must restore its original limits
    and remaining allowance through a qualified recovery implementation.
    """

    def __init__(
        self, limits: RolloutLimits, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.limits = limits
        self._clock = clock
        self._started = clock()
        if not math.isfinite(self._started):
            raise ValueError("clock must return a finite monotonic value")
        self._last_sample = self._started
        self._output_high_water = 0
        self._decision: Optional[RolloutLimitDecision] = None

    @property
    def decision(self) -> Optional[RolloutLimitDecision]:
        return self._decision

    def observe(self, *, output_bytes: int) -> Optional[RolloutLimitDecision]:
        if type(output_bytes) is not int or output_bytes < 0:
            raise ValueError("output_bytes must be a non-negative integer")
        now = self._clock()
        if not math.isfinite(now) or now < self._last_sample:
            raise ValueError("monotonic clock regressed or became non-finite")
        self._last_sample = now
        self._output_high_water = max(self._output_high_water, output_bytes)
        if self._decision is not None:
            return self._decision
        elapsed = now - self._started
        # Time wins simultaneous observations deterministically. The output
        # high-water remains in the decision so the other observed fact survives.
        if elapsed >= self.limits.timeout_seconds:
            self._decision = RolloutLimitDecision(
                RolloutLimitKind.WORK_TIME,
                self.limits.timeout_seconds,
                elapsed,
                elapsed,
                self._output_high_water,
            )
        elif self._output_high_water > self.limits.max_output_bytes:
            self._decision = RolloutLimitDecision(
                RolloutLimitKind.OUTPUT_BYTES,
                self.limits.max_output_bytes,
                self._output_high_water,
                elapsed,
                self._output_high_water,
            )
        return self._decision
