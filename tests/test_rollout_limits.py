import math

import pytest

from synth_containers.rollout_limits import (
    RolloutLimitKind,
    RolloutLimits,
    RolloutLimitSupervisor,
)


class Clock:
    value = 10.0

    def __call__(self):
        return self.value


def test_deadline_is_inclusive_and_stop_is_sticky():
    clock = Clock()
    supervisor = RolloutLimitSupervisor(RolloutLimits(5, 100), clock=clock)
    clock.value = 14.9
    assert supervisor.observe(output_bytes=100) is None
    clock.value = 15
    decision = supervisor.observe(output_bytes=100)
    assert decision.kind == RolloutLimitKind.WORK_TIME
    clock.value = 20
    assert supervisor.observe(output_bytes=0) is decision
    assert decision.to_payload()["enforcement"] == "observed_threshold"


def test_output_limit_does_not_reset_when_candidate_removes_files():
    clock = Clock()
    supervisor = RolloutLimitSupervisor(RolloutLimits(5, 100), clock=clock)
    assert supervisor.observe(output_bytes=100) is None
    decision = supervisor.observe(output_bytes=101)
    assert decision.kind == RolloutLimitKind.OUTPUT_BYTES
    assert decision.observed == 101
    assert supervisor.observe(output_bytes=0) is decision


def test_simultaneous_limits_retain_output_observation():
    clock = Clock()
    supervisor = RolloutLimitSupervisor(RolloutLimits(5, 100), clock=clock)
    clock.value = 15
    decision = supervisor.observe(output_bytes=101)
    assert decision.kind == RolloutLimitKind.WORK_TIME
    assert decision.output_high_water_bytes == 101


@pytest.mark.parametrize("timeout", [0, -1, True, math.inf, math.nan])
def test_invalid_deadlines_are_rejected(timeout):
    with pytest.raises(ValueError):
        RolloutLimits(timeout, 100)


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_output_limits_are_rejected(size):
    with pytest.raises(ValueError):
        RolloutLimits(5, size)


def test_regressed_clock_is_an_error_not_more_allowance():
    clock = Clock()
    supervisor = RolloutLimitSupervisor(RolloutLimits(5, 100), clock=clock)
    clock.value = 9
    with pytest.raises(ValueError, match="regressed"):
        supervisor.observe(output_bytes=0)
