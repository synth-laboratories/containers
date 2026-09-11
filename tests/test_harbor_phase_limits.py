import math

import pytest

from synth_containers.harbor_phase_limits import NativeHarborPhaseLimits


def test_work_and_verification_are_separate_and_cannot_widen_source():
    manifest = {"agent": {"timeout_sec": 3600}, "verifier": {"timeout_sec": 200}}
    receipt = NativeHarborPhaseLimits(600, 300).apply(manifest)
    assert manifest == {"agent": {"timeout_sec": 600}, "verifier": {"timeout_sec": 200}}
    assert receipt["source_seconds"] == {"agent": 3600, "verifier": 200}
    assert receipt["resolved_seconds"] == {"agent": 600, "verifier": 200}
    assert receipt["worker_failure_survival"] is False
    assert receipt["overall_deadline"] is None


@pytest.mark.parametrize("value", [None, True, "600", 0, -1, math.inf, math.nan, 86401])
def test_invalid_limits_never_mean_unlimited(value):
    with pytest.raises(ValueError):
        NativeHarborPhaseLimits(value, 300)


def test_invalid_verifier_leaves_agent_source_unchanged():
    manifest = {"agent": {"timeout_sec": 3600}}
    with pytest.raises(ValueError, match="explicit finite source"):
        NativeHarborPhaseLimits(600, 300).apply(manifest)
    assert manifest == {"agent": {"timeout_sec": 3600}}
