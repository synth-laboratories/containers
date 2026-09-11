import pytest
from synth_containers.limit_capabilities import (
    LimitCapability, LimitDimension as D, LimitEnforcement as E,
    unsupported_limit_capabilities,
)


def test_survival_can_satisfy_weaker_requirement_but_not_reverse():
    strong = LimitCapability(D.WORK_TIME, E.OBSERVED_THRESHOLD, True)
    weak = LimitCapability(D.WORK_TIME, E.OBSERVED_THRESHOLD)
    assert unsupported_limit_capabilities((weak,), (strong,)) == ()
    assert unsupported_limit_capabilities((strong,), (weak,)) == (strong,)


def test_distinct_mechanisms_are_not_silently_equivalent():
    native = LimitCapability(D.OUTPUT_BYTES, E.NATIVE_CONTROL)
    sampled = LimitCapability(D.OUTPUT_BYTES, E.OBSERVED_THRESHOLD)
    assert unsupported_limit_capabilities((native,), (sampled,)) == (native,)
    assert unsupported_limit_capabilities((sampled,), (native,)) == (sampled,)


@pytest.mark.parametrize('dimension,enforcement,survival', [
    ('work_time', E.OBSERVED_THRESHOLD, False),
    (D.WORK_TIME, 'unknown', False),
    (D.WORK_TIME, E.OBSERVED_THRESHOLD, 'false'),
])
def test_malformed_requirements_are_rejected(dimension, enforcement, survival):
    with pytest.raises(ValueError):
        LimitCapability(dimension, enforcement, survival)


def test_required_capability_count_is_bounded():
    item = LimitCapability(D.WORK_TIME, E.OBSERVED_THRESHOLD)
    with pytest.raises(ValueError):
        unsupported_limit_capabilities((item,) * 33, ())
