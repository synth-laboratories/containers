from types import SimpleNamespace

import pytest

from synth_containers.platform.state import CompatPlatform


def _run(raiser, *, terminal, status):
    pin = SimpleNamespace(status=status, terminal=terminal)
    host = SimpleNamespace(_simulate=raiser)
    with pytest.raises(RuntimeError):
        CompatPlatform._simulate_or_fail(host, pin, None)
    return pin


def _boom(pin, log):
    raise RuntimeError("client disconnected")


def test_error_before_outcome_marks_rollout_failed():
    pin = _run(_boom, terminal=False, status="running")
    assert (pin.status, pin.terminal) == ("failed", True)


def test_error_after_outcome_keeps_recorded_status():
    pin = _run(_boom, terminal=True, status="completed")
    assert (pin.status, pin.terminal) == ("completed", True)
