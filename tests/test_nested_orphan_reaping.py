"""Orphaned trial containers must die on their own.

`NestedTrial.timeout_seconds` is enforced by the CLIENT: it kills `docker run`,
not the container. A client that is SIGKILLed — a killed tmux session, the OOM
killer, an agent tearing down — leaves the container running on the daemon, and
`reap_rollout` never fires because its `finally` never runs. A GameBench DEO
verifier is a 45-minute grade that spawns its own sandbox containers per
candidate, so a few orphans accumulating across attempts is enough to exhaust
the host. That is not hypothetical: five of them were found running at once.
"""

from __future__ import annotations

import subprocess
from typing import Any

from synth_containers.nested import DEADLINE_LABEL, NestedTrial, reap_expired

_PINNED = "img@sha256:" + "a" * 64


def test_every_trial_carries_its_own_deadline() -> None:
    trial = NestedTrial(
        image=_PINNED, rollout_id="r1", command=("true",), timeout_seconds=600, allow_nonzero=True
    )
    argv = trial.argv(parent="p1", name="n1")
    stamped = [argv[index + 1] for index, item in enumerate(argv) if item == "--label"]
    deadline = next(item for item in stamped if item.startswith(f"{DEADLINE_LABEL}="))
    seconds = int(deadline.split("=", 1)[1])
    # Roughly now + the timeout; the point is that it is ON the container, so a
    # later process can judge it without knowing who started it.
    import time

    assert time.time() + 300 < seconds <= time.time() + 601


def _fake_docker(monkeypatch: Any, listing: str, *, returncode: int = 0) -> list[list[str]]:
    calls: list[list[str]] = []

    class _Completed:
        def __init__(self) -> None:
            self.stdout = listing
            self.returncode = returncode

    def _run(argv: list[str], **_kwargs: Any) -> _Completed:
        calls.append(list(argv))
        return _Completed()

    monkeypatch.setattr(subprocess, "run", _run)
    return calls


def test_expired_containers_are_removed(monkeypatch) -> None:
    calls = _fake_docker(monkeypatch, "abc123 1000\ndef456 5000\n")

    removed = reap_expired(now=2000.0)

    assert removed == ["abc123"]
    assert calls[-1][:3] == ["docker", "rm", "-f"]
    assert calls[-1][3:] == ["abc123"]


def test_a_live_container_is_left_alone(monkeypatch) -> None:
    calls = _fake_docker(monkeypatch, "def456 5000\n")

    assert reap_expired(now=2000.0) == []
    # One listing call and no removal: reaping something still inside its
    # deadline would kill a running grade.
    assert len(calls) == 1


def test_an_unreachable_daemon_is_not_an_error(monkeypatch) -> None:
    """This runs before a rollout starts; it must never fail one."""

    _fake_docker(monkeypatch, "", returncode=1)
    assert reap_expired(now=2000.0) == []


def test_an_unparseable_deadline_is_not_treated_as_expired(monkeypatch) -> None:
    calls = _fake_docker(monkeypatch, "abc123 not-a-number\n")
    assert reap_expired(now=2000.0) == []
    assert len(calls) == 1
