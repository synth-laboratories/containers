# SPDX-License-Identifier: Apache-2.0
"""Executor boundary checks with a fake OCI process; no provider qualification."""

import subprocess

import pytest

from dataclasses import dataclass
from synth_containers import oci_trial as executor


@dataclass(frozen=True)
class TrialLimits:
    max_parallel_trials: int
    timeout_seconds: int
    cpus: float
    memory_mb: int
    max_output_bytes: int


class Process:
    returncode = None

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fixture", timeout)
        return self.returncode

    def poll(self):
        return self.returncode

    def kill(self):
        self.returncode = -9


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    process = Process()
    stops = []
    monkeypatch.setattr(executor.shutil, "which", lambda _: "/fixture/docker")
    monkeypatch.setattr(executor.subprocess, "Popen", lambda *a, **k: process)

    def stop(argv, **kwargs):
        assert kwargs["timeout"] == 10
        stops.append(argv)
        process.kill()
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(executor.subprocess, "run", stop)
    for name in ("input", "policy", "output"):
        (tmp_path / name).mkdir()
    request = executor.TrialRunRequest(
        trial_id="trial-fixture",
        image_reference="sha256:" + "ab" * 32,
        input_dir=tmp_path / "input",
        policy_dir=tmp_path / "policy",
        output_dir=tmp_path / "output",
        limits=TrialLimits(1, 30, 1, 128, 100),
        network="none",
    )
    return executor.OciTrialExecutor(), request, process, stops


def test_live_output_stop_has_honest_event_and_typed_outcome(runtime):
    driver, request, process, stops = runtime
    (request.output_dir / "oversize").write_bytes(b"x" * 101)
    events = []
    result = driver.run(
        request, on_event=events.append, should_cancel=lambda: False, heartbeat=lambda: None
    )
    assert result.output_limit_exceeded
    assert not result.timed_out
    assert not result.cancelled
    assert process.returncode == -9
    assert len(stops) == 1
    assert events[0]["limit_kind"] == "output_bytes"
    assert events[0]["enforcement"] == "observed_threshold"


def test_broken_observer_cannot_prevent_stop(runtime):
    driver, request, process, stops = runtime
    (request.output_dir / "oversize").write_bytes(b"x" * 101)

    def broken_observer(_):
        raise RuntimeError("observer disconnected")

    with pytest.raises(RuntimeError, match="observer disconnected"):
        driver.run(
            request, on_event=broken_observer, should_cancel=lambda: False, heartbeat=lambda: None
        )
    assert process.returncode == -9
    assert len(stops) == 1


def test_heartbeat_failure_still_stops_owned_execution(runtime):
    driver, request, process, stops = runtime

    def broken_heartbeat():
        raise RuntimeError("lease lost")

    with pytest.raises(RuntimeError, match="lease lost"):
        driver.run(
            request,
            on_event=lambda _: None,
            should_cancel=lambda: False,
            heartbeat=broken_heartbeat,
        )
    assert process.returncode == -9
    assert len(stops) == 1


def test_fast_exit_is_checked_for_output_overflow(runtime, monkeypatch):
    driver, request, process, stops = runtime

    def finish(timeout=None):
        (request.output_dir / "oversize").write_bytes(b"x" * 101)
        process.returncode = 0
        return 0

    monkeypatch.setattr(process, "wait", finish)
    result = driver.run(
        request, on_event=lambda _: None, should_cancel=lambda: False, heartbeat=lambda: None
    )
    assert result.output_limit_exceeded
    assert result.exit_code == 0
    assert stops == []


def test_unconfirmed_stop_is_not_reported_as_success(runtime, monkeypatch):
    driver, request, process, _ = runtime

    def timeout(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(executor.subprocess, "run", timeout)
    cancellation = iter([False, True])
    with pytest.raises(executor.ContainerRuntimeError, match="stop timed out"):
        driver.run(
            request,
            on_event=lambda _: None,
            should_cancel=lambda: next(cancellation),
            heartbeat=lambda: None,
        )
    assert process.returncode == -9  # Local CLI reaped; remote stop remains unknown.


def test_cancelled_admission_never_starts_a_process(runtime, monkeypatch):
    driver, request, _, stops = runtime

    def forbidden_start(*args, **kwargs):
        pytest.fail("cancelled trial spawned a process")

    monkeypatch.setattr(executor.subprocess, "Popen", forbidden_start)
    result = driver.run(
        request, on_event=lambda _: None, should_cancel=lambda: True, heartbeat=lambda: None
    )
    assert result.cancelled
    assert result.exit_code is None
    assert stops == []


def test_output_scan_ignores_symlink_targets_and_bounds_entries(tmp_path):
    root = tmp_path / "output"
    root.mkdir()
    (root / "real").write_bytes(b"123")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "large").write_bytes(b"x" * 1000)
    (root / "linked-dir").symlink_to(outside, target_is_directory=True)
    (root / "linked-file").symlink_to(outside / "large")
    assert executor._output_bytes(root) == 3
    with pytest.raises(executor.ContainerRuntimeError, match="entry bound"):
        executor._output_bytes(root, max_entries=1)


@pytest.mark.parametrize("reason", list(executor.StopFailureReason))
def test_stop_failure_exposes_resource_identity(runtime, monkeypatch, reason):
    driver, request, process, _ = runtime

    def fail(argv, **kwargs):
        if reason == executor.StopFailureReason.TIMEOUT:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return subprocess.CompletedProcess(argv, 1)

    monkeypatch.setattr(executor.subprocess, "run", fail)
    with pytest.raises(executor.ContainerStopUnconfirmed) as caught:
        driver._kill("owned-container", process)
    assert caught.value.to_payload() == {
        "container_id": "owned-container",
        "runtime": "docker",
        "stop_status": "unconfirmed",
        "reason": reason.value,
        "reconciliation_required": True,
    }


def test_stderr_tail_reads_only_the_requested_suffix(tmp_path):
    path = tmp_path / "stderr"
    with path.open("wb") as handle:
        handle.seek(10_000_000)
        handle.write(b"last message")
    assert executor._tail_text(path, limit=7) == "message"


@pytest.mark.parametrize("record", [b"{broken}\n", b"[]\n", b'{"x":NaN}\n', b"\xff\n", b'{"x":1}'])
def test_final_target_drain_refuses_corrupt_or_partial_records(tmp_path, record):
    import threading

    path = tmp_path / "events.jsonl"
    path.write_bytes(record)
    stop = threading.Event()
    stop.set()
    with pytest.raises(executor.ContainerEventStreamError):
        executor._tail_events(path, lambda _: None, stop)


def test_target_drain_has_a_record_bound(tmp_path):
    import threading

    path = tmp_path / "events.jsonl"
    path.write_bytes(b"x" * 33)
    stop = threading.Event()
    stop.set()
    with pytest.raises(executor.ContainerEventStreamError, match="record bound"):
        executor._tail_events(path, lambda _: None, stop, max_record_bytes=32)


def test_target_drain_pages_all_retained_records(tmp_path):
    import threading

    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"event":"fixture"}\n' * 600)
    stop = threading.Event()
    stop.set()
    events = []
    executor._tail_events(path, events.append, stop)
    assert len(events) == 600


def test_live_partial_record_waits_until_completed(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"event":')

    class Stop:
        done = False

        def is_set(self):
            return self.done

        def wait(self, timeout):
            with path.open("ab") as handle:
                handle.write(b'"fixture"}\n')
            self.done = True

    events = []
    executor._tail_events(path, events.append, Stop())
    assert events == [{"event": "fixture"}]


def test_event_thread_failure_reaches_owner_and_stops_process(runtime, monkeypatch):
    driver, request, process, stops = runtime
    failed = executor.threading.Event()

    def corrupt(*args, **kwargs):
        failed.set()
        raise ValueError("fixture corruption")

    monkeypatch.setattr(executor, "_tail_events", corrupt)

    def heartbeat():
        assert failed.wait(1)

    with pytest.raises(executor.ContainerEventStreamError):
        driver.run(
            request, on_event=lambda _: None, should_cancel=lambda: False, heartbeat=heartbeat
        )
    assert process.returncode == -9
    assert len(stops) == 1


def test_event_stream_replacement_is_visible(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"event":"first"}\n')

    class Stop:
        def is_set(self):
            return False

        def wait(self, timeout):
            replacement = tmp_path / "replacement"
            replacement.write_bytes(b'{"event":"next"}\n')
            replacement.replace(path)

    with pytest.raises(executor.ContainerEventStreamError, match="replaced"):
        executor._tail_events(path, lambda _: None, Stop())


@pytest.mark.parametrize(
    "change",
    [
        {"image_reference": "mutable:latest"},
        {"image_reference": "--bad@sha256:" + "ab" * 32},
        {"network": "typo"},
        {"secrets": {"BAD=KEY": "value"}},
    ],
)
def test_invalid_direct_request_never_starts(runtime, monkeypatch, change):
    from dataclasses import replace

    driver, request, _, _ = runtime
    monkeypatch.setattr(
        executor.subprocess, "Popen", lambda *a, **k: pytest.fail("invalid request started")
    )
    with pytest.raises(executor.ExecutionContractError):
        driver.run(
            replace(request, **change),
            on_event=lambda _: None,
            should_cancel=lambda: False,
            heartbeat=lambda: None,
        )


@pytest.mark.parametrize("cpu", [float("nan"), float("inf"), True, -1, 10**1000])
def test_invalid_cpu_is_rejected_before_execution(runtime, cpu):
    from dataclasses import replace

    _, request, _, _ = runtime
    with pytest.raises(executor.ExecutionContractError):
        executor.validate_trial_request(replace(request, limits=replace(request.limits, cpus=cpu)))
