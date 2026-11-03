"""Owned recorder checks; real Docker cases require an explicit cached image."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import selectors
import sys
import time
from types import SimpleNamespace
from uuid import uuid4

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from synth_containers import finite_stdio_capture as recorder


def test_final_input_write_delivers_eof_without_second_ready_event():
    reader, writer = os.pipe()
    handle = os.fdopen(writer, "wb", buffering=0)
    selector = selectors.DefaultSelector()
    selector.register(handle, selectors.EVENT_WRITE)
    os.set_blocking(reader, False)
    try:
        body = b'{"original":true}'
        assert recorder.write_original_input(selector, handle, body, 0) == len(body)
        assert handle.closed and not selector.get_map()
        assert os.read(reader, 1024) == body
        assert os.read(reader, 1024) == b""  # Actual pipe EOF, no second selector event.
    finally:
        selector.close()
        handle.close()
        os.close(reader)


def test_timed_out_control_client_is_reaped_but_original_stays_in_doubt(monkeypatch):
    class Client:
        returncode = None
        killed = False
        calls = 0

        def communicate(self, *, timeout):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(["docker", "create"], timeout)
            assert self.killed and timeout == 2
            return b"", b""

        def kill(self):
            self.killed = True

    client = Client()
    monkeypatch.setattr(recorder.subprocess, "Popen", lambda *args, **kwargs: client)
    with pytest.raises(recorder.FiniteRecorderRefusal, match="finite_original_control_in_doubt"):
        recorder.owner_command("docker", ["create"], {})
    assert client.killed and client.calls == 2


def test_control_client_never_inherits_open_ambient_worker_input():
    reader, writer = os.pipe()
    original = os.dup(0)
    os.write(writer, b"ambient-worker-input-must-not-be-forwarded")
    try:
        os.dup2(reader, 0)
        output = recorder.owner_command(sys.executable,
            ["-c", "import sys; print(repr(sys.stdin.buffer.read()))"], dict(os.environ))
        assert output == b"b''\n"
    finally:
        os.dup2(original, 0)
        os.close(original)
        os.close(reader)
        os.close(writer)


@pytest.fixture
def signer(tmp_path, monkeypatch):
    private = Ed25519PrivateKey.generate()
    key_path = tmp_path / "test-only-recorder.pem"
    key_path.write_bytes(private.private_bytes(serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    key_path.chmod(0o600)
    spool = tmp_path / "spool"
    spool.mkdir(mode=0o700)
    cohort = {"schema_version": "synth.rhodes-recorder-cohort.v1", "cohort_id": "test-only",
        "key_id": "test-only", "module_sha256": hashlib.sha256(Path(recorder.__file__).read_bytes()).hexdigest(),
        "public_key_pem": private.public_key().public_bytes(serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo).decode()}
    monkeypatch.setenv("RHODES_FINITE_RECORDER_COHORT_JSON", json.dumps(cohort))
    monkeypatch.setenv("RHODES_FINITE_RECORDER_PRIVATE_KEY_FILE", str(key_path))
    monkeypatch.setenv("RHODES_FINITE_RECORDER_SPOOL_DIR", str(spool))
    try:
        yield recorder.operator_signer("test-only")
    finally:
        key_path.unlink(missing_ok=True)


def test_operator_binding_rejects_source_pin_change(signer, monkeypatch):
    cohort = json.loads(os.environ["RHODES_FINITE_RECORDER_COHORT_JSON"])
    cohort["module_sha256"] = "0" * 64
    monkeypatch.setenv("RHODES_FINITE_RECORDER_COHORT_JSON", json.dumps(cohort))
    with pytest.raises(recorder.FiniteRecorderRefusal, match="recorder_cohort_pin_differs"):
        recorder.operator_signer("test-only")


def test_inspection_transport_failure_is_not_absence(monkeypatch):
    monkeypatch.setattr(recorder, "daemon_identity", lambda *args: "original-daemon")
    monkeypatch.setattr(recorder.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(returncode=1, stdout=b"", stderr=b"Cannot connect to Docker daemon"))
    with pytest.raises(recorder.FiniteRecorderRefusal, match="finite_original_presence_unknown"):
        recorder.inspect_original("docker", "original", {}, expected_daemon_id="original-daemon")


def test_inspection_changed_daemon_refuses_before_original_lookup(monkeypatch):
    monkeypatch.setattr(recorder, "daemon_identity", lambda *args: "different-daemon")
    monkeypatch.setattr(recorder.subprocess, "run", lambda *args, **kwargs: pytest.fail("wrong daemon lookup"))
    with pytest.raises(recorder.FiniteRecorderRefusal, match="finite_original_daemon_differs"):
        recorder.inspect_original("docker", "original", {}, expected_daemon_id="original-daemon")


@pytest.mark.parametrize("message", ["Error: No such object: ", "error: no such object: "])
def test_absence_requires_two_exact_reads_from_same_daemon(monkeypatch, message):
    calls = []
    monkeypatch.setattr(recorder, "daemon_identity", lambda *args: "original-daemon")
    def read(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=1, stdout=b"[]", stderr=(message + "original\n").encode())
    monkeypatch.setattr(recorder.subprocess, "run", read)
    assert recorder.inspect_original("docker", "original", {}, expected_daemon_id="original-daemon") is None
    assert calls == [["docker", "inspect", "original"], ["docker", "inspect", "original"]]


def test_exact_container_id_survives_rename(monkeypatch):
    identity = "a" * 64
    calls = []
    monkeypatch.setattr(recorder, "daemon_identity", lambda *args: "original-daemon")
    def read(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout=json.dumps([{"Id": identity, "Name": "/renamed"}]).encode())
    monkeypatch.setattr(recorder.subprocess, "run", read)
    assert recorder.inspect_original("docker", "original", {}, expected_daemon_id="original-daemon",
        container_id=identity)["Name"] == "/renamed"
    assert calls == [["docker", "inspect", identity]]


def test_no_effect_preflight_recovery_never_contacts_provider(monkeypatch):
    monkeypatch.setattr(recorder.shutil, "which", lambda *args: pytest.fail("preflight has no provider effect"))
    assert recorder.recover_original_stdio(rollout_id="original", epoch="epoch", state="preflight",
        cohort_id="test-only") is None


def test_expired_original_deadline_refuses_before_daemon_read(signer, monkeypatch):
    monkeypatch.setattr(recorder, "daemon_identity", lambda *args: pytest.fail("expired deadline"))
    with pytest.raises(recorder.FiniteRecorderRefusal, match="finite_deadline_exhausted"):
        recorder.run_isolated_stdio(rollout_id="expired", epoch="epoch", image_digest="sha256:" + "a" * 64,
            command=("/bin/cat",), input_bytes=b"{}", cpus=1, memory_bytes=256 * 1024**2,
            timeout_seconds=30, deadline_timestamp=1, maximum_output_bytes=1024, closure={}, signer=signer)


def test_owner_ack_latency_consumes_original_deadline_without_create(signer, monkeypatch):
    monkeypatch.setattr(recorder, "daemon_identity", lambda *args: "original-daemon")
    monkeypatch.setattr(recorder, "inspect_original", lambda *args, **kwargs: None)
    monkeypatch.setattr(recorder.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("expired owner ACK dispatched"))
    with pytest.raises(recorder.FiniteRecorderRefusal, match="finite_cancelled_before_create"):
        recorder.run_isolated_stdio(rollout_id="slow-owner", epoch="epoch", image_digest="sha256:" + "a" * 64,
            command=("/bin/cat",), input_bytes=b"{}", cpus=1, memory_bytes=256 * 1024**2,
            timeout_seconds=0.01, maximum_output_bytes=1024, closure={}, signer=signer,
            on_owner=lambda value: time.sleep(0.02))


@pytest.fixture
def docker_execution(signer):
    image = os.environ.get("RHODES_FINITE_TEST_IMAGE")
    if not image:
        pytest.skip("requires explicit owned-test cached image digest")
    details = json.loads(subprocess.check_output(["docker", "image", "inspect", image], timeout=10))[0]
    platform = {"os": details["Os"], "architecture": details["Architecture"]}
    rollout_id, epoch = "finite-test-" + uuid4().hex, uuid4().hex
    name = "rhodes-finite-" + hashlib.sha256((rollout_id + ":" + epoch).encode()).hexdigest()[:40]
    values = dict(rollout_id=rollout_id, epoch=epoch, image_digest=details["Id"], command=("/bin/cat",),
        input_bytes=b'{"success":true,"reward":1}', cpus=1, memory_bytes=256 * 1024**2,
        timeout_seconds=60, maximum_output_bytes=4096,
        closure={"image_platform": "sha256:" + hashlib.sha256(recorder.canonical(platform)).hexdigest()}, signer=signer)
    try:
        yield values, name
    finally:
        # Only this random test's exact labelled allocation may be removed.
        observed = subprocess.run(["docker", "inspect", name], capture_output=True, timeout=10)
        if observed.returncode == 0:
            container = json.loads(observed.stdout)[0]
            assert container["Config"]["Labels"]["ai.synth.rhodes.finite"] == rollout_id + ":" + epoch
            subprocess.run(["docker", "rm", "--force", container["Id"]], capture_output=True, timeout=15, check=True)


def test_real_finite_original_terminal_and_same_archive_recovery(docker_execution):
    values, name = docker_execution
    owner = {}
    phases = []
    result = recorder.run_isolated_stdio(**values, on_owner=lambda value: owner.update(value), on_state=phases.append)
    assert result.result["success"] is True
    assert result.stdout == values["input_bytes"]
    assert phases == ["create_pending", "created", "running", "cleanup_confirmed"]
    assert result.proof["body"]["docker_daemon_id"] == owner["docker_daemon_id"]
    assert result.proof["body"]["container_id"] == owner["container_id"]
    recovered = recorder.recover_original_stdio(rollout_id=values["rollout_id"], epoch=values["epoch"],
        state="cleanup_confirmed", cohort_id="test-only", expected_daemon_id=owner["docker_daemon_id"])
    assert recovered.archive == result.archive
    assert recovered.proof == result.proof
    assert recovered.journal == result.journal
    with pytest.raises(recorder.FiniteRecorderRefusal, match="finite_original_lifecycle_differs"):
        recorder.recover_original_stdio(rollout_id=values["rollout_id"], epoch=values["epoch"],
            state="cleanup_confirmed", cohort_id="test-only", expected_daemon_id="different-daemon")
    with pytest.raises(FileExistsError):
        recorder.run_isolated_stdio(**values)


def test_real_lost_created_ack_preserves_cleanup_original(docker_execution):
    values, name = docker_execution
    owner = {}
    def acknowledge(value):
        owner.update(value)
        if "container_id" in value:
            raise TimeoutError("fixture lost create acknowledgment")
    with pytest.raises(TimeoutError, match="fixture lost create"):
        recorder.run_isolated_stdio(**values, on_owner=acknowledge)
    assert recorder.recover_original_stdio(rollout_id=values["rollout_id"], epoch=values["epoch"],
        state="create_pending", cohort_id="test-only", expected_daemon_id=owner["docker_daemon_id"]) is None


def test_real_lost_terminal_ack_recovers_exact_original(docker_execution):
    values, name = docker_execution
    owner = {}
    retained = []
    def acknowledge(value):
        retained.append(value)
        raise TimeoutError("fixture lost terminal acknowledgment")
    with pytest.raises(TimeoutError, match="fixture lost terminal"):
        recorder.run_isolated_stdio(**values, on_owner=lambda value: owner.update(value), on_terminal=acknowledge)
    recovered = recorder.recover_original_stdio(rollout_id=values["rollout_id"], epoch=values["epoch"],
        state="running", cohort_id="test-only", expected_daemon_id=owner["docker_daemon_id"])
    assert recovered.archive == retained[0].archive
    assert recovered.proof == retained[0].proof


def test_real_first_archive_seal_recovery_from_signed_original(docker_execution, monkeypatch, tmp_path):
    from synth_containers.tracing.inspection import inspect_trace_input

    values, name = docker_execution
    owner = {}
    seal = recorder.seal_original_stdio
    def crash_before_seal(**kwargs):
        raise RuntimeError("fixture crash before first archive seal")
    monkeypatch.setattr(recorder, "seal_original_stdio", crash_before_seal)
    with pytest.raises(RuntimeError, match="fixture crash before first"):
        recorder.run_isolated_stdio(**values, on_owner=lambda value: owner.update(value))
    monkeypatch.setattr(recorder, "seal_original_stdio", seal)
    recovery = dict(rollout_id=values["rollout_id"], epoch=values["epoch"], state="running",
        cohort_id="test-only", expected_daemon_id=owner["docker_daemon_id"])
    with pytest.raises(recorder.FiniteRecorderRefusal, match="finite_pinned_archive_missing"):
        recorder.recover_original_stdio(**recovery, allow_first_seal=False)
    recovered = recorder.recover_original_stdio(**recovery, allow_first_seal=True)
    assert recovered.stdout == values["input_bytes"]
    assert recorder.recover_original_stdio(**recovery).archive == recovered.archive
    archive = tmp_path / "recovered-original.zip"
    archive.write_bytes(recovered.archive)
    inspected = inspect_trace_input(archive)
    assert inspected.validation.valid is True
    assert inspected.traces[0].verified is True
