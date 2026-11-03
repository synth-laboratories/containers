"""Protected host recorder for a finite read-only, networkless stdio workload.

No candidate callback runs in this process. The child receives only exact stdin
bytes; the signing key stays in this owner process. Ordinary HTTP/agent images
must not advertise this execution profile unless they satisfy its fixed shape.
"""

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from contextlib import nullcontext


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def daemon_identity(docker, env):
    value = subprocess.run([docker, "info", "--format", "{{.ID}}"], capture_output=True, env=env, timeout=10)
    identity = value.stdout.decode(errors="strict").strip()
    if value.returncode or not 1 <= len(identity) <= 256 or any(char.isspace() for char in identity):
        raise FiniteRecorderRefusal("finite_owner_daemon_unavailable")
    return identity


def inspect_original(docker, name, env, *, expected_daemon_id=None, container_id=None):
    """Absence needs an exact daemon response, never a transport exit code."""
    def read():
        return subprocess.run([docker, "inspect", container_id or name], capture_output=True, env=env, timeout=10)
    def absent(value):
        error = value.stderr.decode(errors="replace").strip()
        return value.returncode != 0 and error in {
            "Error: No such object: " + (container_id or name),
            "error: no such object: " + (container_id or name),
            "Error response from daemon: No such container: " + (container_id or name),
        }
    identity = daemon_identity(docker, env)
    if expected_daemon_id is not None and identity != expected_daemon_id:
        raise FiniteRecorderRefusal("finite_original_daemon_differs")
    value = read()
    if value.returncode == 0:
        rows = json.loads(value.stdout)
        if (len(rows) != 1 or container_id is not None and rows[0].get("Id") != container_id
            or container_id is None and rows[0].get("Name") != "/" + name):
            raise FiniteRecorderRefusal("finite_original_identity_differs")
        if daemon_identity(docker, env) != identity:
            raise FiniteRecorderRefusal("finite_original_daemon_changed")
        return rows[0]
    if not absent(value):
        raise FiniteRecorderRefusal("finite_original_presence_unknown")
    if daemon_identity(docker, env) != identity or not absent(read()):
        raise FiniteRecorderRefusal("finite_original_absence_unconfirmed")
    return None


class FiniteRecorderRefusal(RuntimeError):
    pass


def write_original_input(selector, handle, body, cursor):
    """Send a bounded chunk and deliver EOF in the final write-ready event."""
    if cursor < len(body):
        try:
            cursor += os.write(handle.fileno(), body[cursor:cursor + 65536])
        except BlockingIOError:
            return cursor  # Readiness can change before the nonblocking write.
        except BrokenPipeError as error:
            raise FiniteRecorderRefusal("finite_input_not_consumed") from error
    if cursor == len(body):
        # A pipe need not emit another write-ready edge after a short write.
        selector.unregister(handle)
        handle.close()
    return cursor


def owner_command(docker, arguments, env):
    """Bound the local control client; an uncertain provider effect stays uncertain."""
    process = subprocess.Popen([docker, *arguments], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    try:
        output, _error = process.communicate(timeout=15)
    except subprocess.TimeoutExpired as error:
        # Reaping this local CLI never proves the original create was absent
        # or cancelled. Its durable create_pending intent still owns recovery.
        process.kill()
        try:
            process.communicate(timeout=2)
        except subprocess.TimeoutExpired as reaping_error:
            raise FiniteRecorderRefusal("finite_control_client_reaping_in_doubt") from reaping_error
        raise FiniteRecorderRefusal("finite_original_control_in_doubt") from error
    if process.returncode:
        raise FiniteRecorderRefusal("finite_owner_command_refused")
    return output


@dataclass(frozen=True)
class RecorderSigner:
    key_id: str
    cohort_id: str
    module_sha256: str
    private_key: object
    public_key: object

    def sign(self, body):
        return {"schema_version": "synth.rhodes-finite-recorder-proof.v1", "key_id": self.key_id,
            "cohort_id": self.cohort_id, "body": body,
            "signature_b64": base64.b64encode(self.private_key.sign(canonical(body))).decode()}


def operator_signer(cohort_id):
    """Explicit installed operator custody; no key generation or auto enrollment."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

    value = os.environ.get("RHODES_FINITE_RECORDER_COHORT_JSON")
    path = os.environ.get("RHODES_FINITE_RECORDER_PRIVATE_KEY_FILE")
    if not value or not path or not Path(path).is_absolute() or len(value) > 16384:
        raise FiniteRecorderRefusal("recorder_operator_binding_unavailable")
    cohort = json.loads(value)
    module_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if (cohort.get("schema_version") != "synth.rhodes-recorder-cohort.v1"
        or cohort.get("cohort_id") != cohort_id or cohort.get("module_sha256") != module_digest):
        raise FiniteRecorderRefusal("recorder_cohort_pin_differs")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > 16384):
            raise FiniteRecorderRefusal("recorder_key_custody_invalid")
        private_bytes = os.read(descriptor, 16385)
    finally:
        os.close(descriptor)
    private = serialization.load_pem_private_key(private_bytes, password=None)
    public = serialization.load_pem_public_key(cohort["public_key_pem"].encode())
    if not isinstance(private, Ed25519PrivateKey) or not isinstance(public, Ed25519PublicKey):
        raise FiniteRecorderRefusal("recorder_ed25519_required")
    encoding = serialization.Encoding.Raw
    fmt = serialization.PublicFormat.Raw
    if private.public_key().public_bytes(encoding, fmt) != public.public_bytes(encoding, fmt):
        raise FiniteRecorderRefusal("recorder_key_pin_differs")
    return RecorderSigner(cohort["key_id"], cohort_id, module_digest, private, public)


@dataclass(frozen=True)
class FiniteStdioResult:
    result: dict
    archive: bytes
    proof: dict
    stdout: bytes
    stderr: bytes
    journal: bytes
    image_digest: str


def seal_original_stdio(*, root, rollout_id, epoch, input_bytes, stdout, stderr,
                        journal, proof, recovery=False):
    """A trace transformation of actual retained observations, never execution."""
    from .tracing.capture.supervisor import CaptureSupervisor, SupervisorConfig
    from .tracing.models.identity import TraceIdentityV5, TraceProvenanceV5
    from .tracing.store.bundle import LocalTraceBundle

    body = proof["body"]
    transformation = {
        "schema_version": "synth.rhodes-finite-spool-transform.v1",
        "original_recorder_proof_sha256": hashlib.sha256(canonical(proof)).hexdigest(),
        "materializer_module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source": "fsynced_signed_original_stdio" if recovery else "live_owned_original_stdio",
    }
    with CaptureSupervisor(SupervisorConfig(bundle_root=Path(root),
        trace_key={"rollout": rollout_id, "epoch": epoch}, trace_id=rollout_id,
        upstream_base_url="http://127.0.0.1:9", interception="application",
        identity=TraceIdentityV5(rollout_id=rollout_id), workload_kind="other",
        provenance=TraceProvenanceV5("rhodes-isolated-stdio", "1",
            container_image_digest=body["image_digest"],
            transformation_chain=("signed-original-stdio-recovery",) if recovery else (),
            extra={"finite_materialization": transformation}))) as capture:
        for path, content in (("input.bin", input_bytes), ("stdout.bin", stdout), ("stderr.bin", stderr),
            ("stdio-journal.json", journal), ("recorder-proof.json", canonical(proof))):
            capture.collector.artifact(role="research_log", media_type="application/octet-stream",
                content=content, logical_name=path, visibility="private")
        capture.collector.event(event_type="environment.finite_stdio.terminal", payload=body)
        capture.finalize(status="completed")
    return LocalTraceBundle(Path(root)).archive_bytes()


def run_isolated_stdio(*, rollout_id, epoch, image_digest, command, input_bytes,
                       cpus, memory_bytes, timeout_seconds, maximum_output_bytes,
                       closure, signer, cancelled=None, on_state=None, on_terminal=None, on_owner=None,
                       deadline_timestamp=None):
    """One foreground-owned container; every observable output byte is retained."""
    docker = shutil.which("docker")
    if docker is None or not 0 < cpus <= 16 or not 0 < timeout_seconds <= 3600:
        raise FiniteRecorderRefusal("finite_runtime_or_limits_unavailable")
    remaining = timeout_seconds
    if deadline_timestamp is not None:
        if not math.isfinite(deadline_timestamp):
            raise FiniteRecorderRefusal("finite_deadline_invalid")
        remaining = min(remaining, deadline_timestamp - time.time())
    if remaining <= 0:
        raise FiniteRecorderRefusal("finite_deadline_exhausted")
    deadline = time.monotonic() + remaining
    if type(memory_bytes) is not int or not 128 * 1024 * 1024 <= memory_bytes <= 8 * 1024**3:
        raise FiniteRecorderRefusal("finite_memory_invalid")
    name = "rhodes-finite-" + hashlib.sha256((rollout_id + ":" + epoch).encode()).hexdigest()[:40]
    label = "ai.synth.rhodes.finite=" + rollout_id + ":" + epoch
    env = {name: os.environ[name] for name in ("PATH", "DOCKER_HOST", "DOCKER_CONFIG", "HOME") if name in os.environ}
    owner_daemon_id = daemon_identity(docker, env)

    def call(args):
        return owner_command(docker, args, env)

    # Refuse a previous original allocation; this runner never redispatches it.
    if inspect_original(docker, name, env, expected_daemon_id=owner_daemon_id) is not None:
        raise FiniteRecorderRefusal("finite_original_allocation_exists")
    journal = []
    journal_bytes = 2  # Canonical list brackets; bound adversarial tiny writes.
    streams = {"stdout": bytearray(), "stderr": bytearray()}
    created = False
    container_id = None
    child = None
    selector = None
    spool_value = os.environ.get("RHODES_FINITE_RECORDER_SPOOL_DIR")
    if not spool_value or not Path(spool_value).is_absolute():
        raise FiniteRecorderRefusal("finite_durable_spool_unavailable")
    spool = Path(spool_value)
    spool_info = spool.lstat()
    if (not stat.S_ISDIR(spool_info.st_mode) or spool_info.st_uid != os.getuid()
        or stat.S_IMODE(spool_info.st_mode) != 0o700):
        raise FiniteRecorderRefusal("finite_spool_custody_invalid")
    attempt = spool / name
    attempt.mkdir(mode=0o700, exist_ok=False)  # Existing original never redispatches.
    descriptor = os.open(spool, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

    def durable(path, body, *, append=False):
        flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | (os.O_APPEND if append else os.O_EXCL)
        descriptor = os.open(attempt / path, flags, 0o600)
        try:
            view = memoryview(body)
            while view:
                view = view[os.write(descriptor, view):]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        parent = os.open(attempt, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)

    durable("input.bin", input_bytes)
    def retain_lifecycle(value, *, container_id=None):
        body = {"schema_version": "synth.rhodes-finite-lifecycle.v1", "rollout_id": rollout_id,
            "execution_epoch": epoch, "container_name": name, "docker_daemon_id": owner_daemon_id,
            "state": value, "container_id": container_id, "recorder_module_sha256": signer.module_sha256}
        durable("lifecycle-" + value + ".json", canonical(signer.sign(body)))
    retain_lifecycle("owner_bound")
    if on_owner:
        on_owner({"docker_daemon_id": owner_daemon_id})
    with nullcontext(str(attempt)) as directory:
        empty = Path(directory) / "readonly-shm"
        empty.mkdir(mode=0o700)
        try:
            if time.monotonic() >= deadline or cancelled is not None and cancelled.is_set():
                raise FiniteRecorderRefusal("finite_cancelled_before_create")
            if on_state:
                on_state("create_pending")  # Durable original intent before the control effect.
            # Explicit stdin attachment makes Docker set StdinOnce: closing
            # the owned input pipe must deliver EOF, rather than retaining an
            # open stdin forever for a future unrelated attach client.
            allocated = call(["create", "--interactive", "--attach", "stdin", "--attach", "stdout", "--attach", "stderr",
                "--pull", "never", "--name", name, "--label", label,
                "--no-healthcheck", "--log-driver", "none",
                "--read-only", "--network", "none", "--ipc", "none", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges", "--user", "65534:65534",
                "--pids-limit", "64", "--cpus", str(cpus), "--memory", str(memory_bytes),
                "--memory-swap", str(memory_bytes), "--mount",
                f"type=bind,source={empty},target=/dev/shm,readonly", "--mount",
                f"type=bind,source={empty},target=/dev/mqueue,readonly", "--entrypoint", command[0],
                image_digest, *command[1:]])
            created = True
            allocated_id = allocated.decode("ascii").strip()
            if len(allocated_id) != 64 or any(char not in "0123456789abcdef" for char in allocated_id):
                raise FiniteRecorderRefusal("finite_original_allocation_identity_invalid")
            container_id = allocated_id
            retain_lifecycle("created", container_id=container_id)
            if on_owner:
                on_owner({"docker_daemon_id": owner_daemon_id, "container_id": container_id})
            if on_state:
                on_state("created")
            if time.monotonic() >= deadline or cancelled is not None and cancelled.is_set():
                raise FiniteRecorderRefusal("finite_cancelled_before_start")
            observed = json.loads(call(["inspect", name]))[0]
            image = json.loads(call(["image", "inspect", image_digest]))[0]
            daemon = json.loads(call(["info", "--format", "{{json .}}"])
                or b"null")
            architecture = {"x86_64": "amd64", "aarch64": "arm64"}.get(
                daemon.get("Architecture"), daemon.get("Architecture"))
            platform = {"os": image.get("Os"), "architecture": image.get("Architecture")}
            if (daemon.get("ID") != owner_daemon_id or observed.get("Id") != container_id
                or image.get("Id") != image_digest or platform["os"] != "linux" or daemon.get("OSType") != "linux"
                or platform["architecture"] != architecture
                or "sha256:" + hashlib.sha256(canonical(platform)).hexdigest() != closure.get("image_platform")):
                raise FiniteRecorderRefusal("finite_native_platform_or_image_differs")
            host, config = observed["HostConfig"], observed["Config"]
            mounts = observed.get("Mounts") or []
            if (observed.get("Image") != image_digest or host.get("ReadonlyRootfs") is not True
                or host.get("NetworkMode") != "none" or host.get("Privileged") is not False
                or set(host.get("CapDrop") or ()) != {"ALL"} or host.get("PidsLimit") != 64
                or host.get("Memory") != memory_bytes or host.get("MemorySwap") != memory_bytes
                or host.get("NanoCpus") != round(cpus * 1_000_000_000)
                or config.get("User") != "65534:65534"
                or config.get("OpenStdin") is not True or config.get("StdinOnce") is not True
                or config.get("AttachStdin") is not True or config.get("AttachStdout") is not True
                or config.get("AttachStderr") is not True or config.get("Tty") is not False
                or len(mounts) != 2 or {item.get("Destination") for item in mounts} != {"/dev/shm", "/dev/mqueue"}
                or any(item.get("RW") is not False or item.get("Source") != str(empty) for item in mounts)
                or (config.get("Healthcheck") or {}).get("Test") != ["NONE"]
                or (host.get("LogConfig") or {}).get("Type") != "none"
                or config.get("Entrypoint") != [command[0]] or (config.get("Cmd") or []) != list(command[1:])
                or not set(host.get("SecurityOpt") or ()) <= {"no-new-privileges", "no-new-privileges:true"}
                or not host.get("SecurityOpt")
                or config.get("Volumes") or host.get("Devices") or host.get("Binds")
                or host.get("PidMode") not in ("", "private") or host.get("IpcMode") != "none"):
                raise FiniteRecorderRefusal("finite_enforced_isolation_differs")
            for item in config.get("Env") or ():
                if item.partition("=")[0] not in {"PATH", "LANG", "LC_ALL", "TZ"}:
                    raise FiniteRecorderRefusal("finite_image_environment_refused")
            if time.monotonic() >= deadline or cancelled is not None and cancelled.is_set():
                raise FiniteRecorderRefusal("finite_cancelled_before_start")
            child = subprocess.Popen([docker, "start", "--attach", "--interactive", container_id],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
            retain_lifecycle("running", container_id=container_id)
            if on_state:
                on_state("running")
            selector = selectors.DefaultSelector()
            for handle, channel in ((child.stdout, "stdout"), (child.stderr, "stderr"), (child.stdin, "stdin")):
                os.set_blocking(handle.fileno(), False)
                selector.register(handle, selectors.EVENT_WRITE if channel == "stdin" else selectors.EVENT_READ, channel)
            cursor = 0
            while selector.get_map():
                if time.monotonic() >= deadline or cancelled is not None and cancelled.is_set():
                    raise FiniteRecorderRefusal("finite_execution_timeout")
                for key, _events in selector.select(0.1):
                    handle, channel = key.fileobj, key.data
                    if channel == "stdin":
                        cursor = write_original_input(selector, handle, input_bytes, cursor)
                    else:
                        data = os.read(handle.fileno(), 65536)
                        if not data:
                            selector.unregister(handle)
                            handle.close()
                            continue
                        streams[channel].extend(data)
                        if sum(len(value) for value in streams.values()) > maximum_output_bytes:
                            raise FiniteRecorderRefusal("finite_output_capture_bound_exceeded")
                        entry = {"sequence": len(journal) + 1, "channel": channel,
                            "data_b64": base64.b64encode(data).decode(),
                            "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
                        journal_bytes += len(canonical(entry)) + bool(journal)
                        if journal_bytes > 2 * 1024 * 1024 or len(journal) >= 65536:
                            raise FiniteRecorderRefusal("finite_journal_capture_bound_exceeded")
                        journal.append(entry)
                        durable(channel + ".bin", data, append=True)
                        durable("stream-journal.jsonl", canonical(journal[-1]) + b"\n", append=True)
            child.wait(timeout=max(0.1, deadline - time.monotonic()))
            terminal = json.loads(call(["inspect", container_id]))[0]
            if (terminal.get("Id") != container_id or terminal["State"].get("Running")
                or terminal["State"].get("ExitCode") != child.returncode):
                raise FiniteRecorderRefusal("finite_terminal_outcome_unverified")
            stdout, stderr = bytes(streams["stdout"]), bytes(streams["stderr"])
            body = {"schema_version": "synth.rhodes-finite-terminal.v1", "rollout_id": rollout_id,
                "execution_epoch": epoch, "profile": "isolated_stdio_v1", "image_digest": image_digest,
                "command": list(command), "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
                "stdout_sha256": hashlib.sha256(stdout).hexdigest(), "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
                "journal_sha256": hashlib.sha256(canonical(journal)).hexdigest(), "exit_code": child.returncode,
                "terminal_observed": True, "streams_drained": True, "filesystem_writes": "forbidden",
                "network": "none", "credential_injection": "none", "closure": closure,
                "docker_daemon_id": owner_daemon_id, "container_id": container_id,
                "resource_limits": {"cpu_cores": cpus, "memory_bytes": memory_bytes, "pids_limit": 64},
                "recorder_module_sha256": signer.module_sha256}
            proof = signer.sign(body)
            durable("stdio-journal.json", canonical(journal))
            durable("recorder-proof.json", canonical(proof))
            archive = seal_original_stdio(root=Path(directory) / "trace", rollout_id=rollout_id,
                epoch=epoch, input_bytes=input_bytes, stdout=stdout, stderr=stderr,
                journal=canonical(journal), proof=proof)
            durable("native-trace.zip", archive)
            try:
                result = json.loads(stdout)
            except ValueError:
                result = {"success": False, "error": "finite_result_not_json"}
            if not isinstance(result, dict):
                result = {"success": False, "error": "finite_result_not_object"}
            result.update(rollout_id=rollout_id, finite_exit_code=child.returncode)
            if child.returncode:
                result["success"] = False
            retained = FiniteStdioResult(result, archive, proof, stdout, stderr, canonical(journal), image_digest)
            if on_terminal:
                on_terminal(retained)
            return retained
        finally:
            try:
                if created:
                    observed = inspect_original(docker, name, env, expected_daemon_id=owner_daemon_id, container_id=container_id)
                    if observed is not None:
                        if (observed.get("Config", {}).get("Labels") or {}).get("ai.synth.rhodes.finite") != rollout_id + ":" + epoch:
                            raise FiniteRecorderRefusal("finite_cleanup_ownership_differs")
                        # After finding the named original, remove only its
                        # immutable ID even if another actor renames it.
                        container_id = observed["Id"]
                        subprocess.run([docker, "rm", "--force", container_id], capture_output=True, env=env, timeout=15)
                    if inspect_original(docker, name, env, expected_daemon_id=owner_daemon_id, container_id=container_id) is not None:
                        raise FiniteRecorderRefusal("finite_owner_cleanup_unconfirmed")
                    retain_lifecycle("cleanup_confirmed", container_id=container_id)
                    if on_state:
                        on_state("cleanup_confirmed")
            finally:
                # Reap only the local attach client. Its exit is never evidence
                # that the original provider allocation was removed.
                if selector is not None:
                    selector.close()
                if child is not None:
                    for handle in (child.stdin, child.stdout, child.stderr):
                        if handle is not None and not handle.closed:
                            handle.close()
                    try:
                        child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        child.terminate()
                        try:
                            child.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait(timeout=2)


def recover_original_stdio(*, rollout_id, epoch, state, cohort_id, allow_first_seal=False,
                           expected_daemon_id=None):
    """Read a completed original spool and stop its exact allocation; never run."""
    if state == "preflight":
        return None  # Durable create_pending never authorized an effect.
    if not isinstance(expected_daemon_id, str) or not expected_daemon_id:
        raise FiniteRecorderRefusal("finite_original_daemon_binding_missing")
    docker = shutil.which("docker")
    if docker is None:
        raise FiniteRecorderRefusal("finite_runtime_unavailable")
    name = "rhodes-finite-" + hashlib.sha256((rollout_id + ":" + epoch).encode()).hexdigest()[:40]
    env = {key: os.environ[key] for key in ("PATH", "DOCKER_HOST", "DOCKER_CONFIG", "HOME") if key in os.environ}
    spool = Path(os.environ.get("RHODES_FINITE_RECORDER_SPOOL_DIR", "/nonexistent"))
    if not spool.is_absolute():
        raise FiniteRecorderRefusal("finite_absolute_spool_required")
    info = spool.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise FiniteRecorderRefusal("finite_spool_custody_invalid")
    attempt = spool / name
    info = attempt.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise FiniteRecorderRefusal("finite_attempt_spool_custody_invalid")
    def retained(path, maximum):
        fd = os.open(attempt / path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            item = os.fstat(fd)
            if not stat.S_ISREG(item.st_mode) or item.st_uid != os.getuid() or stat.S_IMODE(item.st_mode) != 0o600 or item.st_size > maximum:
                raise FiniteRecorderRefusal("finite_retained_spool_invalid")
            body = bytearray()
            while chunk := os.read(fd, min(65536, maximum + 1 - len(body))):
                body.extend(chunk)
            if len(body) != item.st_size:
                raise FiniteRecorderRefusal("finite_retained_spool_changed")
            return bytes(body)
        finally:
            os.close(fd)
    def lifecycle(value, *, optional=False):
        try:
            proof = json.loads(retained("lifecycle-" + value + ".json", 65536))
        except FileNotFoundError:
            if optional:
                return None
            raise FiniteRecorderRefusal("finite_original_lifecycle_missing") from None
        body = verify_recorder_proof(proof, cohort_id=cohort_id)
        if (body.get("schema_version") != "synth.rhodes-finite-lifecycle.v1"
            or body.get("rollout_id") != rollout_id or body.get("execution_epoch") != epoch
            or body.get("container_name") != name or body.get("state") != value
            or body.get("docker_daemon_id") != expected_daemon_id):
            raise FiniteRecorderRefusal("finite_original_lifecycle_differs")
        identity = body.get("container_id")
        if identity is not None and (not isinstance(identity, str) or len(identity) != 64
            or any(char not in "0123456789abcdef" for char in identity)):
            raise FiniteRecorderRefusal("finite_original_allocation_identity_invalid")
        return body
    lifecycle("owner_bound")
    cleaned = lifecycle("cleanup_confirmed", optional=True)
    created = None if cleaned else lifecycle("created", optional=True)
    expected_id = (cleaned or created or {}).get("container_id")
    observed = inspect_original(docker, name, env, expected_daemon_id=expected_daemon_id, container_id=expected_id)
    if observed is None and state in {"admitted", "create_pending"} and not (created or cleaned):
        raise FiniteRecorderRefusal("finite_original_create_in_doubt")
    if observed is not None:
        if ((observed.get("Config", {}).get("Labels") or {}).get("ai.synth.rhodes.finite") != rollout_id + ":" + epoch
            or expected_id is not None and observed.get("Id") != expected_id):
            raise FiniteRecorderRefusal("finite_cleanup_ownership_differs")
        if cleaned:
            raise FiniteRecorderRefusal("finite_allocation_reappeared_after_cleanup")
        expected_id = observed["Id"]
        subprocess.run([docker, "rm", "--force", expected_id], capture_output=True, env=env, timeout=15)
        if inspect_original(docker, name, env, expected_daemon_id=expected_daemon_id, container_id=expected_id) is not None:
            raise FiniteRecorderRefusal("finite_owner_cleanup_unconfirmed")
    try:
        proof = json.loads(retained("recorder-proof.json", 1024 * 1024))
        body = verify_recorder_proof(proof, cohort_id=cohort_id)
        input_bytes = retained("input.bin", 256 * 1024)
        journal = retained("stdio-journal.json", 2 * 1024 * 1024)
        stdout = retained("stdout.bin", 256 * 1024) if (attempt / "stdout.bin").exists() else b""
        stderr = retained("stderr.bin", 256 * 1024) if (attempt / "stderr.bin").exists() else b""
        if (body.get("rollout_id") != rollout_id or body.get("execution_epoch") != epoch
            or body.get("docker_daemon_id") != expected_daemon_id
            or body.get("terminal_observed") is not True or body.get("streams_drained") is not True
            or body.get("profile") != "isolated_stdio_v1" or body.get("filesystem_writes") != "forbidden"
            or body.get("network") != "none" or body.get("credential_injection") != "none"
            or any(body.get(key) != hashlib.sha256(value).hexdigest() for key, value in (
                ("input_sha256", input_bytes), ("stdout_sha256", stdout),
                ("stderr_sha256", stderr), ("journal_sha256", journal)))):
            raise FiniteRecorderRefusal("finite_retained_terminal_differs")
        entries = json.loads(journal)
        rebuilt = {"stdout": bytearray(), "stderr": bytearray()}
        if not isinstance(entries, list) or len(entries) > 65536:
            raise FiniteRecorderRefusal("finite_retained_journal_invalid")
        for sequence, item in enumerate(entries, 1):
            data = base64.b64decode(item["data_b64"], validate=True)
            if (item.get("sequence") != sequence or item.get("channel") not in rebuilt
                or item.get("size_bytes") != len(data) or item.get("sha256") != hashlib.sha256(data).hexdigest()):
                raise FiniteRecorderRefusal("finite_retained_stream_journal_differs")
            rebuilt[item["channel"]].extend(data)
            if sum(map(len, rebuilt.values())) > 256 * 1024:
                raise FiniteRecorderRefusal("finite_retained_stream_bound_exceeded")
        if rebuilt != {"stdout": stdout, "stderr": stderr}:
            raise FiniteRecorderRefusal("finite_retained_stream_bytes_differ")
        try:
            archive = retained("native-trace.zip", 8 * 1024 * 1024)
        except FileNotFoundError:
            if not allow_first_seal:
                raise FiniteRecorderRefusal("finite_pinned_archive_missing")
            with tempfile.TemporaryDirectory(prefix="first-seal-", dir=attempt) as temporary:
                archive = seal_original_stdio(root=Path(temporary) / "trace", rollout_id=rollout_id,
                    epoch=epoch, input_bytes=input_bytes, stdout=stdout, stderr=stderr,
                    journal=journal, proof=proof, recovery=True)
            if len(archive) > 8 * 1024 * 1024:
                raise FiniteRecorderRefusal("finite_native_archive_bound_exceeded")
            # O_EXCL is the first immutable archive identity. A competing first
            # seal or a partial existing file is never silently overwritten.
            fd = os.open(attempt / "native-trace.zip", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            try:
                view = memoryview(archive)
                while view:
                    view = view[os.write(fd, view):]
                os.fsync(fd)
            finally:
                os.close(fd)
            fd = os.open(attempt, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        return FiniteStdioResult({}, archive, proof, stdout, stderr, journal, body["image_digest"])
    except FileNotFoundError:
        # Cleanup is proven, but a partial journal never becomes terminal evidence.
        return None


def verify_recorder_proof(proof, *, cohort_id):
    """Only the operator public pin verifies a retained recorder observation."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    value = os.environ.get("RHODES_FINITE_RECORDER_COHORT_JSON")
    if not value:
        raise FiniteRecorderRefusal("recorder_public_cohort_unavailable")
    cohort = json.loads(value)
    if (cohort.get("schema_version") != "synth.rhodes-recorder-cohort.v1"
        or cohort.get("cohort_id") != cohort_id or proof.get("cohort_id") != cohort_id
        or proof.get("key_id") != cohort.get("key_id")
        or proof.get("body", {}).get("recorder_module_sha256") != cohort.get("module_sha256")):
        raise FiniteRecorderRefusal("recorder_public_pin_differs")
    public = serialization.load_pem_public_key(cohort["public_key_pem"].encode())
    if not isinstance(public, Ed25519PublicKey):
        raise FiniteRecorderRefusal("recorder_public_key_invalid")
    public.verify(base64.b64decode(proof["signature_b64"], validate=True), canonical(proof["body"]))
    return proof["body"]
