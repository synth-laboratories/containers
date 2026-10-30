import base64
import io
import tarfile

import pytest
from synth_containers.rhodes_cardcodebench import load_task, task_payload, bundle_digest


def fixture(tmp_path):
    public = tmp_path / "public"
    workspace = public / "environment/workspace"
    workspace.mkdir(parents=True)
    (workspace / "card.py").write_text("pass\n")
    (public / "instruction.md").write_text("Implement the card")
    (public / "task.toml").write_text("""[task]
name="cardcode/task"
[agent]
timeout_sec=600
[verifier]
timeout_sec=240
[environment]
cpus=2
memory_mb=2048
""")
    private = tmp_path / "verifier"
    private.mkdir()
    (private / "test.sh").write_text("#!/bin/sh\necho private-proof\n")
    digest = bundle_digest({"test.sh": (private / "test.sh").read_bytes()})
    return public, private, digest


def files(encoded):
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(encoded))) as archive:
        return {
            entry.name: archive.extractfile(entry).read() for entry in archive if entry.isfile()
        }


def test_private_verifier_only_enters_isolated_post_agent_phase(tmp_path):
    public, private, digest = fixture(tmp_path)
    task = load_task(
        public,
        private,
        image="registry/worker@sha256:" + "a" * 64,
        verifier_sha256=digest,
        agent_user="agent",
    )
    payload = task_payload(task, task_id="task", agent={"name": "codex"})
    build = files(payload["context_tar_base64"])
    assert set(build) == {"Dockerfile", "workspace/card.py"}
    assert all(b"private-proof" not in body for body in build.values())
    phases = payload["harbor_native_phases"]
    assert phases["verifier_environment_mode"] == "isolated"
    assert phases["allow_internet"] is False
    assert b"private-proof" in files(phases["tests_archive_b64"])["test.sh"]
    assert payload["lifecycle_profile"] == "cardcodebench"
    assert payload["limits"]["pids_limit"] == 128
    assert payload["metadata"]["verifier_bundle_sha256"] == digest


def test_unpinned_private_bundle_and_private_public_paths_refuse(tmp_path):
    public, private, digest = fixture(tmp_path)
    kwargs = dict(
        image="registry/worker@sha256:" + "a" * 64, verifier_sha256="0" * 64, agent_user="agent"
    )
    with pytest.raises(ValueError, match="digest_mismatch"):
        load_task(public, private, **kwargs)
    kwargs["verifier_sha256"] = digest
    (public / "environment/workspace/cases.json").write_text("hidden")
    with pytest.raises(ValueError, match="private_paths"):
        load_task(public, private, **kwargs)


def test_isolated_verifier_resource_requirement_is_reserved(tmp_path):
    public, private, digest = fixture(tmp_path)
    with (public / "task.toml").open("a") as stream:
        stream.write("\n[verifier.environment]\ncpus=4\nmemory_mb=3072\n")
    task = load_task(
        public,
        private,
        image="registry/worker@sha256:" + "a" * 64,
        verifier_sha256=digest,
        agent_user="agent",
    )
    payload = task_payload(task, task_id="task", agent={"name": "fixed_solution"})
    assert payload["limits"]["cpu_cores"] == 4
    assert payload["limits"]["memory_mb"] == 3072
