import base64
import io
import tarfile
from pathlib import Path

from synth_containers.rhodes_harbor_daytona import load_task, oracle_script, task_payload

TASK_TOML = """
[metadata]
base_commit_hash = "abc123"
[verifier]
environment_mode = "separate"
timeout_sec = 1800.0
[verifier.environment]
cpus = 2
memory_mb = 8192
[agent]
timeout_sec = 5400.0
"""


def _corpus(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "corpus" / "tasks" / "demo"
    (root / "tests").mkdir(parents=True)
    (root / "solution").mkdir()
    (root / "task.toml").write_text(TASK_TOML)
    (root / "instruction.md").write_text("Fix it.")
    (root / "tests" / "Dockerfile").write_text("FROM agent")
    (root / "tests" / "test.sh").write_text("#!/bin/bash\n")
    (root / "tests" / "grader.py").write_text("print(1)\n")
    (root / "solution" / "solution.patch").write_text("diff --git a/x b/x\n")
    pins = tmp_path / "pins.toml"
    pins.write_text('["deepswe/demo"]\nagent = "registry/img@sha256:' + "a" * 64 + '"\n')
    return tmp_path / "corpus", pins


def test_task_runs_shared_with_hidden_tests_injected(tmp_path):
    corpus, pins = _corpus(tmp_path)
    task = load_task(corpus, pins, "demo")
    payload = task_payload(task, task_id="demo", agent={"name": "fixed_solution", "script_b64": "eA=="})
    native = payload["harbor_native_phases"]
    assert native["verifier_environment_mode"] == "shared"
    assert native["allow_internet"] is False and native["workdir"] == "/app"
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(native["tests_archive_b64"]))) as archive:
        assert sorted(archive.getnames()) == ["grader.py", "test.sh"]
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload["context_tar_base64"]))) as archive:
        assert archive.extractfile("Dockerfile").read() == f"FROM {task.image}\n".encode()
    assert payload["limits"]["timeout_s"] == 5400 + 1800 + 300


def test_oracle_exports_the_patch_against_the_base_commit(tmp_path):
    corpus, pins = _corpus(tmp_path)
    script = oracle_script(load_task(corpus, pins, "demo"))
    assert "git diff --binary abc123 HEAD > /logs/artifacts/model.patch" in script
    assert script.index("safe.directory") < script.index("git apply")
