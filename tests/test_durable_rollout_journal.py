"""Real origin process crashes must neither erase custody nor repeat effects."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import time
from uuid import uuid4

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from synth_containers.rollout_journal import RolloutJournal, RolloutJournalError


def initial(rollout_id: str) -> dict:
    return {"rollout_id": rollout_id, "status": "running", "success_status": "running",
            "created_at": "2026-10-09T00:00:00Z", "updated_at": "2026-10-09T00:00:00Z",
            "metadata": {}, "summary": {}}


def test_single_owner_and_runtime_scope(tmp_path):
    journal = RolloutJournal(tmp_path, "origin-a")
    instance = journal.origin_instance
    try:
        with pytest.raises(RolloutJournalError, match="owner_conflict"):
            RolloutJournal(tmp_path, "origin-a")
        journal.admit("uncertain", "digest-a", initial("uncertain"))
    finally:
        journal.close()
    with pytest.raises(RolloutJournalError, match="runtime_conflict"):
        RolloutJournal(tmp_path, "origin-b")
    reopened = RolloutJournal(tmp_path, "origin-a")
    try:
        assert reopened.origin_instance == instance
        assert reopened.get("uncertain")["status"] == "in_doubt"
        with pytest.raises(RolloutJournalError, match="identity_conflict"):
            reopened.admit("uncertain", "different", initial("uncertain"))
    finally:
        reopened.close()


def test_previous_journal_migration_and_corrupt_identity_refusal(tmp_path):
    database = tmp_path / "rollouts.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE identity(runtime_id TEXT PRIMARY KEY)")
        db.execute("INSERT INTO identity VALUES('origin-a')")
    journal = RolloutJournal(tmp_path, "origin-a")
    instance = journal.origin_instance
    journal.close()
    with sqlite3.connect(database) as db:
        db.execute("UPDATE identity SET origin_instance='invalid'")
    with pytest.raises(RolloutJournalError, match="identity_invalid"):
        RolloutJournal(tmp_path, "origin-a")
    with sqlite3.connect(database) as db:
        db.execute("UPDATE identity SET origin_instance=?", (instance,))
    journal = RolloutJournal(tmp_path, "origin-a")
    assert journal.origin_instance == instance
    journal.close()


def test_history_does_not_consume_unresolved_capacity(tmp_path):
    journal = RolloutJournal(tmp_path, "origin-a")
    try:
        for index in range(4097):
            key = f"finished-{index}"
            payload = initial(key)
            journal.admit(key, str(index), payload)
            journal.write(key, {**payload, "status": "completed", "success_status": "success"})
        journal.MAX_UNRESOLVED = 2
        journal.admit("pending-a", "a", initial("pending-a"))
        journal.admit("pending-b", "b", initial("pending-b"))
        with pytest.raises(RolloutJournalError, match="unresolved_limit"):
            journal.admit("pending-c", "c", initial("pending-c"))
        with pytest.raises(RolloutJournalError, match="terminal_conflict"):
            journal.write("finished-0", initial("finished-0"))
        assert journal.get("finished-0")["status"] == "completed"
        assert journal.get("pending-c") is None
    finally:
        journal.close()


def test_record_limit_refuses_before_admission(tmp_path):
    journal = RolloutJournal(tmp_path, "origin-a")
    try:
        payload = {**initial("large"), "summary": {"body": "x" * journal.MAX_RECORD_BYTES}}
        with pytest.raises(RolloutJournalError, match="record_limit"):
            journal.admit("large", "large", payload)
        assert journal.get("large") is None
    finally:
        journal.close()


@contextmanager
def origin_process(directory: Path):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    log = (directory / f"origin-{time.monotonic_ns()}.log").open("x")
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--serve", str(directory), str(port)],
                               stdout=log, stderr=subprocess.STDOUT)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5) as client:
            deadline = time.monotonic() + 30
            while True:
                assert process.poll() is None, "origin process failed before readiness"
                try:
                    if client.get("/health").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                assert time.monotonic() < deadline, "origin readiness timed out"
                time.sleep(0.025)
            yield process, client
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        log.close()
        assert process.poll() is not None


def effects(directory: Path) -> list[str]:
    file = directory / "effects.jsonl"
    return [] if not file.exists() else [json.loads(line)["rollout_id"] for line in file.read_text().splitlines()]


def origin_binding(client, payload):
    from synth_containers.rollout_lifecycle import request_identity
    from synth_containers.sdk import RolloutSubmissionPayload
    metadata = client.get("/metadata").json()
    return {"origin_instance": metadata["runtime"]["origin_instance"],
            "input_sha256": request_identity(RolloutSubmissionPayload.model_validate(payload).to_payload())}


def test_sigkill_after_completed_result_retains_exact_result(tmp_path):
    payload = {"rollout_id": "completed", "submission_mode": "sync", "value": 7}
    with origin_process(tmp_path) as (process, client):
        response = client.post("/rollouts", json=payload)
        assert response.status_code == 200
        completed = response.json()
        assert completed["status"] == "completed"
        if not os.environ.get("SYNTH_DURABLE_CONTROL_SDK"):
            binding = origin_binding(client, payload)
            receipt = client.get("/operations/rollouts/completed", params=binding)
            assert receipt.status_code == 200 and receipt.json()["record"] == completed
            assert client.post("/operations/rollouts/completed/cancel", json=binding).json()["record"] == completed
        process.kill()
        process.wait(timeout=5)
    with origin_process(tmp_path) as (_, client):
        lookup = client.get("/rollouts/completed")
        assert lookup.status_code == 200, "origin restart lost completed operation custody"
        assert lookup.json() == completed
        assert client.post("/rollouts", json=payload).json() == completed
        assert client.post("/rollouts", json={**payload, "value": 8}).status_code == 409
        assert effects(tmp_path) == ["completed"], "restart replayed a completed origin effect"
        if not os.environ.get("SYNTH_DURABLE_CONTROL_SDK"):
            assert origin_binding(client, payload) == binding
            assert client.get("/operations/rollouts/completed", params=binding).json()["record"] == completed
            assert client.post("/operations/rollouts/completed/cancel", json={**binding, "input_sha256": "f" * 64}).status_code == 409
            assert client.post("/operations/rollouts/completed/cancel", json={**binding, "origin_instance": str(uuid4())}).status_code == 409


def test_sigkill_after_effect_before_result_remains_uncertain_without_replay(tmp_path):
    payload = {"rollout_id": "uncertain", "submission_mode": "async", "block_after_effect": True}
    with origin_process(tmp_path) as (process, client):
        assert client.post("/rollouts", json=payload).status_code == 200
        deadline = time.monotonic() + 5
        while effects(tmp_path) != ["uncertain"]:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        if not os.environ.get("SYNTH_DURABLE_CONTROL_SDK"):
            binding = origin_binding(client, payload)
            refused = client.post("/operations/rollouts/uncertain/cancel", json={**binding, "input_sha256": "f" * 64})
            assert refused.status_code == 409
            assert not client.get("/rollouts/uncertain/state").json()["metadata"].get("cancel_requested")
        process.kill()
        process.wait(timeout=5)
    with origin_process(tmp_path) as (_, client):
        lookup = client.get("/rollouts/uncertain/state")
        assert lookup.status_code == 200, "origin restart lost uncertain operation custody"
        state = lookup.json()
        assert state["status"] == "in_doubt" and state["success_status"] == "unknown"
        assert state["metadata"]["origin_outcome"] == "unverified"
        if not os.environ.get("SYNTH_DURABLE_CONTROL_SDK"):
            assert origin_binding(client, payload) == binding
            receipt = client.post("/operations/rollouts/uncertain/cancel", json=binding).json()
            assert receipt["record"]["status"] == "in_doubt"
            assert receipt["record"]["metadata"]["cancel_requested"]
            assert receipt["record"]["metadata"]["resource_cleanup"] == "unverified"
        assert client.post("/rollouts", json=payload).json()["status"] == "in_doubt"
        cancellation = client.post("/rollouts/uncertain/terminate").json()
        assert cancellation["status"] == "in_doubt"
        assert cancellation["metadata"]["cancel_requested"]
        assert effects(tmp_path) == ["uncertain"], "restart repeated an uncertain side effect"
        sibling = client.post("/rollouts", json={"rollout_id": "sibling", "submission_mode": "sync"})
        assert sibling.status_code == 200 and sibling.json()["status"] == "completed"
        assert effects(tmp_path) == ["uncertain", "sibling"]


def serve(directory: Path, port: int):
    from synth_containers.sdk import Container
    import uvicorn
    container = Container("durable-origin", operation_directory=directory / "operations")

    @container.rollout
    async def callback(payload):
        with (directory / "effects.jsonl").open("a") as file:
            file.write(json.dumps({"rollout_id": payload["rollout_id"]}) + "\n")
            file.flush()
            os.fsync(file.fileno())
        if payload.get("block_after_effect"):
            await asyncio.Event().wait()
        return {**initial(payload["rollout_id"]), "status": "completed", "success_status": "success",
                "summary": {"value": payload.get("value", 0)}}

    control = os.environ.get("SYNTH_DURABLE_CONTROL_SDK")
    if control:
        import ast
        import synth_containers.sdk as sdk
        tree = ast.parse(Path(control).read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Container")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "fastapi")
        namespace = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), control, "exec"), sdk.__dict__, namespace)
        app = namespace["fastapi"](container)
    else:
        app = container.fastapi()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    assert sys.argv[1] == "--serve"
    serve(Path(sys.argv[2]), int(sys.argv[3]))
