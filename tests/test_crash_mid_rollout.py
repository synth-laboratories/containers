"""Lock: SIGKILL mid-simulate recovers crashed with real identity and refuses an unsealed log."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


_SERVER = """
import sys
import uvicorn
from synth_containers.platform import create_compat_app

storage_root, port, hold = sys.argv[1], int(sys.argv[2]), sys.argv[3]
app = create_compat_app(
    "craftax_engine",
    storage_root=storage_root,
    runtime_config={"simulate_hold_path": hold, "instance_id": "crash-mid-rollout"},
)
uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")
"""


def test_sigkill_mid_simulate_recovers_crashed_identity_and_refuses_unsealed_log(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "root"
    storage_root.mkdir()
    hold = tmp_path / "hold"
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-c", _SERVER, str(storage_root), str(port), str(hold)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    task_instance_id = "seed:7"
    rollout_id = "roll_crash_mid"
    try:
        deadline = time.time() + 20
        while time.time() < deadline:
            if proc.poll() is not None:
                err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
                raise RuntimeError(f"uvicorn exited early: {err}")
            try:
                httpx.get(f"http://127.0.0.1:{port}/health", timeout=0.2)
                break
            except httpx.HTTPError:
                time.sleep(0.05)
        else:
            raise RuntimeError("uvicorn did not start")

        post_error: list[BaseException] = []

        def _post() -> None:
            try:
                httpx.post(
                    f"http://127.0.0.1:{port}/rollouts",
                    json={
                        "rollout_id": rollout_id,
                        "task_instance_id": task_instance_id,
                        "policy_ref": {"harness": "react", "config": "luna_med"},
                        "telemetry": {"enabled": True, "transport": "sse"},
                    },
                    timeout=30.0,
                )
            except BaseException as exc:  # connection reset after SIGKILL is expected
                post_error.append(exc)

        worker = threading.Thread(target=_post, daemon=True)
        worker.start()
        hold_deadline = time.time() + 20
        while time.time() < hold_deadline:
            if hold.exists():
                break
            if proc.poll() is not None:
                err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
                raise RuntimeError(f"uvicorn died before hold: {err}")
            time.sleep(0.05)
        else:
            raise RuntimeError("simulate never held")
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)
        worker.join(timeout=2)
    finally:
        if proc.poll() is None:
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)

    recovered = create_compat_app("craftax_engine", storage_root=storage_root)
    pin = recovered.state.platform.pins[rollout_id]
    assert pin.status == "crashed"
    assert pin.task_instance_id == task_instance_id
    assert pin.seed == 7
    assert pin.terminal is True
    assert pin.policy_ref.get("harness") == "react"
    client = TestClient(recovered)
    trace = client.get(f"/rollouts/{rollout_id}/trace")
    assert trace.status_code == 409, trace.text
    body = trace.json()
    assert body["error"] == "unsealed_log"
    assert body["task_instance_id"] == task_instance_id
    assert not (storage_root / "seals" / f"{rollout_id}.trace-v5.json").is_file()
    lease_files = list((storage_root / "leases").glob("*.json"))
    assert lease_files == []


def test_orphaned_lease_without_identity_is_refused(tmp_path: Path) -> None:
    storage_root = tmp_path / "root"
    leases = storage_root / "leases"
    leases.mkdir(parents=True)
    rollout_id = "roll_legacy_lease"
    key = __import__("hashlib").sha256(rollout_id.encode("utf-8")).hexdigest()
    (leases / f"{key}.json").write_text(
        json.dumps(
            {
                "schema": "synth.containers.lease.v1",
                "target_id": "craftax_engine",
                "instance_id": "dead",
                "rollout_id": rollout_id,
                "owner_id": "ws-a",
                "owner_kind": "workshop_instance",
                "status": "running",
                "accepted_at": "2026-08-21T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="orphaned_lease_identity_missing"):
        create_compat_app("craftax_engine", storage_root=storage_root)
