"""SYN-5206: the engine guard waits on observed progress and never strands a container.

A fake Docker CLI (a script on disk, injected as ``docker_cli``) simulates a
loaded daemon: slow create, slow start, slow stop, and a guard killed mid-way.
The parent's post-handshake reporting-liveness bound is set far below each slow
phase, so a pass proves waiting is progress-bound rather than wall-bound. Guard
interpreter start-up precedes the hello handshake and is never counted as silence.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading

import pytest

from synth_containers.eval_models import EvaluationPlanV1
from synth_containers.native_engine import NativeEngineError, owned_gold_engine

D = "sha256:" + "a" * 64
FAKE_DOCKER = r'''
import json, os, signal, sys, time
from pathlib import Path
root = Path(__file__).parent
config = json.loads((root / "config.json").read_text())
state_path = root / "containers.json"
def load():
    return json.loads(state_path.read_text()) if state_path.exists() else {}
def save(state):
    state_path.write_text(json.dumps(state))
with open(root / "calls.log", "a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\n")
command, args = sys.argv[1], sys.argv[2:]
if command == "create":
    name = args[args.index("--name") + 1]
    labels = dict(args[i + 1].split("=", 1) for i, a in enumerate(args) if a == "--label")
    time.sleep(config.get("create_delay", 0))
    state = load(); state[name] = {"id": "id-" + name, "status": "created", "labels": labels}; save(state)
    print("id-" + name)
elif command == "start":
    time.sleep(config.get("start_delay", 0))
    if config.get("kill_guard_on_start"):
        os.kill(os.getppid(), signal.SIGKILL)
        sys.exit(1)
    state = load(); state[args[0]]["status"] = "running"; save(state)
elif command == "inspect":
    entry = load().get(args[-1])
    if entry is None:
        sys.exit(1)
    print(entry["status"])
elif command == "ps":
    key, value = args[args.index("--filter") + 1].removeprefix("label=").split("=", 1)
    for entry in load().values():
        if entry["labels"].get(key) == value:
            print(entry["id"])
elif command == "port":
    print("127.0.0.1:" + str(config["health_port"]))
elif command == "rm":
    time.sleep(config.get("rm_delay", 0))
    state = load(); state.pop(args[-1], None); save(state)
'''


class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def health_port():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Health)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()


def fake_docker(tmp_path: Path, health_port: int, **config) -> Path:
    root = tmp_path / "docker"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"health_port": health_port, **config}))
    (root / "fake_docker.py").write_text(FAKE_DOCKER)
    cli = root / "docker"
    cli.write_text(f"#!/bin/sh\nexec {sys.executable} {root / 'fake_docker.py'} \"$@\"\n")
    cli.chmod(0o755)
    return cli


def plan():
    return EvaluationPlanV1.model_validate({"benchmark": "craftax", "adapter": "local-gold-scripted-v1",
        "source_digest": D, "dependencies_digest": D, "scorer_digest": D, "image_digest": D,
        "task": {"world": {"classic": True}}, "limits": {"steps": 5, "calls": 8, "active_seconds": 60}})


def containers(cli: Path) -> dict:
    path = cli.parent / "containers.json"
    return json.loads(path.read_text()) if path.exists() else {}


def run_engine(cli: Path, receipt: Path) -> str:
    with owned_gold_engine(plan(), owner="eval-owner", receipt=receipt, docker_cli=str(cli),
                           heartbeat_seconds=.1, silence_seconds=.8) as url:
        assert len(containers(cli)) == 1
        return url


def states(receipt: Path) -> list[str]:
    return [t["state"] for t in json.loads(receipt.read_text())["transitions"]]


@pytest.mark.parametrize("phase", ["create_delay", "start_delay", "rm_delay"], ids=["slow-create", "slow-start", "slow-stop"])
def test_slow_docker_phase_is_awaited_and_removal_is_observed(tmp_path, health_port, phase):
    cli = fake_docker(tmp_path, health_port, **{phase: 2.5})
    receipt = tmp_path / "receipt.json"
    url = run_engine(cli, receipt)
    assert url == f"http://127.0.0.1:{health_port}"
    record = json.loads(receipt.read_text())
    assert record["cleanup"] == "removed" and record["last_observed_state"] == "absent"
    assert record["recorded_by"] == "guard" and record["label"] == "ai.synth.eval.owner=eval-owner"
    sequence = states(receipt)
    for expected in ("creating", "created", "starting", "running", "ready", "stopping", "removed"):
        assert expected in sequence
    assert sequence.index("created") < sequence.index("running") < sequence.index("ready") < sequence.index("removed")
    assert [t["observed"] for t in record["transitions"] if t["state"] in ("created", "running")] == ["created", "running"]
    assert containers(cli) == {}


def test_guard_dying_mid_way_records_unconfirmed_with_findable_name(tmp_path, health_port):
    cli = fake_docker(tmp_path, health_port, start_delay=.5, kill_guard_on_start=True)
    receipt = tmp_path / "receipt.json"
    with pytest.raises(NativeEngineError) as caught:
        run_engine(cli, receipt)
    assert caught.value.code == "native_engine_cleanup_unconfirmed"
    record = json.loads(receipt.read_text())
    assert record["cleanup"] == "unconfirmed" and record["recorded_by"] == "parent"
    assert record["reason"].startswith("guard_exited_") and record["last_observed_state"] == "starting"
    stranded = containers(cli)
    assert list(stranded) == [record["name"]] == [caught.value.name]
    assert stranded[record["name"]]["labels"]["ai.synth.eval.owner"] == "eval-owner"
    assert record["label"] == "ai.synth.eval.owner=eval-owner"


def test_guard_dying_before_handshake_is_exited_not_silent(tmp_path):
    receipt = tmp_path / "receipt.json"
    slow_then_die = (sys.executable, "-c", "import time, sys; time.sleep(1.5); sys.exit(3)")
    with pytest.raises(NativeEngineError) as caught:
        with owned_gold_engine(plan(), owner="eval-owner", receipt=receipt, heartbeat_seconds=.1,
                               silence_seconds=.2, guard_command=slow_then_die):
            pass
    assert caught.value.code == "native_engine_startup_failed"
    assert caught.value.cause == "guard_exited_3_before_hello"
    record = json.loads(receipt.read_text())
    assert record["cleanup"] == "not_allocated" and record["reason"] == "guard_exited_3"
    assert record["recorded_by"] == "parent" and record["name"] is None


def test_slow_guard_start_is_not_silence(tmp_path, health_port):
    """Interpreter start-up slower than the silence budget still reaches ready."""
    cli = fake_docker(tmp_path, health_port)
    receipt = tmp_path / "receipt.json"
    delayed = (sys.executable, "-c", "import time, runpy; time.sleep(1.5); "
               "runpy.run_module('synth_containers.engine_guard', run_name='__main__')")
    with owned_gold_engine(plan(), owner="eval-owner", receipt=receipt, docker_cli=str(cli),
                           heartbeat_seconds=.1, silence_seconds=.8, guard_command=delayed) as url:
        assert url == f"http://127.0.0.1:{health_port}"
    assert json.loads(receipt.read_text())["cleanup"] == "removed"


def test_slow_durable_cleanup_receipt_remains_observable(tmp_path, health_port):
    """Removal alone is not a durable receipt; keep reporting while it is saved."""
    import time

    cli = fake_docker(tmp_path, health_port)
    receipt = tmp_path / "receipt.json"
    slow_receipt = (sys.executable, "-c", """
import time
import synth_containers.engine_guard as guard
write = guard.durable_write
def delayed(*args, **kwargs):
    time.sleep(2.0)
    return write(*args, **kwargs)
guard.durable_write = delayed
guard.main()
""")
    try:
        with owned_gold_engine(plan(), owner="eval-owner", receipt=receipt, docker_cli=str(cli),
                               heartbeat_seconds=.1, silence_seconds=.8, guard_command=slow_receipt) as url:
            assert url == f"http://127.0.0.1:{health_port}"
    finally:
        # If an old parent raises early, let its detached guard finish its owned
        # receipt before pytest removes the temporary directory.
        deadline = time.monotonic() + 8
        while not receipt.exists() and time.monotonic() < deadline:
            time.sleep(.1)
    assert json.loads(receipt.read_text())["cleanup"] == "removed"
    assert containers(cli) == {}


def test_guard_silent_after_removal_remains_typed_and_unconfirmed(tmp_path):
    """A removed event without a durable receipt cannot become a false success."""
    import time

    receipt = tmp_path / "receipt.json"
    silent = (sys.executable, "-c", """
import json, sys, time
print(json.dumps({'event':'hello','name':'guard-test'}), flush=True)
sys.stdin.readline()
print(json.dumps({'event':'state','name':'guard-test','state':'ready','url':'http://127.0.0.1:1'}), flush=True)
sys.stdin.read()
print(json.dumps({'event':'state','name':'guard-test','state':'removed','observed':'absent'}), flush=True)
time.sleep(2.0)
""")
    try:
        with pytest.raises(NativeEngineError) as refused:
            with owned_gold_engine(plan(), owner="eval-owner", receipt=receipt, silence_seconds=.3,
                                   guard_command=silent):
                pass
        assert refused.value.code == "native_engine_cleanup_unconfirmed"
        assert refused.value.cause == "guard_silent_after_cleanup"
        record = json.loads(receipt.read_text())
        assert record["cleanup"] == "unconfirmed" and record["recorded_by"] == "parent"
        assert record["reason"] == "guard_silent_after_cleanup"
    finally:
        # The guard is deliberately not killed by the parent; give this fake
        # process time to finish before the temporary test workspace disappears.
        time.sleep(2.2)


def test_receipt_persistence_emits_progress_until_durable_write_finishes(tmp_path, monkeypatch):
    import time
    from synth_containers import engine_guard

    receipt = tmp_path / "receipt.json"
    original = engine_guard.durable_write
    def delayed(*args, **kwargs):
        time.sleep(.5)
        return original(*args, **kwargs)
    monkeypatch.setattr(engine_guard, "durable_write", delayed)
    reporter = engine_guard.GuardReporter(name="guard-test", heartbeat=.05)
    events = []
    monkeypatch.setattr(reporter, "emit", events.append)
    engine_guard.persist_receipt(reporter, receipt, {"cleanup": "removed"})
    assert json.loads(receipt.read_text()) == {"cleanup": "removed"}
    assert any(event.get("event") == "heartbeat" and event.get("phase") == "receipt" for event in events)
