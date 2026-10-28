"""SYN-5209 follow-up: declared egress ``none`` is an internal network plus a forwarding sidecar.

Uses the fake Docker CLI from the guard progress tests. The guard must create an
owned internal network, start the engine there with nothing published, start a
sidecar on bridge + internal network publishing loopback, verify that placement by
inspection, and remove sidecar, engine and network, confirming each.
"""

from __future__ import annotations

import json

import pytest

from synth_containers.engine_egress import EgressNames, egress_capability, sidecar_create_argv
from synth_containers.engine_guard import docker_create_argv
from synth_containers.eval_models import EvaluationPlanV1
from synth_containers.native_engine import NativeEngineError, owned_gold_engine
from tests.test_engine_guard_progress import D, containers, fake_docker, health_port, states  # noqa: F401

OWNER = "eval-owner"


def plan(egress="none"):
    return EvaluationPlanV1.model_validate({"benchmark": "craftax", "adapter": "local-gold-scripted-v1",
        "source_digest": D, "dependencies_digest": D, "scorer_digest": D, "image_digest": D, "egress": egress,
        "task": {"world": {"classic": True}}, "limits": {"steps": 5, "calls": 8, "active_seconds": 60}})


def networks(cli):
    path = cli.parent / "networks.json"
    return json.loads(path.read_text()) if path.exists() else {}


def calls(cli):
    return [json.loads(line) for line in (cli.parent / "calls.log").read_text().splitlines()]


def run(cli, receipt, egress="none"):
    with owned_gold_engine(plan(egress), owner=OWNER, receipt=receipt, docker_cli=str(cli),
                           heartbeat_seconds=.1, silence_seconds=4.0) as url:
        live = containers(cli)
        return url, live, networks(cli)


def test_none_runs_engine_on_internal_network_behind_sidecar(tmp_path, health_port):
    cli = fake_docker(tmp_path, health_port)
    receipt = tmp_path / "receipt.json"
    url, live, nets = run(cli, receipt)
    assert url == f"http://127.0.0.1:{health_port}"
    engine = next(name for name in live if not name.endswith("-fwd"))
    names = EgressNames(engine)
    assert set(live) == {engine, names.sidecar} and nets[names.network]["internal"] is True
    assert "-p" not in live[engine]["argv"] and live[engine]["host"]["NetworkMode"] == names.network
    assert set(live[names.sidecar]["networks"]) == {"bridge", names.network}
    assert live[names.sidecar]["argv"][-4:] == ["-e", "nc", "10.9.0.2", "8098"]
    record = json.loads(receipt.read_text())
    egress = record["egress"]
    assert record["cleanup"] == "removed" and record["remaining"] == [] and record["code"] is None
    assert egress["declared"] == "none" and egress["observed"] == "none(internal+sidecar)"
    assert record["limits"]["egress"] == "none(internal+sidecar)"
    assert record["containers"] == {"engine": "id-" + engine, "sidecar": "id-" + names.sidecar}
    assert egress["network"] == {"name": names.network, "id": "net-" + names.network}
    assert egress["forwarding"] == {"host": f"127.0.0.1:{health_port}", "sidecar": f"{names.sidecar}:8098",
                                    "engine": "10.9.0.2:8098"}
    assert egress["observed_placement"]["engine_networks"] == [names.network]
    assert egress["observed_placement"]["engine_port_bindings"] is None
    assert "egress_verified" in states(receipt)
    removals = [c for c in calls(cli) if c[0] == "rm" or c[:2] == ["network", "rm"]]
    assert [c[-1] for c in removals] == [names.sidecar, engine, names.network]
    assert containers(cli) == {} and networks(cli) == {}


def test_unrestricted_bridge_keeps_published_engine_and_no_network(tmp_path, health_port):
    cli = fake_docker(tmp_path, health_port)
    receipt = tmp_path / "receipt.json"
    _, live, nets = run(cli, receipt, egress="unrestricted_bridge")
    (engine,) = live
    assert nets == {} and "-p" in live[engine]["argv"] and live[engine]["host"]["NetworkMode"] == "bridge"
    record = json.loads(receipt.read_text())
    assert record["egress"]["observed"] == "unrestricted_bridge" and record["containers"]["sidecar"] is None
    assert record["cleanup"] == "removed"


def test_missing_sidecar_image_is_typed_unsupported_and_allocates_nothing(tmp_path, health_port):
    cli = fake_docker(tmp_path, health_port, sidecar_image_missing=True)
    receipt = tmp_path / "receipt.json"
    with pytest.raises(NativeEngineError) as caught:
        run(cli, receipt)
    assert caught.value.code == "egress_isolation_unsupported" and caught.value.stage == "egress"
    record = json.loads(receipt.read_text())
    assert record["egress"]["reason"] == "sidecar image alpine:3.20 not present locally"
    assert record["cleanup"] == "removed" and not any(c[0] == "create" for c in calls(cli))
    assert not any(c[:2] == ["network", "create"] for c in calls(cli))


def test_engine_also_on_bridge_is_egress_mismatch_and_everything_removed(tmp_path, health_port):
    cli = fake_docker(tmp_path, health_port, engine_networks_extra=["bridge"])
    receipt = tmp_path / "receipt.json"
    with pytest.raises(NativeEngineError) as caught:
        run(cli, receipt)
    assert caught.value.code == "native_engine_limits_mismatch" and caught.value.stage == "egress"
    assert "engine_networks" in caught.value.cause
    record = json.loads(receipt.read_text())
    assert record["stage"] == "egress" and record["egress"]["observed"] is None
    assert record["egress"]["observed_placement"]["engine_networks"][0] == "bridge"
    assert record["cleanup"] == "removed" and containers(cli) == {} and networks(cli) == {}
    assert "ready" not in states(receipt)


def test_network_left_behind_is_unconfirmed_naming_it(tmp_path, health_port):
    cli = fake_docker(tmp_path, health_port, network_rm_fails=True)
    receipt = tmp_path / "receipt.json"
    with pytest.raises(NativeEngineError) as caught:
        run(cli, receipt)
    assert caught.value.code == "native_engine_cleanup_unconfirmed"
    record = json.loads(receipt.read_text())
    assert record["cleanup"] == "unconfirmed" and containers(cli) == {}
    assert record["remaining"] == [record["egress"]["network"]["name"] + ":present"]


def test_argv_shapes_and_capability_detection(tmp_path, health_port):
    names = EgressNames("synth-eval-x")
    request = {"owner": OWNER, "image": D, "cpu": 1, "memory_mb": 512, "port": 8098, "egress": "none"}
    engine = docker_create_argv(request, names.engine, network=names.network)
    assert engine[engine.index("--network") + 1] == names.network and "-p" not in engine
    sidecar = sidecar_create_argv("docker", names, "o=x", "a", 8098, "10.0.0.2", "sha256:" + "b" * 64)
    assert sidecar[sidecar.index("--network") + 1] == "bridge" and "127.0.0.1::8098" in sidecar
    assert "--pull" in sidecar and sidecar[sidecar.index("--pull") + 1] == "never"
    with pytest.raises(ValueError, match="egress_network_missing"):
        docker_create_argv(request, names.engine)
    assert egress_capability(str(fake_docker(tmp_path, health_port)))["none"]["supported"] is True
    (tmp_path / "m").mkdir()
    missing = egress_capability(str(fake_docker(tmp_path / "m", health_port, sidecar_image_missing=True)))
    assert missing["none"]["code"] == "egress_isolation_unsupported" and missing["supported"] == ["unrestricted_bridge"]
