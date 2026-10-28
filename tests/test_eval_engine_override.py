"""Plan-declared engine command/port reach the owned Docker guard; no Docker here."""

import json
from pathlib import Path

import pytest

from synth_containers.engine_guard import docker_run_argv
from synth_containers.eval_models import EvaluationPlanV1, EvaluationRunDeclarationV1
from synth_containers.eval_run import describe_evaluation
from synth_containers.native_engine import guard_request

D = "sha256:" + "a" * 64
GOLD = "/opt/gamebench/tasks/craftax-singleplayer/gold_rust/target/release/craftax_gold"
COMMAND = [GOLD, "--host", "0.0.0.0", "--port", "8098"]


def plan(**updates):
    return EvaluationPlanV1.model_validate({"benchmark": "craftax", "adapter": "local-gold-scripted-v1",
        "source_digest": D, "dependencies_digest": D, "scorer_digest": D, "image_digest": D,
        "task": {"world": {"classic": True}}, "limits": {"steps": 5, "calls": 8, "active_seconds": 60}, **updates})


def test_unset_engine_fields_leave_canonical_dump_and_digest_unchanged():
    p = plan()
    dumped = p.model_dump(mode="json")
    assert "engine_command" not in dumped and "engine_port" not in dumped
    assert p.digest == plan(engine_command=None, engine_port=None).digest
    run = EvaluationRunDeclarationV1(run_id="r", lineage_id="l", plan=p, candidates=())
    assert "engine_command" not in run.model_dump(mode="json")["plan"]
    assert EvaluationPlanV1.model_validate_json(p.model_dump_json()) == p
    assert p.effective_engine_port == 8098


def test_declared_engine_fields_are_part_of_identity_and_round_trip():
    p = plan(engine_command=COMMAND, engine_port=8098)
    assert p.engine_command == tuple(COMMAND)
    assert p.digest != plan().digest != plan(engine_port=8080).digest
    assert EvaluationPlanV1.model_validate_json(p.model_dump_json()) == p
    run = EvaluationRunDeclarationV1(run_id="r", lineage_id="l", plan=p, candidates=())
    assert EvaluationRunDeclarationV1.model_validate_json(run.model_dump_json()).plan == p


@pytest.mark.parametrize("bad", [
    {"engine_command": []}, {"engine_command": [""]}, {"engine_command": ["  "]},
    {"engine_command": ["a\nb"]}, {"engine_command": [1]}, {"engine_command": "craftax_gold"},
    {"engine_port": 0}, {"engine_port": 65536}, {"engine_port": "8098"}, {"engine_port": 80.0},
    {"engine_command": COMMAND, "adapter": "rhodes-v1"},
], ids=["empty", "blank", "space", "control", "nonstring", "bare-string",
        "port-zero", "port-high", "port-string", "port-float", "wrong-adapter"])
def test_engine_override_validation_refuses(bad):
    with pytest.raises(ValueError):
        plan(**bad)


def test_guard_request_and_docker_argv_shape(tmp_path):
    request = guard_request(plan(engine_command=COMMAND, engine_port=9000), owner="eval-x", receipt=tmp_path / "r.json")
    assert json.loads(json.dumps(request)) == request
    assert request["port"] == 9000 and request["command"] == COMMAND
    argv = docker_run_argv(request, "synth-eval-n")
    image = argv.index(D)
    assert argv[image + 1:] == COMMAND
    head = argv[:image]
    for fixed in (["--pull", "never"], ["--read-only"], ["--tmpfs", "/tmp:rw,size=67108864"],
                  ["--pids-limit", "128"], ["-p", "127.0.0.1::9000"], ["-e", "PORT=9000"],
                  ["--label", "ai.synth.eval.owner=eval-x"], ["--memory", "512m"]):
        assert any(head[i:i + len(fixed)] == fixed for i in range(len(head))), fixed


def test_default_guard_request_keeps_image_cmd_and_8098(tmp_path):
    request = guard_request(plan(), owner="o", receipt=Path("r.json"))
    assert request["command"] is None and request["port"] == 8098
    argv = docker_run_argv(request, "n")
    assert argv[-1] == D and "127.0.0.1::8098" in argv and "PORT=8098" in argv


@pytest.mark.parametrize("request_update", [{"port": 0}, {"port": "80"}, {"command": []}, {"command": [""]}],
                         ids=["port-zero", "port-string", "empty-command", "blank-arg"])
def test_guard_refuses_malformed_requests(request_update):
    request = {"owner": "o", "image": D, "cpu": 1, "memory_mb": 512, **request_update}
    with pytest.raises(ValueError):
        docker_run_argv(request, "n")


def test_describe_shows_effective_engine():
    default = describe_evaluation(plan(), ())["engine"]
    assert default == {"image": D, "port": 8098, "command": None, "command_source": "image-default"}
    declared = describe_evaluation(plan(engine_command=COMMAND), ())["engine"]
    assert declared["command"] == COMMAND and declared["command_source"] == "plan" and declared["port"] == 8098
