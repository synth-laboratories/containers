from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app
from synth_containers.platform.gold_rogue_world import GoldRogueWorld
from synth_containers.platform.react import OpenRouterReAct
from synth_containers.platform.runtime import runtime_for
from synth_containers.platform.targets import TARGETS


def test_policy_candidate_contract_is_published() -> None:
    schema = json.loads(
        (Path(__file__).parents[1] / "schemas/policy_candidate.v1.schema.json").read_text()
    )
    assert schema["$id"].endswith("policy_candidate.v1.schema.json")
    assert schema["properties"]["request"]["properties"]["contract"]["const"] == (
        "policy_candidate.v1"
    )
    assert schema["properties"]["response"]["properties"]["decision"]["required"] == ["actions"]


def test_rogue_target_publishes_native_harness_affordances() -> None:
    spec = TARGETS["rogue_react"]
    assert runtime_for(spec) is not None
    client = TestClient(create_compat_app(spec))
    info = client.get("/info").json()
    assert info["runtime_family"] == "rogue"
    assert info["environment_ref"] == "env:rogue_gold"
    assert info["max_episode_steps"] == 400
    affordances = info["affordances"]["environment"]
    assert affordances["bind_policy_config"] == "native"
    assert affordances["update_policy_code"] == "native"
    assert affordances["true_checkpoint"] == "native"
    assert affordances["restore"] == "native"
    assert affordances["simulate"] == "native"

    program = client.get("/program").json()
    assert program["version"] == "prompt_program.v1"
    assert program["program_id"] == "rogue.react.v1"
    assert program["target_modules"][0]["objective"] == "rogue_graded_progress"
    levers = client.get("/levers").json()["levers"]
    assert {item["lever_id"] for item in levers} == {
        "react_system_prompt",
        "policy_config",
        "harness_code",
    }

    planner = OpenRouterReAct(config_id="rogue_luna_medium", config={"environment_name": "Rogue"})
    assert "Rogue" in planner._messages[0]["content"]
    assert "Rogue" in planner._observation_prompt(
        {"observation_text": "|@..%|"}, ["h", "j", "k", "l", ">"]
    )


def test_rogue_gold_adapter_preserves_action_alphabet_and_checkpoint(monkeypatch) -> None:
    calls: list[tuple[str, str, dict | None]] = []
    responses = iter(
        [
            {
                "rollout_id": "rogue-1",
                "readout": {
                    "ascii": "|@..%|",
                    "valid_actions": ["h", "j", "k", "l", ">"],
                    "public": {},
                    "private": {"turn": 0, "total_reward": 0.0},
                },
                "nev_cursor": 0,
            },
            {
                "rollout_id": "rogue-1",
                "readout": {
                    "ascii": "|.@.%|",
                    "valid_actions": ["h", "j", "k", "l", ">"],
                    "public": {},
                    "private": {"turn": 1, "total_reward": 0.25},
                },
                "reward": 0.25,
                "nev_cursor": 1,
            },
            {"checkpoint_id": "cp-1", "blob": "encoded", "bytes": 7},
        ]
    )

    def fake_request(method: str, path: str, body: dict | None = None) -> dict:
        calls.append((method, path, body))
        return next(responses)

    world = GoldRogueWorld(max_steps=40)
    monkeypatch.setattr(world, "_request", fake_request)
    opened = world.reset(17)
    assert opened.valid_actions == ["h", "j", "k", "l", ">"]
    assert opened.observation["ascii"] == "|@..%|"
    stepped = world.step("l")
    assert stepped.reward == 0.25
    assert stepped.env_steps == 1
    assert world.checkpoint() == {"checkpoint_id": "cp-1", "blob": "encoded", "bytes": 7}
    assert calls[0][2]["task"]["objective"] == "descend"
    assert calls[1] == ("POST", "/rollouts/rogue-1/step", {"action": "l"})


def test_rogue_gold_adapter_uses_graded_progress_delta(monkeypatch) -> None:
    responses = iter(
        [
            {
                "rollout_id": "rogue-graded",
                "readout": {
                    "ascii": "|@..%|",
                    "valid_actions": ["l"],
                    "public": {},
                    "private": {"step_index": 0, "total_reward": 0.0},
                    "progress_metrics": {"synth_shaped_reward": 9.0},
                },
            },
            {
                "rollout_id": "rogue-graded",
                "readout": {
                    "ascii": "|.@.%|",
                    "valid_actions": ["l"],
                    "public": {},
                    "private": {"step_index": 1, "total_reward": 0.0},
                    "progress_metrics": {"synth_shaped_reward": 13.5},
                },
                "reward": 0.0,
            },
        ]
    )
    world = GoldRogueWorld(max_steps=40)
    monkeypatch.setattr(world, "_request", lambda *_args, **_kwargs: next(responses))

    opened = world.reset(7)
    stepped = world.step("l")
    assert opened.observation["progress_metrics"]["synth_shaped_reward"] == 9.0
    assert stepped.reward == 4.5
    assert world.previous_total_reward == 0.0
