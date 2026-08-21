"""Harbor trial/verifier fold. ATIF is a projection of the log, not the log."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ...event_log import RolloutEventLog
from ..pin import RewardNode
from ..state import CompatPlatform, RolloutPin
from ..targets import TARGETS, ScriptNode
from .harbor_docker import DOCKER_ENVIRONMENT, run_docker_trial


class HarborRuntime:
    def simulate(self, platform: CompatPlatform, pin: RolloutPin, log: RolloutEventLog) -> None:
        if platform.spec.environment_ref == DOCKER_ENVIRONMENT:
            run_docker_trial(platform, pin, log)
            return
        if platform.spec.script_node == ScriptNode.HELDOUT_GATE:
            self._nested_child(platform, pin, log)
            return
        # Agent and verifier are distinct executions. C5 still requires
        # trial.planned + verifier.reward.txt on this parent log.
        log.append("trial.planned", {"instruction": "solve the public fixture"})
        log.append("trial.launched", {"sandbox": pin.environment_ref})
        log.append("span.agent.opened", {"role": "agent", "execution": "distinct"})
        log.append("tools", {"name": "bash", "stdout": "ok"})
        log.append("stdout", {"text": "wrote answer"})
        log.append("span.agent.closed", {"role": "agent"})
        log.append("span.verifier.opened", {"role": "verifier", "execution": "distinct"})
        log.append("verifier", {"script": "tests/test.sh", "reward.txt": 1.0})
        log.append("span.verifier.closed", {"role": "verifier"})
        pin.native_script_reward = 1.0
        pin.status = "completed"
        pin.terminal = True
        pin.usage = {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}
        log.append("status", {"status": "completed"})
        log.mark_closed()

    def _nested_child(self, platform: CompatPlatform, pin: RolloutPin, log: RolloutEventLog) -> None:
        child_id = f"{pin.rollout_id}:child"
        # Code-policy child so PUT /policy and POST /policy/restart are native
        # (`update_policy_code=native`). Parent /reward stays held-out gate +
        # baseline delta, not a copy of the child env-sum.
        # rollout_id is caller-controlled; never use it as a path component.
        child_platform = CompatPlatform(
            TARGETS["craftax_code_policy"],
            storage_root=platform.storage_root / "children" / platform._durable_key(child_id),
        )
        from ..http_requests import parse_create_rollout

        child = child_platform.start_rollout(
            parse_create_rollout(
                {
                    "rollout_id": child_id,
                    "telemetry": {"enabled": True, "transport": "sse"},
                    "task_instance_id": "seed:0",
                    "policy_ref": {"harness": "isolated_policy_process"},
                }
            )
        )
        pin.child_rollout_id = child["rollout_id"]
        child_stream = child["stream"]
        pin.child_resource_ref = {
            "schema": "synth.resource-ref.v1",
            "kind": "container_rollout",
            "id": child_id,
            "attributes": {
                "stream_id": child_stream["id"],
                "reward_url": child_stream["reward"]["url"],
            },
        }
        platform.logs[child_id] = child_platform.logs[child_id]
        platform.stream_bindings[child_id] = child_platform.stream_bindings[child_id]
        platform.pins[child_id] = child_platform.pins[child_id]
        platform.seals[child_id] = child_platform.seals[child_id]
        platform.artifacts.update(child_platform.artifacts)

        child_pin = child_platform.pins[child_id]
        child_score = float(sum(float(value) for value in child_pin.reward_signals if value is not None))
        # Fixture baseline distinct from the child env-sum. Pin a 0.1 improvement
        # so the held-out gate passes without copying env-sum onto the parent.
        baseline = child_score - 0.1 if child_score > 0 else 0.0
        delta = round(child_score - baseline, 10)
        gate_passed = delta > 0
        pin.hillclimb_nodes = (
            RewardNode(
                node_id="heldout_gate",
                kind="gate",
                authority="trusted_scorer",
                status="scored" if gate_passed else "gated",
                value=1.0 if gate_passed else None,
            ),
            RewardNode(
                node_id="baseline_delta",
                kind="aggregate",
                authority="trusted_scorer",
                status="scored",
                value=delta,
            ),
        )
        pin.native_script_reward = 1.0 if gate_passed else None

        log.append(
            "trial.planned",
            {"child_rollout_id": child_id, "child_resource_ref": pin.child_resource_ref},
        )
        log.append("trial.launched", {"author": "codex"})
        log.append("verifier", {"heldout_gate": gate_passed, "delta": delta})
        pin.status = "completed"
        pin.terminal = True
        log.append("status", {"status": "completed"})
        log.mark_closed()


def project_harbor_atif(envelopes: list[dict[str, Any]]) -> dict[str, Any]:
    """ATIF-shaped dict derived from Harbor trial/verifier envelopes."""
    semantic = [item for item in envelopes if not item.get("control")]
    planned = [item for item in semantic if item.get("kind") == "trial.planned"]
    launched = [item for item in semantic if item.get("kind") == "trial.launched"]
    tools = [item for item in semantic if item.get("kind") == "tools"]
    stdout = [item for item in semantic if item.get("kind") == "stdout"]
    verifier = next((item for item in reversed(semantic) if item.get("kind") == "verifier"), None)
    reward_txt = None
    if verifier is not None:
        raw_payload = verifier.get("payload")
        payload: dict[str, Any] = raw_payload if isinstance(raw_payload, dict) else {}
        value = payload.get("reward.txt")
        if isinstance(value, (int, float)):
            reward_txt = float(value)
    return {
        "schema": "ATIF-v1.7",
        "source": "projection",
        "trials": [
            {"planned": item.get("payload"), "sequence": item.get("sequence")}
            for item in planned
        ],
        "attempts": [
            {"launched": item.get("payload"), "sequence": item.get("sequence")}
            for item in launched
        ],
        "tools": [item.get("payload") for item in tools],
        "stdout": [item.get("payload") for item in stdout],
        "verifier": None if verifier is None else dict(verifier.get("payload") or {}),
        "reward.txt": reward_txt,
    }


def atif_is_projection(log_envelopes: list[dict[str, Any]]) -> bool:
    projected = project_harbor_atif(log_envelopes)
    mutated = deepcopy(projected)
    mutated["reward.txt"] = 0.0
    mutated["verifier"] = {"reward.txt": 0.0}
    return project_harbor_atif(log_envelopes)["reward.txt"] == projected["reward.txt"]
