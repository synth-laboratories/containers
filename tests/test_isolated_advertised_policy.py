"""The /info code-policy identity must be usable unchanged by an orchestrator."""
from dataclasses import replace

from synth_containers.platform import create_compat_app
from synth_containers.platform.http_requests import parse_create_rollout
from synth_containers.platform.targets import OPENENV_ECHO, PolicySeed


def test_declared_empty_isolated_seed_is_accepted_but_binding_is_refused(tmp_path):
    spec = replace(OPENENV_ECHO, default_policy_harness="isolated_policy_process",
                   policy_seeds=(PolicySeed("heuristic", "isolated_policy_process", {}),),
                   admission=lambda *_: {"admission_reached": True})
    platform = create_compat_app(spec, storage_root=tmp_path).state.platform
    # Stop before engine dispatch: this test verifies real admission, while the
    # native Craftax/DungeonGrid acceptance runs exercise the actual engines.
    def start(config):
        return platform.start_rollout(parse_create_rollout({
            "rollout_id": f"check-{config}", "policy_ref": {"harness":"isolated_policy_process", "config":config},
            "telemetry":{"enabled":True,"transport":"sse"},
        }))
    assert start("heuristic") == {"admission_reached": True}
    assert start("arbitrary")["error"] == "bind_refused"
    platform.policy_configs["heuristic"].config["unexpected"] = True
    # Only immutable declarations are accepted, never newly registered aliases.
    assert start("heuristic")["error"] == "bind_refused"
