"""The isolated code-policy boundary receives public gold observations only."""

from synth_containers.platform.policy_process import IsolatedPolicyProcess


def test_isolated_policy_receives_symbolic_observation_without_private_gold_state() -> None:
    code = b"""
def choose_actions(*, observation_text, session, valid_actions, engine, seed, ply, readout):
    assert observation_text == "public readout"
    assert readout["observation"]["inventory"]["wood"] == 2
    assert readout["observation"]["player"]["health"] == 9
    assert "private" not in readout["observation"]
    assert "_frame_png" not in readout["observation"]
    assert "public" not in readout["observation"]
    return {"actions": ["noop"], "policy_reason": "public_symbolic_state"}
"""
    process = IsolatedPolicyProcess(code)
    try:
        actions = process.choose(
            {
                "observation_text": "public readout",
                "ascii": "P..",
                "valid_actions": ["noop", "right"],
                "inventory": {"wood": 2},
                "player": {"health": 9},
                "private": {"hidden_seed": 5000},
                "public": {"done": False},
                "_frame_png": b"private image bytes",
                "env_steps": 1,
            },
            ply=1,
        )
        assert actions == ["noop"]
    finally:
        process.close()
