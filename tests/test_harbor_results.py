import json

import pytest

from synth_containers.harbor_results import (
    HarborResultError,
    HarborTrialRecord,
    finite_number,
    read_result_object,
)


def trial(**updates):
    return {
        "task_name": "fixture",
        "trial_name": "attempt",
        "exception_info": None,
        "agent_result": {},
        "verifier_result": {"rewards": {"reward": 0.0}},
        **updates,
    }


def test_real_zero_is_preserved_and_missing_reward_is_not_zero():
    assert HarborTrialRecord.from_mapping(trial()).require_reward() == 0.0
    for verifier in (None, {}, {"rewards": {}}, {"rewards": {"reward": None}}):
        with pytest.raises(HarborResultError):
            HarborTrialRecord.from_mapping(trial(verifier_result=verifier)).require_reward()


@pytest.mark.parametrize("reward", [True, False, "1", float("nan"), float("inf"), 10**1000])
def test_non_numeric_or_non_finite_reward_refused(reward):
    with pytest.raises(HarborResultError):
        finite_number(reward, field="reward")


def test_staged_validation_leaves_exception_harvest_and_scorer_precedence_to_consumer():
    record = HarborTrialRecord.from_mapping(trial(exception_info="malformed", agent_result=None))
    with pytest.raises(HarborResultError, match="exception_info"):
        record.exception_info()
    with pytest.raises(HarborResultError, match="agent_result"):
        record.agent_result()
    interrupted = HarborTrialRecord.from_mapping(
        trial(exception_info={"exception_type": "AgentTimeoutError"})
    )
    assert interrupted.exception_info() == {"exception_type": "AgentTimeoutError"}
    assert interrupted.require_reward() == 0.0


def test_result_file_read_is_bounded_and_rejects_non_object(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    monkeypatch.setattr("synth_containers.harbor_results.MAX_RESULT_BYTES", 32)
    path.write_bytes(b" " * 33)
    with pytest.raises(HarborResultError, match="exceeds 32 bytes"):
        read_result_object(path, label="trial")
    path.write_text("[]")
    with pytest.raises(HarborResultError, match="JSON object"):
        read_result_object(path, label="trial")
    path.write_text(json.dumps({"reward": 0}))
    assert read_result_object(path, label="trial") == {"reward": 0}
