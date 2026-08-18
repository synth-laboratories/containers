"""GSM8K one-turn world through the Containers HTTP surface.

The invariants under test are the ones that make a number honest: the reference
answer is env-private, an unparseable completion is not a wrong answer, a
missing signal stays ``null``, and the two splits share neither a question nor
a normalized answer.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app
from synth_containers.platform.gsm8k_world import (
    HELDOUT_INDICES,
    HELDOUT_SPLIT,
    TRAIN_INDICES,
    TRAIN_SPLIT,
    _FIXTURE_POOL,
    load_row,
    normalize_number,
    parse_answer,
    split_size,
)


TELEMETRY = {"enabled": True, "transport": "sse", "retention": "run"}
TARGET = "gsm8k_solve"


def _client() -> TestClient:
    return TestClient(create_compat_app(TARGET))


def _prepare_start(client: TestClient, *, rollout_id: str, body: dict) -> dict:
    prepared = client.post(
        "/rollouts/prepare",
        json={"rollout_id": rollout_id, "telemetry": TELEMETRY},
    )
    assert prepared.status_code == 200, prepared.text
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": rollout_id,
            "telemetry": TELEMETRY,
            "slot": "stream",
            **body,
        },
    )
    assert started.status_code == 200, started.text
    return started.json()


def _events(client: TestClient, rollout_id: str) -> list[dict]:
    return client.get(f"/rollouts/{rollout_id}/events", params={"after": 0}).json()["events"]


def _register(client: TestClient, config_id: str, config: dict) -> None:
    response = client.post(
        "/policy-configs",
        json={"config_id": config_id, "harness": "solve", "config": config},
    )
    assert response.status_code == 200, response.text


# --- world registration -----------------------------------------------------


def test_metadata_is_content_not_a_fold() -> None:
    meta = _client().get("/metadata").json()
    assert meta["runtime_family"] == "gsm8k"
    assert meta["target_id"] == TARGET
    assert meta["adapter_chain"] == []
    assert meta["world_ref"] == "world:gsm8k@heldout"
    assert meta["environment_ref"] == "env:gsm8k_dataset"
    assert meta["evaluation_plan_ref"] == "gsm8k_eval.v1"
    assert meta["policy_ref"]["harness"] == "solve"
    assert meta["live_frames"] == "unsupported"
    assert meta["reward_authority"] == "environment"
    seed = meta["input_schema"]["properties"]["seed"]
    assert seed["minimum"] == 0
    assert seed["maximum"] == split_size(HELDOUT_SPLIT) - 1


# --- requirement 1: the observation is the question only ---------------------


def test_public_observation_never_carries_the_answer() -> None:
    client = _client()
    for split in (TRAIN_SPLIT, HELDOUT_SPLIT):
        for seed in range(split_size(split)):
            rollout_id = f"gsm8k_private_{split}_{seed}"
            _prepare_start(
                client,
                rollout_id=rollout_id,
                body={
                    "world_ref": f"world:gsm8k@{split}",
                    "task_instance_id": f"seed:{seed}",
                    "policy_ref": {"harness": "solve", "config": "solve"},
                },
            )
            observation = next(
                row for row in _events(client, rollout_id) if row["kind"] == "observation"
            )
            payload = observation["payload"]
            row = load_row(split, seed)
            assert row is not None
            assert payload["question"] == row.question
            assert "answer" not in payload
            assert "answer_text" not in payload
            serialized = json.dumps(payload)
            # The system prompt names the `#### <answer>` format, so the bare
            # marker is expected; the gold *line* and the gold reasoning are not.
            assert f"#### {row.answer}" not in serialized
            assert row.answer_text not in serialized
            assert row.answer_text.splitlines()[0] not in serialized


def test_answer_value_is_absent_from_the_observation_for_a_row_that_would_show_it() -> None:
    # Seed 0's question mentions 48 and 2; its answer is 72. If the answer ever
    # leaked into the observation this assertion is the one that catches it.
    client = _client()
    _prepare_start(
        client,
        rollout_id="gsm8k_leak_probe",
        body={
            "world_ref": "world:gsm8k@train",
            "task_instance_id": "seed:0",
            "policy_ref": {"harness": "solve", "config": "solve"},
        },
    )
    observation = next(
        row for row in _events(client, "gsm8k_leak_probe") if row["kind"] == "observation"
    )
    row = load_row(TRAIN_SPLIT, 0)
    assert row is not None
    assert row.answer == "72"
    assert row.answer not in json.dumps(observation["payload"])


# --- requirement 2: the parser ----------------------------------------------


@pytest.mark.parametrize(
    ("completion", "expected", "source"),
    [
        (r"Therefore \boxed{42} is the answer.", "42", "boxed"),
        (r"work work \boxed{ -17 } more prose", "-17", "boxed"),
        (r"\boxed{1,024} widgets", "1024", "boxed"),
        (r"\boxed{3/4}", "0.75", "boxed"),
        ("Step one.\n#### 72", "72", "hash_marker"),
        ("#### 72\nThanks for reading!", "72", "hash_marker"),
        ("#### -5", "-5", "hash_marker"),
        ("#### +7", "7", "hash_marker"),
        ("#### 1,234,567", "1234567", "hash_marker"),
        ("#### 4.50", "4.5", "hash_marker"),
        ("#### $18", "18", "hash_marker"),
        ("#### 3/4", "0.75", "hash_marker"),
        ("#### 1/3", "1/3", "hash_marker"),
        ("So she has 6 left.", "6", "trailing_number"),
        ("12 + 24 = 36 points in total", "36", "trailing_number"),
        ("The rate is 0.2 per minute, so 10 dollars.", "10", "trailing_number"),
        # `\boxed{}` wins over a `####` that appears earlier, and both win over
        # the loose arithmetic a chain of thought leaves lying around.
        ("#### 11\nOn reflection: \\boxed{12}", "12", "boxed"),
        ("2 + 2 = 4\n#### 5", "5", "hash_marker"),
    ],
)
def test_parser_reads_the_documented_formats(completion: str, expected: str, source: str) -> None:
    parsed = parse_answer(completion)
    assert parsed.value == expected
    assert parsed.source == source
    assert parsed.parsed is True
    assert parsed.raw == completion


@pytest.mark.parametrize(
    "completion",
    ["", "I cannot answer that.", "#### ", r"\boxed{}", None, "no digits anywhere"],
)
def test_parse_failure_is_recorded_as_unparsed_not_as_a_number(completion: str | None) -> None:
    parsed = parse_answer(completion)
    assert parsed.value is None
    assert parsed.parsed is False
    assert parsed.source == "unparsed"
    assert parsed.raw == (completion or "")


def test_normalization_collapses_only_equal_values() -> None:
    assert normalize_number("1,000") == normalize_number("+1000") == normalize_number("1000.0")
    assert normalize_number("3/4") == normalize_number("0.75") == "0.75"
    assert normalize_number("-0") == "0"
    assert normalize_number("1/3") == "1/3"
    assert normalize_number("0.333") != normalize_number("1/3")
    assert normalize_number("abc") is None


# --- requirements 3 and 6: reward is exact match, missing stays null ---------


def test_dataset_gold_scores_exact_match() -> None:
    client = _client()
    started = _prepare_start(
        client,
        rollout_id="gsm8k_gold",
        body={
            "world_ref": "world:gsm8k@train",
            "task_instance_id": "seed:0",
            "policy_ref": {"harness": "dataset_gold", "config": "dataset_gold"},
        },
    )
    assert started["status"] == "completed"
    scored = client.post("/reward", json={"rollout_id": "gsm8k_gold", "mode": "terminal"}).json()
    assert scored["status"] == "scored"
    assert scored["reward"] == 1.0
    assert scored["node_results"][0]["authority"] == "environment"
    action = next(row for row in _events(client, "gsm8k_gold") if row["kind"] == "action")
    assert action["payload"]["answer"] == "72"
    assert action["payload"]["parse_status"] == "parsed"


def test_wrong_answer_and_unparseable_answer_both_score_zero() -> None:
    client = _client()
    _register(client, "forced_right", {"forced_completion": "reasoning...\n#### 72"})
    _register(client, "forced_wrong", {"forced_completion": "reasoning...\n#### 71"})
    _register(client, "forced_prose", {"forced_completion": "I would rather not say."})
    for rollout_id, config_id in (
        ("gsm8k_right", "forced_right"),
        ("gsm8k_wrong", "forced_wrong"),
        ("gsm8k_prose", "forced_prose"),
    ):
        _prepare_start(
            client,
            rollout_id=rollout_id,
            body={
                "world_ref": "world:gsm8k@train",
                "task_instance_id": "seed:0",
                "policy_ref": {"harness": "solve", "config": config_id},
            },
        )
    right = client.post("/reward", json={"rollout_id": "gsm8k_right", "mode": "terminal"}).json()
    wrong = client.post("/reward", json={"rollout_id": "gsm8k_wrong", "mode": "terminal"}).json()
    prose = client.post("/reward", json={"rollout_id": "gsm8k_prose", "mode": "terminal"}).json()
    assert right["reward"] == 1.0
    assert wrong["reward"] == 0.0
    assert wrong["status"] == "scored"
    # The one that matters: a completion the policy DID produce, that states no
    # number, is a failed attempt and scores 0.0. Scoring it None would let the
    # eval layer drop it from the denominator and inflate the model's accuracy.
    assert prose["reward"] == 0.0
    assert prose["status"] == "scored"

    action = next(row for row in _events(client, "gsm8k_prose") if row["kind"] == "action")
    assert action["payload"]["answer"] is None
    assert action["payload"]["parse_status"] == "unparsed"
    # The raw completion is kept next to the failed parse, not thrown away.
    assert action["payload"]["text"] == "I would rather not say."
    signal = next(row for row in _events(client, "gsm8k_prose") if row["kind"] == "reward_signal")
    assert signal["payload"]["value"] == 0.0


def test_a_failed_attempt_and_an_absent_signal_are_not_the_same_thing() -> None:
    """The distinction the eval layer depends on.

    `absent` is excluded from the denominator; `scored` is not. So "the policy
    produced text that stated no number" must be 0.0 (it tried and failed) while
    "the policy produced nothing at all" must be None (there is no attempt to
    score). Collapsing these in the null direction silently inflates accuracy by
    deleting a model's worst trials.
    """
    client = _client()
    _register(client, "forced_prose2", {"forced_completion": "hmm, unclear."})
    _prepare_start(
        client,
        rollout_id="gsm8k_attempted",
        body={
            "world_ref": "world:gsm8k@train",
            "task_instance_id": "seed:0",
            "policy_ref": {"harness": "solve", "config": "forced_prose2"},
        },
    )
    _prepare_start(
        client,
        rollout_id="gsm8k_no_attempt",
        body={
            "world_ref": "world:gsm8k@train",
            "task_instance_id": "seed:0",
            "policy_ref": {"harness": "solve", "config": "solve"},
        },
    )
    attempted = client.post(
        "/reward", json={"rollout_id": "gsm8k_attempted", "mode": "terminal"}
    ).json()
    no_attempt = client.post(
        "/reward", json={"rollout_id": "gsm8k_no_attempt", "mode": "terminal"}
    ).json()
    assert (attempted["reward"], attempted["status"]) == (0.0, "scored")
    assert (no_attempt["reward"], no_attempt["status"]) == (None, "absent")


def test_solve_without_a_sampler_is_absent_not_zero() -> None:
    client = _client()
    _prepare_start(
        client,
        rollout_id="gsm8k_absent",
        body={
            "world_ref": "world:gsm8k@heldout",
            "task_instance_id": "seed:1",
            "policy_ref": {"harness": "solve", "config": "solve"},
        },
    )
    scored = client.post("/reward", json={"rollout_id": "gsm8k_absent", "mode": "terminal"}).json()
    assert scored["status"] == "absent"
    assert scored["reward"] is None


def test_omit_reward_stays_null() -> None:
    client = _client()
    _prepare_start(
        client,
        rollout_id="gsm8k_omit",
        body={
            "world_ref": "world:gsm8k@train",
            "task_instance_id": "seed:2",
            "omit_reward": True,
            "policy_ref": {"harness": "dataset_gold", "config": "dataset_gold"},
        },
    )
    scored = client.post("/reward", json={"rollout_id": "gsm8k_omit", "mode": "terminal"}).json()
    assert scored["reward"] is None
    assert scored["status"] == "absent"


def test_unknown_seed_does_not_wrap() -> None:
    client = _client()
    overflow = split_size(HELDOUT_SPLIT) + 3
    started = _prepare_start(
        client,
        rollout_id="gsm8k_overflow",
        body={
            "world_ref": "world:gsm8k@heldout",
            "task_instance_id": f"seed:{overflow}",
            "policy_ref": {"harness": "dataset_gold", "config": "dataset_gold"},
        },
    )
    assert started["status"] == "failed"
    scored = client.post(
        "/reward", json={"rollout_id": "gsm8k_overflow", "mode": "terminal"}
    ).json()
    assert scored["reward"] is None
    assert scored["status"] == "absent"


# --- requirement 4: the split is deterministic and does not leak -------------


def test_split_indices_are_persisted_deterministic_and_a_partition() -> None:
    assert TRAIN_INDICES == tuple(range(0, 16))
    assert HELDOUT_INDICES == tuple(range(16, 32))
    assert set(TRAIN_INDICES).isdisjoint(HELDOUT_INDICES)
    assert sorted(TRAIN_INDICES + HELDOUT_INDICES) == list(range(len(_FIXTURE_POOL)))
    for position, index in enumerate(TRAIN_INDICES):
        assert load_row(TRAIN_SPLIT, position) is _FIXTURE_POOL[index]
    for position, index in enumerate(HELDOUT_INDICES):
        assert load_row(HELDOUT_SPLIT, position) is _FIXTURE_POOL[index]


def test_no_duplicate_questions_and_no_answer_leakage_across_splits() -> None:
    train = [load_row(TRAIN_SPLIT, seed) for seed in range(split_size(TRAIN_SPLIT))]
    heldout = [load_row(HELDOUT_SPLIT, seed) for seed in range(split_size(HELDOUT_SPLIT))]
    assert all(row is not None for row in train + heldout)

    train_questions = [row.question.strip() for row in train]
    heldout_questions = [row.question.strip() for row in heldout]
    assert len(set(train_questions)) == len(train_questions), "duplicate question inside train"
    assert len(set(heldout_questions)) == len(heldout_questions), "duplicate question inside heldout"
    shared_questions = set(train_questions) & set(heldout_questions)
    assert not shared_questions, f"question leaked across splits: {sorted(shared_questions)}"

    train_answers = [row.answer for row in train]
    heldout_answers = [row.answer for row in heldout]
    assert all(train_answers) and all(heldout_answers), "a fixture reference answer will not parse"
    assert len(set(train_answers)) == len(train_answers), "duplicate answer inside train"
    assert len(set(heldout_answers)) == len(heldout_answers), "duplicate answer inside heldout"
    shared_answers = set(train_answers) & set(heldout_answers)
    assert not shared_answers, f"normalized answer leaked across splits: {sorted(shared_answers)}"


def test_train_and_heldout_worlds_are_distinct_over_http() -> None:
    client = _client()
    for rollout_id, split in (("gsm8k_train", "train"), ("gsm8k_held", "heldout")):
        _prepare_start(
            client,
            rollout_id=rollout_id,
            body={
                "world_ref": f"world:gsm8k@{split}",
                "task_instance_id": "seed:0",
                "policy_ref": {"harness": "dataset_gold", "config": "dataset_gold"},
            },
        )
    train_question = next(
        row for row in _events(client, "gsm8k_train") if row["kind"] == "observation"
    )["payload"]["question"]
    heldout_question = next(
        row for row in _events(client, "gsm8k_held") if row["kind"] == "observation"
    )["payload"]["question"]
    assert train_question != heldout_question


# --- requirement 5: fixtures are the default source, real data is opt-in -----


def test_fixture_is_the_default_source_and_needs_no_download(monkeypatch) -> None:
    from synth_containers.platform import gsm8k_world

    monkeypatch.delenv("SYNTH_GSM8K_SOURCE", raising=False)
    assert gsm8k_world.source_name() == "fixture"
    assert split_size(TRAIN_SPLIT) == len(TRAIN_INDICES)
    assert split_size(HELDOUT_SPLIT) == len(HELDOUT_INDICES)

    def refuse(*_args, **_kwargs):  # pragma: no cover - guards a download
        raise AssertionError("the fixture path must never touch HuggingFace")

    monkeypatch.setattr(gsm8k_world, "_hf_rows", refuse)
    assert load_row(TRAIN_SPLIT, 0) is not None


def test_hf_source_is_opt_in_by_env_var(monkeypatch) -> None:
    from synth_containers.platform import gsm8k_world

    monkeypatch.setenv("SYNTH_GSM8K_SOURCE", "hf")
    assert gsm8k_world.source_name() == "hf"
    called: dict[str, str] = {}

    def fake_rows(split: str):
        called["split"] = split
        return ()

    monkeypatch.setattr(gsm8k_world, "_hf_rows", fake_rows)
    assert gsm8k_world.split_size(TRAIN_SPLIT) == 0
    assert called["split"] == TRAIN_SPLIT


# --- requirement 2 (task 2 tie-in): the local provider is admitted here too --


def test_local_mlx_provider_samples_over_loopback(monkeypatch) -> None:
    client = _client()
    captured: dict[str, object] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit: int) -> bytes:
            return json.dumps(
                {
                    "choices": [{"message": {"content": "reasoning\n#### 72"}}],
                    "usage": {"prompt_tokens": 40, "completion_tokens": 9, "total_tokens": 49},
                }
            ).encode()

    def fake_urlopen(request, *, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        del timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    _register(
        client,
        "local_mlx",
        {
            "provider": "synth_mlx_rl",
            "model": "Qwen/Qwen3.5-0.8B",
            "base_url": "http://127.0.0.1:8765/v1",
            "max_tokens": 256,
        },
    )
    started = _prepare_start(
        client,
        rollout_id="gsm8k_local_mlx",
        body={
            "world_ref": "world:gsm8k@train",
            "task_instance_id": "seed:0",
            "policy_ref": {"harness": "solve", "config": "local_mlx"},
        },
    )
    assert started["status"] == "completed"
    assert started["usage"]["total_tokens"] == 49
    assert captured["url"] == "http://127.0.0.1:8765/v1/chat/completions"
    # A loopback proxy issues no bearer of its own; none is invented.
    assert captured["authorization"] is None
    scored = client.post(
        "/reward", json={"rollout_id": "gsm8k_local_mlx", "mode": "terminal"}
    ).json()
    assert scored["reward"] == 1.0


def test_local_mlx_provider_refuses_a_public_origin_before_the_network(monkeypatch) -> None:
    client = _client()

    def unexpected(*_args, **_kwargs):  # pragma: no cover - the point is it is never reached
        raise AssertionError("network must not run")

    monkeypatch.setattr("urllib.request.urlopen", unexpected)
    _register(
        client,
        "local_mlx_public",
        {
            "provider": "synth_mlx_rl",
            "model": "Qwen/Qwen3.5-0.8B",
            "base_url": "http://exfil.example/v1",
        },
    )
    started = _prepare_start(
        client,
        rollout_id="gsm8k_local_refused",
        body={
            "world_ref": "world:gsm8k@train",
            "task_instance_id": "seed:0",
            "policy_ref": {"harness": "solve", "config": "local_mlx_public"},
        },
    )
    assert started["status"] == "failed"
    closed = next(
        row for row in _events(client, "gsm8k_local_refused") if row["kind"] == "span.policy.closed"
    )
    assert closed["payload"]["error_code"] == "synth_mlx_rl_endpoint_refused"
    scored = client.post(
        "/reward", json={"rollout_id": "gsm8k_local_refused", "mode": "terminal"}
    ).json()
    assert scored["reward"] is None


def test_the_proxy_request_id_is_sealed_into_the_trace(monkeypatch) -> None:
    """The join key between this container's reward and the proxy's token record.

    The proxy owns the authoritative token ids and rollout logprobs; the
    container owns the reward. If the container does not seal the proxy request
    id, nothing connects the two and an on-policy trace is unusable however
    complete each half is on its own.
    """
    client = _client()
    captured: dict[str, object] = {}

    class Response:
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self, *_):
            return json.dumps(
                {
                    "choices": [{"message": {"content": "the answer is #### 18"}}],
                    "usage": {"prompt_tokens": 40, "completion_tokens": 9, "total_tokens": 49},
                    "synth": {
                        "proxy_request_ids": ["prid_abc123"],
                        "policy_snapshot_id": "snap_v7",
                        "training_version": 7,
                        "tokenizer_digest": "tok_digest",
                        "template_digest": "tpl_digest",
                        "api_family": "chat_completions",
                    },
                }
            ).encode()

    def fake_urlopen(request, *, timeout):
        captured["url"] = request.full_url
        del timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    _register(
        client,
        "local_join",
        {
            "provider": "synth_mlx_rl",
            "model": "Qwen/Qwen3.5-0.8B",
            "base_url": "http://127.0.0.1:8765/v1",
            "max_tokens": 256,
        },
    )
    _prepare_start(
        client,
        rollout_id="gsm8k_join",
        body={
            "world_ref": "world:gsm8k@train",
            "task_instance_id": "seed:0",
            "policy_ref": {"harness": "solve", "config": "local_join"},
        },
    )
    capture = next(
        row for row in _events(client, "gsm8k_join") if row["kind"] == "token_capture"
    )
    payload = capture["payload"]
    assert payload["proxy_request_ids"] == ["prid_abc123"]
    assert payload["policy_snapshot_id"] == "snap_v7"
    assert payload["tokenizer_digest"] == "tok_digest"
    # A reference, never the tokens themselves: a container must not relay a
    # training record it does not own.
    assert "rollout_logprobs" not in payload and "completion_token_ids" not in payload
    assert payload["provenance"] == "observed_provider"


def test_no_join_key_is_emitted_when_there_is_no_proxy_record() -> None:
    """A gold action has no proxy call. Emitting an empty capture would be worse
    than none: it would claim a record exists that never did."""
    client = _client()
    _prepare_start(
        client,
        rollout_id="gsm8k_gold_nojoin",
        body={
            "world_ref": "world:gsm8k@train",
            "task_instance_id": "seed:0",
            "policy_ref": {"harness": "dataset_gold", "config": "dataset_gold"},
        },
    )
    kinds = [row["kind"] for row in _events(client, "gsm8k_gold_nojoin")]
    assert "token_capture" not in kinds
