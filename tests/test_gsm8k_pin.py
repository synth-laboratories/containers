"""The GSM8K dataset pin, its profiles, and the per-trial parse mode.

What makes a GSM8K number reportable: the rows are ``openai/gsm8k`` at one
recorded revision whose split digests reproduce, the profile that supplied
them is declared in code (not read off an env var), the seed→row order is a
recorded permutation, and every trial says whether it was scored on a marked
answer (``exact``) or on the last-number fallback (``trailing_number``).
"""

from __future__ import annotations

import tempfile
import importlib.util
import json
import os
import re
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app
from synth_containers.platform import gsm8k_world as world
from synth_containers.platform.gsm8k_world import (
    HELDOUT_SPLIT,
    HF_CONFIG,
    HF_DATASET,
    HF_REVISION,
    PARSE_MODE_EXACT,
    PARSE_MODE_TRAILING,
    PARSE_MODE_UNPARSED,
    SHUFFLE_SEED,
    SPLIT_PINS,
    TRAIN_SPLIT,
    Gsm8kRow,
    SplitPin,
    clear_profile,
    dataset_manifest,
    declare_profile,
    load_row,
    parse_answer,
    rows_digest,
    shuffled_order,
    split_size,
    write_snapshot,
)
from synth_containers.training_rollout import (
    ROLLOUT_REQUEST_SCHEMA_VERSION,
    HostedSamplerClient,
    SamplerResult,
)

TELEMETRY = {"enabled": True, "transport": "sse", "retention": "run"}
TARGET = "gsm8k_solve"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--openai--gsm8k" / "snapshots"


@pytest.fixture(autouse=True)
def _no_declared_profile(monkeypatch):
    monkeypatch.delenv(world.LEGACY_SOURCE_ENV, raising=False)
    clear_profile()
    yield
    clear_profile()


def _tiny_rows(count: int = 5) -> tuple[Gsm8kRow, ...]:
    return tuple(
        Gsm8kRow(f"Question {index}: what is {index} + {index}?", f"{index} + {index} = {2 * index}\n#### {2 * index}")
        for index in range(count)
    )


def _pin_tiny(monkeypatch, split: str, rows: tuple[Gsm8kRow, ...]) -> SplitPin:
    count, digest = rows_digest(rows)
    pin = SplitPin(split, SPLIT_PINS[split].hf_split, count, digest)
    monkeypatch.setitem(world.SPLIT_PINS, split, pin)
    return pin


def _fake_datasets(monkeypatch, rows_by_hf_split: dict[str, tuple[Gsm8kRow, ...]], calls: list[dict]):
    module = types.ModuleType("datasets")

    def load_dataset(path, name=None, *, split=None, revision=None, **kwargs):
        calls.append({"path": path, "name": name, "split": split, "revision": revision, **kwargs})
        return [{"question": row.question, "answer": row.answer_text} for row in rows_by_hf_split[split]]

    module.load_dataset = load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)


# --- the pin is in code ------------------------------------------------------


def test_the_pin_is_a_full_commit_and_two_digests() -> None:
    assert HF_DATASET == "openai/gsm8k"
    assert HF_CONFIG == "main"
    assert re.fullmatch(r"[0-9a-f]{40}", HF_REVISION)
    assert set(SPLIT_PINS) == {TRAIN_SPLIT, HELDOUT_SPLIT}
    assert SPLIT_PINS[TRAIN_SPLIT].hf_split == "train"
    assert SPLIT_PINS[HELDOUT_SPLIT].hf_split == "test"
    for pin in SPLIT_PINS.values():
        assert pin.rows > 1000
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", pin.digest)
    assert isinstance(SHUFFLE_SEED, int)


def test_the_pin_never_comes_from_the_environment(monkeypatch) -> None:
    # The legacy switch can choose hf over the fixture; it cannot touch the pin.
    monkeypatch.setenv(world.LEGACY_SOURCE_ENV, "hf")
    manifest = dataset_manifest()
    assert manifest["revision"] == HF_REVISION
    assert manifest["profile"] == "hf"
    assert manifest["profile_source"] == "env"
    assert manifest["splits"][HELDOUT_SPLIT]["digest"] == SPLIT_PINS[HELDOUT_SPLIT].digest
    assert not any(name.startswith("SYNTH_GSM8K") and "REVISION" in name for name in os.environ)


def test_metadata_exposes_the_manifest_and_hashes_it_into_the_capability_digest(tmp_path) -> None:
    client = TestClient(create_compat_app(TARGET, storage_root=tmp_path / "p0"))
    meta = client.get("/metadata").json()
    dataset = meta["dataset"]
    assert dataset["schema_version"] == world.MANIFEST_SCHEMA
    assert dataset["dataset"] == "openai/gsm8k"
    assert dataset["revision"] == HF_REVISION
    assert dataset["splits"][TRAIN_SPLIT] == SPLIT_PINS[TRAIN_SPLIT].to_json()
    assert dataset["splits"][HELDOUT_SPLIT] == SPLIT_PINS[HELDOUT_SPLIT].to_json()
    assert dataset["profile"] == "fixture"
    assert dataset["pinned"] is False
    assert dataset["shuffle_seed"] is None
    assert dataset["parse_modes"] == [PARSE_MODE_EXACT, PARSE_MODE_TRAILING, PARSE_MODE_UNPARSED]
    assert dataset["fixture"][TRAIN_SPLIT]["rows"] == 16
    capabilities = client.get("/training/capabilities").json()
    platform = client.app.state.platform if hasattr(client.app.state, "platform") else None
    del platform
    # The digest binds the dataset: the same target with another manifest is
    # another container.
    assert capabilities["container_digest"] == meta["capabilities_digest"]
    assert "dataset" in json.dumps(meta)


# --- profiles are declared, not inferred -------------------------------------


def test_declared_profile_wins_over_the_legacy_env_var(monkeypatch) -> None:
    monkeypatch.setenv(world.LEGACY_SOURCE_ENV, "hf")
    profile = declare_profile("fixture")
    assert profile.source == "declared"
    assert dataset_manifest()["profile"] == "fixture"
    assert dataset_manifest()["profile_source"] == "declared"
    assert split_size(TRAIN_SPLIT) == 16  # fixture rows, not a download
    clear_profile()
    assert world.active_profile().name == "hf"
    assert world.active_profile().source == "env"


def test_unknown_profiles_and_half_declared_snapshots_are_refused(tmp_path) -> None:
    with pytest.raises(ValueError):
        declare_profile("parquet")
    with pytest.raises(ValueError):
        declare_profile("snapshot")
    with pytest.raises(ValueError):
        declare_profile("hf", snapshot_dir=tmp_path)
    with pytest.raises(RuntimeError, match="gsm8k_snapshot_invalid"):
        declare_profile("snapshot", snapshot_dir=tmp_path)  # no jsonl files


def test_hf_profile_loads_the_pinned_revision_and_orders_by_the_recorded_seed(monkeypatch) -> None:
    train = _tiny_rows(7)
    heldout = _tiny_rows(5)
    _pin_tiny(monkeypatch, TRAIN_SPLIT, train)
    _pin_tiny(monkeypatch, HELDOUT_SPLIT, heldout)
    calls: list[dict] = []
    _fake_datasets(monkeypatch, {"train": train, "test": heldout}, calls)
    declare_profile("hf")

    assert split_size(HELDOUT_SPLIT) == 5
    assert calls[0] == {"path": HF_DATASET, "name": HF_CONFIG, "split": "test", "revision": HF_REVISION}
    order = shuffled_order(5)
    assert order != tuple(range(5))
    for seed, index in enumerate(order):
        assert load_row(HELDOUT_SPLIT, seed) == heldout[index]
    assert load_row(HELDOUT_SPLIT, 5) is None
    manifest = dataset_manifest()
    assert manifest["pinned"] is True
    assert manifest["shuffle_seed"] == SHUFFLE_SEED


def test_hf_rows_that_do_not_reproduce_the_digest_are_refused(monkeypatch) -> None:
    rows = _tiny_rows(5)
    _pin_tiny(monkeypatch, HELDOUT_SPLIT, rows)
    tampered = rows[:-1] + (Gsm8kRow(rows[-1].question, "#### 999"),)
    _fake_datasets(monkeypatch, {"test": tampered, "train": rows}, [])
    declare_profile("hf")
    with pytest.raises(RuntimeError, match="gsm8k_split_digest_mismatch:heldout"):
        split_size(HELDOUT_SPLIT)
    # A short split is refused just the same: the pin is rows *and* digest.
    _fake_datasets(monkeypatch, {"test": rows[:-1], "train": rows}, [])
    clear_profile()
    declare_profile("hf")
    with pytest.raises(RuntimeError, match="gsm8k_split_digest_mismatch"):
        split_size(HELDOUT_SPLIT)


def test_snapshot_profile_round_trips_and_refuses_tampering(monkeypatch, tmp_path) -> None:
    train = _tiny_rows(6)
    heldout = _tiny_rows(4)
    _pin_tiny(monkeypatch, TRAIN_SPLIT, train)
    _pin_tiny(monkeypatch, HELDOUT_SPLIT, heldout)
    snapshot = write_snapshot(tmp_path / "snap", {TRAIN_SPLIT: train, HELDOUT_SPLIT: heldout})
    manifest = json.loads((snapshot / "manifest.json").read_text())
    assert manifest["revision"] == HF_REVISION
    assert manifest["splits"][HELDOUT_SPLIT]["digest"] == world.SPLIT_PINS[HELDOUT_SPLIT].digest
    assert (snapshot / "train.jsonl").is_file() and (snapshot / "test.jsonl").is_file()

    declare_profile("snapshot", snapshot_dir=snapshot)
    assert split_size(TRAIN_SPLIT) == 6
    assert load_row(HELDOUT_SPLIT, 0) == heldout[shuffled_order(4)[0]]
    assert dataset_manifest()["snapshot_dir"] == str(snapshot)
    assert dataset_manifest()["pinned"] is True

    lines = (snapshot / "test.jsonl").read_text().splitlines()
    lines[0] = lines[0].replace("#### 0", "#### 1")
    (snapshot / "test.jsonl").write_text("\n".join(lines) + "\n")
    clear_profile()
    declare_profile("snapshot", snapshot_dir=snapshot)
    with pytest.raises(RuntimeError, match="gsm8k_split_digest_mismatch"):
        split_size(HELDOUT_SPLIT)


def test_write_snapshot_refuses_rows_that_are_not_the_pin(monkeypatch, tmp_path) -> None:
    _pin_tiny(monkeypatch, HELDOUT_SPLIT, _tiny_rows(4))
    with pytest.raises(RuntimeError, match="gsm8k_split_digest_mismatch"):
        write_snapshot(tmp_path / "bad", {HELDOUT_SPLIT: _tiny_rows(3)})
    assert not (tmp_path / "bad").exists()


@pytest.mark.skipif(
    not (HF_CACHE / HF_REVISION).is_dir() or importlib.util.find_spec("datasets") is None,
    reason="pinned openai/gsm8k revision not in the local HF cache (no download in tests)",
)
def test_the_real_pinned_revision_reproduces_both_digests_offline(monkeypatch) -> None:
    """Runs wherever the pinned snapshot is already cached; never downloads."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    declare_profile("hf")
    # split_size verifies the digest on the way in; a mismatch raises here.
    assert split_size(TRAIN_SPLIT) == SPLIT_PINS[TRAIN_SPLIT].rows == 7473
    assert split_size(HELDOUT_SPLIT) == SPLIT_PINS[HELDOUT_SPLIT].rows == 1319
    first = load_row(HELDOUT_SPLIT, 0)
    assert first is not None and first.answer
    assert dataset_manifest()["pinned"] is True


# --- parse mode per trial ----------------------------------------------------


@pytest.mark.parametrize(
    ("completion", "mode", "compliant"),
    [
        ("Step one.\n#### 72", PARSE_MODE_EXACT, True),
        (r"Therefore \boxed{72}.", PARSE_MODE_EXACT, True),
        ("So she has 72 left.", PARSE_MODE_TRAILING, False),
        ("I would rather not say.", PARSE_MODE_UNPARSED, False),
        (None, PARSE_MODE_UNPARSED, False),
    ],
)
def test_parse_mode_separates_marked_answers_from_the_fallback(completion, mode, compliant) -> None:
    parsed = parse_answer(completion)
    assert parsed.parse_mode == mode
    assert parsed.format_compliant is compliant


def _client() -> TestClient:
    return TestClient(create_compat_app(TARGET, storage_root=tempfile.mkdtemp(prefix="test_gsm8k_pin-")))


def _run_forced(client: TestClient, rollout_id: str, completion: str) -> tuple[dict, dict]:
    config_id = f"cfg_{rollout_id}"
    assert client.post(
        "/policy-configs",
        json={"config_id": config_id, "harness": "solve", "config": {"forced_completion": completion}},
    ).status_code == 200
    assert client.post("/rollouts/prepare", json={"rollout_id": rollout_id, "telemetry": TELEMETRY}).status_code == 200
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": rollout_id,
            "telemetry": TELEMETRY,
            "slot": "stream",
            "world_ref": "world:gsm8k@train",
            "task_instance_id": "seed:0",
            "policy_ref": {"harness": "solve", "config": config_id},
        },
    )
    assert started.status_code == 200, started.text
    events = client.get(f"/rollouts/{rollout_id}/events", params={"after": 0}).json()["events"]
    action = next(row["payload"] for row in events if row["kind"] == "action")
    closed = next(row["payload"] for row in events if row["kind"] == "span.policy.closed")
    reward = client.post("/reward", json={"rollout_id": rollout_id, "mode": "terminal"}).json()
    return {**action, "reward": reward["reward"]}, closed


def test_every_trial_reports_its_parse_mode() -> None:
    client = _client()
    exact, exact_closed = _run_forced(client, "mode_exact", "48 / 2 = 24, 48 + 24 = 72\n#### 72")
    trailing, trailing_closed = _run_forced(client, "mode_trailing", "48 / 2 = 24, so altogether 72")
    prose, prose_closed = _run_forced(client, "mode_prose", "I would rather not say.")

    assert (exact["parse_mode"], exact["format_compliant"], exact["reward"]) == ("exact", True, 1.0)
    # Scored correct through the fallback: counted, and visibly *not* format compliance.
    assert (trailing["parse_mode"], trailing["format_compliant"], trailing["reward"]) == (
        "trailing_number",
        False,
        1.0,
    )
    assert trailing["parse_status"] == "parsed"
    assert (prose["parse_mode"], prose["format_compliant"], prose["reward"]) == ("unparsed", False, 0.0)
    for closed, mode in ((exact_closed, "exact"), (trailing_closed, "trailing_number"), (prose_closed, "unparsed")):
        assert closed["parse_mode"] == mode


# --- the training boundary samples GSM8K through the hosted sampler contract -


def test_training_rollout_samples_gsm8k_and_stamps_the_action(monkeypatch) -> None:
    seen: list[dict] = []

    def sample(_client: HostedSamplerClient, payload: dict, *, idempotency_key: str) -> SamplerResult:
        seen.append({"payload": payload, "key": idempotency_key})
        return SamplerResult(
            text="In May she sold 24 clips.\nAltogether 72.\n#### 72",
            prompt_token_ids=(10, 11, 12),
            token_ids=(1, 2, 3),
            log_probs=(-0.1, -0.2, -0.3),
            usage={"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
        )

    monkeypatch.setattr(HostedSamplerClient, "sample", sample)
    client = _client()
    request = {
        "schema_version": ROLLOUT_REQUEST_SCHEMA_VERSION,
        "job_id": "job-gsm8k",
        "attempt_id": "attempt-1",
        "rollout_id": "roll_gsm8k_training_1",
        "idempotency_key": "job-gsm8k:rollout-1",
        "policy_version": "snapshot:abc",
        "sampler": {"url": "https://sampler.example/v1/training/sample", "bearer_token": "job-token"},
        "task": {"world_ref": "world:gsm8k@train", "task_instance_id": "seed:0", "max_tokens": 9000, "temperature": 0.7},
    }
    response = client.post("/training/rollouts", json=request)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["policy_version"] == "snapshot:abc"
    assert body["reward"]["reward"] == 1.0
    action = body["actions"][0]
    assert action["token_ids"] == [1, 2, 3]
    assert action["log_probs"] == [-0.1, -0.2, -0.3]
    assert action["prompt_token_ids"] == [10, 11, 12]
    assert action["completion"].endswith("#### 72")

    payload = seen[0]["payload"]
    assert payload["policy_version"] == "snapshot:abc"
    assert payload["messages"][0]["content"] == world.SOLVE_SYSTEM
    assert "Natalia" in payload["messages"][1]["content"]
    assert payload["max_tokens"] == 4096  # the container's own ceiling, not Banking77's 32
    assert payload["temperature"] == 0.7
    assert body["usage"]["total_tokens"] == 6

    events = client.get("/rollouts/roll_gsm8k_training_1/events", params={"after": 0}).json()["events"]
    action_event = next(row["payload"] for row in events if row["kind"] == "action")
    assert action_event["parse_mode"] == "exact"
    assert action_event["training_action"]["policy_version"] == "snapshot:abc"
    assert "token_capture" not in {row["kind"] for row in events}  # no proxy record to join to


def test_training_rollout_requires_the_boundary_stamp(monkeypatch) -> None:
    """An `inference_target` without the boundary's stamp is not sampled."""
    from synth_containers.platform.runtimes import gsm8k as runtime

    assert runtime._training_sampler_target({"inference_target": {"provider_endpoint_id": "https://x/v1"}}) is None
    assert runtime._training_sampler_target(
        {"training_sampler_endpoint": True, "inference_target": {"provider_endpoint_id": "tinker://run"}}
    ) is None
    target = runtime._training_sampler_target(
        {"training_sampler_endpoint": True, "inference_target": {"provider_endpoint_id": "https://x/v1/sample"}}
    )
    assert target == {"provider_endpoint_id": "https://x/v1/sample"}
