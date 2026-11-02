"""Current-Harbor (0.22.0) evidence qualification against producer-derived golden samples.

Goldens live in ``tests/fixtures/harbor_current``:

- ``observed/`` — the Resin engine job of real run 20261101T160739Z-632079bb,
  byte-copied with only the producer home prefix sanitized;
- ``fixture_derived/`` — the deterministic failed trial produced by the
  resinbench ``tools/harbor_fixture.py`` generator (error path; not observed).

``SOURCE.json`` records the original and stored sha256 of every file.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from synth_containers.tracing.adapters.harbor_evidence import (
    qualify_harbor_job,
    qualify_harbor_trial,
    requalify_from_raw,
)
from synth_containers.tracing.adapters.harbor_formats import (
    HarborFormatRefusal,
    HarborRefusalCode,
    mapping_declaration,
)

GOLDEN = Path(__file__).parent / "fixtures" / "harbor_current"
OBSERVED_JOB = GOLDEN / "observed" / "jobs" / "resin"
OBSERVED_TRIAL = "codex-mq-queue-views__Dqb272n"
FAILED_TRIAL = GOLDEN / "fixture_derived" / "jobs" / "vllm" / "trace-arrival-replay__uLt8vDJ"


def _source_manifest() -> dict[str, Any]:
    return json.loads((GOLDEN / "SOURCE.json").read_text(encoding="utf-8"))


def test_golden_bytes_match_recorded_source_hashes() -> None:
    manifest = _source_manifest()
    assert manifest["producer_observed"]["harbor_version_advertised"] == "0.22.0"
    for entry in manifest["files"]:
        content = (GOLDEN / entry["path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == entry["stored_sha256"], entry["path"]
        if not entry["sanitized"]:
            assert entry["stored_sha256"] == entry["source_sha256"]


def test_observed_job_qualifies_with_pinned_versions_and_raw_digests() -> None:
    evidence = qualify_harbor_job(OBSERVED_JOB)
    record = evidence.normalized
    assert record["versions"] == {"harbor": "0.22.0", "job_lock_schema": 3}
    assert record["job"]["timing"]["timezone"] == "unknown"
    assert record["job"]["retry"]["max_retries"] == 0
    assert record["job"]["retry"]["n_retries"] == 0
    stored = {
        entry["path"].split("jobs/resin/", 1)[1]: "sha256:" + entry["stored_sha256"]
        for entry in _source_manifest()["files"]
        if entry["origin"] == "producer_observed"
    }
    assert {ref["relpath"]: ref["digest"] for ref in record["raw_sources"]} == stored
    assert len(record["trials"]) == 2
    trial = next(item for item in record["trials"] if item["trial_dir"] == OBSERVED_TRIAL)
    assert trial["versions"]["trial_lock_schema"] == 2
    assert trial["agent"]["name"] == "resinbench-shell-agent"
    assert trial["attempt"]["status"] == "completed"
    assert trial["attempt"]["termination"] == "final_answer_tool"
    assert trial["attempt"]["attempt_number"] is None
    assert [step["index"] for step in trial["steps"]] == [0, 1, 2]
    assert [tool["name"] for step in trial["steps"] for tool in step["tools"]] == [
        "shell",
        "shell",
        "final_answer",
    ]
    assert trial["steps"][2]["tools"][0]["return_code_recorded"] is False
    assert all(step["timestamp"] is None for step in trial["steps"])
    assert trial["usage"]["input_tokens"] == sum(
        step["usage"]["input_tokens"] for step in trial["steps"]
    )
    assert trial["reward"]["values"] == {"reward": 0.0}
    assert trial["reward"]["reward_file"] == "0"
    assert trial["timing"]["agent_execution"]["duration_s"] == "4.991223"
    assert trial["timing"]["monotonic"] is None
    media = {item["relpath"]: item for item in trial["media"]}
    assert media["artifacts/app/queue.py"]["digest"].startswith("sha256:")
    assert media["artifacts/logs/artifacts"]["retained"] is False
    assert trial["producer_losses"]


def test_coverage_declares_unknown_and_raw_only_fields() -> None:
    coverage = qualify_harbor_job(OBSERVED_JOB).coverage_dict()
    result = coverage[f"{OBSERVED_TRIAL}/result.json"]
    assert "agent_result.cost_usd" in result["unknown"]
    assert "agent_result.n_cache_tokens" in result["unknown"]
    assert "exception_info" in result["absent"]
    assert "trial_uri" in result["raw_only"]
    assert "task_checksum" in result["mapped"]
    agent = coverage[f"{OBSERVED_TRIAL}/agent/resinbench-agent.json"]
    assert "steps[].usage.completion_tokens_details.reasoning_tokens" in agent["unknown"]
    assert {"lock.json", "result.json"} <= set(coverage)


def test_job_round_trips_from_preserved_raw_bytes(tmp_path: Path) -> None:
    evidence = qualify_harbor_job(OBSERVED_JOB)
    again = requalify_from_raw(evidence)
    assert again.normalized_digest == evidence.normalized_digest
    assert again.raw_digest == evidence.raw_digest
    restored = evidence.write_raw(tmp_path / "restored")
    for item in evidence.raw:
        assert (restored / item.relpath).read_bytes() == (OBSERVED_JOB / item.relpath).read_bytes()
    assert qualify_harbor_job(OBSERVED_JOB).normalized_digest == evidence.normalized_digest


def test_fixture_derived_failed_trial_maps_error_and_absences() -> None:
    evidence = qualify_harbor_trial(FAILED_TRIAL)
    record = evidence.normalized
    assert record["attempt"]["status"] == "errored"
    assert record["error"]["exception_type"] == "AgentTimeoutError"
    assert record["error"]["occurred_at"] is None
    assert record["steps"] == [] and record["step_source"] == "absent"
    assert record["timing"]["verifier"] is None
    absent = evidence.coverage_dict()["result.json"]["absent"]
    assert {"agent_result", "verifier", "verifier_result"} <= set(absent)
    assert requalify_from_raw(evidence).normalized_digest == evidence.normalized_digest


def test_mapping_declaration_pins_versions_and_lists_losses() -> None:
    declaration = mapping_declaration()
    assert declaration["supported"]["harbor_versions"] == ["0.22.0"]
    assert declaration["supported"]["trial_lock_schema_versions"] == [2]
    assert "ATIF-v1.7" in declaration["supported"]["atif_versions"]
    assert "trial/result.json" in declaration["files"]
    assert declaration["producer_losses"] and declaration["raw_only_files"]


# ------------------------------------------------------------------ refusals


def _job_copy(tmp_path: Path) -> Path:
    target = tmp_path / "job"
    shutil.copytree(OBSERVED_JOB, target)
    return target


def _edit_json(path: Path, edit: Callable[[Any], None]) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    edit(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def _drop_harbor_version(job: Path) -> None:
    _edit_json(job / "lock.json", lambda value: value["harbor"].pop("version"))


def _future_harbor(job: Path) -> None:
    _edit_json(job / "lock.json", lambda value: value["harbor"].update(version="0.23.0"))


def _drop_trial_lock_version(job: Path) -> None:
    _edit_json(job / OBSERVED_TRIAL / "lock.json", lambda value: value.pop("schema_version"))


def _future_trial_lock(job: Path) -> None:
    _edit_json(job / OBSERVED_TRIAL / "lock.json", lambda value: value.update(schema_version=3))


def _undeclared_result_field(job: Path) -> None:
    _edit_json(job / OBSERVED_TRIAL / "result.json", lambda value: value.update(new_field=1))


def _truncate_result(job: Path) -> None:
    path = job / OBSERVED_TRIAL / "result.json"
    path.write_bytes(path.read_bytes()[:40])


def _unpinned_agent(job: Path) -> None:
    _edit_json(
        job / OBSERVED_TRIAL / "result.json",
        lambda value: value["agent_info"].update(version="0.2.0"),
    )


def _reward_mismatch(job: Path) -> None:
    (job / OBSERVED_TRIAL / "verifier" / "reward.txt").write_text("1", encoding="utf-8")


def _agent_file_drift(job: Path) -> None:
    _edit_json(
        job / OBSERVED_TRIAL / "agent" / "resinbench-agent.json",
        lambda value: value.update(termination="max_turns"),
    )


def _naive_trial_clock(job: Path) -> None:
    _edit_json(
        job / OBSERVED_TRIAL / "result.json",
        lambda value: value.update(started_at="2026-11-01T16:09:16.330879"),
    )


def _write_atif(payload: dict[str, Any]) -> Callable[[Path], None]:
    def apply(job: Path) -> None:
        (job / OBSERVED_TRIAL / "agent" / "trajectory.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    return apply


_ATIF_STEP = {"step_id": 1, "source": "user", "message": "fix the queue"}
_ATIF_AGENT = {"name": "terminus", "version": "1.0"}

REFUSAL_CASES = [
    (
        "job-lock-harbor-version-missing",
        _drop_harbor_version,
        HarborRefusalCode.UNADVERTISED_VERSION,
    ),
    ("job-lock-harbor-version-future", _future_harbor, HarborRefusalCode.UNSUPPORTED_VERSION),
    (
        "trial-lock-version-missing",
        _drop_trial_lock_version,
        HarborRefusalCode.UNADVERTISED_VERSION,
    ),
    ("trial-lock-version-future", _future_trial_lock, HarborRefusalCode.UNSUPPORTED_VERSION),
    ("trial-result-undeclared-field", _undeclared_result_field, HarborRefusalCode.UNDECLARED_FIELD),
    ("trial-result-truncated", _truncate_result, HarborRefusalCode.MALFORMED_SOURCE),
    ("agent-metadata-unpinned", _unpinned_agent, HarborRefusalCode.UNSUPPORTED_VERSION),
    ("reward-file-mismatch", _reward_mismatch, HarborRefusalCode.INCONSISTENT_SOURCE),
    ("agent-file-drift", _agent_file_drift, HarborRefusalCode.INCONSISTENT_SOURCE),
    ("trial-clock-without-offset", _naive_trial_clock, HarborRefusalCode.MALFORMED_SOURCE),
    (
        "atif-version-unadvertised",
        _write_atif({"agent": _ATIF_AGENT, "steps": [_ATIF_STEP]}),
        HarborRefusalCode.UNADVERTISED_VERSION,
    ),
    (
        "atif-version-bare-number",
        _write_atif({"schema_version": "1.7", "agent": _ATIF_AGENT, "steps": [_ATIF_STEP]}),
        HarborRefusalCode.UNSUPPORTED_VERSION,
    ),
    (
        "atif-version-future",
        _write_atif({"schema_version": "ATIF-v2.0", "agent": _ATIF_AGENT, "steps": [_ATIF_STEP]}),
        HarborRefusalCode.UNSUPPORTED_VERSION,
    ),
    (
        "atif-steps-malformed",
        _write_atif(
            {
                "schema_version": "ATIF-v1.7",
                "agent": _ATIF_AGENT,
                "steps": [{**_ATIF_STEP, "step_id": 2}],
            }
        ),
        HarborRefusalCode.MALFORMED_ATIF,
    ),
]


@pytest.mark.parametrize(
    ("mutate", "code"),
    [pytest.param(mutate, code, id=case_id) for case_id, mutate, code in REFUSAL_CASES],
)
def test_malformed_or_unadvertised_sources_are_refused(
    tmp_path: Path, mutate: Callable[[Path], None], code: HarborRefusalCode
) -> None:
    job = _job_copy(tmp_path)
    mutate(job)
    with pytest.raises(HarborFormatRefusal) as refused:
        qualify_harbor_job(job)
    assert refused.value.code is code
    detail = refused.value.to_dict()
    assert detail["stage"] == "harbor_qualification" and detail["source"]


def test_advertised_atif_trajectory_is_inventoried(tmp_path: Path) -> None:
    job = _job_copy(tmp_path)
    _write_atif({"schema_version": "ATIF-v1.7", "agent": _ATIF_AGENT, "steps": [_ATIF_STEP]})(job)
    trial = next(
        item
        for item in qualify_harbor_job(job).normalized["trials"]
        if item["trial_dir"] == OBSERVED_TRIAL
    )
    assert trial["atif"]["source_format"] == "ATIF-v1.7"
    assert trial["atif"]["step_count"] == 1
    assert trial["atif"]["losses"]
