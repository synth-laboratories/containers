from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_containers.portable_candidates import AdmissionError, PortableCandidateRegistry

FIXTURES = Path(__file__).parent / "fixtures" / "portable-contracts-v1"


def documents() -> tuple[bytes, bytes, bytes]:
    return (
        (FIXTURES / "candidate.json").read_bytes(),
        (FIXTURES / "candidate-content.bin").read_bytes(),
        (FIXTURES / "task.json").read_bytes(),
    )


def test_register_then_run_binds_exact_candidate_task_and_refs() -> None:
    candidate, content, task = documents()
    registry = PortableCandidateRegistry()
    receipt = registry.register(
        candidate_bytes=candidate,
        candidate_content_bytes=content,
        task_bytes=task,
        recorded_at="2026-08-24T14:00:00Z",
    )
    admission = registry.admit_run(
        registration_id=receipt.registration_id,
        candidate_bytes=candidate,
        candidate_content_bytes=content,
        task_bytes=task,
        run_id="run-1",
        evaluator_id="nanohorizon-mac-v1",
        seed=91004,
        idempotency_key="run:1",
    )
    assert admission.candidate_digest == receipt.candidate_digest
    assert admission.task_contract_digest == receipt.task_contract_digest
    assert admission.execution_ref["uri"] == "run://run-1"
    assert admission.evaluation_ref["admission_digest"] == admission.digest
    assert admission.trajectory_ref["admission_digest"] == admission.digest
    assert admission.evaluation_ref["digest"] != admission.trajectory_ref["digest"]


def test_unregistered_run_is_rejected_before_execution() -> None:
    candidate, content, task = documents()
    with pytest.raises(AdmissionError, match="registration_not_found"):
        PortableCandidateRegistry().admit_run(
            registration_id="reg_missing",
            candidate_bytes=candidate,
            candidate_content_bytes=content,
            task_bytes=task,
            run_id="run-1",
            evaluator_id="nanohorizon-mac-v1",
            seed=91004,
            idempotency_key="run:1",
        )


@pytest.mark.parametrize("which", ["candidate", "task"])
def test_changed_bytes_are_rejected_even_when_json_is_semantically_equal(which: str) -> None:
    candidate, content, task = documents()
    registry = PortableCandidateRegistry()
    receipt = registry.register(
        candidate_bytes=candidate,
        candidate_content_bytes=content,
        task_bytes=task,
        recorded_at="2026-08-24T14:00:00Z",
    )
    if which == "candidate":
        candidate += b"\n"
    else:
        task += b"\n"
    with pytest.raises(AdmissionError, match=f"{which}_bytes_changed"):
        registry.admit_run(
            registration_id=receipt.registration_id,
            candidate_bytes=candidate,
            candidate_content_bytes=content,
            task_bytes=task,
            run_id="run-1",
            evaluator_id="nanohorizon-mac-v1",
            seed=91004,
            idempotency_key="run:1",
        )


def test_candidate_task_substitution_is_rejected_at_registration() -> None:
    candidate, content, task = documents()
    changed = json.loads(task)
    changed["revision"] = "different"
    # Even a stale offered digest fails closed before the candidate can join it.
    with pytest.raises(AdmissionError, match="digest_mismatch"):
        PortableCandidateRegistry().register(
            candidate_bytes=candidate,
            candidate_content_bytes=content,
            task_bytes=json.dumps(changed).encode(),
            recorded_at="2026-08-24T14:00:00Z",
        )


def test_candidate_content_is_checked_at_registration_and_run() -> None:
    candidate, content, task = documents()
    registry = PortableCandidateRegistry()
    with pytest.raises(AdmissionError, match="candidate_content_mismatch"):
        registry.register(
            candidate_bytes=candidate,
            candidate_content_bytes=b"substitution",
            task_bytes=task,
            recorded_at="2026-08-24T14:00:00Z",
        )
    receipt = registry.register(
        candidate_bytes=candidate,
        candidate_content_bytes=content,
        task_bytes=task,
        recorded_at="2026-08-24T14:00:00Z",
    )
    with pytest.raises(AdmissionError, match="candidate_content_changed"):
        registry.admit_run(
            registration_id=receipt.registration_id,
            candidate_bytes=candidate,
            candidate_content_bytes=b"substitution",
            task_bytes=task,
            run_id="run-1",
            evaluator_id="nanohorizon-mac-v1",
            seed=91004,
            idempotency_key="run:1",
        )


@pytest.mark.parametrize(
    ("evaluator", "seed", "error"),
    [
        ("wrong-evaluator", 91004, "evaluator_mismatch"),
        ("nanohorizon-mac-v1", 2, "seed_not_admitted"),
    ],
)
def test_evaluator_and_seed_authority_cannot_be_substituted(
    evaluator: str, seed: int, error: str
) -> None:
    candidate, content, task = documents()
    registry = PortableCandidateRegistry()
    receipt = registry.register(
        candidate_bytes=candidate,
        candidate_content_bytes=content,
        task_bytes=task,
        recorded_at="2026-08-24T14:00:00Z",
    )
    with pytest.raises(AdmissionError, match=error):
        registry.admit_run(
            registration_id=receipt.registration_id,
            candidate_bytes=candidate,
            candidate_content_bytes=content,
            task_bytes=task,
            run_id="run-1",
            evaluator_id=evaluator,
            seed=seed,
            idempotency_key="run:1",
        )


def test_run_idempotency_replays_same_receipt_and_rejects_conflict() -> None:
    candidate, content, task = documents()
    registry = PortableCandidateRegistry()
    receipt = registry.register(
        candidate_bytes=candidate,
        candidate_content_bytes=content,
        task_bytes=task,
        recorded_at="2026-08-24T14:00:00Z",
    )
    kwargs = dict(
        registration_id=receipt.registration_id,
        candidate_bytes=candidate,
        candidate_content_bytes=content,
        task_bytes=task,
        run_id="run-1",
        evaluator_id="nanohorizon-mac-v1",
        seed=91004,
        idempotency_key="run:1",
    )
    assert registry.admit_run(**kwargs) is registry.admit_run(**kwargs)
    kwargs["run_id"] = "run-2"
    with pytest.raises(AdmissionError, match="idempotency_conflict"):
        registry.admit_run(**kwargs)
