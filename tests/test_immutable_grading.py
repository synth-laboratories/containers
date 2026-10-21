"""Frozen evidence, physical attempts, rubric revisions and publication retries."""

import json
import shutil

import pytest

import synth_containers.grading as grading
from synth_containers.grading_models import GradingPlanV1, GradingAuthorityV1
from synth_containers.tracing.adapters.native import import_native_to_bundle
from synth_containers.tracing.canonical import canonical_bytes
from synth_containers.tracing.models.standards import CriterionDefinitionV1, RubricDefinitionV2
from synth_containers.tracing.store.bundle import LocalTraceBundle, rebuild_catalog
from synth_containers.tracing.validation.rehydrate import trace_document_from_payload
from synth_containers.tracing.validation.validator import validate_evidence

D = "sha256:" + "a" * 64


def fixture(tmp_path, *, scorer="json-exact-v1", values=None, case_ids=("a", "b"), operation="grade-a"):
    source = tmp_path / "source.jsonl"
    source.write_text('{"type":"item.completed","item":{"id":"i","type":"agent_message","text":"frozen submission"}}\n')
    bundle = LocalTraceBundle(tmp_path / "bundle")
    imported = import_native_to_bundle(source, source_format="codex_jsonl", bundle=bundle)
    document = trace_document_from_payload(bundle.read_trace(imported["trace_digest"]))
    criteria = tuple(CriterionDefinitionV1(name, name, "retained-evidence rule", max_score=100 if scorer.startswith("craftax") else 1).sealed() for name in case_ids)
    rubric = RubricDefinitionV2("rubric", "Frozen fixture rubric", "fixture", criteria).sealed()
    values = values or {"cases": {"a": {"actual": 1, "expected": 1}, "b": {"actual": 0, "expected": 1}}}
    values = {**values, "rubric": json.loads(canonical_bytes(rubric))}
    inputs = []
    for name, value in values.items():
        body = canonical_bytes(value)
        digest = bundle.blobs.put(body)
        inputs.append(dict(name=name, digest=digest, size_bytes=len(body), media_type="application/json",
                           visibility="private", retention_root="bundle"))
    plan = GradingPlanV1(operation_id=operation, trace_id=document.trace_id, trace_digest=document.content_digest,
        task_digest=D, world_digest=D, dataset_digest=D, split_digest=D, scorer=scorer,
        scorer_program_digest=grading.grader_program_digest(scorer), dependencies_digest=grading.grader_dependencies_digest(),
        rubric_digest=rubric.content_digest, seed=0, judge_configuration_digest=D,
        execution_profile="trusted-offline-rubric-v1", permissions_digest=D, limits={}, case_ids=case_ids,
        inputs=tuple(inputs))
    bundle.write_manifest()
    return bundle, document, plan


def test_grades_append_without_changing_trace_and_both_versions_survive_cleanup(tmp_path):
    bundle, document, plan = fixture(tmp_path)
    before = canonical_bytes(document)
    root = tmp_path / "grades"
    a = grading.run_grading(plan, document, bundle.blobs, root, source_bundle=bundle)
    assert a.execution_status == "completed" and a.mean_score == .5 and a.scored_cases == 2
    assert grading.run_grading(plan, document, bundle.blobs, root) == a
    evidence = grading.read_grading_evidence(root / plan.operation_id, a)
    errors = [f for f in validate_evidence(document, evidence)[0] if str(f.severity) == "error"]
    assert errors == [], [item.to_dict() for item in errors]
    grading.publish_grading(bundle, root, a)
    grading.publish_grading(bundle, root, a)
    other = GradingPlanV1.model_validate({**plan.model_dump(mode="python"), "operation_id": "grade-b"})
    b = grading.run_grading(other, document, bundle.blobs, root, source_bundle=bundle)
    grading.publish_grading(bundle, root, b)
    assert a.result_id != b.result_id and a.evidence_digest != b.evidence_digest
    assert len(bundle.read_manifest()["evidence"]) == 2
    assert canonical_bytes(trace_document_from_payload(bundle.read_trace(document.content_digest))) == before
    selected = grading.aggregate_grades((a, b, a), GradingAuthorityV1(name="reviewed-a", result_ids=(a.result_id,)))
    assert selected["mean_score"] == .5 and selected["scored_cases"] == 2
    shutil.rmtree(bundle.root)
    (tmp_path / "source.jsonl").unlink()
    restored = grading.export_grading(root, b, tmp_path / "fresh")
    assert restored.verify_self_contained() == (True, ())
    assert restored.read_trace(document.content_digest) == json.loads(before)
    assert len(restored.read_manifest()["evidence"]) == 2
    assert rebuild_catalog(restored)["evidence_bundles"] == 2


@pytest.mark.parametrize("fault", ["missing", "corrupt", "wrong-program", "wrong-runtime", "private-denied"])
def test_preflight_refuses_unavailable_or_incompatible_evidence_without_compute(tmp_path, monkeypatch, fault):
    bundle, document, plan = fixture(tmp_path)
    monkeypatch.setattr(grading, "_worker", lambda *a: pytest.fail("preflight invoked compute"))
    if fault in {"missing", "corrupt", "private-denied"}:
        path = bundle.blobs.path_for(plan.inputs[0].digest)
        if fault == "missing":
            path.unlink()
        elif fault == "corrupt":
            path.chmod(0o600)
            path.write_bytes(b"wrong")
        else:
            original = bundle.blobs.get
            def denied(digest, **kwargs):
                if digest == plan.inputs[0].digest:
                    raise PermissionError("private")
                return original(digest, **kwargs)
            monkeypatch.setattr(bundle.blobs, "get", denied)
    else:
        field = "scorer_program_digest" if fault == "wrong-program" else "dependencies_digest"
        plan = GradingPlanV1.model_validate({**plan.model_dump(mode="python"), field: D})
    preflight = grading.preflight_grading(plan, document, bundle.blobs)
    assert not preflight.sufficient and preflight.compute_invoked is False
    with pytest.raises(ValueError):
        grading.run_grading(plan, document, bundle.blobs, tmp_path / "grades")


def test_lost_result_publication_reply_reuses_completed_worker_without_rerun(tmp_path, monkeypatch):
    bundle, document, plan = fixture(tmp_path)
    original = grading.durable_write
    def interrupted(path, body, **kwargs):
        if path.name == "evidence.json":
            raise OSError("crash before evidence publication")
        return original(path, body, **kwargs)
    root = tmp_path / "grades"
    with monkeypatch.context() as patch:
        patch.setattr(grading, "durable_write", interrupted)
        with pytest.raises(OSError):
            grading.run_grading(plan, document, bundle.blobs, root, source_bundle=bundle)
    assert len(list((root / plan.operation_id / "attempts").glob("*/worker-result.json"))) == 1
    shutil.rmtree(bundle.root)
    monkeypatch.setattr(grading, "_worker", lambda *a: pytest.fail("completed grader rerun"))
    # Inputs, trace and completed worker output are all retained in the operation.
    result = grading.run_grading(plan, None, bundle.blobs, root)
    assert result.mean_score == .5
    assert grading.run_grading(plan, None, bundle.blobs, root) == result


def test_unknown_attempt_refuses_implicit_retry_and_operation_plan_changes(tmp_path):
    bundle, document, plan = fixture(tmp_path)
    root = tmp_path / "grades"
    pending = root / plan.operation_id / "attempts/pending"
    pending.mkdir(parents=True)
    (pending / "started.json").write_text('{}')
    with pytest.raises(ValueError, match="outcome_unknown"):
        grading.run_grading(plan, document, bundle.blobs, root)
    result = grading.run_grading(plan, document, bundle.blobs, root, retry_interrupted=True)
    assert result.mean_score == .5
    changed = GradingPlanV1.model_validate({**plan.model_dump(mode="python"), "seed": 1})
    with pytest.raises(FileExistsError):
        grading.run_grading(changed, document, bundle.blobs, root)


def test_correction_reuses_existing_v5_supersession_and_alternative_run_coexists(tmp_path):
    bundle, document, plan = fixture(tmp_path)
    root = tmp_path / "grades"
    first = grading.run_grading(plan, document, bundle.blobs, root, source_bundle=bundle)
    prior = grading.read_grading_evidence(root / plan.operation_id, first)
    body = canonical_bytes(prior)
    digest = bundle.blobs.put(body)
    correction = GradingPlanV1.model_validate({**plan.model_dump(mode="python"), "operation_id": "correction",
        "supersedes_result_id": first.result_id, "inputs": [*plan.inputs, dict(name="prior_evidence", digest=digest,
            media_type="application/json", size_bytes=len(body), visibility="private", retention_root="bundle")]})
    second = grading.run_grading(correction, document, bundle.blobs, root)
    evidence = grading.read_grading_evidence(root / correction.operation_id, second)
    assert len(evidence.verifier_results) == 2
    assert evidence.verifier_results[0].verifier_result_id == first.result_id
    assert evidence.verifier_results[1].supersedes_id == first.result_id
    assert grading.compare_grades(plan, first, correction, second)["matching_scored_cases"] == ["a", "b"]


def test_worker_committed_before_controller_receipt_is_recovered_without_compute(tmp_path, monkeypatch):
    bundle, document, plan = fixture(tmp_path)
    original = grading.durable_write
    def lose_completion(path, body, **kwargs):
        if path.name == "completion.json":
            raise OSError("controller died after worker completion")
        return original(path, body, **kwargs)
    root = tmp_path / "grades"
    with monkeypatch.context() as patch:
        patch.setattr(grading, "durable_write", lose_completion)
        with pytest.raises(OSError):
            grading.run_grading(plan, document, bundle.blobs, root)
    monkeypatch.setattr(grading, "_worker", lambda *a: pytest.fail("worker computation repeated"))
    recovered = grading.run_grading(plan, None, bundle.blobs, root)
    assert recovered.mean_score == .5 and recovered.elapsed_ns is None
    assert recovered.cleanup == "worker-exit-not-observed"
    assert grading.run_grading(plan, None, bundle.blobs, root) == recovered


def test_craftax_metrics_refuse_uncaptured_state_instead_of_zero(tmp_path):
    bundle, document, plan = fixture(tmp_path, scorer="craftax-trajectory-metrics-v1",
        values={"trajectory": {"records": [{"step": 0, "action": "noop"}]}}, case_ids=("achievements",))
    preflight = grading.preflight_grading(plan, document, bundle.blobs)
    assert not preflight.sufficient and "craftax_required_state_not_captured" in preflight.blockers
    with pytest.raises(ValueError, match="state_not_captured"):
        grading.run_grading(plan, document, bundle.blobs, tmp_path / "grades")


def test_craftax_captured_metric_and_worker_failure_keep_missing_score(tmp_path):
    bundle, document, plan = fixture(tmp_path, scorer="craftax-trajectory-metrics-v1",
        values={"trajectory": {"records": [{"step": 0, "action": "noop", "state": {"achievements": ["wood"]}, "reward": 1}]}},
        case_ids=("achievements", "actions", "reward"))
    result = grading.run_grading(plan, document, bundle.blobs, tmp_path / "grades")
    assert result.mean_score == .01 and result.scored_cases == 3
    assert set(result.case_scores.values()) == {1}


def test_plan_result_models_never_accept_a_path_as_operation_identity():
    with pytest.raises(ValueError):
        GradingAuthorityV1(name="../outside", result_ids=())

@pytest.mark.parametrize(
    "required", ["unknown_required_fact", "full_state", "trace_bound_observation"]
)
def test_required_fact_cannot_be_silently_accepted_by_an_unqualified_profile(tmp_path, required):
    bundle, document, plan = fixture(tmp_path)
    requested = GradingPlanV1.model_validate(
        {**plan.model_dump(mode="python"), "required_facts": [required]}
    )
    preflight = grading.preflight_grading(requested, document, bundle.blobs)
    assert preflight.sufficient is False and preflight.compute_invoked is False


def test_achievement_list_is_not_full_simulator_state(tmp_path):
    bundle, document, plan = fixture(
        tmp_path,
        scorer="craftax-trajectory-metrics-v1",
        case_ids=("achievements",),
        values={
            "trajectory": {
                "records": [{"step": 0, "action": "noop", "state": {"achievements": []}}]
            }
        },
    )
    requested = GradingPlanV1.model_validate(
        {**plan.model_dump(mode="python"), "required_facts": ["full_state"]}
    )
    preflight = grading.preflight_grading(requested, document, bundle.blobs)
    assert "grading_full_state_capability_unqualified" in preflight.blockers

