"""Observed trajectory grades require complete, unambiguous sealed occurrences."""

from copy import deepcopy
from dataclasses import replace

import pytest

from synth_containers.grading_craftax import observed_craftax_trajectory
from synth_containers.tracing.models.events import EventV5
from synth_containers.tracing.models.document import TraceDocumentV5, TraceCaptureSummaryV5
from synth_containers.tracing.models.identity import TraceIdentityV5, TraceProvenanceV5
from synth_containers.tracing.models.completeness import TraceLifecycleV5, TraceCompletenessV5


def observed_document(tmp_path):
    document = TraceDocumentV5(
        trace_id="craftax-observed",
        trace_kind="evaluation_attempt",
        identity=TraceIdentityV5(run_id="run", task_id="craftax", benchmark="craftax", seed=0),
        lifecycle=TraceLifecycleV5(status="completed", started_at="2026-10-21T00:00:00Z"),
        capture=TraceCaptureSummaryV5("capture", "binding", "digest", "native", "none", "test"),
        provenance=TraceProvenanceV5(producer="test", producer_version="1"),
        completeness=TraceCompletenessV5(capture_status="complete", terminal_event_observed=True),
    )
    rows = [
        ("observation", {"step": 0, "readout": {"env_steps": 0, "achievements": []}}),
        ("action", {"step": 1, "action": "do"}),
        ("reward_signal", {"step": 1, "authority": "environment", "value": 0.8}),
        ("observation", {"step": 1, "readout": {"env_steps": 1, "achievements": ["wood"]}}),
        ("env.episode.closed", {"steps": 1, "status": "completed"}),
    ]
    events = tuple(
        EventV5(
            str(index), kind, "env", "session", "2026-10-21T00:00:00Z", payload=payload
        ).sealed()
        for index, (kind, payload) in enumerate(rows)
    )
    return replace(document, events=events).sealed()


def test_observed_trajectory_preserves_trace_lineage_and_copies_readouts(tmp_path):
    document = observed_document(tmp_path)
    result = observed_craftax_trajectory(document)
    assert result["trace_digest"] == document.content_digest
    assert result["state_semantics"] == "observed_readout"
    assert result["records"][0]["source_events"] == {
        "action": "1",
        "reward": "2",
        "observation": "3",
    }
    assert result["records"][0]["reward"] == 0.8
    result["records"][0]["state"]["achievements"].append("invented")
    assert document.events[3].payload["readout"]["achievements"] == ["wood"]


@pytest.mark.parametrize(
    "fault",
    [
        "unsealed-mutation",
        "missing-action",
        "missing-reward",
        "missing-observation",
        "duplicate-step",
        "duplicate-event-id",
        "foreign-step",
        "boolean-step",
        "wrong-terminal-count",
        "duplicate-terminal",
        "missing-terminal",
        "wrong-readout-step",
        "missing-achievements",
        "duplicate-achievements",
        "non-environment-reward",
        "boolean-reward",
        "nonfinite-reward",
        "empty-action",
    ],
)
def test_observed_trajectory_refuses_incomplete_or_ambiguous_capture(tmp_path, fault):
    document = observed_document(tmp_path)
    events = deepcopy(list(document.events))
    if fault == "unsealed-mutation":
        events[1].payload["action"] = "changed"
        with pytest.raises(ValueError, match="trace_identity_invalid"):
            observed_craftax_trajectory(replace(document, events=tuple(events)))
        return
    if fault.startswith("missing-") and fault != "missing-achievements":
        index = {
            "missing-action": 1,
            "missing-reward": 2,
            "missing-observation": 3,
            "missing-terminal": 4,
        }[fault]
        events.pop(index)
    elif fault == "duplicate-step":
        events.append(replace(events[1], event_id="duplicate"))
    elif fault == "duplicate-event-id":
        events[2] = replace(events[2], event_id=events[1].event_id)
    elif fault == "foreign-step":
        events[1].payload["step"] = 2
    elif fault == "boolean-step":
        events[1].payload["step"] = True
    elif fault == "wrong-terminal-count":
        events[4].payload["steps"] = 2
    elif fault == "duplicate-terminal":
        events.append(replace(events[4], event_id="duplicate"))
    elif fault == "wrong-readout-step":
        events[3].payload["readout"]["env_steps"] = 2
    elif fault == "missing-achievements":
        del events[3].payload["readout"]["achievements"]
    elif fault == "duplicate-achievements":
        events[3].payload["readout"]["achievements"] = ["wood", "wood"]
    elif fault == "non-environment-reward":
        events[2].payload["authority"] = "policy"
    elif fault == "boolean-reward":
        events[2].payload["value"] = True
    elif fault == "nonfinite-reward":
        # Nonfinite values cannot be sealed in canonical JSON either.
        events[2].payload["value"] = float("inf")
    elif fault == "empty-action":
        events[1].payload["action"] = ""
    with pytest.raises(ValueError):
        observed_craftax_trajectory(replace(document, events=tuple(events)).sealed())


def test_preflight_refuses_forged_observed_inputs_without_invoking_worker(tmp_path, monkeypatch):
    import json
    import synth_containers.grading as grading
    from synth_containers.grading_models import GradingPlanV1
    from synth_containers.tracing.canonical import canonical_bytes
    from synth_containers.tracing.store.bundle import LocalTraceBundle
    from synth_containers.tracing.models.standards import CriterionDefinitionV1, RubricDefinitionV2

    document = observed_document(tmp_path)
    bundle = LocalTraceBundle(tmp_path / "bundle")
    rubric = RubricDefinitionV2(
        "observed",
        "Observed actions",
        "test",
        (CriterionDefinitionV1("actions", "Actions", "Captured actions", max_score=10).sealed(),),
    ).sealed()
    trajectory = observed_craftax_trajectory(document)
    trajectory["records"][0]["action"] = "invented"
    inputs = []
    for name, value in {
        "rubric": json.loads(canonical_bytes(rubric)),
        "trajectory": trajectory,
    }.items():
        body = canonical_bytes(value)
        inputs.append(
            dict(
                name=name,
                digest=bundle.blobs.put(body),
                size_bytes=len(body),
                media_type="application/json",
                visibility="private",
                retention_root="bundle",
            )
        )
    digest = "sha256:" + "a" * 64
    plan = GradingPlanV1(
        operation_id="forged-observed",
        trace_id=document.trace_id,
        trace_digest=document.content_digest,
        task_digest=digest,
        world_digest=digest,
        dataset_digest=digest,
        split_digest=digest,
        scorer="craftax-trajectory-metrics-v1",
        scorer_program_digest=grading.grader_program_digest("craftax-trajectory-metrics-v1"),
        dependencies_digest=grading.grader_dependencies_digest(),
        rubric_digest=rubric.content_digest,
        seed=0,
        judge_configuration_digest=digest,
        permissions_digest=digest,
        execution_profile="trusted-offline-rubric-v1",
        limits={},
        case_ids=("actions",),
        inputs=tuple(inputs),
        required_facts=("trace_bound_observation",),
    )
    monkeypatch.setattr(grading, "_worker", lambda *args: pytest.fail("invoked compute"))
    result = grading.preflight_grading(plan, document, bundle.blobs)
    assert not result.sufficient and not result.compute_invoked
    assert "craftax_observed_trajectory_trace_mismatch" in result.blockers
    with pytest.raises(ValueError):
        grading.run_grading(plan, document, bundle.blobs, tmp_path / "grades")
