"""Harbor terminal logs must cross into Workshop as inspectable Trace V5 bundles."""

from __future__ import annotations

import base64

from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app
from synth_containers.tracing.inspection import inspect_trace_input
from synth_containers.tracing.projections.rollout_inspector import rollout_inspector_from_sealed
from synth_containers.tracing.store.bundle import LocalTraceBundle
from synth_containers.tracing.validation.rehydrate import trace_document_from_payload
from synth_containers.event_log import RolloutEventLog
from synth_containers.platform.targets import HARBOR_PUBLIC
from synth_containers.platform.trace_bundle import materialize_harbor_trace_bundle


BODY = {
    "rollout_id": "harbor_trace_bundle_1",
    "task_instance_id": "seed:17",
    "policy_ref": {"harness": "harbor_fused", "config": "luna_med"},
    "telemetry": {"enabled": True, "transport": "sse", "retention": "run"},
}

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _frame_bundle(tmp_path, *, repeated: bool = False, viewer: str | None = None,
                  producer_commit: str | None = None, container_image_digest: str | None = None):
    rollout_id = "frame_bundle_rollout"
    journal = tmp_path / "event_journals" / "frame.jsonl"
    log = RolloutEventLog(
        rollout_id=rollout_id,
        stream_id=f"stream:{rollout_id}",
        journal_path=journal,
    )
    for step in range(2 if repeated else 1):
        url = log.persist_frame(step, PNG_1X1, name=viewer)
        assert url
        log.append("frame", {"step": step, "format": "png", "url": url, **({"viewer": viewer} if viewer else {})})
    log.mark_closed()
    output = tmp_path / "portable-frame-trace.zip"
    materialize_harbor_trace_bundle(
        output_path=output,
        spec=HARBOR_PUBLIC,
        log=log,
        seal={"content_digest": "sha256:" + "a" * 64},
        pin={"task_instance_id": "seed:1"},
        status="completed",
        producer_commit=producer_commit,
        container_image_digest=container_image_digest,
    )
    return output


def test_native_frames_are_embedded_and_bound_to_their_trace_events(tmp_path) -> None:
    archive = _frame_bundle(tmp_path, repeated=True)
    extracted = LocalTraceBundle.extract_archive(archive, tmp_path / "portable")
    manifest = extracted.read_manifest()
    document = extracted.read_trace(manifest["traces"][0]["trace_digest"])
    assert len(document["artifacts"]) == 1  # identical pixels deduplicate physically
    artifact = document["artifacts"][0]
    assert artifact["media_type"] == "image/png"
    assert artifact["metadata"]["steps"] == [0, 1]
    assert extracted.blobs.get(artifact["digest"]) == PNG_1X1
    frames = [event for event in document["events"] if event["event_type"] == "frame"]
    assert len(frames) == 2  # repeated logical frames remain in the trajectory
    assert all(event["artifact_ids"] == [artifact["artifact_id"]] for event in frames)
    visual = rollout_inspector_from_sealed(trace_document_from_payload(document)).visual
    projected_frames = [item for item in visual.items if item.kind == "frame"]
    assert len(projected_frames) == 2
    assert projected_frames[0].detail["artifacts"][0]["digest"] == artifact["digest"]


def test_missing_declared_native_frame_fails_portable_promotion(tmp_path) -> None:
    rollout_id = "missing_frame_rollout"
    log = RolloutEventLog(
        rollout_id=rollout_id,
        stream_id=f"stream:{rollout_id}",
        journal_path=tmp_path / "event_journals" / "missing.jsonl",
    )
    log.append(
        "frame",
        {"step": 0, "format": "png", "url": f"/rollouts/{rollout_id}/frames/0.png"},
    )
    log.mark_closed()
    try:
        materialize_harbor_trace_bundle(
            output_path=tmp_path / "missing.zip",
            spec=HARBOR_PUBLIC,
            log=log,
            seal={"content_digest": "sha256:" + "b" * 64},
            pin={},
            status="completed",
        )
    except ValueError as error:
        assert "native_frame_artifact_missing" in str(error)
    else:  # pragma: no cover - the failure is the contract
        raise AssertionError("declared frame without bytes produced a portable bundle")


def test_harbor_terminal_run_serves_a_self_contained_inspectable_bundle(tmp_path) -> None:
    app = create_compat_app("harbor_public", storage_root=tmp_path)
    client = TestClient(app)
    prepared = client.post(
        "/rollouts/prepare",
        json={"rollout_id": BODY["rollout_id"], "telemetry": BODY["telemetry"]},
    )
    assert prepared.status_code == 200, prepared.text
    started = client.post("/rollouts", json=BODY)
    assert started.status_code == 200, started.text

    status = client.get(f"/rollouts/{BODY['rollout_id']}")
    assert status.status_code == 200, status.text
    reference = status.json()["trace"]
    assert reference["bundle_url"] == f"/rollouts/{BODY['rollout_id']}/trace/bundle"
    assert reference["bundle_trace_id"] == BODY["rollout_id"]
    assert reference["bundle_digest"].startswith("sha256:")

    served = client.get(reference["bundle_url"])
    assert served.status_code == 200, served.text
    assert served.headers["content-type"].startswith("application/zip")
    archive = tmp_path / "served-trace.zip"
    archive.write_bytes(served.content)

    inspection = inspect_trace_input(archive)
    assert inspection.input_kind == "bundle_archive"
    assert inspection.compatibility == "native"
    assert inspection.self_contained is True
    assert inspection.trusted is True
    assert inspection.validation.valid is True
    assert [trace.trace_id for trace in inspection.traces] == [BODY["rollout_id"]]
    assert inspection.traces[0].projectable is True

    extracted = LocalTraceBundle.extract_archive(archive, tmp_path / "extracted")
    self_contained, errors = extracted.verify_self_contained()
    assert self_contained, errors
    manifest = extracted.read_manifest()
    assert manifest["metadata"]["promotion_schema"] == "synth.containers.harbor-trace-promotion.v1"
    assert manifest["metadata"]["source_blob_digest"].startswith("sha256:")
    document = extracted.read_trace(manifest["traces"][0]["trace_digest"])
    kinds = {event["event_type"] for event in document["events"]}
    assert "application.event" not in kinds
    assert any(kind.startswith("trial.") for kind in kinds)
    assert all("source_event_type" in event["payload"] for event in document["events"])


def test_portable_trace_binds_explicit_runtime_identity_not_ambient_environment(
    tmp_path, monkeypatch
) -> None:
    image = "sha256:" + "b" * 64
    revision = "66b75560dc9ced6c72e94c7ead126a30d2474219"
    monkeypatch.setenv("SYNTH_CONTAINER_IMAGE_DIGEST", "sha256:" + "c" * 64)
    monkeypatch.setenv("SYNTH_CONTAINER_PRODUCER_SOURCE_REVISION", "d" * 40)
    archive = _frame_bundle(tmp_path, producer_commit=revision, container_image_digest=image)
    extracted = LocalTraceBundle.extract_archive(archive, tmp_path / "identity")
    manifest = extracted.read_manifest()
    document = extracted.read_trace(manifest["traces"][0]["trace_digest"])
    assert document["provenance"]["container_image_digest"] == image
    assert document["provenance"]["producer_commit"] == revision


def test_harbor_trace_bundle_is_retained_and_reannounced_after_restart(tmp_path) -> None:
    from dataclasses import replace
    spec = replace(HARBOR_PUBLIC, environment_version="sha256:" + "a" * 64)
    first = TestClient(create_compat_app(spec, storage_root=tmp_path))
    assert first.post("/rollouts", json=BODY).status_code == 200
    before = first.get(f"/rollouts/{BODY['rollout_id']}/trace/bundle")
    assert before.status_code == 200, before.text

    reopened = TestClient(create_compat_app(replace(spec, environment_version="sha256:" + "b" * 64), storage_root=tmp_path))
    status = reopened.get(f"/rollouts/{BODY['rollout_id']}").json()
    assert status["research_context"]["environmentVersion"] == "sha256:" + "a" * 64
    assert status["trace"]["bundle_url"] == f"/rollouts/{BODY['rollout_id']}/trace/bundle"
    after = reopened.get(status["trace"]["bundle_url"])
    assert after.status_code == 200, after.text
    assert after.content == before.content


def test_harbor_rebuilds_a_corrupt_derived_bundle_from_retained_raw_evidence(tmp_path) -> None:
    first = TestClient(create_compat_app("harbor_public", storage_root=tmp_path))
    assert first.post("/rollouts", json=BODY).status_code == 200
    original_seal = first.get(f"/rollouts/{BODY['rollout_id']}/trace").json()
    archive = next((tmp_path / "trace_bundles").glob("*.zip"))
    archive.chmod(0o644)
    archive.write_bytes(b"not a trace bundle")

    reopened = TestClient(create_compat_app("harbor_public", storage_root=tmp_path))
    rebuilt = reopened.get(f"/rollouts/{BODY['rollout_id']}/trace/bundle")
    assert rebuilt.status_code == 200, rebuilt.text
    restored = tmp_path / "restored.zip"
    restored.write_bytes(rebuilt.content)
    inspection = inspect_trace_input(restored)
    assert inspection.trusted is True
    assert inspection.validation.valid is True
    # Regeneration derives from the same retained compact seal and journal;
    # neither is replaced just because the portable derivative was damaged.
    assert reopened.get(f"/rollouts/{BODY['rollout_id']}/trace").json() == original_seal


def test_non_harbor_runtime_also_serves_its_terminal_journal_as_trace_v5(tmp_path) -> None:
    """External/OpenEnv-style targets must not lose the portable trace edge."""
    rollout_id = "openenv_trace_bundle_1"
    client = TestClient(create_compat_app("openenv_echo", storage_root=tmp_path))
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": rollout_id,
            "task_instance_id": "seed:23",
            "policy_ref": {"harness": "gym_loop", "config": "echo"},
            "telemetry": {"enabled": True, "transport": "poll", "retention": "run"},
        },
    )
    assert started.status_code == 200, started.text

    status = client.get(f"/rollouts/{rollout_id}").json()
    assert status["trace"]["bundle_url"] == f"/rollouts/{rollout_id}/trace/bundle"
    served = client.get(status["trace"]["bundle_url"])
    assert served.status_code == 200, served.text
    archive = tmp_path / "openenv-trace.zip"
    archive.write_bytes(served.content)
    inspection = inspect_trace_input(archive)
    assert inspection.trusted is True
    assert inspection.self_contained is True
    assert [trace.trace_id for trace in inspection.traces] == [rollout_id]


def test_named_agent_frames_survive_portable_bundle(tmp_path):
    archive = _frame_bundle(tmp_path, viewer="agent_1")
    inspection = inspect_trace_input(archive)
    assert inspection.trusted and inspection.self_contained
    extracted = LocalTraceBundle.extract_archive(archive, tmp_path / "agent-view")
    document = extracted.read_trace(inspection.traces[0].trace_digest)
    frame = next(e for e in document["events"] if e["event_type"] == "frame")
    assert frame["payload"]["viewer"] == "agent_1"
    assert frame["artifact_ids"]


def test_explicit_action_actors_are_distinct_but_viewers_are_not_actors():
    from synth_containers.platform.trace_bundle import _event_actor_key
    first = _event_actor_key({"kind":"action_applied","payload":{"agent_id":"hero_1"}})
    second = _event_actor_key({"kind":"action_applied","payload":{"agent_id":"hero_2"}})
    assert first != second
    assert _event_actor_key({"kind":"frame","payload":{"viewer":"hero_1"}}) == "environment"


def test_cancelled_rollout_is_interrupted_not_completed(tmp_path):
    log = RolloutEventLog('cancelled-rollout', 'stream:cancelled-rollout')
    log.append('status', {'status':'cancelled'})
    log.mark_closed()
    archive = tmp_path / 'cancelled.zip'
    materialize_harbor_trace_bundle(output_path=archive,spec=HARBOR_PUBLIC,log=log,seal={},pin={},status='cancelled')
    bundle = LocalTraceBundle.extract_archive(archive,tmp_path/'unpacked-cancelled')
    inspection=inspect_trace_input(archive)
    document=bundle.read_trace(inspection.traces[0].trace_digest)
    assert document['lifecycle']['status']=='interrupted'
    assert all(session['status']=='interrupted' for session in document['sessions'])


def test_declared_terminal_reward_has_stable_typed_semantics_and_missing_stays_missing(tmp_path):
    from dataclasses import replace
    from synth_containers.tracing.projections.inspector import load_bundle
    definition = {"primary_metric": "craftax.episode_return", "version": "craftax.env-signal-sum.v1", "units": "craftax_reward", "intent": "Sum of emitted environment signals", "source_kind": "deterministic_metric"}
    spec = replace(HARBOR_PUBLIC, task_family="craftax", task_id="craftax", reward_definition=definition)
    digests = []
    for i, terminal in enumerate([{"status":"scored","value":0.0},{"status":"scored","value":2.0},{"status":"absent","value":None}]):
        log=RolloutEventLog(rollout_id=f"typed-{i}",stream_id=f"stream:{i}")
        log.append("reward_signal",{"value":terminal["value"]});log.mark_closed()
        archive=tmp_path/f"typed-{i}.zip"
        result=materialize_harbor_trace_bundle(output_path=archive,spec=spec,log=log,seal={},pin={"task_instance_id":"seed:1","terminal_reward":terminal},status="completed")
        bundle=LocalTraceBundle.extract_archive(archive,tmp_path/f"bundle-{i}")
        inspected=load_bundle(bundle.root)[0]
        assert inspect_trace_input(archive).trusted
        assert result.bundle_digest==bundle.read_manifest()["content_digest"]
        if terminal["status"]=="absent":
            assert inspected.evidence is None or not inspected.evidence.reward_records
        else:
            reward=inspected.evidence.reward_records[0]
            assert reward.value==terminal["value"]
            assert str(reward.subject.kind)=="trace"
            assert inspected.evidence.reward_definitions[0].units=="craftax_reward"
            digests.append(reward.reward_digest)
    assert digests[0]==digests[1], "reward values and run IDs must not change the definition"
