"""Portable joint cuts, strict-position selection and explicit fork refusals."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

import pytest
from pydantic import ValidationError

from synth_containers.checkpoints import (
    ApplicationCheckpoint,
    CheckpointBundleStore,
    CheckpointCommitAnchorV1,
    CheckpointComponentV1,
    CheckpointCursorV1,
    CheckpointExecutionV1,
    CheckpointFailure,
    CheckpointLiabilityV1,
    CheckpointRestoreRequestV1,
    CheckpointRuntimeSupportV1,
    CheckpointSemantics,
    FileArtifactStore,
    ParticipantCut,
    ParticipantIdentity,
    PortableCheckpointManifestV1,
    SessionFence,
    check_checkpoint_restore,
    export_checkpoint_bundle,
    inspect_portable_checkpoint,
    load_checkpoint_bundle,
    select_strict_checkpoint,
)
from synth_containers.checkpoints.contracts import ParticipantSnapshot
from synth_containers.checkpoints.cli import main
from synth_containers.package_models import content_digest


def digest(value):
    return content_digest(value)


def fixture(root, *, sequence=5):
    root.mkdir(parents=True, exist_ok=True)
    store = FileArtifactStore(root / "source", namespace="private-checkpoint")
    operation, checkpoint = uuid4(), uuid4()
    fence = SessionFence(connector=uuid4(), session=uuid4(), source_generation=1,
                         generation=2, permit_barrier_ms=0)
    snapshots, components = [], []
    for name in ("environment", "policy_a", "policy_b"):
        coverage = ("state", "rng", "mailboxes") if name == "environment" else ("history", "rng", "pending_plan")
        identity = ParticipantIdentity(
            name=name, service_identity="scripted-" + name, origin_instance=uuid4(),
            semantics=CheckpointSemantics.APPLICATION_COLD_RESTORE,
        )
        snapshots.append(ParticipantSnapshot(
            cut=ParticipantCut(operation=operation, receipt=uuid4(), participant=identity,
                               pending_operations=0, unresolved_operations=0),
            artifact=store.put(json.dumps({"state": name, "position": sequence}).encode()),
        ))
        components.append(CheckpointComponentV1(participant=name, codec=digest(name), coverage=coverage))
    application = ApplicationCheckpoint(
        checkpoint=checkpoint, operation=operation, fence=fence,
        semantics=CheckpointSemantics.APPLICATION_COLD_RESTORE, participants=tuple(snapshots),
    )
    configuration = store.put(b'{"checkpoint_cadence":1,"step_cap":20,"deadline":"unchanged"}')
    policies = store.put(b'{"policy_a":{"model":"scripted","seed":10},"policy_b":{"seed":11}}')
    execution = CheckpointExecutionV1(
        source=digest("source"), image=digest("image"), dependencies=digest("dependencies"),
        task=digest("task"), world=digest("world"), scorer=digest("scorer"), protocol=digest("protocol"),
        configuration=configuration, policies=policies,
    )
    manifest = PortableCheckpointManifestV1(
        application=store.put(application.canonical_bytes()), checkpoint=checkpoint,
        run_id="scripted-run", lineage_id=uuid4(), sequence=sequence,
        chain_head=digest(["committed-event-prefix", sequence]), execution=execution,
        components=tuple(components),
        cursors=tuple(CheckpointCursorV1(name=name, position=sequence, chain_head=digest(["committed-event-prefix", sequence]) if name == "event" else digest([name, sequence]))
                      for name in ("action", "observation", "reward", "event")),
        liabilities=(
            CheckpointLiabilityV1(resource="calls", provider="scripted", unit="count", consumed="5", outstanding="0"),
            CheckpointLiabilityV1(resource="cost", provider="scripted", unit="usd", consumed="0.003125", outstanding=None),
            CheckpointLiabilityV1(resource="active_time", provider="runtime", unit="seconds", consumed="1.25", outstanding="0"),
        ),
        clock_policy="active_runtime", terminal=False,
        modes=("inspect", "replay", "strict_continue", "same_policy_fork", "edited_policy_fork"),
    )
    support = CheckpointRuntimeSupportV1(
        modes=manifest.modes, components=manifest.components,
        required_cursors=("action", "observation", "reward", "event"),
        trusted_decoders=tuple(row.codec for row in manifest.components), clock_policy="active_runtime",
    )
    return manifest, application, store, support


def request_for(manifest, **updates):
    value = dict(
        mode="strict_continue", source_run_id=manifest.run_id, source_lineage_id=manifest.lineage_id,
        target_run_id=manifest.run_id, target_lineage_id=manifest.lineage_id, execution=manifest.execution,
        commit=CheckpointCommitAnchorV1(manifest=manifest.digest(), sequence=manifest.sequence, chain_head=manifest.chain_head),
    )
    value.update(updates)
    return CheckpointRestoreRequestV1(**value)


def at_position(manifest, sequence, head):
    return manifest.model_copy(update={
        "sequence": sequence, "chain_head": head,
        "cursors": tuple(row.model_copy(update={"position": sequence, "chain_head": head}) if row.name == "event" else row for row in manifest.cursors),
    })


def test_export_is_fresh_portable_private_closure_and_does_not_decode(tmp_path):
    manifest, _, store, support = fixture(tmp_path)
    original = {path.name: path.read_bytes() for path in (tmp_path / "source/sha256").iterdir()}
    bundle = tmp_path / "export"
    inspection = export_checkpoint_bundle(manifest, store, bundle)
    assert inspection.verified_blobs == 6
    assert inspection.publication == "not_checked" and inspection.decoder_execution == "not_run"
    assert original == {path.name: path.read_bytes() for path in (tmp_path / "source/sha256").iterdir()}
    assert bundle.stat().st_mode & 0o077 == 0
    assert all(path.stat().st_mode & 0o077 == 0 for path in bundle.rglob("*") if path.is_file())
    assert not (bundle / ".manifest.pending").exists()
    shutil.rmtree(tmp_path / "source")
    moved = tmp_path / "fresh-process-input"
    bundle.rename(moved)
    assert load_checkpoint_bundle(moved) == inspection
    reader = CheckpointBundleStore(moved, namespace=manifest.application.namespace)
    report = check_checkpoint_restore(manifest, reader, request_for(manifest), support)
    assert report.declaration_supported
    assert report.retained_liabilities[1].outstanding is None
    assert report.retained_liabilities[1].consumed == "0.003125"
    assert report.admission == report.target_inputs == "not_checked"
    assert report.suffix_reconciliation == report.liability_reconciliation == "required"
    assert report.decoder_execution == "not_run"
    with pytest.raises(CheckpointFailure, match="destination_exists"):
        export_checkpoint_bundle(manifest, reader, moved)


@pytest.mark.parametrize("mutation,blocker", [
    ("terminal", "terminal_not_resumable"),
    ("unresolved", "unresolved_effects"),
    ("codec", "codec_incompatible"),
    ("trust", "decoder_untrusted"),
    ("coverage", "component_coverage_missing"),
    ("participant", "participant_set_mismatch"),
    ("cursor", "cursor_coverage_missing"),
    ("mode", "mode_unsupported"),
    ("clock", "clock_policy_mismatch"),
    ("lineage", "source_lineage_mismatch"),
    ("commit", "commit_mismatch"),
    ("configuration", "execution_identity_mismatch"),
    ("policies", "execution_identity_mismatch"),
    ("source", "execution_identity_mismatch"),
])
def test_strict_preflight_refusals_do_not_dispatch(tmp_path, mutation, blocker):
    manifest, _, store, support = fixture(tmp_path)
    overrides = {}
    if mutation == "terminal":
        manifest = manifest.model_copy(update={"terminal": True})
    elif mutation == "unresolved":
        manifest = manifest.model_copy(update={"unresolved_effects": 1})
    elif mutation == "codec":
        support = support.model_copy(update={"components": (support.components[0].model_copy(update={"codec": digest("other")}), *support.components[1:])})
    elif mutation == "trust":
        support = support.model_copy(update={"trusted_decoders": ()})
    elif mutation == "coverage":
        component = manifest.components[1].model_copy(update={"coverage": ("history",)})
        manifest = manifest.model_copy(update={"components": (manifest.components[0], component, manifest.components[2])})
    elif mutation == "participant":
        support = support.model_copy(update={"components": support.components[:1]})
    elif mutation == "cursor":
        manifest = manifest.model_copy(update={"cursors": manifest.cursors[-1:]})
    elif mutation == "mode":
        support = support.model_copy(update={"modes": ("inspect",)})
    elif mutation == "clock":
        support = support.model_copy(update={"clock_policy": "absolute_deadline"})
    elif mutation == "lineage":
        overrides.update(source_lineage_id=uuid4())
        overrides["target_lineage_id"] = overrides["source_lineage_id"]
    elif mutation == "commit":
        overrides["commit"] = CheckpointCommitAnchorV1(manifest=digest("wrong"), sequence=manifest.sequence, chain_head=manifest.chain_head)
    else:
        change = {mutation: store.put(b"other-config")} if mutation != "source" else {"source": digest("other-source")}
        overrides["execution"] = manifest.execution.model_copy(update=change)
    report = check_checkpoint_restore(manifest, store, request_for(manifest, **overrides), support)
    assert not report.declaration_supported
    assert any(blocker in value for value in report.blockers)
    assert report.decoder_execution == "not_run"


@pytest.mark.parametrize("kind", ["missing", "corrupt", "symlink", "parent_symlink", "duplicate_json", "identity", "namespace"])
def test_missing_corrupt_private_and_inconsistent_closure_refuse(tmp_path, kind):
    manifest, application, store, support = fixture(tmp_path)
    bundle = tmp_path / "export"
    export_checkpoint_bundle(manifest, store, bundle)
    item = application.participants[0].artifact
    blob = bundle / "blobs" / item.object_key
    if kind == "missing":
        blob.unlink()
    elif kind == "corrupt":
        blob.write_bytes(b"x" * item.size_bytes)
    elif kind == "symlink":
        blob.unlink()
        blob.symlink_to(tmp_path / "source" / item.object_key)
    elif kind == "parent_symlink":
        shutil.rmtree(bundle / "blobs")
        (bundle / "blobs").symlink_to(tmp_path / "source", target_is_directory=True)
    elif kind == "duplicate_json":
        path = bundle / "manifest.json"
        path.write_text('{"run_id":"secret-value",' + path.read_text()[1:])
    elif kind == "identity":
        manifest = manifest.model_copy(update={"checkpoint": uuid4()})
        (bundle / "manifest.json").write_text(manifest.model_dump_json())
    else:
        configuration = manifest.execution.configuration.model_copy(update={"namespace": "another-org"})
        manifest = manifest.model_copy(update={"execution": manifest.execution.model_copy(update={"configuration": configuration})})
        (bundle / "manifest.json").write_text(manifest.model_dump_json())
    with pytest.raises((CheckpointFailure, ValidationError)) as error:
        load_checkpoint_bundle(bundle)
    assert "secret-value" not in str(error.value)
    if kind not in ("duplicate_json",):
        report = check_checkpoint_restore(manifest, CheckpointBundleStore(bundle, namespace=manifest.application.namespace), request_for(manifest), support)
        assert not report.declaration_supported


def test_forks_are_explicit_and_cannot_change_world_or_parent(tmp_path):
    manifest, _, store, support = fixture(tmp_path)
    before = manifest.model_dump_json()
    changed = manifest.execution.model_copy(update={"policies": store.put(b'{"new_policy":"explicit"}')})
    child = dict(target_run_id="child", target_lineage_id=uuid4(), child_budget=store.put(b'{"cap_usd":"1.00"}'))
    for mode, expected in (("same_policy_fork", False), ("edited_policy_fork", True)):
        report = check_checkpoint_restore(manifest, store, request_for(manifest, mode=mode, execution=changed, **child), support)
        assert report.declaration_supported == expected
    same = check_checkpoint_restore(manifest, store, request_for(manifest, mode="same_policy_fork", **child), support)
    assert same.declaration_supported
    changed_world = changed.model_copy(update={"world": digest("other-world")})
    assert not check_checkpoint_restore(manifest, store, request_for(manifest, mode="edited_policy_fork", execution=changed_world, **child), support).declaration_supported
    with pytest.raises(ValidationError, match="child_identity_and_budget"):
        request_for(manifest, mode="edited_policy_fork")
    with pytest.raises(ValidationError, match="continuation_identity_changed"):
        request_for(manifest, target_run_id="silent-new-run")
    assert manifest.model_dump_json() == before


def test_inspection_is_separate_from_decoder_trust_and_terminal_restore(tmp_path):
    manifest, _, store, support = fixture(tmp_path)
    terminal = manifest.model_copy(update={"terminal": True})
    support = support.model_copy(update={"trusted_decoders": ()})
    assert check_checkpoint_restore(terminal, store, request_for(terminal, mode="inspect"), support).declaration_supported
    report = check_checkpoint_restore(terminal, store, request_for(terminal, mode="replay"), support)
    assert not report.declaration_supported
    assert "checkpoint_decoder_untrusted:environment" in report.blockers


def test_strict_selection_uses_position_and_refuses_ambiguous_or_mixed_lineage(tmp_path):
    manifest, _, store, support = fixture(tmp_path)
    early = at_position(manifest, 2, digest("early"))
    early_report = check_checkpoint_restore(early, store, request_for(early), support)
    latest_report = check_checkpoint_restore(manifest, store, request_for(manifest), support)
    assert select_strict_checkpoint([latest_report, early_report]) == latest_report
    assert select_strict_checkpoint([early_report, latest_report, latest_report]) == latest_report
    terminal = manifest.model_copy(update={"terminal": True})
    rejected = check_checkpoint_restore(terminal, store, request_for(terminal), support)
    assert select_strict_checkpoint([rejected, early_report]) == early_report
    assert select_strict_checkpoint([rejected]) is None
    ambiguous = at_position(manifest, manifest.sequence, digest("rewritten-prefix"))
    ambiguous_report = check_checkpoint_restore(ambiguous, store, request_for(ambiguous), support)
    with pytest.raises(CheckpointFailure, match="position_ambiguous"):
        select_strict_checkpoint([latest_report, ambiguous_report])
    other = manifest.model_copy(update={"lineage_id": uuid4()})
    with pytest.raises(CheckpointFailure, match="selection_scope"):
        select_strict_checkpoint([latest_report, check_checkpoint_restore(other, store, request_for(other), support)])
    with pytest.raises(CheckpointFailure, match="requires_strict_mode"):
        select_strict_checkpoint([check_checkpoint_restore(manifest, store, request_for(manifest, mode="inspect"), support)])


@pytest.mark.parametrize("amount", ["-1", "NaN", "Infinity", "1e3", "01", 0.1, True])
def test_liability_numbers_never_coerce_unknown_or_lose_decimal_precision(amount):
    with pytest.raises(ValidationError):
        CheckpointLiabilityV1(resource="cost", provider="judge", unit="usd", consumed=amount, outstanding=None)


def test_publication_failure_never_exposes_partial_manifest(tmp_path, monkeypatch):
    manifest, _, store, _ = fixture(tmp_path)
    import synth_containers.checkpoints.portable as portable

    original = portable.os.link
    def fail_marker(source, destination):
        if Path(destination).name == "manifest.json":
            raise OSError("injected-private-content")
        return original(source, destination)
    monkeypatch.setattr(portable.os, "link", fail_marker)
    output = tmp_path / "export"
    with pytest.raises(CheckpointFailure, match="publication_in_doubt"):
        export_checkpoint_bundle(manifest, store, output)
    assert not (output / "manifest.json").exists()
    assert (output / ".manifest.pending").exists()
    with pytest.raises(CheckpointFailure, match="destination_exists"):
        export_checkpoint_bundle(manifest, store, output)


def test_cli_export_inspect_check_and_clean_error_surface(tmp_path, capsys):
    manifest, _, _, support = fixture(tmp_path)
    manifest_file, request_file, support_file = (tmp_path / (name + ".json") for name in ("descriptor", "request", "support"))
    for path, model in ((manifest_file, manifest), (request_file, request_for(manifest)), (support_file, support)):
        path.write_text(model.model_dump_json())
    output = tmp_path / "bundle"
    assert main(["export", str(manifest_file), "--store", str(tmp_path / "source"), "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["decoder_execution"] == "not_run"
    assert main(["inspect", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["manifest_digest"] == manifest.digest()
    assert main(["check", str(output), "--request", str(request_file), "--runtime-support", str(support_file)]) == 0
    assert json.loads(capsys.readouterr().out)["admission"] == "not_checked"
    request_file.write_text('{"secret-input":"never echo this"}')
    assert main(["check", str(output), "--request", str(request_file), "--runtime-support", str(support_file)]) == 2
    assert json.loads(capsys.readouterr().out) == {"error": "checkpoint_descriptor_invalid"}
    absent = tmp_path / "does-not-exist"
    assert main(["inspect", str(absent)]) == 2
    assert not absent.exists()
    assert "never echo this" not in capsys.readouterr().out


def test_export_then_actual_existing_joint_restore_runs_in_fresh_process(tmp_path):
    """Exercise the real restore coordinator after deleting all source-path state."""
    manifest, application, store, support = fixture(tmp_path)
    # Two policies alternate actions, retaining per-policy RNG, history and a
    # partially consumed plan. The world carries RNG and queued actor messages.
    scripted_world = '''
def advance(states):
    env = states["environment"]
    actor = "policy_a" if env["turn"] % 2 == 0 else "policy_b"
    policy = states[actor]
    if not policy["plan"]:
        policy["rng"] = (1103515245 * policy["rng"] + 12345) % (2**31)
        policy["plan"] = [policy["rng"] % 5 + 1, policy["rng"] % 3 + 1]
    action = policy["plan"].pop(0)
    inbox = env["mailboxes"][actor]
    observed = sum(inbox)
    inbox.clear()
    env["rng"] = (1664525 * env["rng"] + 1013904223) % (2**32)
    env["value"] += action + env["rng"] % 2 + observed
    env["turn"] += 1
    other = "policy_b" if actor == "policy_a" else "policy_a"
    env["mailboxes"][other].append(action % 2)
    reward = env["value"] % 7
    policy["history"].append([action, observed, reward])
    return [actor, action, observed, reward, env["value"]]
'''
    namespace = {}
    exec(scripted_world, namespace)
    advance = namespace["advance"]
    states = {
        "environment": {"rng": 17, "turn": 0, "value": 0, "mailboxes": {"policy_a": [], "policy_b": []}},
        "policy_a": {"rng": 42, "history": [], "plan": []},
        "policy_b": {"rng": 43, "history": [], "plan": []},
    }
    prefix = [advance(states) for _ in range(5)]
    assert any(states[name]["plan"] for name in ("policy_a", "policy_b"))
    assert any(states["environment"]["mailboxes"].values())
    snapshots = tuple(row.model_copy(update={"artifact": store.put(json.dumps(states[row.cut.participant.name]).encode())})
                      for row in application.participants)
    application = application.model_copy(update={"participants": snapshots})
    manifest = manifest.model_copy(update={"application": store.put(application.canonical_bytes())})
    uninterrupted_suffix = [advance(states) for _ in range(7)]
    expected_final_state = json.loads(json.dumps(states))
    bundle = tmp_path / "bundle"
    export_checkpoint_bundle(manifest, store, bundle)
    shutil.rmtree(tmp_path / "source")
    # This is a trusted deterministic codec fixture. Only the child decodes it,
    # after the provider-free preflight. No environment/provider is contacted.
    script = scripted_world + '''
import asyncio, json, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4
from synth_containers.checkpoints import *
root = Path(sys.argv[1])
inspection = load_checkpoint_bundle(root / "bundle")
m = inspection.manifest
store = CheckpointBundleStore(root / "bundle", namespace=m.application.namespace)
app = ApplicationCheckpoint.model_validate_json(store.get(m.application))
support = CheckpointRuntimeSupportV1.model_validate_json((root / "support.json").read_bytes())
request = CheckpointRestoreRequestV1.model_validate_json((root / "request.json").read_bytes())
preflight = check_checkpoint_restore(m, store, request, support)
assert preflight.declaration_supported
states, receipts, calls = {}, {}, []
class Participant:
    def __init__(self, snapshot):
        self.identity = snapshot.cut.participant.model_copy(update={"origin_instance":uuid4()})
    async def restore(self, request, source, payload, fence):
        calls.append(self.identity.name)
        states[self.identity.name] = json.loads(payload)
        receipt = ParticipantRestore(operation=request.operation, checkpoint=m.checkpoint,
            checkpoint_sha256=m.application.sha256, participant_name=self.identity.name,
            service_identity=self.identity.service_identity,
            source_origin_instance=source.cut.participant.origin_instance,
            restored_origin_instance=self.identity.origin_instance, receipt=uuid4())
        receipts[self.identity.name] = receipt
        return receipt
class Authority:
    async def restart(self, request, manifest):
        return app.fence.model_copy(update={"source_generation":2,"generation":3})
    async def complete_restore(self, request, result):
        pass
result_store = FileArtifactStore(root / "result-blobs", namespace="restore-results")
class RestoreStore:
    def get(self, reference):
        return (result_store if reference.namespace == "restore-results" else store).get(reference)
    def put(self, payload):
        return result_store.put(payload)
async def run():
    participants = tuple(Participant(row) for row in app.participants)
    restore = RestoreRequest(operation=uuid4(), connector=app.fence.connector,
        session=app.fence.session, expected_generation=2, artifact=m.application,
        deadline_at=datetime.now(timezone.utc)+timedelta(seconds=30))
    coordinator = ApplicationRestoreCoordinator(authority=Authority(), store=RestoreStore(),
        journal=FileCheckpointJournal(root / "restore-journal"))
    result = await coordinator.restore(restore, participants)
    again = await coordinator.restore(restore, participants)
    assert again == result and len(receipts) == len(calls) == 3
    suffix = [advance(states) for _ in range(7)]
    print(json.dumps({"states":states, "receipts":len(result.receipts),
        "suffix":suffix,
        "liabilities":[row.model_dump() for row in preflight.retained_liabilities]}))
asyncio.run(run())
'''
    (tmp_path / "support.json").write_text(support.model_dump_json())
    (tmp_path / "request.json").write_text(request_for(manifest).model_dump_json())
    process = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=20)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result["receipts"] == 3
    assert set(result["states"]) == {row.cut.participant.name for row in application.participants}
    assert result["suffix"] == uninterrupted_suffix
    assert prefix[-1] != result["suffix"][0]
    assert result["states"] == expected_final_state
    assert result["liabilities"][1]["outstanding"] is None


def test_no_new_identity_or_boolean_coercions_at_public_boundaries(tmp_path):
    manifest, _, store, _ = fixture(tmp_path)
    with pytest.raises(ValidationError):
        inspect_portable_checkpoint(manifest.model_copy(update={"sequence": True}), store)
    with pytest.raises(ValidationError):
        PortableCheckpointManifestV1.model_validate({**manifest.model_dump(mode="json"), "unknown": "value"})
    with pytest.raises(ValidationError):
        PortableCheckpointManifestV1.model_validate({**manifest.model_dump(mode="json"), "lineage_id": "00000000-0000-0000-0000-000000000000"})
    bad_ref = manifest.execution.configuration.model_copy(update={"object_key": "../private"})
    bad = manifest.model_copy(update={"execution": manifest.execution.model_copy(update={"configuration": bad_ref})})
    with pytest.raises(ValidationError):
        inspect_portable_checkpoint(bad, store)
    with pytest.raises(ValidationError, match="event_cursor_inconsistent"):
        inspect_portable_checkpoint(manifest.model_copy(update={"sequence": 2}), store)
