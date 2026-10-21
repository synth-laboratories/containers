"""Portable CLI declaration, event views and checkpoint selection boundaries."""

import io
import json
import shutil

import pytest

from synth_containers.eval_cli import TerminalProgress, candidates_from, _example_toml, main
from synth_containers.eval_models import EvaluationPlanV1, CandidateBindingV1, ScriptedCandidateV1, EvaluationRunDeclarationV1
from synth_containers.eval_run import (
    describe_evaluation, local_execution_source_digest, local_execution_dependencies_digest,
    read_actions, _append_action,
)
from synth_containers.package_models import content_digest
from synth_containers.checkpoints.gold import commit_gold_checkpoint, restore_gold_checkpoint
from synth_containers.checkpoints.portable import load_checkpoint_bundle
from synth_containers.checkpoints.portable_models import CheckpointCommitAnchorV1
from synth_containers.gold_http import StepResult

D = "sha256:" + "a" * 64


def plan(**updates):
    return EvaluationPlanV1.model_validate({"benchmark": "craftax", "adapter": "local-gold-scripted-v1",
        "source_digest": local_execution_source_digest(), "dependencies_digest": local_execution_dependencies_digest(),
        "scorer_digest": D, "image_digest": D, "task": {"world": {"classic": True}},
        "seeds": [0, 1], "parallelism": 2, "limits": {"steps": 5, "calls": 8, "active_seconds": 60}, **updates})


def test_config_identity_candidates_and_credential_refusal(tmp_path):
    source = tmp_path / "candidate.toml"
    source.write_text('kind="scripted-actions-v1"\nactions=["noop"]\n')
    candidates = candidates_from(["label=candidate.toml"], relative=tmp_path)
    p = plan()
    description = describe_evaluation(p, candidates)
    assert description["supported"] and description["aggregate_maximum_provider_cost_usd"] == "0"
    assert description["candidate_bindings"][0]["model"] == "scripted"
    source.write_text('kind="scripted-actions-v1"\nactions=["left"]\n')
    assert candidates_from(["label=candidate.toml"], relative=tmp_path)[0].content.digest != candidates[0].content.digest
    with pytest.raises(ValueError):
        plan(credentials={"OPENAI_API_KEY": "plaintext"})
    assert not describe_evaluation(plan(adapter="rhodes-v1"), candidates)["supported"]
    import tomllib
    assert EvaluationPlanV1.model_validate(tomllib.loads(_example_toml(json.loads(p.model_dump_json(exclude_none=True))))) == p


def test_tty_at_most_ten_and_short_terminal_adaptation(monkeypatch):
    monkeypatch.setattr(shutil, "get_terminal_size", lambda _: (100, 24))
    stream = io.StringIO()
    renderer = TerminalProgress(stream, mode="tty")
    for index in range(15):
        renderer({"seq": index, "run_id": "r", "lane": f"lane-{index}", "kind": "eval.usage.sampled",
            "payload": {"steps": index, "step_limit": 20}})
    last = stream.getvalue().rsplit("\x1b[12A", 1)[-1]
    assert last.count(" | ") == 10 and "+5 additional rollouts" in last
    monkeypatch.setattr(shutil, "get_terminal_size", lambda _: (40, 5))
    short = io.StringIO()
    renderer = TerminalProgress(short, mode="tty")
    renderer.lanes = {str(i): {"steps": None, "limit": None, "latest": "", "status": "running"} for i in range(15)}
    renderer({"seq": 1, "run_id": "r", "lane": None, "kind": "eval.phase", "payload": {}})
    assert short.getvalue().count(" | ") == 2
    plain = io.StringIO()
    TerminalProgress(plain, mode="json")({"seq": 1, "run_id": "r", "kind": "eval.phase", "payload": {}})
    assert "\x1b" not in plain.getvalue() and json.loads(plain.getvalue())["seq"] == 1


def test_action_journal_rejects_corrupt_and_partial_prefix(tmp_path):
    path = tmp_path / "actions.jsonl"
    row = {"position": 1, "previous_head": content_digest({"action_journal": "genesis-v1"}), "action": "noop",
           "optional": None, "nested": {"unknown": None, "known": 0}}
    row["chain_head"] = content_digest(row)
    _append_action(path, row)
    assert read_actions(path) == [row]
    path.write_bytes(path.read_bytes() + b'{"incomplete":')
    with pytest.raises(ValueError, match="incomplete"):
        read_actions(path)


class DeterministicWorld:
    """Only a contract fixture; actual Craftax qualification is a separate native run."""
    def __init__(self):
        self.position = 0
    def checkpoint(self):
        return {"blob": str(self.position), "checkpoint_id": "native", "step_index": self.position}
    def restore(self, blob):
        self.position = int(blob)
        return StepResult({"position": self.position}, 0, False, ["noop"], "", "", self.position)
    def drain_native_events(self):
        return []


def test_native_joint_cut_identity_forks_and_terminal_refusal(tmp_path):
    from uuid import uuid4
    candidate = ScriptedCandidateV1(kind="scripted-actions-v1", actions=("noop",), repeat=True)
    run = EvaluationRunDeclarationV1(run_id="parent", lineage_id=str(uuid4()), plan=plan(),
        candidates=(CandidateBindingV1(label="p", content=candidate),))
    world = DeterministicWorld()
    head = content_digest({"action_journal": "genesis-v1"})
    bundle = commit_gold_checkpoint(tmp_path, run, candidate, world=world, rows=[], chain_head=head, calls=0, active_ns=0, terminal=False)
    inspection = load_checkpoint_bundle(bundle)
    anchor = CheckpointCommitAnchorV1(manifest=inspection.manifest_digest, sequence=0, chain_head=head)
    restored, policy, result = restore_gold_checkpoint(bundle, run, candidate, world=world, expected_commit=anchor)
    assert policy["calls"] == 0 and result.env_steps == 0
    changed = ScriptedCandidateV1(kind="scripted-actions-v1", actions=("left",), repeat=True)
    with pytest.raises(ValueError, match="execution_identity"):
        restore_gold_checkpoint(bundle, run, changed, world=world, expected_commit=anchor)
    child = run.model_copy(update={"run_id": "child", "lineage_id": str(uuid4())})
    with pytest.raises(ValueError, match="child_identity_and_budget"):
        restore_gold_checkpoint(bundle, child, changed, world=world, mode="edited_policy_fork", expected_commit=anchor)
    restore_gold_checkpoint(bundle, child, changed, world=world, mode="edited_policy_fork",
        child_budget={"authority": "owner", "reference": "child"}, expected_commit=anchor)
    # A newer committed point with a different policy does not replace the
    # greatest compatible strict cut. Selection never falls back to fresh run.
    from synth_containers.eval_run import _latest_checkpoint
    rows = [{"action": "noop", "reward": 0, "chain_head": D}]
    world.position = 1
    commit_gold_checkpoint(tmp_path, run, changed, world=world, rows=rows, chain_head=D,
                           calls=1, active_ns=1, terminal=False)
    selected, selected_anchor = _latest_checkpoint(tmp_path, rows, run=run, candidate=candidate)
    assert selected == bundle and selected_anchor.sequence == 0


def test_explicit_detach_refusal_before_allocating_root(tmp_path, capsys):
    source = tmp_path / "evaluation.json"
    source.write_text(plan().model_dump_json())
    candidate = tmp_path / "candidate.toml"
    candidate.write_text('kind="scripted-actions-v1"\nactions=["noop"]\n')
    assert main(["run", str(source), "--candidate", "script=candidate.toml", "--root", str(tmp_path / "run"), "--detach"]) == 2
    assert not (tmp_path / "run").exists()


def test_deadline_and_source_mismatch_refuse():
    with pytest.raises(ValueError):
        plan(limits={"steps": 1, "calls": 2, "active_seconds": 60, "clock_policy": "absolute_deadline"})
    p = plan(source_digest=D)
    assert "execution_source_identity_mismatch" in describe_evaluation(p, ()) ["blockers"]


def test_retention_roots_survive_restart_and_protect_ring_eviction(tmp_path):
    from synth_containers.platform.state import CompatPlatform
    from synth_containers.platform.targets import OPENENV_ECHO
    from synth_containers.checkpoint_schedule import apply_checkpoint_keep_last, parse_checkpoint_schedule
    platform = CompatPlatform(OPENENV_ECHO, storage_root=tmp_path)
    for index in range(4):
        platform.record_checkpoint({"checkpoint_id": f"native_{index}", "restore_eligible": True})
    platform.retain_checkpoint("native_0", owner_kind="fork", owner_id="child")
    recovered = CompatPlatform(OPENENV_ECHO, storage_root=tmp_path)
    with pytest.raises(RuntimeError, match="retention_roots"):
        recovered.drop_checkpoint("native_0")
    apply_checkpoint_keep_last(recovered, parse_checkpoint_schedule({"mode": "per_policy_call", "checkpoint_id_prefix": "native", "keep_last": 1}))
    assert recovered.checkpoint_is_retained("native_0") and "native_0" in recovered.checkpoints
    recovered.release_checkpoint_root("native_0", owner_kind="fork", owner_id="child")
    recovered.drop_checkpoint("native_0")
    assert "native_0" not in recovered.checkpoints
