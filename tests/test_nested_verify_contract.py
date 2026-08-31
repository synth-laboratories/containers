"""`NestedTrialRuntime._verify`: what the grader is allowed to see.

Both defects these pin produced a *number*, not an error, which is why they
need tests rather than an exception path:

* A grader that writes outside the workspace (DeepSWE's absolute ``/logs``) has
  to share that directory with the collection step, because the two run in
  separate containers. Without the shared mount the collected artifact is gone
  by the time the grader looks for it and the base state is graded instead.
* A benchmark declaring ``environment_mode = "separate"`` grades its own
  pristine checkout and takes the agent's work only as the collected patch.
  Mounting the mutated tree over it leaves edits outside the patch in the graded
  state and resets test files to the agent's HEAD, which fails pass-to-pass
  tests that have nothing to do with the agent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from synth_containers.nested_runtime import NestedTrialRuntime, TrialImage


class _Log:
    def __init__(self) -> None:
        self.rows: list[tuple[str, dict[str, Any]]] = []

    def append(self, kind: str, payload: dict[str, Any]) -> None:
        self.rows.append((kind, payload))


class _Workspace:
    """Identity translation; the host-path mapping is covered elsewhere."""

    def to_host(self, path: Path) -> Path:
        return path


class _Result:
    def __init__(self, name: str) -> None:
        self.name = name
        self.exit_code = 0


def _capture(monkeypatch: Any) -> list[dict[str, Any]]:
    """Record every sibling container the runtime would start."""

    seen: list[dict[str, Any]] = []

    class _FakeTrial:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def run(self) -> _Result:
            seen.append(dict(self.kwargs))
            return _Result("fake-container")

    monkeypatch.setattr("synth_containers.nested_runtime.NestedTrial", _FakeTrial)
    return seen


def _runtime(trial: TrialImage) -> NestedTrialRuntime:
    return NestedTrialRuntime(
        environment_ref="env:test",
        trial_images={trial.id: trial},
        default_trial=trial.id,
    )


def _separate_verifier_trial() -> TrialImage:
    return TrialImage(
        id="deepswe/task",
        image="agent:v1",
        verify_image="verify:v1",
        workspace_source="/app",
        workspace_mount="/app",
        logs_mount="/logs",
        verify_mounts_workspace=False,
        collect_commands=(("bash", "-lc", "git diff > /logs/artifacts/model.patch"),),
        reward_json_path="verifier/reward.json",
        result_path="verifier/reward.json",
        allow_unpinned=True,
    )


def test_collection_and_grading_share_the_logs_mount(tmp_path: Path, monkeypatch) -> None:
    seen = _capture(monkeypatch)
    trial = _separate_verifier_trial()
    rollout_dir = tmp_path / "r1"
    (rollout_dir / "workspace").mkdir(parents=True)

    _runtime(trial)._verify(trial, rollout_dir, _Workspace(), "r1", _Log())

    collect, verify = seen
    logs_host = str(rollout_dir / "logs")
    # The artifact the collect step writes must survive into the grader.
    assert collect["mounts"][logs_host] == "/logs"
    assert verify["mounts"][logs_host] == "/logs"


def test_a_separate_verifier_never_sees_the_agent_tree(tmp_path: Path, monkeypatch) -> None:
    seen = _capture(monkeypatch)
    trial = _separate_verifier_trial()
    rollout_dir = tmp_path / "r1"
    (rollout_dir / "workspace").mkdir(parents=True)

    _runtime(trial)._verify(trial, rollout_dir, _Workspace(), "r1", _Log())

    collect, verify = seen
    # Collection reads the agent's work; grading must not.
    assert "/app" in collect["mounts"].values()
    assert "/app" not in verify["mounts"].values()
    assert verify["image"] == "verify:v1"


def test_in_place_grading_still_mounts_the_workspace(tmp_path: Path, monkeypatch) -> None:
    seen = _capture(monkeypatch)
    # The GameBench DEO shape: one image, grade the mutated tree in place.
    trial = TrialImage(id="gb-cpo-rogue", image="gb-cpo-rogue:local", allow_unpinned=True)
    rollout_dir = tmp_path / "r1"
    (rollout_dir / "workspace").mkdir(parents=True)

    _runtime(trial)._verify(trial, rollout_dir, _Workspace(), "r1", _Log())

    (verify,) = seen
    assert verify["mounts"] == {str(rollout_dir / "workspace"): "/workspace"}
    assert verify["image"] == "gb-cpo-rogue:local"


def test_reward_is_read_from_the_logs_mount(tmp_path: Path, monkeypatch) -> None:
    trial = _separate_verifier_trial()
    rollout_dir = tmp_path / "r1"
    (rollout_dir / "workspace").mkdir(parents=True)

    class _GradingTrial:
        """Writes reward.json the way the real grader does: into the mount."""

        def __init__(self, **kwargs: Any) -> None:
            self.mounts = kwargs["mounts"]

        def run(self) -> _Result:
            host = next(h for h, c in self.mounts.items() if c == "/logs")
            target = Path(host) / "verifier"
            target.mkdir(parents=True, exist_ok=True)
            (target / "reward.json").write_text(json.dumps({"reward": 1, "f2p": 1.0}))
            return _Result("fake-container")

    monkeypatch.setattr("synth_containers.nested_runtime.NestedTrial", _GradingTrial)
    reward, payload = _runtime(trial)._verify(
        trial, rollout_dir, _Workspace(), "r1", _Log()
    )

    # Read from the logs mount, not from the workspace: the grader never wrote
    # anything there, and falling back to it would report an absent reward.
    assert reward == 1.0
    assert payload["f2p"] == 1.0


def _staged_candidate_trial() -> TrialImage:
    """The GameBench DEO shape after the freeze fix: only the candidate crosses."""

    return TrialImage(
        id="gb-cpo-rogue",
        image="gb-cpo-rogue:local",
        logs_mount="/out",
        reward_path="verifier/reward.txt",
        verify_mounts_workspace=False,
        candidate_paths=("candidates/rogue",),
        candidate_sandbox_docker=True,
        environment={"GAMEBENCH_TASK": "rogue-singleplayer", "CANDIDATE_SUBDIR": "rogue"},
        allow_unpinned=True,
    )


def test_only_the_candidate_copy_reaches_the_grader(tmp_path: Path, monkeypatch) -> None:
    """The freeze item, pinned.

    The grader reads exactly one thing out of the workspace. Everything else it
    needs — runner, suite, baseline — resolves under the baked `/task`, so the
    agent's tree has no reason to be there and every reason not to be: it is the
    tree the untrusted candidate was just written into.
    """

    monkeypatch.setenv("SYNTH_PLATFORM_ID", "test-platform")
    seen = _capture(monkeypatch)
    trial = _staged_candidate_trial()
    rollout_dir = tmp_path / "r1"
    workspace = rollout_dir / "workspace"
    (workspace / "candidates" / "rogue").mkdir(parents=True)
    (workspace / "candidates" / "rogue" / "policy.py").write_text("# candidate\n")
    # Things the agent could plant that must not cross.
    (workspace / "gamebench").mkdir()
    (workspace / "gamebench" / "suite.json").write_text("{}\n")

    _runtime(trial)._verify(trial, rollout_dir, _Workspace(), "r1", _Log())

    (verify,) = seen
    mounted_at_workspace = [
        host for host, container in verify["mounts"].items() if container == "/workspace"
    ]
    assert mounted_at_workspace == [str(rollout_dir / "candidate")]
    assert str(workspace) not in verify["mounts"]

    staged = rollout_dir / "candidate"
    assert (staged / "candidates" / "rogue" / "policy.py").read_text() == "# candidate\n"
    assert not (staged / "gamebench").exists()


def test_the_candidate_sandbox_binds_the_copy_not_the_agent_tree(
    tmp_path: Path, monkeypatch
) -> None:
    """Otherwise the staging is cosmetic.

    The grader starts throwaway containers for each untrusted candidate and
    binds the HOST side of its own workspace mount into them. Left pointing at
    the agent's directory, those sandboxes would mount the very tree the staging
    exists to keep out.
    """

    monkeypatch.setenv("SYNTH_PLATFORM_ID", "test-platform")
    seen = _capture(monkeypatch)
    trial = _staged_candidate_trial()
    rollout_dir = tmp_path / "r1"
    (rollout_dir / "workspace" / "candidates" / "rogue").mkdir(parents=True)

    _runtime(trial)._verify(trial, rollout_dir, _Workspace(), "r1", _Log())

    (verify,) = seen
    assert verify["environment"]["GAMEBENCH_POLICY_SANDBOX_HOST_MAP"] == (
        f"/workspace:{rollout_dir / 'candidate'}"
    )


def test_a_missing_candidate_is_reported_not_papered_over(tmp_path: Path, monkeypatch) -> None:
    """An agent that produced nothing must be graded as having produced nothing."""

    monkeypatch.setenv("SYNTH_PLATFORM_ID", "test-platform")
    seen = _capture(monkeypatch)
    log = _Log()
    trial = _staged_candidate_trial()
    rollout_dir = tmp_path / "r1"
    (rollout_dir / "workspace").mkdir(parents=True)

    _runtime(trial)._verify(trial, rollout_dir, _Workspace(), "r1", log)

    kinds = {kind for kind, _ in log.rows}
    assert "nested.candidate.missing" in kinds
    (verify,) = seen
    # Still mounted, still graded: the grader decides what an empty candidate
    # scores, not the runtime.
    assert str(rollout_dir / "candidate") in verify["mounts"]


def test_the_receipt_names_what_actually_crossed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SYNTH_PLATFORM_ID", "test-platform")
    _capture(monkeypatch)
    log = _Log()
    trial = _staged_candidate_trial()
    rollout_dir = tmp_path / "r1"
    (rollout_dir / "workspace" / "candidates" / "rogue").mkdir(parents=True)

    _runtime(trial)._verify(trial, rollout_dir, _Workspace(), "r1", log)

    kinds = [kind for kind, _ in log.rows]
    assert kinds.index("span.verifier.opened") < kinds.index("nested.verified")
    assert kinds.index("nested.verified") < kinds.index("span.verifier.closed")
    verified = next(payload for kind, payload in log.rows if kind == "nested.verified")
    assert verified["workspace_mounted_into_verifier"] is False
    assert verified["staged_candidate_paths"] == ["candidates/rogue"]
    assert verified["candidate_paths_declared"] == ["candidates/rogue"]
