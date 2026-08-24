"""``TargetSpec.runtime`` for nested platforms: one rollout = sibling containers.

The sequence is the one ``_run_harbor_agent.sh`` already runs on the host, moved
inside a platform image that talks to the **same** daemon:

    1. resolve the trial image        (digest-pinned rollout_environment)
    2. docker run --rm <img> tar …    extract the task workspace onto the shared
                                      host workspace root
    3. run the agent policy           against that workspace (the policy ladder:
                                      codex_agentic writes candidate code there)
    4. docker run <img> <verify>      score the mutated workspace
    5. read reward.txt                the verifier's number is the reward
    6. reap                           every sibling labelled with this rollout

Every step appends to the durable event log, so a rollout that dies mid-way
still seals a terminal status instead of looking active forever.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .event_log import RolloutEventLog
from .nested import (
    HostWorkspace,
    NestedError,
    NestedTrial,
    platform_id,
    reap_expired,
    reap_rollout,
)
from .platform.state import CompatPlatform, RolloutPin
from .policies import build_planner
from .tracing.capture.redaction import scrub_text

__all__ = ["NestedTrialRuntime", "TrialImage"]


@dataclass(frozen=True, slots=True)
class TrialImage:
    """A digest-pinned per-trial image a rollout may name.

    ``extract_command`` streams the baked task workspace out of the image;
    ``verify_command`` scores the mutated workspace. Both run as siblings.
    """

    id: str
    image: str
    workspace_source: str = "/workspace"
    extract_command: tuple[str, ...] = ("bash", "-lc", "tar -C /workspace -cf - .")
    verify_command: tuple[str, ...] = ("bash", "-lc", "/task/tests/test.sh")
    workspace_mount: str = "/workspace"
    reward_path: str = "logs/verifier/reward.txt"
    result_path: str = "logs/verifier/result.json"
    environment: Mapping[str, str] = field(default_factory=dict)
    verify_timeout_seconds: float = 900.0
    allow_unpinned: bool = False
    # Some benchmarks grade in a SEPARATE image that carries hidden tests the
    # agent must never see (DeepSWE builds one FROM the agent image). Empty
    # means "grade in the same image", which is the GameBench DEO shape.
    verify_image: str = ""
    # A grader that writes a JSON object rather than a bare float. The float at
    # `reward_key` inside it is the reward; `reward_path` stays the fallback.
    reward_json_path: str = ""
    reward_key: str = "reward"
    # Ordered commands run in the verifier container before the grader, e.g.
    # DeepSWE's `[[verifier.collect]]` step that produces model.patch.
    collect_commands: tuple[tuple[str, ...], ...] = ()
    # Harbor bundles bake the task statement beside the tests, not inside the
    # workspace. Without it the agent is handed a directory and no objective,
    # so copy it in during extraction when the workspace does not carry one.
    instruction_source: str = "/task/instruction.md"
    # A grader that writes outside the workspace (DeepSWE uses an absolute
    # `/logs`) needs that directory to be a host mount: collection and grading
    # are two separate containers, so an artifact written to the container
    # filesystem by the first is gone by the time the second runs. Empty means
    # the grader writes inside the workspace, which is the GameBench DEO shape.
    logs_mount: str = ""
    # Some corpora keep the task statement beside the task definition rather
    # than inside the image (DeepSWE ships `instruction.md` in the corpus). A
    # platform that can read it supplies the text here; the runtime passes it
    # straight to the agent instead of writing a file into the workspace, which
    # for a git-graded benchmark would land in the diff.
    instruction_text: str = ""
    # Whether the grader sees the agent's tree. GameBench DEO grades the
    # workspace in place, so it must. DeepSWE declares
    # `verifier.environment_mode = "separate"`: it grades its own pristine
    # checkout and takes the agent's work ONLY as the collected model.patch,
    # resetting and applying it itself. Mounting the mutated tree there leaves
    # edits outside the patch in the graded state and resets the test files to
    # the agent's HEAD instead of the base commit, which silently fails
    # pass-to-pass tests that have nothing to do with the agent.
    verify_mounts_workspace: bool = True
    # The narrower alternative to mounting the whole tree. Workspace-relative
    # paths naming what the agent PRODUCED — for the GameBench DEO hillclimb,
    # `candidates/<subdir>`, which is the only thing its scorer reads from the
    # workspace at all (suite, baseline and runner all resolve under the baked
    # `/task`). They are copied into a throwaway directory that is mounted at
    # `workspace_mount` for the grader, so the grader sees the candidate and
    # nothing else, and anything it writes cannot reach back into the tree the
    # agent was editing. Requires `verify_mounts_workspace=False`; the two
    # together are what the freeze item "only the candidate artifact crosses
    # into the verifier" actually asks for.
    candidate_paths: tuple[str, ...] = ()
    # Benchmarks whose grader signals "scored zero" with a non-zero exit set
    # this; the default is to treat a failed verifier as ungraded, not as zero.
    reward_on_nonzero_exit: bool = False
    # Graders that sandbox untrusted candidate code in their own throwaway
    # containers. They need the host daemon and a way to translate an
    # in-container path into the host path the daemon will resolve for `-v`.
    # Only safe when the grader itself runs image-owned code: handing the socket
    # to a verifier that executes files from the agent's workspace hands the
    # agent the daemon.
    candidate_sandbox_docker: bool = False
    # The agent's wall-clock budget when the benchmark states one. A corpus that
    # declares `[agent] timeout_sec` is the authority for it; a platform-wide
    # default silently truncates a run that the benchmark intended to allow,
    # and a truncated agent grades as a weak one.
    agent_timeout_seconds: float = 0.0
    # Static, secret-free release projection stamped into retained evidence.
    # It makes two attempts comparable after all sibling containers are gone.
    environment_release: Mapping[str, Any] = field(default_factory=dict)

    def verifier_image(self) -> str:
        return self.verify_image or self.image


@dataclass(frozen=True)
class NestedTrialRuntime:
    """Runtime for a Harbor/Apex-style platform. Never Docker-in-Docker."""

    environment_ref: str
    trial_images: Mapping[str, TrialImage]
    default_trial: str = ""
    agent_timeout_seconds: float = 3600.0

    # ------------------------------------------------------------------ public

    def simulate(self, platform: CompatPlatform, pin: RolloutPin, log: RolloutEventLog) -> None:
        if platform.spec.environment_ref != self.environment_ref:
            raise ValueError(f"unknown_environment:{platform.spec.environment_ref}")
        rollout_id = pin.rollout_id
        trial = self._trial_for(pin)
        workspace = HostWorkspace.from_env()
        parent = platform_id()
        log.append(
            "env.episode.opened",
            {
                "trial_image_id": trial.id,
                "image": trial.image,
                "nested": "host_docker",
                **(
                    {"environment_release": dict(trial.environment_release)}
                    if trial.environment_release
                    else {}
                ),
            },
        )
        # Sweep orphans from earlier runs before adding to the pile. A verifier
        # whose client was killed keeps grading on the daemon until its own
        # deadline, and several of those at once is enough to exhaust the box.
        expired = reap_expired()
        if expired:
            log.append("nested.reaped_expired", {"containers": expired})
        rollout_dir = workspace.rollout_dir(rollout_id)
        try:
            self._extract(trial, rollout_dir, workspace, rollout_id, log)
            actions = self._run_agent(platform, pin, log, rollout_dir, trial, workspace)
            reward, result = self._verify(trial, rollout_dir, workspace, rollout_id, log)
        except Exception as exc:  # noqa: BLE001 — seal a terminal status, then re-raise nothing
            # The type alone cannot tell an operator whether the trial image was
            # missing, the workspace empty, or the agent unauthenticated. Carry
            # the message, scrubbed the same way any persisted string is.
            message, _ = scrub_text(str(exc))
            log.append(
                "status",
                {
                    "status": "failed",
                    "reason": "nested_trial_error",
                    "error_type": type(exc).__name__,
                    "error": message[:600],
                },
            )
            log.append("env.episode.closed", {"status": "failed"})
            high_water = log.high_water
            log.append("capture.high_water", {"high_water": high_water})
            log.append("capture.closed", {"high_water": high_water})
            log.mark_closed()
            pin.status = "failed"
            pin.terminal = True
            reap_rollout(rollout_id, parent=parent)
            return
        finally:
            reaped = reap_rollout(rollout_id, parent=parent)
            if reaped:
                log.append("nested.reaped", {"containers": reaped, "rollout_id": rollout_id})

        if reward is not None:
            log.append("reward_signal", {"value": reward, "authority": "verifier"})
            pin.reward_signals = [float(reward)]
            # The verifier's number is what a `script` evaluation plan scores.
            # Emitting the event alone leaves the trace showing a reward while
            # the sealed evaluation reports `absent`, which is worse than either
            # a score or an error: the run looks graded and is not.
            pin.native_script_reward = float(reward)
        log.append(
            "task_resolved",
            {
                "result": result,
                "actions": len(actions),
                **(
                    {"environment_release": dict(trial.environment_release)}
                    if trial.environment_release
                    else {}
                ),
            },
        )
        log.append("terminal", {"status": "completed", "reward": reward})
        log.append("env.episode.closed", {"status": "completed"})
        log.append("status", {"status": "completed"})
        high_water = log.high_water
        log.append("capture.high_water", {"high_water": high_water})
        log.append("capture.closed", {"high_water": high_water})
        log.mark_closed()
        pin.status = "completed"
        pin.terminal = True

    # ----------------------------------------------------------------- stages

    def _trial_for(self, pin: RolloutPin) -> TrialImage:
        """Resolve the trial the rollout named. A named trial is never defaulted.

        `task_instance_id` is the selector the operator API exposes and the one
        every image README documents; a recipe key overrides it for callers that
        pin by image rather than by task. If the caller named something, an
        unresolvable name must refuse: falling back to the default here would
        run one environment and report it under another's id, which is worse
        than an error because the resulting number looks legitimate.
        """

        requested = ""
        recipe = getattr(pin, "recipe", None)
        if isinstance(recipe, dict):
            requested = str(recipe.get("trial_image") or recipe.get("rollout_environment") or "")
        if not requested:
            metadata = getattr(pin, "metadata", None)
            if isinstance(metadata, dict):
                requested = str(metadata.get("trial_image") or "")
        if not requested:
            requested = str(getattr(pin, "task_instance_id", "") or "")
        named_by_caller = bool(requested)
        requested = requested or self.default_trial
        if not requested:
            raise NestedError("nested_trial_image_unnamed")
        trial = self.trial_images.get(requested)
        if trial is None:
            known = ",".join(sorted(self.trial_images))
            if named_by_caller:
                raise NestedError(f"nested_trial_image_unknown:{requested}:{known}")
            raise NestedError(f"nested_default_trial_unknown:{requested}:{known}")
        return trial

    def _extract(
        self,
        trial: TrialImage,
        rollout_dir: Path,
        workspace: HostWorkspace,
        rollout_id: str,
        log: RolloutEventLog,
    ) -> None:
        """Stream the baked workspace out of the trial image into the shared root."""

        target = rollout_dir / "workspace"
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        host_target = workspace.to_host(target)
        result = NestedTrial(
            image=trial.image,
            rollout_id=rollout_id,
            command=(
                "bash",
                "-lc",
                f"cp -a {trial.workspace_source}/. /out/"
                + (
                    f" && {{ [ -e /out/instruction.md ] || "
                    f"[ ! -f {trial.instruction_source} ] || "
                    f"cp {trial.instruction_source} /out/instruction.md; }}"
                    if trial.instruction_source
                    else ""
                ),
            ),
            mounts={str(host_target): "/out"},
            environment=dict(trial.environment),
            require_pinned=not trial.allow_unpinned,
            timeout_seconds=600.0,
        ).run()
        log.append(
            "nested.workspace.extracted",
            {"container": result.name, "image": trial.image, "path": str(target)},
        )
        if not any(target.iterdir()):
            raise NestedError("nested_workspace_extract_empty")

    def _run_agent(
        self,
        platform: CompatPlatform,
        pin: RolloutPin,
        log: RolloutEventLog,
        rollout_dir: Path,
        trial: TrialImage,
        workspace: HostWorkspace,
    ) -> Sequence[str]:
        """The policy half. The agent mutates the extracted workspace in place."""

        harness = str(pin.policy_ref.get("harness") or "").strip()
        if not harness:
            raise ValueError("simulate requires policy_ref.harness; start must not fill a default")
        config_id = str(pin.policy_ref.get("config") or "").strip()
        if not config_id:
            raise ValueError("simulate requires policy_ref.config; start must not default a model")
        policy = platform.policy_configs.get(config_id)
        config = dict(policy.config) if policy is not None else {}
        config.setdefault("workspace_root", str(rollout_dir / "workspace"))
        config.setdefault("timeout_seconds", self.agent_timeout_seconds)
        if trial.agent_timeout_seconds > 0:
            config["timeout_seconds"] = float(trial.agent_timeout_seconds)
        planner = build_planner(harness, config_id=config_id, config=config)

        log.append(
            "span.policy.opened",
            {"harness": harness, "config": config_id, "metadata": planner.metadata()},
        )
        observation = {
            "observation_text": trial.instruction_text.strip()
            or _instruction(rollout_dir / "workspace"),
            "valid_actions": ["done"],
            "workspace": str(rollout_dir / "workspace"),
        }
        # Forward the harness's own event stream into the trace. Without this a
        # nested rollout records the final action list and a token total and
        # throws the whole trajectory away — every command the agent ran, every
        # file it touched, every reasoning step — which is the part worth having
        # for an agentic benchmark. The gold lanes already do this; this lane
        # did not, so its traces looked complete and were not.
        def on_delta(payload: dict[str, Any]) -> None:
            if isinstance(payload, dict) and payload:
                log.append("span.policy.data", payload)

        try:
            with _environment(self._agent_environment(trial, rollout_dir, workspace)):
                actions = planner.plan(observation, on_delta=on_delta)
        finally:
            closer = getattr(planner, "close", None)
            if callable(closer):
                closer()
        trace_data = getattr(planner, "trace_data", None)
        if callable(trace_data):
            data = trace_data()
            if data:
                log.append("span.policy.data", data)
        log.append("span.policy.plan", {"actions": list(actions), "length": len(actions)})
        usage = planner.usage()
        log.append("span.policy.closed", {"status": "ok", "usage": usage})
        log.append("policy.session.closed", {"status": "ok", "calls": usage.get("calls")})
        pin.usage = dict(usage)
        return actions

    def _agent_environment(
        self, trial: TrialImage, rollout_dir: Path, workspace: HostWorkspace
    ) -> dict[str, str]:
        """What the agent's own tooling needs to reproduce the verifier's grade.

        The instruction tells the agent to run the task's hillclimb runner and
        iterate on what it reports. That runner is the same one the verifier
        runs -- but the agent runs it HERE, in the platform container, not in the
        trial image, so none of the trial image's ENV reaches it:

          * `GAMEBENCH_TASK` / `CANDIDATE_SUBDIR` are baked into the trial image
            and the runner refuses without them.
          * the candidate sandbox needs the container->host path map, or it
            degrades to an unconfined subprocess and the sweep refuses to score.

        Without these the agent cannot measure a candidate at all, and the lane
        silently stops being a hillclimb: it grades whatever was guessed blind.
        """

        environment = {k: str(v) for k, v in dict(trial.environment).items()}
        if trial.candidate_sandbox_docker:
            target = rollout_dir / "workspace"
            environment["GAMEBENCH_POLICY_SANDBOX_HOST_MAP"] = (
                f"{target}:{workspace.to_host(target)}"
            )
        return environment

    def _verify(
        self,
        trial: TrialImage,
        rollout_dir: Path,
        workspace: HostWorkspace,
        rollout_id: str,
        log: RolloutEventLog,
    ) -> tuple[float | None, dict[str, Any]]:
        """Score the mutated workspace in a second sibling. Its number is the reward."""

        target = rollout_dir / "workspace"
        host_target = workspace.to_host(target)
        verifier_image = trial.verifier_image()
        mounts = {str(host_target): trial.workspace_mount}
        # Reward and result paths are read back from wherever the grader writes.
        artifact_root = target
        if trial.logs_mount:
            artifact_root = rollout_dir / "logs"
            if artifact_root.exists():
                shutil.rmtree(artifact_root)
            artifact_root.mkdir(parents=True)
            mounts[str(workspace.to_host(artifact_root))] = trial.logs_mount
        environment = {
            **dict(trial.environment),
            "GAMEBENCH_WORKSPACE_ROOT": trial.workspace_mount,
            # When a logs mount exists the grader writes THERE, outside the
            # workspace the agent could write to. Without it the reward file
            # sits in agent-writable space and can be planted before grading.
            "HARBOR_LOG_DIR": (
                f"{trial.logs_mount}/verifier"
                if trial.logs_mount
                else f"{trial.workspace_mount}/logs/verifier"
            ),
        }
        if trial.candidate_sandbox_docker:
            socket_path = os.environ.get("SYNTH_DOCKER_SOCKET", "/var/run/docker.sock")
            mounts[socket_path] = "/var/run/docker.sock"
            # The grader creates its sandbox under the workspace mount; the host
            # daemon only understands the host side of that bind.
            environment["GAMEBENCH_POLICY_SANDBOX_HOST_MAP"] = (
                f"{trial.workspace_mount}:{host_target}"
            )
            # Candidate containers carry the platform's labels so `down` reaps
            # them with the same filter it uses for trials.
            environment["SYNTH_PARENT"] = platform_id()
            environment["SYNTH_ROLLOUT"] = rollout_id
        # Collection runs in the verifier, before the grader: a benchmark whose
        # grader reads an artifact (DeepSWE grades `model.patch`, not the tree)
        # scores the base state instead of the agent's work without it.
        for index, command in enumerate(trial.collect_commands):
            collected = NestedTrial(
                image=verifier_image,
                rollout_id=rollout_id,
                command=command,
                mounts=dict(mounts),
                environment=environment,
                require_pinned=not trial.allow_unpinned,
                timeout_seconds=trial.verify_timeout_seconds,
                allow_nonzero=True,
            ).run()
            log.append(
                "nested.collected",
                {"step": index, "exit_code": collected.exit_code, "container": collected.name},
            )
        verify_mounts = (
            dict(mounts)
            if trial.verify_mounts_workspace
            else {k: v for k, v in mounts.items() if v != trial.workspace_mount}
        )
        verify_environment = dict(environment)
        staged: list[str] = []
        if trial.candidate_paths and not trial.verify_mounts_workspace:
            staging = rollout_dir / "candidate"
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)
            for relative in trial.candidate_paths:
                source = target / relative
                if not source.exists():
                    # An absent candidate is the agent producing nothing, which
                    # the grader must see as such. Staging silently in that case
                    # would be indistinguishable from staging a real artifact.
                    log.append(
                        "nested.candidate.missing",
                        {"path": relative, "workspace": str(trial.workspace_mount)},
                    )
                    continue
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if source.is_dir():
                    shutil.copytree(source, destination)
                else:
                    shutil.copy2(source, destination)
                staged.append(relative)
            staged_host = workspace.to_host(staging)
            verify_mounts[str(staged_host)] = trial.workspace_mount
            if trial.candidate_sandbox_docker:
                # The grader's throwaway candidate containers bind the host side
                # of ITS workspace mount. Left pointing at the agent's tree, the
                # staging would be cosmetic: the sandbox would mount the very
                # directory the staging exists to keep out.
                verify_environment["GAMEBENCH_POLICY_SANDBOX_HOST_MAP"] = (
                    f"{trial.workspace_mount}:{staged_host}"
                )
            log.append(
                "nested.candidate.staged",
                {"paths": staged, "mounted_at": trial.workspace_mount},
            )
        log.append(
            "span.verifier.opened",
            {
                "role": "verifier",
                "image": verifier_image,
                "separate_verifier": bool(trial.verify_image),
                "isolation_mechanism": _isolation_mechanism(trial),
            },
        )
        result = NestedTrial(
            image=verifier_image,
            rollout_id=rollout_id,
            command=trial.verify_command,
            mounts=verify_mounts,
            environment=verify_environment,
            require_pinned=not trial.allow_unpinned,
            timeout_seconds=trial.verify_timeout_seconds,
            allow_nonzero=True,
        ).run()
        log.append(
            "nested.verified",
            {
                "container": result.name,
                "exit_code": result.exit_code,
                "image": verifier_image,
                "separate_verifier": bool(trial.verify_image),
                # A receipt that does not name the mechanism lets a report claim
                # isolation the run never had. This describes what was actually
                # configured for THIS verification, not what was intended.
                "isolation_mechanism": _isolation_mechanism(trial),
                "verifier_network": "none",
                "workspace_mounted_into_verifier": bool(trial.verify_mounts_workspace),
                "reward_outside_agent_workspace": bool(trial.logs_mount),
                # What actually crossed. A receipt that says "not the workspace"
                # without saying what DID cross is the same unverifiable claim
                # one level down.
                "staged_candidate_paths": list(staged),
                "candidate_paths_declared": list(trial.candidate_paths),
            },
        )
        log.append(
            "span.verifier.closed",
            {
                "role": "verifier",
                "status": "completed" if result.exit_code == 0 else "failed",
                "exit_code": result.exit_code,
            },
        )
        payload = _read_json(artifact_root / trial.result_path)
        # A verifier that failed did not grade. Reward files live in the
        # workspace the agent just wrote to, so trusting one written by a run
        # that exited non-zero means trusting whatever was already on disk --
        # including a value the agent planted before the grader ever ran.
        if result.exit_code != 0 and not trial.reward_on_nonzero_exit:
            return None, {
                "exit_code": result.exit_code,
                "reward_status": "verifier_failed",
                **payload,
            }
        reward: float | None = None
        if trial.reward_json_path:
            graded = _read_json(artifact_root / trial.reward_json_path)
            payload = {**graded, **payload}
            value = graded.get(trial.reward_key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                reward = float(value)
        if reward is None:
            reward = _read_float(artifact_root / trial.reward_path)
        return reward, {"exit_code": result.exit_code, **payload}


@contextlib.contextmanager
def _environment(overrides: Mapping[str, str]) -> Iterator[None]:
    """Apply `overrides` to this process for the duration of the block.

    The agent harness runs as a subprocess that inherits this environment, so
    this is how a trial's identity reaches it. Restored afterwards -- one
    rollout must not leak its task id into the next one on the same platform.
    """

    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _instruction(workspace: Path) -> str:
    for name in ("instruction.md", "AGENTS.md", "README.md"):
        path = workspace / name
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")[:20000]
    return f"Workspace at {workspace}. No instruction file was baked into the trial image."


def _read_float(path: Path) -> float | None:
    if not path.is_file():
        return None
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _isolation_mechanism(trial: "TrialImage") -> str:
    """The isolation this verification actually runs under.

    `host_docker_sibling` means the grader was handed the host daemon so it can
    run each candidate as a throwaway sibling. That is stronger than an
    unconfined subprocess and weaker than a locked-down sandbox, and the receipt
    has to say which one it was.
    """

    if getattr(trial, "candidate_sandbox_docker", False):
        return "sibling_container+host_docker_socket"
    return "sibling_container"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def trial_images_from_env(variable: str = "SYNTH_TRIAL_IMAGES") -> dict[str, TrialImage]:
    """Rollout images named by the operator, as ``id=image@sha256:…`` pairs.

    The catalog is the normal home for these. This is the escape hatch for a
    freshly baked digest that has no catalog row yet.
    """

    raw = os.environ.get(variable, "").strip()
    images: dict[str, TrialImage] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        images[key.strip()] = TrialImage(id=key.strip(), image=value.strip())
    return images
