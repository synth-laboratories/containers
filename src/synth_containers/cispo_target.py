"""Installing a real runtime behind the two declared CISPO ports.

``cispo_contract`` declares seventeen routes and two ports; ``cispo_handshake``,
``cispo_probe``, ``cispo_policy``, ``cispo_rollout``, ``cispo_evidence`` and
``cispo_reward`` implement the behavior behind them. Nothing wires the two
together, so a container process advertises the whole surface and answers a
typed 501 on every route but ``capabilities``. This module is that wiring, and
it is the one place in the CISPO stream that is allowed to name a runtime:
which target a deployment installs is a deployment concern, not an algorithm
one.

The runtime installed here is the package's own
:class:`~synth_containers.reference_runtime.CounterRuntime`, behind
:class:`~synth_containers.reference_runtime.ReferenceManagedRuntime`. It is the
cheapest runtime in this repo that can be served *honestly*:

* Of the runtimes this package serves directly it is the one that already
  publishes a :class:`~synth_containers.capabilities.RuntimeCapabilitySurface`,
  and the capability document is assembled from that surface rather than from a
  literal. ``platform.echo_world`` is a bare gym world -- one reset, one step,
  no surface, no metadata, no trace and no terminate/state support -- and
  ``platform.runtime`` is a ``Protocol`` with no capabilities of its own, so
  serving CISPO over either would mean *declaring* five mandatory capabilities
  that nothing publishes: exactly the direction ``_assert_mandatory`` exists to
  refuse.
* It counts steps, so its horizon is a real ``steps`` horizon with a declared
  ``seconds_per_unit`` rather than a wall clock nobody measured.
* It steps deterministically in memory: no sleeping, no network, no spend.
* Its reward is the environment's own outcome, so the container is the reward
  authority in fact and not only in the declaration.

Three seams are injected rather than assumed:

``SamplerTransport``
    One outbound sampling call. :class:`DeterministicSampler` is the stand-in a
    conformance run uses: it reaches no network, spends nothing, and returns
    token ids and per-token logprobs, which is what makes the evidence
    trainable. A transport whose response carries no logprobs is refused rather
    than recorded as ``engine_meta``.

``SamplerReachability``
    Whether a bound origin is up. The package default is fail-closed, so a
    deployment that installs no probe binds nothing.

``clock``
    The lifecycle's monotone clock. Nothing here sleeps.

What this module does *not* do is invent a measure. The reward request may not
carry one: the container reads its own episode outcome, and a caller-supplied
measure is a refusal, because a reward the executor chose is not a
container-authoritative reward.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from .capabilities import RuntimeCapabilitySurface
from .cispo_contract import (
    AgentInstanceDeclaration,
    CispoRuntimeDeclaration,
    DiscoveryDeclaration,
    EvidenceDeclaration,
    HorizonDeclaration,
    LeaseDeclaration,
    LifecycleDeclaration,
    PolicyBindingDeclaration,
    RecoveryDeclaration,
    RendererProfileDeclaration,
    RewardDeclaration,
    TeamDeclaration,
    TopologyDeclaration,
    build_cispo_capability_document,
    cispo_declared_routes,
)
from .cispo_evidence import (
    CispoEvidenceBuilder,
    InferenceCallV1,
    SealedEvidenceV1,
)
from .cispo_handshake import (
    CispoHandshakeAdapter,
    RuntimeFacts,
    TaskRow,
    task_content_digest,
)
from .cispo_policy import (
    CispoPolicyAdapter,
    SamplerBinding,
    SamplerOriginV1,
    SamplerReachability,
    SamplerTransport,
    BoundPolicySession,
)
from .cispo_reward import CispoRewardAuthority
from .cispo_rollout import (
    AttemptRecordV1,
    CispoRolloutAdapter,
    CispoRolloutLifecycle,
)
from .event_log import RolloutEventLog
from .reference_runtime import CounterRuntime, ReferenceManagedRuntime
from .tracing.canonical import utc_now

TARGET_SCHEMA_VERSION = "cispo.target.v1"

#: The evaluation plan the installed target scores under. It is an identity,
#: not a switch: two reads of the same episode may not come from two plans.
DEFAULT_EVALUATION_PLAN_ID = "cispo.container.outcome.v1"

#: Vocabulary base for the container-side renderer below. Kept clear of the
#: declared stop token ids so a rendered token can never read as a stop.
RENDER_VOCAB_BASE = 100_000
RENDER_VOCAB_SIZE = 50_000

#: Declared stop tokens. They are part of the renderer identity, so they are
#: pinned here beside the vocabulary the same renderer emits.
RENDER_STOP_TOKEN_IDS: tuple[int, ...] = (200_002, 199_999)


class CispoTargetError(RuntimeError):
    """The installed target refused. Never degrade one of these to a reward."""


# --------------------------------------------------------------------------- #
# The capability surface installing the target actually adds
# --------------------------------------------------------------------------- #


def cispo_capability_surface(base: RuntimeCapabilitySurface) -> RuntimeCapabilitySurface:
    """The runtime's own surface plus exactly what installing the target supplies.

    Three flags turn on here and nothing else does. The CISPO binding routes
    every model call through a session-scoped sampler origin the container
    proxies, and the record of that call retains the wire objects
    (``proxied_inference``) together with the prompt and generation token ids
    and their per-token behavior logprobs (``token_emission``). Those are
    capabilities of the *target*, not of the counter environment, which is why
    they are added by installing it rather than declared by a runtime that does
    not have them.

    Everything else is read from ``base``: a runtime that publishes no
    ``trace_support``, ``reward_support``, ``state_support`` or
    ``terminate_support`` still cannot serve CISPO after this, and
    ``build_cispo_capability_document`` refuses to assemble a document for it.
    """

    if not isinstance(base, RuntimeCapabilitySurface):
        raise CispoTargetError(
            "a CISPO target installs over a RuntimeCapabilitySurface, got "
            f"{type(base).__name__}"
        )
    emission = replace(
        base.token_emission,
        token_ids=True,
        tokens=True,
        logprobs=True,
        metadata={**dict(base.token_emission.metadata), "source": "cispo_sampler_capture"},
    )
    return replace(
        base,
        token_emission=emission,
        proxied_inference=True,
        metadata={**dict(base.metadata), "cispo_target": TARGET_SCHEMA_VERSION},
    )


# --------------------------------------------------------------------------- #
# The container-side renderer
# --------------------------------------------------------------------------- #


def render_tokens(text: str) -> tuple[int, ...]:
    """Render one string to token ids, deterministically and reproducibly.

    A pinned renderer is an identity, not a version string, so this one is
    content-addressed: the same text renders to the same ids in every process,
    which is what lets the strict-prefix rule hold across a restart. The digest
    of this function's rule is what ``tokenizer_digest`` names below.
    """

    words = text.split()
    if not words:
        words = [""]
    return tuple(
        RENDER_VOCAB_BASE
        + int(hashlib.sha256(word.encode("utf-8")).hexdigest()[:8], 16) % RENDER_VOCAB_SIZE
        for word in words
    )


def render_logprobs(token_ids: Sequence[int]) -> tuple[float, ...]:
    """One finite, non-zero behavior logprob per generated token.

    An importance ratio has no denominator without these, and an identically
    zero vector is refused by ``InferenceCall.validate_for_training``, so the
    values vary with the token rather than being a constant stand-in.
    """

    return tuple(
        round(-0.05 - ((int(token_id) + index) % 97) / 500.0, 6)
        for index, token_id in enumerate(token_ids)
    )


def reference_renderer_profile() -> RendererProfileDeclaration:
    """The pinned identity of the renderer above."""

    rule = json.dumps(
        {
            "vocab_base": RENDER_VOCAB_BASE,
            "vocab_size": RENDER_VOCAB_SIZE,
            "split": "whitespace",
            "hash": "sha256[:8]",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = "sha256:" + hashlib.sha256(rule.encode("utf-8")).hexdigest()
    return RendererProfileDeclaration(
        profile_id="synth_containers.cispo.render.v1",
        package="synth_containers",
        package_version=_package_version(),
        config_digest=digest,
        tokenizer_id="synth_containers.cispo.whitespace.v1",
        tokenizer_digest=digest,
        stop_token_ids=RENDER_STOP_TOKEN_IDS,
        modalities=("text",),
        add_generation_prompt=True,
    )


def _package_version() -> str:
    try:  # pragma: no cover - trivial
        from importlib.metadata import version

        return version("synth-containers")
    except Exception:  # noqa: BLE001 - an uninstalled checkout still has an identity
        return "0.0.0+source"


# --------------------------------------------------------------------------- #
# Discovery: the taskset the runtime actually has
# --------------------------------------------------------------------------- #


def task_rows_for(runtime: Any) -> tuple[TaskRow, ...]:
    """One duplicate-free row per task in the runtime's own catalog.

    The digest is of the row's content, so a task that changes gets a new
    digest and every agreement built on the old one stops matching.
    """

    catalog = runtime.task_catalog()
    topology_ref = _topology_id(runtime)
    rows: list[TaskRow] = []
    seen: set[str] = set()
    for task in getattr(catalog, "tasks", ()) or ():
        task_id = str(getattr(task, "task_id", "") or "")
        if not task_id or task_id in seen:
            continue
        seen.add(task_id)
        payload = {
            "task_id": task_id,
            "task_name": str(getattr(task, "task_name", "") or ""),
            "task_family": str(getattr(task, "task_family", "") or ""),
            "version": str(getattr(task, "version", "") or ""),
            "benchmark": str(getattr(task, "benchmark", "") or ""),
            "metadata": dict(getattr(task, "metadata", {}) or {}),
        }
        rows.append(
            TaskRow(
                task_id=task_id,
                content_digest=task_content_digest(payload),
                topology_ref=topology_ref,
                task_family=payload["task_family"],
            )
        )
    if not rows:
        raise CispoTargetError(
            "the runtime's catalog holds no task; a taskset route that answers with "
            "nothing is not discovery"
        )
    return tuple(rows)


def _topology_id(runtime: Any) -> str:
    return f"{runtime.metadata().runtime_id}.solo.v1"


# --------------------------------------------------------------------------- #
# The declaration
# --------------------------------------------------------------------------- #


def reference_cispo_declaration(
    runtime: ReferenceManagedRuntime,
    *,
    container_image_digest: str,
    max_steps: int = 8,
    seconds_per_step: float = 0.25,
    splits: Sequence[str] = ("train",),
    advertised_concurrency: int = 8,
    lease_ttl_seconds: float = 120.0,
    evaluation_plan_id: str = DEFAULT_EVALUATION_PLAN_ID,
    settlement_window_seconds: float = 0.0,
) -> CispoRuntimeDeclaration:
    """Everything the installed runtime declares that its surface cannot express.

    The horizon is a ``steps`` horizon because the runtime counts steps, and it
    declares its own ``seconds_per_unit`` because a step carries no duration and
    a lease may not be guessed from one. The roster is a single trainable
    instance on a single trainable team, so the topology never claims a
    multi-actor episode the surface does not support.
    """

    metadata = runtime.metadata()
    topology_id = _topology_id(runtime)
    task_info = runtime.task_info()
    return CispoRuntimeDeclaration(
        container_id=str(metadata.runtime_id),
        container_image_digest=container_image_digest,
        renderer_profile=reference_renderer_profile(),
        discovery=DiscoveryDeclaration(
            taskset_id=f"{task_info.task.task_family or task_info.task.task_id}.catalog",
            taskset_version=str(task_info.task.version or "v1"),
            splits=tuple(str(item) for item in splits),
            task_content_digests=True,
            deterministic_lookup=True,
            duplicate_free=True,
        ),
        policy=PolicyBindingDeclaration(
            binding_transport="message_in_capture_out",
            wire_api="chat_completions",
            session_scoped_sampler_origin=True,
            embeds_credentials=False,
            revision_immutable_after_admission=True,
            records_policy_revision=True,
            probe_binding=True,
        ),
        lifecycle=LifecycleDeclaration(
            advertised_concurrency=int(advertised_concurrency),
            supports_idempotency=True,
            exactly_one_terminal_result=True,
            lease=LeaseDeclaration(
                ttl_seconds=float(lease_ttl_seconds),
                renewable=True,
                straggler_grace_seconds=5.0,
            ),
        ),
        evidence=EvidenceDeclaration(
            strict_prefix=True,
            masking=True,
            wire_objects=True,
            # The attempt produces one trace and no out-of-band recording, so
            # evidence is served inline. That is the declared substitute for
            # artifact-by-reference, not a missing capability pretending to be
            # one.
            artifact_by_reference=False,
            tokens_in_tokens_out=False,
        ),
        reward=RewardDeclaration(
            authority="container",
            binds_trace_digest=True,
            quiescence=True,
            horizon_clipping=True,
            channels=("score",),
            evaluation_plan_id=evaluation_plan_id,
            settlement_window_seconds=float(settlement_window_seconds),
            deferred_scoring=False,
        ),
        recovery=RecoveryDeclaration(restart=True, stale_discard=True),
        topology=TopologyDeclaration(
            topology_id=topology_id,
            turn_model="sequential",
            actuation_model="direct_action",
            reward_relation="cooperative",
            agent_instances=(
                AgentInstanceDeclaration(
                    agent_instance_id="instance-0",
                    role_id="actor",
                    policy_type_id="policy-0",
                    team_id="team-0",
                    trainable=True,
                ),
            ),
            teams=(TeamDeclaration(team_id="team-0", trainable=True, minimum_viable_roster=1),),
            horizon=HorizonDeclaration(
                horizon_kind="steps",
                value=float(max_steps),
                seconds_per_unit=float(seconds_per_step),
            ),
            communication_channels=(),
            parameter_groups={"policy-0": "pg-0"},
        ),
        clock_skew_tolerance_seconds=5.0,
    )


# --------------------------------------------------------------------------- #
# The sampler seam
# --------------------------------------------------------------------------- #


class DeterministicSampler:
    """A sampler stand-in that reaches no network and spends nothing.

    It satisfies both injected seams: :class:`SamplerReachability` (it is always
    up, because it is in this process) and :class:`SamplerTransport`. What it
    returns is what makes the evidence trainable -- generation token ids and one
    finite behavior logprob per token -- so a conformance run walks the same
    path a paid provider would without buying anything.

    The action is chosen the way a model would choose it: from the legal list
    carried in the last message, first entry, deterministically. Nothing here
    reads the environment.
    """

    def __init__(self, *, action_index: int = 0) -> None:
        self._action_index = int(action_index)
        self._calls = 0
        self._lock = threading.Lock()

    @property
    def calls(self) -> int:
        return self._calls

    def reachable(self, origin: SamplerOriginV1) -> bool:
        del origin
        return True

    def post(
        self, url: str, *, headers: Mapping[str, str], body: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if "Authorization" not in headers:
            raise CispoTargetError(
                f"sampling call to {url} carries no Authorization header; the container "
                "carries the session credential and never mints one"
            )
        messages = body.get("messages")
        if not isinstance(messages, Sequence) or not messages:
            raise CispoTargetError("a chat_completions sampling call carries messages")
        action = self._choose(str((messages[-1] or {}).get("content") or ""))
        token_ids = list(render_tokens(action))
        logprobs = list(render_logprobs(token_ids))
        with self._lock:
            self._calls += 1
            index = self._calls
        return {
            "id": f"det-{index}",
            "object": "chat.completion",
            "model": str(body.get("model") or ""),
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": action},
                }
            ],
            "token_ids": {"completion": token_ids},
            "logprobs": {"completion": logprobs},
            "usage": {"completion_tokens": len(token_ids)},
        }

    def _choose(self, prompt: str) -> str:
        marker = "valid_actions="
        start = prompt.rfind(marker)
        if start < 0:
            raise CispoTargetError(
                "the rendered prompt names no legal action list; a sampler that guesses "
                "an action is not sampling the environment's own vocabulary"
            )
        raw = prompt[start + len(marker) :].strip().splitlines()[0]
        try:
            actions = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CispoTargetError(f"unreadable legal action list {raw!r}") from exc
        if not isinstance(actions, list) or not actions:
            raise CispoTargetError("the rendered prompt names an empty legal action list")
        return str(actions[min(self._action_index, len(actions) - 1)])


# --------------------------------------------------------------------------- #
# One attempt over the installed runtime
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class AttemptPlan:
    """What one accepted attempt is: a task, a seed, and one bound policy."""

    rollout_id: str
    task_id: str
    seed: int
    binding: SamplerBinding
    handshake_id: str
    agreement_digest: str
    task_digest: str
    group_id: str = ""
    sample_index: int = 0
    run_id: str = ""


@dataclass(frozen=True, slots=True)
class AttemptResult:
    """What the attempt produced: sealed evidence and the environment's measure."""

    rollout_id: str
    evidence: SealedEvidenceV1
    measure: float
    passed: bool
    steps: int
    snapshot: dict[str, Any] = field(default_factory=dict)


class CounterAttemptRuntime:
    """The :class:`~synth_containers.cispo_rollout.RolloutRuntime` seam.

    ``start`` runs one counter episode to completion in memory: it renders the
    observation, samples through the bound policy session, steps the
    environment, and records one immutable inference call per turn. Nothing here
    sleeps and nothing here opens a socket -- the transport does, and it is
    injected.

    ``poll`` reports the episode's own status. It never finalizes on its own:
    the attempt is terminal when the horizon is attested, which is the
    ``finalize`` route's job, and conflating the two is how a reward comes to
    describe whatever was still running afterwards.
    """

    def __init__(
        self,
        *,
        runtime_factory: Callable[[], CounterRuntime],
        transport: SamplerTransport,
        max_steps: int,
        topology_instance_id: str,
        topology_team_id: str,
        role_id: str,
        policy_type_id: str,
    ) -> None:
        self._runtime_factory = runtime_factory
        self._transport = transport
        self._max_steps = int(max_steps)
        self._instance_id = topology_instance_id
        self._team_id = topology_team_id
        self._role_id = role_id
        self._policy_type_id = policy_type_id
        self._lock = threading.RLock()
        self._plans: dict[str, AttemptPlan] = {}
        self._results: dict[str, AttemptResult] = {}
        self._cancelled: dict[str, str] = {}

    # -- plans ------------------------------------------------------------ #

    def register(self, plan: AttemptPlan) -> AttemptPlan:
        with self._lock:
            self._plans[plan.rollout_id] = plan
        return plan

    def plan(self, rollout_id: str) -> AttemptPlan:
        with self._lock:
            found = self._plans.get(rollout_id)
        if found is None:
            raise CispoTargetError(f"attempt {rollout_id!r} has no admitted plan")
        return found

    def result(self, rollout_id: str) -> AttemptResult:
        with self._lock:
            found = self._results.get(rollout_id)
        if found is None:
            raise CispoTargetError(
                f"attempt {rollout_id!r} sealed no evidence; missing evidence is a terminal "
                "failure, not a zero-reward trajectory"
            )
        return found

    def evidence(self, rollout_id: str) -> SealedEvidenceV1:
        return self.result(rollout_id).evidence

    # -- the RolloutRuntime protocol -------------------------------------- #

    def start(self, attempt: AttemptRecordV1, log: RolloutEventLog) -> None:
        plan = self.plan(attempt.rollout_id)
        result = self._run_episode(plan, log)
        with self._lock:
            self._results[attempt.rollout_id] = result

    def poll(self, attempt: AttemptRecordV1, log: RolloutEventLog) -> str | None:
        del log
        with self._lock:
            if attempt.rollout_id in self._cancelled:
                return "cancelled"
            finished = attempt.rollout_id in self._results
        return "completed" if finished else None

    def quiesce(self, attempt: AttemptRecordV1) -> tuple[str, ...]:
        """Nothing policy-authored outlives the episode: it ran in this call stack.

        The lifecycle has already stopped every process the attempt adopted, so
        an empty residual list here is a reading rather than an assumption.
        """

        del attempt
        return ()

    def horizon_snapshot(self, attempt: AttemptRecordV1) -> Mapping[str, Any]:
        with self._lock:
            found = self._results.get(attempt.rollout_id)
        if found is None:
            return {"rollout_id": attempt.rollout_id, "steps": 0, "state": "unstarted"}
        return dict(found.snapshot)

    def cancel(self, attempt: AttemptRecordV1, reason: str) -> None:
        with self._lock:
            self._cancelled[attempt.rollout_id] = str(reason)

    # -- the episode ------------------------------------------------------ #

    def _run_episode(self, plan: AttemptPlan, log: RolloutEventLog) -> AttemptResult:
        binding = plan.binding
        origin = binding.origin
        if origin is None:  # pragma: no cover - a trainable binding always has one
            raise CispoTargetError(
                f"attempt {plan.rollout_id!r} is bound to a policy with no sampler origin"
            )
        session = BoundPolicySession(binding, transport=self._transport)
        env = self._runtime_factory()
        actions = _legal_actions(env)
        observation = env.reset(seed=plan.seed)

        builder = CispoEvidenceBuilder(
            rollout_id=plan.rollout_id,
            task_id=plan.task_id,
            binding=binding.behavior_binding(),
            seed=plan.seed,
            group_id=plan.group_id,
            sample_index=plan.sample_index,
        )
        system = (
            "You act in a counting environment. Answer with exactly one legal action name."
        )
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        rendered: list[int] = list(render_tokens(system))
        steps = 0
        while steps < self._max_steps and not bool(observation.channels.get("done")):
            user = _observation_prompt(observation, actions)
            messages.append({"role": "user", "content": user})
            rendered.extend(render_tokens(user))
            prompt_tokens = tuple(rendered)

            payload = {"messages": list(messages), "max_tokens": 32}
            wire_request = session.request(payload)
            record, response = session.call(payload)
            action, generation, logprobs = _read_completion(response)

            call = InferenceCallV1(
                call_id=f"{plan.rollout_id}:call:{steps}",
                proxy_request_id=record.proxy_request_id,
                rollout_id=plan.rollout_id,
                group_id=plan.group_id,
                sample_index=plan.sample_index,
                behavior_fingerprint=binding.behavior_fingerprint,
                policy_revision=int(binding.policy_revision),
                wire_api=binding.wire_api,
                sampling_transport=binding.sampling_transport,
                token_capture_provenance="engine_meta",
                prompt_token_ids=prompt_tokens,
                generation_token_ids=generation,
                generation_logprobs=logprobs,
                sampled_mask=tuple(1 for _ in generation),
                finish_reason="stop_token",
                stop_token_ids=tuple(binding.renderer_profile.stop_token_ids),
                renderer_profile_fingerprint=binding.renderer_profile.fingerprint,
                agent_instance_id=binding.agent_instance_id or self._instance_id,
                team_id=binding.team_id or self._team_id,
                role_id=self._role_id,
                policy_type_id=self._policy_type_id,
                parameter_group_id=binding.parameter_group_id,
                effect_tick_start=steps,
                effect_tick_end=steps + 1,
                author_kind="policy",
                wire_request=dict(wire_request),
                wire_response=dict(response),
                usage={
                    "prompt_tokens": len(prompt_tokens),
                    "completion_tokens": len(generation),
                },
                created_at=utc_now(),
            )
            builder.observe(call)

            messages.append({"role": "assistant", "content": action})
            rendered.extend(generation)
            observation = env.step(action, actor_id=self._instance_id)
            steps += 1
            log.append(
                "rollout.attempt.step",
                {
                    "rollout_id": plan.rollout_id,
                    "step_index": steps,
                    "action": action,
                    "reward": observation.channels.get("reward"),
                    "done": bool(observation.channels.get("done")),
                },
            )

        evidence = builder.seal(
            terminal_status="completed",
            correlation={
                "run_id": plan.run_id,
                "group_id": plan.group_id,
                "sample_index": plan.sample_index,
                "seed": plan.seed,
                "handshake_id": plan.handshake_id,
                "agreement_digest": plan.agreement_digest,
                "task_digest": plan.task_digest,
            },
        )
        # The container refuses its own invalid evidence here rather than letting
        # a reward describe it later.
        evidence.validate()
        outcome = env.outcome()
        state = env.read_state()
        return AttemptResult(
            rollout_id=plan.rollout_id,
            evidence=evidence,
            measure=float(outcome.reward or 0.0),
            passed=bool(outcome.passed),
            steps=steps,
            snapshot={
                "rollout_id": plan.rollout_id,
                "state_id": state.state_id,
                "values": dict(state.values),
                "steps": steps,
            },
        )


def _legal_actions(env: CounterRuntime) -> tuple[str, ...]:
    """The environment's own action vocabulary, from its declared tool schema."""

    for tool in env.tools():
        schema = dict(tool.input_schema or {})
        properties = dict(schema.get("properties") or {})
        enum = (properties.get("action") or {}).get("enum")
        if isinstance(enum, Sequence) and enum:
            return tuple(str(item) for item in enum)
    raise CispoTargetError(
        "the runtime declares no action vocabulary; a policy may not be handed a legal "
        "list the environment never published"
    )


def _observation_prompt(observation: Any, actions: Sequence[str]) -> str:
    return (
        f"observation: {observation.content}\n"
        f"valid_actions={json.dumps(list(actions))}"
    )


def _read_completion(response: Mapping[str, Any]) -> tuple[str, tuple[int, ...], tuple[float, ...]]:
    """Lift one sampler response into text, token ids and behavior logprobs.

    A response that carries no token ids or no logprobs is refused: recording it
    as ``engine_meta`` would produce a call that enters a batch and trains on an
    importance ratio with no denominator.
    """

    choices = response.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        raise CispoTargetError("sampler response carries no choice")
    message = dict((choices[0] or {}).get("message") or {})
    text = str(message.get("content") or "").strip()
    if not text:
        raise CispoTargetError("sampler response carries no generated text")
    token_ids = tuple(
        int(item) for item in ((response.get("token_ids") or {}).get("completion") or ())
    )
    logprobs = tuple(
        float(item) for item in ((response.get("logprobs") or {}).get("completion") or ())
    )
    if not token_ids:
        raise CispoTargetError(
            "sampler response carries no generation token ids; there is nothing to train on"
        )
    if len(logprobs) != len(token_ids):
        raise CispoTargetError(
            f"sampler returned {len(logprobs)} logprobs for {len(token_ids)} tokens; an "
            "importance ratio has no denominator"
        )
    return text, token_ids, logprobs


# --------------------------------------------------------------------------- #
# The rollout port
# --------------------------------------------------------------------------- #


class CispoTargetRolloutPort(CispoRolloutAdapter):
    """The rollout port with a real attempt behind it.

    Two things are added to the base adapter and nothing else is changed.

    * **Submission admits and starts.** An attempt is gated against its
      handshake and its policy binding before the lifecycle ever sees it, and
      the episode is started as part of accepting it, because a container with
      no worker behind ``/rollout`` accepts attempts that never run.
    * **The reward is the container's own.** ``cispo_reward`` reads the measure
      from the episode outcome. A request that carries a measure is a refusal:
      a reward the executor chose is not a container-authoritative reward.
    """

    def __init__(self, target: "CispoReferenceTarget") -> None:
        super().__init__(
            target.lifecycle,
            evidence_source=target.attempts.evidence,
            reward_source=lambda: target.reward_authority,
            artifact_source=target.artifacts,
        )
        self._target = target

    def cispo_submit_rollout(self, request: Mapping[str, Any]) -> dict[str, Any]:
        plan = self._target.admit_attempt(request)
        payload = super().cispo_submit_rollout(request)
        if not payload.get("duplicate"):
            record = self._lifecycle.start(plan.rollout_id)
            payload["state"] = record.state.value
        payload["config_id"] = plan.binding.binding_id
        payload["agent_instance_id"] = plan.binding.agent_instance_id
        payload["team_id"] = plan.binding.team_id
        payload["task_id"] = plan.task_id
        payload["task_content_digest"] = plan.task_digest
        return payload

    def cispo_reward(self, request: Mapping[str, Any]) -> dict[str, Any]:
        for name in ("measure", "measures"):
            if request.get(name) is not None:
                raise CispoTargetError(
                    f"reward request carries {name!r}; this container is the reward "
                    "authority and the executor never scores locally"
                )
        rollout_id = str(request["rollout_id"])
        result = self._target.attempts.result(rollout_id)
        offered = str(request.get("trace_digest") or "").strip()
        if offered and offered != result.evidence.trace_digest:
            raise CispoTargetError(
                f"reward asked to score trace {offered} but attempt {rollout_id} sealed "
                f"{result.evidence.trace_digest}"
            )
        plan = self._target.attempts.plan(rollout_id)
        merged = dict(request)
        merged["trace_digest"] = result.evidence.trace_digest
        merged["measures"] = {plan.binding.team_id or "team-0": result.measure}
        merged["optimized_team_id"] = plan.binding.team_id or "team-0"
        merged["metadata"] = {
            **dict(request.get("metadata") or {}),
            "task_id": plan.task_id,
            "task_content_digest": plan.task_digest,
            "steps": result.steps,
            "passed": result.passed,
            "policy_revision": int(plan.binding.policy_revision),
            "behavior_identity": plan.binding.behavior_identity,
        }
        return super().cispo_reward(merged)


# --------------------------------------------------------------------------- #
# The target
# --------------------------------------------------------------------------- #


class CispoReferenceTarget:
    """The reference counter runtime, installed behind both declared ports.

    The capability document is assembled once, from the declaration and the
    installed surface, and the handshake facts are derived from *that document*
    rather than reassembled -- so the hash the handshake echoes is byte for byte
    the hash the capabilities route serves.
    """

    def __init__(
        self,
        runtime: ReferenceManagedRuntime,
        *,
        declaration: CispoRuntimeDeclaration,
        surface: RuntimeCapabilitySurface | None = None,
        routes: Mapping[str, str] | None = None,
        transport: SamplerTransport | None = None,
        reachability: SamplerReachability | None = None,
        clock: Callable[[], float] | None = None,
        handshake_clock: Callable[[], datetime] | None = None,
        handshake_ttl_seconds: float = 900.0,
        container_version: str = "",
    ) -> None:
        self._runtime = runtime
        self._declaration = declaration
        published = surface if surface is not None else runtime.metadata().capabilities
        self._routes = dict(routes or cispo_declared_routes())
        self._document = build_cispo_capability_document(
            declaration, published, routes=self._routes
        )
        self._facts = RuntimeFacts.from_capability_document(
            self._document,
            task_rows=task_rows_for(runtime),
            handshake_ttl_seconds=float(handshake_ttl_seconds),
            container_version=container_version or _package_version(),
        )

        sampler = transport if transport is not None else DeterministicSampler()
        probe = reachability
        if probe is None:
            probe = sampler if isinstance(sampler, SamplerReachability) else None
        self._sampler = sampler

        self._admission = CispoHandshakeAdapter(self._facts, clock=handshake_clock)
        self._admission.policies = CispoPolicyAdapter(self._facts, reachability=probe)

        instance = declaration.topology.agent_instances[0]
        self._attempts = CounterAttemptRuntime(
            runtime_factory=lambda: CounterRuntime(target=_target_count(runtime)),
            transport=sampler,
            max_steps=int(declaration.topology.horizon.value),
            topology_instance_id=instance.agent_instance_id,
            topology_team_id=instance.team_id,
            role_id=instance.role_id,
            policy_type_id=instance.policy_type_id,
        )
        self._lifecycle = CispoRolloutLifecycle(
            self._attempts,
            lease_ttl_seconds=declaration.lifecycle.lease.ttl_seconds,
            lease_renewable=declaration.lifecycle.lease.renewable,
            max_concurrency=declaration.lifecycle.advertised_concurrency,
            clock=clock or time.monotonic,
        )
        self._reward = CispoRewardAuthority(
            evaluation_plan_id=declaration.reward.evaluation_plan_id,
            reward_relation=declaration.topology.reward_relation,
            deferred_scoring=declaration.reward.deferred_scoring,
            settlement_window_seconds=declaration.reward.settlement_window_seconds,
        )
        self._rollouts = CispoTargetRolloutPort(self)

    # -- installation ----------------------------------------------------- #

    @classmethod
    def install(
        cls,
        runtime: ReferenceManagedRuntime,
        *,
        container_image_digest: str = "sha256:reference-counter",
        declaration: CispoRuntimeDeclaration | None = None,
        **kwargs: Any,
    ) -> "CispoReferenceTarget":
        """Install this target behind ``runtime``'s two CISPO ports.

        Installing publishes the surface the target itself supplies -- proxied
        inference and token capture -- on the runtime's metadata, because that
        is what actually changes when the target is installed. Everything else
        on the surface is the runtime's own, so a runtime that cannot trace,
        score, read state or terminate still refuses to serve a document.
        """

        metadata = runtime.metadata()
        surface = cispo_capability_surface(metadata.capabilities)
        metadata.capabilities = surface
        target = cls(
            runtime,
            declaration=declaration
            or reference_cispo_declaration(
                runtime, container_image_digest=container_image_digest
            ),
            surface=surface,
            **kwargs,
        )
        runtime.install_cispo_target(target)
        return target

    # -- what the runtime's ports return ---------------------------------- #

    @property
    def declaration(self) -> CispoRuntimeDeclaration:
        return self._declaration

    @property
    def admission(self) -> CispoHandshakeAdapter:
        return self._admission

    @property
    def rollouts(self) -> CispoTargetRolloutPort:
        return self._rollouts

    # -- the pieces behind them ------------------------------------------- #

    @property
    def facts(self) -> RuntimeFacts:
        return self._facts

    @property
    def capability_document(self) -> dict[str, Any]:
        return dict(self._document)

    @property
    def capability_hash(self) -> str:
        return str(self._document["capability_hash"])

    @property
    def lifecycle(self) -> CispoRolloutLifecycle:
        return self._lifecycle

    @property
    def attempts(self) -> CounterAttemptRuntime:
        return self._attempts

    @property
    def reward_authority(self) -> CispoRewardAuthority:
        return self._reward

    @property
    def sampler(self) -> SamplerTransport:
        return self._sampler

    @property
    def policy_registry(self) -> Any:
        return self._admission.policies.registry

    # -- admission -------------------------------------------------------- #

    def admit_attempt(self, request: Mapping[str, Any]) -> AttemptPlan:
        """Gate one submission against its handshake, its task, and its binding.

        Every one of the three is a refusal rather than a default: an attempt
        with no admitted agreement, on a task the agreement never named, or on a
        binding whose sampler was never reachable, is refused before the
        lifecycle accepts it.
        """

        agreement = self._admission.admit_attempt(request)
        rollout_id = str(request.get("rollout_id") or "").strip()
        if not rollout_id:
            raise CispoTargetError("a submission names no rollout_id")
        task_id = str(
            request.get("task_id")
            or (request.get("task") or {}).get("task_id")
            or ""
        ).strip()
        if not task_id:
            raise CispoTargetError(
                f"attempt {rollout_id!r} names no task; a submission that picks its own "
                "task is not the one the agreement resolved"
            )
        digest = agreement.task_digest(task_id)

        config_id = str(
            request.get("config_id") or request.get("policy_config_id") or ""
        ).strip()
        if not config_id:
            raise CispoTargetError(
                f"attempt {rollout_id!r} names no bound policy config; an unbound attempt "
                "has nothing to collect behavior logprobs from"
            )
        binding = self.policy_registry.admit(rollout_id, config_id)

        correlation = dict(request.get("correlation") or {})
        seed = correlation.get("seed", request.get("seed"))
        return self._attempts.register(
            AttemptPlan(
                rollout_id=rollout_id,
                task_id=task_id,
                seed=int(seed) if isinstance(seed, int) else 0,
                binding=binding,
                handshake_id=agreement.handshake_id,
                agreement_digest=agreement.agreement_digest,
                task_digest=digest,
                group_id=str(correlation.get("group_id") or ""),
                sample_index=int(correlation.get("sample_index") or 0),
                run_id=str(correlation.get("run_id") or agreement.run_id or ""),
            )
        )

    # -- artifacts -------------------------------------------------------- #

    def artifacts(self, rollout_id: str) -> list[dict[str, Any]]:
        """The attempt's inventory: the sealed trace, served inline.

        The container declares no artifact-by-reference, so the one artifact an
        attempt produces is named with its content digest and pointed at the
        declared trace route rather than at a fetch handle it does not serve.
        """

        evidence = self._attempts.evidence(rollout_id)
        payload = evidence.document.to_dict()
        return [
            {
                "artifact_id": f"{rollout_id}:trace",
                "role": "trace",
                "media_type": "application/json",
                "content_digest": evidence.trace_digest,
                "size_bytes": len(
                    json.dumps(payload, sort_keys=True, separators=(",", ":"))
                ),
                "inline": True,
                "route": self._routes.get("trace_route", ""),
            }
        ]


def _target_count(runtime: ReferenceManagedRuntime) -> int:
    """The counter target the runtime already publishes in its own task limits."""

    limits = dict(runtime.task_info().limits or {})
    value = limits.get("target")
    if not isinstance(value, int) or value < 1:
        raise CispoTargetError(
            "the runtime publishes no positive counter target; the episode has no "
            "objective to score against"
        )
    return int(value)


__all__ = [
    "DEFAULT_EVALUATION_PLAN_ID",
    "RENDER_STOP_TOKEN_IDS",
    "RENDER_VOCAB_BASE",
    "RENDER_VOCAB_SIZE",
    "TARGET_SCHEMA_VERSION",
    "AttemptPlan",
    "AttemptResult",
    "CispoReferenceTarget",
    "CispoTargetError",
    "CispoTargetRolloutPort",
    "CounterAttemptRuntime",
    "DeterministicSampler",
    "cispo_capability_surface",
    "reference_cispo_declaration",
    "reference_renderer_profile",
    "render_logprobs",
    "render_tokens",
    "task_rows_for",
]
