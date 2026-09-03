"""The ``probe`` policy-binding kind: claims turned into evidence, at zero cost.

A probe attempt walks the whole path the executor will depend on -- submit,
state, events, lease renewal, trace, reward, finalize, terminate, an idempotent
resubmit of the same key, and one cancellation -- while the bound policy
returns deterministic canned generations instead of calling a provider. Nothing
here sleeps, nothing here spends, and nothing here is negotiable after the
handshake: a probe binding is refused unless it names an admitted agreement.

**Probe evidence must be distinguishable from real evidence.** The optimizer
fails conformance for a container whose probe evidence could be mistaken for the
real thing, so the marking here is structural rather than cosmetic --
:data:`PROBE_MARKERS` names each mark and :func:`assert_probe_distinguishable`
refuses to emit an attempt that is missing one:

===================================  ====================================================
Mark                                 Why it is structural
===================================  ====================================================
``token_capture_provenance``         ``probe_synthetic`` is outside ``TRAINABLE_PROVENANCE``,
                                     so ``InferenceCall.validate_for_training`` raises.
``trainable = False`` per call       The training gate refuses the call on its own field.
``episode.probe = True``             ``TrainableEpisode.validate`` raises before a group
                                     or a batch can ever hold it.
``rollout_id`` prefix ``probe_``     Probe attempts occupy a separate id namespace, so a
                                     probe id cannot collide with a real rollout id.
``reward.metadata.probe``            The reward record says it scored a synthetic episode
                                     at zero provider spend.
synthetic wire objects               ``wire_request``/``wire_response`` declare
                                     ``synthetic: true`` and name no provider, so the
                                     persisted semantic record is marked too.
===================================  ====================================================

Two turns are mandatory: prefix consistency is only *provable* across a pair,
and the second turn's prompt here is a byte-for-byte extension of the first
turn's prompt-plus-generation, on one branch, so the strict-prefix rule holds
by construction rather than by assertion.

Payload field names deliberately match the optimizer's record constructors
(``InferenceCall``, ``TrainableSegment``, ``TrainableEpisode``, ``RewardRecord``,
``ProbeAttempt``) so a validator can build its own typed records from
:meth:`ProbeAttempt.to_payload` without a translation layer.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .cispo_handshake import (
    AdmittedHandshake,
    RendererProfileFacts,
    RuntimeFacts,
    canonical_digest,
)
from .serde import JsonDataclassMixin

PROBE_POLICY_KIND = "probe"
PROBE_TOKEN_CAPTURE_PROVENANCE = "probe_synthetic"
PROBE_ROLLOUT_PREFIX = "probe_"
PROBE_REPORT_SCHEMA_VERSION = "cispo.probe_report.v1"
INFERENCE_CALL_SCHEMA_VERSION = "cispo.inference_call.v2"
TRAINABLE_EPISODE_SCHEMA_VERSION = "cispo.trainable_episode.v1"
REWARD_RECORD_SCHEMA_VERSION = "cispo.reward_record.v1"

#: The operations one probe attempt must exercise before real attempts are
#: admitted. Mirrors ``synth_optimizers.rl.probe.REQUIRED_PROBE_OPERATIONS``.
REQUIRED_PROBE_OPERATIONS: frozenset[str] = frozenset(
    {
        "submit",
        "state",
        "events",
        "renew",
        "trace",
        "reward",
        "finalize",
        "terminate",
        "idempotent_resubmit",
        "cancellation",
    }
)

#: Each structural mark that makes probe evidence unmistakable, and nothing that
#: is merely a label a real attempt could also carry.
PROBE_MARKERS: tuple[str, ...] = (
    "call.token_capture_provenance == 'probe_synthetic'",
    "call.trainable is False",
    "episode.probe is True",
    "rollout_id starts with 'probe_'",
    "reward.metadata.probe is True",
    "wire objects declare synthetic: true and no provider",
)

TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled"})
FINISH_REASONS: frozenset[str] = frozenset({"stop_token", "length_cap", "container_abort"})

#: vLLM uses this both for missing evidence and as a lower-bound clamp, so it can
#: never prove a real logprob came back. A probe must not emit it either.
LOGPROB_SENTINEL = -9999.0


class ProbeError(ValueError):
    """A probe attempt did not exercise or shape the evidence path correctly."""

    status_code = 409
    error = "probe_error"

    def payload(self) -> dict[str, Any]:
        return {"error": self.error, "reason": str(self), "status_code": self.status_code}


class ProbeUnsupported(ProbeError):
    """This build declares no ``probe`` policy-binding kind."""

    error = "probe_unsupported"


class ProbeNotDistinguishable(ProbeError):
    """Probe evidence could be mistaken for real evidence. Conformance failure."""

    error = "probe_not_distinguishable"


# --------------------------------------------------------------------------- #
# Deterministic canned generation
# --------------------------------------------------------------------------- #


def _stream(material: Sequence[Any], count: int, *, low: int, high: int) -> tuple[int, ...]:
    """A deterministic integer stream derived from content, not from a table."""

    if count <= 0:
        return ()
    span = high - low
    values: list[int] = []
    block = 0
    while len(values) < count:
        raw = hashlib.sha256(
            (canonical_digest(list(material)) + f":{block}").encode("utf-8")
        ).digest()
        for index in range(0, len(raw), 2):
            if len(values) >= count:
                break
            values.append(low + (int.from_bytes(raw[index : index + 2], "big") % span))
        block += 1
    return tuple(values)


def _logprobs(material: Sequence[Any], count: int) -> tuple[float, ...]:
    """Finite, negative, never the provider sentinel, never identically zero."""

    return tuple(
        -(1 + value) / 100.0 for value in _stream([*material, "logprobs"], count, low=0, high=400)
    )


# --------------------------------------------------------------------------- #
# Behavior identity (mirrors BehaviorFingerprint / SamplingProfile)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ProbeSamplingProfile(JsonDataclassMixin):
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int | None = None
    seed: int | None = None

    @property
    def key(self) -> str:
        return canonical_digest(
            {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "seed": self.seed,
            },
            length=16,
        )


@dataclass(frozen=True, slots=True)
class ProbeBehavior(JsonDataclassMixin):
    """What the probe's tokens were produced by. Groups may not mix these."""

    renderer_profile: RendererProfileFacts
    model_family: str
    model_id: str
    policy_revision: int
    wire_api: str
    sampling_transport: str
    sampling: ProbeSamplingProfile = field(default_factory=ProbeSamplingProfile)

    @property
    def value(self) -> str:
        return canonical_digest(
            {
                "renderer": self.renderer_profile.fingerprint,
                "model_family": self.model_family,
                "model_id": self.model_id,
                "policy_revision": self.policy_revision,
                "wire_api": self.wire_api,
                "sampling_transport": self.sampling_transport,
                "sampling": self.sampling.key,
            },
            length=32,
        )

    @property
    def binding_id(self) -> str:
        """A probe binding is still a binding, and a binding has an id.

        The executor correlates an attempt to what it bound by this id, and it
        makes no exception for the unpaid kind — a probe that returns none
        cannot be submitted against.
        """

        return f"probe_{self.value[:16]}"

    def to_payload(self) -> dict[str, Any]:
        return {
            "config_id": self.binding_id,
            "binding_id": self.binding_id,
            "kind": PROBE_POLICY_KIND,
            "renderer_profile": self.renderer_profile.to_payload(),
            "model_family": self.model_family,
            "model_id": self.model_id,
            "policy_revision": self.policy_revision,
            "wire_api": self.wire_api,
            "sampling_transport": self.sampling_transport,
            "sampling": {
                "temperature": self.sampling.temperature,
                "top_p": self.sampling.top_p,
                "max_tokens": self.sampling.max_tokens,
                "seed": self.sampling.seed,
            },
        }


# --------------------------------------------------------------------------- #
# Evidence records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ProbeCall(JsonDataclassMixin):
    """One canned model call, shaped as the optimizer's ``InferenceCall``."""

    call_id: str
    proxy_request_id: str
    rollout_id: str
    group_id: str
    sample_index: int
    behavior_fingerprint: str
    policy_revision: int
    wire_api: str
    sampling_transport: str
    prompt_token_ids: tuple[int, ...]
    generation_token_ids: tuple[int, ...]
    generation_logprobs: tuple[float, ...]
    sampled_mask: tuple[int, ...]
    stop_token_ids: tuple[int, ...]
    renderer_profile_fingerprint: str
    finish_reason: str = "stop_token"
    branch_id: str = "root"
    token_capture_provenance: str = PROBE_TOKEN_CAPTURE_PROVENANCE
    trainable: bool = False
    created_at: str = ""

    def __post_init__(self) -> None:
        if self.finish_reason not in FINISH_REASONS:
            raise ProbeError(f"unknown finish_reason {self.finish_reason!r}")
        generated = len(self.generation_token_ids)
        if not self.prompt_token_ids:
            raise ProbeError(f"probe call {self.call_id} has no prompt tokens")
        if not generated:
            raise ProbeError(f"probe call {self.call_id} has no generated tokens")
        if len(self.generation_logprobs) != generated:
            raise ProbeError(f"probe call {self.call_id} logprob length mismatch")
        if len(self.sampled_mask) != generated:
            raise ProbeError(f"probe call {self.call_id} sampled mask length mismatch")
        if not self.stop_token_ids:
            raise ProbeError(f"probe call {self.call_id} records no renderer stop token ids")
        for index, value in enumerate(self.generation_logprobs):
            if not math.isfinite(value):
                raise ProbeError(f"probe call {self.call_id} logprob {index} is not finite")
            if value == LOGPROB_SENTINEL:
                raise ProbeError(
                    f"probe call {self.call_id} logprob {index} is the provider sentinel"
                )
        if all(value == 0.0 for value in self.generation_logprobs):
            raise ProbeError(f"probe call {self.call_id} logprobs are identically zero")

    @property
    def full_sequence(self) -> tuple[int, ...]:
        return tuple(self.prompt_token_ids) + tuple(self.generation_token_ids)

    def wire_request(self) -> dict[str, Any]:
        return {
            "synthetic": True,
            "provider": None,
            "probe": True,
            "policy_kind": PROBE_POLICY_KIND,
            "wire_api": self.wire_api,
        }

    def wire_response(self) -> dict[str, Any]:
        return {
            "synthetic": True,
            "provider": None,
            "probe": True,
            "paid": False,
            "finish_reason": self.finish_reason,
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "proxy_request_id": self.proxy_request_id,
            "rollout_id": self.rollout_id,
            "group_id": self.group_id,
            "sample_index": self.sample_index,
            "behavior_fingerprint": self.behavior_fingerprint,
            "policy_revision": self.policy_revision,
            "wire_api": self.wire_api,
            "sampling_transport": self.sampling_transport,
            "token_capture_provenance": self.token_capture_provenance,
            "prompt_token_ids": tuple(self.prompt_token_ids),
            "generation_token_ids": tuple(self.generation_token_ids),
            "generation_logprobs": tuple(self.generation_logprobs),
            "sampled_mask": tuple(self.sampled_mask),
            "finish_reason": self.finish_reason,
            "stop_token_ids": tuple(self.stop_token_ids),
            "renderer_profile_fingerprint": self.renderer_profile_fingerprint,
            "trainable": self.trainable,
            "branch_id": self.branch_id,
            "wire_request": self.wire_request(),
            "wire_response": self.wire_response(),
            "usage": {"provider_requests": 0, "billed": False},
            "created_at": self.created_at,
            "schema_version": INFERENCE_CALL_SCHEMA_VERSION,
        }


@dataclass(frozen=True, slots=True)
class ProbeSegment(JsonDataclassMixin):
    """One stitched trainer sequence, shaped as ``TrainableSegment``."""

    token_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]
    behavior_logprobs: tuple[float, ...]
    call_ids: tuple[str, ...]
    branch_id: str = "root"
    author_kind: str = "policy"
    policy_revision: int | None = None

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise ProbeError("probe segment has no tokens")
        if len(self.loss_mask) != len(self.token_ids):
            raise ProbeError("probe segment loss mask length mismatch")
        if len(self.behavior_logprobs) != len(self.token_ids):
            raise ProbeError("probe segment behavior logprob length mismatch")

    def to_payload(self) -> dict[str, Any]:
        return {
            "token_ids": tuple(self.token_ids),
            "loss_mask": tuple(self.loss_mask),
            "behavior_logprobs": tuple(self.behavior_logprobs),
            "call_ids": tuple(self.call_ids),
            "branch_id": self.branch_id,
            "author_kind": self.author_kind,
            "policy_revision": self.policy_revision,
        }


@dataclass(frozen=True, slots=True)
class ProbeEpisode(JsonDataclassMixin):
    """The training view of the probe attempt -- refused for training by field."""

    rollout_id: str
    task_id: str
    seed: int
    policy_revision: int
    behavior_fingerprint: str
    segments: tuple[ProbeSegment, ...]
    terminal_status: str
    trace_digest: str
    probe: bool = True

    def __post_init__(self) -> None:
        if not self.probe:
            raise ProbeNotDistinguishable(
                f"probe episode {self.rollout_id} is not marked probe; it would be "
                "indistinguishable from real evidence"
            )
        if self.terminal_status not in TERMINAL_STATUSES:
            raise ProbeError(f"unknown terminal status {self.terminal_status!r}")
        if not self.segments:
            raise ProbeError("probe episode carries no segment to check")

    def to_payload(self) -> dict[str, Any]:
        return {
            "rollout_id": self.rollout_id,
            "task_id": self.task_id,
            "seed": self.seed,
            "policy_revision": self.policy_revision,
            "behavior_fingerprint": self.behavior_fingerprint,
            "segments": tuple(segment.to_payload() for segment in self.segments),
            "terminal_status": self.terminal_status,
            "trace_digest": self.trace_digest,
            "probe": True,
            "usage": {"provider_requests": 0, "billed": False},
            "schema_version": TRAINABLE_EPISODE_SCHEMA_VERSION,
        }


@dataclass(frozen=True, slots=True)
class ProbeRewardChannel(JsonDataclassMixin):
    channel_id: str
    team_id: str | None
    measure: float
    rank: int | None = None


@dataclass(frozen=True, slots=True)
class ProbeHorizonEvidence(JsonDataclassMixin):
    """When the reward was read, and whether the environment was still moving."""

    horizon_kind: str
    horizon_value: float
    scored_at_offset_seconds: float
    clipped: bool
    quiescence_attested: bool
    settlement_window_seconds: float = 0.0
    credited_settlement_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not self.quiescence_attested and not self.clipped:
            raise ProbeError(
                "probe reward has neither a quiescence attestation nor a "
                "horizon-clipped snapshot"
            )


@dataclass(frozen=True, slots=True)
class ProbeReward(JsonDataclassMixin):
    """Container-authoritative reward, bound to the rollout and trace digest."""

    reward_id: str
    rollout_id: str
    trace_digest: str
    channels: tuple[ProbeRewardChannel, ...]
    optimized_channel: str
    terminal_status: str
    evaluation_plan_id: str
    horizon: ProbeHorizonEvidence

    def __post_init__(self) -> None:
        if not self.channels:
            raise ProbeError(f"probe reward {self.reward_id} carries no channel")
        if self.optimized_channel not in {item.channel_id for item in self.channels}:
            raise ProbeError(
                f"probe reward {self.reward_id} optimizes a channel it does not carry"
            )
        if self.terminal_status not in TERMINAL_STATUSES:
            raise ProbeError(f"unknown terminal status {self.terminal_status!r}")
        if not self.trace_digest:
            raise ProbeError(f"probe reward {self.reward_id} is not bound to a trace digest")

    def to_payload(self) -> dict[str, Any]:
        return {
            "reward_id": self.reward_id,
            "rollout_id": self.rollout_id,
            "trace_digest": self.trace_digest,
            "channels": tuple(
                {
                    "channel_id": item.channel_id,
                    "team_id": item.team_id,
                    "measure": item.measure,
                    "rank": item.rank,
                }
                for item in self.channels
            ),
            "optimized_channel": self.optimized_channel,
            "terminal_status": self.terminal_status,
            "evaluation_plan_id": self.evaluation_plan_id,
            "horizon": {
                "horizon_kind": self.horizon.horizon_kind,
                "horizon_value": self.horizon.horizon_value,
                "scored_at_offset_seconds": self.horizon.scored_at_offset_seconds,
                "clipped": self.horizon.clipped,
                "quiescence_attested": self.horizon.quiescence_attested,
                "settlement_window_seconds": self.horizon.settlement_window_seconds,
                "credited_settlement_seconds": self.horizon.credited_settlement_seconds,
            },
            "metadata": {"probe": True, "paid": False, "provider_requests": 0},
            "schema_version": REWARD_RECORD_SCHEMA_VERSION,
        }


@dataclass(frozen=True, slots=True)
class ProbeAttempt(JsonDataclassMixin):
    """Everything one probe attempt produced, as the executor will receive it."""

    rollout_id: str
    behavior: ProbeBehavior
    calls: tuple[ProbeCall, ...]
    episode: ProbeEpisode
    reward: ProbeReward
    event_cursors: tuple[int, ...]
    terminal_results: tuple[str, ...]
    operations: frozenset[str]
    resubmit_rollout_id: str
    cancelled_rollout_id: str
    trace_digest: str
    handshake_id: str = ""
    agreement_digest: str = ""

    def to_payload(self) -> dict[str, Any]:
        """Keyed by the optimizer's ``ProbeAttempt`` field names, so a validator
        can build its own typed records from this without a translation layer."""

        return {
            "rollout_id": self.rollout_id,
            "behavior": self.behavior.to_payload(),
            "calls": tuple(call.to_payload() for call in self.calls),
            "episode": self.episode.to_payload(),
            "reward": self.reward.to_payload(),
            "event_cursors": tuple(self.event_cursors),
            "terminal_results": tuple(self.terminal_results),
            "operations": tuple(sorted(self.operations)),
            "resubmit_rollout_id": self.resubmit_rollout_id,
            "cancelled_rollout_id": self.cancelled_rollout_id,
            "trace_digest": self.trace_digest,
            "metadata": {
                "probe": True,
                "policy_kind": PROBE_POLICY_KIND,
                "handshake_id": self.handshake_id,
                "agreement_digest": self.agreement_digest,
                "markers": list(PROBE_MARKERS),
                "provider_requests": 0,
            },
        }


@dataclass(frozen=True, slots=True)
class ProbeReport(JsonDataclassMixin):
    """What the probe proved, for the run receipt. Never a training record."""

    rollout_id: str
    calls_checked: int
    segments_checked: int
    operations: tuple[str, ...]
    renderer_fingerprint: str
    trace_digest: str
    reward_id: str
    quiescence_attested: bool
    trainable: bool = False
    schema_version: str = PROBE_REPORT_SCHEMA_VERSION

    @property
    def evidence_digest(self) -> str:
        return "sha256:" + canonical_digest(
            {
                "rollout_id": self.rollout_id,
                "trace_digest": self.trace_digest,
                "reward_id": self.reward_id,
                "renderer_fingerprint": self.renderer_fingerprint,
                "probe": True,
            }
        )


# --------------------------------------------------------------------------- #
# Distinguishability, checked by the container before it answers
# --------------------------------------------------------------------------- #


def assert_probe_distinguishable(attempt: ProbeAttempt) -> None:
    """Refuse to emit probe evidence a reader could mistake for the real thing.

    The optimizer runs the same check on its side and fails conformance; doing
    it here as well means the container never puts an indistinguishable record
    on the wire in the first place.
    """

    if not attempt.rollout_id.startswith(PROBE_ROLLOUT_PREFIX):
        raise ProbeNotDistinguishable(
            f"probe rollout {attempt.rollout_id!r} does not use the "
            f"{PROBE_ROLLOUT_PREFIX!r} id namespace"
        )
    if not attempt.episode.probe:
        raise ProbeNotDistinguishable("probe episode is not marked probe")
    for call in attempt.calls:
        if call.token_capture_provenance != PROBE_TOKEN_CAPTURE_PROVENANCE:
            raise ProbeNotDistinguishable(
                f"probe call {call.call_id} declares provenance "
                f"{call.token_capture_provenance!r}, not "
                f"{PROBE_TOKEN_CAPTURE_PROVENANCE!r}"
            )
        if call.trainable:
            raise ProbeNotDistinguishable(
                f"probe call {call.call_id} is marked trainable; a probe episode may "
                "never enter a group or a batch"
            )
        if not call.wire_response().get("synthetic"):
            raise ProbeNotDistinguishable(
                f"probe call {call.call_id} persists a wire object that does not "
                "declare itself synthetic"
            )
    if not attempt.reward.to_payload()["metadata"].get("probe"):
        raise ProbeNotDistinguishable("probe reward does not mark itself probe-derived")


def assert_probe_path_exercised(attempt: ProbeAttempt) -> tuple[str, ...]:
    """Every required operation ran, cursors are monotone, one terminal result."""

    missing = tuple(sorted(REQUIRED_PROBE_OPERATIONS - set(attempt.operations)))
    if missing:
        raise ProbeError(f"probe attempt did not exercise {missing}")
    unknown = tuple(sorted(set(attempt.operations) - REQUIRED_PROBE_OPERATIONS))
    if unknown:
        raise ProbeError(f"probe attempt reports unknown operations {unknown}")
    if attempt.resubmit_rollout_id != attempt.rollout_id:
        raise ProbeError(
            "idempotent resubmit produced a second logical attempt: "
            f"{attempt.resubmit_rollout_id!r} != {attempt.rollout_id!r}"
        )
    if not attempt.cancelled_rollout_id.strip():
        raise ProbeError("probe attempt records no cancellation")
    if not attempt.event_cursors:
        raise ProbeError("probe attempt returned no event cursor")
    for previous, following in zip(
        attempt.event_cursors, attempt.event_cursors[1:], strict=False
    ):
        if following <= previous:
            raise ProbeError(
                f"probe event cursor is not monotone: {previous} then {following}"
            )
    if len(attempt.terminal_results) != 1:
        raise ProbeError(
            f"probe attempt produced {len(attempt.terminal_results)} terminal results; "
            "exactly one is allowed"
        )
    return tuple(sorted(attempt.operations))


def assert_strict_prefix(previous: ProbeCall, following: ProbeCall) -> None:
    """Two calls stitch only on a byte-for-byte token prefix, on one branch."""

    sequence = previous.full_sequence
    prompt = tuple(following.prompt_token_ids)
    if prompt[: len(sequence)] != sequence:
        raise ProbeError(
            f"probe call {following.call_id} is not a strict token prefix continuation "
            f"of {previous.call_id}; a probe may not fork or retokenize"
        )
    if following.branch_id != previous.branch_id:
        raise ProbeError(
            f"probe call {following.call_id} is a strict prefix continuation but "
            "changed branch"
        )


# --------------------------------------------------------------------------- #
# The probe policy binding and its session
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ProbeBinding(JsonDataclassMixin):
    """A bound ``probe`` policy, pinned to one admitted agreement.

    The binding carries no credential and reaches no provider: the whole point
    is that the evidence path can be walked before one paid request exists.
    """

    policy_kind: str
    handshake_id: str
    agreement_digest: str
    behavior: ProbeBehavior
    evaluation_plan_id: str
    reward_channels: tuple[str, ...]
    team_id: str | None
    horizon_kind: str
    horizon_value: float
    settlement_window_seconds: float
    quiescence_accepted: bool
    max_prompt_tokens: int = 0

    @property
    def binding_id(self) -> str:
        return f"probe_{self.behavior.value[:16]}"

    def to_payload(self) -> dict[str, Any]:
        return {
            # A probe binding is still a binding, and the executor correlates
            # an attempt to what it bound by this id. The unpaid kind gets no
            # exception: a binding that returns no id cannot be submitted
            # against, and the run fails a layer away from the cause.
            "config_id": self.binding_id,
            "binding_id": self.binding_id,
            "policy_kind": self.policy_kind,
            "kind": self.policy_kind,
            "probe": True,
            "trainable": False,
            "handshake_id": self.handshake_id,
            "agreement_digest": self.agreement_digest,
            "behavior": self.behavior.to_payload(),
            "renderer_profile": self.behavior.renderer_profile.to_payload(),
            "behavior_fingerprint": self.behavior.value,
            "evaluation_plan_id": self.evaluation_plan_id,
            "reward_channels": list(self.reward_channels),
            "quiescence_accepted": self.quiescence_accepted,
            "sampler_origin": None,
            "credential": None,
            "provider_requests_expected": 0,
        }


def bind_probe_policy(
    facts: RuntimeFacts,
    agreement: AdmittedHandshake,
    *,
    model_family: str,
    model_id: str,
    policy_revision: int = 0,
    team_id: str | None = None,
    sampling: ProbeSamplingProfile | None = None,
) -> ProbeBinding:
    """Bind the ``probe`` policy kind, or refuse.

    Refuses when this build declares no probe binding, and refuses when the
    agreement it is asked to run under is not one the container admitted -- the
    handshake gates the probe exactly as it gates a real attempt.
    """

    if not facts.probe_binding:
        raise ProbeUnsupported(
            "container declares no probe policy-binding kind; a single real paid "
            "canary attempt is the declared alternative"
        )
    if not agreement.agreement_digest:
        raise ProbeError("probe binding names no agreement digest")
    if not facts.reward.channels:
        raise ProbeError("container declares no reward channel to score a probe on")
    return ProbeBinding(
        policy_kind=PROBE_POLICY_KIND,
        handshake_id=agreement.handshake_id,
        agreement_digest=agreement.agreement_digest,
        behavior=ProbeBehavior(
            renderer_profile=facts.renderer_profile,
            model_family=model_family,
            model_id=model_id,
            policy_revision=policy_revision,
            wire_api=facts.policy.wire_api,
            sampling_transport=facts.policy.binding_transport,
            sampling=sampling or ProbeSamplingProfile(),
        ),
        evaluation_plan_id=facts.reward.evaluation_plan_id,
        reward_channels=tuple(facts.reward.channels),
        team_id=team_id,
        horizon_kind=facts.horizon.horizon_kind,
        horizon_value=float(facts.horizon.value),
        settlement_window_seconds=float(agreement.obligations.settlement_window_seconds),
        quiescence_accepted=agreement.quiescence_accepted,
    )


class ProbeSession:
    """Walks the probe path, one operation at a time, recording what it ran.

    Operations are recorded as they happen rather than asserted at the end, so
    :meth:`attempt` cannot report a path the session did not actually take.
    """

    def __init__(
        self,
        binding: ProbeBinding,
        *,
        task_id: str,
        seed: int,
        group_id: str = "probe-group",
        sample_index: int = 0,
        idempotency_key: str = "",
        turns: int = 2,
    ) -> None:
        if turns < 2:
            raise ProbeError(
                "a probe must exercise at least two turns so prefix consistency is provable"
            )
        self.binding = binding
        self.task_id = task_id
        self.seed = int(seed)
        self.group_id = group_id
        self.sample_index = int(sample_index)
        self.turns = int(turns)
        self.idempotency_key = idempotency_key or canonical_digest(
            [binding.agreement_digest, task_id, seed, group_id, sample_index], length=16
        )
        self.rollout_id = PROBE_ROLLOUT_PREFIX + canonical_digest(
            [binding.agreement_digest, self.idempotency_key], length=20
        )
        self.cancelled_rollout_id = self.rollout_id + "_cancelled"
        self._operations: set[str] = set()
        self._events: list[dict[str, Any]] = []
        self._cursor = 0
        self._calls: tuple[ProbeCall, ...] = ()
        self._terminal: list[str] = []
        self._trace_digest = ""

    # -- events -------------------------------------------------------- #

    def _emit(self, kind: str, rollout_id: str, **extra: Any) -> dict[str, Any]:
        self._cursor += 1
        event = {"cursor": self._cursor, "kind": kind, "rollout_id": rollout_id, **extra}
        self._events.append(event)
        return event

    @property
    def event_cursors(self) -> tuple[int, ...]:
        return tuple(int(event["cursor"]) for event in self._events)

    @property
    def operations(self) -> frozenset[str]:
        return frozenset(self._operations)

    # -- the path ------------------------------------------------------ #

    def submit(self) -> dict[str, Any]:
        if self._calls:
            raise ProbeError("probe attempt already submitted")
        self._operations.add("submit")
        self._calls = self._generate_calls()
        self._trace_digest = "sha256:" + canonical_digest(
            [call.to_payload() for call in self._calls]
        )
        self._emit("accepted", self.rollout_id, idempotency_key=self.idempotency_key)
        return {
            "rollout_id": self.rollout_id,
            "status": "running",
            "probe": True,
            "handshake_id": self.binding.handshake_id,
            "agreement_digest": self.binding.agreement_digest,
        }

    def resubmit(self) -> dict[str, Any]:
        """The same key yields the same logical attempt, not a second one."""

        self._require("submit")
        self._operations.add("idempotent_resubmit")
        self._emit("resubmit_deduplicated", self.rollout_id, idempotency_key=self.idempotency_key)
        return {"rollout_id": self.rollout_id, "deduplicated": True, "probe": True}

    def state(self) -> dict[str, Any]:
        self._require("submit")
        self._operations.add("state")
        event = self._emit("state", self.rollout_id, status="running")
        return {
            "rollout_id": self.rollout_id,
            "status": "running",
            "probe": True,
            "cursor": event["cursor"],
        }

    def events(self, *, after_cursor: int = 0) -> list[dict[str, Any]]:
        self._require("submit")
        self._operations.add("events")
        return [event for event in self._events if int(event["cursor"]) > after_cursor]

    def renew_lease(self) -> dict[str, Any]:
        self._require("submit")
        self._operations.add("renew")
        event = self._emit("lease.renewed", self.rollout_id)
        return {"rollout_id": self.rollout_id, "renewed": True, "cursor": event["cursor"]}

    def trace(self) -> dict[str, Any]:
        self._require("submit")
        self._operations.add("trace")
        self._emit("trace.sealed", self.rollout_id, trace_digest=self._trace_digest)
        return {
            "rollout_id": self.rollout_id,
            "trace_digest": self._trace_digest,
            "probe": True,
            "calls": [call.to_payload() for call in self._calls],
        }

    def reward(self) -> ProbeReward:
        self._require("trace")
        self._operations.add("reward")
        record = self._build_reward()
        self._emit("reward.scored", self.rollout_id, reward_id=record.reward_id)
        return record

    def finalize(self) -> ProbeEpisode:
        self._require("reward")
        self._operations.add("finalize")
        episode = self._build_episode()
        self._terminal.append("completed")
        self._emit("episode", self.rollout_id, terminal_status="completed")
        return episode

    def terminate(self) -> dict[str, Any]:
        self._require("finalize")
        self._operations.add("terminate")
        self._emit("terminated", self.rollout_id)
        return {"rollout_id": self.rollout_id, "terminated": True}

    def cancel_sibling(self) -> dict[str, Any]:
        """One cancellation, on its own attempt, so the probe keeps one terminal."""

        self._operations.add("cancellation")
        self._emit("cancellation", self.cancelled_rollout_id, terminal_status="cancelled")
        return {"rollout_id": self.cancelled_rollout_id, "status": "cancelled"}

    def _require(self, operation: str) -> None:
        if operation not in self._operations:
            raise ProbeError(f"probe operation ran out of order: {operation!r} has not run")

    # -- evidence ------------------------------------------------------ #

    def _generate_calls(self) -> tuple[ProbeCall, ...]:
        profile = self.binding.behavior.renderer_profile
        stop_token = int(profile.stop_token_ids[0])
        material = [self.binding.agreement_digest, self.rollout_id, self.task_id, self.seed]
        prompt = list(_stream([*material, "prompt"], 6, low=1000, high=60000))
        calls: list[ProbeCall] = []
        for turn in range(1, self.turns + 1):
            generated = list(_stream([*material, "gen", turn], 3, low=1000, high=60000))
            generated.append(stop_token)
            logprobs = _logprobs([*material, "gen", turn], len(generated))
            call = ProbeCall(
                call_id=f"{self.rollout_id}:call-{turn}",
                proxy_request_id=f"{self.rollout_id}:proxy-{turn}",
                rollout_id=self.rollout_id,
                group_id=self.group_id,
                sample_index=self.sample_index,
                behavior_fingerprint=self.binding.behavior.value,
                policy_revision=self.binding.behavior.policy_revision,
                wire_api=self.binding.behavior.wire_api,
                sampling_transport=self.binding.behavior.sampling_transport,
                prompt_token_ids=tuple(prompt),
                generation_token_ids=tuple(generated),
                generation_logprobs=logprobs,
                sampled_mask=tuple(1 for _ in generated),
                stop_token_ids=tuple(profile.stop_token_ids),
                renderer_profile_fingerprint=profile.fingerprint,
            )
            calls.append(call)
            # The next turn's prompt is the previous prompt-plus-generation with
            # a tool observation appended: a byte-for-byte token prefix, so the
            # two turns stitch and prefix consistency is provable.
            observation = list(_stream([*material, "observation", turn], 2, low=1000, high=60000))
            prompt = prompt + generated + observation
        for previous, following in zip(calls, calls[1:], strict=False):
            assert_strict_prefix(previous, following)
        return tuple(calls)

    def _build_episode(self) -> ProbeEpisode:
        last = self._calls[-1]
        token_ids = list(last.full_sequence)
        loss_mask = [0] * len(token_ids)
        logprobs = [0.0] * len(token_ids)
        # Mark only the tokens the policy actually sampled; everything the
        # harness or the environment contributed stays masked. The stitched
        # sequence is the last turn's prompt-plus-generation because every
        # earlier turn is a byte-for-byte prefix of it.
        for call in self._calls:
            start = len(call.prompt_token_ids)
            for index in range(len(call.generation_token_ids)):
                loss_mask[start + index] = int(bool(call.sampled_mask[index]))
                logprobs[start + index] = float(call.generation_logprobs[index])
        return ProbeEpisode(
            rollout_id=self.rollout_id,
            task_id=self.task_id,
            seed=self.seed,
            policy_revision=self.binding.behavior.policy_revision,
            behavior_fingerprint=self.binding.behavior.value,
            segments=(
                ProbeSegment(
                    token_ids=tuple(token_ids),
                    loss_mask=tuple(loss_mask),
                    behavior_logprobs=tuple(logprobs),
                    call_ids=tuple(call.call_id for call in self._calls),
                    policy_revision=self.binding.behavior.policy_revision,
                ),
            ),
            terminal_status="completed",
            trace_digest=self._trace_digest,
        )

    def _build_reward(self) -> ProbeReward:
        binding = self.binding
        channels = tuple(
            ProbeRewardChannel(
                channel_id=channel_id,
                team_id=binding.team_id,
                measure=float(index),
                rank=None,
            )
            for index, channel_id in enumerate(binding.reward_channels)
        )
        return ProbeReward(
            reward_id="probe_rw_" + canonical_digest([self.rollout_id, "reward"], length=16),
            rollout_id=self.rollout_id,
            trace_digest=self._trace_digest,
            channels=channels,
            optimized_channel=channels[0].channel_id,
            terminal_status="completed",
            evaluation_plan_id=binding.evaluation_plan_id,
            horizon=ProbeHorizonEvidence(
                horizon_kind=binding.horizon_kind,
                horizon_value=float(binding.horizon_value),
                scored_at_offset_seconds=0.0,
                # Quiescence and clipping answer the same question two ways, and
                # which one the probe attests is the clause verdict, not a choice.
                clipped=not binding.quiescence_accepted,
                quiescence_attested=binding.quiescence_accepted,
                settlement_window_seconds=float(binding.settlement_window_seconds),
                credited_settlement_seconds=0.0,
            ),
        )

    # -- the whole path ------------------------------------------------ #

    def attempt(self) -> ProbeAttempt:
        """Assemble what ran. Refuses a path that was not actually walked."""

        self._require("terminate")
        episode = self._build_episode()
        reward = self._build_reward()
        attempt = ProbeAttempt(
            rollout_id=self.rollout_id,
            behavior=self.binding.behavior,
            calls=self._calls,
            episode=episode,
            reward=reward,
            event_cursors=self.event_cursors,
            terminal_results=tuple(self._terminal),
            operations=self.operations,
            resubmit_rollout_id=self.rollout_id,
            cancelled_rollout_id=self.cancelled_rollout_id,
            trace_digest=self._trace_digest,
            handshake_id=self.binding.handshake_id,
            agreement_digest=self.binding.agreement_digest,
        )
        assert_probe_path_exercised(attempt)
        assert_probe_distinguishable(attempt)
        return attempt


def run_probe(
    facts: RuntimeFacts,
    agreement: AdmittedHandshake,
    *,
    task_id: str,
    seed: int = 0,
    model_family: str = "",
    model_id: str = "",
    policy_revision: int = 0,
    team_id: str | None = None,
    turns: int = 2,
) -> tuple[ProbeAttempt, ProbeReport]:
    """Bind a probe policy and walk the whole evidence path once, at zero cost."""

    binding = bind_probe_policy(
        facts,
        agreement,
        model_family=model_family or facts.renderer_profile.tokenizer_id,
        model_id=model_id or facts.renderer_profile.tokenizer_id,
        policy_revision=policy_revision,
        team_id=team_id or _first_trainable_team(facts),
    )
    session = ProbeSession(binding, task_id=task_id, seed=seed, turns=turns)
    session.submit()
    session.state()
    session.renew_lease()
    session.trace()
    session.reward()
    session.finalize()
    session.terminate()
    session.resubmit()
    session.cancel_sibling()
    session.events()
    attempt = session.attempt()
    return attempt, probe_report(attempt, facts=facts)


def _first_trainable_team(facts: RuntimeFacts) -> str | None:
    for instance in facts.topology.agent_instances:
        if instance.trainable:
            return instance.team_id
    return None


def probe_report(attempt: ProbeAttempt, *, facts: RuntimeFacts) -> ProbeReport:
    """The receipt line for the probe. Never a training record."""

    return ProbeReport(
        rollout_id=attempt.rollout_id,
        calls_checked=len(attempt.calls),
        segments_checked=len(attempt.episode.segments),
        operations=tuple(sorted(attempt.operations)),
        renderer_fingerprint=facts.renderer_profile.fingerprint,
        trace_digest=attempt.trace_digest,
        reward_id=attempt.reward.reward_id,
        quiescence_attested=bool(attempt.reward.horizon.quiescence_attested),
    )


class CispoProbeAdapter:
    """The probe half of the policy-binding surface, in wire terms.

    Kept to one class for the same reason ``CispoHandshakeAdapter`` is: the
    coupling to ``cispo_contract``'s port is one object wide.
    """

    def __init__(self, facts: RuntimeFacts) -> None:
        self._facts = facts

    def declares_probe_binding(self) -> bool:
        return bool(self._facts.probe_binding)

    def bind(self, agreement: AdmittedHandshake, body: Mapping[str, Any]) -> dict[str, Any]:
        kind = str(body.get("kind") or body.get("policy_kind") or "")
        if kind != PROBE_POLICY_KIND:
            raise ProbeError(f"policy binding kind {kind!r} is not {PROBE_POLICY_KIND!r}")
        binding = bind_probe_policy(
            self._facts,
            agreement,
            model_family=str(body.get("model_family") or ""),
            model_id=str(body.get("model_id") or ""),
            policy_revision=int(body.get("policy_revision") or 0),
            team_id=body.get("team_id"),
        )
        return binding.to_payload()

    def run(self, agreement: AdmittedHandshake, body: Mapping[str, Any]) -> dict[str, Any]:
        attempt, report = run_probe(
            self._facts,
            agreement,
            task_id=str(body.get("task_id") or ""),
            seed=int(body.get("seed") or 0),
            model_family=str(body.get("model_family") or ""),
            model_id=str(body.get("model_id") or ""),
            policy_revision=int(body.get("policy_revision") or 0),
            team_id=body.get("team_id"),
        )
        return {"attempt": attempt.to_payload(), "report": report.to_dict()}


__all__ = [
    "INFERENCE_CALL_SCHEMA_VERSION",
    "LOGPROB_SENTINEL",
    "PROBE_MARKERS",
    "PROBE_POLICY_KIND",
    "PROBE_REPORT_SCHEMA_VERSION",
    "PROBE_ROLLOUT_PREFIX",
    "PROBE_TOKEN_CAPTURE_PROVENANCE",
    "REQUIRED_PROBE_OPERATIONS",
    "REWARD_RECORD_SCHEMA_VERSION",
    "TRAINABLE_EPISODE_SCHEMA_VERSION",
    "CispoProbeAdapter",
    "ProbeAttempt",
    "ProbeBehavior",
    "ProbeBinding",
    "ProbeCall",
    "ProbeEpisode",
    "ProbeError",
    "ProbeHorizonEvidence",
    "ProbeNotDistinguishable",
    "ProbeReport",
    "ProbeReward",
    "ProbeRewardChannel",
    "ProbeSamplingProfile",
    "ProbeSegment",
    "ProbeSession",
    "ProbeUnsupported",
    "assert_probe_distinguishable",
    "assert_probe_path_exercised",
    "assert_strict_prefix",
    "bind_probe_policy",
    "probe_report",
    "run_probe",
]
