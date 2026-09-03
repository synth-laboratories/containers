"""Container-side CISPO trainable evidence, built from the existing Trace V5 capture.

The wire object is the semantic record and the tokens are the training record;
neither substitutes for the other, so both are kept. What this module produces:

* **One immutable record per proxied model call, before any flattening.** Exact
  prompt and generation token ids, per-token behavior logprobs taken from under
  the wire, the sampled mask, the renderer profile fingerprint, the behavior
  policy revision, the finish reason and declared stop tokens, and the original
  wire request/response objects retained beside the tokens.
* **Strict-prefix stitching.** Two calls concatenate into one trainer sequence
  only when the next prompt is a byte-for-byte token prefix of the previous
  prompt-plus-generation. A tool loop stitches. A compaction forks a branch,
  seals the prior segment, and loss-masks the retained prefix. An unexplained
  divergence is an evidence failure, never a silent re-tokenization.
* **Declared foreign authorship.** Another instance's messages, an opponent's
  spans, verifier text, and judge text are emitted as untrainable segments with
  their author named. A zero mask never implies authorship.
* **A sealed Trace V5 document** built from this package's own models and sealed
  with its own canonical digest, so the reward has something to bind to.

The reader of the existing capture is
:func:`inference_call_from_span`: it takes a
:class:`~synth_containers.tracing.models.spans.SpanV5` whose ``token_capture``
is the package's :class:`~synth_containers.tracing.models.tokens.TokenCaptureV5`
and lifts it into a training record. The Trace V5 capture models carry token ids
and completion logprobs but have no field for the sampled mask, the finish
reason, the renderer-profile fingerprint, or the behavior policy revision, so
those are read from ``SpanV5.metadata["cispo"]``, which is the declared
extension point for exactly this.

No task, harness, or environment name appears here.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .serde import JsonDataclassMixin
from .tracing.canonical import record_id, utc_now
from .tracing.capture.binding import (
    BindingCaptureV1,
    BindingContainerV1,
    BindingContextV1,
    BindingWorkloadV1,
    CaptureMode,
    CapturePolicyV1,
    Interception,
    TokenCaptureLevel,
    WorkloadKind,
    mint_binding,
)
from .tracing.models.actors import (
    ActorKind,
    ActorV5,
    CoverageState,
    SessionCoverageV5,
    SessionStatus,
    SessionV5,
)
from .tracing.models.completeness import (
    CaptureStatus,
    TerminationV5,
    TraceCompletenessV5,
    TraceLifecycleV5,
    TraceStatus,
)
from .tracing.models.document import TraceCaptureSummaryV5, TraceDocumentV5
from .tracing.models.identity import TraceIdentityV5, TraceKind, TraceProvenanceV5
from .tracing.models.spans import SpanKind, SpanV5, UsageProvenance, UsageV5
from .tracing.models.tokens import TokenCaptureProvenance, TokenCaptureV5, TokenSequenceRefV1

INFERENCE_CALL_SCHEMA_VERSION = "cispo.inference_call.v2"
TRAINABLE_EPISODE_SCHEMA_VERSION = "cispo.trainable_episode.v1"
RENDERER_PROFILE_SCHEMA_VERSION = "cispo.renderer_profile.v1"
EVIDENCE_BUNDLE_SCHEMA_VERSION = "cispo.evidence_bundle.v1"

#: Namespace for the CISPO facts the Trace V5 capture models have no field for.
SPAN_EXTENSION_KEY = "cispo"

WIRE_APIS = frozenset({"chat_completions", "responses"})
SAMPLING_TRANSPORTS = frozenset({"message_in_capture_out", "tokens_in_tokens_out"})
FINISH_REASONS = frozenset({"stop_token", "length_cap", "container_abort"})

#: Foreign authorship must be declared, never implied by a zero mask.
AUTHOR_KINDS = frozenset(
    {"policy", "foreign_agent", "opponent", "verifier", "judge", "harness"}
)
TRAINABLE_AUTHOR_KINDS = frozenset({"policy"})

TOKEN_CAPTURE_PROVENANCE = frozenset({"engine_meta", "probe_synthetic", "wire_derived"})
TRAINABLE_PROVENANCE = frozenset({"engine_meta"})

#: vLLM uses this both for missing sampled-token evidence and as a lower-bound
#: clamp, so receiving it can never prove a real logprob came back.
LOGPROB_SENTINEL = -9999.0

#: How the existing capture's provenance enum lands in the training plane. Only
#: ``OBSERVED_PROVIDER`` is engine-level token capture; everything else was seen
#: at or reconstructed from the wire and is therefore untrainable. ``UNAVAILABLE``
#: has no training-plane spelling at all: missing evidence is a terminal failure,
#: not a degraded record.
CAPTURE_PROVENANCE_MAP: Mapping[str, str] = {
    TokenCaptureProvenance.OBSERVED_PROVIDER.value: "engine_meta",
    TokenCaptureProvenance.OBSERVED_HARNESS.value: "wire_derived",
    TokenCaptureProvenance.IMPORTED.value: "wire_derived",
    TokenCaptureProvenance.DERIVED_RETOKENIZED.value: "wire_derived",
}

_AUTHOR_ACTOR_KIND: Mapping[str, str] = {
    "policy": ActorKind.AGENT.value,
    "foreign_agent": ActorKind.AGENT.value,
    "opponent": ActorKind.AGENT.value,
    "verifier": ActorKind.VERIFIER.value,
    "judge": ActorKind.EVALUATOR.value,
    "harness": ActorKind.ORCHESTRATOR.value,
}


class RecordError(ValueError):
    """A record was incomplete, malformed, or internally inconsistent."""


class EvidenceError(RecordError):
    """Evidence that cannot be trained on. Never degrade this to zero reward."""


def digest(payload: Any, *, length: int = 64) -> str:
    """Canonical sha256 over a JSON-serializable payload.

    Deliberately the training plane's own digest rather than this package's
    ``sha256:``-prefixed canonical digest: these values travel to the optimizer
    and must be byte-identical to the ones it computes.
    """

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()[:length]


def _ints(values: Any) -> tuple[int, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise RecordError("expected a sequence of integers")
    return tuple(int(item) for item in values)


def _floats(values: Any) -> tuple[float, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise RecordError("expected a sequence of floats")
    return tuple(float(item) for item in values)


# --------------------------------------------------------------------------- #
# Pinned identities
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RendererProfileV1(JsonDataclassMixin):
    """Pinned renderer identity. A version string alone is not an identity."""

    profile_id: str
    package: str
    package_version: str
    config_digest: str
    tokenizer_id: str
    tokenizer_digest: str
    stop_token_ids: tuple[int, ...]
    modalities: tuple[str, ...] = ("text",)
    add_generation_prompt: bool = True

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise RecordError("renderer profile_id is required")
        if not self.stop_token_ids:
            raise RecordError("renderer profile must declare stop token ids")

    @property
    def fingerprint(self) -> str:
        """Digest of everything that changes what a token sequence means."""

        return digest(
            {
                "schema_version": RENDERER_PROFILE_SCHEMA_VERSION,
                "profile_id": self.profile_id,
                "package": self.package,
                "package_version": self.package_version,
                "config_digest": self.config_digest,
                "tokenizer_id": self.tokenizer_id,
                "tokenizer_digest": self.tokenizer_digest,
                "stop_token_ids": list(self.stop_token_ids),
                "modalities": list(self.modalities),
                "add_generation_prompt": self.add_generation_prompt,
            },
            length=32,
        )

    def assert_matches(self, other: "RendererProfileV1") -> None:
        if self.fingerprint != other.fingerprint:
            raise RecordError(
                "renderer profile mismatch: "
                f"{self.profile_id}@{self.fingerprint} != {other.profile_id}@{other.fingerprint}"
            )


@dataclass(frozen=True, slots=True)
class SamplingProfileV1(JsonDataclassMixin):
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int | None = None
    seed: int | None = None

    @property
    def key(self) -> str:
        return digest(
            {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "seed": self.seed,
            },
            length=16,
        )


@dataclass(frozen=True, slots=True)
class BehaviorBindingV1(JsonDataclassMixin):
    """What the tokens were produced by. A group may not mix these."""

    renderer_profile: RendererProfileV1
    model_family: str
    model_id: str
    policy_revision: int
    wire_api: str
    sampling_transport: str
    sampling: SamplingProfileV1 = field(default_factory=SamplingProfileV1)

    def __post_init__(self) -> None:
        if self.wire_api not in WIRE_APIS:
            raise RecordError(f"unknown wire_api {self.wire_api!r}")
        if self.sampling_transport not in SAMPLING_TRANSPORTS:
            raise RecordError(f"unknown sampling_transport {self.sampling_transport!r}")
        if self.policy_revision < 0:
            raise RecordError("policy_revision must be non-negative")

    @property
    def fingerprint(self) -> str:
        return digest(
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


@dataclass(frozen=True, slots=True)
class CompactionProvenanceV1(JsonDataclassMixin):
    """Why a turn's prompt is not a strict prefix of the previous sequence."""

    rule: str
    divergence_index: int
    removed_message_indices: tuple[int, ...] = ()
    authored_by_policy: bool = False

    def __post_init__(self) -> None:
        if not self.rule.strip():
            raise RecordError("compaction rule is required")
        if self.divergence_index < 0:
            raise RecordError("divergence_index must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "divergence_index": self.divergence_index,
            "removed_message_indices": list(self.removed_message_indices),
            "authored_by_policy": self.authored_by_policy,
        }


# --------------------------------------------------------------------------- #
# The per-call record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class InferenceCallV1(JsonDataclassMixin):
    """One immutable record per proxied model call, before any flattening."""

    call_id: str
    proxy_request_id: str
    rollout_id: str
    group_id: str
    sample_index: int
    behavior_fingerprint: str
    policy_revision: int
    wire_api: str
    sampling_transport: str
    token_capture_provenance: str
    prompt_token_ids: tuple[int, ...]
    generation_token_ids: tuple[int, ...]
    generation_logprobs: tuple[float, ...]
    sampled_mask: tuple[int, ...]
    finish_reason: str
    stop_token_ids: tuple[int, ...] = ()
    content_mask: tuple[int, ...] = ()
    renderer_profile_fingerprint: str = ""
    trainable: bool = True
    branch_id: str = "root"
    parent_branch_id: str | None = None
    compaction: CompactionProvenanceV1 | None = None
    agent_instance_id: str | None = None
    team_id: str | None = None
    role_id: str | None = None
    policy_type_id: str | None = None
    parameter_group_id: str | None = None
    policy_set_revision_id: str | None = None
    effect_tick_start: int | None = None
    effect_tick_end: int | None = None
    author_kind: str = "policy"
    wire_request: dict[str, Any] = field(default_factory=dict)
    wire_response: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    schema_version: str = INFERENCE_CALL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.wire_api not in WIRE_APIS:
            raise RecordError(f"unknown wire_api {self.wire_api!r}")
        if self.sampling_transport not in SAMPLING_TRANSPORTS:
            raise RecordError(f"unknown sampling_transport {self.sampling_transport!r}")
        if self.token_capture_provenance not in TOKEN_CAPTURE_PROVENANCE:
            raise RecordError(f"unknown provenance {self.token_capture_provenance!r}")
        if self.finish_reason not in FINISH_REASONS:
            raise RecordError(f"unknown finish_reason {self.finish_reason!r}")
        if self.author_kind not in AUTHOR_KINDS:
            raise RecordError(f"unknown author_kind {self.author_kind!r}")

    def validate_for_training(self) -> None:
        """Every reason a call may not enter a batch. Raises, never degrades."""

        if not self.trainable:
            raise EvidenceError(f"call {self.call_id} is marked non-trainable")
        if self.token_capture_provenance not in TRAINABLE_PROVENANCE:
            raise EvidenceError(
                f"call {self.call_id} captured via {self.token_capture_provenance}; "
                "training requires engine-level token capture"
            )
        if not self.prompt_token_ids:
            raise EvidenceError(f"call {self.call_id} has no prompt tokens")
        if not self.generation_token_ids:
            raise EvidenceError(f"call {self.call_id} has no generated tokens")
        generated = len(self.generation_token_ids)
        if len(self.generation_logprobs) != generated:
            raise EvidenceError(
                f"call {self.call_id} logprob length {len(self.generation_logprobs)} "
                f"!= generated token count {generated}"
            )
        if self.sampled_mask and len(self.sampled_mask) != generated:
            raise EvidenceError(f"call {self.call_id} sampled mask length mismatch")
        if self.content_mask and len(self.content_mask) != generated:
            raise EvidenceError(f"call {self.call_id} content mask length mismatch")
        for index, value in enumerate(self.generation_logprobs):
            if math.isnan(value) or math.isinf(value):
                raise EvidenceError(f"call {self.call_id} logprob {index} is not finite")
            if value == LOGPROB_SENTINEL:
                raise EvidenceError(
                    f"call {self.call_id} logprob {index} is the provider sentinel "
                    f"{LOGPROB_SENTINEL}; presence of the sentinel cannot prove a real logprob"
                )
        if all(value == 0.0 for value in self.generation_logprobs):
            raise EvidenceError(f"call {self.call_id} logprobs are identically zero")

    @property
    def full_sequence(self) -> tuple[int, ...]:
        return tuple(self.prompt_token_ids) + tuple(self.generation_token_ids)

    @property
    def loss_mask(self) -> tuple[int, ...]:
        prompt = (0,) * len(self.prompt_token_ids)
        if self.sampled_mask:
            return prompt + tuple(int(bool(flag)) for flag in self.sampled_mask)
        return prompt + (1,) * len(self.generation_token_ids)

    def to_dict(self) -> dict[str, Any]:
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
            "prompt_token_ids": list(self.prompt_token_ids),
            "generation_token_ids": list(self.generation_token_ids),
            "generation_logprobs": list(self.generation_logprobs),
            "sampled_mask": list(self.sampled_mask),
            "content_mask": list(self.content_mask),
            "finish_reason": self.finish_reason,
            "stop_token_ids": list(self.stop_token_ids),
            "renderer_profile_fingerprint": self.renderer_profile_fingerprint,
            "trainable": self.trainable,
            "branch_id": self.branch_id,
            "parent_branch_id": self.parent_branch_id,
            "compaction": None if self.compaction is None else self.compaction.to_dict(),
            "agent_instance_id": self.agent_instance_id,
            "team_id": self.team_id,
            "role_id": self.role_id,
            "policy_type_id": self.policy_type_id,
            "parameter_group_id": self.parameter_group_id,
            "policy_set_revision_id": self.policy_set_revision_id,
            "effect_tick_start": self.effect_tick_start,
            "effect_tick_end": self.effect_tick_end,
            "wire_request": dict(self.wire_request),
            "wire_response": dict(self.wire_response),
            "usage": dict(self.usage),
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }


def assert_strict_prefix(previous: InferenceCallV1, following: InferenceCallV1) -> None:
    """Two calls stitch only on a byte-for-byte token prefix.

    Anything else forks a branch and seals the prior segment. Never retokenize
    new text onto old ids; an unexplained divergence is an evidence failure.
    """

    sequence = previous.full_sequence
    prompt = tuple(following.prompt_token_ids)
    if prompt[: len(sequence)] == sequence:
        if following.branch_id != previous.branch_id:
            raise EvidenceError(
                f"call {following.call_id} is a strict prefix continuation but changed branch"
            )
        return
    if following.compaction is None:
        content_divergence = next(
            (i for i, (a, b) in enumerate(zip(sequence, prompt, strict=False)) if a != b),
            None,
        )
        if content_divergence is None:
            raise EvidenceError(
                f"call {following.call_id} truncates {previous.call_id}: its prompt agrees for "
                f"{len(prompt)} tokens but the previous sequence is {len(sequence)} long, "
                "with no branch record and no declared compaction"
            )
        raise EvidenceError(
            f"call {following.call_id} diverges from {previous.call_id} at token "
            f"{content_divergence} with no branch record and no declared compaction"
        )
    if following.parent_branch_id != previous.branch_id:
        raise EvidenceError(
            f"call {following.call_id} declares compaction but does not fork from "
            f"branch {previous.branch_id!r}"
        )
    if following.branch_id == previous.branch_id:
        raise EvidenceError(
            f"call {following.call_id} declares compaction and must open a new branch"
        )


def assert_wire_not_flattened(call: InferenceCallV1) -> None:
    """A declared wire may not be persisted as the other one."""

    for name, payload in (("request", call.wire_request), ("response", call.wire_response)):
        declared = payload.get("wire")
        if declared and str(declared) != call.wire_api:
            raise EvidenceError(
                f"call {call.call_id} declares wire {call.wire_api!r} but persisted its "
                f"{name} as {declared!r}; flattening one wire into the other makes two "
                "datasets look like one"
            )


# --------------------------------------------------------------------------- #
# Trainer-facing views
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TrainableSegmentV1(JsonDataclassMixin):
    """One contiguous trainer sequence with its mask and behavior logprobs."""

    token_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]
    behavior_logprobs: tuple[float, ...]
    branch_id: str = "root"
    parameter_group_id: str | None = None
    agent_instance_id: str | None = None
    call_ids: tuple[str, ...] = ()
    author_kind: str = "policy"
    role_id: str | None = None
    policy_type_id: str | None = None
    team_id: str | None = None
    policy_revision: int | None = None
    policy_set_revision_id: str | None = None
    effect_tick_start: int | None = None
    effect_tick_end: int | None = None

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise RecordError("segment has no tokens")
        if len(self.loss_mask) != len(self.token_ids):
            raise RecordError("segment loss mask length mismatch")
        if len(self.behavior_logprobs) != len(self.token_ids):
            raise RecordError("segment behavior logprob length mismatch")
        if self.author_kind not in AUTHOR_KINDS:
            raise RecordError(f"unknown author_kind {self.author_kind!r}")
        if self.author_kind not in TRAINABLE_AUTHOR_KINDS and self.trainable_tokens:
            raise RecordError(
                f"segment authored by {self.author_kind!r} carries trainable tokens; "
                "foreign authorship is never trainable"
            )
        if (
            self.effect_tick_start is not None
            and self.effect_tick_end is not None
            and self.effect_tick_end < self.effect_tick_start
        ):
            raise RecordError("segment effect interval ends before it starts")

    @property
    def trainable_tokens(self) -> int:
        return sum(1 for flag in self.loss_mask if flag)

    @property
    def trainable(self) -> bool:
        return self.author_kind in TRAINABLE_AUTHOR_KINDS and bool(self.trainable_tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_ids": list(self.token_ids),
            "loss_mask": list(self.loss_mask),
            "behavior_logprobs": list(self.behavior_logprobs),
            "branch_id": self.branch_id,
            "parameter_group_id": self.parameter_group_id,
            "agent_instance_id": self.agent_instance_id,
            "call_ids": list(self.call_ids),
            "author_kind": self.author_kind,
            "role_id": self.role_id,
            "policy_type_id": self.policy_type_id,
            "team_id": self.team_id,
            "policy_revision": self.policy_revision,
            "policy_set_revision_id": self.policy_set_revision_id,
            "effect_tick_start": self.effect_tick_start,
            "effect_tick_end": self.effect_tick_end,
        }


@dataclass(frozen=True, slots=True)
class TrainableEpisodeV1(JsonDataclassMixin):
    """The common training view of one completed attempt, per policy instance."""

    rollout_id: str
    task_id: str
    seed: int
    policy_revision: int
    behavior_fingerprint: str
    segments: tuple[TrainableSegmentV1, ...]
    terminal_status: str
    usage: dict[str, Any] = field(default_factory=dict)
    agent_instance_id: str | None = None
    team_id: str | None = None
    policy_set_revision_id: str | None = None
    root_rollout_id: str | None = None
    trace_digest: str = ""
    probe: bool = False
    schema_version: str = TRAINABLE_EPISODE_SCHEMA_VERSION

    def validate(self) -> None:
        if self.probe:
            raise EvidenceError(
                f"episode {self.rollout_id} is probe-derived and may not enter a group or batch"
            )
        if not self.segments:
            raise EvidenceError(f"episode {self.rollout_id} has no trainable segments")
        if not self.trace_digest:
            raise EvidenceError(f"episode {self.rollout_id} has no sealed trace digest")
        if not any(segment.trainable_tokens for segment in self.segments):
            raise EvidenceError(f"episode {self.rollout_id} has no trainable tokens")

    @property
    def parameter_groups(self) -> tuple[str, ...]:
        seen: list[str] = []
        for segment in self.segments:
            group = segment.parameter_group_id
            if group is not None and group not in seen:
                seen.append(group)
        return tuple(seen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollout_id": self.rollout_id,
            "task_id": self.task_id,
            "seed": self.seed,
            "policy_revision": self.policy_revision,
            "behavior_fingerprint": self.behavior_fingerprint,
            "terminal_status": self.terminal_status,
            "usage": dict(self.usage),
            "agent_instance_id": self.agent_instance_id,
            "team_id": self.team_id,
            "policy_set_revision_id": self.policy_set_revision_id,
            "root_rollout_id": self.root_rollout_id,
            "trace_digest": self.trace_digest,
            "probe": self.probe,
            "segments": [segment.to_dict() for segment in self.segments],
        }


@dataclass(frozen=True, slots=True)
class SealedEvidenceV1(JsonDataclassMixin):
    """Everything one attempt hands the training plane, with its seal."""

    rollout_id: str
    trace_id: str
    trace_digest: str
    evidence_digest: str
    calls: tuple[InferenceCallV1, ...]
    episodes: tuple[TrainableEpisodeV1, ...]
    context_segments: tuple[TrainableSegmentV1, ...]
    document: TraceDocumentV5
    schema_version: str = EVIDENCE_BUNDLE_SCHEMA_VERSION

    def validate(self) -> None:
        """Terminal evidence failure, or nothing. Never a zero-reward trajectory."""

        if not self.episodes:
            raise EvidenceError(f"attempt {self.rollout_id} produced no episode")
        for call in self.calls:
            assert_wire_not_flattened(call)
            if call.trainable:
                call.validate_for_training()
        for episode in self.episodes:
            episode.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rollout_id": self.rollout_id,
            "trace_id": self.trace_id,
            "trace_digest": self.trace_digest,
            "evidence_digest": self.evidence_digest,
            "calls": [call.to_dict() for call in self.calls],
            "episodes": [episode.to_dict() for episode in self.episodes],
            "context_segments": [item.to_dict() for item in self.context_segments],
        }


# --------------------------------------------------------------------------- #
# Reading the existing capture
# --------------------------------------------------------------------------- #


def _author_kind_for(
    *,
    declared: str | None,
    role_id: str | None,
    policy_type_id: str | None,
    trainable: bool,
    provenance: str,
) -> str:
    """Who sampled these tokens. Declared where declared, derived only as a fallback."""

    if declared:
        if declared not in AUTHOR_KINDS:
            raise RecordError(f"unknown author_kind {declared!r}")
        return declared
    if role_id == "judge" or policy_type_id == "judge":
        return "judge"
    if role_id == "verifier" or policy_type_id == "verifier":
        return "verifier"
    if not trainable and provenance == "wire_derived":
        return "opponent"
    return "policy"


def inference_call_from_span(
    span: SpanV5,
    *,
    binding: BehaviorBindingV1,
    rollout_id: str,
    group_id: str = "",
    sample_index: int = 0,
    probe: bool = False,
) -> InferenceCallV1:
    """Lift one captured model-call span into a training record.

    ``span.token_capture`` supplies the token ids and the completion logprobs —
    the two things the Trace V5 capture models already model. Everything the
    training plane also needs and those models have no field for lives under
    ``span.metadata["cispo"]``.
    """

    if str(span.span_kind) != SpanKind.MODEL_CALL.value:
        raise RecordError(f"span {span.span_id} is not a model call")
    capture = span.token_capture
    if capture is None:
        raise EvidenceError(
            f"span {span.span_id} carries no token capture; missing token evidence is a "
            "terminal evidence failure, not an untrainable record"
        )
    raw_provenance = str(capture.provenance)
    if raw_provenance == TokenCaptureProvenance.UNAVAILABLE.value:
        raise EvidenceError(
            f"span {span.span_id} declares token capture unavailable; missing evidence is "
            "a terminal evidence failure"
        )
    extension = dict(span.metadata.get(SPAN_EXTENSION_KEY) or {})
    provenance = "probe_synthetic" if probe else CAPTURE_PROVENANCE_MAP.get(raw_provenance)
    if provenance is None:
        raise EvidenceError(f"span {span.span_id} has unmappable provenance {raw_provenance!r}")

    prompt = _ints(capture.prompt.token_ids if capture.prompt else ())
    generation = _ints(capture.completion.token_ids if capture.completion else ())
    logprobs = _floats(capture.completion_logprobs)
    sampled_mask = _ints(extension.get("sampled_mask")) or tuple(1 for _ in generation)
    role_id = extension.get("role_id")
    policy_type_id = extension.get("policy_type_id")
    trainable = bool(extension.get("trainable", not probe))
    author_kind = _author_kind_for(
        declared=extension.get("author_kind"),
        role_id=role_id,
        policy_type_id=policy_type_id,
        trainable=trainable,
        provenance=provenance,
    )
    if author_kind not in TRAINABLE_AUTHOR_KINDS:
        trainable = False
        sampled_mask = tuple(0 for _ in generation)

    compaction_payload = extension.get("compaction")
    compaction = (
        None
        if not compaction_payload
        else CompactionProvenanceV1(
            rule=str(compaction_payload["rule"]),
            divergence_index=int(compaction_payload["divergence_index"]),
            removed_message_indices=_ints(compaction_payload.get("removed_message_indices")),
            authored_by_policy=bool(compaction_payload.get("authored_by_policy")),
        )
    )
    revision = int(extension.get("policy_revision", binding.policy_revision))
    usage = dict(extension.get("usage") or {})
    if not usage and span.usage is not None:
        usage = {
            "prompt_tokens": span.usage.prompt_tokens,
            "completion_tokens": span.usage.completion_tokens,
        }
    return InferenceCallV1(
        call_id=str(extension.get("call_id") or span.span_id),
        proxy_request_id=str(extension.get("proxy_request_id") or span.span_id),
        rollout_id=rollout_id,
        group_id=group_id,
        sample_index=sample_index,
        behavior_fingerprint=str(extension.get("behavior_fingerprint") or binding.fingerprint),
        policy_revision=revision,
        wire_api=str(extension.get("wire_api") or binding.wire_api),
        sampling_transport=str(
            extension.get("sampling_transport") or binding.sampling_transport
        ),
        token_capture_provenance=provenance,
        prompt_token_ids=prompt,
        generation_token_ids=generation,
        generation_logprobs=logprobs,
        sampled_mask=sampled_mask,
        finish_reason=str(extension.get("finish_reason") or "stop_token"),
        stop_token_ids=binding.renderer_profile.stop_token_ids,
        content_mask=_ints(extension.get("content_mask")),
        renderer_profile_fingerprint=binding.renderer_profile.fingerprint,
        trainable=trainable,
        branch_id=str(extension.get("branch_id") or span.branch_id or "root"),
        parent_branch_id=extension.get("parent_branch_id"),
        compaction=compaction,
        agent_instance_id=extension.get("agent_instance_id"),
        team_id=extension.get("team_id"),
        role_id=role_id,
        policy_type_id=policy_type_id,
        parameter_group_id=extension.get("parameter_group_id"),
        policy_set_revision_id=extension.get("policy_set_revision_id"),
        effect_tick_start=extension.get("effect_tick_start"),
        effect_tick_end=extension.get("effect_tick_end"),
        author_kind=author_kind,
        wire_request=dict(span.detail.get("wire_request") or {}),
        wire_response=dict(span.detail.get("wire_response") or {}),
        usage=usage,
        created_at=span.started_at,
    )


# --------------------------------------------------------------------------- #
# Stitching
# --------------------------------------------------------------------------- #


def stitch(calls: Sequence[InferenceCallV1], *, author_kind: str = "policy") -> TrainableSegmentV1:
    """One contiguous trainer sequence. Retained prefixes are loss-masked.

    A segment authored by anything but the policy carries a fully zero mask and
    names its author: the shared record refuses foreign authorship that carries
    trainable tokens, and that refusal is the point.
    """

    if not calls:
        raise EvidenceError("cannot stitch an empty span group")
    final = calls[-1]
    total = final.full_sequence
    mask = [0] * len(total)
    logprobs = [0.0] * len(total)
    policy_authored = author_kind in TRAINABLE_AUTHOR_KINDS
    for call in calls:
        start = len(call.prompt_token_ids)
        if start + len(call.generation_token_ids) > len(total):
            continue
        flags = call.sampled_mask or tuple(1 for _ in call.generation_token_ids)
        for offset, flag in enumerate(flags):
            mask[start + offset] = int(bool(flag)) if policy_authored else 0
        for offset, value in enumerate(call.generation_logprobs):
            if start + offset < len(logprobs):
                logprobs[start + offset] = value
    starts = [item.effect_tick_start for item in calls if item.effect_tick_start is not None]
    ends = [item.effect_tick_end for item in calls if item.effect_tick_end is not None]
    return TrainableSegmentV1(
        token_ids=total,
        loss_mask=tuple(mask),
        behavior_logprobs=tuple(logprobs),
        branch_id=final.branch_id,
        parameter_group_id=final.parameter_group_id,
        agent_instance_id=final.agent_instance_id,
        call_ids=tuple(item.call_id for item in calls),
        author_kind=author_kind,
        role_id=final.role_id,
        policy_type_id=final.policy_type_id,
        team_id=final.team_id,
        policy_revision=final.policy_revision,
        policy_set_revision_id=final.policy_set_revision_id,
        effect_tick_start=min(starts) if starts else None,
        effect_tick_end=max(ends) if ends else None,
    )


def segments_by_branch(
    calls: Sequence[InferenceCallV1], *, author_kind: str
) -> tuple[TrainableSegmentV1, ...]:
    """A compaction sealed the prior segment; grouping by branch is that seal."""

    branches: dict[str, list[InferenceCallV1]] = {}
    for call in calls:
        branches.setdefault(call.branch_id, []).append(call)
    return tuple(stitch(group, author_kind=author_kind) for group in branches.values())


# --------------------------------------------------------------------------- #
# The builder
# --------------------------------------------------------------------------- #


class CispoEvidenceBuilder:
    """Accumulate captured calls, check stitching, then seal a Trace V5 document."""

    def __init__(
        self,
        *,
        rollout_id: str,
        task_id: str,
        binding: BehaviorBindingV1,
        seed: int = 0,
        group_id: str = "",
        sample_index: int = 0,
        probe: bool = False,
    ) -> None:
        self._rollout_id = rollout_id
        self._task_id = task_id
        self._binding = binding
        self._seed = int(seed)
        self._group_id = group_id
        self._sample_index = int(sample_index)
        self._probe = bool(probe)
        self._calls: list[InferenceCallV1] = []
        self._previous: dict[str | None, InferenceCallV1] = {}

    @property
    def calls(self) -> tuple[InferenceCallV1, ...]:
        return tuple(self._calls)

    def observe_span(self, span: SpanV5) -> InferenceCallV1:
        return self.observe(
            inference_call_from_span(
                span,
                binding=self._binding,
                rollout_id=self._rollout_id,
                group_id=self._group_id,
                sample_index=self._sample_index,
                probe=self._probe,
            )
        )

    def observe(self, call: InferenceCallV1) -> InferenceCallV1:
        """Append a call, enforcing the strict-prefix rule within its own stream.

        Per-instance streams are checked independently: a concurrent real-time
        topology has no global turn order to serialize into.
        """

        assert_wire_not_flattened(call)
        previous = self._previous.get(call.agent_instance_id)
        if previous is not None and call.author_kind == previous.author_kind:
            assert_strict_prefix(previous, call)
        self._calls.append(call)
        self._previous[call.agent_instance_id] = call
        return call

    # -- views ------------------------------------------------------------ #

    def context_segments(self) -> tuple[TrainableSegmentV1, ...]:
        """Foreign-authored spans, recorded as untrainable context with an author."""

        grouped: dict[tuple[str, str | None], list[InferenceCallV1]] = {}
        for call in self._calls:
            if call.author_kind in TRAINABLE_AUTHOR_KINDS:
                continue
            grouped.setdefault((call.author_kind, call.agent_instance_id), []).append(call)
        segments: list[TrainableSegmentV1] = []
        for (author, _instance), group in grouped.items():
            segments.extend(segments_by_branch(group, author_kind=author))
        return tuple(segments)

    def episodes(self, *, terminal_status: str, trace_digest: str) -> tuple[TrainableEpisodeV1, ...]:
        """One trajectory per policy-authored agent instance."""

        by_instance: dict[str | None, list[InferenceCallV1]] = {}
        for call in self._calls:
            if call.author_kind not in TRAINABLE_AUTHOR_KINDS:
                continue
            by_instance.setdefault(call.agent_instance_id, []).append(call)
        episodes: list[TrainableEpisodeV1] = []
        for instance_id, calls in by_instance.items():
            head = calls[0]
            episodes.append(
                TrainableEpisodeV1(
                    rollout_id=self._rollout_id,
                    task_id=self._task_id,
                    seed=self._seed,
                    policy_revision=head.policy_revision,
                    behavior_fingerprint=head.behavior_fingerprint,
                    segments=segments_by_branch(calls, author_kind="policy"),
                    terminal_status=terminal_status,
                    usage={
                        "calls": len(calls),
                        "prompt_tokens": sum(len(c.prompt_token_ids) for c in calls),
                        "completion_tokens": sum(len(c.generation_token_ids) for c in calls),
                        "provider_request_ids": [c.proxy_request_id for c in calls],
                    },
                    agent_instance_id=instance_id,
                    team_id=head.team_id,
                    policy_set_revision_id=head.policy_set_revision_id,
                    root_rollout_id=self._rollout_id,
                    trace_digest=trace_digest,
                    probe=self._probe,
                )
            )
        return tuple(episodes)

    # -- sealing ---------------------------------------------------------- #

    def seal(
        self,
        *,
        terminal_status: str = "completed",
        started_at: str | None = None,
        ended_at: str | None = None,
        correlation: Mapping[str, Any] | None = None,
    ) -> SealedEvidenceV1:
        """Build and seal the Trace V5 document, then bind the episodes to its digest."""

        if not self._calls:
            raise EvidenceError(
                f"attempt {self._rollout_id} sealed no model call; missing trainable "
                "evidence is a terminal evidence failure"
            )
        started = started_at or self._calls[0].created_at or utc_now()
        ended = ended_at or self._calls[-1].created_at or started
        document = self._document(
            terminal_status=terminal_status,
            started_at=started,
            ended_at=ended,
            correlation=dict(correlation or {}),
        ).sealed()
        trace_digest = document.content_digest
        episodes = self.episodes(terminal_status=terminal_status, trace_digest=trace_digest)
        context = self.context_segments()
        evidence_digest = digest(
            {
                "trace_digest": trace_digest,
                "calls": [call.to_dict() for call in self._calls],
                "episodes": [episode.to_dict() for episode in episodes],
                "context_segments": [item.to_dict() for item in context],
            }
        )
        return SealedEvidenceV1(
            rollout_id=self._rollout_id,
            trace_id=document.trace_id,
            trace_digest=trace_digest,
            evidence_digest=evidence_digest,
            calls=tuple(self._calls),
            episodes=episodes,
            context_segments=context,
            document=document,
        )

    def _actor_key(self, call: InferenceCallV1) -> tuple[str, str]:
        return (call.author_kind, call.agent_instance_id or "solo")

    def _document(
        self,
        *,
        terminal_status: str,
        started_at: str,
        ended_at: str,
        correlation: Mapping[str, Any],
    ) -> TraceDocumentV5:
        trace_id = record_id(
            "trace",
            kind="cispo_attempt",
            scope=(self._rollout_id,),
            key={"task_id": self._task_id, "binding": self._binding.fingerprint},
        )
        capture_id = record_id(
            "cap", kind="cispo_attempt_capture", scope=(trace_id,), key={"seed": self._seed}
        )
        actor_ids: dict[tuple[str, str], str] = {}
        session_ids: dict[tuple[str, str], str] = {}
        for call in self._calls:
            key = self._actor_key(call)
            if key in actor_ids:
                continue
            actor_ids[key] = record_id("act", kind="cispo_author", scope=(trace_id,), key=key)
            session_ids[key] = record_id("ses", kind="cispo_author", scope=(trace_id,), key=key)

        first = self._calls[0]
        root_key = self._actor_key(first)
        binding_record = mint_binding(
            trace_id=trace_id,
            capture_id=capture_id,
            trace_kind=TraceKind.AGENT_ROLLOUT,
            policy=CapturePolicyV1(
                profile="cispo_trainable_evidence",
                raw_capture="inline_wire_objects",
                token_level=TokenCaptureLevel.FULL_TRAINING,
                retention_class="local_only",
            ),
            workload=BindingWorkloadV1(
                kind=WorkloadKind.OTHER,
                root_actor_id=actor_ids[root_key],
                actor_session_id=session_ids[root_key],
                run_id=str(correlation.get("run_id") or "") or None,
                rollout_id=self._rollout_id,
            ),
            capture=BindingCaptureV1(
                interception=Interception.PROVIDER_PROXY,
                mode=CaptureMode.REQUIRED,
                proxy_profile=self._binding.wire_api,
                output_artifact_root="local_only",
            ),
            container=BindingContainerV1(contract_version=EVIDENCE_BUNDLE_SCHEMA_VERSION),
            context=BindingContextV1(task_id=self._task_id, seed=self._seed),
            metadata={"behavior_fingerprint": self._binding.fingerprint},
        )
        # ``mint_binding`` stamps wall-clock time. Pin it to the attempt's own
        # start so one attempt's evidence seals to one digest however often it is
        # rebuilt; a trace digest that drifts with the clock is not a binding.
        binding_record = replace(binding_record, created_at=started_at).sealed()
        coverage = SessionCoverageV5(
            model_calls=CoverageState.COMPLETE,
            usage=CoverageState.COMPLETE,
            raw_provider=CoverageState.COMPLETE,
            reasons=("every proxied model call is retained with its wire objects",),
        )
        actors = tuple(
            ActorV5(
                actor_id=actor_id,
                kind=_AUTHOR_ACTOR_KIND[author],
                display_name=f"{author}:{instance}",
                role=author,
                model=self._binding.model_id if author == "policy" else None,
                task_id=self._task_id,
                metadata={"author_kind": author, "agent_instance_id": instance},
            ).sealed()
            for (author, instance), actor_id in actor_ids.items()
        )
        sessions = tuple(
            SessionV5(
                session_id=session_id,
                actor_id=actor_ids[key],
                started_at=started_at,
                ended_at=ended_at,
                capture_id=capture_id,
                status=(
                    SessionStatus.COMPLETED
                    if terminal_status == "completed"
                    else SessionStatus.FAILED
                ),
                coverage=coverage,
                metadata={"author_kind": key[0], "agent_instance_id": key[1]},
            ).sealed()
            for key, session_id in session_ids.items()
        )
        spans = tuple(self._span_for(call, trace_id, actor_ids, session_ids) for call in self._calls)
        return TraceDocumentV5(
            trace_id=trace_id,
            trace_kind=TraceKind.AGENT_ROLLOUT,
            identity=TraceIdentityV5(
                rollout_id=self._rollout_id,
                run_id=str(correlation.get("run_id") or "") or None,
                task_id=self._task_id,
                seed=self._seed,
            ),
            lifecycle=TraceLifecycleV5(
                status=(
                    TraceStatus.COMPLETED if terminal_status == "completed" else TraceStatus.FAILED
                ),
                started_at=started_at,
                ended_at=ended_at,
                termination=TerminationV5(reason=terminal_status),
            ),
            capture=TraceCaptureSummaryV5(
                capture_id=capture_id,
                binding_id=binding_record.binding_id,
                binding_digest=binding_record.content_digest,
                capture_profile="cispo_trainable_evidence",
                interception=Interception.PROVIDER_PROXY.value,
                mode=CaptureMode.REQUIRED.value,
                raw_record_count=len(self._calls),
            ),
            provenance=TraceProvenanceV5(
                producer="synth-containers-cispo",
                producer_version=EVIDENCE_BUNDLE_SCHEMA_VERSION,
                source_format=INFERENCE_CALL_SCHEMA_VERSION,
                model=self._binding.model_id,
                captured_at=ended_at,
                extra={
                    "behavior_fingerprint": self._binding.fingerprint,
                    "renderer_profile_fingerprint": self._binding.renderer_profile.fingerprint,
                    "wire_api": self._binding.wire_api,
                    "sampling_transport": self._binding.sampling_transport,
                    "correlation": dict(correlation),
                },
            ),
            completeness=TraceCompletenessV5(
                capture_status=CaptureStatus.COMPLETE,
                terminal_event_observed=True,
                model_calls=CoverageState.COMPLETE,
                raw_provider=CoverageState.COMPLETE,
                usage=CoverageState.COMPLETE,
                expected_record_count=len(self._calls),
                captured_record_count=len(self._calls),
            ),
            actors=actors,
            sessions=sessions,
            spans=spans,
            usage=UsageV5(
                provenance=UsageProvenance.OBSERVED_PROVIDER,
                prompt_tokens=sum(len(c.prompt_token_ids) for c in self._calls),
                completion_tokens=sum(len(c.generation_token_ids) for c in self._calls),
                requests=len(self._calls),
            ),
        )

    def _span_for(
        self,
        call: InferenceCallV1,
        trace_id: str,
        actor_ids: Mapping[tuple[str, str], str],
        session_ids: Mapping[tuple[str, str], str],
    ) -> SpanV5:
        key = self._actor_key(call)
        aligned = len(call.generation_logprobs) == len(call.generation_token_ids)
        capture = TokenCaptureV5(
            provenance=(
                TokenCaptureProvenance.OBSERVED_PROVIDER
                if call.token_capture_provenance == "engine_meta"
                else TokenCaptureProvenance.OBSERVED_HARNESS
            ),
            level="token_ids",
            tokenizer=self._binding.renderer_profile.tokenizer_id,
            tokenizer_revision=self._binding.renderer_profile.tokenizer_digest,
            prompt=TokenSequenceRefV1(
                token_ids=call.prompt_token_ids, count=len(call.prompt_token_ids)
            ),
            completion=TokenSequenceRefV1(
                token_ids=call.generation_token_ids, count=len(call.generation_token_ids)
            ),
            completion_logprobs=call.generation_logprobs if aligned else (),
            # A misaligned vector is recorded as unavailable here rather than
            # dropped: the training record still carries it, and
            # ``validate_for_training`` is what refuses it.
            unavailable_fields=() if aligned else ("completion_logprobs_length",),
            metadata={
                "token_capture_provenance": call.token_capture_provenance,
                "renderer_profile_fingerprint": call.renderer_profile_fingerprint,
                "observed_logprob_count": len(call.generation_logprobs),
            },
        )
        return SpanV5(
            span_id=record_id(
                "span", kind="cispo_model_call", scope=(trace_id,), key={"call": call.call_id}
            ),
            span_kind=SpanKind.MODEL_CALL,
            actor_id=actor_ids[key],
            session_id=session_ids[key],
            started_at=call.created_at or utc_now(),
            branch_id=call.branch_id,
            detail={
                "wire_request": dict(call.wire_request),
                "wire_response": dict(call.wire_response),
                "finish_reason": call.finish_reason,
            },
            usage=UsageV5(
                provenance=UsageProvenance.OBSERVED_PROVIDER,
                prompt_tokens=len(call.prompt_token_ids),
                completion_tokens=len(call.generation_token_ids),
                requests=1,
            ),
            token_capture=capture,
            metadata={SPAN_EXTENSION_KEY: call.to_dict()},
        ).sealed()


# --------------------------------------------------------------------------- #
# The one coupling point to the route layer
# --------------------------------------------------------------------------- #


class CispoEvidenceAdapter:
    """Route-shaped facade over :class:`CispoEvidenceBuilder`.

    Method names match the declared route table: ``trace`` returns the sealed
    document and its digest, ``evidence`` returns the training records.
    """

    def __init__(self, evidence: SealedEvidenceV1) -> None:
        self._evidence = evidence

    @property
    def evidence(self) -> SealedEvidenceV1:
        return self._evidence

    @property
    def trace_digest(self) -> str:
        return self._evidence.trace_digest

    def trace(self) -> dict[str, Any]:
        return self._evidence.document.to_dict()

    def trace_reference(self) -> dict[str, Any]:
        return {
            "schema_version": self._evidence.document.schema_version,
            "trace_id": self._evidence.trace_id,
            "content_digest": self._evidence.trace_digest,
            "span_count": len(self._evidence.document.spans),
            "url": f"/rollouts/{self._evidence.rollout_id}/trace",
        }

    def evidence_payload(self) -> dict[str, Any]:
        return self._evidence.to_dict()


__all__ = [
    "AUTHOR_KINDS",
    "CAPTURE_PROVENANCE_MAP",
    "EVIDENCE_BUNDLE_SCHEMA_VERSION",
    "INFERENCE_CALL_SCHEMA_VERSION",
    "LOGPROB_SENTINEL",
    "SPAN_EXTENSION_KEY",
    "TRAINABLE_AUTHOR_KINDS",
    "TRAINABLE_EPISODE_SCHEMA_VERSION",
    "BehaviorBindingV1",
    "CispoEvidenceAdapter",
    "CispoEvidenceBuilder",
    "CompactionProvenanceV1",
    "EvidenceError",
    "InferenceCallV1",
    "RecordError",
    "RendererProfileV1",
    "SamplingProfileV1",
    "SealedEvidenceV1",
    "TrainableEpisodeV1",
    "TrainableSegmentV1",
    "assert_strict_prefix",
    "assert_wire_not_flattened",
    "digest",
    "inference_call_from_span",
    "segments_by_branch",
    "stitch",
]
