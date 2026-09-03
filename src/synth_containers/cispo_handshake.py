"""Container side of the CISPO startup handshake and the admission gate.

Reading an advertisement is discovery; this module is agreement. Before the
executor creates a training session or issues one paid provider request it
POSTs a requirement document here, and this module answers *every* clause in
the canonical registry with a verdict and a reason -- never a bare boolean,
because a boolean says a run will fail without saying what to change.

Three properties are load-bearing, and each is enforced structurally:

1. **Answers are derived, not tabulated.** Every clause verdict is a pure
   function of :class:`RuntimeFacts`, which is itself derived from the build's
   declared runtime surface (``RuntimeCapabilitySurface`` plus the platform's
   own declarations). There is no table of what "usually" works, and no clause
   answer inspects a task, harness, or environment name.
2. **The digest is computed the way the executor computes it.** A digest both
   sides compute differently is worse than no digest, so
   :func:`compute_agreement_digest` mirrors
   ``synth_optimizers.rl.handshake.compute_agreement_digest`` field for field,
   over the canonical obligations payload the executor will reconstruct after
   ``Obligations.from_payload`` drops the container's extra keys.
3. **The agreement gates every attempt.** :meth:`HandshakeRegistry.assert_admissible`
   refuses an attempt whose handshake is absent, unknown, expired, revoked, or
   whose agreement digest is not the one agreed. Renewal re-reads the
   capability document and fails closed on any change; the container revokes
   when it degrades.

Interop hazard, deliberate, and reported rather than papered over:
``lifecycle.clock_skew`` is *conditional* -- a ``steps`` or ``env_ticks``
horizon reads no wall clock, so the clause does not apply, which is a different
statement from declining it. This module therefore omits it from ``clauses``
when it does not apply and names it in ``not_applicable_clauses`` instead. The
optimizer's ``_unanswered_clauses`` currently walks ``MANDATORY_CLAUSES``
without consulting ``applies()``, so it would read that omission as a rejected
mandatory clause. Answering a clause that does not apply would be the worse
lie; the fix belongs on the executor side.

The canonical clause identifiers live in
``synth_optimizers/contracts/rl_clauses.py``. That module is authoritative;
the mirror below exists only because this package does not depend on the
optimizer, and ``tests/test_cispo_handshake.py`` pins the mirror against the
original whenever the optimizer source is reachable.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any

from .capabilities import RuntimeCapabilitySurface
from .serde import JsonDataclassMixin

# --------------------------------------------------------------------------- #
# Mirror of the canonical clause registry (synth_optimizers/contracts/rl_clauses)
# --------------------------------------------------------------------------- #

HANDSHAKE_SCHEMA_VERSION = "cispo.handshake.v1"
CAPABILITY_SCHEMA_VERSION = "cispo.capabilities.v1"
CISPO_CONTRACT_VERSION = "synth_optimizers.cispo.v1"

VERDICTS: tuple[str, ...] = ("accepted", "degraded", "rejected", "unsupported")

CLAUSE_GROUPS: dict[str, tuple[str, ...]] = {
    "contract": ("contract.version", "contract.routes"),
    "discovery": ("discovery.taskset", "discovery.task_digests", "discovery.topology"),
    "policy": (
        "policy.binding_transport",
        "policy.renderer_profile_match",
        "policy.revision_immutability",
        "policy.no_embedded_credentials",
        "policy.session_scoped_origin",
    ),
    "lifecycle": (
        "lifecycle.idempotency",
        "lifecycle.lease_renewal",
        "lifecycle.cancellation",
        "lifecycle.concurrency",
        "lifecycle.exactly_one_terminal",
        "lifecycle.pause_resume",
        "lifecycle.clock_skew",
    ),
    "evidence": (
        "evidence.trace_v5",
        "evidence.behavior_logprobs",
        "evidence.strict_prefix",
        "evidence.masking",
        "evidence.wire_objects",
        "evidence.artifact_reference",
        "evidence.tito",
    ),
    "reward": (
        "reward.authority",
        "reward.binding_digest",
        "reward.horizon_quiescence",
        "reward.settlement_window",
        "reward.channels",
    ),
    "recovery": ("recovery.restart", "recovery.stale_discard"),
    "topology": (
        "topology.roster",
        "topology.channels",
        "topology.minimum_roster",
        "topology.opponent_pinning",
    ),
}

ALL_CLAUSES: tuple[str, ...] = tuple(
    clause for clauses in CLAUSE_GROUPS.values() for clause in clauses
)

CONDITIONAL_CLAUSES: dict[str, str] = {
    "lifecycle.clock_skew": "horizon_kind == 'wall_clock'",
}

CLAUSE_SUBSTITUTES: dict[str, tuple[str, ...]] = {
    "reward.horizon_quiescence": ("horizon_clipped_snapshot",),
    "evidence.artifact_reference": ("inline_evidence_only",),
    "lifecycle.pause_resume": ("cancel_and_replace",),
}

OPTIONAL_CLAUSES: frozenset[str] = frozenset(
    {
        "evidence.tito",
        "evidence.artifact_reference",
        "reward.settlement_window",
        "lifecycle.pause_resume",
        "topology.channels",
        "topology.minimum_roster",
        "topology.opponent_pinning",
    }
)

MANDATORY_CLAUSES: tuple[str, ...] = tuple(
    clause for clause in ALL_CLAUSES if clause not in OPTIONAL_CLAUSES
)

UNCONDITIONAL_MANDATORY_CLAUSES: tuple[str, ...] = tuple(
    clause for clause in MANDATORY_CLAUSES if clause not in CONDITIONAL_CLAUSES
)

HORIZON_KINDS: frozenset[str] = frozenset({"wall_clock", "steps", "env_ticks"})
TURN_MODELS: frozenset[str] = frozenset({"sequential", "concurrent_realtime"})
ACTUATION_MODELS: frozenset[str] = frozenset({"direct_action", "deferred_program"})
REWARD_RELATIONS: frozenset[str] = frozenset(
    {"cooperative", "competitive_rank", "competitive_margin", "mixed"}
)
PARTIAL_ROSTER_DISPOSITIONS: frozenset[str] = frozenset(
    {"refuse", "drop_instance", "refuse_team"}
)
SAMPLING_TRANSPORTS: frozenset[str] = frozenset(
    {"message_in_capture_out", "tokens_in_tokens_out"}
)
WIRE_APIS: frozenset[str] = frozenset({"chat_completions", "responses"})

#: Route names the contract must declare. Route presence and behavior support
#: are separate questions: a container that cannot serve artifacts by reference
#: still declares the route it would serve them on.
MANDATORY_ROUTE_NAMES: tuple[str, ...] = (
    "health_route",
    "capabilities_route",
    "handshake_route",
    "taskset_route",
    "taskset_tasks_route",
    "topology_route",
    "policy_bind_route",
    "policy_set_bind_route",
    "rollout_route",
    "rollout_state_route",
    "rollout_events_route",
    "rollout_renew_route",
    "rollout_finalize_route",
    "rollout_terminate_route",
    "trace_route",
    "artifacts_route",
    "reward_route",
)


def applies(clause_id: str, *, horizon_kind: str | None = None) -> bool:
    """Whether a conditional clause applies to a run of this shape."""

    if clause_id not in CONDITIONAL_CLAUSES:
        return True
    if clause_id == "lifecycle.clock_skew":
        return horizon_kind == "wall_clock"
    raise KeyError(f"conditional clause {clause_id!r} has no applicability rule")


def substitutes_for(clause_id: str) -> tuple[str, ...]:
    """Declared substitutes that satisfy a mandatory clause by other means."""

    return CLAUSE_SUBSTITUTES.get(clause_id, ())


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class HandshakeError(ValueError):
    """Base for every refusal this module issues, with an HTTP shape."""

    status_code = 400
    error = "handshake_error"

    def payload(self) -> dict[str, Any]:
        return {"error": self.error, "reason": str(self), "status_code": self.status_code}


class MalformedRequirementDocument(HandshakeError):
    """The requirement document is absent, mis-versioned, or self-inconsistent."""

    status_code = 400
    error = "malformed_requirement_document"


class UndeclaredSubstitute(HandshakeError):
    """``accept_degraded`` acknowledged a substitute nobody declared."""

    status_code = 400
    error = "accept_degraded_undeclared"


class AdmissionRefused(HandshakeError):
    """An attempt did not pass the agreement gate. No work is admitted."""

    status_code = 403
    error = "admission_refused"


class HandshakeAbsent(AdmissionRefused):
    status_code = 400
    error = "handshake_absent"


class HandshakeUnknown(AdmissionRefused):
    status_code = 403
    error = "handshake_unknown"


class HandshakeExpired(AdmissionRefused):
    status_code = 403
    error = "handshake_expired"


class HandshakeRevoked(AdmissionRefused):
    status_code = 403
    error = "handshake_revoked"


class AgreementMismatch(AdmissionRefused):
    status_code = 409
    error = "agreement_digest_mismatch"


class CapabilityDrift(HandshakeError):
    """The capability document changed under a live handshake. Fail closed."""

    status_code = 409
    error = "capability_document_changed"


# --------------------------------------------------------------------------- #
# Canonical digests and time
# --------------------------------------------------------------------------- #


def canonical_digest(payload: Any, *, length: int = 64) -> str:
    """Canonical sha256 over a JSON-serialisable payload.

    Byte-identical to ``synth_optimizers.contracts.rl_records.digest``: sorted
    keys, no whitespace, ``default=str``, ASCII-escaped, truncated to ``length``.
    """

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()[:length]


def canonical_capability_hash(document: Mapping[str, Any]) -> str:
    """Hash every field but the offered hash itself.

    Mirrors ``synth_optimizers.rl.capabilities.canonical_capability_hash``,
    including ``ensure_ascii=False``, which differs from
    :func:`canonical_digest` and must not be unified with it.
    """

    unhashed = {key: value for key, value in document.items() if key != "capability_hash"}
    raw = json.dumps(unhashed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"sha256:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def format_rfc3339(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_rfc3339(value: Any, *, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise MalformedRequirementDocument(f"{field_name} is required as an RFC3339 timestamp")
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise MalformedRequirementDocument(
            f"{field_name} is not an RFC3339 timestamp: {value!r}"
        ) from exc
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Declared runtime facts. Every clause answer is a function of these.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RendererProfileFacts(JsonDataclassMixin):
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
            raise ValueError("renderer profile_id is required")
        if not self.stop_token_ids:
            raise ValueError("renderer profile must declare stop token ids")

    def to_payload(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "package": self.package,
            "package_version": self.package_version,
            "config_digest": self.config_digest,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_digest": self.tokenizer_digest,
            "stop_token_ids": list(self.stop_token_ids),
            "modalities": list(self.modalities),
            "add_generation_prompt": self.add_generation_prompt,
        }

    @property
    def fingerprint(self) -> str:
        """Mirrors ``RendererProfile.fingerprint`` -- same payload, same length."""

        return canonical_digest(
            {
                "schema_version": "cispo.renderer_profile.v1",
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


@dataclass(frozen=True, slots=True)
class TaskRow(JsonDataclassMixin):
    """One resolvable taskset row, with the digest of its own content."""

    task_id: str
    content_digest: str
    topology_ref: str
    task_family: str = ""

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task row has no task_id")
        if not self.content_digest.strip():
            raise ValueError(f"task {self.task_id} has no content digest")


def task_content_digest(task_payload: Mapping[str, Any]) -> str:
    """Content address of one task row, derived from the row itself."""

    return "sha256:" + canonical_digest(task_payload)


@dataclass(frozen=True, slots=True)
class DiscoveryFacts(JsonDataclassMixin):
    taskset_id: str
    taskset_version: str
    splits: tuple[str, ...]
    rows: tuple[TaskRow, ...]
    task_content_digests: bool = False
    deterministic_lookup: bool = False
    duplicate_free: bool = False

    def row(self, task_id: str) -> TaskRow | None:
        for item in self.rows:
            if item.task_id == task_id:
                return item
        return None


@dataclass(frozen=True, slots=True)
class PolicyFacts(JsonDataclassMixin):
    binding_transport: str = "message_in_capture_out"
    wire_api: str = "chat_completions"
    tokens_in_tokens_out: bool = False
    session_scoped_sampler_origin: bool = False
    embeds_credentials: bool = True
    revision_immutable_after_admission: bool = False
    records_policy_revision: bool = False

    def __post_init__(self) -> None:
        if self.binding_transport not in SAMPLING_TRANSPORTS:
            raise ValueError(f"unknown binding_transport {self.binding_transport!r}")
        if self.wire_api not in WIRE_APIS:
            raise ValueError(f"unknown wire_api {self.wire_api!r}")

    @property
    def supported_transports(self) -> tuple[str, ...]:
        transports = [self.binding_transport]
        if self.tokens_in_tokens_out and "tokens_in_tokens_out" not in transports:
            transports.append("tokens_in_tokens_out")
        return tuple(transports)


@dataclass(frozen=True, slots=True)
class LifecycleFacts(JsonDataclassMixin):
    max_concurrency: int = 1
    lease_ttl_seconds: float = 0.0
    lease_renewable: bool = False
    supports_idempotency: bool = False
    supports_cancellation: bool = False
    exactly_one_terminal_result: bool = False
    supports_pause_resume: bool = False
    straggler_grace_seconds: float = 0.0
    clock_skew_tolerance_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")


@dataclass(frozen=True, slots=True)
class EvidenceFacts(JsonDataclassMixin):
    trace_v5: bool = False
    behavior_logprobs: bool = False
    strict_prefix: bool = False
    masking: bool = False
    wire_objects: bool = False
    artifact_reference: bool = False
    inline_evidence: bool = False
    tokens_in_tokens_out: bool = False


@dataclass(frozen=True, slots=True)
class RewardFacts(JsonDataclassMixin):
    authority: str = ""
    evaluation_plan_id: str = ""
    reward_relation: str = "cooperative"
    channels: tuple[str, ...] = ()
    binds_trace_digest: bool = False
    quiescence: bool = False
    horizon_clipping: bool = False
    settlement_window_seconds: float = 0.0
    deferred_scoring: bool = False

    def __post_init__(self) -> None:
        if self.reward_relation not in REWARD_RELATIONS:
            raise ValueError(f"unknown reward_relation {self.reward_relation!r}")


@dataclass(frozen=True, slots=True)
class RecoveryFacts(JsonDataclassMixin):
    restart: bool = False
    stale_discard: bool = False


@dataclass(frozen=True, slots=True)
class AgentInstanceFacts(JsonDataclassMixin):
    agent_instance_id: str
    role_id: str
    policy_type_id: str
    team_id: str
    trainable: bool = True
    pinned_identity: str | None = None


@dataclass(frozen=True, slots=True)
class TeamFacts(JsonDataclassMixin):
    team_id: str
    trainable: bool = True
    minimum_viable_roster: int = 0


@dataclass(frozen=True, slots=True)
class ChannelFacts(JsonDataclassMixin):
    channel_id: str
    scope: str
    trainable_for_author: bool = True


@dataclass(frozen=True, slots=True)
class TopologyFacts(JsonDataclassMixin):
    topology_id: str
    turn_model: str = "sequential"
    actuation_model: str = "direct_action"
    agent_instances: tuple[AgentInstanceFacts, ...] = ()
    teams: tuple[TeamFacts, ...] = ()
    communication_channels: tuple[ChannelFacts, ...] = ()
    parameter_groups: Mapping[str, str] = field(default_factory=dict)
    partial_roster_disposition: str = "refuse"
    pins_opponent_identity: bool = False

    def __post_init__(self) -> None:
        if self.turn_model not in TURN_MODELS:
            raise ValueError(f"unknown turn_model {self.turn_model!r}")
        if self.actuation_model not in ACTUATION_MODELS:
            raise ValueError(f"unknown actuation_model {self.actuation_model!r}")
        if self.partial_roster_disposition not in PARTIAL_ROSTER_DISPOSITIONS:
            raise ValueError(
                f"unknown partial roster disposition {self.partial_roster_disposition!r}"
            )

    @property
    def opponent_instances(self) -> tuple[AgentInstanceFacts, ...]:
        return tuple(item for item in self.agent_instances if not item.trainable)

    @property
    def declares_minimum_roster(self) -> bool:
        return any(team.minimum_viable_roster > 0 for team in self.teams)


@dataclass(frozen=True, slots=True)
class HorizonFacts(JsonDataclassMixin):
    """A declared episode horizon. A step horizon carries no duration of its own."""

    horizon_kind: str = "wall_clock"
    value: float = 0.0
    time_dilation: float = 1.0
    grace_seconds: float = 0.0
    seconds_per_unit: float | None = None

    def __post_init__(self) -> None:
        if self.horizon_kind not in HORIZON_KINDS:
            raise ValueError(f"unknown horizon_kind {self.horizon_kind!r}")
        if self.value <= 0:
            raise ValueError("horizon value must be positive")
        if self.time_dilation <= 0:
            raise ValueError("time_dilation must be positive")
        if self.seconds_per_unit is not None and self.seconds_per_unit <= 0:
            raise ValueError("seconds_per_unit must be positive when declared")

    @property
    def seconds(self) -> float:
        """The horizon in seconds, or a refusal. Never a guess."""

        if self.horizon_kind == "wall_clock":
            return float(self.value)
        if self.seconds_per_unit is None:
            raise ValueError(
                f"a {self.horizon_kind} horizon must declare seconds_per_unit; "
                "a lease may not be guessed from a unit with no duration"
            )
        return float(self.value) * float(self.seconds_per_unit)


@dataclass(frozen=True, slots=True)
class RuntimeFacts(JsonDataclassMixin):
    """What this build can actually do, as it declares it.

    Nothing in this object names a task, a harness, or an environment, and
    every clause answer below is a function of these fields alone.
    """

    container_id: str
    container_image_digest: str
    renderer_profile: RendererProfileFacts
    discovery: DiscoveryFacts
    policy: PolicyFacts
    lifecycle: LifecycleFacts
    evidence: EvidenceFacts
    reward: RewardFacts
    recovery: RecoveryFacts
    topology: TopologyFacts
    horizon: HorizonFacts
    declared_routes: Mapping[str, str] = field(default_factory=dict)
    probe_binding: bool = False
    handshake_ttl_seconds: float = 900.0
    contract_version: str = CISPO_CONTRACT_VERSION
    container_version: str = ""
    #: When these facts were parsed out of an existing capability document, that
    #: document is the one that gets hashed and advertised. Reassembling it here
    #: would give the executor a second document to disagree with.
    source_document: Mapping[str, Any] | None = None

    # -- capability document ------------------------------------------- #

    def capability_document(self) -> dict[str, Any]:
        """The advertisement, in the schema the executor parses and hashes."""

        if self.source_document is not None:
            return dict(self.source_document)
        document: dict[str, Any] = {
            "schema_version": CAPABILITY_SCHEMA_VERSION,
            "container_id": self.container_id,
            "container_image_digest": self.container_image_digest,
            "contract_version": self.contract_version,
            "renderer_profile": self.renderer_profile.to_payload(),
            "discovery": {
                "taskset_id": self.discovery.taskset_id,
                "taskset_version": self.discovery.taskset_version,
                "splits": list(self.discovery.splits),
                "task_content_digests": bool(self.discovery.task_content_digests),
                "deterministic_lookup": bool(self.discovery.deterministic_lookup),
                "duplicate_free": bool(self.discovery.duplicate_free),
            },
            "policy": {
                "binding_transport": self.policy.binding_transport,
                "wire_api": self.policy.wire_api,
                "session_scoped_sampler_origin": bool(
                    self.policy.session_scoped_sampler_origin
                ),
                "embeds_credentials": bool(self.policy.embeds_credentials),
                "revision_immutable_after_admission": bool(
                    self.policy.revision_immutable_after_admission
                ),
                "records_policy_revision": bool(self.policy.records_policy_revision),
            },
            "lifecycle": {
                "max_concurrency": int(self.lifecycle.max_concurrency),
                "lease_ttl_seconds": float(self.lifecycle.lease_ttl_seconds),
                "supports_idempotency": bool(self.lifecycle.supports_idempotency),
                "supports_cancellation": bool(self.lifecycle.supports_cancellation),
                "supports_lease_renewal": bool(self.lifecycle.lease_renewable),
                "exactly_one_terminal_result": bool(
                    self.lifecycle.exactly_one_terminal_result
                ),
                "supports_pause_resume": bool(self.lifecycle.supports_pause_resume),
                "straggler_grace_seconds": float(self.lifecycle.straggler_grace_seconds),
            },
            "evidence": {
                "trace_v5": bool(self.evidence.trace_v5),
                "behavior_logprobs": bool(self.evidence.behavior_logprobs),
                "strict_prefix": bool(self.evidence.strict_prefix),
                "masking": bool(self.evidence.masking),
                "wire_objects": bool(self.evidence.wire_objects),
                "artifact_reference": bool(self.evidence.artifact_reference),
                "tokens_in_tokens_out": bool(self.evidence.tokens_in_tokens_out),
                "probe_binding": bool(self.probe_binding),
            },
            "reward": {
                "authority": self.reward.authority,
                "binds_trace_digest": bool(self.reward.binds_trace_digest),
                "quiescence": bool(self.reward.quiescence),
                "horizon_clipping": bool(self.reward.horizon_clipping),
                "channels": list(self.reward.channels),
                "reward_relation": self.reward.reward_relation,
                "evaluation_plan_id": self.reward.evaluation_plan_id,
                "settlement_window_seconds": float(self.reward.settlement_window_seconds),
                "deferred_scoring": bool(self.reward.deferred_scoring),
            },
            "recovery": {
                "restart": bool(self.recovery.restart),
                "stale_discard": bool(self.recovery.stale_discard),
            },
            "topology": {
                "topology_id": self.topology.topology_id,
                "turn_model": self.topology.turn_model,
                "actuation_model": self.topology.actuation_model,
                "reward_relation": self.reward.reward_relation,
                "agent_instances": [
                    {
                        "agent_instance_id": item.agent_instance_id,
                        "role_id": item.role_id,
                        "policy_type_id": item.policy_type_id,
                        "team_id": item.team_id,
                        "trainable": bool(item.trainable),
                        "pinned_identity": item.pinned_identity,
                    }
                    for item in self.topology.agent_instances
                ],
                "teams": [
                    {
                        "team_id": item.team_id,
                        "trainable": bool(item.trainable),
                        "minimum_viable_roster": int(item.minimum_viable_roster),
                    }
                    for item in self.topology.teams
                ],
                "communication_channels": [
                    {
                        "channel_id": item.channel_id,
                        "scope": item.scope,
                        "trainable_for_author": bool(item.trainable_for_author),
                    }
                    for item in self.topology.communication_channels
                ],
                "parameter_groups": dict(self.topology.parameter_groups),
                "partial_roster": self.topology.partial_roster_disposition,
                "horizon": {
                    "horizon_kind": self.horizon.horizon_kind,
                    "value_seconds": float(self.horizon.value),
                    "time_dilation": float(self.horizon.time_dilation),
                    "grace_seconds": float(self.horizon.grace_seconds),
                    "seconds_per_unit": self.horizon.seconds_per_unit,
                },
            },
            "routes": dict(sorted(self.declared_routes.items())),
            "clock": {"skew_tolerance_seconds": float(self.lifecycle.clock_skew_tolerance_seconds)},
        }
        document["capability_hash"] = canonical_capability_hash(document)
        return document

    @property
    def capability_hash(self) -> str:
        return canonical_capability_hash(self.capability_document())

    @property
    def horizon_seconds(self) -> float:
        """The declared horizon in seconds; ``0.0`` where no conversion exists."""

        try:
            return self.horizon.seconds
        except ValueError:
            return 0.0

    # -- derivation from an already-assembled capability document ------- #

    @classmethod
    def from_capability_document(
        cls,
        document: Mapping[str, Any],
        *,
        task_rows: Sequence[TaskRow] = (),
        declared_routes: Mapping[str, str] | None = None,
        handshake_ttl_seconds: float = 900.0,
        container_version: str = "",
    ) -> "RuntimeFacts":
        """Derive the clause inputs from a capability document already assembled.

        The document stays authoritative: it is what gets hashed and what the
        executor compared its preflight against, so it is carried through
        verbatim rather than reassembled. ``task_rows`` come from the taskset
        rather than the document, which declares only that per-row digests
        exist.
        """

        def block(name: str) -> Mapping[str, Any]:
            value = document.get(name)
            if not isinstance(value, Mapping):
                raise MalformedRequirementDocument(
                    f"capability document field {name!r} must be an object"
                )
            return value

        discovery_raw = block("discovery")
        policy_raw = block("policy")
        lifecycle_raw = block("lifecycle")
        evidence_raw = block("evidence")
        reward_raw = block("reward")
        recovery_raw = block("recovery")
        topology_raw = block("topology")
        clock_raw = document.get("clock")
        clock = clock_raw if isinstance(clock_raw, Mapping) else {}
        routes = document.get("routes")
        renderer_raw = block("renderer_profile")
        probe = bool(
            policy_raw.get("probe_binding", evidence_raw.get("probe_binding", False))
        )
        return cls(
            container_id=str(document.get("container_id") or ""),
            container_image_digest=str(document.get("container_image_digest") or ""),
            renderer_profile=RendererProfileFacts(
                profile_id=str(renderer_raw.get("profile_id") or ""),
                package=str(renderer_raw.get("package") or ""),
                package_version=str(renderer_raw.get("package_version") or ""),
                config_digest=str(renderer_raw.get("config_digest") or ""),
                tokenizer_id=str(renderer_raw.get("tokenizer_id") or ""),
                tokenizer_digest=str(renderer_raw.get("tokenizer_digest") or ""),
                stop_token_ids=tuple(
                    int(item) for item in renderer_raw.get("stop_token_ids") or ()
                ),
                modalities=tuple(
                    str(item) for item in renderer_raw.get("modalities") or ("text",)
                ),
                add_generation_prompt=bool(renderer_raw.get("add_generation_prompt", True)),
            ),
            discovery=DiscoveryFacts(
                taskset_id=str(discovery_raw.get("taskset_id") or ""),
                taskset_version=str(discovery_raw.get("taskset_version") or ""),
                splits=tuple(str(item) for item in discovery_raw.get("splits") or ()),
                rows=tuple(task_rows),
                task_content_digests=bool(discovery_raw.get("task_content_digests", False)),
                deterministic_lookup=bool(discovery_raw.get("deterministic_lookup", False)),
                duplicate_free=bool(discovery_raw.get("duplicate_free", False)),
            ),
            policy=PolicyFacts(
                binding_transport=str(
                    policy_raw.get("binding_transport") or "message_in_capture_out"
                ),
                wire_api=str(policy_raw.get("wire_api") or "chat_completions"),
                tokens_in_tokens_out=bool(evidence_raw.get("tokens_in_tokens_out", False)),
                session_scoped_sampler_origin=bool(
                    policy_raw.get("session_scoped_sampler_origin", False)
                ),
                embeds_credentials=bool(policy_raw.get("embeds_credentials", True)),
                revision_immutable_after_admission=bool(
                    policy_raw.get("revision_immutable_after_admission", False)
                ),
                records_policy_revision=bool(policy_raw.get("records_policy_revision", False)),
            ),
            lifecycle=LifecycleFacts(
                max_concurrency=int(lifecycle_raw.get("max_concurrency") or 1),
                lease_ttl_seconds=float(lifecycle_raw.get("lease_ttl_seconds") or 0.0),
                lease_renewable=bool(lifecycle_raw.get("supports_lease_renewal", False)),
                supports_idempotency=bool(lifecycle_raw.get("supports_idempotency", False)),
                supports_cancellation=bool(lifecycle_raw.get("supports_cancellation", False)),
                exactly_one_terminal_result=bool(
                    lifecycle_raw.get("exactly_one_terminal_result", False)
                ),
                supports_pause_resume=bool(lifecycle_raw.get("supports_pause_resume", False)),
                straggler_grace_seconds=float(
                    lifecycle_raw.get("straggler_grace_seconds") or 0.0
                ),
                clock_skew_tolerance_seconds=float(
                    clock.get("skew_tolerance_seconds") or 0.0
                ),
            ),
            evidence=EvidenceFacts(
                trace_v5=bool(evidence_raw.get("trace_v5", False)),
                behavior_logprobs=bool(evidence_raw.get("behavior_logprobs", False)),
                strict_prefix=bool(evidence_raw.get("strict_prefix", False)),
                masking=bool(evidence_raw.get("masking", False)),
                wire_objects=bool(evidence_raw.get("wire_objects", False)),
                artifact_reference=bool(evidence_raw.get("artifact_reference", False)),
                # A container that does not serve artifacts by reference serves
                # them inline; that is the declared substitute, not a guess.
                inline_evidence=not bool(evidence_raw.get("artifact_reference", False)),
                tokens_in_tokens_out=bool(evidence_raw.get("tokens_in_tokens_out", False)),
            ),
            reward=RewardFacts(
                authority=str(reward_raw.get("authority") or ""),
                evaluation_plan_id=str(reward_raw.get("evaluation_plan_id") or ""),
                reward_relation=str(reward_raw.get("reward_relation") or "cooperative"),
                channels=tuple(str(item) for item in reward_raw.get("channels") or ()),
                binds_trace_digest=bool(reward_raw.get("binds_trace_digest", False)),
                quiescence=bool(reward_raw.get("quiescence", False)),
                horizon_clipping=bool(reward_raw.get("horizon_clipping", False)),
                settlement_window_seconds=float(
                    reward_raw.get("settlement_window_seconds") or 0.0
                ),
                deferred_scoring=bool(reward_raw.get("deferred_scoring", False)),
            ),
            recovery=RecoveryFacts(
                restart=bool(recovery_raw.get("restart", False)),
                stale_discard=bool(recovery_raw.get("stale_discard", False)),
            ),
            topology=TopologyFacts(
                topology_id=str(topology_raw.get("topology_id") or ""),
                turn_model=str(topology_raw.get("turn_model") or "sequential"),
                actuation_model=str(topology_raw.get("actuation_model") or "direct_action"),
                agent_instances=tuple(
                    AgentInstanceFacts(
                        agent_instance_id=str(item.get("agent_instance_id") or ""),
                        role_id=str(item.get("role_id") or ""),
                        policy_type_id=str(item.get("policy_type_id") or ""),
                        team_id=str(item.get("team_id") or ""),
                        trainable=bool(item.get("trainable", True)),
                        pinned_identity=(
                            str(item["pinned_identity"]) if item.get("pinned_identity") else None
                        ),
                    )
                    for item in topology_raw.get("agent_instances") or ()
                    if isinstance(item, Mapping)
                ),
                teams=tuple(
                    TeamFacts(
                        team_id=str(item.get("team_id") or ""),
                        trainable=bool(item.get("trainable", True)),
                        minimum_viable_roster=int(item.get("minimum_viable_roster") or 0),
                    )
                    for item in topology_raw.get("teams") or ()
                    if isinstance(item, Mapping)
                ),
                communication_channels=tuple(
                    ChannelFacts(
                        channel_id=str(item.get("channel_id") or ""),
                        scope=str(item.get("scope") or ""),
                        trainable_for_author=bool(item.get("trainable_for_author", True)),
                    )
                    for item in topology_raw.get("communication_channels") or ()
                    if isinstance(item, Mapping)
                ),
                parameter_groups={
                    str(key): str(value)
                    for key, value in (topology_raw.get("parameter_groups") or {}).items()
                },
                partial_roster_disposition=str(
                    topology_raw.get("partial_roster") or "refuse"
                ),
                # A non-trainable instance is only pinned if it declares an
                # immutable identity; an alias is not an identity.
                pins_opponent_identity=all(
                    bool(item.get("pinned_identity"))
                    for item in topology_raw.get("agent_instances") or ()
                    if isinstance(item, Mapping) and not item.get("trainable", True)
                ),
            ),
            horizon=_capability_horizon(document),
            declared_routes=dict(
                declared_routes
                if declared_routes is not None
                else (routes if isinstance(routes, Mapping) else {})
            ),
            probe_binding=probe,
            handshake_ttl_seconds=handshake_ttl_seconds,
            contract_version=str(document.get("contract_version") or CISPO_CONTRACT_VERSION),
            container_version=container_version,
            source_document=dict(document),
        )

    # -- derivation from the repo's own runtime surface ----------------- #

    @classmethod
    def from_runtime_surface(
        cls,
        surface: RuntimeCapabilitySurface,
        *,
        container_id: str,
        container_image_digest: str,
        renderer_profile: RendererProfileFacts,
        discovery: DiscoveryFacts,
        policy: PolicyFacts,
        lifecycle: LifecycleFacts,
        evidence: EvidenceFacts,
        reward: RewardFacts,
        recovery: RecoveryFacts,
        topology: TopologyFacts,
        horizon: HorizonFacts,
        declared_routes: Mapping[str, str] | None = None,
        probe_binding: bool = False,
        handshake_ttl_seconds: float = 900.0,
        contract_version: str = CISPO_CONTRACT_VERSION,
    ) -> "RuntimeFacts":
        """Intersect the declarations with what the runtime surface admits.

        The intersection only ever removes a promise. A build may declare less
        than its surface allows (a feature not wired up yet), but it may never
        promise a capability the surface says it does not have -- that is the
        fail-closed direction, and it is why no clause answer needs a table of
        what "usually" works.
        """

        emission = surface.token_emission
        evidence = replace(
            evidence,
            trace_v5=bool(evidence.trace_v5 and surface.trace_support),
            behavior_logprobs=bool(evidence.behavior_logprobs and emission.logprobs),
            wire_objects=bool(evidence.wire_objects and surface.proxied_inference),
            artifact_reference=bool(evidence.artifact_reference and surface.artifact_support),
            tokens_in_tokens_out=bool(evidence.tokens_in_tokens_out and emission.token_ids),
        )
        policy = replace(
            policy,
            tokens_in_tokens_out=bool(policy.tokens_in_tokens_out and emission.token_ids),
        )
        lifecycle = replace(
            lifecycle,
            supports_cancellation=bool(
                lifecycle.supports_cancellation and surface.terminate_support
            ),
            supports_pause_resume=bool(
                lifecycle.supports_pause_resume
                and surface.pause_support
                and surface.resume_support
            ),
        )
        reward = replace(
            reward,
            channels=tuple(reward.channels) if surface.reward_support else (),
            binds_trace_digest=bool(reward.binds_trace_digest and surface.trace_support),
        )
        recovery = replace(
            recovery, restart=bool(recovery.restart and surface.resume_support)
        )
        topology = replace(
            topology,
            communication_channels=(
                topology.communication_channels if surface.multi_actor else ()
            ),
        )
        routes = dict(declared_routes or {})
        if not routes:
            routes = _routes_from_hints(surface)
        return cls(
            container_id=container_id,
            container_image_digest=container_image_digest,
            renderer_profile=renderer_profile,
            discovery=discovery,
            policy=policy,
            lifecycle=lifecycle,
            evidence=evidence,
            reward=reward,
            recovery=recovery,
            topology=topology,
            horizon=horizon,
            declared_routes=routes,
            probe_binding=probe_binding,
            handshake_ttl_seconds=handshake_ttl_seconds,
            contract_version=contract_version,
        )


def _capability_horizon(document: Mapping[str, Any]) -> HorizonFacts:
    """Read the declared horizon from wherever the document carries it.

    ``value_seconds`` is the key the optimizer's parser reads; ``value`` and
    ``seconds_per_unit`` say what the number actually means for a step or tick
    horizon, where the two are not the same.
    """

    topology = document.get("topology")
    raw = topology.get("horizon") if isinstance(topology, Mapping) else None
    if not isinstance(raw, Mapping):
        raw = document.get("horizon")
    if not isinstance(raw, Mapping):
        raise MalformedRequirementDocument("capability document declares no horizon")
    seconds_per_unit = raw.get("seconds_per_unit")
    kind = str(raw.get("horizon_kind") or "")
    return HorizonFacts(
        horizon_kind=kind,
        value=float(raw.get("value", raw.get("value_seconds", 0.0)) or 0.0),
        time_dilation=float(raw.get("time_dilation", 1.0) or 1.0),
        grace_seconds=float(raw.get("grace_seconds", 0.0) or 0.0),
        seconds_per_unit=(
            None
            if seconds_per_unit is None or kind == "wall_clock"
            else float(seconds_per_unit)
        ),
    )


def _routes_from_hints(surface: RuntimeCapabilitySurface) -> dict[str, str]:
    """Name the declared routes from the surface's own route hints.

    Route presence is a declaration, so a hint list that names no route leaves
    the route name absent and ``contract.routes`` rejects -- rather than the
    module inventing a path the container does not actually serve.
    """

    hints = surface.route_hints

    def first(values: Sequence[str]) -> str:
        return str(values[0]) if values else ""

    candidates = {
        "capabilities_route": first(hints.compatibility_routes),
        "taskset_route": first(hints.taskset_routes),
        "taskset_tasks_route": first(hints.taskset_routes[1:]),
        "rollout_route": first(hints.rollout_routes),
        "rollout_state_route": first(hints.state_routes),
        "rollout_events_route": first(hints.event_routes),
        "rollout_terminate_route": first(hints.terminate_routes),
        "trace_route": first(hints.trace_routes),
        "artifacts_route": first(hints.artifact_routes),
    }
    return {name: route for name, route in candidates.items() if route}


# --------------------------------------------------------------------------- #
# The requirement document
# --------------------------------------------------------------------------- #


def _mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise MalformedRequirementDocument(f"requirement document field {name!r} must be an object")
    return value


def _text(payload: Mapping[str, Any], name: str, *, where: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise MalformedRequirementDocument(f"{where}.{name} is required")
    return value.strip()


def _int(payload: Mapping[str, Any], name: str, *, where: str, default: int | None = None) -> int:
    value = payload.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise MalformedRequirementDocument(f"{where}.{name} must be an integer")
    return value


def _float(
    payload: Mapping[str, Any],
    name: str,
    *,
    where: str,
    default: float | None = None,
) -> float:
    value = payload.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MalformedRequirementDocument(f"{where}.{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise MalformedRequirementDocument(f"{where}.{name} must be finite")
    return number


@dataclass(frozen=True, slots=True)
class RunPlanRequest(JsonDataclassMixin):
    group_size: int
    groups_per_step: int
    max_execution_slots: int
    maximum_policy_lag: int
    target_train_updates: int
    expected_horizon_seconds: float


@dataclass(frozen=True, slots=True)
class TasksetRequest(JsonDataclassMixin):
    taskset_id: str
    split: str
    task_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HandshakeRequest(JsonDataclassMixin):
    """The executor's requirement document, as received, plus its raw payload.

    ``raw`` is what the agreement digest binds. The executor digests
    ``HandshakeRequest.to_payload()``; keeping the bytes we were handed is the
    only way the two computations can agree.
    """

    run_id: str
    attempt: int
    policy_transport: str
    renderer_profile: Mapping[str, Any]
    requirements: tuple[str, ...]
    accept_degraded: Mapping[str, str]
    expected_topology_id: str
    trainable_teams: tuple[str, ...]
    partial_roster: str
    run_plan: RunPlanRequest
    taskset: TasksetRequest
    executor_time: datetime
    renew_of: str = ""
    schema_version: str = HANDSHAKE_SCHEMA_VERSION
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> "HandshakeRequest":
        if not isinstance(payload, Mapping):
            raise MalformedRequirementDocument("requirement document must be an object")
        schema = payload.get("schema_version")
        if schema != HANDSHAKE_SCHEMA_VERSION:
            raise MalformedRequirementDocument(
                f"handshake schema {schema!r} is unsupported; expected "
                f"{HANDSHAKE_SCHEMA_VERSION}"
            )
        requirements_raw = payload.get("requirements") or ()
        if not isinstance(requirements_raw, Sequence) or isinstance(requirements_raw, (str, bytes)):
            raise MalformedRequirementDocument("requirements must be a list of clause ids")
        requirements = tuple(str(item) for item in requirements_raw)
        unknown = tuple(item for item in requirements if item not in ALL_CLAUSES)
        if unknown:
            raise MalformedRequirementDocument(
                f"requirement document names unknown clauses: {unknown}"
            )
        accept_raw = payload.get("accept_degraded") or {}
        if not isinstance(accept_raw, Mapping):
            raise MalformedRequirementDocument(
                "accept_degraded must be an object of clause id to substitute"
            )
        accept_degraded = {str(key): str(value) for key, value in accept_raw.items()}
        undeclared = tuple(
            clause for clause in accept_degraded if clause not in CLAUSE_SUBSTITUTES
        )
        if undeclared:
            raise UndeclaredSubstitute(
                f"accept_degraded names clauses with no declared substitute: {undeclared}"
            )
        wrong = tuple(
            f"{clause}={substitute!r}"
            for clause, substitute in accept_degraded.items()
            if substitute not in CLAUSE_SUBSTITUTES[clause]
        )
        if wrong:
            raise UndeclaredSubstitute(
                f"accept_degraded names substitutes nobody declared: {wrong}"
            )
        policy_raw = _mapping(payload, "policy")
        topology_raw = _mapping(payload, "topology")
        plan_raw = _mapping(payload, "run_plan")
        taskset_raw = _mapping(payload, "taskset")
        clock_raw = _mapping(payload, "clock")
        task_ids_raw = taskset_raw.get("task_ids") or ()
        if not isinstance(task_ids_raw, Sequence) or isinstance(task_ids_raw, (str, bytes)):
            raise MalformedRequirementDocument("taskset.task_ids must be a list")
        task_ids = tuple(str(item) for item in task_ids_raw)
        if not task_ids:
            raise MalformedRequirementDocument("taskset must name at least one task id")
        if len(set(task_ids)) != len(task_ids):
            raise MalformedRequirementDocument("taskset repeats a task id")
        partial_roster = str(topology_raw.get("partial_roster") or "refuse")
        if partial_roster not in PARTIAL_ROSTER_DISPOSITIONS:
            raise MalformedRequirementDocument(
                f"unknown partial roster disposition {partial_roster!r}"
            )
        teams_raw = topology_raw.get("trainable_teams") or ()
        if not isinstance(teams_raw, Sequence) or isinstance(teams_raw, (str, bytes)):
            raise MalformedRequirementDocument("topology.trainable_teams must be a list")
        renderer_raw = payload.get("renderer_profile") or {}
        if not isinstance(renderer_raw, Mapping):
            raise MalformedRequirementDocument("renderer_profile must be an object")
        return cls(
            run_id=_text(payload, "run_id", where="handshake"),
            attempt=_int(payload, "attempt", where="handshake", default=1),
            policy_transport=_text(policy_raw, "transport", where="policy"),
            renderer_profile=dict(renderer_raw),
            requirements=requirements,
            accept_degraded=accept_degraded,
            expected_topology_id=str(topology_raw.get("expected_topology_id") or ""),
            trainable_teams=tuple(str(item) for item in teams_raw),
            partial_roster=partial_roster,
            run_plan=RunPlanRequest(
                group_size=_int(plan_raw, "group_size", where="run_plan"),
                groups_per_step=_int(plan_raw, "groups_per_step", where="run_plan"),
                max_execution_slots=_int(plan_raw, "max_execution_slots", where="run_plan"),
                maximum_policy_lag=_int(
                    plan_raw, "maximum_policy_lag", where="run_plan", default=0
                ),
                target_train_updates=_int(
                    plan_raw, "target_train_updates", where="run_plan", default=1
                ),
                expected_horizon_seconds=_float(
                    plan_raw, "expected_horizon_seconds", where="run_plan"
                ),
            ),
            taskset=TasksetRequest(
                taskset_id=_text(taskset_raw, "taskset_id", where="taskset"),
                split=_text(taskset_raw, "split", where="taskset"),
                task_ids=task_ids,
            ),
            executor_time=parse_rfc3339(
                clock_raw.get("executor_time"), field_name="clock.executor_time"
            ),
            renew_of=str(payload.get("renew_of") or ""),
            schema_version=str(schema),
            raw=dict(payload),
        )

    @property
    def digest_payload(self) -> dict[str, Any]:
        """Exactly what the executor digests: its own ``to_payload()``.

        ``renew_of`` is a transport concern the executor's requirement document
        does not carry, so it is stripped before the digest rather than making
        a renewal disagree with the agreement it renews.
        """

        return {key: value for key, value in self.raw.items() if key != "renew_of"}

    @property
    def requested_renderer_fingerprint(self) -> str:
        return str(self.renderer_profile.get("fingerprint") or "")


# --------------------------------------------------------------------------- #
# Clause answers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ClauseAnswer(JsonDataclassMixin):
    """One clause, one verdict, one reason. Never a bare boolean."""

    clause_id: str
    verdict: str
    reason: str = ""
    substitute: str = ""

    def __post_init__(self) -> None:
        if self.clause_id not in ALL_CLAUSES:
            raise ValueError(f"unknown clause {self.clause_id!r}")
        if self.verdict not in VERDICTS:
            raise ValueError(f"unknown verdict {self.verdict!r} for {self.clause_id}")
        if not self.reason.strip():
            raise ValueError(f"clause {self.clause_id} answered without a reason")
        if self.substitute and self.substitute not in substitutes_for(self.clause_id):
            raise ValueError(
                f"{self.clause_id} has no declared substitute {self.substitute!r}"
            )

    @property
    def mandatory(self) -> bool:
        return self.clause_id not in OPTIONAL_CLAUSES

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "clause_id": self.clause_id,
            "verdict": self.verdict,
            "reason": self.reason,
        }
        if self.substitute:
            payload["substitute"] = self.substitute
        return payload


def _answer(
    clause_id: str,
    ok: bool,
    *,
    reason: str,
    substitute: str = "",
    substitute_available: bool = False,
) -> ClauseAnswer:
    """The one place a verdict is chosen, so the rule is uniform.

    Satisfied outright is ``accepted``. Not satisfied but answerable by a
    declared substitute this build actually offers is ``degraded`` naming that
    substitute -- neither a rejection, which stops the run, nor a silent pass.
    Not satisfied and optional is ``unsupported``; the run records its fallback.
    Anything else is ``rejected``, which stops the run before spend.
    """

    if ok:
        return ClauseAnswer(clause_id, "accepted", reason)
    if substitute and substitute_available:
        return ClauseAnswer(clause_id, "degraded", reason, substitute=substitute)
    if clause_id in OPTIONAL_CLAUSES:
        return ClauseAnswer(clause_id, "unsupported", reason)
    return ClauseAnswer(clause_id, "rejected", reason)


@dataclass(frozen=True, slots=True)
class _Context:
    facts: RuntimeFacts
    request: HandshakeRequest
    measured_skew_seconds: float


def _contract_version(ctx: _Context) -> ClauseAnswer:
    declared = ctx.facts.contract_version
    ok = declared == CISPO_CONTRACT_VERSION
    return _answer(
        "contract.version",
        ok,
        reason=(
            f"container declares contract {declared!r} and speaks handshake schema "
            f"{HANDSHAKE_SCHEMA_VERSION}"
            if ok
            else f"container declares contract {declared!r}, not {CISPO_CONTRACT_VERSION!r}"
        ),
    )


def _contract_routes(ctx: _Context) -> ClauseAnswer:
    declared = dict(ctx.facts.declared_routes)
    missing = tuple(
        name
        for name in MANDATORY_ROUTE_NAMES
        if not str(declared.get(name) or "").strip().startswith("/")
    )
    return _answer(
        "contract.routes",
        not missing,
        reason=(
            f"all {len(MANDATORY_ROUTE_NAMES)} contract routes are declared"
            if not missing
            else "route presence and behavior support are separate questions; "
            f"undeclared routes: {missing}"
        ),
    )


def _discovery_taskset(ctx: _Context) -> ClauseAnswer:
    facts = ctx.facts.discovery
    wanted = ctx.request.taskset
    reasons: list[str] = []
    if not facts.taskset_id:
        reasons.append("container declares no taskset id")
    elif wanted.taskset_id and wanted.taskset_id != facts.taskset_id:
        reasons.append(
            f"requested taskset {wanted.taskset_id!r} is not the declared "
            f"{facts.taskset_id!r}"
        )
    if wanted.split not in facts.splits:
        reasons.append(f"split {wanted.split!r} is not among declared splits {facts.splits}")
    if not facts.deterministic_lookup:
        reasons.append("lookup by task id is not declared deterministic")
    if not facts.duplicate_free:
        reasons.append("taskset responses are not declared duplicate-free")
    return _answer(
        "discovery.taskset",
        not reasons,
        reason=(
            f"taskset {facts.taskset_id}@{facts.taskset_version} serves split "
            f"{wanted.split!r} deterministically and duplicate-free"
            if not reasons
            else "; ".join(reasons)
        ),
    )


def _discovery_task_digests(ctx: _Context) -> ClauseAnswer:
    facts = ctx.facts.discovery
    if not facts.task_content_digests:
        return _answer(
            "discovery.task_digests",
            False,
            reason="container does not declare a content digest per task row",
        )
    missing = tuple(
        task_id for task_id in ctx.request.taskset.task_ids if facts.row(task_id) is None
    )
    return _answer(
        "discovery.task_digests",
        not missing,
        reason=(
            f"{len(ctx.request.taskset.task_ids)} requested task ids resolve to a content digest"
            if not missing
            else f"container cannot resolve requested task ids: {missing}"
        ),
    )


def _discovery_topology(ctx: _Context) -> ClauseAnswer:
    declared = ctx.facts.topology.topology_id
    wanted = ctx.request.expected_topology_id
    ok = bool(declared) and (not wanted or wanted == declared)
    return _answer(
        "discovery.topology",
        ok,
        reason=(
            f"container declares topology {declared!r}"
            if ok
            else (
                "container declares no topology id"
                if not declared
                else f"executor expects topology {wanted!r}, container declares {declared!r}"
            )
        ),
    )


def _policy_binding_transport(ctx: _Context) -> ClauseAnswer:
    supported = ctx.facts.policy.supported_transports
    wanted = ctx.request.policy_transport
    ok = wanted in supported
    return _answer(
        "policy.binding_transport",
        ok,
        reason=(
            f"container binds {wanted!r} over wire {ctx.facts.policy.wire_api!r}"
            if ok
            else f"container declares transports {supported}, executor asked for {wanted!r}"
        ),
    )


def _policy_renderer_profile_match(ctx: _Context) -> ClauseAnswer:
    declared = ctx.facts.renderer_profile
    wanted = ctx.request.renderer_profile
    fingerprint = ctx.request.requested_renderer_fingerprint
    if fingerprint:
        ok = fingerprint == declared.fingerprint
        return _answer(
            "policy.renderer_profile_match",
            ok,
            reason=(
                f"renderer fingerprint {declared.fingerprint} matches the training session"
                if ok
                else "renderer profile mismatch: container "
                f"{declared.fingerprint} != session {fingerprint}"
            ),
        )
    payload = declared.to_payload()
    mismatched = tuple(
        sorted(name for name, value in wanted.items() if name in payload and payload[name] != value)
    )
    return _answer(
        "policy.renderer_profile_match",
        not mismatched,
        reason=(
            f"renderer profile agrees on every field the executor pinned "
            f"({declared.fingerprint})"
            if not mismatched
            else f"renderer profile differs on {mismatched}"
        ),
    )


def _policy_revision_immutability(ctx: _Context) -> ClauseAnswer:
    facts = ctx.facts.policy
    ok = facts.revision_immutable_after_admission and facts.records_policy_revision
    return _answer(
        "policy.revision_immutability",
        ok,
        reason=(
            "behavior-policy revision is recorded on every trainable call and frozen "
            "at admission"
            if ok
            else "container declares revision_immutable_after_admission="
            f"{facts.revision_immutable_after_admission}, records_policy_revision="
            f"{facts.records_policy_revision}"
        ),
    )


def _policy_no_embedded_credentials(ctx: _Context) -> ClauseAnswer:
    embeds = ctx.facts.policy.embeds_credentials
    return _answer(
        "policy.no_embedded_credentials",
        not embeds,
        reason=(
            "rollout requests carry no raw sampler credential"
            if not embeds
            else "container declares it embeds raw credentials in rollout requests"
        ),
    )


def _policy_session_scoped_origin(ctx: _Context) -> ClauseAnswer:
    scoped = ctx.facts.policy.session_scoped_sampler_origin
    return _answer(
        "policy.session_scoped_origin",
        scoped,
        reason=(
            "sampler origin carries the per-attempt request identity in its path"
            if scoped
            else "container declares a global sampler origin; a leaked credential "
            "would cross rollouts"
        ),
    )


def _lifecycle_idempotency(ctx: _Context) -> ClauseAnswer:
    ok = ctx.facts.lifecycle.supports_idempotency
    return _answer(
        "lifecycle.idempotency",
        ok,
        reason=(
            "a resubmitted attempt key yields the same logical attempt"
            if ok
            else "container does not declare idempotent rollout/attempt ids"
        ),
    )


def _lifecycle_lease_renewal(ctx: _Context) -> ClauseAnswer:
    facts = ctx.facts.lifecycle
    horizon_seconds = ctx.facts.horizon_seconds
    requested = ctx.request.run_plan.expected_horizon_seconds
    if not facts.lease_renewable:
        covers = facts.lease_ttl_seconds >= max(horizon_seconds, requested)
        return _answer(
            "lifecycle.lease_renewal",
            covers,
            reason=(
                f"lease ttl {facts.lease_ttl_seconds}s is not renewable but already covers "
                f"the {max(horizon_seconds, requested)}s horizon"
                if covers
                else f"lease ttl {facts.lease_ttl_seconds}s cannot be extended and is shorter "
                f"than the {max(horizon_seconds, requested)}s horizon; an hour-scale episode "
                "may not depend on an HTTP request staying open"
            ),
        )
    if horizon_seconds and requested > horizon_seconds:
        return ClauseAnswer(
            "lifecycle.lease_renewal",
            "degraded",
            f"leases renew by heartbeat every {facts.lease_ttl_seconds}s but the declared "
            f"horizon is {horizon_seconds}s and the run plan expects {requested}s; "
            "lower the plan horizon and re-handshake",
        )
    return _answer(
        "lifecycle.lease_renewal",
        True,
        reason=(
            f"leases of {facts.lease_ttl_seconds}s renew by heartbeat for the declared "
            f"{ctx.facts.horizon.horizon_kind} horizon"
        ),
    )


def _lifecycle_cancellation(ctx: _Context) -> ClauseAnswer:
    ok = ctx.facts.lifecycle.supports_cancellation
    return _answer(
        "lifecycle.cancellation",
        ok,
        reason=(
            "an accepted attempt can be cancelled and reports a terminal cancellation"
            if ok
            else "container does not declare cancellation"
        ),
    )


def _lifecycle_concurrency(ctx: _Context) -> ClauseAnswer:
    available = ctx.facts.lifecycle.max_concurrency
    plan = ctx.request.run_plan
    requested = max(plan.max_execution_slots, plan.group_size * plan.groups_per_step)
    if requested <= available:
        return _answer(
            "lifecycle.concurrency",
            True,
            reason=f"{available} leases available, {requested} requested in flight",
        )
    return ClauseAnswer(
        "lifecycle.concurrency",
        "degraded",
        f"{available} leases available, {requested} requested in flight "
        f"(group_size {plan.group_size} x groups_per_step {plan.groups_per_step}, "
        f"slots {plan.max_execution_slots}) exceeds the pool",
    )


def _lifecycle_exactly_one_terminal(ctx: _Context) -> ClauseAnswer:
    ok = ctx.facts.lifecycle.exactly_one_terminal_result
    return _answer(
        "lifecycle.exactly_one_terminal",
        ok,
        reason=(
            "every accepted attempt reaches exactly one of episode, failure, cancellation"
            if ok
            else "container does not declare exactly one terminal result per attempt"
        ),
    )


def _lifecycle_pause_resume(ctx: _Context) -> ClauseAnswer:
    facts = ctx.facts.lifecycle
    return _answer(
        "lifecycle.pause_resume",
        facts.supports_pause_resume,
        reason=(
            "an attempt can be paused and resumed in place"
            if facts.supports_pause_resume
            else "container cannot pause an attempt in place"
        ),
        substitute="cancel_and_replace",
        substitute_available=facts.supports_cancellation,
    )


def _lifecycle_clock_skew(ctx: _Context) -> ClauseAnswer:
    tolerance = ctx.facts.lifecycle.clock_skew_tolerance_seconds
    skew = abs(ctx.measured_skew_seconds)
    ok = skew <= tolerance
    return _answer(
        "lifecycle.clock_skew",
        ok,
        reason=(
            f"measured skew {skew}s is within the declared tolerance {tolerance}s "
            "for a wall-clock horizon"
            if ok
            else f"measured skew {skew}s exceeds the declared tolerance {tolerance}s "
            "and the horizon is the instant the reward is read"
        ),
    )


def _evidence_trace_v5(ctx: _Context) -> ClauseAnswer:
    ok = ctx.facts.evidence.trace_v5
    return _answer(
        "evidence.trace_v5",
        ok,
        reason=(
            "container seals a trace v5 record per attempt"
            if ok
            else "container does not declare trace v5 capture"
        ),
    )


def _evidence_behavior_logprobs(ctx: _Context) -> ClauseAnswer:
    ok = ctx.facts.evidence.behavior_logprobs
    return _answer(
        "evidence.behavior_logprobs",
        ok,
        reason=(
            "per-token behavior logprobs come back from the sampling forward pass"
            if ok
            else "container does not declare per-token behavior logprobs; a later "
            "recomputation is not evidence"
        ),
    )


def _evidence_strict_prefix(ctx: _Context) -> ClauseAnswer:
    ok = ctx.facts.evidence.strict_prefix
    return _answer(
        "evidence.strict_prefix",
        ok,
        reason=(
            "turns stitch only on a byte-for-byte token prefix; anything else forks "
            "a branch and seals the prior segment"
            if ok
            else "container does not declare strict-prefix stitching"
        ),
    )


def _evidence_masking(ctx: _Context) -> ClauseAnswer:
    ok = ctx.facts.evidence.masking
    return _answer(
        "evidence.masking",
        ok,
        reason=(
            "loss masks are the renderer's sampled mask intersected with policy authorship"
            if ok
            else "container does not declare the fixed loss-mask convention"
        ),
    )


def _evidence_wire_objects(ctx: _Context) -> ClauseAnswer:
    ok = ctx.facts.evidence.wire_objects
    return _answer(
        "evidence.wire_objects",
        ok,
        reason=(
            "original wire objects are persisted alongside the token evidence"
            if ok
            else "container does not persist the original wire objects; the wire object "
            "is the semantic record and the tokens are the training record"
        ),
    )


def _evidence_artifact_reference(ctx: _Context) -> ClauseAnswer:
    facts = ctx.facts.evidence
    return _answer(
        "evidence.artifact_reference",
        facts.artifact_reference,
        reason=(
            "large evidence is served by reference"
            if facts.artifact_reference
            else "container serves evidence inline only"
        ),
        substitute="inline_evidence_only",
        substitute_available=facts.inline_evidence,
    )


def _evidence_tito(ctx: _Context) -> ClauseAnswer:
    ok = ctx.facts.evidence.tokens_in_tokens_out
    return _answer(
        "evidence.tito",
        ok,
        reason=(
            "container sends prompt token ids and receives generation token ids under "
            "the identical renderer profile"
            if ok
            else "container does not speak tokens in / tokens out"
        ),
    )


def _reward_authority(ctx: _Context) -> ClauseAnswer:
    facts = ctx.facts.reward
    ok = bool(facts.authority.strip()) and bool(facts.evaluation_plan_id.strip())
    return _answer(
        "reward.authority",
        ok,
        reason=(
            f"reward authority {facts.authority!r} under evaluation plan "
            f"{facts.evaluation_plan_id!r} is container-side"
            if ok
            else "container declares no reward authority or no evaluation plan; the "
            "executor never scores an episode itself"
        ),
    )


def _reward_binding_digest(ctx: _Context) -> ClauseAnswer:
    ok = ctx.facts.reward.binds_trace_digest
    return _answer(
        "reward.binding_digest",
        ok,
        reason=(
            "every reward record is bound to its rollout id and trace digest"
            if ok
            else "container does not bind reward to a trace digest"
        ),
    )


def _reward_horizon_quiescence(ctx: _Context) -> ClauseAnswer:
    facts = ctx.facts.reward
    return _answer(
        "reward.horizon_quiescence",
        facts.quiescence,
        reason=(
            "container stops every agent-authored background process at the horizon "
            "and attests no mutation before the scored read"
            if facts.quiescence
            else "container cannot kill agent-authored background processes; it serves "
            "a horizon-clipped state snapshot taken at the horizon instead"
        ),
        substitute="horizon_clipped_snapshot",
        substitute_available=facts.horizon_clipping,
    )


def _reward_settlement_window(ctx: _Context) -> ClauseAnswer:
    window = ctx.facts.reward.settlement_window_seconds
    ok = window > 0
    return _answer(
        "reward.settlement_window",
        ok,
        reason=(
            f"scored state may lag the horizon by up to {window}s"
            if ok
            else "scored state does not lag the horizon; the reward is a single read"
        ),
    )


def _reward_channels(ctx: _Context) -> ClauseAnswer:
    channels = ctx.facts.reward.channels
    return _answer(
        "reward.channels",
        bool(channels),
        reason=(
            f"container declares reward channels {tuple(channels)} under relation "
            f"{ctx.facts.reward.reward_relation!r}"
            if channels
            else "container declares no reward channel; an absent measure is not zero"
        ),
    )


def _recovery_restart(ctx: _Context) -> ClauseAnswer:
    ok = ctx.facts.recovery.restart
    return _answer(
        "recovery.restart",
        ok,
        reason=(
            "queued, active, scored and train-ready work survives a restart and is "
            "re-admitted only against the same agreement digest"
            if ok
            else "container does not declare restart recovery"
        ),
    )


def _recovery_stale_discard(ctx: _Context) -> ClauseAnswer:
    ok = ctx.facts.recovery.stale_discard
    return _answer(
        "recovery.stale_discard",
        ok,
        reason=(
            "work whose agreement no longer holds is discarded rather than resumed"
            if ok
            else "container does not declare stale-work discard"
        ),
    )


def _topology_roster(ctx: _Context) -> ClauseAnswer:
    facts = ctx.facts.topology
    instances = facts.agent_instances
    teams = {team.team_id for team in facts.teams}
    orphans = tuple(
        sorted(item.agent_instance_id for item in instances if item.team_id not in teams)
    )
    wanted = set(ctx.request.trainable_teams)
    trainable = {item.team_id for item in instances if item.trainable}
    unbound = tuple(sorted(wanted - trainable))
    reasons: list[str] = []
    if not instances:
        reasons.append("topology declares no agent instance")
    if orphans:
        reasons.append(f"instances name undeclared teams: {orphans}")
    if unbound:
        reasons.append(f"executor asked to train teams with no trainable instance: {unbound}")
    return _answer(
        "topology.roster",
        not reasons,
        reason=(
            f"{len(instances)} declared instances across {len(teams)} teams, "
            f"trainable teams {tuple(sorted(trainable))}"
            if not reasons
            else "; ".join(reasons)
        ),
    )


def _topology_channels(ctx: _Context) -> ClauseAnswer:
    channels = ctx.facts.topology.communication_channels
    return _answer(
        "topology.channels",
        bool(channels),
        reason=(
            f"container declares {len(channels)} communication channels with scopes "
            f"{tuple(sorted({item.scope for item in channels}))}"
            if channels
            else "container declares no communication channel, so channel completeness "
            "cannot be checked"
        ),
    )


def _topology_minimum_roster(ctx: _Context) -> ClauseAnswer:
    facts = ctx.facts.topology
    ok = facts.declares_minimum_roster
    return _answer(
        "topology.minimum_roster",
        ok,
        reason=(
            "each team declares a minimum viable roster, so a partial roster is "
            f"resolved by the declared disposition {facts.partial_roster_disposition!r}"
            if ok
            else "no team declares a minimum viable roster"
        ),
    )


def _topology_opponent_pinning(ctx: _Context) -> ClauseAnswer:
    facts = ctx.facts.topology
    opponents = facts.opponent_instances
    unpinned = tuple(
        sorted(item.agent_instance_id for item in opponents if not item.pinned_identity)
    )
    ok = bool(opponents) and facts.pins_opponent_identity and not unpinned
    return _answer(
        "topology.opponent_pinning",
        ok,
        reason=(
            f"{len(opponents)} non-trainable instances carry an immutable pinned identity"
            if ok
            else (
                "topology declares no non-trainable instance"
                if not opponents
                else f"non-trainable instances without an immutable identity: {unpinned}"
                if unpinned
                else "container does not declare opponent identity pinning"
            )
        ),
    )


CLAUSE_EVALUATORS: dict[str, Callable[[_Context], ClauseAnswer]] = {
    "contract.version": _contract_version,
    "contract.routes": _contract_routes,
    "discovery.taskset": _discovery_taskset,
    "discovery.task_digests": _discovery_task_digests,
    "discovery.topology": _discovery_topology,
    "policy.binding_transport": _policy_binding_transport,
    "policy.renderer_profile_match": _policy_renderer_profile_match,
    "policy.revision_immutability": _policy_revision_immutability,
    "policy.no_embedded_credentials": _policy_no_embedded_credentials,
    "policy.session_scoped_origin": _policy_session_scoped_origin,
    "lifecycle.idempotency": _lifecycle_idempotency,
    "lifecycle.lease_renewal": _lifecycle_lease_renewal,
    "lifecycle.cancellation": _lifecycle_cancellation,
    "lifecycle.concurrency": _lifecycle_concurrency,
    "lifecycle.exactly_one_terminal": _lifecycle_exactly_one_terminal,
    "lifecycle.pause_resume": _lifecycle_pause_resume,
    "lifecycle.clock_skew": _lifecycle_clock_skew,
    "evidence.trace_v5": _evidence_trace_v5,
    "evidence.behavior_logprobs": _evidence_behavior_logprobs,
    "evidence.strict_prefix": _evidence_strict_prefix,
    "evidence.masking": _evidence_masking,
    "evidence.wire_objects": _evidence_wire_objects,
    "evidence.artifact_reference": _evidence_artifact_reference,
    "evidence.tito": _evidence_tito,
    "reward.authority": _reward_authority,
    "reward.binding_digest": _reward_binding_digest,
    "reward.horizon_quiescence": _reward_horizon_quiescence,
    "reward.settlement_window": _reward_settlement_window,
    "reward.channels": _reward_channels,
    "recovery.restart": _recovery_restart,
    "recovery.stale_discard": _recovery_stale_discard,
    "topology.roster": _topology_roster,
    "topology.channels": _topology_channels,
    "topology.minimum_roster": _topology_minimum_roster,
    "topology.opponent_pinning": _topology_opponent_pinning,
}

if set(CLAUSE_EVALUATORS) != set(ALL_CLAUSES):  # pragma: no cover - import-time invariant
    raise RuntimeError(
        "every canonical clause needs an evaluator; missing "
        f"{sorted(set(ALL_CLAUSES) - set(CLAUSE_EVALUATORS))}, unknown "
        f"{sorted(set(CLAUSE_EVALUATORS) - set(ALL_CLAUSES))}"
    )


def evaluate_clauses(
    facts: RuntimeFacts,
    request: HandshakeRequest,
    *,
    measured_skew_seconds: float,
) -> tuple[tuple[ClauseAnswer, ...], tuple[str, ...]]:
    """Answer every clause that applies. Returns ``(answers, not_applicable)``."""

    context = _Context(facts=facts, request=request, measured_skew_seconds=measured_skew_seconds)
    answers: list[ClauseAnswer] = []
    not_applicable: list[str] = []
    for clause_id in ALL_CLAUSES:
        if not applies(clause_id, horizon_kind=facts.horizon.horizon_kind):
            not_applicable.append(clause_id)
            continue
        answers.append(CLAUSE_EVALUATORS[clause_id](context))
    return tuple(answers), tuple(not_applicable)


# --------------------------------------------------------------------------- #
# Obligations, resolution, agreement digest
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Obligations(JsonDataclassMixin):
    """What the container commits to for this run, and only this run."""

    max_concurrency: int
    lease_ttl_seconds: float
    lease_renewable: bool
    deferred_scoring: bool
    quiescence: bool
    settlement_window_seconds: float
    horizon_kind: str
    horizon_value: float
    time_dilation: float
    grace_seconds: float
    seconds_per_unit: float | None = None
    partial_roster: str = "refuse"
    probe_binding: bool = False

    def canonical_payload(self) -> dict[str, Any]:
        """Exactly what ``Obligations.from_payload(...).to_payload()`` reconstructs.

        The executor's typed obligations drop every key it does not model, so
        the digest must be taken over this subset and these coercions -- an int
        lease TTL on the wire becomes a float there, and a float here.
        """

        return {
            "max_concurrency": int(self.max_concurrency),
            "lease_ttl_seconds": float(self.lease_ttl_seconds),
            "deferred_scoring": bool(self.deferred_scoring),
            "quiescence": bool(self.quiescence),
            "settlement_window_seconds": float(self.settlement_window_seconds),
            "horizon": {
                "horizon_kind": self.horizon_kind,
                "value_seconds": float(self.horizon_value),
                "time_dilation": float(self.time_dilation),
                "grace_seconds": float(self.grace_seconds),
            },
        }

    def wire_payload(self) -> dict[str, Any]:
        """The canonical subset plus what the container additionally commits to."""

        payload = self.canonical_payload()
        payload["lease_renewable"] = bool(self.lease_renewable)
        payload["partial_roster"] = self.partial_roster
        payload["probe_binding"] = bool(self.probe_binding)
        payload["horizon"] = {
            **payload["horizon"],
            "value": float(self.horizon_value),
            "seconds_per_unit": self.seconds_per_unit,
        }
        return payload


def build_obligations(facts: RuntimeFacts, answers: Sequence[ClauseAnswer]) -> Obligations:
    """Obligations follow the clause answers; they never contradict them."""

    by_id = {answer.clause_id: answer for answer in answers}
    quiescence = by_id["reward.horizon_quiescence"].verdict == "accepted"
    settlement = (
        facts.reward.settlement_window_seconds
        if by_id["reward.settlement_window"].verdict == "accepted"
        else 0.0
    )
    return Obligations(
        max_concurrency=int(facts.lifecycle.max_concurrency),
        lease_ttl_seconds=float(facts.lifecycle.lease_ttl_seconds),
        lease_renewable=bool(facts.lifecycle.lease_renewable),
        deferred_scoring=bool(facts.reward.deferred_scoring),
        quiescence=quiescence,
        settlement_window_seconds=float(settlement),
        horizon_kind=facts.horizon.horizon_kind,
        horizon_value=float(facts.horizon.value),
        time_dilation=float(facts.horizon.time_dilation),
        grace_seconds=float(facts.horizon.grace_seconds),
        seconds_per_unit=facts.horizon.seconds_per_unit,
        partial_roster=facts.topology.partial_roster_disposition,
        probe_binding=bool(facts.probe_binding),
    )


def resolve_taskset(
    facts: RuntimeFacts, request: HandshakeRequest
) -> tuple[TaskRow, ...]:
    """One row per requested task id, in request order, duplicate-free."""

    rows: list[TaskRow] = []
    seen: set[str] = set()
    for task_id in request.taskset.task_ids:
        if task_id in seen:
            continue
        row = facts.discovery.row(task_id)
        if row is None:
            continue
        seen.add(task_id)
        rows.append(row)
    return tuple(rows)


def compute_agreement_digest(
    *,
    request: HandshakeRequest,
    handshake_id: str,
    capability_hash: str,
    renderer_fingerprint: str,
    taskset_resolution: Sequence[TaskRow],
    obligations: Obligations,
    clauses: Sequence[ClauseAnswer],
) -> str:
    """Bind both documents, the capability hash, the renderer, tasks, obligations.

    Field for field the same payload as
    ``synth_optimizers.rl.handshake.compute_agreement_digest``. The clause list
    carries only ``clause_id`` and ``verdict`` because the executor merges in
    its own local clause results that the container cannot know, and a digest
    only one side can compute is not an agreement.
    """

    return "sha256:" + canonical_digest(
        {
            "schema_version": HANDSHAKE_SCHEMA_VERSION,
            "handshake_id": handshake_id,
            "request": request.digest_payload,
            "capability_hash": capability_hash,
            "renderer_fingerprint": renderer_fingerprint,
            "task_digests": sorted(
                [item.task_id, item.content_digest, item.topology_ref]
                for item in taskset_resolution
            ),
            "obligations": obligations.canonical_payload(),
            "clauses": sorted([item.clause_id, item.verdict] for item in clauses),
        }
    )


# --------------------------------------------------------------------------- #
# The verdict, and the admitted agreement
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class HandshakeVerdict(JsonDataclassMixin):
    """The container's answer: per clause, with obligations and an expiry."""

    handshake_id: str
    accepted: bool
    clauses: tuple[ClauseAnswer, ...]
    not_applicable_clauses: tuple[str, ...]
    obligations: Obligations
    taskset_resolution: tuple[TaskRow, ...]
    capability_hash: str
    agreement_digest: str
    expires_at: datetime
    container_time: datetime
    measured_skew_seconds: float
    skew_tolerance_seconds: float
    rejected_mandatory_clauses: tuple[str, ...] = ()
    degraded_clauses: tuple[str, ...] = ()
    unaccepted_degraded_clauses: tuple[str, ...] = ()
    acknowledged_substitutes: Mapping[str, str] = field(default_factory=dict)
    schema_version: str = HANDSHAKE_SCHEMA_VERSION

    def clause(self, clause_id: str) -> ClauseAnswer | None:
        for answer in self.clauses:
            if answer.clause_id == clause_id:
                return answer
        return None

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "handshake_id": self.handshake_id,
            "accepted": bool(self.accepted),
            "clauses": [answer.to_payload() for answer in self.clauses],
            "not_applicable_clauses": list(self.not_applicable_clauses),
            "rejected_mandatory_clauses": list(self.rejected_mandatory_clauses),
            "degraded_clauses": list(self.degraded_clauses),
            "unaccepted_degraded_clauses": list(self.unaccepted_degraded_clauses),
            "acknowledged_substitutes": dict(self.acknowledged_substitutes),
            "obligations": self.obligations.wire_payload(),
            "taskset_resolution": [
                {
                    "task_id": item.task_id,
                    "content_digest": item.content_digest,
                    "topology_ref": item.topology_ref,
                    "task_family": item.task_family,
                }
                for item in self.taskset_resolution
            ],
            "capability_hash": self.capability_hash,
            "agreement_digest": self.agreement_digest,
            "expires_at": format_rfc3339(self.expires_at),
            "clock": {
                "container_time": format_rfc3339(self.container_time),
                "measured_skew_seconds": self.measured_skew_seconds,
                "tolerance_seconds": self.skew_tolerance_seconds,
            },
        }


@dataclass(frozen=True, slots=True)
class AdmittedHandshake(JsonDataclassMixin):
    """An accepted agreement. Every attempt is gated against this object."""

    handshake_id: str
    run_id: str
    agreement_digest: str
    capability_hash: str
    renderer_fingerprint: str
    obligations: Obligations
    taskset_resolution: tuple[TaskRow, ...]
    clauses: tuple[ClauseAnswer, ...]
    expires_at: datetime
    admitted_at: datetime

    def task_digest(self, task_id: str) -> str:
        for item in self.taskset_resolution:
            if item.task_id == task_id:
                return item.content_digest
        raise AgreementMismatch(f"task {task_id!r} is not part of this agreement")

    @property
    def quiescence_accepted(self) -> bool:
        answer = next(
            (item for item in self.clauses if item.clause_id == "reward.horizon_quiescence"),
            None,
        )
        return bool(answer is not None and answer.verdict == "accepted")


# --------------------------------------------------------------------------- #
# The registry: issue, admit, renew, revoke
# --------------------------------------------------------------------------- #


class HandshakeRegistry:
    """Issues agreements and gates every attempt against one.

    The records are frozen; only this object holds state. One lock guards it,
    and nothing here does background work.
    """

    def __init__(
        self,
        facts: RuntimeFacts,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._facts = facts
        self._clock = clock or utc_now
        self._lock = threading.RLock()
        self._admitted: dict[str, AdmittedHandshake] = {}
        self._revoked: dict[str, str] = {}
        self._issued = 0

    # -- state --------------------------------------------------------- #

    @property
    def facts(self) -> RuntimeFacts:
        return self._facts

    @property
    def capability_hash(self) -> str:
        return self._facts.capability_hash

    def agreements(self) -> Mapping[str, AdmittedHandshake]:
        with self._lock:
            return dict(self._admitted)

    def now(self) -> datetime:
        return self._clock()

    # -- issuing ------------------------------------------------------- #

    def handshake(self, payload: Any) -> HandshakeVerdict:
        """Answer one requirement document. Admits only when it is accepted."""

        request = HandshakeRequest.from_payload(payload)
        with self._lock:
            if request.renew_of:
                return self._renew_locked(request.renew_of, request=request)
            return self._issue_locked(request)

    def _issue_locked(self, request: HandshakeRequest) -> HandshakeVerdict:
        facts = self._facts
        moment = self._clock()
        skew = (moment - request.executor_time).total_seconds()
        answers, not_applicable = evaluate_clauses(
            facts, request, measured_skew_seconds=skew
        )
        obligations = build_obligations(facts, answers)
        resolution = resolve_taskset(facts, request)

        rejected = tuple(
            answer.clause_id
            for answer in answers
            if answer.mandatory and answer.verdict in {"rejected", "unsupported"}
        )
        degraded = tuple(answer.clause_id for answer in answers if answer.verdict == "degraded")
        acknowledged = {
            answer.clause_id: request.accept_degraded[answer.clause_id]
            for answer in answers
            if answer.verdict == "degraded"
            and answer.clause_id in request.accept_degraded
            and request.accept_degraded[answer.clause_id] == answer.substitute
            and answer.substitute
        }
        unaccepted = tuple(
            clause
            for clause in degraded
            if clause not in acknowledged and clause not in OPTIONAL_CLAUSES
        )
        accepted = not rejected and not unaccepted

        self._issued += 1
        handshake_id = "hs_" + canonical_digest(
            [facts.container_id, self._issued, request.raw, format_rfc3339(moment)],
            length=16,
        )
        digest_value = compute_agreement_digest(
            request=request,
            handshake_id=handshake_id,
            capability_hash=facts.capability_hash,
            renderer_fingerprint=facts.renderer_profile.fingerprint,
            taskset_resolution=resolution,
            obligations=obligations,
            clauses=answers,
        )
        expires_at = moment + timedelta(seconds=float(facts.handshake_ttl_seconds))
        verdict = HandshakeVerdict(
            handshake_id=handshake_id,
            accepted=accepted,
            clauses=answers,
            not_applicable_clauses=not_applicable,
            obligations=obligations,
            taskset_resolution=resolution,
            capability_hash=facts.capability_hash,
            agreement_digest=digest_value,
            expires_at=expires_at,
            container_time=moment,
            measured_skew_seconds=skew,
            skew_tolerance_seconds=float(facts.lifecycle.clock_skew_tolerance_seconds),
            rejected_mandatory_clauses=rejected,
            degraded_clauses=degraded,
            unaccepted_degraded_clauses=unaccepted,
            acknowledged_substitutes=acknowledged,
        )
        if accepted:
            self._admitted[handshake_id] = AdmittedHandshake(
                handshake_id=handshake_id,
                run_id=request.run_id,
                agreement_digest=digest_value,
                capability_hash=facts.capability_hash,
                renderer_fingerprint=facts.renderer_profile.fingerprint,
                obligations=obligations,
                taskset_resolution=resolution,
                clauses=answers,
                expires_at=expires_at,
                admitted_at=moment,
            )
        return verdict

    # -- the gate ------------------------------------------------------ #

    def assert_admissible(
        self,
        handshake_id: str | None,
        agreement_digest: str | None,
        *,
        now: datetime | None = None,
    ) -> AdmittedHandshake:
        """The gate every attempt, resume, and probe binding passes.

        Absent, unknown, expired, revoked, or mismatched is a refusal. A run
        cannot drift out from under its own agreement.
        """

        with self._lock:
            if not handshake_id or not str(handshake_id).strip():
                raise HandshakeAbsent("no handshake_id on the attempt")
            key = str(handshake_id).strip()
            if key in self._revoked:
                raise HandshakeRevoked(f"handshake {key!r} was revoked: {self._revoked[key]}")
            agreement = self._admitted.get(key)
            if agreement is None:
                raise HandshakeUnknown(f"handshake {key!r} was never admitted")
            moment = now or self._clock()
            if agreement.expires_at <= moment:
                raise HandshakeExpired(
                    f"handshake {key!r} expired at {format_rfc3339(agreement.expires_at)}"
                )
            if not agreement_digest or not str(agreement_digest).strip():
                raise HandshakeAbsent(f"attempt on handshake {key!r} names no agreement digest")
            if str(agreement_digest).strip() != agreement.agreement_digest:
                raise AgreementMismatch(
                    f"attempt names agreement {agreement_digest} but handshake {key} "
                    f"agreed {agreement.agreement_digest}"
                )
            return agreement

    # -- renewal and revocation ---------------------------------------- #

    def renew(
        self, handshake_id: str, *, request: HandshakeRequest | None = None
    ) -> HandshakeVerdict:
        """Extend an agreement, re-reading the capability document first."""

        with self._lock:
            return self._renew_locked(handshake_id, request=request)

    def _renew_locked(
        self, handshake_id: str, *, request: HandshakeRequest | None
    ) -> HandshakeVerdict:
        key = str(handshake_id or "").strip()
        if not key:
            raise HandshakeAbsent("renewal names no handshake_id")
        if key in self._revoked:
            raise HandshakeRevoked(f"handshake {key!r} was revoked: {self._revoked[key]}")
        agreement = self._admitted.get(key)
        if agreement is None:
            raise HandshakeUnknown(f"handshake {key!r} was never admitted")
        current = self._facts.capability_hash
        if current != agreement.capability_hash:
            self._revoke_locked(key, "capability document changed under a live handshake")
            raise CapabilityDrift(
                "capability document changed under a live handshake: "
                f"{agreement.capability_hash} != {current}"
            )
        moment = self._clock()
        if agreement.expires_at <= moment:
            self._revoke_locked(key, "handshake expired before renewal")
            raise HandshakeExpired(
                f"handshake {key!r} expired at {format_rfc3339(agreement.expires_at)}; "
                "re-handshake instead of renewing"
            )
        skew = (
            (moment - request.executor_time).total_seconds()
            if request is not None
            else 0.0
        )
        expires_at = moment + timedelta(seconds=float(self._facts.handshake_ttl_seconds))
        self._admitted[key] = replace(agreement, expires_at=expires_at, admitted_at=moment)
        return HandshakeVerdict(
            handshake_id=key,
            accepted=True,
            clauses=agreement.clauses,
            not_applicable_clauses=tuple(
                clause
                for clause in CONDITIONAL_CLAUSES
                if not applies(clause, horizon_kind=self._facts.horizon.horizon_kind)
            ),
            obligations=agreement.obligations,
            taskset_resolution=agreement.taskset_resolution,
            capability_hash=agreement.capability_hash,
            # A renewal never changes the agreement: a changed digest means
            # re-handshake, not renew.
            agreement_digest=agreement.agreement_digest,
            expires_at=expires_at,
            container_time=moment,
            measured_skew_seconds=skew,
            skew_tolerance_seconds=float(self._facts.lifecycle.clock_skew_tolerance_seconds),
        )

    def revoke(self, handshake_id: str, reason: str) -> None:
        """Stop admitting new attempts; in-flight work finishes or cancels."""

        with self._lock:
            self._revoke_locked(handshake_id, reason)

    def _revoke_locked(self, handshake_id: str, reason: str) -> None:
        key = str(handshake_id or "").strip()
        if not key:
            raise HandshakeAbsent("revocation names no handshake_id")
        self._revoked[key] = reason or "revoked by container"
        self._admitted.pop(key, None)

    def degrade(self, facts: RuntimeFacts, *, reason: str) -> tuple[str, ...]:
        """Adopt a new capability document and revoke every stale agreement.

        This is the container degrading under a live run: the executor must
        stop admitting new attempts, finish or cancel in-flight ones, and
        re-handshake before resuming.
        """

        with self._lock:
            self._facts = facts
            new_hash = facts.capability_hash
            stale = tuple(
                handshake_id
                for handshake_id, agreement in self._admitted.items()
                if agreement.capability_hash != new_hash
            )
            for handshake_id in stale:
                self._revoke_locked(handshake_id, reason)
            return stale


# --------------------------------------------------------------------------- #
# The single coupling point to the handshake/admission port in cispo_contract
# --------------------------------------------------------------------------- #


class CispoHandshakeAdapter:
    """The handshake/admission half of ``cispo_contract.CispoAdmissionPort``.

    This class is the *only* thing in this module that speaks the port's
    vocabulary, so the coupling is one class wide and everything above it is
    typed and importable on its own. The port is a ``runtime_checkable``
    ``Protocol``, so this satisfies it structurally without importing it --
    which also means an edit to the port's shape shows up as a failing
    ``isinstance`` check in this module's tests rather than an import error.

    ``cispo_bind_policy`` dispatches on the requested kind: the unpaid ``probe``
    kind is served by ``cispo_probe``, a real sampler binding by
    ``cispo_policy``, and any other kind is a typed refusal rather than a silent
    stub. Neither module is imported at module scope, so this one still stands
    on its own.
    """

    def __init__(
        self,
        facts: RuntimeFacts,
        *,
        clock: Callable[[], datetime] | None = None,
        registry: HandshakeRegistry | None = None,
    ) -> None:
        self.registry = registry or HandshakeRegistry(facts, clock=clock)

    @property
    def facts(self) -> RuntimeFacts:
        return self.registry.facts

    # -- discovery ----------------------------------------------------- #

    def cispo_health(self) -> dict[str, Any]:
        facts = self.facts
        return {
            "status": "ok",
            "container_id": facts.container_id,
            "container_version": facts.container_version or facts.contract_version,
            "container_image_digest": facts.container_image_digest,
            "contract_version": facts.contract_version,
            "capability_hash": facts.capability_hash,
        }

    def cispo_capabilities(self) -> dict[str, Any]:
        return self.capability_document()

    def capability_document(self) -> dict[str, Any]:
        return self.facts.capability_document()

    def capability_hash(self) -> str:
        return self.registry.capability_hash

    def cispo_taskset(self) -> dict[str, Any]:
        discovery = self.facts.discovery
        return {
            "taskset_id": discovery.taskset_id,
            "taskset_version": discovery.taskset_version,
            "splits": list(discovery.splits),
            "task_content_digests": bool(discovery.task_content_digests),
            "deterministic_lookup": bool(discovery.deterministic_lookup),
            "duplicate_free": bool(discovery.duplicate_free),
        }

    def cispo_taskset_tasks(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """One duplicate-free row per requested id. A missing id is a refusal.

        The shipped client spells the field ``ids``; the handshake document
        spells the same idea ``task_ids``. Both are read here, because a
        container that answers only its own preferred spelling silently
        resolves nothing and the run fails several layers away.
        """

        raw = request.get("ids")
        if raw is None:
            raw = request.get("task_ids") or ()
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise MalformedRequirementDocument("task ids must be a list")
        wanted = tuple(dict.fromkeys(str(item) for item in raw))
        discovery = self.facts.discovery
        missing = tuple(task_id for task_id in wanted if discovery.row(task_id) is None)
        if missing:
            raise MalformedRequirementDocument(
                f"container holds no rows for task ids {missing}; a shorter list "
                "would look like a successful lookup"
            )
        return {
            "taskset_id": discovery.taskset_id,
            "taskset_version": discovery.taskset_version,
            # The shipped client reads "rows"; so do the reference fakes. A
            # container that answers with its own preferred spelling resolves
            # nothing and fails several layers away.
            "rows": [
                {
                    "task_id": row.task_id,
                    "content_digest": row.content_digest,
                    "topology_ref": row.topology_ref,
                    "task_family": row.task_family,
                }
                for row in (discovery.row(task_id) for task_id in wanted)
                if row is not None
            ],
        }

    def cispo_topology(self, topology_id: str) -> dict[str, Any]:
        declared = self.facts.topology.topology_id
        if str(topology_id).strip() != declared:
            raise MalformedRequirementDocument(
                f"container declares topology {declared!r}, not {topology_id!r}"
            )
        document = self.capability_document()
        topology = document.get("topology")
        if not isinstance(topology, Mapping):  # pragma: no cover - built above
            raise MalformedRequirementDocument("capability document declares no topology")
        return dict(topology)

    # -- agreement ----------------------------------------------------- #

    def cispo_handshake(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """``POST handshake_route``. Answers clause by clause, never a boolean."""

        return self.registry.handshake(request).to_payload()

    # Kept as the plain-language alias the rest of this module uses.
    handshake = cispo_handshake

    def renew(self, handshake_id: str) -> dict[str, Any]:
        return self.registry.renew(handshake_id).to_payload()

    def revoke(self, handshake_id: str, reason: str) -> None:
        self.registry.revoke(handshake_id, reason)

    def degrade(self, facts: RuntimeFacts, *, reason: str) -> tuple[str, ...]:
        return self.registry.degrade(facts, reason=reason)

    # -- policy binding ------------------------------------------------ #

    @property
    def policies(self) -> Any:
        """The sampler-binding half, kept for the life of the run.

        Created on first use so a build that only ever probes never constructs
        it, and settable so a container can supply its own reachability probe
        and registry without this class growing a second constructor argument.
        """

        adapter = getattr(self, "_policies", None)
        if adapter is None:
            from .cispo_policy import CispoPolicyAdapter

            adapter = CispoPolicyAdapter(self.facts)
            self._policies = adapter
        return adapter

    @policies.setter
    def policies(self, adapter: Any) -> None:
        self._policies = adapter

    @property
    def probe_bindings(self) -> dict[str, dict[str, Any]]:
        """Probe bindings this adapter has issued, by binding id."""

        held = getattr(self, "_probe_bindings", None)
        if held is None:
            held = {}
            self._probe_bindings = held
        return held

    def cispo_bind_policy(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Bind one policy. ``probe`` is unpaid; a sampler kind reaches a provider."""

        from .cispo_probe import PROBE_POLICY_KIND, CispoProbeAdapter, ProbeError

        kind = str(request.get("kind") or request.get("policy_kind") or "")
        agreement = self.admit_attempt(request)
        if kind == PROBE_POLICY_KIND:
            payload = CispoProbeAdapter(self.facts).bind(agreement, request)
            # A probe attempt is submitted against its binding like any other,
            # so the binding has to be findable afterwards. The sampler
            # registry cannot hold it — a probe has no sampler origin — so the
            # adapter remembers its own.
            binding_id = str(payload.get("config_id") or "")
            if binding_id:
                self.probe_bindings[binding_id] = dict(payload)
            return payload
        from .cispo_policy import SAMPLER_POLICY_KINDS

        if kind in SAMPLER_POLICY_KINDS:
            return self.policies.bind(agreement, request)
        raise ProbeError(
            f"this adapter binds {PROBE_POLICY_KIND!r} and "
            f"{sorted(SAMPLER_POLICY_KINDS)}; {kind!r} is none of them"
        )

    def cispo_bind_policy_set(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Bind every instance of a joint episode, or none of them."""

        from .cispo_probe import PROBE_POLICY_KIND, ProbeError

        bindings = request.get("bindings") or ()
        if not isinstance(bindings, Sequence) or isinstance(bindings, (str, bytes)):
            raise MalformedRequirementDocument("policy set bindings must be a list")
        kinds = {str(request.get("kind") or request.get("policy_kind") or "")} | {
            str(item.get("kind") or item.get("policy_kind") or "")
            for item in bindings
            if isinstance(item, Mapping)
        }
        agreement = self.admit_attempt(request)
        if kinds - {"", PROBE_POLICY_KIND}:
            return self.policies.bind_set(agreement, request)
        resolved: list[dict[str, Any]] = []
        for item in bindings:
            if not isinstance(item, Mapping):
                raise MalformedRequirementDocument("each policy set binding must be an object")
            merged = {
                "handshake_id": agreement.handshake_id,
                "agreement_digest": agreement.agreement_digest,
                # The kind is declared once for the set, and each binding is
                # bound one at a time. Without carrying it down, every item
                # arrives naming no kind at all and the binder refuses a
                # roster that was perfectly well formed.
                "kind": str(request.get("kind") or request.get("policy_kind") or "")
                or PROBE_POLICY_KIND,
                **item,
            }
            resolved.append(self.cispo_bind_policy(merged))
        if not resolved:
            raise ProbeError("a policy set with no binding would start a half-bound episode")
        set_id = canonical_digest([agreement.agreement_digest, resolved], length=20)
        return {
            "policy_set_revision": set_id,
            # A set binding is still a binding, and the executor correlates an
            # attempt to what it bound by this id. A roster that returns none
            # cannot be submitted against.
            "config_id": f"probe_set_{set_id}",
            "policy_set_id": f"probe_set_{set_id}",
            "handshake_id": agreement.handshake_id,
            "agreement_digest": agreement.agreement_digest,
            "bindings": resolved,
        }

    # -- admission ----------------------------------------------------- #

    def assert_admissible(
        self, handshake_id: str | None, agreement_digest: str | None
    ) -> AdmittedHandshake:
        return self.registry.assert_admissible(handshake_id, agreement_digest)

    def admit_attempt(self, body: Mapping[str, Any]) -> AdmittedHandshake:
        """Gate one request body carrying ``handshake_id``/``agreement_digest``."""

        return self.assert_admissible(
            body.get("handshake_id"), body.get("agreement_digest")
        )


__all__ = [
    "ALL_CLAUSES",
    "CAPABILITY_SCHEMA_VERSION",
    "CISPO_CONTRACT_VERSION",
    "CLAUSE_EVALUATORS",
    "CLAUSE_GROUPS",
    "CLAUSE_SUBSTITUTES",
    "CONDITIONAL_CLAUSES",
    "HANDSHAKE_SCHEMA_VERSION",
    "MANDATORY_CLAUSES",
    "MANDATORY_ROUTE_NAMES",
    "OPTIONAL_CLAUSES",
    "UNCONDITIONAL_MANDATORY_CLAUSES",
    "VERDICTS",
    "AdmittedHandshake",
    "AgentInstanceFacts",
    "AgreementMismatch",
    "CapabilityDrift",
    "ChannelFacts",
    "CispoHandshakeAdapter",
    "ClauseAnswer",
    "DiscoveryFacts",
    "EvidenceFacts",
    "HandshakeAbsent",
    "HandshakeError",
    "HandshakeExpired",
    "HandshakeRegistry",
    "HandshakeRequest",
    "HandshakeRevoked",
    "HandshakeUnknown",
    "HandshakeVerdict",
    "HorizonFacts",
    "LifecycleFacts",
    "MalformedRequirementDocument",
    "Obligations",
    "PolicyFacts",
    "RecoveryFacts",
    "RendererProfileFacts",
    "RewardFacts",
    "RunPlanRequest",
    "RuntimeFacts",
    "TaskRow",
    "TasksetRequest",
    "TeamFacts",
    "TopologyFacts",
    "UndeclaredSubstitute",
    "applies",
    "build_obligations",
    "canonical_capability_hash",
    "canonical_digest",
    "compute_agreement_digest",
    "evaluate_clauses",
    "format_rfc3339",
    "parse_rfc3339",
    "resolve_taskset",
    "substitutes_for",
    "task_content_digest",
    "utc_now",
]
