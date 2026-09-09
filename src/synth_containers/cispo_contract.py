"""Container-side CISPO contract: the declared route table and the hashed
capability document.

An optimizer discovers this container by reading
``metadata.optimizer_contracts.cispo``. Routes are *declared* there, never
guessed, exactly as ``optimizer_contracts.gepa`` already is. Discovery is not
agreement: the capability document says what this build can do in general, and
the handshake (stream B) says whether it can honor one particular run.

Three rules shape this module.

1. **Nothing is hardcoded that a target could declare.** Every value that ends
   up in the capability document comes either from the runtime's
   :class:`~synth_containers.capabilities.RuntimeCapabilitySurface` (the flags
   the runtime already publishes) or from its
   :class:`CispoRuntimeDeclaration` (the CISPO-specific facts the surface has
   no field for: renderer profile, topology, horizon, lease, discovery). A
   target contributes its declared capabilities and nothing else.
2. **A capability that cannot be declared is a refusal, not a default.**
   Building a document from a runtime that does not support, say, behavior
   logprobs raises :class:`CispoCapabilityError`. Emitting a plausible-looking
   ``true`` there would buy a run that trains on evidence that was never valid.
3. **The content hash is fail-closed.** It covers the whole document minus the
   hash field itself, canonicalized with sorted keys and compact separators,
   and is prefixed ``sha256:``. Any change at all invalidates a prior preflight
   and every handshake built on it. This is byte-for-byte the optimizer's
   ``synth_optimizers.rl.capabilities.canonical_capability_hash``.

No task, harness, or environment name appears anywhere in this module.

Streams B and C own the behavior behind the routes. This module owns the
advertisement, the document, and the dispatch; it defines
:class:`CispoAdmissionPort` (handshake and admission) and
:class:`CispoRolloutPort` (rollout and evidence) as the contract those streams
implement, and ships null implementations that answer 501 until they do.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .capabilities import RuntimeCapabilitySurface
from .serde import JsonDataclassMixin

CISPO_OPTIMIZER_CONTRACT_VERSION = "synth_optimizers.cispo.v1"

#: Schema version of the capability document. The optimizer refuses any other.
CISPO_CAPABILITY_SCHEMA_VERSION = "cispo.capabilities.v1"

#: The CISPO surface is served under its own prefix. Ten of the seventeen
#: canonical paths (``/health``, ``/taskset``, ``/rollout``, ``/reward``, the
#: ``/rollouts/{rollout_id}/...`` family) already exist on this app with
#: blocking, GEPA-era semantics that differ from the CISPO ones -- a blocking
#: rollout result is not a 202 with a lease expiry. Registering the CISPO
#: handlers on those same paths would either shadow the existing surface or be
#: shadowed by it. The contract explicitly allows this: "A container may add or
#: rename any of them", because the executor calls only declared routes. So the
#: canonical shape is kept verbatim (``cispo_optimizer_contract(prefix="")``)
#: and mounted under this prefix for advertisement.
CISPO_ROUTE_PREFIX = "/cispo"

#: The canonical route table, unprefixed, verbatim from the declared contract.
CISPO_DECLARED_ROUTES: Mapping[str, str] = {
    "health_route": "/health",
    "capabilities_route": "/training/capabilities",
    "handshake_route": "/training/handshake",
    "taskset_route": "/taskset",
    "taskset_tasks_route": "/taskset/tasks",
    "topology_route": "/topologies/{topology_id}",
    "policy_bind_route": "/policy-configs",
    "policy_set_bind_route": "/policy-sets",
    "rollout_route": "/rollout",
    "rollout_state_route": "/rollouts/{rollout_id}",
    "rollout_events_route": "/rollouts/{rollout_id}/events",
    "rollout_renew_route": "/rollouts/{rollout_id}/renew",
    "rollout_finalize_route": "/rollouts/{rollout_id}/finalize",
    "rollout_terminate_route": "/rollouts/{rollout_id}/terminate",
    "trace_route": "/rollouts/{rollout_id}/trace",
    "artifacts_route": "/rollouts/{rollout_id}/artifacts",
    "reward_route": "/reward",
}

#: Method the executor uses per declared route, mirroring the optimizer's
#: ``ROUTE_METHODS``. ``reward_route`` is served on both verbs: the declared
#: contract table says "GET/POST" and the optimizer client only ever GETs it.
CISPO_ROUTE_METHODS: Mapping[str, str] = {
    "health_route": "GET",
    "capabilities_route": "GET",
    "handshake_route": "POST",
    "taskset_route": "GET",
    "taskset_tasks_route": "POST",
    "topology_route": "GET",
    "policy_bind_route": "POST",
    "policy_set_bind_route": "POST",
    "rollout_route": "POST",
    "rollout_state_route": "GET",
    "rollout_events_route": "GET",
    "rollout_renew_route": "POST",
    "rollout_finalize_route": "POST",
    "rollout_terminate_route": "POST",
    "trace_route": "GET",
    "artifacts_route": "GET",
    "reward_route": "GET",
}

#: Placeholders the executor knows how to substitute. A route declaring any
#: other placeholder is unresolvable and would be refused at parse time.
CISPO_ROUTE_PLACEHOLDERS: frozenset[str] = frozenset({"rollout_id", "topology_id"})

HORIZON_KINDS: frozenset[str] = frozenset({"wall_clock", "steps", "env_ticks"})
TURN_MODELS: frozenset[str] = frozenset({"sequential", "concurrent_realtime"})
ACTUATION_MODELS: frozenset[str] = frozenset({"direct_action", "deferred_program"})
REWARD_RELATIONS: frozenset[str] = frozenset(
    {"cooperative", "competitive_rank", "competitive_margin", "mixed"}
)
CHANNEL_SCOPES: frozenset[str] = frozenset({"intra_team", "cross_team", "private"})
SAMPLING_TRANSPORTS: frozenset[str] = frozenset(
    {"message_in_capture_out", "tokens_in_tokens_out"}
)
WIRE_APIS: frozenset[str] = frozenset({"chat_completions", "responses"})

#: Reward authority is not a dial. A container that does not own the reward
#: cannot be the reward authority, and the executor never scores locally.
CONTAINER_REWARD_AUTHORITY = "container"


class CispoContractError(ValueError):
    """The CISPO declaration is absent, malformed, or self-inconsistent."""


class CispoCapabilityError(CispoContractError):
    """A capability the contract makes mandatory was not declared.

    Raised instead of emitting a default. The whole point of a fail-closed
    preflight is that an undeclared capability stops a run before it spends.
    """


class CispoNotImplementedError(RuntimeError):
    """A declared route has no implementation behind it yet.

    Carries a 501 and a typed payload rather than a 404 or an empty body: the
    route *is* declared -- route presence and behavior support are separate
    questions -- but the port that answers it has not been supplied.
    """

    status = 501

    def __init__(self, port: str, operation: str, reason: str = "") -> None:
        self.port = port
        self.operation = operation
        self.reason = reason or (
            f"{port}.{operation} is declared but not implemented by this build"
        )
        super().__init__(f"HTTP 501: {self.reason}")

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": "cispo_port_not_implemented",
            "port": self.port,
            "operation": self.operation,
            "reason": self.reason,
            "contract_version": CISPO_OPTIMIZER_CONTRACT_VERSION,
        }


# --------------------------------------------------------------------------- #
# Route table
# --------------------------------------------------------------------------- #


def _placeholders(route: str) -> tuple[str, ...]:
    found: list[str] = []
    rest = route
    while "{" in rest:
        _head, _, rest = rest.partition("{")
        name, closer, rest = rest.partition("}")
        if not closer:
            raise CispoContractError(f"route {route!r} has an unterminated placeholder")
        found.append(name)
    return tuple(found)


def cispo_optimizer_contract(prefix: str = CISPO_ROUTE_PREFIX) -> dict[str, str]:
    """The advertisement block for ``metadata.optimizer_contracts.cispo``.

    All seventeen declared routes, absolute, with the contract version. With an
    empty prefix this is the canonical table verbatim.
    """

    normalized = str(prefix or "").strip().rstrip("/")
    if normalized and not normalized.startswith("/"):
        raise CispoContractError(f"route prefix must be absolute, got {prefix!r}")
    block: dict[str, str] = {"version": CISPO_OPTIMIZER_CONTRACT_VERSION}
    for name, route in CISPO_DECLARED_ROUTES.items():
        full = f"{normalized}{route}"
        if not full.startswith("/"):
            raise CispoContractError(f"declared route {name} must be absolute, got {full!r}")
        unknown = tuple(
            item for item in _placeholders(full) if item not in CISPO_ROUTE_PLACEHOLDERS
        )
        if unknown:
            raise CispoContractError(
                f"declared route {name} carries placeholders the executor cannot "
                f"substitute: {unknown}"
            )
        block[name] = full
    return block


def cispo_declared_routes(prefix: str = CISPO_ROUTE_PREFIX) -> dict[str, str]:
    """Just the routes, without the ``version`` key."""

    return {
        name: value
        for name, value in cispo_optimizer_contract(prefix).items()
        if name != "version"
    }


# --------------------------------------------------------------------------- #
# What a runtime declares
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RendererProfileDeclaration(JsonDataclassMixin):
    """Pinned renderer identity. A version string alone is not an identity.

    The executor compares this against the profile its training session will
    use, before any paid request. Everything that changes what a token sequence
    means belongs here.
    """

    profile_id: str
    package: str
    package_version: str
    config_digest: str
    tokenizer_id: str
    tokenizer_digest: str
    stop_token_ids: tuple[int, ...]
    modalities: tuple[str, ...] = ("text",)
    add_generation_prompt: bool = True
    canary_digest: str = ""

    def __post_init__(self) -> None:
        for name in (
            "profile_id",
            "package",
            "package_version",
            "config_digest",
            "tokenizer_id",
            "tokenizer_digest",
        ):
            if not str(getattr(self, name) or "").strip():
                raise CispoCapabilityError(f"renderer profile must declare {name}")
        if not self.stop_token_ids:
            raise CispoCapabilityError("renderer profile must declare stop token ids")
        if not self.modalities:
            raise CispoCapabilityError("renderer profile must declare at least one modality")

    def to_payload(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "package": self.package,
            "package_version": self.package_version,
            "config_digest": self.config_digest,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_digest": self.tokenizer_digest,
            "stop_token_ids": [int(item) for item in self.stop_token_ids],
            "modalities": [str(item) for item in self.modalities],
            "add_generation_prompt": bool(self.add_generation_prompt),
            "canary_digest": str(self.canary_digest),
        }


@dataclass(frozen=True, slots=True)
class HorizonDeclaration(JsonDataclassMixin):
    """How long one episode runs, in the unit the container actually counts.

    A ``steps`` or ``env_ticks`` horizon carries no duration of its own, so it
    must declare ``seconds_per_unit``. Reading 500 steps as 500 seconds is
    exactly the guess a lease may not be built on, so an undeclared conversion
    is a refusal rather than a default of 1.0.
    """

    horizon_kind: str
    value: float
    seconds_per_unit: float | None = None
    time_dilation: float = 1.0
    grace_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.horizon_kind not in HORIZON_KINDS:
            raise CispoCapabilityError(f"unknown horizon_kind {self.horizon_kind!r}")
        if self.value <= 0:
            raise CispoCapabilityError("horizon value must be positive")
        if self.time_dilation <= 0:
            raise CispoCapabilityError("time_dilation must be positive")
        if self.grace_seconds < 0:
            raise CispoCapabilityError("grace_seconds may not be negative")
        if self.horizon_kind == "wall_clock":
            if self.seconds_per_unit is not None and self.seconds_per_unit != 1.0:
                raise CispoCapabilityError(
                    "a wall_clock horizon is already seconds; seconds_per_unit must be 1.0"
                )
        elif self.seconds_per_unit is None or self.seconds_per_unit <= 0:
            raise CispoCapabilityError(
                f"a {self.horizon_kind} horizon must declare a positive seconds_per_unit; "
                "a lease may not be guessed from a unit with no duration"
            )

    def declared_seconds_per_unit(self) -> float:
        if self.horizon_kind == "wall_clock":
            return 1.0
        assert self.seconds_per_unit is not None  # enforced in __post_init__
        return float(self.seconds_per_unit)

    def horizon_seconds(self) -> float:
        """Wall-clock duration the declared horizon actually covers."""

        return float(self.value) * self.declared_seconds_per_unit()

    def to_payload(self) -> dict[str, Any]:
        return {
            "horizon_kind": self.horizon_kind,
            "value": float(self.value),
            "seconds_per_unit": self.declared_seconds_per_unit(),
            "time_dilation": float(self.time_dilation),
            "grace_seconds": float(self.grace_seconds),
            "horizon_seconds": self.horizon_seconds(),
        }


@dataclass(frozen=True, slots=True)
class AgentInstanceDeclaration(JsonDataclassMixin):
    """One seat in the roster. The executor binds these; it never infers them."""

    agent_instance_id: str
    role_id: str
    policy_type_id: str
    team_id: str
    trainable: bool
    #: A non-trainable instance must pin an immutable identity. An alias such as
    #: ``latest`` is not an identity and is refused.
    pinned_identity: str | None = None

    def __post_init__(self) -> None:
        for name in ("agent_instance_id", "role_id", "policy_type_id", "team_id"):
            if not str(getattr(self, name) or "").strip():
                raise CispoCapabilityError(f"agent instance must declare {name}")

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "agent_instance_id": self.agent_instance_id,
            "role_id": self.role_id,
            "policy_type_id": self.policy_type_id,
            "team_id": self.team_id,
            "trainable": bool(self.trainable),
        }
        if self.pinned_identity:
            payload["pinned_identity"] = str(self.pinned_identity)
        return payload


@dataclass(frozen=True, slots=True)
class TeamDeclaration(JsonDataclassMixin):
    """A team and the smallest roster that still makes its episode meaningful."""

    team_id: str
    trainable: bool
    minimum_viable_roster: int

    def __post_init__(self) -> None:
        if not str(self.team_id or "").strip():
            raise CispoCapabilityError("team must declare team_id")
        if self.minimum_viable_roster < 1:
            raise CispoCapabilityError(
                f"team {self.team_id!r} must declare a minimum_viable_roster of at least 1"
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "team_id": self.team_id,
            "trainable": bool(self.trainable),
            "minimum_viable_roster": int(self.minimum_viable_roster),
        }


@dataclass(frozen=True, slots=True)
class CommunicationChannelDeclaration(JsonDataclassMixin):
    """A declared channel. Declaring one obliges the container to carry it:
    a declared channel that returns no messages is a dropped-channel evidence
    failure, not an empty result."""

    channel_id: str
    scope: str
    trainable_for_author: bool = True

    def __post_init__(self) -> None:
        if not str(self.channel_id or "").strip():
            raise CispoCapabilityError("channel must declare channel_id")
        if self.scope not in CHANNEL_SCOPES:
            raise CispoCapabilityError(f"unknown channel scope {self.scope!r}")

    def to_payload(self) -> dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "scope": self.scope,
            "trainable_for_author": bool(self.trainable_for_author),
        }


@dataclass(frozen=True, slots=True)
class TopologyDeclaration(JsonDataclassMixin):
    """The container-declared roster, turn model, actuation model and horizon."""

    topology_id: str
    turn_model: str
    actuation_model: str
    reward_relation: str
    agent_instances: tuple[AgentInstanceDeclaration, ...]
    teams: tuple[TeamDeclaration, ...]
    horizon: HorizonDeclaration
    communication_channels: tuple[CommunicationChannelDeclaration, ...] = ()
    #: policy_type_id -> parameter group. Instances sharing a group share
    #: parameters and must publish atomically.
    parameter_groups: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.topology_id or "").strip():
            raise CispoCapabilityError("topology must declare topology_id")
        if self.turn_model not in TURN_MODELS:
            raise CispoCapabilityError(f"unknown turn_model {self.turn_model!r}")
        if self.actuation_model not in ACTUATION_MODELS:
            raise CispoCapabilityError(f"unknown actuation_model {self.actuation_model!r}")
        if self.reward_relation not in REWARD_RELATIONS:
            raise CispoCapabilityError(f"unknown reward_relation {self.reward_relation!r}")
        if not self.agent_instances:
            raise CispoCapabilityError("topology declares no agent instances")
        if not self.teams:
            raise CispoCapabilityError("topology declares no teams")
        instance_ids = [item.agent_instance_id for item in self.agent_instances]
        if len(set(instance_ids)) != len(instance_ids):
            raise CispoCapabilityError("duplicate agent_instance_id in topology")
        team_ids = [team.team_id for team in self.teams]
        if len(set(team_ids)) != len(team_ids):
            raise CispoCapabilityError("duplicate team_id in topology")
        known = set(team_ids)
        for instance in self.agent_instances:
            if instance.team_id not in known:
                raise CispoCapabilityError(
                    f"instance {instance.agent_instance_id} names undeclared team "
                    f"{instance.team_id!r}"
                )
        for team in self.teams:
            roster = sum(
                1 for item in self.agent_instances if item.team_id == team.team_id
            )
            if roster < team.minimum_viable_roster:
                raise CispoCapabilityError(
                    f"team {team.team_id!r} declares minimum_viable_roster "
                    f"{team.minimum_viable_roster} but rosters {roster} instances"
                )
        if not any(team.trainable for team in self.teams):
            raise CispoCapabilityError("topology declares no trainable team")
        for instance in self.agent_instances:
            if not instance.trainable and not instance.pinned_identity:
                raise CispoCapabilityError(
                    f"non-trainable instance {instance.agent_instance_id} must pin an "
                    "immutable identity; an opponent may not resolve by alias"
                )

    @property
    def trainable_instances(self) -> tuple[AgentInstanceDeclaration, ...]:
        return tuple(item for item in self.agent_instances if item.trainable)

    def to_payload(self) -> dict[str, Any]:
        horizon = self.horizon.to_payload()
        # ``value_seconds`` is the key the optimizer's ``_parse_topology`` reads
        # into ``Horizon.value``; it carries the value in horizon *units*, which
        # for a step or tick horizon is not seconds. The name is the client's,
        # kept verbatim so the document parses; ``value`` and
        # ``seconds_per_unit`` alongside it say what the number actually means.
        horizon["value_seconds"] = float(self.horizon.value)
        return {
            "topology_id": self.topology_id,
            "turn_model": self.turn_model,
            "actuation_model": self.actuation_model,
            "reward_relation": self.reward_relation,
            "agent_instances": [item.to_payload() for item in self.agent_instances],
            "teams": [item.to_payload() for item in self.teams],
            "communication_channels": [
                item.to_payload() for item in self.communication_channels
            ],
            "horizon": horizon,
            "parameter_groups": {str(k): str(v) for k, v in self.parameter_groups.items()},
        }


@dataclass(frozen=True, slots=True)
class DiscoveryDeclaration(JsonDataclassMixin):
    """Taskset identity and the guarantees its lookup makes."""

    taskset_id: str
    taskset_version: str
    splits: tuple[str, ...]
    task_content_digests: bool
    deterministic_lookup: bool
    duplicate_free: bool

    def __post_init__(self) -> None:
        if not str(self.taskset_id or "").strip():
            raise CispoCapabilityError("discovery must declare taskset_id")
        if not str(self.taskset_version or "").strip():
            raise CispoCapabilityError("discovery must declare taskset_version")
        if not self.splits:
            raise CispoCapabilityError("discovery must declare at least one split")

    def to_payload(self) -> dict[str, Any]:
        return {
            "taskset_id": self.taskset_id,
            "taskset_version": self.taskset_version,
            "splits": [str(item) for item in self.splits],
            "task_content_digests": bool(self.task_content_digests),
            "deterministic_lookup": bool(self.deterministic_lookup),
            "duplicate_free": bool(self.duplicate_free),
        }


@dataclass(frozen=True, slots=True)
class PolicyBindingDeclaration(JsonDataclassMixin):
    """How this container accepts a policy binding."""

    binding_transport: str
    wire_api: str
    session_scoped_sampler_origin: bool
    embeds_credentials: bool
    revision_immutable_after_admission: bool
    records_policy_revision: bool
    #: The unpaid ``probe`` binding kind: deterministic synthetic generations
    #: that walk the whole evidence path at zero provider cost.
    probe_binding: bool = False

    def __post_init__(self) -> None:
        if self.binding_transport not in SAMPLING_TRANSPORTS:
            raise CispoCapabilityError(
                f"unknown binding_transport {self.binding_transport!r}"
            )
        if self.wire_api not in WIRE_APIS:
            raise CispoCapabilityError(f"unknown wire_api {self.wire_api!r}")

    def to_payload(self) -> dict[str, Any]:
        return {
            "binding_transport": self.binding_transport,
            "wire_api": self.wire_api,
            "session_scoped_sampler_origin": bool(self.session_scoped_sampler_origin),
            "embeds_credentials": bool(self.embeds_credentials),
            "revision_immutable_after_admission": bool(
                self.revision_immutable_after_admission
            ),
            "records_policy_revision": bool(self.records_policy_revision),
            "probe_binding": bool(self.probe_binding),
        }


@dataclass(frozen=True, slots=True)
class LeaseDeclaration(JsonDataclassMixin):
    """The container's own lease obligation.

    A lease is never derived from the horizon: the horizon says how long an
    episode runs, the TTL says how long one grant survives without a heartbeat,
    and they are different clocks. ``heartbeat_route`` is not declared here --
    it is filled from the declared renew route, so the obligation can never
    name a path the container does not actually serve.
    """

    ttl_seconds: float
    renewable: bool
    straggler_grace_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.ttl_seconds <= 0:
            raise CispoCapabilityError("lease ttl_seconds must be positive")
        if self.straggler_grace_seconds < 0:
            raise CispoCapabilityError("straggler_grace_seconds may not be negative")

    def to_payload(self, *, heartbeat_route: str) -> dict[str, Any]:
        return {
            "ttl_seconds": float(self.ttl_seconds),
            "renewable": bool(self.renewable),
            "heartbeat_route": heartbeat_route,
        }


@dataclass(frozen=True, slots=True)
class LifecycleDeclaration(JsonDataclassMixin):
    """Queue-facing obligations. Cancellation and pause/resume are read from
    the runtime capability surface rather than redeclared here."""

    advertised_concurrency: int
    supports_idempotency: bool
    exactly_one_terminal_result: bool
    lease: LeaseDeclaration

    def __post_init__(self) -> None:
        if self.advertised_concurrency < 1:
            raise CispoCapabilityError("advertised_concurrency must be at least 1")

    def to_payload(
        self, *, heartbeat_route: str, supports_cancellation: bool, supports_pause_resume: bool
    ) -> dict[str, Any]:
        return {
            "max_concurrency": int(self.advertised_concurrency),
            "lease_ttl_seconds": float(self.lease.ttl_seconds),
            "lease": self.lease.to_payload(heartbeat_route=heartbeat_route),
            "supports_idempotency": bool(self.supports_idempotency),
            "supports_cancellation": bool(supports_cancellation),
            "supports_lease_renewal": bool(self.lease.renewable),
            "exactly_one_terminal_result": bool(self.exactly_one_terminal_result),
            "supports_pause_resume": bool(supports_pause_resume),
            "straggler_grace_seconds": float(self.lease.straggler_grace_seconds),
        }


@dataclass(frozen=True, slots=True)
class EvidenceDeclaration(JsonDataclassMixin):
    """Evidence guarantees the capability surface has no field for.

    ``trace_v5`` and ``behavior_logprobs`` are deliberately absent: those are
    read from ``RuntimeCapabilitySurface.trace_support`` and
    ``token_emission.logprobs``, so a runtime cannot claim here what it does
    not already publish there.
    """

    strict_prefix: bool
    masking: bool
    wire_objects: bool
    artifact_by_reference: bool = False
    tokens_in_tokens_out: bool = False

    def to_payload(self, *, trace_v5: bool, behavior_logprobs: bool) -> dict[str, Any]:
        return {
            "trace_v5": bool(trace_v5),
            "behavior_logprobs": bool(behavior_logprobs),
            "strict_prefix": bool(self.strict_prefix),
            "masking": bool(self.masking),
            "wire_objects": bool(self.wire_objects),
            "artifact_reference": bool(self.artifact_by_reference),
            "tokens_in_tokens_out": bool(self.tokens_in_tokens_out),
        }


@dataclass(frozen=True, slots=True)
class RewardDeclaration(JsonDataclassMixin):
    """Container-authoritative reward. ``reward_relation`` is not declared here:
    it belongs to the topology, so the two can never disagree."""

    authority: str
    binds_trace_digest: bool
    quiescence: bool
    horizon_clipping: bool
    channels: tuple[str, ...]
    evaluation_plan_id: str
    settlement_window_seconds: float = 0.0
    deferred_scoring: bool = False

    def __post_init__(self) -> None:
        if self.authority != CONTAINER_REWARD_AUTHORITY:
            raise CispoCapabilityError(
                f"reward authority must be {CONTAINER_REWARD_AUTHORITY!r}, got "
                f"{self.authority!r}; the executor never scores locally"
            )
        if not self.channels:
            raise CispoCapabilityError("reward must declare at least one channel")
        if not str(self.evaluation_plan_id or "").strip():
            raise CispoCapabilityError("reward must declare evaluation_plan_id")
        if self.settlement_window_seconds < 0:
            raise CispoCapabilityError("settlement_window_seconds may not be negative")
        if not (self.quiescence or self.horizon_clipping):
            raise CispoCapabilityError(
                "a container that cannot quiesce at the horizon must declare "
                "horizon_clipping as its substitute; neither leaves the reward "
                "describing whatever was still running later"
            )

    def to_payload(self, *, reward_relation: str) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "binds_trace_digest": bool(self.binds_trace_digest),
            "quiescence": bool(self.quiescence),
            "horizon_clipping": bool(self.horizon_clipping),
            "channels": [str(item) for item in self.channels],
            "reward_relation": reward_relation,
            "evaluation_plan_id": self.evaluation_plan_id,
            "settlement_window_seconds": float(self.settlement_window_seconds),
            "deferred_scoring": bool(self.deferred_scoring),
        }


@dataclass(frozen=True, slots=True)
class RecoveryDeclaration(JsonDataclassMixin):
    """Restart behavior. Both are mandatory: a container that cannot discard a
    stale item will happily train on one."""

    restart: bool
    stale_discard: bool

    def to_payload(self) -> dict[str, Any]:
        return {"restart": bool(self.restart), "stale_discard": bool(self.stale_discard)}


@dataclass(frozen=True, slots=True)
class CispoRuntimeDeclaration(JsonDataclassMixin):
    """Everything a target declares for CISPO that the capability surface
    cannot already express.

    A target supplies exactly this and its existing
    :class:`RuntimeCapabilitySurface`; the document is assembled from the two.
    """

    container_id: str
    container_image_digest: str
    renderer_profile: RendererProfileDeclaration
    discovery: DiscoveryDeclaration
    policy: PolicyBindingDeclaration
    lifecycle: LifecycleDeclaration
    evidence: EvidenceDeclaration
    reward: RewardDeclaration
    recovery: RecoveryDeclaration
    topology: TopologyDeclaration
    #: Tolerated skew between the two clocks. The horizon is the instant the
    #: reward is read, so a wall-clock horizon with skew beyond this is a
    #: rejected clause rather than a warning.
    clock_skew_tolerance_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not str(self.container_id or "").strip():
            raise CispoCapabilityError("declaration must name container_id")
        if not str(self.container_image_digest or "").strip():
            raise CispoCapabilityError(
                "declaration must name container_image_digest; a receipt has to be "
                "able to name the exact build it ran against"
            )
        if self.clock_skew_tolerance_seconds < 0:
            raise CispoCapabilityError("clock_skew_tolerance_seconds may not be negative")


# --------------------------------------------------------------------------- #
# The capability document
# --------------------------------------------------------------------------- #


def canonical_capability_hash(document: Mapping[str, Any]) -> str:
    """Hash every field but the offered hash itself.

    Byte-for-byte the optimizer's ``canonical_capability_hash``: sorted keys,
    compact separators, no ASCII escaping, ``sha256:`` prefix.
    """

    unhashed = {key: value for key, value in document.items() if key != "capability_hash"}
    raw = json.dumps(unhashed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"sha256:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


def _assert_mandatory(capabilities: RuntimeCapabilitySurface) -> None:
    """Refuse to build a document a runtime cannot actually honor.

    Each check below maps to a mandatory clause. A false flag here is a
    capability the runtime *did not declare*; emitting ``true`` anyway would
    produce a run that starts successfully and then trains on evidence that was
    never valid, which is strictly worse than not starting.
    """

    if not capabilities.trace_support:
        raise CispoCapabilityError(
            "evidence.trace_v5 is mandatory but the runtime does not declare "
            "trace_support; no trainable evidence can be produced"
        )
    if not capabilities.token_emission.logprobs:
        raise CispoCapabilityError(
            "evidence.behavior_logprobs is mandatory but the runtime does not "
            "declare token_emission.logprobs; an importance ratio has no denominator"
        )
    if not capabilities.reward_support:
        raise CispoCapabilityError(
            "reward.authority is mandatory but the runtime does not declare "
            "reward_support; the executor never scores locally"
        )
    if not capabilities.state_support:
        raise CispoCapabilityError(
            "the declared rollout_state route must answer, but the runtime does "
            "not declare state_support"
        )
    if not capabilities.terminate_support:
        raise CispoCapabilityError(
            "lifecycle.cancellation is mandatory but the runtime does not declare "
            "terminate_support"
        )


def _assert_consistent(
    declaration: CispoRuntimeDeclaration, capabilities: RuntimeCapabilitySurface
) -> None:
    """Refuse a declaration that contradicts the surface it sits beside."""

    if declaration.evidence.artifact_by_reference and not capabilities.artifact_support:
        raise CispoCapabilityError(
            "evidence.artifact_reference was declared but the runtime does not "
            "declare artifact_support"
        )
    if declaration.evidence.tokens_in_tokens_out and not capabilities.token_emission.token_ids:
        raise CispoCapabilityError(
            "evidence.tokens_in_tokens_out was declared but the runtime does not "
            "declare token_emission.token_ids"
        )
    if (
        declaration.policy.binding_transport == "tokens_in_tokens_out"
        and not declaration.evidence.tokens_in_tokens_out
    ):
        raise CispoCapabilityError(
            "policy.binding_transport is tokens_in_tokens_out but evidence does "
            "not declare it"
        )
    if len(declaration.topology.agent_instances) > 1 and not capabilities.multi_actor:
        raise CispoCapabilityError(
            "the declared topology rosters more than one agent instance but the "
            "runtime does not declare multi_actor"
        )


def build_cispo_capability_document(
    declaration: CispoRuntimeDeclaration,
    capabilities: RuntimeCapabilitySurface,
    *,
    routes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble the hashed capability document from what the runtime declares.

    Nothing here is invented. The evidence, lifecycle and reward flags come
    from ``capabilities`` where the surface already carries them and from
    ``declaration`` where it does not; the renderer profile, topology
    reference, horizon, actuation model, reward relation, minimum viable
    roster, advertised concurrency and lease obligation come from
    ``declaration``. The lease's ``heartbeat_route`` is taken from the declared
    renew route so it can never name a path this container does not serve.
    """

    if not isinstance(declaration, CispoRuntimeDeclaration):
        raise CispoContractError(
            "capability document needs a CispoRuntimeDeclaration, got "
            f"{type(declaration).__name__}"
        )
    if not isinstance(capabilities, RuntimeCapabilitySurface):
        raise CispoContractError(
            "capability document needs a RuntimeCapabilitySurface, got "
            f"{type(capabilities).__name__}"
        )
    _assert_mandatory(capabilities)
    _assert_consistent(declaration, capabilities)

    table = dict(routes or cispo_declared_routes())
    heartbeat_route = str(table.get("rollout_renew_route") or "").strip()
    if not heartbeat_route.startswith("/"):
        raise CispoContractError(
            "the lease obligation needs a declared, absolute rollout_renew_route "
            f"to name as its heartbeat route, got {heartbeat_route!r}"
        )

    topology = declaration.topology
    document: dict[str, Any] = {
        "schema_version": CISPO_CAPABILITY_SCHEMA_VERSION,
        "contract_version": CISPO_OPTIMIZER_CONTRACT_VERSION,
        "container_id": declaration.container_id,
        "container_image_digest": declaration.container_image_digest,
        "renderer_profile": declaration.renderer_profile.to_payload(),
        "discovery": declaration.discovery.to_payload(),
        "policy": declaration.policy.to_payload(),
        "lifecycle": declaration.lifecycle.to_payload(
            heartbeat_route=heartbeat_route,
            supports_cancellation=bool(capabilities.terminate_support),
            supports_pause_resume=bool(
                capabilities.pause_support and capabilities.resume_support
            ),
        ),
        "evidence": declaration.evidence.to_payload(
            trace_v5=bool(capabilities.trace_support),
            behavior_logprobs=bool(capabilities.token_emission.logprobs),
        ),
        "reward": declaration.reward.to_payload(reward_relation=topology.reward_relation),
        "recovery": declaration.recovery.to_payload(),
        "topology": topology.to_payload(),
        "topology_ref": topology.topology_id,
        "actuation_model": topology.actuation_model,
        "horizon": topology.horizon.to_payload(),
        "advertised_concurrency": int(declaration.lifecycle.advertised_concurrency),
        "minimum_viable_roster": {
            team.team_id: int(team.minimum_viable_roster) for team in topology.teams
        },
        "clock": {
            "skew_tolerance_seconds": float(declaration.clock_skew_tolerance_seconds)
        },
        "routes": dict(sorted(table.items())),
    }
    document["capability_hash"] = canonical_capability_hash(document)
    return document


# --------------------------------------------------------------------------- #
# Ports: streams B and C implement these
# --------------------------------------------------------------------------- #


@runtime_checkable
class CispoAdmissionPort(Protocol):
    """Everything that happens before an attempt is admitted. Stream B owns it.

    Discovery, the two-sided handshake, and policy binding. Every method may be
    implemented synchronously or as a coroutine; the HTTP adapter awaits the
    result either way. Every method returns a JSON object, never a bare
    boolean: a verdict without a reason cannot be put in a receipt.

    Implementations raise :class:`CispoNotImplementedError` for an operation
    they do not serve; any other failure should surface as the container's own
    typed refusal.
    """

    def cispo_health(self) -> Mapping[str, Any]:
        """Liveness plus the build identity a receipt has to name.

        Returns at least ``status``, ``container_version`` and
        ``container_image_digest``. Called before anything else and before any
        spend, so it must not touch the queue.
        """
        ...

    def cispo_handshake(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Answer the executor's requirement document, clause by clause.

        ``request`` is a ``cispo.handshake.v1`` requirement document: run id,
        optimizer identity, policy and renderer profile, requested clauses,
        topology expectations, run plan, taskset selection, and the executor's
        clock reading.

        Returns a ``cispo.handshake.v1`` response carrying ``handshake_id``,
        ``accepted``, a ``clauses`` list of ``{clause_id, verdict, reason}``
        where verdict is one of accepted/degraded/rejected/unsupported, the
        ``obligations`` the container commits to, ``taskset_resolution`` with
        one content digest and ``topology_ref`` per requested task,
        ``capability_hash`` echoing the document this agreement was built on,
        ``agreement_digest`` binding both documents, ``expires_at``, and the
        container's ``clock`` with the measured skew.

        The handshake is the run's admission ticket: every later attempt
        carries its ``handshake_id`` and must be refused if that handshake is
        absent, expired, revoked, or bound to a different agreement digest.
        """
        ...

    def cispo_taskset(self) -> Mapping[str, Any]:
        """Taskset identity and the splits this container declares.

        Returns at least ``taskset_id``, ``taskset_version`` and ``splits``.
        Discovery, not configuration: the executor selects from what comes
        back rather than being told a taskset out of band.
        """
        ...

    def cispo_taskset_tasks(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Resolve requested task ids to rows.

        ``request`` carries ``task_ids`` and optionally ``split``. Returns
        ``tasks``: exactly one duplicate-free row per requested id, each row
        naming its own ``topology_ref`` and ``content_digest``, so a family
        whose variants differ in roster resolves per task rather than per
        container. An id the container does not hold is a refusal, never a
        silently shorter list.
        """
        ...

    def cispo_topology(self, topology_id: str) -> Mapping[str, Any]:
        """The full declared topology for one ``topology_id``.

        Returns the roster, teams, communication channels, turn and actuation
        model, horizon, parameter groups, and each team's minimum viable
        roster. The executor binds exactly this and never infers a roster from
        an agent count.
        """
        ...

    def cispo_bind_policy(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Bind one sampler for one instance.

        ``request`` names the handshake, the policy revision, the sampling
        transport, and a session-scoped sampler origin -- never an embedded
        credential. Returns ``config_id``, the resolved ``renderer_profile``,
        and sampler readiness. The binding is immutable once admitted: a
        revision may not be retired while an attempt is still sampling from it.
        """
        ...

    def cispo_bind_policy_set(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Bind every instance of a joint episode in one atomic operation.

        ``request`` carries one binding per agent instance, trainable and
        pinned-opponent alike, each opponent pinned to an immutable identity
        rather than an alias. Returns a ``policy_set_revision`` plus the
        per-instance resolved bindings. Either the whole roster binds or none
        of it does: no episode may start half-bound.
        """
        ...


@runtime_checkable
class CispoRolloutPort(Protocol):
    """Everything from submission to evidence. Stream C owns it.

    Submission is asynchronous and idempotent, the lease is the liveness
    contract, and the reward is the container's own. Every method may be
    implemented synchronously or as a coroutine; the HTTP adapter awaits the
    result either way. Implementations raise
    :class:`CispoNotImplementedError` for an operation they do not serve.
    """

    def cispo_submit_rollout(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Accept one attempt for asynchronous execution.

        ``request`` carries the ``handshake_id``, an ``idempotency_key``, the
        task row, the policy or policy-set binding, and the correlation fields
        (run, group, sample index, seed, policy revision, agent instance, team,
        policy set revision, match set revision) that must be echoed verbatim.

        Returns ``rollout_id``, the initial ``lease_expires_at``, and the
        accepted correlation echo, for a 202. Resubmitting the same
        idempotency key returns the same logical attempt rather than starting a
        second one -- a lost HTTP response must not double-spend.
        """
        ...

    def cispo_rollout_state(self, rollout_id: str) -> Mapping[str, Any]:
        """Current state of one attempt, for polling and straggler detection.

        Returns ``state``, ``lease_expires_at``, and per-instance liveness for
        a joint episode. Cheap enough to poll: it must not compute a reward or
        seal a trace.
        """
        ...

    def cispo_rollout_events(
        self,
        rollout_id: str,
        *,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> Mapping[str, Any]:
        """One ordered page of events, resumable from ``cursor``.

        Returns ``events`` (each with a monotone cursor value) and
        ``next_cursor``. The cursor is monotone and durable across a restart,
        so a reconnecting executor resumes rather than replays. Exactly one
        terminal event is ever emitted per attempt.
        """
        ...

    def cispo_renew_lease(
        self, rollout_id: str, request: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Extend the lease on a still-live attempt.

        ``request`` carries the ``handshake_id`` and may carry the requested
        extension. Returns the new ``lease_expires_at``. Hour-scale episodes
        must not depend on an open HTTP request, so this is the heartbeat the
        capability document advertises; renewing a lapsed or terminal lease is
        a refusal.
        """
        ...

    def cispo_finalize_rollout(
        self, rollout_id: str, request: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Clip the attempt at its horizon and attest quiescence.

        Returns the horizon-clipped state snapshot, the horizon actually
        applied, and either a quiescence attestation or the declared clipping
        substitute. The reward must describe the horizon, not whatever a
        policy-authored background process was still doing afterwards.
        """
        ...

    def cispo_terminate_rollout(
        self, rollout_id: str, request: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Cancel an attempt, exactly once.

        Returns the terminal state and the cancellation reason. Terminating an
        already-terminal attempt is idempotent and does not emit a second
        terminal event.
        """
        ...

    def cispo_rollout_trace(self, rollout_id: str) -> Mapping[str, Any]:
        """The sealed Trace V5 evidence for one attempt.

        Returns either the trace inline or a reference plus its digest, along
        with the ``trace_digest`` the reward receipt binds to. Spans carry
        behavior logprobs, the renderer-profile stamp, masks, and the wire
        objects; probe evidence is explicitly marked non-trainable.
        """
        ...

    def cispo_rollout_artifacts(self, rollout_id: str) -> Mapping[str, Any]:
        """Inventory of the attempt's artifacts.

        Returns one row per artifact with its content digest, size, media type,
        and a fetch handle. Multi-hundred-megabyte recordings are referenced,
        never inlined into a job store.
        """
        ...

    def cispo_reward(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """The container-authoritative reward receipt.

        ``request`` names the ``rollout_id`` and the ``trace_digest`` the
        caller believes it is scoring. Returns a receipt bound to both, with
        per-team channels, the horizon applied, whether the state was clipped
        or quiesced, and the settlement window. A zero reward is a scored
        result; an absent reward is an evidence failure, and the two are never
        conflated.
        """
        ...


class NullCispoAdmissionPort:
    """Declared, not implemented. Every call is a typed 501.

    Route presence and behavior support are separate questions: this container
    advertises the whole surface -- an executor must be able to see the shape
    it would talk to -- and answers 501 until stream B supplies a real port.
    """

    __slots__ = ()

    def _refuse(self, operation: str) -> Mapping[str, Any]:
        raise CispoNotImplementedError("CispoAdmissionPort", operation)

    def cispo_health(self) -> Mapping[str, Any]:
        return self._refuse("cispo_health")

    def cispo_handshake(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        del request
        return self._refuse("cispo_handshake")

    def cispo_taskset(self) -> Mapping[str, Any]:
        return self._refuse("cispo_taskset")

    def cispo_taskset_tasks(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        del request
        return self._refuse("cispo_taskset_tasks")

    def cispo_topology(self, topology_id: str) -> Mapping[str, Any]:
        del topology_id
        return self._refuse("cispo_topology")

    def cispo_bind_policy(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        del request
        return self._refuse("cispo_bind_policy")

    def cispo_bind_policy_set(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        del request
        return self._refuse("cispo_bind_policy_set")


class NullCispoRolloutPort:
    """Declared, not implemented. Every call is a typed 501 until stream C
    supplies a real port."""

    __slots__ = ()

    def _refuse(self, operation: str) -> Mapping[str, Any]:
        raise CispoNotImplementedError("CispoRolloutPort", operation)

    def cispo_submit_rollout(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        del request
        return self._refuse("cispo_submit_rollout")

    def cispo_rollout_state(self, rollout_id: str) -> Mapping[str, Any]:
        del rollout_id
        return self._refuse("cispo_rollout_state")

    def cispo_rollout_events(
        self,
        rollout_id: str,
        *,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> Mapping[str, Any]:
        del rollout_id, cursor, limit
        return self._refuse("cispo_rollout_events")

    def cispo_renew_lease(
        self, rollout_id: str, request: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        del rollout_id, request
        return self._refuse("cispo_renew_lease")

    def cispo_finalize_rollout(
        self, rollout_id: str, request: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        del rollout_id, request
        return self._refuse("cispo_finalize_rollout")

    def cispo_terminate_rollout(
        self, rollout_id: str, request: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        del rollout_id, request
        return self._refuse("cispo_terminate_rollout")

    def cispo_rollout_trace(self, rollout_id: str) -> Mapping[str, Any]:
        del rollout_id
        return self._refuse("cispo_rollout_trace")

    def cispo_rollout_artifacts(self, rollout_id: str) -> Mapping[str, Any]:
        del rollout_id
        return self._refuse("cispo_rollout_artifacts")

    def cispo_reward(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        del request
        return self._refuse("cispo_reward")


# --------------------------------------------------------------------------- #
# Runtime discovery helpers
# --------------------------------------------------------------------------- #


def cispo_declaration_of(runtime: Any) -> CispoRuntimeDeclaration | None:
    """The runtime's CISPO declaration, or ``None`` if it makes none.

    A runtime opts into the CISPO surface by exposing
    ``cispo_declaration()``. Anything else it returns is a configuration error
    rather than a silent opt-out.
    """

    handler = getattr(runtime, "cispo_declaration", None)
    if not callable(handler):
        return None
    value = handler()
    if value is None:
        return None
    if not isinstance(value, CispoRuntimeDeclaration):
        raise CispoContractError(
            "runtime.cispo_declaration() must return a CispoRuntimeDeclaration, got "
            f"{type(value).__name__}"
        )
    return value


def cispo_admission_port(runtime: Any) -> CispoAdmissionPort:
    """The runtime's admission port, or the null port that answers 501."""

    handler = getattr(runtime, "cispo_admission", None)
    if callable(handler):
        value = handler()
        if value is not None:
            return value
    return NullCispoAdmissionPort()


def cispo_rollout_port(runtime: Any) -> CispoRolloutPort:
    """The runtime's rollout/evidence port, or the null port that answers 501."""

    handler = getattr(runtime, "cispo_rollouts", None)
    if callable(handler):
        value = handler()
        if value is not None:
            return value
    return NullCispoRolloutPort()


def cispo_capability_document_for(
    runtime: Any, *, routes: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Build the document from a runtime's declaration and capability surface."""

    declaration = cispo_declaration_of(runtime)
    if declaration is None:
        raise CispoCapabilityError(
            "runtime declares no CISPO capability document; there is nothing to "
            "advertise and nothing to hash"
        )
    metadata = runtime.metadata()
    capabilities = getattr(metadata, "capabilities", None)
    if not isinstance(capabilities, RuntimeCapabilitySurface):
        raise CispoContractError(
            "runtime.metadata().capabilities must be a RuntimeCapabilitySurface"
        )
    return build_cispo_capability_document(declaration, capabilities, routes=routes)


__all__: Sequence[str] = (
    "ACTUATION_MODELS",
    "AgentInstanceDeclaration",
    "CHANNEL_SCOPES",
    "CISPO_CAPABILITY_SCHEMA_VERSION",
    "CISPO_DECLARED_ROUTES",
    "CISPO_OPTIMIZER_CONTRACT_VERSION",
    "CISPO_ROUTE_METHODS",
    "CISPO_ROUTE_PLACEHOLDERS",
    "CISPO_ROUTE_PREFIX",
    "CONTAINER_REWARD_AUTHORITY",
    "CispoAdmissionPort",
    "CispoCapabilityError",
    "CispoContractError",
    "CispoNotImplementedError",
    "CispoRolloutPort",
    "CispoRuntimeDeclaration",
    "CommunicationChannelDeclaration",
    "DiscoveryDeclaration",
    "EvidenceDeclaration",
    "HORIZON_KINDS",
    "HorizonDeclaration",
    "LeaseDeclaration",
    "LifecycleDeclaration",
    "NullCispoAdmissionPort",
    "NullCispoRolloutPort",
    "PolicyBindingDeclaration",
    "REWARD_RELATIONS",
    "RecoveryDeclaration",
    "RendererProfileDeclaration",
    "RewardDeclaration",
    "SAMPLING_TRANSPORTS",
    "TURN_MODELS",
    "TeamDeclaration",
    "TopologyDeclaration",
    "WIRE_APIS",
    "build_cispo_capability_document",
    "canonical_capability_hash",
    "cispo_admission_port",
    "cispo_capability_document_for",
    "cispo_declaration_of",
    "cispo_declared_routes",
    "cispo_optimizer_contract",
    "cispo_rollout_port",
)
