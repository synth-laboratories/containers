"""CISPO clause registry, canonical identities and declared runtime facts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
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
    canary_digest: str = ""

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
            "canary_digest": self.canary_digest,
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
    seed: int | None = None

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
                canary_digest=str(renderer_raw.get("canary_digest") or ""),
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
