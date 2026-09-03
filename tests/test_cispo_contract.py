"""The CISPO advertisement and the hashed capability document.

Three things have to hold for a container to be safely discoverable.

Every declared route must be present and absolute, because the executor calls
only what it reads from ``/metadata`` and refuses a route it cannot resolve.

The capability document must be assembled from what the runtime actually
declares. A runtime that declares little gets a small document; a runtime that
declares much gets a large one; a runtime that cannot declare a mandatory
capability gets no document at all. That last case is the load-bearing one: a
plausible default there buys a run that starts successfully and then trains on
evidence that was never valid.

And the content hash must be stable over key order and unstable over any value,
byte-identical to the optimizer's ``canonical_capability_hash``, because a
preflight and every handshake built on it fail closed on a hash change.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from synth_containers.capabilities import (
    RuntimeCapabilitySurface,
    TokenEmissionCapabilities,
)
from synth_containers.cispo_contract import (
    CISPO_CAPABILITY_SCHEMA_VERSION,
    CISPO_DECLARED_ROUTES,
    CISPO_OPTIMIZER_CONTRACT_VERSION,
    CISPO_ROUTE_METHODS,
    CISPO_ROUTE_PREFIX,
    AgentInstanceDeclaration,
    CispoCapabilityError,
    CispoContractError,
    CispoRuntimeDeclaration,
    CommunicationChannelDeclaration,
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
    canonical_capability_hash,
    cispo_declared_routes,
    cispo_optimizer_contract,
)
from synth_containers.ontology import CapabilityLevel, PrimitiveProtocol

# The seventeen keys of the declared contract, verbatim from the note.
DECLARED_KEYS = (
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

CANONICAL_TABLE = {
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

PROFILE = RendererProfileDeclaration(
    profile_id="renderers.profile.low.v1",
    package="renderers",
    package_version="0.4.0",
    config_digest="sha256:config",
    tokenizer_id="tokenizer-a",
    tokenizer_digest="sha256:tokenizer",
    stop_token_ids=(200002,),
)


def minimal_surface(**overrides: Any) -> RuntimeCapabilitySurface:
    """A runtime that declares only what CISPO makes mandatory."""

    values: dict[str, Any] = {
        "state_support": True,
        "terminate_support": True,
        "trace_support": True,
        "reward_support": True,
        "token_emission": TokenEmissionCapabilities(logprobs=True),
    }
    values.update(overrides)
    return RuntimeCapabilitySurface(**values)


def rich_surface(**overrides: Any) -> RuntimeCapabilitySurface:
    """A runtime that declares everything the document can carry."""

    values: dict[str, Any] = {
        "state_support": True,
        "pause_support": True,
        "resume_support": True,
        "terminate_support": True,
        "trace_support": True,
        "reward_support": True,
        "artifact_support": True,
        "multi_actor": True,
        "protocol_fidelity": {PrimitiveProtocol.MULTI_ACTOR: CapabilityLevel.NATIVE},
        "token_emission": TokenEmissionCapabilities(
            token_ids=True, tokens=True, logprobs=True, old_logprobs=True
        ),
    }
    values.update(overrides)
    return RuntimeCapabilitySurface(**values)


def minimal_declaration(**overrides: Any) -> CispoRuntimeDeclaration:
    """One trainable instance, one team, a wall-clock horizon, one channel."""

    values: dict[str, Any] = {
        "container_id": "container-small",
        "container_image_digest": "sha256:image-small",
        "renderer_profile": PROFILE,
        "discovery": DiscoveryDeclaration(
            taskset_id="taskset-small",
            taskset_version="1",
            splits=("train",),
            task_content_digests=True,
            deterministic_lookup=True,
            duplicate_free=True,
        ),
        "policy": PolicyBindingDeclaration(
            binding_transport="message_in_capture_out",
            wire_api="chat_completions",
            session_scoped_sampler_origin=True,
            embeds_credentials=False,
            revision_immutable_after_admission=True,
            records_policy_revision=True,
        ),
        "lifecycle": LifecycleDeclaration(
            advertised_concurrency=1,
            supports_idempotency=True,
            exactly_one_terminal_result=True,
            lease=LeaseDeclaration(ttl_seconds=120.0, renewable=False),
        ),
        "evidence": EvidenceDeclaration(strict_prefix=True, masking=True, wire_objects=True),
        "reward": RewardDeclaration(
            authority="container",
            binds_trace_digest=True,
            quiescence=False,
            horizon_clipping=True,
            channels=("outcome",),
            evaluation_plan_id="plan-small",
        ),
        "recovery": RecoveryDeclaration(restart=True, stale_discard=True),
        "topology": TopologyDeclaration(
            topology_id="topology-small",
            turn_model="sequential",
            actuation_model="direct_action",
            reward_relation="cooperative",
            agent_instances=(
                AgentInstanceDeclaration(
                    agent_instance_id="instance-1",
                    role_id="role-1",
                    policy_type_id="policy-1",
                    team_id="team-1",
                    trainable=True,
                ),
            ),
            teams=(TeamDeclaration(team_id="team-1", trainable=True, minimum_viable_roster=1),),
            horizon=HorizonDeclaration(horizon_kind="wall_clock", value=600.0),
        ),
    }
    values.update(overrides)
    return CispoRuntimeDeclaration(**values)


def rich_declaration(**overrides: Any) -> CispoRuntimeDeclaration:
    """Two teams, four instances, a pinned opponent, a step horizon."""

    values: dict[str, Any] = {
        "container_id": "container-large",
        "container_image_digest": "sha256:image-large",
        "renderer_profile": RendererProfileDeclaration(
            profile_id="renderers.profile.high.v1",
            package="renderers",
            package_version="0.4.0",
            config_digest="sha256:config-high",
            tokenizer_id="tokenizer-a",
            tokenizer_digest="sha256:tokenizer",
            stop_token_ids=(200002, 199999),
            modalities=("text", "image"),
        ),
        "discovery": DiscoveryDeclaration(
            taskset_id="taskset-large",
            taskset_version="7",
            splits=("train", "eval", "holdout"),
            task_content_digests=True,
            deterministic_lookup=True,
            duplicate_free=True,
        ),
        "policy": PolicyBindingDeclaration(
            binding_transport="tokens_in_tokens_out",
            wire_api="responses",
            session_scoped_sampler_origin=True,
            embeds_credentials=False,
            revision_immutable_after_admission=True,
            records_policy_revision=True,
            probe_binding=True,
        ),
        "lifecycle": LifecycleDeclaration(
            advertised_concurrency=30,
            supports_idempotency=True,
            exactly_one_terminal_result=True,
            lease=LeaseDeclaration(
                ttl_seconds=900.0, renewable=True, straggler_grace_seconds=120.0
            ),
        ),
        "evidence": EvidenceDeclaration(
            strict_prefix=True,
            masking=True,
            wire_objects=True,
            artifact_by_reference=True,
            tokens_in_tokens_out=True,
        ),
        "reward": RewardDeclaration(
            authority="container",
            binds_trace_digest=True,
            quiescence=True,
            horizon_clipping=True,
            channels=("rank", "margin"),
            evaluation_plan_id="plan-large",
            settlement_window_seconds=150.0,
            deferred_scoring=True,
        ),
        "recovery": RecoveryDeclaration(restart=True, stale_discard=True),
        "topology": TopologyDeclaration(
            topology_id="topology-large",
            turn_model="concurrent_realtime",
            actuation_model="deferred_program",
            reward_relation="competitive_rank",
            agent_instances=(
                AgentInstanceDeclaration(
                    agent_instance_id="instance-1",
                    role_id="role-a",
                    policy_type_id="policy-trainable",
                    team_id="team-trainable",
                    trainable=True,
                ),
                AgentInstanceDeclaration(
                    agent_instance_id="instance-2",
                    role_id="role-b",
                    policy_type_id="policy-trainable",
                    team_id="team-trainable",
                    trainable=True,
                ),
                AgentInstanceDeclaration(
                    agent_instance_id="instance-3",
                    role_id="role-a",
                    policy_type_id="policy-pinned",
                    team_id="team-pinned",
                    trainable=False,
                    pinned_identity="checkpoint:sha256:opponent",
                ),
                AgentInstanceDeclaration(
                    agent_instance_id="instance-4",
                    role_id="role-b",
                    policy_type_id="policy-pinned",
                    team_id="team-pinned",
                    trainable=False,
                    pinned_identity="checkpoint:sha256:opponent",
                ),
            ),
            teams=(
                TeamDeclaration(
                    team_id="team-trainable", trainable=True, minimum_viable_roster=2
                ),
                TeamDeclaration(team_id="team-pinned", trainable=False, minimum_viable_roster=1),
            ),
            horizon=HorizonDeclaration(
                horizon_kind="steps",
                value=500.0,
                seconds_per_unit=4.0,
                time_dilation=4.0,
                grace_seconds=30.0,
            ),
            communication_channels=(
                CommunicationChannelDeclaration(channel_id="team-chat", scope="intra_team"),
                CommunicationChannelDeclaration(
                    channel_id="taunt", scope="cross_team", trainable_for_author=False
                ),
            ),
            parameter_groups={"policy-trainable": "group-1", "policy-pinned": "group-pinned"},
        ),
        "clock_skew_tolerance_seconds": 2.0,
    }
    values.update(overrides)
    return CispoRuntimeDeclaration(**values)


# --------------------------------------------------------------------------- #
# Contract version and route declaration
# --------------------------------------------------------------------------- #


def test_contract_version_is_the_declared_one() -> None:
    assert CISPO_OPTIMIZER_CONTRACT_VERSION == "synth_optimizers.cispo.v1"
    assert cispo_optimizer_contract()["version"] == "synth_optimizers.cispo.v1"


def test_unprefixed_table_is_the_canonical_declaration_verbatim() -> None:
    """The canonical paths, exactly as the contract declares them."""

    assert cispo_declared_routes(prefix="") == CANONICAL_TABLE
    assert dict(CISPO_DECLARED_ROUTES) == CANONICAL_TABLE


def test_every_declared_route_is_present_and_absolute() -> None:
    block = cispo_optimizer_contract()
    assert set(block) == {"version", *DECLARED_KEYS}
    for name in DECLARED_KEYS:
        route = block[name]
        assert route.startswith("/"), f"{name} must be absolute, got {route!r}"
        assert route.strip() == route
        assert route.startswith(CISPO_ROUTE_PREFIX + "/")


def test_no_declared_route_carries_an_unsubstitutable_placeholder() -> None:
    """A placeholder the executor has no value for makes a route unresolvable."""

    known = {"rollout_id", "topology_id"}
    for name in DECLARED_KEYS:
        route = cispo_optimizer_contract()[name]
        found = {chunk.split("}")[0] for chunk in route.split("{")[1:]}
        assert found <= known, f"{name} declares unknown placeholders {found - known}"
        assert route.count("{") == route.count("}")


def test_templated_routes_address_the_identifier_they_need() -> None:
    block = cispo_optimizer_contract()
    for name in (
        "rollout_state_route",
        "rollout_events_route",
        "rollout_renew_route",
        "rollout_finalize_route",
        "rollout_terminate_route",
        "trace_route",
        "artifacts_route",
    ):
        assert "{rollout_id}" in block[name], name
    assert "{topology_id}" in block["topology_route"]
    for name in ("health_route", "capabilities_route", "handshake_route", "reward_route"):
        assert "{" not in block[name], name


def test_route_methods_cover_every_declared_route() -> None:
    assert set(CISPO_ROUTE_METHODS) == set(DECLARED_KEYS)


def test_a_relative_prefix_is_refused() -> None:
    with pytest.raises(CispoContractError):
        cispo_optimizer_contract(prefix="cispo")


# --------------------------------------------------------------------------- #
# The capability document, built from real declarations
# --------------------------------------------------------------------------- #


def test_document_from_a_runtime_that_declares_little() -> None:
    document = build_cispo_capability_document(minimal_declaration(), minimal_surface())

    assert document["schema_version"] == CISPO_CAPABILITY_SCHEMA_VERSION
    assert document["contract_version"] == CISPO_OPTIMIZER_CONTRACT_VERSION
    assert document["container_id"] == "container-small"
    assert document["container_image_digest"] == "sha256:image-small"

    # Flags the surface carries are read from the surface, not from the
    # declaration, so a runtime cannot claim what it does not publish.
    assert document["lifecycle"]["supports_cancellation"] is True
    assert document["lifecycle"]["supports_pause_resume"] is False
    assert document["evidence"]["trace_v5"] is True
    assert document["evidence"]["behavior_logprobs"] is True
    assert document["evidence"]["artifact_reference"] is False
    assert document["evidence"]["tokens_in_tokens_out"] is False

    # A non-renewable lease says so; it is not upgraded to renewable because
    # the horizon happens to be longer than the TTL.
    assert document["lifecycle"]["lease"] == {
        "ttl_seconds": 120.0,
        "renewable": False,
        "heartbeat_route": cispo_declared_routes()["rollout_renew_route"],
    }
    assert document["lifecycle"]["supports_lease_renewal"] is False
    assert document["advertised_concurrency"] == 1
    assert document["topology_ref"] == "topology-small"
    assert document["actuation_model"] == "direct_action"
    assert document["reward"]["reward_relation"] == "cooperative"
    assert document["minimum_viable_roster"] == {"team-1": 1}
    assert document["horizon"]["horizon_kind"] == "wall_clock"
    assert document["horizon"]["seconds_per_unit"] == 1.0
    assert document["horizon"]["horizon_seconds"] == 600.0
    assert document["topology"]["communication_channels"] == []


def test_document_from_a_runtime_that_declares_much() -> None:
    document = build_cispo_capability_document(rich_declaration(), rich_surface())

    assert document["container_id"] == "container-large"
    assert document["advertised_concurrency"] == 30
    assert document["lifecycle"]["supports_pause_resume"] is True
    assert document["lifecycle"]["straggler_grace_seconds"] == 120.0
    assert document["lifecycle"]["lease"]["renewable"] is True
    assert document["evidence"]["artifact_reference"] is True
    assert document["evidence"]["tokens_in_tokens_out"] is True
    assert document["policy"]["probe_binding"] is True
    assert document["reward"]["channels"] == ["rank", "margin"]
    assert document["reward"]["reward_relation"] == "competitive_rank"
    assert document["actuation_model"] == "deferred_program"
    assert document["topology"]["turn_model"] == "concurrent_realtime"
    assert document["minimum_viable_roster"] == {"team-trainable": 2, "team-pinned": 1}
    assert len(document["topology"]["agent_instances"]) == 4
    assert [item["channel_id"] for item in document["topology"]["communication_channels"]] == [
        "team-chat",
        "taunt",
    ]
    assert document["clock"]["skew_tolerance_seconds"] == 2.0
    assert document["renderer_profile"]["modalities"] == ["text", "image"]
    assert document["discovery"]["splits"] == ["train", "eval", "holdout"]


def test_a_step_horizon_declares_its_conversion() -> None:
    """500 steps is not 500 seconds, and the document has to say which."""

    document = build_cispo_capability_document(rich_declaration(), rich_surface())
    horizon = document["horizon"]
    assert horizon["horizon_kind"] == "steps"
    assert horizon["value"] == 500.0
    assert horizon["seconds_per_unit"] == 4.0
    assert horizon["horizon_seconds"] == 2000.0
    assert document["topology"]["horizon"]["seconds_per_unit"] == 4.0


def test_a_unit_horizon_without_a_conversion_is_refused() -> None:
    with pytest.raises(CispoCapabilityError, match="seconds_per_unit"):
        HorizonDeclaration(horizon_kind="steps", value=500.0)


def test_the_heartbeat_route_is_the_declared_renew_route() -> None:
    """The lease obligation may not name a path this container does not serve."""

    routes = cispo_declared_routes(prefix="/elsewhere")
    document = build_cispo_capability_document(
        minimal_declaration(), minimal_surface(), routes=routes
    )
    assert (
        document["lifecycle"]["lease"]["heartbeat_route"]
        == "/elsewhere/rollouts/{rollout_id}/renew"
    )


# --------------------------------------------------------------------------- #
# Refusals: a capability that cannot be declared is not defaulted
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("override", "needle"),
    [
        ({"trace_support": False}, "trace_v5"),
        ({"token_emission": TokenEmissionCapabilities()}, "behavior_logprobs"),
        ({"reward_support": False}, "reward_support"),
        ({"state_support": False}, "state_support"),
        ({"terminate_support": False}, "terminate_support"),
    ],
)
def test_a_missing_mandatory_capability_refuses_the_document(
    override: dict[str, Any], needle: str
) -> None:
    """No document at all, rather than a plausible default in its place."""

    surface = minimal_surface(**override)
    with pytest.raises(CispoCapabilityError, match=needle):
        build_cispo_capability_document(minimal_declaration(), surface)


def test_a_declaration_contradicting_the_surface_is_refused() -> None:
    """Artifacts by reference from a runtime with no artifact support."""

    declaration = minimal_declaration(
        evidence=EvidenceDeclaration(
            strict_prefix=True, masking=True, wire_objects=True, artifact_by_reference=True
        )
    )
    with pytest.raises(CispoCapabilityError, match="artifact_support"):
        build_cispo_capability_document(declaration, minimal_surface())


def test_tito_evidence_needs_declared_token_ids() -> None:
    declaration = minimal_declaration(
        evidence=EvidenceDeclaration(
            strict_prefix=True, masking=True, wire_objects=True, tokens_in_tokens_out=True
        )
    )
    with pytest.raises(CispoCapabilityError, match="token_ids"):
        build_cispo_capability_document(declaration, minimal_surface())


def test_a_joint_roster_needs_declared_multi_actor() -> None:
    with pytest.raises(CispoCapabilityError, match="multi_actor"):
        build_cispo_capability_document(rich_declaration(), rich_surface(multi_actor=False))


def test_reward_authority_may_not_be_anything_but_the_container() -> None:
    with pytest.raises(CispoCapabilityError, match="authority"):
        RewardDeclaration(
            authority="executor",
            binds_trace_digest=True,
            quiescence=True,
            horizon_clipping=True,
            channels=("outcome",),
            evaluation_plan_id="plan",
        )


def test_neither_quiescence_nor_clipping_is_refused() -> None:
    """Clipping is the declared substitute for quiescence; declaring neither
    leaves the reward describing whatever was still running afterwards."""

    with pytest.raises(CispoCapabilityError, match="horizon_clipping"):
        RewardDeclaration(
            authority="container",
            binds_trace_digest=True,
            quiescence=False,
            horizon_clipping=False,
            channels=("outcome",),
            evaluation_plan_id="plan",
        )


def test_an_unpinned_opponent_is_refused() -> None:
    with pytest.raises(CispoCapabilityError, match="pin an immutable identity"):
        TopologyDeclaration(
            topology_id="t",
            turn_model="sequential",
            actuation_model="direct_action",
            reward_relation="competitive_rank",
            agent_instances=(
                AgentInstanceDeclaration(
                    agent_instance_id="a",
                    role_id="r",
                    policy_type_id="p",
                    team_id="team-1",
                    trainable=True,
                ),
                AgentInstanceDeclaration(
                    agent_instance_id="b",
                    role_id="r",
                    policy_type_id="p",
                    team_id="team-2",
                    trainable=False,
                ),
            ),
            teams=(
                TeamDeclaration(team_id="team-1", trainable=True, minimum_viable_roster=1),
                TeamDeclaration(team_id="team-2", trainable=False, minimum_viable_roster=1),
            ),
            horizon=HorizonDeclaration(horizon_kind="wall_clock", value=60.0),
        )


def test_a_minimum_roster_larger_than_the_roster_is_refused() -> None:
    with pytest.raises(CispoCapabilityError, match="minimum_viable_roster"):
        TopologyDeclaration(
            topology_id="t",
            turn_model="sequential",
            actuation_model="direct_action",
            reward_relation="cooperative",
            agent_instances=(
                AgentInstanceDeclaration(
                    agent_instance_id="a",
                    role_id="r",
                    policy_type_id="p",
                    team_id="team-1",
                    trainable=True,
                ),
            ),
            teams=(TeamDeclaration(team_id="team-1", trainable=True, minimum_viable_roster=4),),
            horizon=HorizonDeclaration(horizon_kind="wall_clock", value=60.0),
        )


def test_a_renderer_profile_without_stop_tokens_is_refused() -> None:
    with pytest.raises(CispoCapabilityError, match="stop token"):
        RendererProfileDeclaration(
            profile_id="p",
            package="renderers",
            package_version="1",
            config_digest="sha256:c",
            tokenizer_id="t",
            tokenizer_digest="sha256:t",
            stop_token_ids=(),
        )


def test_a_declaration_of_the_wrong_type_is_refused() -> None:
    with pytest.raises(CispoContractError):
        build_cispo_capability_document({"container_id": "x"}, minimal_surface())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# The content hash
# --------------------------------------------------------------------------- #


def _reordered(value: Any) -> Any:
    """The same document with every mapping's keys in reverse order."""

    if isinstance(value, dict):
        return {key: _reordered(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [_reordered(item) for item in value]
    return value


def test_hash_covers_the_document_minus_its_own_hash_field() -> None:
    document = build_cispo_capability_document(rich_declaration(), rich_surface())
    offered = document["capability_hash"]
    assert offered.startswith("sha256:")
    assert canonical_capability_hash(document) == offered
    # Removing the field entirely must not change the answer.
    stripped = {k: v for k, v in document.items() if k != "capability_hash"}
    assert canonical_capability_hash(stripped) == offered


def test_hash_is_stable_across_key_reordering() -> None:
    document = build_cispo_capability_document(rich_declaration(), rich_surface())
    shuffled = _reordered(document)
    assert list(shuffled) != list(document)
    assert canonical_capability_hash(shuffled) == document["capability_hash"]


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda d: d.__setitem__("container_id", "other"), id="container_id"),
        pytest.param(
            lambda d: d["lifecycle"].__setitem__("max_concurrency", 31), id="max_concurrency"
        ),
        pytest.param(
            lambda d: d["lifecycle"]["lease"].__setitem__("ttl_seconds", 901.0), id="lease_ttl"
        ),
        pytest.param(lambda d: d["evidence"].__setitem__("masking", False), id="masking"),
        pytest.param(
            lambda d: d["reward"].__setitem__("channels", ["rank"]), id="reward_channels"
        ),
        pytest.param(
            lambda d: d["topology"]["horizon"].__setitem__("seconds_per_unit", 5.0),
            id="seconds_per_unit",
        ),
        pytest.param(
            lambda d: d["renderer_profile"].__setitem__("config_digest", "sha256:other"),
            id="renderer_config_digest",
        ),
        pytest.param(
            lambda d: d["clock"].__setitem__("skew_tolerance_seconds", 3.0), id="skew_tolerance"
        ),
    ],
)
def test_hash_is_unstable_on_any_value_change(mutate: Any) -> None:
    """Fail-closed means every field, not a chosen subset of them."""

    document = build_cispo_capability_document(rich_declaration(), rich_surface())
    before = document["capability_hash"]
    mutate(document)
    assert canonical_capability_hash(document) != before


def test_two_runtimes_that_differ_only_in_concurrency_hash_differently() -> None:
    first = build_cispo_capability_document(minimal_declaration(), minimal_surface())
    second = build_cispo_capability_document(
        minimal_declaration(
            lifecycle=LifecycleDeclaration(
                advertised_concurrency=2,
                supports_idempotency=True,
                exactly_one_terminal_result=True,
                lease=LeaseDeclaration(ttl_seconds=120.0, renewable=False),
            )
        ),
        minimal_surface(),
    )
    assert first["capability_hash"] != second["capability_hash"]


def test_the_same_declaration_hashes_identically_twice() -> None:
    first = build_cispo_capability_document(rich_declaration(), rich_surface())
    second = build_cispo_capability_document(rich_declaration(), rich_surface())
    assert first == second


# --------------------------------------------------------------------------- #
# Byte-for-byte agreement with the optimizer's canonicalization
# --------------------------------------------------------------------------- #

#: The optimizer's ``canonical_capability_hash``, reimplemented here because
#: ``synth_optimizers`` is not a dependency of this repo and is not importable
#: in this environment. The test below imports the real one when it is present
#: and falls back to this plus a checked-in digest when it is not, so the two
#: implementations are pinned either way.
def _optimizer_side_hash(document: dict[str, Any]) -> str:
    unhashed = {key: value for key, value in document.items() if key != "capability_hash"}
    raw = json.dumps(unhashed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"sha256:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


#: The digest of the fixture document below under that canonicalization,
#: recorded so a change to either side shows up as a failing test rather than
#: as a preflight rejection in a paid run.
EXPECTED_FIXTURE_DIGEST = (
    "sha256:591d0a540cd77138a21df5d8a2f93341fb3e6e606eccaf71f859ca99f12f9123"
)

#: The optimizer's own compliant capability document, verbatim from
#: ``tests/rl/test_handshake.py::capability_payload``. Hashing this exact
#: object is what proves the two canonicalizations agree, independently of the
#: document this container happens to emit.
FIXTURE_DOCUMENT: dict[str, Any] = {
    "schema_version": "cispo.capabilities.v1",
    "container_id": "container-1",
    "container_image_digest": "sha256:image",
    "contract_version": "synth_optimizers.cispo.v1",
    "renderer_profile": {
        "profile_id": "renderers.profile.low.v1",
        "package": "renderers",
        "package_version": "0.4.0",
        "config_digest": "sha256:config",
        "tokenizer_id": "tokenizer-a",
        "tokenizer_digest": "sha256:tokenizer",
        "stop_token_ids": [200002],
        "modalities": ["text"],
        "add_generation_prompt": True,
    },
    "discovery": {
        "taskset_id": "taskset-1",
        "taskset_version": "3",
        "splits": ["train", "eval"],
        "task_content_digests": True,
        "deterministic_lookup": True,
        "duplicate_free": True,
    },
    "policy": {
        "binding_transport": "message_in_capture_out",
        "wire_api": "chat_completions",
        "session_scoped_sampler_origin": True,
        "embeds_credentials": False,
        "revision_immutable_after_admission": True,
        "records_policy_revision": True,
    },
    "lifecycle": {
        "max_concurrency": 30,
        "lease_ttl_seconds": 900.0,
        "supports_idempotency": True,
        "supports_cancellation": True,
        "supports_lease_renewal": True,
        "exactly_one_terminal_result": True,
        "supports_pause_resume": False,
        "straggler_grace_seconds": 120.0,
    },
    "evidence": {
        "trace_v5": True,
        "behavior_logprobs": True,
        "strict_prefix": True,
        "masking": True,
        "wire_objects": True,
        "artifact_reference": True,
        "tokens_in_tokens_out": False,
    },
    "reward": {
        "authority": "container",
        "binds_trace_digest": True,
        "quiescence": True,
        "horizon_clipping": True,
        "channels": ["outcome"],
        "reward_relation": "cooperative",
        "evaluation_plan_id": "plan-1",
        "settlement_window_seconds": 0.0,
        "deferred_scoring": False,
    },
    "recovery": {"restart": True, "stale_discard": True},
    "topology": {
        "topology_id": "topology-1",
        "turn_model": "sequential",
        "actuation_model": "direct_action",
        "reward_relation": "cooperative",
        "agent_instances": [
            {
                "agent_instance_id": "instance-1",
                "role_id": "role-1",
                "policy_type_id": "policy-1",
                "team_id": "team-1",
                "trainable": True,
            }
        ],
        "teams": [{"team_id": "team-1", "trainable": True, "minimum_viable_roster": 1}],
        "communication_channels": [
            {"channel_id": "channel-1", "scope": "intra_team", "trainable_for_author": True}
        ],
        "horizon": {
            "horizon_kind": "wall_clock",
            "value_seconds": 5400.0,
            "time_dilation": 1.0,
        },
        "parameter_groups": {"policy-1": "group-1"},
    },
    "clock": {"skew_tolerance_seconds": 2.0},
}


def test_hash_matches_the_optimizer_side_canonicalization() -> None:
    """``synth_optimizers`` is not importable here, so the canonicalization is
    reimplemented above and pinned to a checked-in digest. When the package is
    importable the real function is used instead and the digest is a bonus."""

    try:  # pragma: no cover - depends on the environment
        from synth_optimizers.rl.capabilities import (  # type: ignore[import-not-found]
            canonical_capability_hash as optimizer_hash,
        )

        source = "synth_optimizers"
    except ImportError:
        optimizer_hash = _optimizer_side_hash
        source = "reimplemented"

    assert optimizer_hash(FIXTURE_DOCUMENT) == EXPECTED_FIXTURE_DIGEST, source
    assert canonical_capability_hash(FIXTURE_DOCUMENT) == optimizer_hash(FIXTURE_DOCUMENT)

    # And on a document this container actually emits, not only the fixture.
    emitted = build_cispo_capability_document(rich_declaration(), rich_surface())
    assert emitted["capability_hash"] == optimizer_hash(emitted)


def test_the_emitted_document_carries_every_field_the_optimizer_parses() -> None:
    """The optimizer's ``CapabilityDocument.from_payload`` reads exactly these."""

    document = build_cispo_capability_document(rich_declaration(), rich_surface())
    for key in FIXTURE_DOCUMENT:
        assert key in document, f"the optimizer parses {key!r} and it is absent"
    for section, keys in (
        ("discovery", FIXTURE_DOCUMENT["discovery"]),
        ("policy", FIXTURE_DOCUMENT["policy"]),
        ("lifecycle", FIXTURE_DOCUMENT["lifecycle"]),
        ("evidence", FIXTURE_DOCUMENT["evidence"]),
        ("reward", FIXTURE_DOCUMENT["reward"]),
        ("recovery", FIXTURE_DOCUMENT["recovery"]),
    ):
        missing = set(keys) - set(document[section])
        assert not missing, f"{section} is missing {sorted(missing)}"
    # ``_parse_topology`` reads the horizon value under ``value_seconds``.
    assert "value_seconds" in document["topology"]["horizon"]
    assert document["topology"]["horizon"]["value_seconds"] == 500.0
