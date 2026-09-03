"""The container half of the readiness agreement, and the gate it installs.

Four things have to hold, and each of them is a way a run silently wastes
provider spend or trains on invalid evidence when it does not.

Every clause in the canonical registry gets a verdict with a reason, derived
from what this build declares. A clause answered from a table of what "usually"
works is worse than no answer, because it reads as agreement.

A conditional clause is answered only where it applies. A ``steps`` horizon
reads no wall clock, so ``lifecycle.clock_skew`` is not applicable there, which
is a different statement from a container declining it.

The agreement digest is computed the way the executor computes it, byte for
byte. A digest the two sides compute differently is worse than no digest: it
fails a healthy run and passes nothing.

And the agreement gates every attempt. Absent, expired, revoked, or bound to a
different digest are four separate refusals, and a run that can slip past any
of them can drift out from under its own agreement mid-flight.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from synth_containers.capabilities import (
    RuntimeCapabilitySurface,
    TokenEmissionCapabilities,
)
from synth_containers.cispo_handshake import (
    ALL_CLAUSES,
    CISPO_CONTRACT_VERSION,
    CLAUSE_GROUPS,
    CLAUSE_SUBSTITUTES,
    CONDITIONAL_CLAUSES,
    HANDSHAKE_SCHEMA_VERSION,
    MANDATORY_CLAUSES,
    MANDATORY_ROUTE_NAMES,
    OPTIONAL_CLAUSES,
    UNCONDITIONAL_MANDATORY_CLAUSES,
    AgentInstanceFacts,
    AgreementMismatch,
    CapabilityDrift,
    ChannelFacts,
    CispoHandshakeAdapter,
    DiscoveryFacts,
    EvidenceFacts,
    HandshakeAbsent,
    HandshakeExpired,
    HandshakeRegistry,
    HandshakeRequest,
    HandshakeRevoked,
    HandshakeUnknown,
    HorizonFacts,
    LifecycleFacts,
    PolicyFacts,
    RecoveryFacts,
    RendererProfileFacts,
    RewardFacts,
    RuntimeFacts,
    TaskRow,
    TeamFacts,
    TopologyFacts,
    UndeclaredSubstitute,
    compute_agreement_digest,
    format_rfc3339,
    task_content_digest,
)

# --------------------------------------------------------------------------- #
# The optimizer's own modules, when this checkout can reach them
# --------------------------------------------------------------------------- #


def _optimizer_src() -> Path | None:
    """Locate a sibling ``optimizers`` checkout, or give up quietly.

    The container package does not depend on the optimizer, so every
    cross-check below is conditional. Where the source is reachable the test
    pins against the real thing; where it is not, the test says so in its
    assertion message rather than passing on a weaker check silently.
    """

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "optimizers" / "src" / "synth_optimizers"
        if candidate.is_dir():
            return candidate.parent
    return None


def optimizer(module: str) -> Any | None:
    """Import one optimizer module, or ``None`` when it is not reachable."""

    try:
        return importlib.import_module(module)
    except ImportError:
        pass
    source = _optimizer_src()
    if source is None:
        return None
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    try:
        return importlib.import_module(module)
    except ImportError:  # pragma: no cover - a broken sibling checkout
        return None


OPTIMIZER_CLAUSES = optimizer("synth_optimizers.contracts.rl_clauses")
OPTIMIZER_HANDSHAKE = optimizer("synth_optimizers.rl.handshake")
OPTIMIZER_RECORDS = optimizer("synth_optimizers.contracts.rl_records")
OPTIMIZER_CAPABILITIES = optimizer("synth_optimizers.rl.capabilities")


# --------------------------------------------------------------------------- #
# A build that declares everything, and the knobs to take pieces away
# --------------------------------------------------------------------------- #

NOW = datetime(2026, 9, 3, 12, 0, 0, tzinfo=timezone.utc)

ROUTES: dict[str, str] = {
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

PROFILE = RendererProfileFacts(
    profile_id="renderers.pinned.low.v1",
    package="renderers",
    package_version="0.1.11",
    config_digest="sha256:cfg",
    tokenizer_id="vendor/policy-20b",
    tokenizer_digest="sha256:tok",
    stop_token_ids=(200002, 199999),
)

TASK_IDS = ("task-0", "task-1", "task-2")
ROWS = tuple(
    TaskRow(
        task_id=task_id,
        content_digest=task_content_digest({"task_id": task_id, "taskset": "ts-1"}),
        topology_ref="topo-1",
        task_family="family-a",
    )
    for task_id in TASK_IDS
)


def facts(**overrides: Any) -> RuntimeFacts:
    """A build that can honor everything, so a test takes exactly one thing away."""

    payload: dict[str, Any] = {
        "container_id": "container-1",
        "container_image_digest": "sha256:image",
        "renderer_profile": PROFILE,
        "discovery": DiscoveryFacts(
            taskset_id="ts-1",
            taskset_version="v1",
            splits=("train", "val"),
            rows=ROWS,
            task_content_digests=True,
            deterministic_lookup=True,
            duplicate_free=True,
        ),
        "policy": PolicyFacts(
            binding_transport="message_in_capture_out",
            wire_api="chat_completions",
            session_scoped_sampler_origin=True,
            embeds_credentials=False,
            revision_immutable_after_admission=True,
            records_policy_revision=True,
        ),
        "lifecycle": LifecycleFacts(
            max_concurrency=30,
            lease_ttl_seconds=900.0,
            lease_renewable=True,
            supports_idempotency=True,
            supports_cancellation=True,
            exactly_one_terminal_result=True,
            supports_pause_resume=True,
            clock_skew_tolerance_seconds=2.0,
        ),
        "evidence": EvidenceFacts(
            trace_v5=True,
            behavior_logprobs=True,
            strict_prefix=True,
            masking=True,
            wire_objects=True,
            artifact_reference=True,
            inline_evidence=True,
            tokens_in_tokens_out=True,
        ),
        "reward": RewardFacts(
            authority="container",
            evaluation_plan_id="plan-1",
            reward_relation="competitive_rank",
            channels=("outcome", "margin"),
            binds_trace_digest=True,
            quiescence=True,
            horizon_clipping=True,
            settlement_window_seconds=150.0,
            deferred_scoring=True,
        ),
        "recovery": RecoveryFacts(restart=True, stale_discard=True),
        "topology": TopologyFacts(
            topology_id="topo-1",
            turn_model="sequential",
            actuation_model="direct_action",
            agent_instances=(
                AgentInstanceFacts("a1", "runner", "pt-trainable", "terra", True),
                AgentInstanceFacts(
                    "a2", "runner", "pt-frozen", "rock", False, "sha256:frozen-ckpt"
                ),
            ),
            teams=(TeamFacts("terra", True, 1), TeamFacts("rock", False, 1)),
            communication_channels=(ChannelFacts("c1", "intra_team"),),
            parameter_groups={"pt-trainable": "pg-1"},
            partial_roster_disposition="drop_instance",
            pins_opponent_identity=True,
        ),
        "horizon": HorizonFacts(
            horizon_kind="wall_clock", value=5400.0, time_dilation=4.0, grace_seconds=30.0
        ),
        "declared_routes": ROUTES,
        "probe_binding": True,
        "handshake_ttl_seconds": 900.0,
    }
    payload.update(overrides)
    return RuntimeFacts(**payload)


def request_payload(**overrides: Any) -> dict[str, Any]:
    """The requirement document the executor sends, in full, before any spend."""

    payload: dict[str, Any] = {
        "schema_version": HANDSHAKE_SCHEMA_VERSION,
        "run_id": "run-1",
        "attempt": 1,
        "optimizer": {"name": "synth_optimizers.cispo", "version": "0.2.20"},
        "policy": {
            "provider": "tinker",
            "model_id": "vendor/policy-20b",
            "transport": "message_in_capture_out",
        },
        "renderer_profile": {
            "profile_id": PROFILE.profile_id,
            "config_digest": PROFILE.config_digest,
            "fingerprint": PROFILE.fingerprint,
        },
        "requirements": [
            *UNCONDITIONAL_MANDATORY_CLAUSES,
            "lifecycle.clock_skew",
        ],
        "accept_degraded": {},
        "topology": {
            "expected_topology_id": "topo-1",
            "trainable_teams": ["terra"],
            "partial_roster": "drop_instance",
        },
        "run_plan": {
            "group_size": 8,
            "groups_per_step": 1,
            "max_execution_slots": 8,
            "maximum_policy_lag": 1,
            "target_train_updates": 10,
            "expected_horizon_seconds": 5400.0,
        },
        "taskset": {"taskset_id": "ts-1", "split": "train", "task_ids": ["task-0", "task-1"]},
        "clock": {
            "executor_time": format_rfc3339(NOW),
            "monotonic_source": "CLOCK_MONOTONIC",
        },
    }
    payload.update(overrides)
    return payload


def registry(runtime: RuntimeFacts | None = None, *, now: datetime = NOW) -> HandshakeRegistry:
    clock: list[datetime] = [now]
    return HandshakeRegistry(runtime or facts(), clock=lambda: clock[0])


def _clock_registry(
    runtime: RuntimeFacts | None = None,
) -> tuple[HandshakeRegistry, list[datetime]]:
    """A registry whose clock a test can move forward."""

    moment = [NOW]
    return HandshakeRegistry(runtime or facts(), clock=lambda: moment[0]), moment


# --------------------------------------------------------------------------- #
# The clause registry mirror
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(OPTIMIZER_CLAUSES is None, reason="optimizer source not reachable")
def test_clause_registry_mirror_matches_the_authoritative_module() -> None:
    """The ids are the optimizer's. This module only mirrors them."""

    assert CLAUSE_GROUPS == OPTIMIZER_CLAUSES.CLAUSE_GROUPS
    assert ALL_CLAUSES == OPTIMIZER_CLAUSES.ALL_CLAUSES
    assert MANDATORY_CLAUSES == OPTIMIZER_CLAUSES.MANDATORY_CLAUSES
    assert OPTIONAL_CLAUSES == OPTIMIZER_CLAUSES.OPTIONAL_CLAUSES
    assert CONDITIONAL_CLAUSES == OPTIMIZER_CLAUSES.CONDITIONAL_CLAUSES
    assert CLAUSE_SUBSTITUTES == OPTIMIZER_CLAUSES.CLAUSE_SUBSTITUTES
    assert (
        UNCONDITIONAL_MANDATORY_CLAUSES
        == OPTIMIZER_CLAUSES.UNCONDITIONAL_MANDATORY_CLAUSES
    )
    assert HANDSHAKE_SCHEMA_VERSION == OPTIMIZER_CLAUSES.HANDSHAKE_SCHEMA_VERSION


# --------------------------------------------------------------------------- #
# Every clause answered, with a reason
# --------------------------------------------------------------------------- #


def test_every_mandatory_clause_is_answered_with_a_reason() -> None:
    verdict = registry().handshake(request_payload())
    answered = {answer.clause_id for answer in verdict.clauses}
    assert set(MANDATORY_CLAUSES) <= answered, sorted(set(MANDATORY_CLAUSES) - answered)
    for answer in verdict.clauses:
        assert answer.reason.strip(), f"{answer.clause_id} answered without a reason"


def test_a_capable_build_answers_every_clause_accepted() -> None:
    """Nothing is degraded or rejected when the build can honor all of it."""

    verdict = registry().handshake(request_payload())
    unhappy = {
        answer.clause_id: (answer.verdict, answer.reason)
        for answer in verdict.clauses
        if answer.verdict != "accepted"
    }
    assert unhappy == {}, unhappy
    assert verdict.accepted is True
    assert verdict.rejected_mandatory_clauses == ()


def test_clause_answers_move_with_the_declared_facts() -> None:
    """One flipped declaration, one changed verdict. No table in between."""

    cases = {
        "evidence.masking": facts(
            evidence=replace(facts().evidence, masking=False)
        ),
        "policy.no_embedded_credentials": facts(
            policy=replace(facts().policy, embeds_credentials=True)
        ),
        "lifecycle.idempotency": facts(
            lifecycle=replace(facts().lifecycle, supports_idempotency=False)
        ),
        "recovery.stale_discard": facts(
            recovery=replace(facts().recovery, stale_discard=False)
        ),
        "reward.channels": facts(reward=replace(facts().reward, channels=())),
    }
    for clause_id, runtime in cases.items():
        verdict = registry(runtime).handshake(request_payload())
        answer = verdict.clause(clause_id)
        assert answer is not None
        assert answer.verdict == "rejected", (clause_id, answer)
        assert verdict.accepted is False


def test_an_undeclared_route_rejects_contract_routes_and_names_it() -> None:
    """Route presence and behavior support are separate questions."""

    thinned = {name: path for name, path in ROUTES.items() if name != "artifacts_route"}
    verdict = registry(facts(declared_routes=thinned)).handshake(request_payload())
    answer = verdict.clause("contract.routes")
    assert answer is not None and answer.verdict == "rejected"
    assert "artifacts_route" in answer.reason


def test_no_clause_answer_reads_a_task_harness_or_environment_name() -> None:
    """The clause list is generic; renaming the taskset must not move a verdict."""

    renamed_rows = tuple(
        replace(row, task_id=row.task_id.replace("task", "scenario"))
        for row in ROWS
    )
    renamed = facts(
        discovery=replace(
            facts().discovery,
            taskset_id="totally-different-taskset",
            rows=renamed_rows,
        )
    )
    request = request_payload(
        taskset={
            "taskset_id": "totally-different-taskset",
            "split": "train",
            "task_ids": ["scenario-0", "scenario-1"],
        }
    )
    baseline = registry().handshake(request_payload())
    renamed_verdict = registry(renamed).handshake(request)
    assert {answer.clause_id: answer.verdict for answer in renamed_verdict.clauses} == {
        answer.clause_id: answer.verdict for answer in baseline.clauses
    }


# --------------------------------------------------------------------------- #
# Conditional clauses
# --------------------------------------------------------------------------- #


def test_clock_skew_is_answered_for_a_wall_clock_horizon() -> None:
    verdict = registry().handshake(request_payload())
    answer = verdict.clause("lifecycle.clock_skew")
    assert answer is not None and answer.verdict == "accepted"
    assert verdict.not_applicable_clauses == ()


def test_clock_skew_is_skipped_where_it_does_not_apply() -> None:
    """A steps horizon reads no wall clock, so the clause is not applicable.

    Not applicable is not the same as declined: an optional clause may be
    declined, a conditional one may not be declined where it applies. Here it
    does not apply, so no verdict is issued at all.
    """

    stepped = facts(
        horizon=HorizonFacts(horizon_kind="steps", value=500.0, seconds_per_unit=2.0)
    )
    verdict = registry(stepped).handshake(
        request_payload(
            requirements=list(UNCONDITIONAL_MANDATORY_CLAUSES),
            # 500 steps at the declared 2s per step; a lease may not be guessed
            # from a unit that carries no duration.
            run_plan={**request_payload()["run_plan"], "expected_horizon_seconds": 1000.0},
        )
    )
    assert verdict.clause("lifecycle.clock_skew") is None
    assert verdict.not_applicable_clauses == ("lifecycle.clock_skew",)
    assert verdict.accepted is True


def test_skew_beyond_the_declared_tolerance_is_a_rejected_clause() -> None:
    """The horizon is the instant the reward is read, so skew is not a warning."""

    late = NOW + timedelta(seconds=30)
    verdict = registry(now=late).handshake(request_payload())
    answer = verdict.clause("lifecycle.clock_skew")
    assert answer is not None and answer.verdict == "rejected"
    assert "30.0" in answer.reason and "2.0" in answer.reason
    assert verdict.accepted is False
    assert verdict.measured_skew_seconds == pytest.approx(30.0)


# --------------------------------------------------------------------------- #
# Substitutes and degradation
# --------------------------------------------------------------------------- #


def test_a_build_that_cannot_quiesce_offers_its_declared_substitute() -> None:
    """Neither a rejection nor a plain degradation: the same question, answered."""

    clipping = facts(
        reward=replace(facts().reward, quiescence=False, horizon_clipping=True)
    )
    verdict = registry(clipping).handshake(request_payload())
    answer = verdict.clause("reward.horizon_quiescence")
    assert answer is not None
    assert answer.verdict == "degraded"
    assert answer.substitute == "horizon_clipped_snapshot"
    assert answer.substitute in CLAUSE_SUBSTITUTES["reward.horizon_quiescence"]
    # Unacknowledged, the run may not start: the executor has to say which
    # answer it will run under.
    assert verdict.accepted is False
    assert verdict.unaccepted_degraded_clauses == ("reward.horizon_quiescence",)
    assert verdict.obligations.quiescence is False


def test_an_acknowledged_substitute_lets_the_handshake_accept() -> None:
    clipping = facts(
        reward=replace(facts().reward, quiescence=False, horizon_clipping=True)
    )
    verdict = registry(clipping).handshake(
        request_payload(
            accept_degraded={"reward.horizon_quiescence": "horizon_clipped_snapshot"}
        )
    )
    assert verdict.accepted is True
    assert verdict.acknowledged_substitutes == {
        "reward.horizon_quiescence": "horizon_clipped_snapshot"
    }
    assert verdict.clause("reward.horizon_quiescence").verdict == "degraded"


def test_a_build_that_can_neither_quiesce_nor_clip_rejects_outright() -> None:
    neither = facts(
        reward=replace(facts().reward, quiescence=False, horizon_clipping=False)
    )
    verdict = registry(neither).handshake(request_payload())
    answer = verdict.clause("reward.horizon_quiescence")
    assert answer is not None and answer.verdict == "rejected"
    assert "reward.horizon_quiescence" in verdict.rejected_mandatory_clauses


def test_a_substitute_nobody_declared_is_refused() -> None:
    with pytest.raises(UndeclaredSubstitute):
        registry().handshake(
            request_payload(
                accept_degraded={"reward.horizon_quiescence": "just_score_it_late"}
            )
        )
    with pytest.raises(UndeclaredSubstitute):
        registry().handshake(
            request_payload(accept_degraded={"lifecycle.concurrency": "fewer_slots"})
        )


def test_concurrency_beyond_the_pool_degrades_rather_than_rejects() -> None:
    """A degraded clause the executor can answer by lowering its own run plan."""

    small = facts(lifecycle=replace(facts().lifecycle, max_concurrency=4))
    verdict = registry(small).handshake(request_payload())
    answer = verdict.clause("lifecycle.concurrency")
    assert answer is not None and answer.verdict == "degraded"
    assert "4 leases available, 8 requested" in answer.reason
    assert verdict.accepted is False
    assert verdict.obligations.max_concurrency == 4


def test_an_unrenewable_lease_shorter_than_the_horizon_is_rejected() -> None:
    """An hour-scale episode may not depend on an HTTP request staying open."""

    short = facts(
        lifecycle=replace(facts().lifecycle, lease_renewable=False, lease_ttl_seconds=60.0)
    )
    verdict = registry(short).handshake(request_payload())
    answer = verdict.clause("lifecycle.lease_renewal")
    assert answer is not None and answer.verdict == "rejected"
    assert verdict.accepted is False


def test_an_optional_clause_the_build_lacks_comes_back_unsupported() -> None:
    without = facts(
        evidence=replace(facts().evidence, tokens_in_tokens_out=False),
        reward=replace(facts().reward, settlement_window_seconds=0.0),
    )
    verdict = registry(without).handshake(request_payload())
    assert verdict.clause("evidence.tito").verdict == "unsupported"
    assert verdict.clause("reward.settlement_window").verdict == "unsupported"
    # Optional clauses never block the run; the executor records its fallback.
    assert verdict.accepted is True
    assert verdict.obligations.settlement_window_seconds == 0.0


# --------------------------------------------------------------------------- #
# The agreement digest
# --------------------------------------------------------------------------- #


def _replica_agreement_digest(
    *,
    request: HandshakeRequest,
    handshake_id: str,
    capability_hash: str,
    renderer_fingerprint: str,
    taskset_resolution: Any,
    obligations: Any,
    clauses: Any,
) -> str:
    """A transcription of the optimizer's ``compute_agreement_digest``.

    Used only when the optimizer source is not reachable from this checkout.
    Transcribed from ``synth_optimizers/rl/handshake.py``; it can catch drift
    in this module but not drift on the other side, which is why the reachable
    path above is the one that matters.
    """

    import hashlib
    import json

    payload = {
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
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


def test_agreement_digest_matches_the_optimizers_computation() -> None:
    """Both sides must compute the same digest over the same two documents."""

    runtime = facts()
    payload = request_payload()
    verdict = registry(runtime).handshake(payload)
    parsed = HandshakeRequest.from_payload(payload)

    # The module's own function, over the same documents, is the digest the
    # verdict carried: the registry does not compute it a second, private way.
    assert verdict.agreement_digest == compute_agreement_digest(
        request=parsed,
        handshake_id=verdict.handshake_id,
        capability_hash=runtime.capability_hash,
        renderer_fingerprint=runtime.renderer_profile.fingerprint,
        taskset_resolution=verdict.taskset_resolution,
        obligations=verdict.obligations,
        clauses=verdict.clauses,
    )

    if OPTIMIZER_HANDSHAKE is None or OPTIMIZER_RECORDS is None:
        expected = _replica_agreement_digest(
            request=parsed,
            handshake_id=verdict.handshake_id,
            capability_hash=runtime.capability_hash,
            renderer_fingerprint=runtime.renderer_profile.fingerprint,
            taskset_resolution=verdict.taskset_resolution,
            obligations=verdict.obligations,
            clauses=verdict.clauses,
        )
        assert verdict.agreement_digest == expected, (
            "digest disagrees with the transcribed replica of the optimizer's "
            "computation (the optimizer source was not reachable from this checkout)"
        )
        return

    opt_request = _optimizer_request(payload)
    opt_verdict = OPTIMIZER_HANDSHAKE.HandshakeVerdict.from_payload(verdict.to_payload())
    expected = OPTIMIZER_HANDSHAKE.compute_agreement_digest(
        opt_request,
        handshake_id=verdict.handshake_id,
        capability_hash=runtime.capability_hash,
        renderer_fingerprint=_optimizer_profile().fingerprint,
        taskset_resolution=opt_verdict.taskset_resolution,
        obligations=opt_verdict.obligations,
        clauses=opt_verdict.clauses,
    )
    assert verdict.agreement_digest == expected


def _optimizer_profile() -> Any:
    records = OPTIMIZER_RECORDS
    payload = PROFILE.to_payload()
    return records.RendererProfile(
        profile_id=payload["profile_id"],
        package=payload["package"],
        package_version=payload["package_version"],
        config_digest=payload["config_digest"],
        tokenizer_id=payload["tokenizer_id"],
        tokenizer_digest=payload["tokenizer_digest"],
        stop_token_ids=tuple(payload["stop_token_ids"]),
        modalities=tuple(payload["modalities"]),
        add_generation_prompt=payload["add_generation_prompt"],
    )


def _optimizer_request(payload: dict[str, Any]) -> Any:
    handshake = OPTIMIZER_HANDSHAKE
    plan = payload["run_plan"]
    taskset = payload["taskset"]
    return handshake.HandshakeRequest(
        run_id=payload["run_id"],
        optimizer=handshake.OptimizerIdentity(**payload["optimizer"]),
        policy=handshake.PolicyRequest(**payload["policy"]),
        renderer_profile=_optimizer_profile(),
        requirements=tuple(payload["requirements"]),
        topology=handshake.TopologyExpectation(
            expected_topology_id=payload["topology"]["expected_topology_id"],
            trainable_teams=tuple(payload["topology"]["trainable_teams"]),
            partial_roster=payload["topology"]["partial_roster"],
        ),
        run_plan=handshake.RunPlan(**plan),
        taskset=handshake.TasksetRequest(
            taskset_id=taskset["taskset_id"],
            split=taskset["split"],
            task_ids=tuple(taskset["task_ids"]),
        ),
        clock=handshake.ClockStamp(**payload["clock"]),
        attempt=payload["attempt"],
        accept_degraded=tuple(payload["accept_degraded"].items()),
    )


@pytest.mark.skipif(
    OPTIMIZER_HANDSHAKE is None, reason="optimizer source not reachable"
)
def test_the_optimizer_reads_this_verdict_as_admissible() -> None:
    """The whole client path, not just the digest: parse, evaluate, admit."""

    runtime = facts()
    payload = request_payload()
    verdict = registry(runtime).handshake(payload)
    capabilities = OPTIMIZER_CAPABILITIES
    contract_module = optimizer("synth_optimizers.rl.contract")
    assert contract_module is not None

    document = capabilities.CapabilityDocument.from_payload(runtime.capability_document())
    assert document.content_hash == runtime.capability_hash

    decision = OPTIMIZER_HANDSHAKE.evaluate_handshake(
        _optimizer_request(payload),
        OPTIMIZER_HANDSHAKE.HandshakeVerdict.from_payload(verdict.to_payload()),
        capability=document,
        contract=contract_module.ContainerContract(
            version=CISPO_CONTRACT_VERSION,
            route_table=contract_module.RouteTable(routes=ROUTES),
        ),
        now=NOW,
    )
    assert decision.outcome == "admissible"
    assert decision.agreement is not None
    assert decision.agreement.agreement_digest == verdict.agreement_digest


def test_the_digest_changes_when_any_bound_document_changes() -> None:
    """It binds both documents, the capability hash, the renderer, and the tasks."""

    base = registry().handshake(request_payload())
    variants = {
        "run plan": request_payload(
            run_plan={
                **request_payload()["run_plan"],
                "target_train_updates": 11,
            }
        ),
        "taskset": request_payload(
            taskset={"taskset_id": "ts-1", "split": "train", "task_ids": ["task-0"]}
        ),
    }
    for name, payload in variants.items():
        other = registry().handshake(payload)
        assert other.agreement_digest != base.agreement_digest, name

    # A different capability document is a different agreement even for the
    # identical requirement document.
    drifted = registry(
        facts(lifecycle=replace(facts().lifecycle, max_concurrency=29))
    ).handshake(request_payload())
    assert drifted.agreement_digest != base.agreement_digest


def test_the_taskset_resolution_carries_a_digest_and_topology_ref_per_task() -> None:
    verdict = registry().handshake(request_payload())
    resolved = {row.task_id: row for row in verdict.taskset_resolution}
    assert set(resolved) == {"task-0", "task-1"}
    for row in resolved.values():
        assert row.content_digest.startswith("sha256:")
        assert row.topology_ref == "topo-1"


def test_an_unresolvable_task_id_rejects_the_digest_clause() -> None:
    verdict = registry().handshake(
        request_payload(
            taskset={"taskset_id": "ts-1", "split": "train", "task_ids": ["task-0", "ghost"]}
        )
    )
    answer = verdict.clause("discovery.task_digests")
    assert answer is not None and answer.verdict == "rejected"
    assert "ghost" in answer.reason


# --------------------------------------------------------------------------- #
# Admission: absent, expired, revoked, mismatched
# --------------------------------------------------------------------------- #


def test_an_admitted_agreement_gates_an_attempt() -> None:
    ledger = registry()
    verdict = ledger.handshake(request_payload())
    agreement = ledger.assert_admissible(verdict.handshake_id, verdict.agreement_digest)
    assert agreement.run_id == "run-1"
    assert agreement.task_digest("task-0").startswith("sha256:")


def test_an_attempt_with_no_handshake_is_refused() -> None:
    ledger = registry()
    ledger.handshake(request_payload())
    with pytest.raises(HandshakeAbsent):
        ledger.assert_admissible(None, "sha256:whatever")
    with pytest.raises(HandshakeAbsent):
        ledger.assert_admissible("  ", "sha256:whatever")


def test_an_attempt_naming_an_unknown_handshake_is_refused() -> None:
    ledger = registry()
    verdict = ledger.handshake(request_payload())
    with pytest.raises(HandshakeUnknown):
        ledger.assert_admissible("hs_never_issued", verdict.agreement_digest)


def test_an_attempt_on_an_expired_handshake_is_refused() -> None:
    ledger, moment = _clock_registry()
    verdict = ledger.handshake(request_payload())
    moment[0] = NOW + timedelta(seconds=901)
    with pytest.raises(HandshakeExpired):
        ledger.assert_admissible(verdict.handshake_id, verdict.agreement_digest)


def test_an_attempt_on_a_revoked_handshake_is_refused() -> None:
    ledger = registry()
    verdict = ledger.handshake(request_payload())
    ledger.revoke(verdict.handshake_id, "container degraded")
    with pytest.raises(HandshakeRevoked) as excinfo:
        ledger.assert_admissible(verdict.handshake_id, verdict.agreement_digest)
    assert "container degraded" in str(excinfo.value)


def test_an_attempt_naming_a_different_agreement_is_refused() -> None:
    ledger = registry()
    verdict = ledger.handshake(request_payload())
    with pytest.raises(AgreementMismatch):
        ledger.assert_admissible(verdict.handshake_id, "sha256:some-other-agreement")
    with pytest.raises(HandshakeAbsent):
        ledger.assert_admissible(verdict.handshake_id, "")


def test_a_rejected_handshake_is_never_admitted() -> None:
    """No session, no binding, no attempt against an agreement that never was."""

    broken = facts(evidence=replace(facts().evidence, strict_prefix=False))
    ledger = registry(broken)
    verdict = ledger.handshake(request_payload())
    assert verdict.accepted is False
    with pytest.raises(HandshakeUnknown):
        ledger.assert_admissible(verdict.handshake_id, verdict.agreement_digest)


# --------------------------------------------------------------------------- #
# Renewal and revocation
# --------------------------------------------------------------------------- #


def test_renewal_extends_the_same_agreement() -> None:
    ledger, moment = _clock_registry()
    verdict = ledger.handshake(request_payload())
    moment[0] = NOW + timedelta(seconds=600)
    renewed = ledger.renew(verdict.handshake_id)
    assert renewed.handshake_id == verdict.handshake_id
    assert renewed.agreement_digest == verdict.agreement_digest
    assert renewed.expires_at == NOW + timedelta(seconds=1500)
    moment[0] = NOW + timedelta(seconds=1000)
    ledger.assert_admissible(verdict.handshake_id, verdict.agreement_digest)


def test_renewal_fails_closed_when_the_capability_document_changed() -> None:
    """Renewal re-reads the document, exactly as the original preflight does."""

    ledger, moment = _clock_registry()
    verdict = ledger.handshake(request_payload())
    moment[0] = NOW + timedelta(seconds=60)
    ledger.degrade(
        facts(lifecycle=replace(facts().lifecycle, max_concurrency=2)),
        reason="pool shrank under a live run",
    )
    with pytest.raises((CapabilityDrift, HandshakeRevoked)):
        ledger.renew(verdict.handshake_id)
    with pytest.raises(HandshakeRevoked):
        ledger.assert_admissible(verdict.handshake_id, verdict.agreement_digest)


def test_degrading_revokes_only_the_agreements_built_on_the_old_document() -> None:
    ledger = registry()
    verdict = ledger.handshake(request_payload())
    revoked = ledger.degrade(
        facts(reward=replace(facts().reward, settlement_window_seconds=10.0)),
        reason="settlement window shortened",
    )
    assert revoked == (verdict.handshake_id,)
    assert ledger.agreements() == {}


def test_renewal_of_an_expired_agreement_is_refused_and_revokes_it() -> None:
    ledger, moment = _clock_registry()
    verdict = ledger.handshake(request_payload())
    moment[0] = NOW + timedelta(seconds=1000)
    with pytest.raises(HandshakeExpired):
        ledger.renew(verdict.handshake_id)
    with pytest.raises(HandshakeRevoked):
        ledger.renew(verdict.handshake_id)


def test_renewal_of_an_unknown_or_revoked_handshake_is_refused() -> None:
    ledger = registry()
    with pytest.raises(HandshakeUnknown):
        ledger.renew("hs_never_issued")
    verdict = ledger.handshake(request_payload())
    ledger.revoke(verdict.handshake_id, "drained")
    with pytest.raises(HandshakeRevoked):
        ledger.renew(verdict.handshake_id)


# --------------------------------------------------------------------------- #
# The requirement document itself
# --------------------------------------------------------------------------- #


def test_a_mis_versioned_requirement_document_is_refused() -> None:
    from synth_containers.cispo_handshake import MalformedRequirementDocument

    with pytest.raises(MalformedRequirementDocument):
        registry().handshake(request_payload(schema_version="cispo.handshake.v0"))


def test_an_unknown_clause_id_in_the_requirement_document_is_refused() -> None:
    from synth_containers.cispo_handshake import MalformedRequirementDocument

    with pytest.raises(MalformedRequirementDocument):
        registry().handshake(
            request_payload(requirements=[*UNCONDITIONAL_MANDATORY_CLAUSES, "evidence.vibes"])
        )


def test_a_renderer_profile_mismatch_rejects_before_any_spend() -> None:
    verdict = registry().handshake(
        request_payload(
            renderer_profile={
                "profile_id": PROFILE.profile_id,
                "config_digest": "sha256:other",
                "fingerprint": "0" * 32,
            }
        )
    )
    answer = verdict.clause("policy.renderer_profile_match")
    assert answer is not None and answer.verdict == "rejected"
    assert verdict.accepted is False


# --------------------------------------------------------------------------- #
# Derivation from the runtime surface, and from an assembled document
# --------------------------------------------------------------------------- #


def _surface(**overrides: Any) -> RuntimeCapabilitySurface:
    payload: dict[str, Any] = {
        "trace_support": True,
        "reward_support": True,
        "state_support": True,
        "terminate_support": True,
        "pause_support": True,
        "resume_support": True,
        "artifact_support": True,
        "proxied_inference": True,
        "multi_actor": True,
        "token_emission": TokenEmissionCapabilities(
            token_ids=True, tokens=True, logprobs=True
        ),
    }
    payload.update(overrides)
    return RuntimeCapabilitySurface(**payload)


def _from_surface(surface: RuntimeCapabilitySurface, **overrides: Any) -> RuntimeFacts:
    base = facts()
    payload: dict[str, Any] = {
        "container_id": base.container_id,
        "container_image_digest": base.container_image_digest,
        "renderer_profile": base.renderer_profile,
        "discovery": base.discovery,
        "policy": base.policy,
        "lifecycle": base.lifecycle,
        "evidence": base.evidence,
        "reward": base.reward,
        "recovery": base.recovery,
        "topology": base.topology,
        "horizon": base.horizon,
        "declared_routes": ROUTES,
        "probe_binding": True,
    }
    payload.update(overrides)
    return RuntimeFacts.from_runtime_surface(surface, **payload)


def test_the_surface_can_only_remove_a_promise_never_add_one() -> None:
    """A declaration the surface contradicts is dropped, not honored."""

    runtime = _from_surface(_surface(trace_support=False, artifact_support=False))
    assert runtime.evidence.trace_v5 is False
    assert runtime.evidence.artifact_reference is False
    verdict = registry(runtime).handshake(request_payload())
    assert verdict.clause("evidence.trace_v5").verdict == "rejected"
    assert verdict.accepted is False


def test_a_surface_with_no_logprobs_cannot_promise_behavior_logprobs() -> None:
    runtime = _from_surface(_surface(token_emission=TokenEmissionCapabilities()))
    assert runtime.evidence.behavior_logprobs is False
    verdict = registry(runtime).handshake(request_payload())
    assert verdict.clause("evidence.behavior_logprobs").verdict == "rejected"


def test_facts_can_be_derived_from_an_already_assembled_capability_document() -> None:
    """The document stays authoritative: the hash is the one it was published with."""

    document = facts().capability_document()
    derived = RuntimeFacts.from_capability_document(
        document, task_rows=ROWS, declared_routes=ROUTES
    )
    assert derived.capability_hash == facts().capability_hash
    assert derived.capability_document() == document
    verdict = registry(derived).handshake(request_payload())
    assert verdict.accepted is True
    assert verdict.capability_hash == facts().capability_hash


def test_facts_derived_from_a_stream_a_document_answer_the_same_clauses() -> None:
    """The handshake reads whatever document discovery published, not its own."""

    contract = pytest.importorskip("synth_containers.cispo_contract")
    declaration = contract.CispoRuntimeDeclaration(
        container_id="container-1",
        container_image_digest="sha256:image",
        renderer_profile=contract.RendererProfileDeclaration(
            **{
                key: (tuple(value) if isinstance(value, list) else value)
                for key, value in PROFILE.to_payload().items()
            }
        ),
        discovery=contract.DiscoveryDeclaration(
            taskset_id="ts-1",
            taskset_version="v1",
            splits=("train", "val"),
            task_content_digests=True,
            deterministic_lookup=True,
            duplicate_free=True,
        ),
        policy=contract.PolicyBindingDeclaration(
            binding_transport="message_in_capture_out",
            wire_api="chat_completions",
            session_scoped_sampler_origin=True,
            embeds_credentials=False,
            revision_immutable_after_admission=True,
            records_policy_revision=True,
            probe_binding=True,
        ),
        lifecycle=contract.LifecycleDeclaration(
            advertised_concurrency=30,
            supports_idempotency=True,
            exactly_one_terminal_result=True,
            lease=contract.LeaseDeclaration(ttl_seconds=900.0, renewable=True),
        ),
        evidence=contract.EvidenceDeclaration(
            strict_prefix=True,
            masking=True,
            wire_objects=True,
            artifact_by_reference=True,
            tokens_in_tokens_out=True,
        ),
        reward=contract.RewardDeclaration(
            authority="container",
            binds_trace_digest=True,
            quiescence=True,
            horizon_clipping=True,
            channels=("outcome", "margin"),
            evaluation_plan_id="plan-1",
            settlement_window_seconds=150.0,
            deferred_scoring=True,
        ),
        recovery=contract.RecoveryDeclaration(restart=True, stale_discard=True),
        topology=contract.TopologyDeclaration(
            topology_id="topo-1",
            turn_model="sequential",
            actuation_model="direct_action",
            reward_relation="competitive_rank",
            agent_instances=(
                contract.AgentInstanceDeclaration(
                    "a1", "runner", "pt-trainable", "terra", True
                ),
                contract.AgentInstanceDeclaration(
                    "a2", "runner", "pt-frozen", "rock", False, "sha256:frozen-ckpt"
                ),
            ),
            teams=(
                contract.TeamDeclaration("terra", True, 1),
                contract.TeamDeclaration("rock", False, 1),
            ),
            horizon=contract.HorizonDeclaration(horizon_kind="wall_clock", value=5400.0),
            communication_channels=(
                contract.CommunicationChannelDeclaration("c1", "intra_team"),
            ),
            parameter_groups={"pt-trainable": "pg-1"},
        ),
        clock_skew_tolerance_seconds=2.0,
    )
    document = contract.build_cispo_capability_document(declaration, _surface())
    derived = RuntimeFacts.from_capability_document(document, task_rows=ROWS)
    assert derived.capability_hash == document["capability_hash"]
    verdict = registry(derived).handshake(request_payload())
    unhappy = {
        answer.clause_id: answer.verdict
        for answer in verdict.clauses
        if answer.verdict != "accepted"
    }
    assert unhappy == {}, unhappy


# --------------------------------------------------------------------------- #
# The port
# --------------------------------------------------------------------------- #


def test_the_adapter_satisfies_the_admission_port_structurally() -> None:
    contract = pytest.importorskip("synth_containers.cispo_contract")
    adapter = CispoHandshakeAdapter(facts(), clock=lambda: NOW)
    assert isinstance(adapter, contract.CispoAdmissionPort)


def test_the_adapter_serves_discovery_and_the_handshake_from_one_set_of_facts() -> None:
    adapter = CispoHandshakeAdapter(facts(), clock=lambda: NOW)
    assert adapter.cispo_health()["capability_hash"] == facts().capability_hash
    assert adapter.cispo_taskset()["splits"] == ["train", "val"]
    rows = adapter.cispo_taskset_tasks({"task_ids": ["task-1", "task-0"]})["rows"]
    assert [row["task_id"] for row in rows] == ["task-1", "task-0"]
    assert adapter.cispo_topology("topo-1")["topology_id"] == "topo-1"
    payload = adapter.cispo_handshake(request_payload())
    assert payload["accepted"] is True
    adapter.assert_admissible(payload["handshake_id"], payload["agreement_digest"])


def test_the_taskset_route_refuses_an_id_it_does_not_hold() -> None:
    from synth_containers.cispo_handshake import MalformedRequirementDocument

    adapter = CispoHandshakeAdapter(facts(), clock=lambda: NOW)
    with pytest.raises(MalformedRequirementDocument):
        adapter.cispo_taskset_tasks({"task_ids": ["task-0", "ghost"]})


def test_the_declared_route_names_are_the_contract_route_names() -> None:
    assert set(MANDATORY_ROUTE_NAMES) == set(ROUTES)
