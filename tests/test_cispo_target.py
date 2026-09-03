"""A real runtime installed behind the two declared CISPO ports.

Everything below asks the same question in different places: is the thing
behind the port real, or is it a shape? A capability that is declared but not
published, an attempt accepted with no worker behind it, a reward the caller
chose, evidence with no denominator -- each of those turns a 501 into something
worse than a 501, which is a run that starts and then trains on nothing.

The helpers here (``installed``, ``handshake_request``, ``admitted``,
``bind_request``) are the same ones the end-to-end test drives over HTTP, so a
drift in the declaration shows up in both.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from synth_containers.capabilities import RuntimeCapabilitySurface
from synth_containers.cispo_contract import (
    CispoCapabilityError,
    build_cispo_capability_document,
    cispo_declared_routes,
)
from synth_containers.cispo_handshake import (
    UNCONDITIONAL_MANDATORY_CLAUSES,
    AgreementMismatch,
    HandshakeUnknown,
    format_rfc3339,
)
from synth_containers.cispo_policy import (
    BehaviorIdentityImmutable,
    GlobalSamplerOrigin,
    SamplerUnreachable,
    UnprobedSampler,
)
from synth_containers.cispo_probe import PROBE_POLICY_KIND
from synth_containers.cispo_rollout import LifecycleError
from synth_containers.cispo_target import (
    CispoReferenceTarget,
    CispoTargetError,
    DeterministicSampler,
    cispo_capability_surface,
    reference_cispo_declaration,
    render_logprobs,
    render_tokens,
    task_rows_for,
)
from synth_containers.reference_runtime import ReferenceManagedRuntime

NOW = datetime(2026, 9, 3, 12, 0, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Helpers -- also imported by the end-to-end test
# --------------------------------------------------------------------------- #


def installed(
    *, target_count: int = 2, **kwargs: Any
) -> tuple[ReferenceManagedRuntime, CispoReferenceTarget]:
    """One runtime with the target installed, on a frozen handshake clock."""

    runtime = ReferenceManagedRuntime.counter_default(target=target_count)
    kwargs.setdefault("handshake_clock", lambda: NOW)
    target = CispoReferenceTarget.install(runtime, **kwargs)
    return runtime, target


def handshake_request(target: CispoReferenceTarget, **overrides: Any) -> dict[str, Any]:
    """The requirement document an executor sends before any spend."""

    facts = target.facts
    profile = facts.renderer_profile
    payload: dict[str, Any] = {
        "schema_version": "cispo.handshake.v1",
        "run_id": "run-1",
        "attempt": 1,
        "optimizer": {"name": "synth_optimizers.cispo", "version": "0.2.20"},
        "policy": {
            "provider": "session",
            "model_id": "family/model",
            "transport": "message_in_capture_out",
        },
        "renderer_profile": {
            "profile_id": profile.profile_id,
            "config_digest": profile.config_digest,
            "fingerprint": profile.fingerprint,
        },
        "requirements": list(UNCONDITIONAL_MANDATORY_CLAUSES),
        "accept_degraded": {},
        "topology": {
            "expected_topology_id": facts.topology.topology_id,
            "trainable_teams": ["team-0"],
            "partial_roster": "refuse",
        },
        "run_plan": {
            "group_size": 2,
            "groups_per_step": 1,
            "max_execution_slots": 2,
            "maximum_policy_lag": 1,
            "target_train_updates": 1,
            "expected_horizon_seconds": 2.0,
        },
        "taskset": {
            "taskset_id": facts.discovery.taskset_id,
            "split": "train",
            "task_ids": [facts.discovery.rows[0].task_id],
        },
        "clock": {
            "executor_time": format_rfc3339(NOW),
            "monotonic_source": "CLOCK_MONOTONIC",
        },
    }
    payload.update(overrides)
    return payload


def admitted(target: CispoReferenceTarget, **overrides: Any) -> dict[str, Any]:
    """One accepted handshake payload. A rejection here fails loudly."""

    verdict = target.admission.cispo_handshake(handshake_request(target, **overrides))
    assert verdict["accepted"], (
        verdict["rejected_mandatory_clauses"],
        verdict["unaccepted_degraded_clauses"],
    )
    return verdict


def bind_request(verdict: dict[str, Any], *, proxy_request_id: str = "prid-1", **over: Any):
    """A trainable binding over one session-scoped origin, credential inside it."""

    payload: dict[str, Any] = {
        "handshake_id": verdict["handshake_id"],
        "agreement_digest": verdict["agreement_digest"],
        "kind": "trainable",
        "policy_revision": 7,
        "model_family": "family",
        "model_id": "family/model",
        "agent_instance_id": "instance-0",
        "team_id": "team-0",
        "sampler_origin": {
            "base_url": f"https://sampler.invalid/v1/sessions/{proxy_request_id}",
            "credential": f"session-{proxy_request_id}",
            "policy_revision": 7,
            "behavior_fingerprint": "bf-7",
            "proxy_request_id": proxy_request_id,
            "wire_api": "chat_completions",
            "sampling_transport": "message_in_capture_out",
        },
        "sampling": {"temperature": 1.0, "top_p": 1.0, "max_tokens": 32, "seed": 3},
    }
    payload.update(over)
    return payload


def submit_request(
    verdict: dict[str, Any],
    config_id: str,
    *,
    rollout_id: str = "rollout_1",
    idempotency_key: str = "k-1",
    task_id: str | None = None,
    **over: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "handshake_id": verdict["handshake_id"],
        "agreement_digest": verdict["agreement_digest"],
        "rollout_id": rollout_id,
        "idempotency_key": idempotency_key,
        "config_id": config_id,
        "task_id": task_id or verdict["taskset_resolution"][0]["task_id"],
        "correlation": {
            "run_id": "run-1",
            "group_id": "group-1",
            "sample_index": 0,
            "seed": 3,
            "policy_revision": 7,
        },
        "horizon": {"horizon_kind": "steps", "value": 8, "seconds_per_unit": 0.25},
    }
    payload.update(over)
    return payload


def bound(target: CispoReferenceTarget, **over: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    verdict = admitted(target)
    binding = target.admission.cispo_bind_policy(bind_request(verdict, **over))
    return verdict, binding


def ran(target: CispoReferenceTarget, **over: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """One attempt, submitted and finalized. The common precondition below."""

    verdict, binding = bound(target)
    submitted = target.rollouts.cispo_submit_rollout(
        submit_request(verdict, binding["config_id"], **over)
    )
    target.rollouts.cispo_finalize_rollout(submitted["rollout_id"], {})
    return verdict, submitted


# --------------------------------------------------------------------------- #
# What installing actually declares
# --------------------------------------------------------------------------- #


def test_installing_adds_only_what_the_target_itself_supplies() -> None:
    """The overlay is three flags wide, and it is not a rewrite of the surface."""

    base = ReferenceManagedRuntime.counter_default(target=2).metadata().capabilities
    surface = cispo_capability_surface(base)

    assert surface.token_emission.token_ids is True
    assert surface.token_emission.logprobs is True
    assert surface.proxied_inference is True
    # Everything the counter runtime publishes is still its own.
    assert surface.trace_support == base.trace_support
    assert surface.reward_support == base.reward_support
    assert surface.terminate_support == base.terminate_support
    assert surface.multi_actor == base.multi_actor is False
    assert surface.checkpoint_semantics == base.checkpoint_semantics


def test_a_runtime_that_cannot_trace_still_serves_no_document() -> None:
    """The overlay never rescues a capability the runtime does not publish."""

    runtime = ReferenceManagedRuntime.counter_default(target=2)
    metadata = runtime.metadata()
    metadata.capabilities = replace(metadata.capabilities, trace_support=False)
    surface = cispo_capability_surface(metadata.capabilities)
    declaration = reference_cispo_declaration(
        runtime, container_image_digest="sha256:image"
    )
    with pytest.raises(CispoCapabilityError, match="trace_v5"):
        build_cispo_capability_document(declaration, surface)


def test_the_declared_horizon_counts_the_unit_the_runtime_counts() -> None:
    """A step horizon carries no duration of its own, so it declares one."""

    _, target = installed()
    horizon = target.declaration.topology.horizon
    assert horizon.horizon_kind == "steps"
    assert horizon.seconds_per_unit == 0.25
    assert horizon.horizon_seconds() == pytest.approx(2.0)


def test_discovery_rows_come_from_the_runtimes_own_catalog() -> None:
    runtime, target = installed()
    catalog_ids = {task.task_id for task in runtime.task_catalog().tasks}
    rows = task_rows_for(runtime)
    assert {row.task_id for row in rows} == catalog_ids
    assert all(row.content_digest.startswith("sha256:") for row in rows)
    assert all(row.topology_ref == target.declaration.topology.topology_id for row in rows)
    # Discovery, not decoration: the taskset route answers with the same rows.
    served = target.admission.cispo_taskset_tasks({"task_ids": sorted(catalog_ids)})
    assert {row["task_id"] for row in served["tasks"]} == catalog_ids


def test_the_hash_the_handshake_echoes_is_the_document_that_is_served() -> None:
    """One document. Reassembling it would give the executor a second one."""

    _, target = installed()
    document = target.capability_document
    assert document["routes"] == dict(sorted(cispo_declared_routes().items()))
    assert target.facts.capability_document() == document
    assert target.facts.capability_hash == document["capability_hash"]
    assert admitted(target)["capability_hash"] == document["capability_hash"]
    assert target.admission.cispo_health()["capability_hash"] == document["capability_hash"]


def test_every_unconditional_mandatory_clause_is_accepted() -> None:
    verdict = admitted(installed()[1])
    answers = {row["clause_id"]: row["verdict"] for row in verdict["clauses"]}
    for clause in UNCONDITIONAL_MANDATORY_CLAUSES:
        assert answers[clause] == "accepted", (clause, answers[clause])
    # A steps horizon does not read a wall clock, so the skew clause does not apply.
    assert "lifecycle.clock_skew" in verdict["not_applicable_clauses"]


def test_the_container_declares_the_unpaid_probe_kind_and_serves_it() -> None:
    _, target = installed()
    verdict = admitted(target)
    binding = target.admission.cispo_bind_policy(
        {
            "handshake_id": verdict["handshake_id"],
            "agreement_digest": verdict["agreement_digest"],
            "kind": PROBE_POLICY_KIND,
            "model_family": "family",
            "model_id": "family/model",
        }
    )
    assert binding["policy_kind"] == PROBE_POLICY_KIND
    assert binding["sampler_origin"] is None
    assert binding["provider_requests_expected"] == 0


# --------------------------------------------------------------------------- #
# Admission: three refusals, none of them a default
# --------------------------------------------------------------------------- #


def test_an_attempt_with_no_admitted_agreement_is_refused() -> None:
    _, target = installed()
    _, binding = bound(target)
    with pytest.raises(HandshakeUnknown):
        target.rollouts.cispo_submit_rollout(
            {
                "handshake_id": "hs_never_issued",
                "agreement_digest": "sha256:whatever",
                "rollout_id": "rollout_x",
                "config_id": binding["config_id"],
                "task_id": target.facts.discovery.rows[0].task_id,
            }
        )


def test_an_attempt_on_a_task_the_agreement_never_named_is_refused() -> None:
    _, target = installed()
    verdict, binding = bound(target)
    with pytest.raises(AgreementMismatch):
        target.rollouts.cispo_submit_rollout(
            submit_request(verdict, binding["config_id"], task_id="task-nobody-agreed")
        )


def test_an_attempt_with_no_bound_policy_is_refused() -> None:
    _, target = installed()
    verdict = admitted(target)
    with pytest.raises(CispoTargetError, match="bound policy config"):
        target.rollouts.cispo_submit_rollout(submit_request(verdict, ""))


def test_a_binding_whose_sampler_was_never_reachable_is_not_dispatchable() -> None:
    """The fail-closed default: no probe supplied means nothing is up."""

    _, target = installed(reachability=UnprobedSampler())
    verdict, binding = bound(target)
    assert binding["sampler_ready"] is False
    with pytest.raises(SamplerUnreachable):
        target.rollouts.cispo_submit_rollout(
            submit_request(verdict, binding["config_id"])
        )


def test_a_global_sampler_origin_is_refused_at_bind_time() -> None:
    _, target = installed()
    verdict = admitted(target)
    request = bind_request(verdict)
    request["sampler_origin"]["base_url"] = "https://sampler.invalid/v1"
    with pytest.raises(GlobalSamplerOrigin):
        target.admission.cispo_bind_policy(request)


def test_a_rollout_may_not_be_rebound_to_a_second_behavior_policy() -> None:
    _, target = installed()
    verdict, first = bound(target)
    other = bind_request(verdict, proxy_request_id="prid-2")
    other["policy_revision"] = 9
    other["sampler_origin"]["policy_revision"] = 9
    second = target.admission.cispo_bind_policy(other)
    target.rollouts.cispo_submit_rollout(submit_request(verdict, first["config_id"]))
    with pytest.raises(BehaviorIdentityImmutable):
        target.admit_attempt(
            submit_request(verdict, second["config_id"], idempotency_key="k-2")
        )


# --------------------------------------------------------------------------- #
# The attempt is real
# --------------------------------------------------------------------------- #


def test_submission_starts_an_episode_rather_than_only_accepting_one() -> None:
    _, target = installed()
    verdict, binding = bound(target)
    submitted = target.rollouts.cispo_submit_rollout(
        submit_request(verdict, binding["config_id"])
    )
    assert submitted["state"] == "running"
    assert submitted["task_content_digest"] == verdict["taskset_resolution"][0][
        "content_digest"
    ]
    # The state route is cheap: it reads, it does not finalize.
    assert target.rollouts.cispo_rollout_state("rollout_1")["state"] == "running"
    assert target.attempts.result("rollout_1").steps == 2


def test_a_resubmitted_key_yields_the_same_attempt_and_no_second_episode() -> None:
    _, target = installed()
    verdict, binding = bound(target)
    first = target.rollouts.cispo_submit_rollout(
        submit_request(verdict, binding["config_id"])
    )
    calls = target.sampler.calls
    again = target.rollouts.cispo_submit_rollout(
        submit_request(verdict, binding["config_id"], rollout_id="rollout_2")
    )
    assert again["duplicate"] is True
    assert again["rollout_id"] == first["rollout_id"]
    assert target.sampler.calls == calls, "a lost response must not double-spend"


def test_the_episode_reaches_the_environments_own_objective() -> None:
    _, target = installed(target_count=2)
    ran(target)
    result = target.attempts.result("rollout_1")
    assert result.passed is True
    assert result.measure == 1.0
    assert result.snapshot["values"]["count"] == 2


def test_the_sampling_call_carries_the_session_credential() -> None:
    """The credential leaves as an Authorization header and by no other path."""

    class _Recording(DeterministicSampler):
        seen: list[dict[str, str]] = []

        def post(self, url: str, *, headers: Any, body: Any) -> Any:
            self.seen.append(dict(headers))
            return super().post(url, headers=headers, body=body)

    sampler = _Recording()
    _, target = installed(transport=sampler, reachability=sampler)
    ran(target)
    assert sampler.seen, "the episode sampled nothing"
    for headers in sampler.seen:
        assert headers["Authorization"] == "Bearer session-prid-1"
        assert headers["X-Proxy-Request-Id"] == "prid-1"


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #


def test_the_attempt_seals_trainable_evidence_with_a_real_denominator() -> None:
    _, target = installed()
    ran(target)
    evidence = target.attempts.evidence("rollout_1")
    evidence.validate()

    assert len(evidence.calls) == 2, "one turn cannot prove a prefix"
    for call in evidence.calls:
        call.validate_for_training()
        assert call.token_capture_provenance == "engine_meta"
        assert len(call.generation_logprobs) == len(call.generation_token_ids)
        assert any(value != 0.0 for value in call.generation_logprobs)
        # The wire objects are kept beside the tokens rather than flattened away.
        assert call.wire_request["messages"]
        assert call.wire_response["choices"]

    # Turn two's prompt is the previous prompt plus what was generated.
    first, second = evidence.calls
    assert second.prompt_token_ids[: len(first.full_sequence)] == first.full_sequence

    (episode,) = evidence.episodes
    episode.validate()
    assert episode.trace_digest == evidence.trace_digest
    assert episode.probe is False


def test_a_sampler_that_returns_no_logprobs_is_refused_rather_than_recorded() -> None:
    """An importance ratio with no denominator is not evidence to fall back on."""

    class _NoLogprobs(DeterministicSampler):
        def post(self, url: str, *, headers: Any, body: Any) -> Any:
            payload = dict(super().post(url, headers=headers, body=body))
            payload["logprobs"] = {"completion": []}
            return payload

    sampler = _NoLogprobs()
    _, target = installed(transport=sampler, reachability=sampler)
    verdict, binding = bound(target)
    with pytest.raises(CispoTargetError, match="denominator"):
        target.rollouts.cispo_submit_rollout(
            submit_request(verdict, binding["config_id"])
        )


def test_an_attempt_that_sealed_nothing_has_no_trace_to_serve() -> None:
    _, target = installed()
    with pytest.raises(CispoTargetError, match="sealed no evidence"):
        target.attempts.evidence("rollout_never_ran")


def test_the_renderer_is_content_addressed_so_a_prefix_survives_a_restart() -> None:
    assert render_tokens("count=1") == render_tokens("count=1")
    assert render_tokens("count=1") != render_tokens("count=2")
    logprobs = render_logprobs(render_tokens("increment"))
    assert all(-1.0 < value < 0.0 for value in logprobs)


# --------------------------------------------------------------------------- #
# Finalize, reward, terminate
# --------------------------------------------------------------------------- #


def test_finalize_attests_quiescence_before_anything_is_scored() -> None:
    _, target = installed()
    verdict, binding = bound(target)
    target.rollouts.cispo_submit_rollout(submit_request(verdict, binding["config_id"]))
    outcome = target.rollouts.cispo_finalize_rollout("rollout_1", {})
    assert outcome["state"] == "completed"
    assert outcome["quiescence"]["quiesced"] is True
    assert outcome["quiescence"]["clipped"] is False
    assert outcome["horizon_applied_seconds"] == pytest.approx(2.0)


def test_scoring_before_the_horizon_attestation_is_refused() -> None:
    _, target = installed()
    verdict, binding = bound(target)
    target.rollouts.cispo_submit_rollout(submit_request(verdict, binding["config_id"]))
    with pytest.raises(LifecycleError, match="not been finalized"):
        target.rollouts.cispo_reward({"rollout_id": "rollout_1"})


def test_the_reward_is_the_containers_own_and_a_caller_may_not_supply_one() -> None:
    _, target = installed()
    ran(target)
    with pytest.raises(CispoTargetError, match="reward authority"):
        target.rollouts.cispo_reward({"rollout_id": "rollout_1", "measure": 1.0})
    with pytest.raises(CispoTargetError, match="reward authority"):
        target.rollouts.cispo_reward(
            {"rollout_id": "rollout_1", "measures": {"team-0": 1.0}}
        )


def test_the_receipt_binds_the_rollout_and_the_sealed_trace() -> None:
    _, target = installed()
    ran(target)
    digest = target.attempts.evidence("rollout_1").trace_digest
    receipt = target.rollouts.cispo_reward(
        {"rollout_id": "rollout_1", "trace_digest": digest}
    )
    assert receipt["rollout_id"] == "rollout_1"
    assert receipt["trace_digest"] == digest
    assert receipt["scoring_state"] == "completed"
    assert receipt["channels"] == [
        {"channel_id": "score::team-0", "team_id": "team-0", "measure": 1.0, "rank": None}
    ]
    assert receipt["optimized_channel"] == "score::team-0"
    assert receipt["horizon"]["quiescence_attested"] is True
    # Final scoring is idempotent: a second read is the same receipt.
    assert target.rollouts.cispo_reward({"rollout_id": "rollout_1"}) == receipt


def test_scoring_a_trace_the_attempt_did_not_seal_is_refused() -> None:
    _, target = installed()
    ran(target)
    with pytest.raises(CispoTargetError, match="sealed"):
        target.rollouts.cispo_reward(
            {"rollout_id": "rollout_1", "trace_digest": "sha256:someone-elses"}
        )


def test_a_zero_measure_is_a_scored_result_and_not_an_absent_one() -> None:
    """A counter whose target is past the horizon scores zero, with a channel."""

    _, target = installed(target_count=9)
    ran(target)
    receipt = target.rollouts.cispo_reward({"rollout_id": "rollout_1"})
    assert receipt["channels"][0]["measure"] == 0.0
    assert receipt["metadata"]["passed"] is False
    assert "missing_evidence" not in receipt["metadata"]


def test_terminating_a_finished_attempt_reports_rather_than_re_terminating() -> None:
    _, target = installed()
    ran(target)
    first = target.rollouts.cispo_terminate_rollout("rollout_1", {"reason": "operator"})
    assert first["already_terminal"] is True
    assert first["terminal"]["terminal_status"] == "completed"
    assert first == target.rollouts.cispo_terminate_rollout("rollout_1", {})


def test_cancelling_a_live_attempt_reaches_exactly_one_terminal_result() -> None:
    _, target = installed()
    verdict, binding = bound(target)
    target.rollouts.cispo_submit_rollout(submit_request(verdict, binding["config_id"]))
    cancelled = target.rollouts.cispo_terminate_rollout("rollout_1", {"reason": "drain"})
    assert cancelled["already_terminal"] is False
    assert cancelled["terminal"]["kind"] == "cancellation"
    with pytest.raises(LifecycleError, match="exactly one terminal result"):
        target.rollouts.cispo_finalize_rollout("rollout_1", {})


# --------------------------------------------------------------------------- #
# Lease, events, artifacts
# --------------------------------------------------------------------------- #


def test_a_heartbeat_extends_the_grant_and_never_moves_the_deadline() -> None:
    clock = [0.0]
    _, target = installed(clock=lambda: clock[0])
    verdict, binding = bound(target)
    submitted = target.rollouts.cispo_submit_rollout(
        submit_request(verdict, binding["config_id"])
    )
    deadline = submitted["lease_deadline_at"]
    clock[0] = 1.0
    renewed = target.rollouts.cispo_renew_lease("rollout_1", {"handshake_id": "hs"})
    assert renewed["renewals"] == 1
    assert renewed["lease_expires_at"] > submitted["lease_expires_at"]
    assert renewed["lease_deadline_at"] == deadline


def test_the_event_cursor_is_monotone_and_resumable() -> None:
    _, target = installed()
    ran(target)
    page = target.rollouts.cispo_rollout_events("rollout_1", limit=3)
    cursors = [int(row["cursor"]) for row in page["events"] if row["cursor"] is not None]
    assert cursors and cursors == sorted(cursors)
    rest = target.rollouts.cispo_rollout_events("rollout_1", cursor=page["next_cursor"])
    resumed = [int(row["cursor"]) for row in rest["events"] if row["cursor"] is not None]
    assert resumed[0] > cursors[-1], "a reconnecting reader resumes rather than replays"
    kinds = {row["kind"] for row in page["events"] + rest["events"]}
    assert {"rollout.attempt.accepted", "rollout.attempt.terminal"} <= kinds


def test_the_artifact_inventory_names_the_trace_it_serves_inline() -> None:
    _, target = installed()
    ran(target)
    inventory = target.rollouts.cispo_rollout_artifacts("rollout_1")
    (row,) = inventory["artifacts"]
    assert row["content_digest"] == target.attempts.evidence("rollout_1").trace_digest
    assert row["inline"] is True
    assert row["route"] == cispo_declared_routes()["trace_route"]
    assert row["size_bytes"] > 0


def test_admission_is_refused_past_the_advertised_concurrency() -> None:
    """Only admission is ever refused; executed work never is."""

    _, target = installed()
    verdict, binding = bound(target)
    live = target.lifecycle.advertised_concurrency
    assert live == target.declaration.lifecycle.advertised_concurrency
    for index in range(live):
        target.rollouts.cispo_submit_rollout(
            submit_request(
                verdict,
                binding["config_id"],
                rollout_id=f"rollout_{index}",
                idempotency_key=f"k-{index}",
            )
        )
    with pytest.raises(LifecycleError, match="admission refused"):
        target.rollouts.cispo_submit_rollout(
            submit_request(
                verdict, binding["config_id"], rollout_id="rollout_over", idempotency_key="k-over"
            )
        )


def test_an_expired_handshake_admits_no_further_attempt() -> None:
    moment = [NOW]
    _, target = installed(handshake_clock=lambda: moment[0])
    verdict, binding = bound(target)
    moment[0] = NOW + timedelta(seconds=target.facts.handshake_ttl_seconds + 1)
    from synth_containers.cispo_handshake import HandshakeExpired

    with pytest.raises(HandshakeExpired):
        target.rollouts.cispo_submit_rollout(
            submit_request(verdict, binding["config_id"])
        )


# --------------------------------------------------------------------------- #
# The runtime hooks
# --------------------------------------------------------------------------- #


def test_an_uninstalled_runtime_still_declares_nothing() -> None:
    runtime = ReferenceManagedRuntime.counter_default(target=2)
    assert runtime.cispo_declaration() is None
    assert runtime.cispo_admission() is None
    assert runtime.cispo_rollouts() is None


def test_installing_publishes_the_declaration_and_both_ports() -> None:
    runtime, target = installed()
    assert runtime.cispo_declaration() is target.declaration
    assert runtime.cispo_admission() is target.admission
    assert runtime.cispo_rollouts() is target.rollouts
    assert isinstance(runtime.metadata().capabilities, RuntimeCapabilitySurface)
    assert runtime.metadata().capabilities.proxied_inference is True
