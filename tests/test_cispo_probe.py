"""The probe episode: the unpaid walk that turns the handshake's claims into evidence.

A probe exists to make a container's claims falsifiable before one paid
request. So it has to do three things, and each of them is a separate way a
container passes conformance while being unfit to train against.

It has to walk the *whole* path -- submit, state, events, lease renewal, trace,
reward, finalize, terminate, an idempotent resubmit of the same key, and one
cancellation. A probe that skips renewal proves nothing about the route the
executor will lean on for an hour-scale episode.

It has to produce evidence the executor can check the *shape* of: token and
logprob lengths that agree, a sampled mask, renderer stamping, a monotone event
cursor, exactly one terminal result, a reward bound to the rollout id and the
trace digest, a quiescence attestation where quiescence was the accepted
clause, and prefix consistency across two turns -- which is why one turn is not
enough.

And it has to be *unmistakable*. A container whose probe evidence looks like
real evidence fails conformance, because the failure it hides is training on a
canned generation. The marking here is structural: every mark is a field the
optimizer's own training gate reads, not a label.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from synth_containers.cispo_handshake import (
    HandshakeRegistry,
    RuntimeFacts,
)
from synth_containers.cispo_probe import (
    PROBE_POLICY_KIND,
    PROBE_ROLLOUT_PREFIX,
    PROBE_TOKEN_CAPTURE_PROVENANCE,
    REQUIRED_PROBE_OPERATIONS,
    CispoProbeAdapter,
    ProbeError,
    ProbeNotDistinguishable,
    ProbeSession,
    ProbeUnsupported,
    assert_probe_distinguishable,
    assert_probe_path_exercised,
    bind_probe_policy,
    run_probe,
)
from tests.test_cispo_handshake import (
    NOW,
    OPTIMIZER_RECORDS,
    facts,
    optimizer,
    registry,
    request_payload,
)

OPTIMIZER_PROBE = optimizer("synth_optimizers.rl.probe")


def admitted(runtime: RuntimeFacts | None = None, **request_overrides: Any):
    """One admitted agreement, because a probe is gated exactly like an attempt."""

    runtime = runtime or facts()
    ledger: HandshakeRegistry = registry(runtime)
    verdict = ledger.handshake(request_payload(**request_overrides))
    assert verdict.accepted, verdict.rejected_mandatory_clauses
    return runtime, ledger, ledger.assert_admissible(
        verdict.handshake_id, verdict.agreement_digest
    )


# --------------------------------------------------------------------------- #
# Binding
# --------------------------------------------------------------------------- #


def test_a_build_without_a_probe_kind_refuses_the_binding() -> None:
    """When probe is unsupported, one real paid canary is the declared fallback."""

    runtime, _, agreement = admitted(facts(probe_binding=False))
    with pytest.raises(ProbeUnsupported):
        bind_probe_policy(runtime, agreement, model_family="f", model_id="m")


def test_the_probe_binding_carries_no_credential_and_expects_no_provider_call() -> None:
    runtime, _, agreement = admitted()
    binding = bind_probe_policy(
        runtime, agreement, model_family="policy_family", model_id="vendor/policy-20b"
    )
    payload = binding.to_payload()
    assert payload["policy_kind"] == PROBE_POLICY_KIND
    assert payload["trainable"] is False
    assert payload["credential"] is None
    assert payload["sampler_origin"] is None
    assert payload["provider_requests_expected"] == 0
    assert payload["agreement_digest"] == agreement.agreement_digest


def test_the_probe_binding_is_gated_by_the_handshake() -> None:
    """A probe is admitted the same way an attempt is: no ticket, no path."""

    from synth_containers.cispo_handshake import HandshakeUnknown

    runtime, _, agreement = admitted()
    adapter = CispoProbeAdapter(runtime)
    assert adapter.declares_probe_binding() is True

    from synth_containers.cispo_handshake import CispoHandshakeAdapter

    gate = CispoHandshakeAdapter(runtime, clock=lambda: NOW)
    with pytest.raises(HandshakeUnknown):
        gate.cispo_bind_policy(
            {
                "kind": PROBE_POLICY_KIND,
                "handshake_id": "hs_never_issued",
                "agreement_digest": agreement.agreement_digest,
            }
        )


def test_a_non_probe_kind_is_a_typed_refusal_not_a_silent_stub() -> None:
    from synth_containers.cispo_handshake import CispoHandshakeAdapter

    runtime = facts()
    gate = CispoHandshakeAdapter(runtime, clock=lambda: NOW)
    payload = gate.cispo_handshake(request_payload())
    with pytest.raises(ProbeError):
        gate.cispo_bind_policy(
            {
                "kind": "tinker_sampler",
                "handshake_id": payload["handshake_id"],
                "agreement_digest": payload["agreement_digest"],
            }
        )


# --------------------------------------------------------------------------- #
# The whole path
# --------------------------------------------------------------------------- #


def test_a_probe_exercises_every_required_operation() -> None:
    runtime, _, agreement = admitted()
    attempt, report = run_probe(runtime, agreement, task_id="task-0", seed=7)
    assert set(attempt.operations) == REQUIRED_PROBE_OPERATIONS
    assert tuple(report.operations) == tuple(sorted(REQUIRED_PROBE_OPERATIONS))
    assert assert_probe_path_exercised(attempt) == tuple(sorted(REQUIRED_PROBE_OPERATIONS))


def test_a_probe_that_skipped_an_operation_is_refused() -> None:
    """Operations are recorded as they run, so the report cannot overclaim."""

    runtime, _, agreement = admitted()
    binding = bind_probe_policy(runtime, agreement, model_family="f", model_id="m")
    session = ProbeSession(binding, task_id="task-0", seed=7)
    session.submit()
    session.state()
    session.trace()
    session.reward()
    session.finalize()
    session.terminate()
    # No renewal, no resubmit, no cancellation.
    with pytest.raises(ProbeError) as excinfo:
        session.attempt()
    message = str(excinfo.value)
    assert "renew" in message and "cancellation" in message


def test_operations_cannot_run_out_of_order() -> None:
    runtime, _, agreement = admitted()
    binding = bind_probe_policy(runtime, agreement, model_family="f", model_id="m")
    session = ProbeSession(binding, task_id="task-0", seed=7)
    with pytest.raises(ProbeError):
        session.reward()
    session.submit()
    with pytest.raises(ProbeError):
        session.finalize()


def test_the_resubmit_yields_the_same_logical_attempt() -> None:
    runtime, _, agreement = admitted()
    binding = bind_probe_policy(runtime, agreement, model_family="f", model_id="m")
    session = ProbeSession(binding, task_id="task-0", seed=7)
    submitted = session.submit()
    resubmitted = session.resubmit()
    assert resubmitted["rollout_id"] == submitted["rollout_id"]
    assert resubmitted["deduplicated"] is True


def test_the_cancellation_is_its_own_attempt_so_the_probe_keeps_one_terminal() -> None:
    runtime, _, agreement = admitted()
    attempt, _ = run_probe(runtime, agreement, task_id="task-0", seed=7)
    assert attempt.terminal_results == ("completed",)
    assert attempt.cancelled_rollout_id != attempt.rollout_id
    assert attempt.cancelled_rollout_id.startswith(attempt.rollout_id)


def test_event_cursors_are_monotone() -> None:
    runtime, _, agreement = admitted()
    attempt, _ = run_probe(runtime, agreement, task_id="task-0", seed=7)
    cursors = list(attempt.event_cursors)
    assert cursors == sorted(set(cursors))
    assert cursors[0] == 1


def test_a_probe_is_deterministic_for_the_same_agreement_and_task() -> None:
    runtime, _, agreement = admitted()
    first, _ = run_probe(runtime, agreement, task_id="task-0", seed=7)
    second, _ = run_probe(runtime, agreement, task_id="task-0", seed=7)
    assert first.to_payload() == second.to_payload()
    other, _ = run_probe(runtime, agreement, task_id="task-1", seed=7)
    assert other.rollout_id != first.rollout_id


# --------------------------------------------------------------------------- #
# Evidence shape
# --------------------------------------------------------------------------- #


def test_two_turns_stitch_under_the_strict_prefix_rule() -> None:
    """Never retokenize new text onto old ids; a divergence has no branch record."""

    runtime, _, agreement = admitted()
    attempt, _ = run_probe(runtime, agreement, task_id="task-0", seed=7)
    assert len(attempt.calls) >= 2
    first, second = attempt.calls[0], attempt.calls[1]
    sequence = first.full_sequence
    assert tuple(second.prompt_token_ids)[: len(sequence)] == sequence
    assert second.branch_id == first.branch_id


def test_a_forked_second_turn_is_refused_before_it_reaches_the_wire() -> None:
    from synth_containers.cispo_probe import assert_strict_prefix

    runtime, _, agreement = admitted()
    attempt, _ = run_probe(runtime, agreement, task_id="task-0", seed=7)
    forked = replace(attempt.calls[1], prompt_token_ids=(9_999, 9_998, 9_997))
    with pytest.raises(ProbeError):
        assert_strict_prefix(attempt.calls[0], forked)


def test_token_logprob_and_mask_lengths_agree_on_every_call() -> None:
    runtime, _, agreement = admitted()
    attempt, _ = run_probe(runtime, agreement, task_id="task-0", seed=7)
    for call in attempt.calls:
        generated = len(call.generation_token_ids)
        assert len(call.generation_logprobs) == generated
        assert len(call.sampled_mask) == generated
        assert call.stop_token_ids == runtime.renderer_profile.stop_token_ids
        assert call.renderer_profile_fingerprint == runtime.renderer_profile.fingerprint
        assert call.finish_reason == "stop_token"


def test_logprobs_are_finite_non_sentinel_and_not_identically_zero() -> None:
    from synth_containers.cispo_probe import LOGPROB_SENTINEL

    runtime, _, agreement = admitted()
    attempt, _ = run_probe(runtime, agreement, task_id="task-0", seed=7)
    values = [value for call in attempt.calls for value in call.generation_logprobs]
    assert values
    assert all(value < 0.0 for value in values)
    assert LOGPROB_SENTINEL not in values


def test_the_segment_masks_only_what_the_policy_sampled() -> None:
    runtime, _, agreement = admitted()
    attempt, _ = run_probe(runtime, agreement, task_id="task-0", seed=7)
    (segment,) = attempt.episode.segments
    assert len(segment.loss_mask) == len(segment.token_ids)
    assert len(segment.behavior_logprobs) == len(segment.token_ids)
    sampled = sum(len(call.generation_token_ids) for call in attempt.calls)
    assert sum(segment.loss_mask) == sampled
    # Everything the harness or the environment contributed stays masked, and
    # carries no behavior logprob to be mistaken for one.
    for flag, value in zip(segment.loss_mask, segment.behavior_logprobs, strict=True):
        assert flag or value == 0.0


def test_the_reward_is_bound_to_the_rollout_and_the_trace_digest() -> None:
    runtime, _, agreement = admitted()
    attempt, report = run_probe(runtime, agreement, task_id="task-0", seed=7)
    assert attempt.reward.rollout_id == attempt.rollout_id
    assert attempt.reward.trace_digest == attempt.trace_digest
    assert attempt.episode.trace_digest == attempt.trace_digest
    assert attempt.reward.optimized_channel in {
        channel.channel_id for channel in attempt.reward.channels
    }
    assert report.trace_digest == attempt.trace_digest


def test_quiescence_is_attested_when_quiescence_was_the_accepted_clause() -> None:
    runtime, _, agreement = admitted()
    assert agreement.quiescence_accepted is True
    attempt, report = run_probe(runtime, agreement, task_id="task-0", seed=7)
    assert attempt.reward.horizon.quiescence_attested is True
    assert attempt.reward.horizon.clipped is False
    assert report.quiescence_attested is True


def test_a_clipping_build_attests_the_snapshot_instead() -> None:
    """Which attestation the probe carries is the clause verdict, not a choice."""

    clipping = facts(
        reward=replace(facts().reward, quiescence=False, horizon_clipping=True)
    )
    runtime, _, agreement = admitted(
        clipping,
        accept_degraded={"reward.horizon_quiescence": "horizon_clipped_snapshot"},
    )
    assert agreement.quiescence_accepted is False
    attempt, report = run_probe(runtime, agreement, task_id="task-0", seed=7)
    assert attempt.reward.horizon.quiescence_attested is False
    assert attempt.reward.horizon.clipped is True
    assert report.quiescence_attested is False


def test_a_single_turn_probe_is_refused() -> None:
    runtime, _, agreement = admitted()
    binding = bind_probe_policy(runtime, agreement, model_family="f", model_id="m")
    with pytest.raises(ProbeError):
        ProbeSession(binding, task_id="task-0", seed=7, turns=1)


# --------------------------------------------------------------------------- #
# Distinguishability
# --------------------------------------------------------------------------- #


def test_probe_evidence_is_structurally_distinguishable() -> None:
    runtime, _, agreement = admitted()
    attempt, report = run_probe(runtime, agreement, task_id="task-0", seed=7)
    assert attempt.rollout_id.startswith(PROBE_ROLLOUT_PREFIX)
    assert attempt.episode.probe is True
    assert report.trainable is False
    for call in attempt.calls:
        assert call.token_capture_provenance == PROBE_TOKEN_CAPTURE_PROVENANCE
        assert call.trainable is False
        assert call.wire_response()["synthetic"] is True
        assert call.wire_response()["provider"] is None
    assert attempt.reward.to_payload()["metadata"]["probe"] is True
    assert_probe_distinguishable(attempt)


def test_an_attempt_missing_a_mark_is_refused_by_the_container_itself() -> None:
    """The container never puts indistinguishable probe evidence on the wire."""

    runtime, _, agreement = admitted()
    attempt, _ = run_probe(runtime, agreement, task_id="task-0", seed=7)

    trainable = replace(
        attempt, calls=(replace(attempt.calls[0], trainable=True), *attempt.calls[1:])
    )
    with pytest.raises(ProbeNotDistinguishable):
        assert_probe_distinguishable(trainable)

    engine_provenance = replace(
        attempt,
        calls=(
            replace(attempt.calls[0], token_capture_provenance="engine_meta"),
            *attempt.calls[1:],
        ),
    )
    with pytest.raises(ProbeNotDistinguishable):
        assert_probe_distinguishable(engine_provenance)

    renamed = replace(attempt, rollout_id="ro-looks-real")
    with pytest.raises(ProbeNotDistinguishable):
        assert_probe_distinguishable(renamed)


def test_a_probe_episode_cannot_be_built_unmarked() -> None:
    from synth_containers.cispo_probe import ProbeEpisode, ProbeSegment

    with pytest.raises(ProbeNotDistinguishable):
        ProbeEpisode(
            rollout_id="probe_x",
            task_id="task-0",
            seed=0,
            policy_revision=0,
            behavior_fingerprint="f",
            segments=(
                ProbeSegment(
                    token_ids=(1, 2),
                    loss_mask=(0, 1),
                    behavior_logprobs=(0.0, -0.5),
                    call_ids=("c",),
                ),
            ),
            terminal_status="completed",
            trace_digest="sha256:t",
            probe=False,
        )


# --------------------------------------------------------------------------- #
# The optimizer's own validator
# --------------------------------------------------------------------------- #


def _optimizer_probe_attempt(attempt: Any) -> Any:
    """Rebuild the optimizer's typed records straight from the wire payload.

    No translation layer: the payload's keys are the optimizer's constructor
    argument names, which is the point of matching them.
    """

    records = OPTIMIZER_RECORDS
    payload = attempt.to_payload()
    behavior_raw = payload["behavior"]
    profile = records.RendererProfile(
        profile_id=behavior_raw["renderer_profile"]["profile_id"],
        package=behavior_raw["renderer_profile"]["package"],
        package_version=behavior_raw["renderer_profile"]["package_version"],
        config_digest=behavior_raw["renderer_profile"]["config_digest"],
        tokenizer_id=behavior_raw["renderer_profile"]["tokenizer_id"],
        tokenizer_digest=behavior_raw["renderer_profile"]["tokenizer_digest"],
        stop_token_ids=tuple(behavior_raw["renderer_profile"]["stop_token_ids"]),
        modalities=tuple(behavior_raw["renderer_profile"]["modalities"]),
        add_generation_prompt=behavior_raw["renderer_profile"]["add_generation_prompt"],
    )
    behavior = records.BehaviorFingerprint(
        renderer_profile=profile,
        model_family=behavior_raw["model_family"],
        model_id=behavior_raw["model_id"],
        policy_revision=behavior_raw["policy_revision"],
        wire_api=behavior_raw["wire_api"],
        sampling_transport=behavior_raw["sampling_transport"],
        sampling=records.SamplingProfile(**behavior_raw["sampling"]),
    )
    episode_raw = payload["episode"]
    episode = records.TrainableEpisode(
        rollout_id=episode_raw["rollout_id"],
        task_id=episode_raw["task_id"],
        seed=episode_raw["seed"],
        policy_revision=episode_raw["policy_revision"],
        behavior_fingerprint=episode_raw["behavior_fingerprint"],
        segments=tuple(
            records.TrainableSegment(**segment) for segment in episode_raw["segments"]
        ),
        terminal_status=episode_raw["terminal_status"],
        usage=episode_raw["usage"],
        trace_digest=episode_raw["trace_digest"],
        probe=episode_raw["probe"],
    )
    reward_raw = payload["reward"]
    reward = records.RewardRecord(
        reward_id=reward_raw["reward_id"],
        rollout_id=reward_raw["rollout_id"],
        trace_digest=reward_raw["trace_digest"],
        channels=tuple(records.RewardChannel(**item) for item in reward_raw["channels"]),
        optimized_channel=reward_raw["optimized_channel"],
        terminal_status=reward_raw["terminal_status"],
        evaluation_plan_id=reward_raw["evaluation_plan_id"],
        horizon=records.HorizonEvidence(**reward_raw["horizon"]),
        metadata=reward_raw["metadata"],
    )
    return (
        OPTIMIZER_PROBE.ProbeAttempt(
            rollout_id=payload["rollout_id"],
            behavior=behavior,
            calls=tuple(records.InferenceCall(**call) for call in payload["calls"]),
            episode=episode,
            reward=reward,
            event_cursors=tuple(payload["event_cursors"]),
            terminal_results=tuple(payload["terminal_results"]),
            operations=frozenset(payload["operations"]),
            resubmit_rollout_id=payload["resubmit_rollout_id"],
            cancelled_rollout_id=payload["cancelled_rollout_id"],
            trace_digest=payload["trace_digest"],
            metadata=payload["metadata"],
        ),
        profile,
    )


@pytest.mark.skipif(
    OPTIMIZER_PROBE is None or OPTIMIZER_RECORDS is None,
    reason="optimizer source not reachable",
)
def test_the_probe_passes_the_optimizers_own_validate_probe() -> None:
    """The real validator, on the real payload, not a restatement of it here."""

    runtime, _, agreement = admitted()
    attempt, _ = run_probe(
        runtime,
        agreement,
        task_id="task-0",
        seed=7,
        model_family="policy_family",
        model_id="vendor/policy-20b",
    )
    opt_attempt, profile = _optimizer_probe_attempt(attempt)
    report = OPTIMIZER_PROBE.validate_probe(
        opt_attempt, expected_profile=profile, quiescence_accepted=True
    )
    assert report.trainable is False
    assert set(report.operations) == set(REQUIRED_PROBE_OPERATIONS)
    assert report.trace_digest == attempt.trace_digest


@pytest.mark.skipif(
    OPTIMIZER_PROBE is None or OPTIMIZER_RECORDS is None,
    reason="optimizer source not reachable",
)
def test_the_optimizers_training_gate_refuses_every_probe_call() -> None:
    """The line around training, drawn by the optimizer's own code."""

    runtime, _, agreement = admitted()
    attempt, _ = run_probe(runtime, agreement, task_id="task-0", seed=7)
    opt_attempt, _ = _optimizer_probe_attempt(attempt)
    OPTIMIZER_PROBE.assert_probe_not_trainable(opt_attempt)
    for call in opt_attempt.calls:
        with pytest.raises(OPTIMIZER_RECORDS.EvidenceError):
            call.validate_for_training()
    with pytest.raises(OPTIMIZER_RECORDS.EvidenceError):
        opt_attempt.episode.validate()


@pytest.mark.skipif(
    OPTIMIZER_PROBE is None or OPTIMIZER_RECORDS is None,
    reason="optimizer source not reachable",
)
def test_the_optimizer_reads_the_two_turns_as_a_strict_prefix_stitch() -> None:
    runtime, _, agreement = admitted()
    attempt, _ = run_probe(runtime, agreement, task_id="task-0", seed=7)
    opt_attempt, _ = _optimizer_probe_attempt(attempt)
    for previous, following in zip(
        opt_attempt.calls, opt_attempt.calls[1:], strict=False
    ):
        OPTIMIZER_RECORDS.assert_strict_prefix(previous, following)


def test_the_same_properties_hold_without_the_optimizer_on_the_path() -> None:
    """The container-side restatement, so this suite still checks the shape.

    The two tests above run the optimizer's real validator when its source is
    reachable. This one asserts the same properties directly and always runs,
    so a checkout with no sibling optimizer is not silently unchecked.
    """

    runtime, _, agreement = admitted()
    attempt, _ = run_probe(runtime, agreement, task_id="task-0", seed=7)
    assert_probe_path_exercised(attempt)
    assert_probe_distinguishable(attempt)
    assert len(attempt.calls) >= 2
    for call in attempt.calls:
        assert call.rollout_id == attempt.rollout_id
        assert call.behavior_fingerprint == attempt.behavior.value
        assert call.policy_revision == attempt.behavior.policy_revision
        assert tuple(call.stop_token_ids) == runtime.renderer_profile.stop_token_ids
    assert attempt.episode.behavior_fingerprint == attempt.behavior.value
    assert attempt.episode.rollout_id == attempt.rollout_id
    assert attempt.reward.rollout_id == attempt.rollout_id
    assert attempt.reward.trace_digest == attempt.trace_digest


# --------------------------------------------------------------------------- #
# The adapter
# --------------------------------------------------------------------------- #


def test_the_probe_adapter_returns_an_attempt_and_a_receipt_line() -> None:
    runtime, _, agreement = admitted()
    adapter = CispoProbeAdapter(runtime)
    payload = adapter.run(agreement, {"task_id": "task-0", "seed": 7})
    assert payload["attempt"]["metadata"]["probe"] is True
    assert payload["attempt"]["metadata"]["provider_requests"] == 0
    assert payload["report"]["trainable"] is False
    assert payload["report"]["schema_version"] == "cispo.probe_report.v1"
