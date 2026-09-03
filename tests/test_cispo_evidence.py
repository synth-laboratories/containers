"""Trainable evidence: stitching, forking, declared authorship, and the seal.

The optimizer's own record contract is imported when it is importable from this
worktree, and its validators are then run against exactly the payloads this
module emits. When it is not importable, the same properties are asserted
directly; :func:`_optimizer_records` is the single place that decides which, and
``test_evidence_passes_the_optimizer_validators`` reports which path it took.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest

from synth_containers.cispo_evidence import (
    LOGPROB_SENTINEL,
    SPAN_EXTENSION_KEY,
    BehaviorBindingV1,
    CispoEvidenceAdapter,
    CispoEvidenceBuilder,
    CompactionProvenanceV1,
    EvidenceError,
    InferenceCallV1,
    RendererProfileV1,
    SamplingProfileV1,
    assert_strict_prefix,
    inference_call_from_span,
)
from synth_containers.tracing.models.spans import SpanKind, SpanV5, UsageProvenance, UsageV5
from synth_containers.tracing.models.tokens import (
    TokenCaptureProvenance,
    TokenCaptureV5,
    TokenSequenceRefV1,
)

PROFILE = RendererProfileV1(
    profile_id="renderers.pinned.low.v1",
    package="renderers",
    package_version="0.1.11",
    config_digest="sha256:config",
    tokenizer_id="pinned/tokenizer",
    tokenizer_digest="sha256:tokenizer",
    stop_token_ids=(200002, 199999),
)

BINDING = BehaviorBindingV1(
    renderer_profile=PROFILE,
    model_family="pinned_family",
    model_id="pinned/model",
    policy_revision=17,
    wire_api="chat_completions",
    sampling_transport="message_in_capture_out",
    sampling=SamplingProfileV1(temperature=1.0, top_p=1.0, seed=11),
)

SYSTEM = (1001, 1002, 1003, 1004)


def _logprobs(count: int, *, start: float = -0.11) -> tuple[float, ...]:
    return tuple(round(start - index * 0.01, 6) for index in range(count))


def _call(
    index: int,
    *,
    prompt: tuple[int, ...],
    generation: tuple[int, ...],
    branch_id: str = "root",
    parent_branch_id: str | None = None,
    compaction: CompactionProvenanceV1 | None = None,
    author_kind: str = "policy",
    role_id: str | None = None,
    policy_type_id: str | None = None,
    trainable: bool = True,
    provenance: str = "engine_meta",
    logprobs: tuple[float, ...] | None = None,
    finish_reason: str = "stop_token",
    wire_response: dict[str, Any] | None = None,
    agent_instance_id: str | None = "instance-a",
) -> InferenceCallV1:
    sampled = tuple(1 if author_kind == "policy" else 0 for _ in generation)
    return InferenceCallV1(
        call_id=f"call-{index}",
        proxy_request_id=f"prid-{index}",
        rollout_id="attempt-1",
        group_id="group-a",
        sample_index=3,
        behavior_fingerprint=BINDING.fingerprint,
        policy_revision=BINDING.policy_revision,
        wire_api=BINDING.wire_api,
        sampling_transport=BINDING.sampling_transport,
        token_capture_provenance=provenance,
        prompt_token_ids=prompt,
        generation_token_ids=generation,
        generation_logprobs=(
            _logprobs(len(generation)) if logprobs is None else logprobs
        ),
        sampled_mask=sampled,
        finish_reason=finish_reason,
        stop_token_ids=PROFILE.stop_token_ids,
        renderer_profile_fingerprint=PROFILE.fingerprint,
        trainable=trainable,
        branch_id=branch_id,
        parent_branch_id=parent_branch_id,
        compaction=compaction,
        agent_instance_id=agent_instance_id,
        team_id="team-a",
        role_id=role_id,
        policy_type_id=policy_type_id,
        parameter_group_id="group-policy" if author_kind == "policy" else None,
        policy_set_revision_id="set-20",
        author_kind=author_kind,
        wire_request={"wire": "chat_completions", "messages": [{"role": "user"}], "turn": index},
        wire_response=wire_response
        or {"wire": "chat_completions", "choices": [{"message": {"role": "assistant"}}]},
        usage={"prompt_tokens": len(prompt), "completion_tokens": len(generation)},
        created_at=f"2026-09-02T12:00:{index:02d}.000000Z",
    )


def _builder(**kwargs: Any) -> CispoEvidenceBuilder:
    return CispoEvidenceBuilder(
        rollout_id="attempt-1",
        task_id="row-7",
        binding=BINDING,
        seed=11,
        group_id="group-a",
        sample_index=3,
        **kwargs,
    )


def _tool_loop() -> CispoEvidenceBuilder:
    """Two turns whose second prompt is a byte-for-byte prefix continuation."""

    builder = _builder()
    first = _call(0, prompt=SYSTEM + (2001, 2002), generation=(3001, 3002, 3003))
    builder.observe(first)
    builder.observe(
        _call(
            1,
            prompt=first.full_sequence + (4001, 4002),
            generation=(3101, 3102),
        )
    )
    return builder


# --------------------------------------------------------------------------- #


def test_a_tool_loop_stitches_into_one_segment() -> None:
    builder = _tool_loop()
    evidence = builder.seal()
    evidence.validate()

    assert len(evidence.episodes) == 1
    episode = evidence.episodes[0]
    assert len(episode.segments) == 1, "a tool loop stitches; it does not fork"
    segment = episode.segments[0]

    calls = builder.calls
    assert segment.token_ids == calls[-1].full_sequence
    assert segment.call_ids == ("call-0", "call-1")
    # Only tokens the policy sampled are trainable; prompt, tool observation and
    # environment step tokens are all zero.
    assert segment.trainable_tokens == 5
    expected = [0] * len(segment.token_ids)
    for call in calls:
        for offset in range(len(call.generation_token_ids)):
            expected[len(call.prompt_token_ids) + offset] = 1
    assert list(segment.loss_mask) == expected
    # Behavior logprobs sit under the tokens they scored and nowhere else.
    for position, flag in enumerate(segment.loss_mask):
        if not flag:
            assert segment.behavior_logprobs[position] == 0.0
        else:
            assert segment.behavior_logprobs[position] < 0.0


def test_a_compaction_forks_a_branch_and_masks_the_retained_prefix() -> None:
    builder = _tool_loop()
    calls = builder.calls
    # The harness rewrote history: a summary replaces the middle, and the tail of
    # the earlier generation is retained as context.
    retained_prefix = calls[-1].generation_token_ids[:1]
    forked = _call(
        2,
        prompt=SYSTEM + (5001, 5002) + retained_prefix,
        generation=(3201, 3202),
        branch_id="branch-2",
        parent_branch_id="root",
        compaction=CompactionProvenanceV1(
            rule="harness_deterministic_compaction",
            divergence_index=len(SYSTEM),
            removed_message_indices=(1, 2),
            authored_by_policy=False,
        ),
    )
    builder.observe(forked)
    evidence = builder.seal()
    evidence.validate()

    episode = evidence.episodes[0]
    assert len(episode.segments) == 2, "a compaction seals the prior segment and opens a branch"
    root, branch = episode.segments
    assert root.branch_id == "root"
    assert branch.branch_id == "branch-2"
    assert root.token_ids == calls[-1].full_sequence, "the prior segment is sealed as it stood"

    # Once the original sample is severed the retained tokens are context, not a
    # sample: they stay in the prompt and are entirely loss-masked.
    prompt_length = len(forked.prompt_token_ids)
    assert set(branch.loss_mask[:prompt_length]) == {0}
    assert branch.token_ids[prompt_length - 1] == retained_prefix[0]
    assert branch.trainable_tokens == len(forked.generation_token_ids)


def test_a_policy_sampled_compaction_is_still_the_policys_own_tokens() -> None:
    builder = _tool_loop()
    forked = _call(
        2,
        prompt=SYSTEM + (5001, 5002),
        generation=(3201, 3202),
        branch_id="branch-2",
        parent_branch_id="root",
        compaction=CompactionProvenanceV1(
            rule="policy_sampled_summary",
            divergence_index=len(SYSTEM),
            authored_by_policy=True,
        ),
    )
    builder.observe(forked)
    evidence = builder.seal()
    branch = evidence.episodes[0].segments[1]
    assert branch.trainable_tokens == 2
    assert builder.calls[-1].compaction is not None
    assert builder.calls[-1].compaction.authored_by_policy is True


def test_an_unexplained_divergence_is_an_evidence_failure() -> None:
    builder = _tool_loop()
    with pytest.raises(EvidenceError, match="no branch record and no declared compaction"):
        builder.observe(_call(2, prompt=SYSTEM + (9999,), generation=(3201,)))


def test_a_truncating_prompt_is_refused_rather_than_silently_stitched() -> None:
    first = _call(0, prompt=SYSTEM, generation=(3001, 3002))
    truncated = _call(1, prompt=SYSTEM[:2], generation=(3101,))
    with pytest.raises(EvidenceError, match="truncates"):
        assert_strict_prefix(first, truncated)


def test_a_compaction_that_keeps_its_branch_is_refused() -> None:
    first = _call(0, prompt=SYSTEM, generation=(3001, 3002))
    same_branch = _call(
        1,
        prompt=(7001, 7002),
        generation=(3101,),
        parent_branch_id="root",
        compaction=CompactionProvenanceV1(rule="rewrite", divergence_index=0),
    )
    with pytest.raises(EvidenceError, match="must open a new branch"):
        assert_strict_prefix(first, same_branch)


def test_a_judge_span_is_untrainable_with_its_author_declared() -> None:
    builder = _tool_loop()
    builder.observe(
        _call(
            9,
            prompt=(8001, 8002, 8003),
            generation=(8101, 8102),
            author_kind="judge",
            role_id="judge",
            policy_type_id="judge",
            trainable=False,
            provenance="wire_derived",
            agent_instance_id=None,
        )
    )
    evidence = builder.seal()
    evidence.validate()

    # The judge's tokens never enter an episode.
    assert all(
        "call-9" not in segment.call_ids
        for episode in evidence.episodes
        for segment in episode.segments
    )
    context = evidence.context_segments
    assert len(context) == 1
    judged = context[0]
    assert judged.author_kind == "judge", "authorship is declared, not implied by a zero mask"
    assert judged.trainable_tokens == 0
    assert judged.trainable is False
    assert judged.role_id == "judge"
    # The declaration survives the wire.
    assert judged.to_dict()["author_kind"] == "judge"


def test_an_opponents_spans_are_recorded_but_never_trainable() -> None:
    builder = _builder()
    builder.observe(_call(0, prompt=SYSTEM, generation=(3001, 3002)))
    builder.observe(
        _call(
            1,
            prompt=(6001, 6002),
            generation=(6101, 6102),
            author_kind="opponent",
            trainable=False,
            provenance="wire_derived",
            agent_instance_id="opponent-b",
        )
    )
    evidence = builder.seal()
    evidence.validate()
    assert [episode.agent_instance_id for episode in evidence.episodes] == ["instance-a"]
    assert [segment.author_kind for segment in evidence.context_segments] == ["opponent"]
    assert evidence.context_segments[0].agent_instance_id == "opponent-b"


def test_a_foreign_segment_may_not_carry_trainable_tokens() -> None:
    from synth_containers.cispo_evidence import TrainableSegmentV1

    with pytest.raises(ValueError, match="foreign authorship is never trainable"):
        TrainableSegmentV1(
            token_ids=(1, 2),
            loss_mask=(0, 1),
            behavior_logprobs=(0.0, -0.5),
            author_kind="opponent",
        )


# --------------------------------------------------------------------------- #
# Reading the existing Trace V5 capture
# --------------------------------------------------------------------------- #


def _model_call_span(
    *,
    provenance: TokenCaptureProvenance,
    extension: dict[str, Any] | None = None,
    token_capture: TokenCaptureV5 | None = None,
) -> SpanV5:
    capture = token_capture or TokenCaptureV5(
        provenance=provenance,
        level="token_ids",
        tokenizer=PROFILE.tokenizer_id,
        prompt=TokenSequenceRefV1(token_ids=SYSTEM + (2001,), count=5),
        completion=TokenSequenceRefV1(token_ids=(3001, 3002, 3003), count=3),
        completion_logprobs=_logprobs(3),
    )
    return SpanV5(
        span_id="span-0",
        span_kind=SpanKind.MODEL_CALL,
        actor_id="actor-0",
        session_id="session-0",
        started_at="2026-09-02T12:00:00.000000Z",
        detail={
            "wire_request": {"wire": "chat_completions", "messages": []},
            "wire_response": {"wire": "chat_completions", "choices": []},
        },
        usage=UsageV5(
            provenance=UsageProvenance.OBSERVED_PROVIDER, prompt_tokens=5, completion_tokens=3
        ),
        token_capture=capture,
        metadata={SPAN_EXTENSION_KEY: dict(extension or {})},
    )


def test_evidence_is_lifted_from_the_existing_trace_capture() -> None:
    span = _model_call_span(
        provenance=TokenCaptureProvenance.OBSERVED_PROVIDER,
        extension={
            "sampled_mask": [1, 1, 1],
            "finish_reason": "length_cap",
            "agent_instance_id": "instance-a",
            "team_id": "team-a",
        },
    )
    call = inference_call_from_span(
        span, binding=BINDING, rollout_id="attempt-1", group_id="group-a", sample_index=3
    )
    call.validate_for_training()
    assert call.prompt_token_ids == SYSTEM + (2001,)
    assert call.generation_token_ids == (3001, 3002, 3003)
    assert call.generation_logprobs == _logprobs(3)
    assert call.token_capture_provenance == "engine_meta"
    assert call.renderer_profile_fingerprint == PROFILE.fingerprint
    assert call.policy_revision == 17
    assert call.finish_reason == "length_cap"
    assert call.stop_token_ids == PROFILE.stop_token_ids
    # The wire objects are retained beside the tokens; neither substitutes for
    # the other.
    assert call.wire_request["wire"] == "chat_completions"
    assert call.wire_response["wire"] == "chat_completions"


def test_a_retokenized_capture_is_never_trainable() -> None:
    span = _model_call_span(provenance=TokenCaptureProvenance.DERIVED_RETOKENIZED)
    call = inference_call_from_span(span, binding=BINDING, rollout_id="attempt-1")
    assert call.token_capture_provenance == "wire_derived"
    with pytest.raises(EvidenceError, match="engine-level token capture"):
        call.validate_for_training()


def test_unavailable_token_capture_is_a_terminal_evidence_failure() -> None:
    span = _model_call_span(
        provenance=TokenCaptureProvenance.UNAVAILABLE,
        token_capture=TokenCaptureV5(
            provenance=TokenCaptureProvenance.UNAVAILABLE,
            level="none",
            unavailable_fields=("prompt_token_ids", "completion_token_ids", "logprobs"),
        ),
    )
    with pytest.raises(EvidenceError, match="terminal evidence failure"):
        inference_call_from_span(span, binding=BINDING, rollout_id="attempt-1")


def test_a_probe_binding_is_marked_non_trainable() -> None:
    span = _model_call_span(provenance=TokenCaptureProvenance.OBSERVED_PROVIDER)
    call = inference_call_from_span(
        span, binding=BINDING, rollout_id="attempt-1", probe=True
    )
    assert call.token_capture_provenance == "probe_synthetic"
    with pytest.raises(EvidenceError):
        call.validate_for_training()


# --------------------------------------------------------------------------- #
# Logprob validity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("logprobs", "message"),
    [
        ((-0.1, -0.2), "logprob length"),
        ((LOGPROB_SENTINEL, -0.2, -0.3), "provider sentinel"),
        ((0.0, 0.0, 0.0), "identically zero"),
        ((float("nan"), -0.2, -0.3), "not finite"),
    ],
)
def test_invalid_behavior_logprobs_are_refused(
    logprobs: tuple[float, ...], message: str
) -> None:
    call = _call(0, prompt=SYSTEM, generation=(3001, 3002, 3003), logprobs=logprobs)
    with pytest.raises(EvidenceError, match=message):
        call.validate_for_training()


def test_a_flattened_wire_is_refused() -> None:
    builder = _builder()
    with pytest.raises(EvidenceError, match="flattening one wire into the other"):
        builder.observe(
            _call(
                0,
                prompt=SYSTEM,
                generation=(3001,),
                wire_response={"wire": "responses", "output": []},
            )
        )


# --------------------------------------------------------------------------- #
# The seal
# --------------------------------------------------------------------------- #


def test_the_sealed_document_carries_the_digest_the_reward_binds_to() -> None:
    evidence = _tool_loop().seal(correlation={"run_id": "run-a"})
    assert evidence.trace_digest.startswith("sha256:")
    assert evidence.document.content_digest == evidence.trace_digest
    assert all(episode.trace_digest == evidence.trace_digest for episode in evidence.episodes)
    assert len(evidence.document.spans) == 2
    assert evidence.document.usage.completion_tokens == 5

    # Sealing the same evidence twice reaches the same digest.
    assert _tool_loop().seal(correlation={"run_id": "run-a"}).trace_digest == evidence.trace_digest

    adapter = CispoEvidenceAdapter(evidence)
    assert adapter.trace_digest == evidence.trace_digest
    assert adapter.trace_reference()["content_digest"] == evidence.trace_digest
    assert adapter.evidence_payload()["episodes"][0]["trace_digest"] == evidence.trace_digest


def test_an_attempt_with_no_model_call_seals_nothing() -> None:
    with pytest.raises(EvidenceError, match="terminal evidence failure"):
        _builder().seal()


def test_the_sealed_document_round_trips_back_into_calls() -> None:
    evidence = _tool_loop().seal()
    rebuilt = [
        inference_call_from_span(
            span, binding=BINDING, rollout_id="attempt-1", group_id="group-a", sample_index=3
        )
        for span in evidence.document.spans
    ]
    assert [call.to_dict() for call in rebuilt] == [
        call.to_dict() for call in evidence.calls
    ]


# --------------------------------------------------------------------------- #
# The optimizer's own validators
# --------------------------------------------------------------------------- #


def _optimizer_records() -> Any:
    """Import the training plane's record contract if it is reachable from here.

    ``synth-optimizers`` is not a dependency of this package, so the import is
    conditional. Setting ``SYNTH_OPTIMIZERS_SRC`` to its ``src`` directory makes
    the real validators run instead of the mirrored assertions.
    """

    root = os.environ.get("SYNTH_OPTIMIZERS_SRC")
    if root and root not in sys.path:
        sys.path.insert(0, root)
    try:
        from synth_optimizers.contracts import rl_records
    except ImportError:
        return None
    return rl_records


def test_evidence_passes_the_optimizer_validators() -> None:
    builder = _tool_loop()
    builder.observe(
        _call(
            9,
            prompt=(8001, 8002, 8003),
            generation=(8101, 8102),
            author_kind="judge",
            role_id="judge",
            policy_type_id="judge",
            trainable=False,
            provenance="wire_derived",
            agent_instance_id=None,
        )
    )
    evidence = builder.seal()
    payload = evidence.to_dict()
    records = _optimizer_records()

    if records is None:
        # synth-optimizers is not importable in this worktree, so the same
        # properties are asserted directly against the emitted payload.
        evidence.validate()
        for row in payload["calls"]:
            assert set(row) >= {
                "call_id",
                "prompt_token_ids",
                "generation_token_ids",
                "generation_logprobs",
                "sampled_mask",
                "renderer_profile_fingerprint",
                "policy_revision",
                "finish_reason",
                "stop_token_ids",
                "wire_request",
                "wire_response",
                "token_capture_provenance",
            }
            if row["trainable"]:
                assert row["token_capture_provenance"] == "engine_meta"
                assert len(row["generation_logprobs"]) == len(row["generation_token_ids"])
                assert LOGPROB_SENTINEL not in row["generation_logprobs"]
                assert any(value != 0.0 for value in row["generation_logprobs"])
        for episode_payload in payload["episodes"]:
            assert episode_payload["trace_digest"] == evidence.trace_digest
            for segment_payload in episode_payload["segments"]:
                assert segment_payload["author_kind"] == "policy"
                assert len(segment_payload["loss_mask"]) == len(segment_payload["token_ids"])
                assert len(segment_payload["behavior_logprobs"]) == len(
                    segment_payload["token_ids"]
                )
        for segment_payload in payload["context_segments"]:
            assert segment_payload["author_kind"] != "policy"
            assert not any(segment_payload["loss_mask"])
        return

    # The real contract, fed exactly what this container puts on the wire.
    calls = [_optimizer_call(records, row) for row in payload["calls"]]
    trainable = [call for call in calls if call.trainable]
    for call in trainable:
        call.validate_for_training()
    records.assert_strict_prefix(trainable[0], trainable[1])
    for episode_payload in payload["episodes"]:
        episode = records.TrainableEpisode(
            rollout_id=episode_payload["rollout_id"],
            task_id=episode_payload["task_id"],
            seed=episode_payload["seed"],
            policy_revision=episode_payload["policy_revision"],
            behavior_fingerprint=episode_payload["behavior_fingerprint"],
            segments=tuple(
                _optimizer_segment(records, item) for item in episode_payload["segments"]
            ),
            terminal_status=episode_payload["terminal_status"],
            trace_digest=episode_payload["trace_digest"],
        )
        episode.validate()
    for segment_payload in payload["context_segments"]:
        segment = _optimizer_segment(records, segment_payload)
        assert segment.author_kind in records.AUTHOR_KINDS
        assert segment.trainable is False


def _optimizer_call(records: Any, row: dict[str, Any]) -> Any:
    compaction = row.get("compaction")
    return records.InferenceCall(
        call_id=row["call_id"],
        proxy_request_id=row["proxy_request_id"],
        rollout_id=row["rollout_id"],
        group_id=row["group_id"],
        sample_index=row["sample_index"],
        behavior_fingerprint=row["behavior_fingerprint"],
        policy_revision=row["policy_revision"],
        wire_api=row["wire_api"],
        sampling_transport=row["sampling_transport"],
        token_capture_provenance=row["token_capture_provenance"],
        prompt_token_ids=tuple(row["prompt_token_ids"]),
        generation_token_ids=tuple(row["generation_token_ids"]),
        generation_logprobs=tuple(row["generation_logprobs"]),
        sampled_mask=tuple(row["sampled_mask"]),
        finish_reason=row["finish_reason"],
        stop_token_ids=tuple(row["stop_token_ids"]),
        content_mask=tuple(row["content_mask"]),
        renderer_profile_fingerprint=row["renderer_profile_fingerprint"],
        trainable=row["trainable"],
        branch_id=row["branch_id"],
        parent_branch_id=row["parent_branch_id"],
        compaction=(
            None
            if not compaction
            else records.CompactionProvenance(
                rule=compaction["rule"],
                divergence_index=compaction["divergence_index"],
                removed_message_indices=tuple(compaction["removed_message_indices"]),
                authored_by_policy=compaction["authored_by_policy"],
            )
        ),
        agent_instance_id=row["agent_instance_id"],
        team_id=row["team_id"],
        role_id=row["role_id"],
        policy_type_id=row["policy_type_id"],
        parameter_group_id=row["parameter_group_id"],
        policy_set_revision_id=row["policy_set_revision_id"],
        effect_tick_start=row["effect_tick_start"],
        effect_tick_end=row["effect_tick_end"],
        wire_request=row["wire_request"],
        wire_response=row["wire_response"],
        usage=row["usage"],
        created_at=row["created_at"],
    )


def _optimizer_segment(records: Any, row: dict[str, Any]) -> Any:
    return records.TrainableSegment(
        token_ids=tuple(row["token_ids"]),
        loss_mask=tuple(row["loss_mask"]),
        behavior_logprobs=tuple(row["behavior_logprobs"]),
        branch_id=row["branch_id"],
        parameter_group_id=row["parameter_group_id"],
        agent_instance_id=row["agent_instance_id"],
        call_ids=tuple(row["call_ids"]),
        author_kind=row["author_kind"],
        role_id=row["role_id"],
        policy_type_id=row["policy_type_id"],
        team_id=row["team_id"],
        policy_revision=row["policy_revision"],
        policy_set_revision_id=row["policy_set_revision_id"],
        effect_tick_start=row["effect_tick_start"],
        effect_tick_end=row["effect_tick_end"],
    )


def test_the_renderer_profile_fingerprint_is_the_training_planes_own_digest() -> None:
    records = _optimizer_records()
    if records is None:
        # Without the contract importable, pin the shape instead: a 32-character
        # hex digest that changes with every field that changes what a token
        # sequence means.
        assert len(PROFILE.fingerprint) == 32
        import dataclasses

        changed = dataclasses.replace(PROFILE, tokenizer_digest="sha256:other")
        assert changed.fingerprint != PROFILE.fingerprint
        return
    theirs = records.RendererProfile(
        profile_id=PROFILE.profile_id,
        package=PROFILE.package,
        package_version=PROFILE.package_version,
        config_digest=PROFILE.config_digest,
        tokenizer_id=PROFILE.tokenizer_id,
        tokenizer_digest=PROFILE.tokenizer_digest,
        stop_token_ids=PROFILE.stop_token_ids,
        modalities=PROFILE.modalities,
        add_generation_prompt=PROFILE.add_generation_prompt,
    )
    assert PROFILE.fingerprint == theirs.fingerprint
