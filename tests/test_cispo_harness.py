"""Handing a shipped harness a per-attempt sampler instead of a global one.

Two properties are worth more than the rest here and everything else supports
them: the bound origin reaches the harness as configuration, and the carried
credential reaches it as one entry of a child environment that was built rather
than inherited. The second is only safe because the process is per attempt, so
that is asserted directly -- two launches, two pids, neither able to see the
other's variable -- rather than assumed from the shape of the code.

Nothing here reaches a provider. The child builds a real shipped harness and is
asked what it resolved; it is never asked to sample.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from synth_containers.cispo_evidence import RendererProfileV1, SamplingProfileV1
from synth_containers.cispo_harness import (
    BOUND_CREDENTIAL_ENV,
    GLOBAL_CREDENTIAL_ENV_NAMES,
    BoundHarnessError,
    BoundHarnessLaunch,
    BoundHarnessProcess,
    CredentialLeak,
    assert_config_carries_no_credential,
    carried_credential,
    launch_bound_harness,
)
from synth_containers.cispo_policy import (
    SamplerBinding,
    SamplerCredential,
    SamplerOriginV1,
)
from synth_containers.cispo_rollout import BackgroundProcess

PROFILE = RendererProfileV1(
    profile_id="renderers.pinned.v1",
    package="renderers",
    package_version="0.1.0",
    config_digest="sha256:config",
    tokenizer_id="pinned/tokenizer",
    tokenizer_digest="sha256:tokenizer",
    stop_token_ids=(200002,),
)

SECRET = "session-credential-b7f1c2"


def origin(
    *,
    proxy_request_id: str = "prid-1",
    credential: str = SECRET,
    wire_api: str = "chat_completions",
    sampling_transport: str = "message_in_capture_out",
) -> SamplerOriginV1:
    return SamplerOriginV1(
        base_url=f"https://sampler.invalid/v1/sessions/{proxy_request_id}",
        credential=SamplerCredential(credential),
        policy_revision=4,
        behavior_fingerprint="bf-4",
        proxy_request_id=proxy_request_id,
        wire_api=wire_api,
        sampling_transport=sampling_transport,
    )


def binding(
    *,
    sampler_ready: bool = True,
    max_tokens: int | None = 96,
    **origin_kwargs: Any,
) -> SamplerBinding:
    bound_origin = origin(**origin_kwargs)
    return SamplerBinding(
        binding_id="pc_bound",
        policy_kind="trainable",
        handshake_id="hs-1",
        agreement_digest="sha256:agreement",
        renderer_profile=PROFILE,
        model_family="family",
        model_id="family/model",
        policy_revision=4,
        behavior_fingerprint="bf-4",
        wire_api=bound_origin.wire_api,
        sampling_transport=bound_origin.sampling_transport,
        sampler_ready=sampler_ready,
        agent_instance_id="instance-0",
        team_id="team-0",
        origin=bound_origin,
        sampling=SamplingProfileV1(max_tokens=max_tokens),
    )


# --------------------------------------------------------------------------- #
# The split: origin is configuration, credential is environment
# --------------------------------------------------------------------------- #


def test_the_config_carries_the_bound_origin_and_the_variables_name_only() -> None:
    launch = BoundHarnessLaunch(harness="single_call", binding=binding())
    config = launch.config()

    assert config["base_url"] == "https://sampler.invalid/v1/sessions/prid-1"
    assert config["api"] == "chat_completions"
    assert config["api_key_env"] == BOUND_CREDENTIAL_ENV
    assert config["model"] == "family/model"
    assert config["max_tokens"] == 96
    assert SECRET not in json.dumps(config)
    assert SECRET not in json.dumps(launch.launch_payload())


def test_a_config_that_carries_the_value_anywhere_inside_it_is_refused() -> None:
    """A config is written to disk, echoed into a trace, and logged."""

    with pytest.raises(CredentialLeak):
        assert_config_carries_no_credential(
            {"inference_target": {"headers": {"Authorization": f"Bearer {SECRET}"}}},
            SECRET,
        )
    with pytest.raises(CredentialLeak):
        BoundHarnessLaunch(
            harness="single_call",
            binding=binding(),
            extra_config={"objective": f"use {SECRET}"},
        ).config()


def test_the_environment_is_built_rather_than_inherited() -> None:
    launch = BoundHarnessLaunch(harness="single_call", binding=binding())
    environment = launch.environment()
    assert set(environment) == {
        "PATH",
        "PYTHONPATH",
        "PYTHONDONTWRITEBYTECODE",
        BOUND_CREDENTIAL_ENV,
    }
    assert environment[BOUND_CREDENTIAL_ENV] == SECRET


def test_the_carried_credential_has_exactly_one_way_out() -> None:
    bound_origin = origin()
    assert carried_credential(bound_origin) == SECRET
    # And every other path is still closed on the way in.
    assert SECRET not in repr(bound_origin.credential)
    assert SECRET not in json.dumps(bound_origin.to_payload())
    assert bound_origin.to_payload()["credential"] is None


# --------------------------------------------------------------------------- #
# Refusals, none of them a default
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(GLOBAL_CREDENTIAL_ENV_NAMES))
def test_a_process_wide_provider_variable_is_refused(name: str) -> None:
    with pytest.raises(BoundHarnessError, match="process-wide"):
        BoundHarnessLaunch(harness="single_call", binding=binding(), api_key_env=name)


def test_an_unreachable_binding_launches_nothing() -> None:
    with pytest.raises(BoundHarnessError, match="not reachable"):
        BoundHarnessLaunch(harness="single_call", binding=binding(sampler_ready=False))


def test_a_binding_with_no_sampler_origin_launches_nothing() -> None:
    pinned = SamplerBinding(
        binding_id="pc_pinned",
        policy_kind="pinned",
        handshake_id="hs-1",
        agreement_digest="sha256:agreement",
        renderer_profile=PROFILE,
        model_family="family",
        model_id="family/model",
        policy_revision=4,
        behavior_fingerprint="bf-4",
        wire_api="chat_completions",
        sampling_transport="message_in_capture_out",
        sampler_ready=True,
        trainable=False,
        pinned_identity="sha256:frozen-checkpoint",
    )
    with pytest.raises(BoundHarnessError, match="no sampler origin"):
        BoundHarnessLaunch(harness="single_call", binding=pinned)


def test_the_two_wire_shapes_are_not_interchangeable() -> None:
    """Presenting a Responses trajectory as a chat distribution is prohibited."""

    responses = binding(wire_api="responses")
    with pytest.raises(BoundHarnessError, match="two datasets"):
        BoundHarnessLaunch(harness="react", binding=responses)
    chat = binding()
    with pytest.raises(BoundHarnessError, match="two datasets"):
        BoundHarnessLaunch(harness="responses_react", binding=chat)
    # The pairing that does hold, holds.
    assert BoundHarnessLaunch(harness="responses_react", binding=responses).config()[
        "api"
    ] == "responses"


def test_a_token_transport_would_make_the_harness_a_second_renderer() -> None:
    with pytest.raises(BoundHarnessError, match="second renderer"):
        BoundHarnessLaunch(
            harness="single_call",
            binding=binding(sampling_transport="tokens_in_tokens_out"),
        )


def test_an_unknown_harness_refuses_rather_than_defaulting() -> None:
    with pytest.raises(BoundHarnessError, match="not a harness"):
        BoundHarnessLaunch(harness="nothing_like_that", binding=binding())


# --------------------------------------------------------------------------- #
# The per-attempt process
# --------------------------------------------------------------------------- #


def test_the_child_resolves_the_bound_origin_and_holds_the_credential() -> None:
    with launch_bound_harness("single_call", binding()) as process:
        described = process.describe()

    assert described["base_url"] == "https://sampler.invalid/v1/sessions/prid-1"
    assert described["api_key_env"] == BOUND_CREDENTIAL_ENV
    assert described["credential_present"] is True
    assert described["model"] == "family/model"
    assert described["metadata"]["harness"] == "single_call"
    # The value has no path out of the child at all.
    assert SECRET not in json.dumps(described)


def test_an_ambient_provider_key_does_not_cross_into_the_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point of building the environment instead of inheriting it."""

    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key-that-must-not-cross")
    monkeypatch.setenv("OPENROUTER_API_KEY", "ambient-key-that-must-not-cross")
    with launch_bound_harness("single_call", binding()) as process:
        names = set(process.describe()["environment_names"])
    assert "OPENAI_API_KEY" not in names
    assert "OPENROUTER_API_KEY" not in names
    assert BOUND_CREDENTIAL_ENV in names


def test_each_attempt_gets_its_own_process_and_sees_only_its_own_credential() -> None:
    """A shared process could not hold two attempts' credentials without racing."""

    first = BoundHarnessLaunch(
        harness="single_call",
        binding=binding(proxy_request_id="prid-a", credential="credential-a"),
        api_key_env="SYNTH_CISPO_SESSION_A",
    )
    second = BoundHarnessLaunch(
        harness="single_call",
        binding=binding(proxy_request_id="prid-b", credential="credential-b"),
        api_key_env="SYNTH_CISPO_SESSION_B",
    )
    with first.spawn() as one, second.spawn() as two:
        left, right = one.describe(), two.describe()
        assert one.pid != two.pid
        assert left["base_url"].endswith("prid-a")
        assert right["base_url"].endswith("prid-b")
        assert "SYNTH_CISPO_SESSION_B" not in left["environment_names"]
        assert "SYNTH_CISPO_SESSION_A" not in right["environment_names"]
        assert one.isolation_receipt["scope"] == "per_attempt"
        assert one.isolation_receipt["environment_inherited"] is False
        assert one.isolation_receipt["proxy_request_id"] == "prid-a"


def test_the_receipt_names_the_binding_and_never_the_value() -> None:
    with launch_bound_harness("single_call", binding()) as process:
        receipt = process.isolation_receipt
    assert receipt["binding_id"] == "pc_bound"
    assert receipt["policy_revision"] == 4
    assert receipt["credential_variable"] == BOUND_CREDENTIAL_ENV
    assert SECRET not in json.dumps(receipt)


def test_the_launch_is_a_background_process_an_attempt_can_adopt() -> None:
    """Adopting it is what makes the credential stop existing at the horizon."""

    process = launch_bound_harness("single_call", binding())
    try:
        assert isinstance(process, BackgroundProcess)
        assert isinstance(process, BoundHarnessProcess)
    finally:
        process.close()
    with pytest.raises(BoundHarnessError, match="dead"):
        process.describe()


def test_a_harness_failure_is_reported_and_never_swallowed() -> None:
    """The floor rung refuses an observation with no legal actions."""

    with launch_bound_harness("single_call", binding()) as process:
        with pytest.raises(BoundHarnessError, match="valid_actions"):
            process.plan({"observation_text": "nothing legal here"})


def test_the_launch_sandbox_is_removed_when_the_attempt_ends() -> None:
    process = launch_bound_harness("single_call", binding())
    sandbox = process._sandbox.name  # noqa: SLF001 - the receipt names no path
    assert os.path.isdir(sandbox)
    process.close()
    assert not os.path.exists(sandbox)
