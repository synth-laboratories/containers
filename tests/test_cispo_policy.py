"""The real sampler binding: the last thing between this container and a run.

The probe proves the evidence path without spending. This is the other half,
and every test here is a way a live run silently wastes provider spend or
trains on evidence nobody can attribute.

A revision that dispatches before its sampler is up burns a whole group against
a dead endpoint. A rollout whose behavior identity can move after admission
produces a group whose members are not comparable samples. A credential the
container reads is a credential the container can leak, log, or branch on. A
joint episode that starts half-bound produces per-instance trajectories with a
hole in them, and the hole is invisible by the time anyone reads the trace. And
a binding whose revision is not the one the executor pinned trains the wrong
parameters with a receipt that says otherwise.

The probe path is exercised here too, unchanged: it is how a run validates this
container before it spends, and a change that broke it would take the cheap
check away.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from synth_containers.cispo_handshake import (
    CispoHandshakeAdapter,
    HandshakeRegistry,
    PolicyFacts,
    RuntimeFacts,
)
from synth_containers.cispo_policy import (
    PINNED_POLICY_KIND,
    TRAINABLE_POLICY_KIND,
    AliasOpponentBinding,
    BehaviorIdentityImmutable,
    BoundPolicySession,
    CispoPolicyAdapter,
    EmbeddedCredential,
    GlobalSamplerOrigin,
    HalfBoundRoster,
    PolicyBindingError,
    PolicyBindingRegistry,
    RevisionPinMismatch,
    SamplerCredential,
    SamplerOriginV1,
    SamplerUnreachable,
    TransportUnsupported,
    UnknownPolicyKind,
    WireShapeMismatch,
    bind_policy_set,
    bind_sampler_policy,
)
from synth_containers.cispo_probe import PROBE_POLICY_KIND
from synth_containers.proxying import CredentialMode, InferenceApiFamily, ProxyMode

from tests.test_cispo_handshake import NOW, facts, optimizer, registry, request_payload

OPTIMIZER_PORTS = optimizer("synth_optimizers.rl.ports")

CREDENTIAL = "cred-live-9f3a-never-read-by-the-container"
REVISION = 17
BEHAVIOR = "sha256:behavior-rev-17"


# --------------------------------------------------------------------------- #
# Fixtures: an admitted agreement, a probe that answers, a transport that records
# --------------------------------------------------------------------------- #


def admitted(runtime: RuntimeFacts | None = None, **overrides: Any):
    """One admitted agreement, because a binding is gated exactly like an attempt."""

    runtime = runtime or facts()
    ledger: HandshakeRegistry = registry(runtime)
    verdict = ledger.handshake(request_payload(**overrides))
    assert verdict.accepted, verdict.rejected_mandatory_clauses
    return runtime, ledger, ledger.assert_admissible(
        verdict.handshake_id, verdict.agreement_digest
    )


class Reachable:
    """A probe whose answer a test controls. Nothing here opens a socket."""

    def __init__(self, ready: bool = True, *, unready: frozenset[str] = frozenset()) -> None:
        self.ready = ready
        self.unready = unready
        self.seen: list[str] = []

    def reachable(self, origin: SamplerOriginV1) -> bool:
        self.seen.append(origin.proxy_request_id)
        if origin.proxy_request_id in self.unready:
            return False
        return self.ready


class RecordingTransport:
    """Captures exactly what would have gone out, and answers a canned body."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    def post(self, url, *, headers, body):
        self.sent.append({"url": url, "headers": dict(headers), "body": dict(body)})
        return {"id": f"resp_{len(self.sent)}", "output_text": "ok"}


def origin_payload(
    *,
    proxy_request_id: str = "pr_attempt_0001",
    revision: int = REVISION,
    behavior: str = BEHAVIOR,
    wire_api: str = "chat_completions",
    transport: str = "message_in_capture_out",
    base_url: str | None = None,
    credential: str = CREDENTIAL,
) -> dict[str, Any]:
    """The versioned external sampler binding, exactly as the executor hands it over."""

    return {
        "base_url": base_url or f"https://sampler.internal/v1/attempts/{proxy_request_id}",
        "credential": credential,
        "policy_revision": revision,
        "behavior_fingerprint": behavior,
        "proxy_request_id": proxy_request_id,
        "wire_api": wire_api,
        "sampling_transport": transport,
        "expires_at": "2026-09-03T13:00:00Z",
    }


def binding_request(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": TRAINABLE_POLICY_KIND,
        "policy_revision": REVISION,
        "behavior_fingerprint": BEHAVIOR,
        "model_family": "policy_family",
        "model_id": "vendor/policy-20b",
        "sampler_origin": origin_payload(),
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------- #
# The origin is session-scoped, and it is the executor's own object
# --------------------------------------------------------------------------- #


def test_the_binding_carries_the_executors_own_origin_fields() -> None:
    """A container-shaped translation of the origin is a second place to disagree."""

    runtime, _, agreement = admitted()
    binding = bind_sampler_policy(
        runtime, agreement, binding_request(), reachability=Reachable(True)
    )
    assert binding.origin is not None
    assert binding.origin.base_url.endswith("pr_attempt_0001")
    assert binding.origin.policy_revision == REVISION
    assert binding.origin.behavior_fingerprint == BEHAVIOR
    assert binding.origin.wire_api == "chat_completions"
    assert binding.origin.sampling_transport == "message_in_capture_out"
    assert binding.origin.expires_at == "2026-09-03T13:00:00Z"
    if OPTIMIZER_PORTS is not None:
        declared = {
            name
            for name in OPTIMIZER_PORTS.SamplerOrigin.__dataclass_fields__
        }
        held = set(SamplerOriginV1.__dataclass_fields__)
        assert declared == held, (
            "the container's origin must be the executor's object field for field, "
            f"not a translation of it: {declared ^ held}"
        )


def test_an_origin_backed_probe_binding_is_dispatchable_but_not_trainable() -> None:
    """A joint probe exercises the roster without admitting its tokens to training."""

    runtime, _, agreement = admitted()
    binding = bind_sampler_policy(
        runtime,
        agreement,
        binding_request(
            kind=PROBE_POLICY_KIND,
            sampler_origin=origin_payload(base_url="probe://local"),
        ),
        reachability=Reachable(True),
    )
    assert binding.policy_kind == PROBE_POLICY_KIND
    assert binding.origin is not None
    assert binding.dispatchable is True
    assert binding.trainable is False


def test_the_reserved_probe_origin_cannot_bind_a_trainable_policy() -> None:
    runtime, _, agreement = admitted()
    with pytest.raises(GlobalSamplerOrigin, match="may only bind"):
        bind_sampler_policy(
            runtime,
            agreement,
            binding_request(sampler_origin=origin_payload(base_url="probe://local")),
            reachability=Reachable(True),
        )


def test_the_binding_hands_the_package_its_own_inference_target() -> None:
    """The existing policy machinery takes this object; no translation layer."""

    runtime, _, agreement = admitted()
    binding = bind_sampler_policy(
        runtime, agreement, binding_request(), reachability=Reachable(True)
    )
    target = binding.inference_target()
    assert target.normalized_api_family() is InferenceApiFamily.CHAT_COMPLETIONS
    assert target.normalized_proxy_mode() is ProxyMode.PROXY_ONLY
    assert target.normalized_credential_mode() is CredentialMode.PROXY
    assert target.inference_url.endswith("/chat/completions")
    assert target.model == "vendor/policy-20b"
    # The container does not know which provider is behind a scoped origin, and
    # naming one would be a guess a receipt then carries.
    assert target.provider == ""
    assert target.config["policy_revision"] == REVISION
    assert CREDENTIAL not in str(target.to_dict())


def test_a_global_origin_is_refused_because_a_leaked_credential_would_cross_rollouts() -> None:
    runtime, _, agreement = admitted()
    request = binding_request(
        sampler_origin=origin_payload(base_url="https://sampler.internal/v1")
    )
    with pytest.raises(GlobalSamplerOrigin):
        bind_sampler_policy(runtime, agreement, request, reachability=Reachable(True))


def test_the_origin_endpoint_uses_the_packages_own_wire_family() -> None:
    runtime, _, agreement = admitted()
    chat = bind_sampler_policy(
        runtime, agreement, binding_request(), reachability=Reachable(True)
    )
    assert chat.origin is not None
    assert chat.origin.api_family is InferenceApiFamily.CHAT_COMPLETIONS
    assert chat.origin.endpoint.endswith("/chat/completions")


# --------------------------------------------------------------------------- #
# Dispatchable only once the sampler is reachable
# --------------------------------------------------------------------------- #


def test_a_revision_is_dispatchable_only_once_its_sampler_is_reachable() -> None:
    runtime, _, agreement = admitted()
    probe = Reachable(False)
    ledger = PolicyBindingRegistry(reachability=probe)
    binding = ledger.register(
        bind_sampler_policy(runtime, agreement, binding_request(), reachability=probe)
    )
    assert binding.sampler_ready is False
    assert binding.dispatchable is False
    with pytest.raises(SamplerUnreachable):
        ledger.assert_dispatchable(binding.binding_id)
    with pytest.raises(SamplerUnreachable):
        ledger.session(binding.binding_id, transport=RecordingTransport())

    probe.ready = True
    refreshed = ledger.recheck(binding.binding_id)
    assert refreshed.dispatchable is True
    assert ledger.assert_dispatchable(binding.binding_id).binding_id == binding.binding_id
    assert probe.seen == ["pr_attempt_0001", "pr_attempt_0001"]


def test_a_build_with_no_reachability_probe_has_no_dispatchable_sampler() -> None:
    """Fail-closed: "probably up" is how a group burns its slots on a dead endpoint."""

    runtime, _, agreement = admitted()
    ledger = PolicyBindingRegistry()
    binding = ledger.register(bind_sampler_policy(runtime, agreement, binding_request()))
    assert binding.dispatchable is False
    with pytest.raises(SamplerUnreachable):
        ledger.assert_dispatchable(binding.binding_id)


# --------------------------------------------------------------------------- #
# Immutable after admission
# --------------------------------------------------------------------------- #


def test_the_behavior_identity_is_immutable_after_admission() -> None:
    runtime, _, agreement = admitted()
    probe = Reachable(True)
    ledger = PolicyBindingRegistry(reachability=probe)
    first = ledger.register(
        bind_sampler_policy(runtime, agreement, binding_request(), reachability=probe)
    )
    ledger.admit("ro_0001", first.binding_id)
    # Re-admitting the same binding is the idempotent retry, not a change.
    ledger.admit("ro_0001", first.binding_id)

    second = ledger.register(
        bind_sampler_policy(
            runtime,
            agreement,
            binding_request(
                policy_revision=18,
                behavior_fingerprint="sha256:behavior-rev-18",
                sampler_origin=origin_payload(
                    proxy_request_id="pr_attempt_0002",
                    revision=18,
                    behavior="sha256:behavior-rev-18",
                ),
            ),
            reachability=probe,
        )
    )
    assert second.binding_id != first.binding_id
    with pytest.raises(BehaviorIdentityImmutable):
        ledger.admit("ro_0001", second.binding_id)
    assert ledger.admitted_identity("ro_0001") == first.behavior_identity


def test_one_per_attempt_route_serves_exactly_one_revision() -> None:
    runtime, _, agreement = admitted()
    probe = Reachable(True)
    ledger = PolicyBindingRegistry(reachability=probe)
    ledger.register(
        bind_sampler_policy(runtime, agreement, binding_request(), reachability=probe)
    )
    rebound = bind_sampler_policy(
        runtime,
        agreement,
        binding_request(
            policy_revision=18,
            behavior_fingerprint="sha256:behavior-rev-18",
            sampler_origin=origin_payload(revision=18, behavior="sha256:behavior-rev-18"),
        ),
        reachability=probe,
    )
    with pytest.raises(BehaviorIdentityImmutable):
        ledger.register(rebound)


# --------------------------------------------------------------------------- #
# The credential is carried, never read
# --------------------------------------------------------------------------- #


def test_the_credential_never_reaches_a_payload_a_repr_or_a_record() -> None:
    runtime, _, agreement = admitted()
    probe = Reachable(True)
    ledger = PolicyBindingRegistry(reachability=probe)
    binding = ledger.register(
        bind_sampler_policy(runtime, agreement, binding_request(), reachability=probe)
    )
    transport = RecordingTransport()
    session = ledger.session(binding.binding_id, transport=transport)
    record, _response = session.call({"messages": [{"role": "user", "content": "hi"}]})

    surfaces = [
        repr(binding),
        str(binding),
        binding.to_json(),
        str(binding.to_payload()),
        str(binding.origin.to_payload()),
        repr(binding.origin.credential),
        f"{binding.origin.credential}",
        str(record.to_payload()),
        record.to_json(),
    ]
    for surface in surfaces:
        assert CREDENTIAL not in surface, surface[:200]
    assert binding.to_payload()["sampler_origin"]["credential"] is None
    assert binding.to_payload()["sampler_origin"]["credential_supplied"] is True

    # Carried, though: the one place it goes is the outbound header.
    assert transport.sent[0]["headers"]["Authorization"] == f"Bearer {CREDENTIAL}"
    assert CREDENTIAL not in str(transport.sent[0]["body"])


def test_the_binding_identity_does_not_depend_on_the_credential() -> None:
    """Never inspected: two bindings that differ only by credential are one binding."""

    runtime, _, agreement = admitted()
    probe = Reachable(True)
    first = bind_sampler_policy(
        runtime, agreement, binding_request(), reachability=probe
    )
    second = bind_sampler_policy(
        runtime,
        agreement,
        binding_request(sampler_origin=origin_payload(credential="an-entirely-different-key")),
        reachability=probe,
    )
    assert first.binding_id == second.binding_id
    assert first.behavior_identity == second.behavior_identity
    assert first.to_payload() == second.to_payload()


def test_a_raw_provider_credential_inline_is_refused_not_forwarded() -> None:
    runtime, _, agreement = admitted()
    for name in ("api_key", "bearer_token", "credential", "provider_api_key"):
        with pytest.raises(EmbeddedCredential):
            bind_sampler_policy(
                runtime,
                agreement,
                binding_request(**{name: "sk-should-never-be-inline"}),
                reachability=Reachable(True),
            )


def test_a_credential_may_not_be_compared_against_a_guess() -> None:
    held = SamplerCredential(CREDENTIAL)
    assert held != SamplerCredential(CREDENTIAL)
    assert held != CREDENTIAL
    assert held == held
    with pytest.raises(EmbeddedCredential):
        SamplerCredential("   ")


# --------------------------------------------------------------------------- #
# The pinned revision, on the binding and on every call
# --------------------------------------------------------------------------- #


def test_a_binding_whose_revision_does_not_match_the_pin_is_refused() -> None:
    runtime, _, agreement = admitted()
    with pytest.raises(RevisionPinMismatch):
        bind_sampler_policy(
            runtime,
            agreement,
            binding_request(sampler_origin=origin_payload(revision=18)),
            reachability=Reachable(True),
        )
    with pytest.raises(RevisionPinMismatch):
        bind_sampler_policy(
            runtime,
            agreement,
            binding_request(sampler_origin=origin_payload(behavior="sha256:other")),
            reachability=Reachable(True),
        )


def test_a_binding_that_names_no_revision_is_refused_rather_than_defaulted() -> None:
    runtime, _, agreement = admitted()
    request = binding_request()
    request.pop("policy_revision")
    with pytest.raises(RevisionPinMismatch):
        bind_sampler_policy(runtime, agreement, request, reachability=Reachable(True))


def test_every_model_call_records_the_pinned_revision() -> None:
    runtime, _, agreement = admitted()
    probe = Reachable(True)
    ledger = PolicyBindingRegistry(reachability=probe)
    binding = ledger.register(
        bind_sampler_policy(runtime, agreement, binding_request(), reachability=probe)
    )
    ledger.admit("ro_0007", binding.binding_id)
    transport = RecordingTransport()
    session = ledger.session(binding.binding_id, transport=transport)
    for turn in range(3):
        session.call({"messages": [{"role": "user", "content": f"turn {turn}"}]})

    assert len(session.calls) == 3
    assert [call.call_index for call in session.calls] == [0, 1, 2]
    for call in session.calls:
        assert call.policy_revision == REVISION
        assert call.behavior_fingerprint == BEHAVIOR
        assert call.behavior_identity == binding.behavior_identity
        assert call.proxy_request_id == "pr_attempt_0001"
        assert call.trainable is True
    # The revision rides in the record and in the credential, never as a header
    # the harness chose for itself.
    assert all("revision" not in key.lower() for key in transport.sent[0]["headers"])
    assert binding.behavior_binding().policy_revision == REVISION


def test_an_unreachable_sampler_cannot_be_driven_even_through_a_session() -> None:
    runtime, _, agreement = admitted()
    binding = bind_sampler_policy(
        runtime, agreement, binding_request(), reachability=Reachable(False)
    )
    session = BoundPolicySession(binding, transport=RecordingTransport())
    with pytest.raises(SamplerUnreachable):
        session.call({"messages": []})


# --------------------------------------------------------------------------- #
# Both declared wire shapes, neither flattened into the other
# --------------------------------------------------------------------------- #


def responses_runtime() -> RuntimeFacts:
    base = facts()
    return replace(base, policy=replace(base.policy, wire_api="responses"))


def test_both_declared_wire_shapes_are_served_in_their_own_shape() -> None:
    chat_runtime, _, chat_agreement = admitted()
    chat = bind_sampler_policy(
        chat_runtime, chat_agreement, binding_request(), reachability=Reachable(True)
    )
    chat_body = BoundPolicySession(chat).request(
        {"messages": [{"role": "user", "content": "hi"}], "temperature": 0}
    )
    assert set(chat_body) == {"model", "messages", "temperature"}

    runtime = responses_runtime()
    _, _, agreement = admitted(runtime, policy={"transport": "message_in_capture_out"})
    responses = bind_sampler_policy(
        runtime,
        agreement,
        binding_request(sampler_origin=origin_payload(wire_api="responses")),
        reachability=Reachable(True),
    )
    assert responses.origin is not None
    assert responses.origin.endpoint.endswith("/responses")
    responses_body = BoundPolicySession(responses).request(
        {"input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}]}
    )
    assert set(responses_body) == {"model", "input"}


def test_neither_wire_is_flattened_into_the_other() -> None:
    chat_runtime, _, chat_agreement = admitted()
    chat = BoundPolicySession(
        bind_sampler_policy(
            chat_runtime, chat_agreement, binding_request(), reachability=Reachable(True)
        )
    )
    with pytest.raises(WireShapeMismatch):
        chat.request({"input": [{"role": "user", "content": []}]})

    runtime = responses_runtime()
    _, _, agreement = admitted(runtime)
    responses = BoundPolicySession(
        bind_sampler_policy(
            runtime,
            agreement,
            binding_request(sampler_origin=origin_payload(wire_api="responses")),
            reachability=Reachable(True),
        )
    )
    with pytest.raises(WireShapeMismatch):
        responses.request({"messages": [{"role": "user", "content": "hi"}]})


def test_a_transport_the_container_never_declared_is_refused() -> None:
    base = facts()
    runtime = replace(
        base,
        policy=PolicyFacts(
            binding_transport="message_in_capture_out",
            wire_api="chat_completions",
            tokens_in_tokens_out=False,
            session_scoped_sampler_origin=True,
            embeds_credentials=False,
            revision_immutable_after_admission=True,
            records_policy_revision=True,
        ),
    )
    _, _, agreement = admitted(runtime)
    with pytest.raises(TransportUnsupported):
        bind_sampler_policy(
            runtime,
            agreement,
            binding_request(
                sampler_origin=origin_payload(transport="tokens_in_tokens_out")
            ),
            reachability=Reachable(True),
        )


def test_a_declared_token_transport_never_accepts_a_text_round_trip() -> None:
    """TiTo must not become the route by which a second renderer enters the run."""

    base = facts()
    runtime = replace(base, policy=replace(base.policy, tokens_in_tokens_out=True))
    _, _, agreement = admitted(runtime)
    binding = bind_sampler_policy(
        runtime,
        agreement,
        binding_request(sampler_origin=origin_payload(transport="tokens_in_tokens_out")),
        reachability=Reachable(True),
    )
    session = BoundPolicySession(binding)
    assert session.request({"prompt_token_ids": [1, 2, 3]})["prompt_token_ids"] == [1, 2, 3]
    with pytest.raises(WireShapeMismatch):
        session.request({"messages": [{"role": "user", "content": "hi"}]})
    with pytest.raises(WireShapeMismatch):
        session.request({"prompt_token_ids": [1], "messages": []})


# --------------------------------------------------------------------------- #
# The joint episode binds atomically or not at all
# --------------------------------------------------------------------------- #


def roster_request(**overrides: Any) -> dict[str, Any]:
    """One binding per declared instance: one trainable, one pinned opponent."""

    payload: dict[str, Any] = {
        "kind": TRAINABLE_POLICY_KIND,
        "policy_revision": REVISION,
        "behavior_fingerprint": BEHAVIOR,
        "model_family": "policy_family",
        "model_id": "vendor/policy-20b",
        "policy_set_revision_id": "policy-set-20",
        "bindings": [
            {
                "agent_instance_id": "a1",
                "sampler_origin": origin_payload(proxy_request_id="pr_attempt_a1"),
            },
            {
                "agent_instance_id": "a2",
                "pinned_identity": "sha256:frozen-ckpt",
                "sampler_origin": origin_payload(proxy_request_id="pr_attempt_a2"),
            },
        ],
    }
    payload.update(overrides)
    return payload


def test_a_policy_set_binds_every_instance_atomically() -> None:
    runtime, _, agreement = admitted()
    probe = Reachable(True)
    ledger = PolicyBindingRegistry(reachability=probe)
    policy_set = ledger.register_set(
        bind_policy_set(runtime, agreement, roster_request(), reachability=probe)
    )
    payload = policy_set.to_payload()
    assert payload["atomic"] is True
    assert payload["policy_set_revision_id"] == "policy-set-20"
    assert payload["policy_set_revision"]
    assert payload["topology_id"] == "topo-1"
    rows = {row["agent_instance_id"]: row for row in payload["bindings"]}
    assert set(rows) == {"a1", "a2"}

    # The trainee is trainable and lands in its declared parameter group; the
    # pinned opponent is neither, and its identity is still recorded.
    assert rows["a1"]["trainable"] is True
    assert rows["a1"]["parameter_group_id"] == "pg-1"
    assert rows["a1"]["kind"] == TRAINABLE_POLICY_KIND
    assert rows["a2"]["trainable"] is False
    assert rows["a2"]["parameter_group_id"] is None
    assert rows["a2"]["kind"] == PINNED_POLICY_KIND
    assert rows["a2"]["pinned_identity"] == "sha256:frozen-ckpt"
    assert policy_set.dispatchable is True
    assert ledger.assert_roster_dispatchable(policy_set.policy_set_id) is policy_set


def test_a_half_bound_roster_is_refused_and_registers_nothing() -> None:
    runtime, _, agreement = admitted()
    probe = Reachable(True)
    ledger = PolicyBindingRegistry(reachability=probe)
    partial = roster_request()
    partial["bindings"] = partial["bindings"][:1]
    with pytest.raises(HalfBoundRoster) as caught:
        bind_policy_set(runtime, agreement, partial, reachability=probe)
    assert "'a2'" in str(caught.value)
    assert ledger.binding_ids == ()
    assert ledger.policy_set_ids == ()

    unknown = roster_request()
    unknown["bindings"] = [*unknown["bindings"], {"agent_instance_id": "a3"}]
    with pytest.raises(HalfBoundRoster):
        bind_policy_set(runtime, agreement, unknown, reachability=probe)
    assert ledger.binding_ids == ()


def test_one_unreachable_instance_refuses_the_whole_roster() -> None:
    """No attempt may start against a roster that is only mostly up."""

    runtime, _, agreement = admitted()
    probe = Reachable(True, unready=frozenset({"pr_attempt_a2"}))
    ledger = PolicyBindingRegistry(reachability=probe)
    with pytest.raises(SamplerUnreachable) as caught:
        bind_policy_set(runtime, agreement, roster_request(), reachability=probe)
    assert "a2" in str(caught.value)
    assert ledger.binding_ids == ()
    assert ledger.policy_set_ids == ()


def test_a_pinned_opponent_may_not_resolve_a_moving_alias() -> None:
    runtime, _, agreement = admitted()
    request = roster_request()
    request["bindings"][1]["pinned_identity"] = "latest"
    with pytest.raises(AliasOpponentBinding):
        bind_policy_set(runtime, agreement, request, reachability=Reachable(True))


def test_a_scripted_opponent_binds_with_no_sampler_and_no_trainable_evidence() -> None:
    runtime, _, agreement = admitted()
    request = roster_request()
    request["bindings"][1] = {
        "agent_instance_id": "a2",
        "pinned_identity": "script:pinned-baseline-v3",
    }
    policy_set = bind_policy_set(runtime, agreement, request, reachability=Reachable(True))
    rows = {row["agent_instance_id"]: row for row in policy_set.to_payload()["bindings"]}
    assert rows["a2"]["sampler_origin"] is None
    assert rows["a2"]["trainable"] is False
    assert rows["a2"]["pinned_identity"] == "script:pinned-baseline-v3"
    assert policy_set.dispatchable is True


def test_a_trainable_instance_with_no_origin_is_refused() -> None:
    runtime, _, agreement = admitted()
    request = roster_request()
    request["bindings"][0] = {"agent_instance_id": "a1", "pinned_identity": "sha256:x"}
    with pytest.raises(PolicyBindingError) as caught:
        bind_policy_set(runtime, agreement, request, reachability=Reachable(True))
    assert "sampler origin" in str(caught.value)


def test_a_joint_rollout_pins_the_whole_roster_at_admission() -> None:
    runtime, _, agreement = admitted()
    probe = Reachable(True)
    ledger = PolicyBindingRegistry(reachability=probe)
    first = ledger.register_set(
        bind_policy_set(runtime, agreement, roster_request(), reachability=probe)
    )
    ledger.admit_set("ro_joint", first.policy_set_id)
    moved = roster_request()
    moved["policy_revision"] = 18
    moved["behavior_fingerprint"] = "sha256:behavior-rev-18"
    for row in moved["bindings"]:
        row["sampler_origin"] = origin_payload(
            proxy_request_id=f"{row['agent_instance_id']}_rev18",
            revision=18,
            behavior="sha256:behavior-rev-18",
        )
    second = ledger.register_set(
        bind_policy_set(runtime, agreement, moved, reachability=probe)
    )
    with pytest.raises(BehaviorIdentityImmutable):
        ledger.admit_set("ro_joint", second.policy_set_id)


# --------------------------------------------------------------------------- #
# Dispatch through the admission adapter, with the probe path intact
# --------------------------------------------------------------------------- #


def gate(runtime: RuntimeFacts, probe: Reachable) -> tuple[CispoHandshakeAdapter, dict[str, Any]]:
    adapter = CispoHandshakeAdapter(runtime, clock=lambda: NOW)
    adapter.policies = CispoPolicyAdapter(runtime, reachability=probe)
    return adapter, adapter.cispo_handshake(request_payload())


def test_the_adapter_binds_a_real_sampler_where_it_used_to_refuse() -> None:
    runtime = facts()
    probe = Reachable(True)
    adapter, verdict = gate(runtime, probe)
    payload = adapter.cispo_bind_policy(
        {
            **binding_request(),
            "handshake_id": verdict["handshake_id"],
            "agreement_digest": verdict["agreement_digest"],
        }
    )
    assert payload["config_id"].startswith("pc_")
    assert payload["probe"] is False
    assert payload["sampler_ready"] is True
    assert payload["immutable"] is True
    assert payload["policy_revision"] == REVISION
    assert payload["sampler_origin"]["credential"] is None
    assert payload["renderer_fingerprint"] == runtime.renderer_profile.fingerprint
    assert CREDENTIAL not in str(payload)


def test_the_adapter_binds_a_whole_roster_atomically() -> None:
    runtime = facts()
    probe = Reachable(True)
    adapter, verdict = gate(runtime, probe)
    payload = adapter.cispo_bind_policy_set(
        {
            **roster_request(),
            "handshake_id": verdict["handshake_id"],
            "agreement_digest": verdict["agreement_digest"],
        }
    )
    assert payload["atomic"] is True
    assert {row["agent_instance_id"] for row in payload["bindings"]} == {"a1", "a2"}
    assert adapter.policies.registry.policy_set_ids == (payload["policy_set_id"],)


def test_a_binding_is_gated_by_the_handshake_exactly_as_an_attempt_is() -> None:
    from synth_containers.cispo_handshake import HandshakeUnknown

    runtime = facts()
    adapter, verdict = gate(runtime, Reachable(True))
    with pytest.raises(HandshakeUnknown):
        adapter.cispo_bind_policy(
            {
                **binding_request(),
                "handshake_id": "hs_never_issued",
                "agreement_digest": verdict["agreement_digest"],
            }
        )


def test_a_build_that_embeds_credentials_may_not_bind_a_sampler() -> None:
    """Binding anyway would promise a property the handshake already refused."""

    base, _, agreement = admitted()
    degraded = replace(base, policy=replace(base.policy, embeds_credentials=True))
    with pytest.raises(PolicyBindingError) as caught:
        CispoPolicyAdapter(degraded, reachability=Reachable(True)).bind(
            agreement, binding_request()
        )
    assert "session-scoped" in str(caught.value)

    global_origin = replace(
        base, policy=replace(base.policy, session_scoped_sampler_origin=False)
    )
    with pytest.raises(PolicyBindingError):
        CispoPolicyAdapter(global_origin, reachability=Reachable(True)).bind(
            agreement, binding_request()
        )


def test_an_unknown_kind_is_still_a_typed_refusal() -> None:
    runtime, _, agreement = admitted()
    with pytest.raises(UnknownPolicyKind):
        bind_sampler_policy(
            runtime,
            agreement,
            binding_request(kind="something_nobody_declared"),
            reachability=Reachable(True),
        )


def test_the_probe_path_still_binds_exactly_as_it_did() -> None:
    """The unpaid check is how a run validates this container without spending."""

    runtime = facts()
    adapter, verdict = gate(runtime, Reachable(True))
    payload = adapter.cispo_bind_policy(
        {
            "kind": PROBE_POLICY_KIND,
            "handshake_id": verdict["handshake_id"],
            "agreement_digest": verdict["agreement_digest"],
            "model_family": "policy_family",
            "model_id": "vendor/policy-20b",
        }
    )
    assert payload["policy_kind"] == PROBE_POLICY_KIND
    assert payload["probe"] is True
    assert payload["trainable"] is False
    assert payload["credential"] is None
    assert payload["sampler_origin"] is None
    assert payload["provider_requests_expected"] == 0
    # And it did not construct, or leak into, the sampler registry.
    assert adapter.policies.registry.binding_ids == ()


def test_a_probe_policy_set_still_takes_the_probe_path() -> None:
    runtime = facts()
    adapter, verdict = gate(runtime, Reachable(True))
    payload = adapter.cispo_bind_policy_set(
        {
            "kind": PROBE_POLICY_KIND,
            "handshake_id": verdict["handshake_id"],
            "agreement_digest": verdict["agreement_digest"],
            "bindings": [
                {"kind": PROBE_POLICY_KIND, "model_family": "f", "model_id": "m"},
                {"kind": PROBE_POLICY_KIND, "model_family": "f", "model_id": "m"},
            ],
        }
    )
    assert len(payload["bindings"]) == 2
    assert all(row["probe"] is True for row in payload["bindings"])
    assert adapter.policies.registry.binding_ids == ()
