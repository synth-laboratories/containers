"""Container-side CISPO policy binding: the real sampler, not the probe.

The probe kind proves the evidence path without spending. This module is the
other half: the binding a live attempt actually samples through. It accepts a
*versioned external sampler binding* -- the same object the executor's
``SamplerOrigin`` carries, field for field -- and turns it into something the
container's existing policy machinery can drive.

Five rules shape it, and each one is a way a run trains on invalid evidence
when it is missing.

1. **The origin is session-scoped, never global.** The bound base URL carries
   the per-attempt request identity in its own path, so stitching is a URL
   parse and a leaked credential cannot cross rollouts. An origin whose path
   does not contain its ``proxy_request_id`` is refused rather than accepted
   as "close enough": a global origin is exactly the shape that lets one
   attempt's credential sample another attempt's revision.
2. **The credential is carried, never read.** It arrives inside the versioned
   origin, is wrapped in :class:`SamplerCredential` on the way in, and leaves
   this container only as an ``Authorization`` header on the outbound sampler
   request. Nothing here parses it, digests it, logs it, or lets it into a
   payload -- and the binding's own identity is computed without it, so two
   otherwise identical bindings under different credentials are the same
   binding. The credential names the group, sample, wire, policy kind, and
   pinned revision; this side never sees those fields and never chooses them.
   Raw provider credentials at the top level of a binding request are a
   refusal, not a field this module quietly forwards.
3. **A revision is dispatchable only once its sampler is reachable.** Binding
   records readiness by probing; dispatch asserts it. The default probe is
   fail-closed -- a build that supplies none has no reachable sampler, because
   "probably up" is how a group burns its slots against a dead endpoint.
4. **A rollout's behavior-policy identity is immutable after admission.**
   Admission pins revision, behavior fingerprint, model, wire, transport, and
   renderer together; a second admission of the same rollout under any other
   identity raises. Rebinding one origin route to a second revision raises for
   the same reason.
5. **A joint episode binds atomically or not at all.** Every declared instance
   binds -- trainable and pinned-opponent alike -- or nothing is registered.
   A half-bound roster is refused before an attempt exists, because an episode
   that starts with a missing instance produces evidence nobody can attribute.

The revision is recorded on every model call, in the call record rather than on
the wire: the pinned revision already rides inside the credential, and a
harness that stamped its own revision header would be choosing a field it is
not allowed to see.

Both declared wire shapes are served. ``chat_completions`` posts ``messages``
to ``{base}/chat/completions`` and ``responses`` posts ``input`` items to
``{base}/responses`` -- the two suffixes
:class:`~synth_containers.proxying.InferenceApiFamily` already resolves, and
the two request shapes this package's existing harnesses already build. The
shapes are not interchangeable and are not flattened into each other: a
``responses`` binding handed ``messages`` is a refusal.

No task, harness, or environment name appears anywhere in this module.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

from .cispo_evidence import BehaviorBindingV1, RendererProfileV1, SamplingProfileV1
from .cispo_handshake import (
    AdmittedHandshake,
    AgentInstanceFacts,
    RuntimeFacts,
    canonical_digest,
)
from .proxying import CredentialMode, InferenceApiFamily, InferenceTarget, ProxyMode
from .serde import JsonDataclassMixin

POLICY_BINDING_SCHEMA_VERSION = "cispo.policy_binding.v1"
POLICY_SET_SCHEMA_VERSION = "cispo.policy_set_binding.v1"
MODEL_CALL_SCHEMA_VERSION = "cispo.policy_model_call.v1"

#: The kinds this module binds. ``probe`` is deliberately absent: it reaches no
#: provider and belongs to the module that mints canned generations.
TRAINABLE_POLICY_KIND = "trainable"
PINNED_POLICY_KIND = "pinned"
SAMPLER_POLICY_KINDS: frozenset[str] = frozenset({TRAINABLE_POLICY_KIND, PINNED_POLICY_KIND})

WIRE_APIS: frozenset[str] = frozenset({"chat_completions", "responses"})
SAMPLING_TRANSPORTS: frozenset[str] = frozenset(
    {"message_in_capture_out", "tokens_in_tokens_out"}
)

#: A moving name is not a pinned identity. An opponent bound to one of these
#: would silently change between two episodes that must be comparable samples.
ALIAS_REFS: frozenset[str] = frozenset({"latest", "head", "stable", "current", "main"})

#: Field names that would carry a raw provider credential in the clear. They are
#: refused at the top level of a binding request rather than forwarded: the only
#: credential this container accepts is the session-scoped one inside the
#: versioned origin, and it never belongs in a rollout request at all.
FORBIDDEN_CREDENTIAL_FIELDS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "authorization",
    "bearer_token",
    "credential",
    "provider_api_key",
    "secret",
    "secret_key",
    "token",
)

#: What a binding is pinned by. Two bindings that agree on every one of these
#: are the same behavior policy; disagreeing on any is a different dataset.
BEHAVIOR_IDENTITY_FIELDS: tuple[str, ...] = (
    "policy_revision",
    "behavior_fingerprint",
    "model_family",
    "model_id",
    "wire_api",
    "sampling_transport",
    "renderer_fingerprint",
)


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


class PolicyBindingError(ValueError):
    """A sampler binding was refused. Never degrade one of these to a reward."""

    status_code = 409
    error = "policy_binding_error"

    def payload(self) -> dict[str, Any]:
        return {"error": self.error, "reason": str(self), "status_code": self.status_code}


class UnknownPolicyKind(PolicyBindingError):
    """The request named a binding kind this module does not serve."""

    error = "unknown_policy_kind"


class TransportUnsupported(PolicyBindingError):
    """The requested sampling transport is not one this build declared."""

    error = "transport_unsupported"


class WireShapeMismatch(PolicyBindingError):
    """The request body is shaped for the other wire. The two are not one dataset."""

    error = "wire_shape_mismatch"
    status_code = 400


class EmbeddedCredential(PolicyBindingError):
    """A raw provider credential was inline. Bindings carry a scoped one instead."""

    error = "embedded_credential"
    status_code = 400


class GlobalSamplerOrigin(PolicyBindingError):
    """The origin is not session-scoped: a leaked credential would cross rollouts."""

    error = "global_sampler_origin"
    status_code = 400


class SamplerUnreachable(PolicyBindingError):
    """The bound revision's sampler is not reachable, so it is not dispatchable."""

    error = "sampler_not_ready"


class RevisionPinMismatch(PolicyBindingError):
    """The origin's revision is not the one the request pinned."""

    error = "policy_revision_mismatch"


class BehaviorIdentityImmutable(PolicyBindingError):
    """A rollout's behavior-policy identity may not change after admission."""

    error = "behavior_identity_immutable"


class HalfBoundRoster(PolicyBindingError):
    """A joint episode binds every declared instance, or none of them."""

    error = "partial_roster_binding"


class AliasOpponentBinding(PolicyBindingError):
    """A pinned opponent may not resolve a moving alias."""

    error = "alias_opponent_binding"


class UnboundPolicy(PolicyBindingError):
    """An attempt named a binding this container never issued."""

    error = "policy_not_bound"
    status_code = 404


# --------------------------------------------------------------------------- #
# The credential this container carries and never reads
# --------------------------------------------------------------------------- #


class SamplerCredential:
    """A bearer the container hands to the sampler and never inspects.

    Not a dataclass on purpose: every path that could leak the value -- repr,
    str, format, equality against a plaintext guess, a dataclass ``asdict``
    walk -- is closed here rather than avoided by convention. The only way the
    value leaves is :meth:`authorization`, which is what the outbound request
    needs and nothing else does.
    """

    __slots__ = ("_value",)

    def __init__(self, value: Any) -> None:
        text = str(value or "")
        if not text.strip():
            raise EmbeddedCredential(
                "a sampler binding names no credential; the container carries one, "
                "it does not mint one"
            )
        object.__setattr__(self, "_value", text)

    def __repr__(self) -> str:
        return "SamplerCredential(<redacted>)"

    __str__ = __repr__

    def __format__(self, spec: str) -> str:
        return repr(self)

    def __eq__(self, other: Any) -> bool:
        # Two credentials are never "equal" to this side: comparing one against
        # a candidate is an inspection, and an oracle for guessing it.
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def __getstate__(self) -> dict[str, str]:
        # A dataclass ``asdict`` walk, a deepcopy, and a pickle all go through
        # here, so none of them is a second way to read the value.
        return {"_value": "<redacted>"}

    def __setstate__(self, state: Mapping[str, str]) -> None:
        object.__setattr__(self, "_value", "<redacted>")

    def authorization(self) -> dict[str, str]:
        """The one outbound header. The value is not returned any other way."""

        return {"Authorization": f"Bearer {self._value}"}


# --------------------------------------------------------------------------- #
# The versioned external sampler binding
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SamplerOriginV1(JsonDataclassMixin):
    """Where a bound policy is reachable, for one attempt only.

    Field for field the executor's ``SamplerOrigin``: this is the object the
    executor hands over, not a container-shaped translation of it.
    """

    base_url: str
    credential: SamplerCredential
    policy_revision: int
    behavior_fingerprint: str
    proxy_request_id: str
    wire_api: str
    sampling_transport: str
    expires_at: str = ""

    def __post_init__(self) -> None:
        parts = urlsplit(self.base_url)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise GlobalSamplerOrigin(
                f"sampler origin {self.base_url!r} is not an absolute http(s) URL"
            )
        if not str(self.proxy_request_id or "").strip():
            raise GlobalSamplerOrigin("sampler origin names no per-attempt request id")
        segments = tuple(item for item in parts.path.split("/") if item)
        if self.proxy_request_id not in segments:
            raise GlobalSamplerOrigin(
                f"sampler origin path {parts.path!r} does not carry the per-attempt id "
                f"{self.proxy_request_id!r}; a global origin lets one attempt's "
                "credential sample another attempt's revision"
            )
        if not isinstance(self.credential, SamplerCredential):
            raise EmbeddedCredential(
                "a sampler origin's credential must be wrapped before it is bound"
            )
        if self.policy_revision < 0:
            raise RevisionPinMismatch("policy_revision must be non-negative")
        if not str(self.behavior_fingerprint or "").strip():
            raise RevisionPinMismatch("sampler origin names no behavior fingerprint")
        if self.wire_api not in WIRE_APIS:
            raise WireShapeMismatch(f"unknown wire_api {self.wire_api!r}")
        if self.sampling_transport not in SAMPLING_TRANSPORTS:
            raise TransportUnsupported(
                f"unknown sampling_transport {self.sampling_transport!r}"
            )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SamplerOriginV1":
        if not isinstance(payload, Mapping):
            raise GlobalSamplerOrigin("sampler_origin must be an object")
        if "policy_revision" not in payload:
            raise RevisionPinMismatch("sampler origin names no policy_revision")
        return cls(
            base_url=str(payload.get("base_url") or ""),
            credential=SamplerCredential(payload.get("credential")),
            policy_revision=int(payload["policy_revision"]),
            behavior_fingerprint=str(payload.get("behavior_fingerprint") or ""),
            proxy_request_id=str(payload.get("proxy_request_id") or ""),
            wire_api=str(payload.get("wire_api") or ""),
            sampling_transport=str(payload.get("sampling_transport") or ""),
            expires_at=str(payload.get("expires_at") or ""),
        )

    @property
    def api_family(self) -> InferenceApiFamily:
        """The wire, resolved by the package's own family enum rather than a table."""

        return InferenceApiFamily.parse(self.wire_api)

    @property
    def endpoint(self) -> str:
        """The absolute URL one sampling call posts to."""

        return f"{self.base_url.rstrip('/')}/{self.api_family.endpoint_suffix}"

    def headers(self) -> dict[str, str]:
        """Outbound headers. The credential passes through; nothing reads it."""

        return {
            "Content-Type": "application/json",
            "X-Proxy-Request-Id": self.proxy_request_id,
            **self.credential.authorization(),
        }

    def to_payload(self) -> dict[str, Any]:
        """The origin as it goes back on the wire, with no credential in it."""

        return {
            "base_url": self.base_url,
            "endpoint": self.endpoint,
            "proxy_request_id": self.proxy_request_id,
            "policy_revision": int(self.policy_revision),
            "behavior_fingerprint": self.behavior_fingerprint,
            "wire_api": self.wire_api,
            "sampling_transport": self.sampling_transport,
            "expires_at": self.expires_at,
            "credential": None,
            "credential_supplied": True,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.to_payload()


# --------------------------------------------------------------------------- #
# Reachability and transport, both injected, neither guessed
# --------------------------------------------------------------------------- #


@runtime_checkable
class SamplerReachability(Protocol):
    """Answers whether one bound origin is actually up."""

    def reachable(self, origin: SamplerOriginV1) -> bool: ...


class UnprobedSampler:
    """The fail-closed default: no probe supplied means nothing is reachable.

    A build that cannot check is not a build whose samplers are up. Defaulting
    the other way spends a whole group against a dead endpoint before anyone
    finds out.
    """

    def reachable(self, origin: SamplerOriginV1) -> bool:
        return False


@runtime_checkable
class SamplerTransport(Protocol):
    """One outbound sampling call. Supplied by the container, never by this module."""

    def post(
        self, url: str, *, headers: Mapping[str, str], body: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


# --------------------------------------------------------------------------- #
# The binding
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SamplerBinding(JsonDataclassMixin):
    """One instance's bound policy, pinned to one admitted agreement."""

    binding_id: str
    policy_kind: str
    handshake_id: str
    agreement_digest: str
    renderer_profile: RendererProfileV1
    model_family: str
    model_id: str
    policy_revision: int
    behavior_fingerprint: str
    wire_api: str
    sampling_transport: str
    sampler_ready: bool
    trainable: bool = True
    agent_instance_id: str | None = None
    team_id: str | None = None
    parameter_group_id: str | None = None
    pinned_identity: str | None = None
    origin: SamplerOriginV1 | None = None
    sampling: SamplingProfileV1 = field(default_factory=SamplingProfileV1)

    def __post_init__(self) -> None:
        if self.policy_kind not in SAMPLER_POLICY_KINDS:
            raise UnknownPolicyKind(f"{self.policy_kind!r} is not a sampler binding kind")
        if self.origin is None and not str(self.pinned_identity or "").strip():
            raise PolicyBindingError(
                "a binding with no sampler origin must name the immutable identity "
                "that drives it"
            )
        if self.origin is None and self.trainable:
            raise PolicyBindingError(
                "a trainable instance must bind a sampler origin; there is nothing "
                "to collect behavior logprobs from otherwise"
            )

    @property
    def dispatchable(self) -> bool:
        """Only a reachable sampler may be dispatched against."""

        return bool(self.sampler_ready)

    @property
    def behavior_identity(self) -> str:
        """The pin two bindings must agree on to be the same behavior policy."""

        return canonical_digest(
            {
                "policy_revision": int(self.policy_revision),
                "behavior_fingerprint": self.behavior_fingerprint,
                "model_family": self.model_family,
                "model_id": self.model_id,
                "wire_api": self.wire_api,
                "sampling_transport": self.sampling_transport,
                "renderer_fingerprint": self.renderer_profile.fingerprint,
            },
            length=32,
        )

    def inference_target(self) -> InferenceTarget:
        """The binding as the package's own policy machinery already accepts it.

        ``provider`` is deliberately empty: the container does not know and does
        not choose which provider is behind a session-scoped origin, and naming
        one here would be a guess a receipt would then carry. The credential is
        not in this object either -- the mode says it is proxy-carried, and the
        session is what actually carries it.
        """

        origin = self.origin
        if origin is None:
            raise UnboundPolicy(
                f"binding {self.binding_id!r} drives a pinned identity with no sampler"
            )
        return InferenceTarget(
            provider="",
            model=self.model_id,
            api_family=origin.api_family,
            inference_url=origin.endpoint,
            base_url=origin.base_url,
            proxy_mode=ProxyMode.PROXY_ONLY,
            credential_mode=CredentialMode.PROXY,
            max_tokens=self.sampling.max_tokens,
            config={
                "policy_revision": int(self.policy_revision),
                "behavior_fingerprint": self.behavior_fingerprint,
                "proxy_request_id": origin.proxy_request_id,
                "sampling_transport": self.sampling_transport,
            },
        )

    def behavior_binding(self) -> BehaviorBindingV1:
        """The evidence-side record of what these tokens were produced by."""

        return BehaviorBindingV1(
            renderer_profile=self.renderer_profile,
            model_family=self.model_family,
            model_id=self.model_id,
            policy_revision=int(self.policy_revision),
            wire_api=self.wire_api,
            sampling_transport=self.sampling_transport,
            sampling=self.sampling,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": POLICY_BINDING_SCHEMA_VERSION,
            "config_id": self.binding_id,
            "binding_id": self.binding_id,
            "policy_kind": self.policy_kind,
            "kind": self.policy_kind,
            "probe": False,
            "trainable": bool(self.trainable),
            "handshake_id": self.handshake_id,
            "agreement_digest": self.agreement_digest,
            "agent_instance_id": self.agent_instance_id,
            "team_id": self.team_id,
            "parameter_group_id": self.parameter_group_id,
            "pinned_identity": self.pinned_identity,
            "renderer_profile": self.renderer_profile.to_dict(),
            "renderer_fingerprint": self.renderer_profile.fingerprint,
            "model_family": self.model_family,
            "model_id": self.model_id,
            "policy_revision": int(self.policy_revision),
            "behavior_fingerprint": self.behavior_fingerprint,
            "behavior_identity": self.behavior_identity,
            "transport": self.sampling_transport,
            "sampling_transport": self.sampling_transport,
            "wire_api": self.wire_api,
            "credential_mode": CredentialMode.PROXY.value,
            "proxy_mode": ProxyMode.PROXY_ONLY.value,
            "sampler_ready": bool(self.sampler_ready),
            "sampler_origin": self.origin.to_payload() if self.origin is not None else None,
            "immutable": True,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.to_payload()


@dataclass(frozen=True, slots=True)
class PolicySetBinding(JsonDataclassMixin):
    """A whole joint-episode roster, bound in one atomic operation."""

    policy_set_id: str
    policy_set_revision_id: str
    policy_set_revision: str
    handshake_id: str
    agreement_digest: str
    topology_id: str
    bindings: tuple[SamplerBinding, ...]

    @property
    def dispatchable(self) -> bool:
        return all(item.dispatchable for item in self.bindings)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": POLICY_SET_SCHEMA_VERSION,
            "config_id": self.policy_set_id,
            "policy_set_id": self.policy_set_id,
            "policy_set_revision_id": self.policy_set_revision_id,
            "policy_set_revision": self.policy_set_revision,
            "handshake_id": self.handshake_id,
            "agreement_digest": self.agreement_digest,
            "topology_id": self.topology_id,
            "probe": False,
            "atomic": True,
            "sampler_ready": self.dispatchable,
            "bindings": [item.to_payload() for item in self.bindings],
        }

    def to_dict(self) -> dict[str, Any]:
        return self.to_payload()


# --------------------------------------------------------------------------- #
# One model call
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ModelCallRecordV1(JsonDataclassMixin):
    """What one sampling call was, including which revision authored it.

    The revision rides here rather than on the wire. It is already inside the
    session-scoped credential, and a harness that stamped its own revision
    header would be choosing a field it is not allowed to see.
    """

    call_index: int
    binding_id: str
    proxy_request_id: str
    policy_revision: int
    behavior_fingerprint: str
    behavior_identity: str
    agent_instance_id: str | None
    trainable: bool
    wire_api: str
    sampling_transport: str
    endpoint: str
    request_digest: str
    response_digest: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": MODEL_CALL_SCHEMA_VERSION,
            "call_index": int(self.call_index),
            "binding_id": self.binding_id,
            "proxy_request_id": self.proxy_request_id,
            "policy_revision": int(self.policy_revision),
            "behavior_fingerprint": self.behavior_fingerprint,
            "behavior_identity": self.behavior_identity,
            "agent_instance_id": self.agent_instance_id,
            "trainable": bool(self.trainable),
            "wire_api": self.wire_api,
            "sampling_transport": self.sampling_transport,
            "endpoint": self.endpoint,
            "request_digest": self.request_digest,
            "response_digest": self.response_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.to_payload()


def shape_wire_request(binding: SamplerBinding, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build one outbound body in the binding's own wire shape.

    The two shapes are two datasets. ``chat_completions`` carries ``messages``,
    ``responses`` carries ``input`` items, and neither is flattened into the
    other -- a request holding the other wire's field is refused here rather
    than translated into something that looks trainable.
    """

    origin = binding.origin
    if origin is None:
        raise UnboundPolicy(
            f"binding {binding.binding_id!r} drives a pinned identity with no sampler"
        )
    extras = {
        key: value
        for key, value in payload.items()
        if key not in {"messages", "input", "prompt_token_ids", "model"}
    }
    body: dict[str, Any] = {"model": binding.model_id, **extras}
    if origin.sampling_transport == "tokens_in_tokens_out":
        if "prompt_token_ids" not in payload:
            raise WireShapeMismatch(
                "a tokens_in_tokens_out binding samples prompt_token_ids; text "
                "retokenized here would be a second renderer entering the run"
            )
        if "messages" in payload or "input" in payload:
            raise WireShapeMismatch(
                "a tokens_in_tokens_out request may not also carry a text round trip"
            )
        body["prompt_token_ids"] = list(payload["prompt_token_ids"])
        return body
    if "prompt_token_ids" in payload:
        raise WireShapeMismatch(
            "this binding declares message_in_capture_out; token ids would make the "
            "container the renderer"
        )
    if origin.api_family is InferenceApiFamily.RESPONSES:
        if "input" not in payload:
            raise WireShapeMismatch("a responses binding samples `input` items")
        if "messages" in payload:
            raise WireShapeMismatch(
                "flattening responses items into chat messages is prohibited"
            )
        body["input"] = list(payload["input"])
        return body
    if "messages" not in payload:
        raise WireShapeMismatch("a chat_completions binding samples `messages`")
    if "input" in payload:
        raise WireShapeMismatch(
            "presenting a chat-completions trajectory as a Responses distribution "
            "is prohibited"
        )
    body["messages"] = list(payload["messages"])
    return body


class BoundPolicySession:
    """Drives one bound policy against its external sampler, recording each call.

    Calls are recorded as they happen rather than reconstructed afterwards, so
    the revision on a record is the revision the call was actually dispatched
    under. Nothing here opens a socket: the transport is the container's.
    """

    def __init__(
        self, binding: SamplerBinding, *, transport: SamplerTransport | None = None
    ) -> None:
        self._binding = binding
        self._transport = transport
        self._calls: list[ModelCallRecordV1] = []
        self._lock = threading.Lock()

    @property
    def binding(self) -> SamplerBinding:
        return self._binding

    @property
    def calls(self) -> tuple[ModelCallRecordV1, ...]:
        return tuple(self._calls)

    def request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """The outbound body, without dispatching it."""

        return shape_wire_request(self._binding, payload)

    def call(self, payload: Mapping[str, Any]) -> tuple[ModelCallRecordV1, Mapping[str, Any]]:
        """Dispatch one sampling call and record the revision that authored it."""

        binding = self._binding
        if not binding.dispatchable:
            raise SamplerUnreachable(
                f"binding {binding.binding_id!r} is not dispatchable: its sampler was "
                "not reachable at bind time"
            )
        origin = binding.origin
        if origin is None:  # pragma: no cover - guarded by __post_init__
            raise UnboundPolicy(f"binding {binding.binding_id!r} has no sampler origin")
        if self._transport is None:
            raise PolicyBindingError(
                "no sampler transport was supplied; this module never opens its own "
                "connection"
            )
        body = shape_wire_request(binding, payload)
        response = self._transport.post(
            origin.endpoint, headers=origin.headers(), body=body
        )
        if not isinstance(response, Mapping):
            raise PolicyBindingError("sampler returned a non-object response")
        with self._lock:
            record = ModelCallRecordV1(
                call_index=len(self._calls),
                binding_id=binding.binding_id,
                proxy_request_id=origin.proxy_request_id,
                policy_revision=int(binding.policy_revision),
                behavior_fingerprint=binding.behavior_fingerprint,
                behavior_identity=binding.behavior_identity,
                agent_instance_id=binding.agent_instance_id,
                trainable=bool(binding.trainable),
                wire_api=binding.wire_api,
                sampling_transport=binding.sampling_transport,
                endpoint=origin.endpoint,
                request_digest=canonical_digest(body, length=32),
                response_digest=canonical_digest(dict(response), length=32),
            )
            self._calls.append(record)
        return record, response


# --------------------------------------------------------------------------- #
# Building a binding out of a request
# --------------------------------------------------------------------------- #


def _renderer_profile(facts: RuntimeFacts) -> RendererProfileV1:
    profile = facts.renderer_profile
    return RendererProfileV1(
        profile_id=profile.profile_id,
        package=profile.package,
        package_version=profile.package_version,
        config_digest=profile.config_digest,
        tokenizer_id=profile.tokenizer_id,
        tokenizer_digest=profile.tokenizer_digest,
        stop_token_ids=tuple(profile.stop_token_ids),
        modalities=tuple(profile.modalities),
        add_generation_prompt=bool(profile.add_generation_prompt),
    )


def _sampling_profile(payload: Any) -> SamplingProfileV1:
    if not isinstance(payload, Mapping):
        return SamplingProfileV1()
    max_tokens = payload.get("max_tokens")
    seed = payload.get("seed")
    return SamplingProfileV1(
        temperature=float(payload.get("temperature", 1.0)),
        top_p=float(payload.get("top_p", 1.0)),
        max_tokens=int(max_tokens) if max_tokens is not None else None,
        seed=int(seed) if seed is not None else None,
    )


def assert_no_embedded_credential(body: Mapping[str, Any]) -> None:
    """Refuse a raw provider credential at the top level of a binding request."""

    inline = tuple(sorted(name for name in FORBIDDEN_CREDENTIAL_FIELDS if name in body))
    if inline:
        raise EmbeddedCredential(
            f"binding request carries {inline} inline; the only credential this "
            "container accepts is the session-scoped one inside sampler_origin"
        )


def bind_sampler_policy(
    facts: RuntimeFacts,
    agreement: AdmittedHandshake,
    body: Mapping[str, Any],
    *,
    reachability: SamplerReachability | None = None,
    instance: AgentInstanceFacts | None = None,
) -> SamplerBinding:
    """Build one sampler binding from a request, or refuse.

    ``instance`` is the declared :class:`AgentInstanceFacts` this binding is for
    when the request is part of a roster; a solo binding has none.
    """

    kind = str(body.get("kind") or body.get("policy_kind") or "")
    if kind not in SAMPLER_POLICY_KINDS:
        raise UnknownPolicyKind(
            f"policy binding kind {kind!r} is not one of {sorted(SAMPLER_POLICY_KINDS)}"
        )
    assert_no_embedded_credential(body)
    if "policy_revision" not in body:
        raise RevisionPinMismatch(
            "a sampler binding must name the policy_revision it pins; a default "
            "would let an attempt sample a revision nobody chose"
        )
    pinned_revision = int(body["policy_revision"])
    pinned_fingerprint = str(body.get("behavior_fingerprint") or "").strip()

    trainable = bool(instance.trainable) if instance is not None else kind != PINNED_POLICY_KIND
    pinned_identity = str(
        body.get("pinned_identity")
        or body.get("policy_ref")
        or (instance.pinned_identity if instance is not None else "")
        or ""
    ).strip()
    if not trainable and pinned_identity.lower() in ALIAS_REFS:
        raise AliasOpponentBinding(
            f"pinned instance may not resolve the moving alias {pinned_identity!r}; "
            "two episodes bound to it are not comparable samples"
        )

    origin_payload = body.get("sampler_origin")
    origin: SamplerOriginV1 | None = None
    if origin_payload is not None:
        origin = SamplerOriginV1.from_payload(origin_payload)
        if origin.policy_revision != pinned_revision:
            raise RevisionPinMismatch(
                f"sampler origin serves revision {origin.policy_revision} but the "
                f"binding pins {pinned_revision}"
            )
        if pinned_fingerprint and origin.behavior_fingerprint != pinned_fingerprint:
            raise RevisionPinMismatch(
                f"sampler origin declares behavior {origin.behavior_fingerprint} but "
                f"the binding pins {pinned_fingerprint}"
            )
        transport = origin.sampling_transport
        if transport not in facts.policy.supported_transports:
            raise TransportUnsupported(
                f"container declares transports {facts.policy.supported_transports}, "
                f"the binding asked for {transport!r}"
            )
        wire_api = origin.wire_api
        if wire_api != facts.policy.wire_api:
            raise WireShapeMismatch(
                f"container binds over wire {facts.policy.wire_api!r}, the binding "
                f"asked for {wire_api!r}"
            )
        behavior_fingerprint = origin.behavior_fingerprint
    elif trainable:
        raise PolicyBindingError(
            "a trainable binding must carry a session-scoped sampler origin"
        )
    else:
        transport = facts.policy.binding_transport
        wire_api = facts.policy.wire_api
        behavior_fingerprint = pinned_fingerprint or canonical_digest(
            {"pinned_identity": pinned_identity, "revision": pinned_revision}, length=32
        )

    probe = reachability or UnprobedSampler()
    ready = True if origin is None else bool(probe.reachable(origin))

    parameter_group_id: str | None = None
    if trainable:
        bound_instance = instance
        if bound_instance is None:
            # A solo binding names no instance, but the container still knows
            # which parameter group its tokens belong to when exactly one
            # instance is trainable. Leaving it unset means the evidence hands
            # the reader an unattributed span, and a reader that has to guess a
            # parameter group guesses one the topology never declared.
            trainees = tuple(item for item in facts.topology.agent_instances if item.trainable)
            bound_instance = trainees[0] if len(trainees) == 1 else None
        if bound_instance is not None:
            parameter_group_id = facts.topology.parameter_groups.get(
                bound_instance.policy_type_id
            )

    binding_id = "pc_" + canonical_digest(
        {
            "agreement_digest": agreement.agreement_digest,
            "agent_instance_id": instance.agent_instance_id if instance is not None else None,
            "kind": kind,
            "policy_revision": pinned_revision,
            "behavior_fingerprint": behavior_fingerprint,
            "proxy_request_id": origin.proxy_request_id if origin is not None else None,
            "pinned_identity": pinned_identity or None,
        },
        length=16,
    )
    return SamplerBinding(
        binding_id=binding_id,
        policy_kind=kind,
        handshake_id=agreement.handshake_id,
        agreement_digest=agreement.agreement_digest,
        renderer_profile=_renderer_profile(facts),
        model_family=str(body.get("model_family") or ""),
        model_id=str(body.get("model_id") or ""),
        policy_revision=pinned_revision,
        behavior_fingerprint=behavior_fingerprint,
        wire_api=wire_api,
        sampling_transport=transport,
        sampler_ready=ready,
        trainable=trainable,
        agent_instance_id=(
            instance.agent_instance_id
            if instance is not None
            else (str(body["agent_instance_id"]) if "agent_instance_id" in body else None)
        ),
        team_id=instance.team_id if instance is not None else body.get("team_id"),
        parameter_group_id=parameter_group_id,
        pinned_identity=pinned_identity or None,
        origin=origin,
        sampling=_sampling_profile(body.get("sampling")),
    )


def bind_policy_set(
    facts: RuntimeFacts,
    agreement: AdmittedHandshake,
    body: Mapping[str, Any],
    *,
    reachability: SamplerReachability | None = None,
) -> PolicySetBinding:
    """Bind every declared instance of a joint episode, or none of them.

    Roster completeness is checked before one binding is built, and readiness is
    checked across the whole roster afterwards, so a set never half-exists: the
    caller either gets a set every member of which is dispatchable, or a typed
    refusal naming what was missing.
    """

    assert_no_embedded_credential(body)
    raw = body.get("bindings")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise HalfBoundRoster("policy set bindings must be a list")
    declared = {item.agent_instance_id: item for item in facts.topology.agent_instances}
    if not declared:
        raise HalfBoundRoster(
            f"topology {facts.topology.topology_id!r} declares no agent instance to bind"
        )
    items: dict[str, Mapping[str, Any]] = {}
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise HalfBoundRoster("each policy set binding must be an object")
        instance_id = str(entry.get("agent_instance_id") or "")
        if instance_id in items:
            raise HalfBoundRoster(
                f"instance {instance_id!r} is bound twice; one instance is one policy"
            )
        items[instance_id] = entry
    missing = tuple(sorted(set(declared) - set(items)))
    unknown = tuple(sorted(set(items) - set(declared)))
    if missing or unknown:
        raise HalfBoundRoster(
            "no episode may start against a half-bound roster: missing "
            f"{missing}, unknown {unknown}"
        )

    inherited = frozenset(
        {
            "kind",
            "policy_kind",
            "policy_revision",
            "behavior_fingerprint",
            "model_family",
            "model_id",
            "sampling",
        }
    )
    shared = {key: value for key, value in body.items() if key in inherited}
    bound: list[SamplerBinding] = []
    for instance_id, instance in declared.items():
        entry = dict(shared)
        entry.update(items[instance_id])
        entry.setdefault(
            "kind", TRAINABLE_POLICY_KIND if instance.trainable else PINNED_POLICY_KIND
        )
        if not instance.trainable:
            entry["kind"] = PINNED_POLICY_KIND
        bound.append(
            bind_sampler_policy(
                facts, agreement, entry, reachability=reachability, instance=instance
            )
        )
    unready = tuple(
        sorted(item.agent_instance_id or "" for item in bound if not item.dispatchable)
    )
    if unready:
        raise SamplerUnreachable(
            f"instances {unready} have no reachable sampler; the whole roster is "
            "refused rather than starting an episode that cannot finish"
        )
    ordered = tuple(bound)
    revision = canonical_digest(
        [agreement.agreement_digest, [item.behavior_identity for item in ordered]],
        length=20,
    )
    set_id = "ps_" + canonical_digest(
        [agreement.agreement_digest, [item.binding_id for item in ordered]], length=16
    )
    return PolicySetBinding(
        policy_set_id=set_id,
        policy_set_revision_id=str(body.get("policy_set_revision_id") or set_id),
        policy_set_revision=revision,
        handshake_id=agreement.handshake_id,
        agreement_digest=agreement.agreement_digest,
        topology_id=facts.topology.topology_id,
        bindings=ordered,
    )


# --------------------------------------------------------------------------- #
# The registry: what is bound, what is dispatchable, what a rollout is pinned to
# --------------------------------------------------------------------------- #


class PolicyBindingRegistry:
    """Holds every issued binding and the identity each rollout is pinned to.

    The records are frozen; only this object holds state, one lock guards it,
    and nothing here does background work.
    """

    def __init__(self, *, reachability: SamplerReachability | None = None) -> None:
        self._reachability: SamplerReachability = reachability or UnprobedSampler()
        self._bindings: dict[str, SamplerBinding] = {}
        self._sets: dict[str, PolicySetBinding] = {}
        self._routes: dict[str, str] = {}
        self._admitted: dict[str, str] = {}
        self._lock = threading.Lock()

    @property
    def reachability(self) -> SamplerReachability:
        return self._reachability

    @property
    def binding_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._bindings)

    @property
    def policy_set_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._sets)

    def register(self, binding: SamplerBinding) -> SamplerBinding:
        with self._lock:
            return self._register_locked(binding)

    def _register_locked(self, binding: SamplerBinding) -> SamplerBinding:
        existing = self._bindings.get(binding.binding_id)
        if existing is not None:
            if existing.behavior_identity != binding.behavior_identity:
                raise BehaviorIdentityImmutable(
                    f"binding {binding.binding_id!r} is already bound to a different "
                    "behavior policy"
                )
            return existing
        if binding.origin is not None:
            route = binding.origin.proxy_request_id
            held = self._routes.get(route)
            if held is not None and self._bindings[held].behavior_identity != (
                binding.behavior_identity
            ):
                raise BehaviorIdentityImmutable(
                    f"sampler route {route!r} is already bound to another revision; "
                    "one per-attempt origin serves exactly one revision"
                )
            self._routes[route] = binding.binding_id
        self._bindings[binding.binding_id] = binding
        return binding

    def register_set(self, policy_set: PolicySetBinding) -> PolicySetBinding:
        """Register a whole roster atomically: every member, or nothing."""

        with self._lock:
            if policy_set.policy_set_id in self._sets:
                return self._sets[policy_set.policy_set_id]
            snapshot_bindings = dict(self._bindings)
            snapshot_routes = dict(self._routes)
            try:
                for item in policy_set.bindings:
                    self._register_locked(item)
            except PolicyBindingError:
                self._bindings = snapshot_bindings
                self._routes = snapshot_routes
                raise
            self._sets[policy_set.policy_set_id] = policy_set
            return policy_set

    def binding(self, binding_id: str) -> SamplerBinding:
        with self._lock:
            found = self._bindings.get(str(binding_id or ""))
        if found is None:
            raise UnboundPolicy(f"policy binding {binding_id!r} is not bound")
        return found

    def policy_set(self, policy_set_id: str) -> PolicySetBinding:
        with self._lock:
            found = self._sets.get(str(policy_set_id or ""))
        if found is None:
            raise UnboundPolicy(f"policy set {policy_set_id!r} is not bound")
        return found

    def recheck(self, binding_id: str) -> SamplerBinding:
        """Re-probe one binding's sampler. Readiness is a reading, not a promise."""

        binding = self.binding(binding_id)
        if binding.origin is None:
            return binding
        ready = bool(self._reachability.reachable(binding.origin))
        with self._lock:
            refreshed = replace(binding, sampler_ready=ready)
            self._bindings[binding.binding_id] = refreshed
            return refreshed

    def assert_dispatchable(self, binding_id: str) -> SamplerBinding:
        """The gate one attempt passes before it samples anything."""

        binding = self.binding(binding_id)
        if not binding.dispatchable:
            raise SamplerUnreachable(
                f"binding {binding_id!r} is bound but its sampler is not reachable; "
                "a revision is dispatchable only once its sampler is ready"
            )
        return binding

    def assert_roster_dispatchable(self, policy_set_id: str) -> PolicySetBinding:
        policy_set = self.policy_set(policy_set_id)
        for item in policy_set.bindings:
            self.assert_dispatchable(item.binding_id)
        return policy_set

    def admit(self, rollout_id: str, binding_id: str) -> SamplerBinding:
        """Pin one rollout's behavior-policy identity. It never changes after this."""

        binding = self.assert_dispatchable(binding_id)
        key = str(rollout_id or "").strip()
        if not key:
            raise PolicyBindingError("admission names no rollout_id")
        with self._lock:
            held = self._admitted.get(key)
            if held is not None and held != binding.behavior_identity:
                raise BehaviorIdentityImmutable(
                    f"rollout {key!r} was admitted under behavior {held} and may not "
                    f"be rebound to {binding.behavior_identity}"
                )
            self._admitted[key] = binding.behavior_identity
        return binding

    def admit_set(self, rollout_id: str, policy_set_id: str) -> PolicySetBinding:
        """Pin a whole joint episode. A half-bound roster never gets this far."""

        policy_set = self.assert_roster_dispatchable(policy_set_id)
        key = str(rollout_id or "").strip()
        if not key:
            raise PolicyBindingError("admission names no rollout_id")
        identity = canonical_digest(
            [item.behavior_identity for item in policy_set.bindings], length=32
        )
        with self._lock:
            held = self._admitted.get(key)
            if held is not None and held != identity:
                raise BehaviorIdentityImmutable(
                    f"rollout {key!r} was admitted under roster identity {held}"
                )
            self._admitted[key] = identity
        return policy_set

    def admitted_identity(self, rollout_id: str) -> str | None:
        with self._lock:
            return self._admitted.get(str(rollout_id or "").strip())

    def session(
        self, binding_id: str, *, transport: SamplerTransport | None = None
    ) -> BoundPolicySession:
        """A driver for one dispatchable binding. Refuses an unready sampler."""

        return BoundPolicySession(self.assert_dispatchable(binding_id), transport=transport)


# --------------------------------------------------------------------------- #
# The single coupling point to the admission port's binding half
# --------------------------------------------------------------------------- #


class CispoPolicyAdapter:
    """The sampler-binding half of the policy-binding surface, in wire terms.

    Kept to one class for the same reason ``CispoProbeAdapter`` is: the coupling
    to the port in ``cispo_contract`` is one object wide, and the typed machinery
    above it stands on its own.
    """

    def __init__(
        self,
        facts: RuntimeFacts,
        *,
        reachability: SamplerReachability | None = None,
        registry: PolicyBindingRegistry | None = None,
    ) -> None:
        self._facts = facts
        self.registry = registry or PolicyBindingRegistry(reachability=reachability)

    @property
    def facts(self) -> RuntimeFacts:
        return self._facts

    def declares_sampler_binding(self) -> bool:
        """A build that embeds credentials or serves a global origin cannot bind."""

        policy = self._facts.policy
        return bool(policy.session_scoped_sampler_origin and not policy.embeds_credentials)

    def _assert_declared(self) -> None:
        if not self.declares_sampler_binding():
            raise PolicyBindingError(
                "container declares no session-scoped credential-free sampler binding; "
                "binding one anyway would promise a property the handshake refused"
            )

    def bind(self, agreement: AdmittedHandshake, body: Mapping[str, Any]) -> dict[str, Any]:
        self._assert_declared()
        binding = bind_sampler_policy(
            self._facts, agreement, body, reachability=self.registry.reachability
        )
        return self.registry.register(binding).to_payload()

    def bind_set(self, agreement: AdmittedHandshake, body: Mapping[str, Any]) -> dict[str, Any]:
        self._assert_declared()
        policy_set = bind_policy_set(
            self._facts, agreement, body, reachability=self.registry.reachability
        )
        return self.registry.register_set(policy_set).to_payload()


__all__ = [
    "ALIAS_REFS",
    "BEHAVIOR_IDENTITY_FIELDS",
    "FORBIDDEN_CREDENTIAL_FIELDS",
    "MODEL_CALL_SCHEMA_VERSION",
    "PINNED_POLICY_KIND",
    "POLICY_BINDING_SCHEMA_VERSION",
    "POLICY_SET_SCHEMA_VERSION",
    "SAMPLER_POLICY_KINDS",
    "SAMPLING_TRANSPORTS",
    "TRAINABLE_POLICY_KIND",
    "WIRE_APIS",
    "AliasOpponentBinding",
    "BehaviorIdentityImmutable",
    "BoundPolicySession",
    "CispoPolicyAdapter",
    "EmbeddedCredential",
    "GlobalSamplerOrigin",
    "HalfBoundRoster",
    "ModelCallRecordV1",
    "PolicyBindingError",
    "PolicyBindingRegistry",
    "PolicySetBinding",
    "RevisionPinMismatch",
    "SamplerBinding",
    "SamplerCredential",
    "SamplerOriginV1",
    "SamplerReachability",
    "SamplerTransport",
    "SamplerUnreachable",
    "TransportUnsupported",
    "UnboundPolicy",
    "UnknownPolicyKind",
    "UnprobedSampler",
    "WireShapeMismatch",
    "assert_no_embedded_credential",
    "bind_policy_set",
    "bind_sampler_policy",
    "shape_wire_request",
]
