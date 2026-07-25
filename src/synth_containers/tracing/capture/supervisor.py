"""``CaptureSupervisor`` — mint a binding, run capture, seal a bundle.

This is the ergonomic entry point both Push 1 consumers use. It owns the ordering that
makes capture trustworthy:

1. mint and materialize the binding *before* the workload starts;
2. start the proxy and prove it is reachable from the caller's reachability class;
3. hand back the injectable provider base URL and the collector;
4. on exit — normal or not — flush the spool, seal the trace, and write a coverage
   receipt that says exactly what was and was not observed.

An exception inside the block does not lose the capture: the bundle is still sealed,
with an interrupted lifecycle and a partial completeness claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
import json
from pathlib import Path
import secrets
import threading
from types import TracebackType
from typing import Any
from collections.abc import Callable, Mapping
from urllib.parse import urlsplit, urlunsplit
import uuid

import httpx

from ..canonical import content_digest, utc_now
from ..models.actors import (
    ActorKind,
    ActorV5,
    SessionCoverageV5,
    SessionStatus,
    SessionV5,
)
from ..models.completeness import TerminationV5, TraceStatus
from ..models.events import EventType
from ..models.identity import (
    AliasV1,
    TraceIdentityV5,
    TraceKind,
    TraceProvenanceV5,
    mint_actor_id,
    mint_capture_id,
    mint_session_id,
    mint_trace_id,
    TraceContextV1,
)
from ..store.bundle import LocalTraceBundle
from .binding import (
    BindingCaptureV1,
    BindingContainerV1,
    BindingContextV1,
    BindingWorkloadV1,
    CaptureMode,
    CapturePolicyV1,
    Interception,
    TraceCaptureBindingV1,
    WorkloadKind,
    mint_binding,
)
from .collector import LocalCollector
from .collector_server import CollectorServer, SessionActivityError
from .coverage import (
    CaptureFinalizationV1,
    finalization_from_dict,
    new_coverage_receipt,
)
from .finalizer import (
    FINALIZER_NAME,
    FINALIZER_VERSION,
    SealedCapture,
    TraceFinalizer,
)
from .proxy import CaptureProxy
from .proxy import (
    ANTHROPIC_MESSAGES_PATH,
    CHAT_COMPLETIONS_PATH,
    RESPONSES_COMPACT_PATH,
    RESPONSES_PATH,
)
from .routes import ProviderEndpointConfig, UpstreamAuthKind
from .egress import EgressAssertion, mitm_environment
from .envelope import RawRecordType
from .mitm import (
    MITM_LIFECYCLE_SCHEMA_VERSION,
    MitmLifecycleReceiptV1,
    MitmStartupError,
    ScopedMitmProxy,
)
from .websocket import ResponsesWebSocketRelay, ResponsesWebSocketServer
from .session import CaptureSession
from .spool import RawSpool


class CaptureNotReady(RuntimeError):
    """Raised when a required-mode capture cannot prove the proxy is reachable."""


@dataclass(slots=True)
class SupervisorConfig:
    """Everything the supervisor needs that it cannot mint itself."""

    bundle_root: Path
    trace_key: dict[str, Any]
    upstream_base_url: str
    provenance: TraceProvenanceV5
    identity: TraceIdentityV5 = field(default_factory=TraceIdentityV5)
    workload_kind: WorkloadKind | str = WorkloadKind.REACT
    trace_kind: TraceKind | str = TraceKind.AGENT_ROLLOUT
    root_actor_name: str = "workload"
    root_actor_kind: ActorKind | str = ActorKind.AGENT
    policy: CapturePolicyV1 = field(default_factory=CapturePolicyV1)
    mode: CaptureMode | str = CaptureMode.REQUIRED
    interception: Interception | str = Interception.PROVIDER_PROXY
    upstream_api_key: str | None = None
    proxy_host: str = "127.0.0.1"
    proxy_port: int = 0
    container: BindingContainerV1 = field(default_factory=BindingContainerV1)
    context: BindingContextV1 = field(default_factory=BindingContextV1)
    run_id: str | None = None
    rollout_id: str | None = None
    workflow_id: str | None = None
    trace_id: str | None = None
    capture_id: str | None = None
    resume: bool = False
    provider_endpoints: tuple[ProviderEndpointConfig, ...] | None = None
    anthropic_base_url: str | None = None
    anthropic_api_key: str | None = None
    collector_host: str = "127.0.0.1"
    collector_port: int = 0
    collector_container_host: str | None = None
    container_binding_path: str | None = None
    container_output_dir: str | None = None
    egress_assertion: Callable[[], EgressAssertion] | None = None
    responses_websocket: bool = False
    websocket_host: str = "127.0.0.1"
    websocket_port: int = 0
    mitmdump_command: tuple[str, ...] = ("mitmdump",)
    mitm_host: str = "127.0.0.1"
    mitm_port: int = 0
    mitm_allowed_hosts: tuple[str, ...] | None = None
    mitm_temp_root: Path | None = None
    mitm_base_ca_bundle_path: Path | None = None
    mitm_container_ca_path: str | None = None
    mitm_startup_timeout: float = 10.0
    binding_path: Path | None = None


class CaptureSupervisor:
    """Context manager that runs one capture session and seals one bundle."""

    def __init__(self, config: SupervisorConfig) -> None:
        self.config = config
        self.bundle = LocalTraceBundle(config.bundle_root)
        if config.binding_path is not None and config.resume:
            raise ValueError("binding_path and resume=True are mutually exclusive")
        if config.capture_id and not config.resume:
            raise ValueError("an explicit capture_id is only valid when resume=True")
        if config.binding_path is not None:
            self.binding = _load_external_binding(config.binding_path)
            self.binding_path = self.bundle.write_binding(self.binding)
            trace_id = self.binding.trace_id
            capture_id = self.binding.capture_id
        elif config.resume:
            if not config.capture_id:
                raise ValueError("resume=True requires the existing capture_id")
            self.binding, self.binding_path = _load_resume_binding(
                self.bundle,
                capture_id=config.capture_id,
                trace_id=config.trace_id,
            )
            trace_id = self.binding.trace_id
            capture_id = self.binding.capture_id
        else:
            trace_id = config.trace_id or mint_trace_id(
                kind=str(config.trace_kind),
                key={"trace_key": config.trace_key, "nonce": uuid.uuid4().hex},
            )
            capture_id = mint_capture_id(
                trace_id=trace_id,
                key={"trace_key": config.trace_key, "nonce": uuid.uuid4().hex},
            )
            actor_id = mint_actor_id(trace_id=trace_id, name=config.root_actor_name)
            session_id = mint_session_id(
                trace_id=trace_id,
                actor_id=actor_id,
                nonce=uuid.uuid4().hex,
            )
            self.binding = mint_binding(
                trace_id=trace_id,
                capture_id=capture_id,
                trace_kind=config.trace_kind,
                policy=config.policy,
                container=config.container,
                context=config.context,
                workload=BindingWorkloadV1(
                    kind=config.workload_kind,
                    root_actor_id=actor_id,
                    actor_session_id=session_id,
                    run_id=config.run_id,
                    rollout_id=config.rollout_id,
                    workflow_id=config.workflow_id,
                ),
                capture=BindingCaptureV1(
                    interception=config.interception,
                    mode=config.mode,
                    output_artifact_root=str(config.bundle_root),
                ),
            )
            self.binding_path = self.bundle.write_binding(self.binding)
        capture_root = self.bundle.capture_root(trace_id)
        self.spool = RawSpool(
            capture_root,
            capture_id=capture_id,
            max_segment_records=self.binding.policy.max_segment_records,
        )
        self.session = CaptureSession(
            binding=self.binding,
            spool=self.spool,
            blobs=self.bundle.blobs,
        )
        self._terminal_resume_candidate = bool(
            config.resume
            and any(
                str(record.get("record_type") or "")
                == str(RawRecordType.CAPTURE_FINISHED)
                for record in self.spool.records()
            )
        )
        if (
            not self._terminal_resume_candidate
            and str(self.binding.capture.mode)
            == CaptureMode.OBSERVE_AND_TRANSFORM
        ):
            raise ValueError(
                "observe_and_transform capture requires an explicit versioned "
                "transformation specification; transformation is not implemented"
            )
        if (
            not self._terminal_resume_candidate
            and str(self.binding.capture.mode)
            == CaptureMode.REQUIRED_EGRESS_ASSERTED
            and config.egress_assertion is None
        ):
            raise ValueError(
                "required_egress_asserted mode requires an egress_assertion callback"
            )
        self._endpoints = (
            _terminal_resume_provider_endpoints()
            if self._terminal_resume_candidate
            else (
                config.provider_endpoints
                or _default_provider_endpoints(config)
            )
        )
        self._collector_token = secrets.token_urlsafe(32)
        root_context = self.binding.context_for_child()
        self._root_context = root_context
        self._websocket_context_token = f"sk_trace_{secrets.token_urlsafe(32)}"
        self._contexts: dict[str, TraceContextV1] = {
            root_context.capture_id: root_context
        }
        self._lifecycle_lock = threading.RLock()
        self._finalizing = False
        self._finalization_started = False
        self._finalization_request: tuple[
            str,
            TerminationV5 | None,
            int | None,
        ] | None = None
        self._finalization_captured_at: str | None = None
        self._finalization_services_stopped = False
        self._finalization_egress_evaluated = False
        self._finalization_egress_checked = False
        self._finalization_mitm_checked = False
        self._finalization_assertion: EgressAssertion | None = None
        self._finalization_mitm_receipt: MitmLifecycleReceiptV1 | None = None
        self._finalization_mitm_failure: str | None = None
        self._finalization_egress_failure: str | None = None
        self._finalization_receipt: Any | None = None
        self._finalization_error: str | None = None
        self._terminal_resume_status: str | None = None
        self._terminal_finalization: CaptureFinalizationV1 | None = None
        self._extra_actors: list[ActorV5] = []
        self._extra_sessions: list[SessionV5] = []
        self._declared_session_terminals: dict[str, tuple[str, str, str]] = {}
        self._aliases: list[AliasV1] = []
        self.proxy = CaptureProxy(
            self.session,
            upstream_base_url=(
                "https://terminal-resume.invalid/v1"
                if self._terminal_resume_candidate
                else config.upstream_base_url
            ),
            upstream_api_key=(
                None
                if self._terminal_resume_candidate
                else config.upstream_api_key
            ),
            provider_endpoints=self._endpoints,
            host=(
                "127.0.0.1"
                if self._terminal_resume_candidate
                else config.proxy_host
            ),
            port=0 if self._terminal_resume_candidate else config.proxy_port,
            context_resolver=self._resolve_provider_context,
            context_activity_begin=lambda context: (
                self.collector_server.begin_context_activity(context)
            ),
        )
        self.collector = LocalCollector(self.session)
        responses_endpoint = next(
            (
                endpoint
                for endpoint in self._endpoints
                if endpoint.route == RESPONSES_PATH
            ),
            None,
        )
        if (
            not self._terminal_resume_candidate
            and config.responses_websocket
            and responses_endpoint is None
        ):
            raise ValueError(
                "Responses WebSocket capture requires a /v1/responses provider endpoint"
            )
        self.websocket_server = (
            ResponsesWebSocketServer(
                ResponsesWebSocketRelay(
                    self.session,
                    upstream_url=_websocket_url(responses_endpoint.upstream_url()),
                    stats=self.proxy.stats,
                    authorization=(
                        (
                            f"{responses_endpoint.auth_scheme} "
                            f"{responses_endpoint.api_key}"
                        ).strip()
                        if responses_endpoint.auth_kind == UpstreamAuthKind.BEARER
                        and responses_endpoint.api_key
                        else None
                    ),
                ),
                host=config.websocket_host,
                port=config.websocket_port,
                context_resolver=self._resolve_provider_context,
                query_context_resolver=self._resolve_websocket_context_token,
                context_activity_begin=lambda context: (
                    self.collector_server.begin_context_activity(context)
                ),
            )
            if config.responses_websocket
            and not self._terminal_resume_candidate
            else None
        )
        self.collector_server = CollectorServer(
            self.collector,
            host=(
                "127.0.0.1"
                if self._terminal_resume_candidate
                else config.collector_host
            ),
            port=0 if self._terminal_resume_candidate else config.collector_port,
            collector_token=self._collector_token,
            on_register_context=self._register_remote_context,
            on_finish_context=self._finish_remote_context,
        )
        interception = str(self.binding.capture.interception)
        mode = str(self.binding.capture.mode)
        self.mitm = (
            ScopedMitmProxy(
                capture_id=capture_id,
                endpoints=self._endpoints,
                capture_proxy_host=config.proxy_host,
                capture_proxy_port=self.proxy.port,
                command=config.mitmdump_command,
                host=config.mitm_host,
                port=config.mitm_port,
                allowed_hosts=config.mitm_allowed_hosts,
                temp_root=config.mitm_temp_root,
                base_ca_bundle_path=config.mitm_base_ca_bundle_path,
                startup_timeout=config.mitm_startup_timeout,
            )
            if interception in {Interception.TLS_MITM, Interception.BOTH}
            and mode != CaptureMode.DISABLED
            and not self._terminal_resume_candidate
            else None
        )
        self.receipt = new_coverage_receipt(
            binding_id=self.binding.binding_id,
            binding_digest=self.binding.content_digest,
            capture_id=capture_id,
            scope="model_calls_and_application",
            requested_mode=str(self.binding.capture.mode),
            resolved_mode=str(self.binding.capture.mode),
            interception=str(self.binding.capture.interception),
            proxy_config_digest=self.binding.capture.proxy_config_digest or "",
            started_at=(
                self.session.first_observed_at
                if config.resume
                else None
            ),
        )
        self.sealed: SealedCapture | None = None
        self._capture_operational = False
        self._startup_failure: str | None = None
        if config.resume:
            try:
                self._restore_child_state()
                if self._terminal_finalization is not None:
                    self.collector_server.stop()
                    self.proxy.stop(reason="terminal_resume")
            except BaseException:
                self.collector_server.stop()
                self.proxy.stop(reason="invalid_resume")
                raise

    def _resolve_provider_context(
        self,
        headers: Any,
    ) -> TraceContextV1 | None:
        normalized = {str(name).lower(): str(value) for name, value in headers.items()}
        capture_id = normalized.get("x-synth-capture-id")
        if not capture_id:
            return None
        token = normalized.get("x-synth-context-token") or ""
        context = self._contexts.get(capture_id)
        if context is None:
            return None
        if not self.collector_server.token_authorizes(capture_id, token):
            return None
        if self.collector_server.is_context_terminal(capture_id):
            return None
        expected = {
            "x-synth-trace-id": context.trace_id,
            "x-synth-actor-id": context.actor_id,
            "x-synth-session-id": context.actor_session_id,
        }
        if any(normalized.get(name) != value for name, value in expected.items()):
            return None
        return context

    def _resolve_websocket_context_token(
        self,
        token: str,
    ) -> TraceContextV1 | None:
        if not secrets.compare_digest(token, self._websocket_context_token):
            return None
        return self._root_context

    def _register_remote_context(
        self,
        context: TraceContextV1,
        actor_payload: dict[str, Any],
        session_payload: dict[str, Any],
    ) -> str:
        return self.register_child_context(
            context,
            actor=_actor_from_payload(actor_payload),
            session=_session_from_payload(session_payload),
        )

    def _restore_child_state(self) -> None:
        """Rebuild child topology and terminal indexes from the raw authority."""

        records = tuple(self.spool.records())
        registered_sessions: set[str] = set()
        terminal_sessions: set[str] = set()
        capture_finished = False
        root_actor_id = self.binding.workload.root_actor_id
        root_session_id = self.binding.workload.actor_session_id
        terminal_statuses = {
            str(SessionStatus.COMPLETED),
            str(SessionStatus.FAILED),
            str(SessionStatus.INTERRUPTED),
        }

        for record in records:
            record_type = str(record.get("record_type") or "")
            actor_id = str(record.get("actor_id") or "")
            session_id = str(record.get("session_id") or "")
            if capture_finished:
                raise ValueError("raw record follows capture.finished")
            if session_id in terminal_sessions:
                raise ValueError(
                    f"raw record follows session.finished for {session_id}"
                )

            if record_type == str(RawRecordType.ACTOR_DECLARED):
                if actor_id != root_actor_id or session_id != root_session_id:
                    raise ValueError(
                        "actor.declared must belong to the root session"
                    )
                payload = record.get("payload")
                actor_payload = (
                    payload.get("actor")
                    if isinstance(payload, Mapping)
                    else None
                )
                if not isinstance(actor_payload, Mapping):
                    raise ValueError("restored actor declaration is invalid")
                actor = _actor_from_payload(actor_payload)
                if actor.actor_id == root_actor_id:
                    raise ValueError(
                        "restored actor declaration collides with the root actor"
                    )
                existing = next(
                    (
                        item
                        for item in self._extra_actors
                        if item.actor_id == actor.actor_id
                    ),
                    None,
                )
                if existing is not None:
                    if existing.content_digest != actor.content_digest:
                        raise ValueError(
                            "restored actor declaration conflicts with prior facts"
                        )
                    continue
                self._extra_actors.append(actor)
                continue

            if record_type == str(RawRecordType.ALIAS_DECLARED):
                if actor_id != root_actor_id or session_id != root_session_id:
                    raise ValueError(
                        "alias.declared must belong to the root session"
                    )
                payload = record.get("payload")
                alias_payload = (
                    payload.get("alias")
                    if isinstance(payload, Mapping)
                    else None
                )
                if not isinstance(alias_payload, Mapping):
                    raise ValueError("restored alias declaration is invalid")
                alias = AliasV1(**dict(alias_payload))
                existing = next(
                    (
                        item
                        for item in self._aliases
                        if (
                            str(item.namespace),
                            item.value,
                            item.target_kind,
                        )
                        == (
                            str(alias.namespace),
                            alias.value,
                            alias.target_kind,
                        )
                    ),
                    None,
                )
                if existing is not None:
                    if existing != alias:
                        raise ValueError(
                            "restored alias declaration conflicts with prior facts"
                        )
                    continue
                self._aliases.append(alias)
                continue

            if record_type == str(RawRecordType.CHILD_REGISTERED):
                if session_id in registered_sessions:
                    raise ValueError(
                        f"duplicate child registration for {session_id}"
                    )
                payload = record.get("payload")
                if not isinstance(payload, Mapping):
                    raise ValueError("restored child registration payload is invalid")
                actor_payload = payload.get("actor")
                session_payload = payload.get("session")
                if not isinstance(actor_payload, Mapping) or not isinstance(
                    session_payload,
                    Mapping,
                ):
                    raise ValueError(
                        "restored child registration identity is invalid"
                    )
                actor = _actor_from_payload(actor_payload)
                session = _session_from_payload(session_payload)
                if (
                    actor_id != actor.actor_id
                    or session_id != session.session_id
                    or session.actor_id != actor.actor_id
                ):
                    raise ValueError(
                        "restored child registration envelope identity mismatch"
                    )
                context_payload = payload.get("context")
                if context_payload is None:
                    normalized_actor, normalized_session = (
                        self._normalize_declared_child(actor, session)
                    )
                    if (
                        normalized_actor.content_digest != actor.content_digest
                        or normalized_session.content_digest
                        != session.content_digest
                    ):
                        raise ValueError(
                            "restored declarative child topology is not normalized"
                        )
                    context = None
                else:
                    if not isinstance(context_payload, Mapping):
                        raise ValueError(
                            "restored child registration context is invalid"
                        )
                    context = TraceContextV1(**dict(context_payload))
                    normalized_actor, normalized_session, normalized_context = (
                        self._normalize_registered_child(
                            actor,
                            session,
                            context,
                        )
                    )
                    if (
                        normalized_actor.content_digest != actor.content_digest
                        or normalized_session.content_digest
                        != session.content_digest
                        or normalized_context != context
                    ):
                        raise ValueError(
                            "restored registered child topology is not normalized"
                        )
                    context = normalized_context
                self._install_child_identity(
                    normalized_actor,
                    normalized_session,
                    context=context,
                    persist=False,
                )
                registered_sessions.add(session_id)
                continue

            if record_type == str(RawRecordType.SESSION_FINISHED):
                session = next(
                    (
                        item
                        for item in self._extra_sessions
                        if item.session_id == session_id
                    ),
                    None,
                )
                if session is None:
                    raise ValueError(
                        f"restored terminal fact references unknown child {session_id}"
                    )
                if actor_id != session.actor_id:
                    raise ValueError(
                        "restored terminal fact actor does not own child session"
                    )
                payload = record.get("payload")
                body = payload.get("body") if isinstance(payload, Mapping) else None
                if (
                    not isinstance(payload, Mapping)
                    or str(payload.get("event_type") or "")
                    != str(EventType.SESSION_FINISHED)
                    or not isinstance(body, Mapping)
                ):
                    raise ValueError("restored terminal child fact is invalid")
                status = str(body.get("status") or "")
                if status not in terminal_statuses:
                    raise ValueError(
                        "restored child session status must be terminal"
                    )
                ended_at = str(body.get("ended_at") or "")
                if ended_at != str(record.get("occurred_at") or ""):
                    raise ValueError(
                        "restored child ended_at must equal record occurrence"
                    )
                self._validate_declared_terminal_time(session, ended_at)
                envelope_id = str(record.get("envelope_id") or "")
                context = next(
                    (
                        item
                        for item in self._contexts.values()
                        if item.actor_session_id == session_id
                        and item.capture_id != self.binding.capture_id
                    ),
                    None,
                )
                if context is not None:
                    self.collector_server.restore_terminal_context(
                        context,
                        status=status,
                        ended_at=ended_at,
                        envelope_id=envelope_id,
                    )
                else:
                    self._declared_session_terminals[session_id] = (
                        status,
                        ended_at,
                        envelope_id,
                    )
                terminal_sessions.add(session_id)
                continue

            if record_type == str(RawRecordType.CAPTURE_FINISHED):
                if actor_id != root_actor_id or session_id != root_session_id:
                    raise ValueError("capture.finished must belong to the root session")
                payload = record.get("payload")
                if not isinstance(payload, Mapping):
                    raise ValueError("capture.finished payload is invalid")
                if not payload.get("schema_version"):
                    raise ValueError(
                        "terminal capture cannot resume without a typed "
                        "capture finalization snapshot"
                    )
                finalization = finalization_from_dict(dict(payload))
                if finalization.captured_at != str(record.get("occurred_at") or ""):
                    raise ValueError(
                        "capture finalization captured_at must equal record occurrence"
                    )
                if (
                    finalization.coverage_seed.capture_id != self.binding.capture_id
                    or finalization.coverage_seed.binding_id
                    != self.binding.binding_id
                    or finalization.coverage_seed.binding_digest
                    != self.binding.content_digest
                ):
                    raise ValueError(
                        "capture finalization coverage does not match its binding"
                    )
                if (
                    finalization.coverage_seed.child_exit_code
                    != finalization.child_exit_code
                ):
                    raise ValueError(
                        "capture finalization child exit code is inconsistent"
                    )
                if finalization.aliases != tuple(self._aliases):
                    raise ValueError(
                        "capture finalization aliases disagree with raw authority"
                    )
                self._restore_finalization(finalization)
                capture_finished = True
                continue

            if session_id != root_session_id:
                session = next(
                    (
                        item
                        for item in self._extra_sessions
                        if item.session_id == session_id
                    ),
                    None,
                )
                if session is None or actor_id != session.actor_id:
                    raise ValueError(
                        "restored raw record references an unknown child identity"
                    )

        if capture_finished:
            self.collector_server.freeze()
            self.proxy.freeze()
            self.collector.freeze()
            self.session.freeze_existing()

    def _restore_finalization(
        self,
        finalization: CaptureFinalizationV1,
    ) -> None:
        """Restore the exact lifecycle and receipt inputs sealed before a crash."""

        assertion = (
            _egress_assertion_from_payload(finalization.egress_assertion)
            if finalization.egress_assertion is not None
            else None
        )
        mitm_receipt = (
            _mitm_receipt_from_payload(finalization.mitm_lifecycle)
            if finalization.mitm_lifecycle is not None
            else None
        )
        normalized_status = str(finalization.status)
        self._terminal_finalization = finalization
        self._terminal_resume_status = normalized_status
        self._finalization_started = True
        self._finalization_request = (
            normalized_status,
            finalization.termination,
            finalization.child_exit_code,
        )
        self._finalization_captured_at = finalization.captured_at
        self._finalization_services_stopped = True
        self._finalization_egress_evaluated = True
        self._finalization_egress_checked = False
        self._finalization_mitm_checked = False
        self._finalization_assertion = assertion
        self._finalization_mitm_receipt = mitm_receipt
        self._finalization_mitm_failure = finalization.mitm_failure
        self._finalization_egress_failure = finalization.egress_failure
        self._finalization_receipt = finalization.coverage_seed
        self.receipt = finalization.coverage_seed

    # -- lifecycle ---------------------------------------------------------------

    def __enter__(self) -> "CaptureSupervisor":
        return self.start_capture()

    def start_capture(self) -> "CaptureSupervisor":
        """Start the selected capture mechanisms and enforce mode semantics."""

        if self._terminal_resume_status is not None:
            raise RuntimeError(
                "terminal capture cannot resume workload activity; call finalize() "
                "to reseal its existing authority"
            )
        mode = str(self.binding.capture.mode)
        if mode == CaptureMode.DISABLED:
            self.receipt = replace(
                self.receipt,
                registration_ok=False,
                readiness_ok=False,
                reachability_detail="capture disabled by policy",
                injected_variables=(),
                completeness_reasons=(
                    *self.receipt.completeness_reasons,
                    "capture disabled by policy",
                ),
            )
            self._capture_operational = False
            return self
        try:
            self.collector_server.start()
            self.proxy.start()
            if self.websocket_server is not None:
                self.websocket_server.start()
            if self.mitm is not None:
                self.mitm.start()
        except BaseException as exc:
            detail = f"capture startup failed: {type(exc).__name__}: {exc}"
            self._startup_failure = detail
            self._stop_capture_services(reason="startup_failed")
            self.receipt = replace(
                self.receipt,
                registration_ok=False,
                readiness_ok=False,
                reachability_detail=detail,
                injected_variables=(),
                completeness_reasons=(
                    *self.receipt.completeness_reasons,
                    detail,
                ),
            )
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if mode == CaptureMode.BEST_EFFORT:
                return self
            if isinstance(exc, CaptureNotReady):
                raise
            raise CaptureNotReady(detail) from exc
        proxy_ready, collector_ready, mitm_ready = self._probe()
        readiness = proxy_ready and collector_ready and mitm_ready
        interception = str(self.binding.capture.interception)
        injected = ["SYNTH_TRACE_COLLECTOR_URL"]
        if interception in {Interception.PROVIDER_PROXY, Interception.BOTH}:
            injected.append("OPENAI_BASE_URL")
        if (
            interception in {Interception.PROVIDER_PROXY, Interception.BOTH}
            and any(item.route == ANTHROPIC_MESSAGES_PATH for item in self._endpoints)
        ):
            injected.append("ANTHROPIC_BASE_URL")
        if self.mitm is not None:
            injected.extend(
                (
                    "HTTP_PROXY",
                    "HTTPS_PROXY",
                    "SSL_CERT_FILE",
                    "REQUESTS_CA_BUNDLE",
                    "NODE_EXTRA_CA_CERTS",
                )
            )
        self.receipt = replace(
            self.receipt,
            registration_ok=True,
            readiness_ok=readiness,
            reachability_detail=(
                "selected capture services answered from the launching namespace"
                if readiness
                else (
                    f"proxy_ready={proxy_ready}; collector_ready={collector_ready}; "
                    f"mitm_ready={mitm_ready}"
                )
            ),
            injected_variables=tuple(sorted(set(injected))),
        )
        if not readiness:
            detail = (
                f"capture readiness failed: proxy_ready={proxy_ready}; "
                f"collector_ready={collector_ready}; mitm_ready={mitm_ready}"
            )
            self._startup_failure = detail
            self._stop_capture_services(reason="readiness_failed")
            self.receipt = replace(
                self.receipt,
                injected_variables=(),
                completeness_reasons=(
                    *self.receipt.completeness_reasons,
                    detail,
                ),
            )
            if mode != CaptureMode.BEST_EFFORT:
                raise CaptureNotReady(detail)
            return self
        self._capture_operational = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        status = TraceStatus.COMPLETED if exc_type is None else TraceStatus.INTERRUPTED
        termination = (
            None
            if exc_type is None
            else TerminationV5(
                reason="workload_exception",
                detail=f"{exc_type.__name__}: {exc}",
            )
        )
        self.finalize(status=status, termination=termination)
        return False

    def _probe(self) -> tuple[bool, bool, bool]:
        try:
            with httpx.Client(timeout=10.0, trust_env=False) as client:
                proxy_response = client.get(f"{self.proxy.base_url}/healthz")
                collector_response = client.get(
                    f"{self.collector_server.base_url}/healthz",
                    headers={
                        "authorization": f"Bearer {self._collector_token}",
                        "x-synth-trace-id": self.binding.trace_id,
                        "x-synth-capture-id": self.binding.capture_id,
                        "x-synth-actor-id": self.binding.workload.root_actor_id,
                        "x-synth-session-id": (
                            self.binding.workload.actor_session_id
                        ),
                    },
                )
        except httpx.HTTPError:
            return False, False, bool(self.mitm is None or self.mitm.ready)
        return (
            proxy_response.status_code == 200,
            collector_response.status_code == 200,
            bool(self.mitm is None or self.mitm.ready),
        )

    def _stop_capture_services(self, *, reason: str) -> MitmLifecycleReceiptV1 | None:
        """Stop every component even when an earlier component failed to start."""

        mitm_receipt: MitmLifecycleReceiptV1 | None = None
        stop_errors: list[str] = []
        if self.mitm is not None:
            try:
                mitm_receipt = self.mitm.stop(reason=reason)
            except BaseException as exc:
                stop_errors.append(f"mitm={type(exc).__name__}: {exc}")
        if self.websocket_server is not None:
            try:
                self.websocket_server.stop()
            except BaseException as exc:
                stop_errors.append(f"websocket={type(exc).__name__}: {exc}")
        try:
            self.collector_server.stop()
        except BaseException as exc:
            stop_errors.append(f"collector={type(exc).__name__}: {exc}")
        try:
            self.proxy.stop(reason=reason)
        except BaseException as exc:
            stop_errors.append(f"provider_proxy={type(exc).__name__}: {exc}")
        self._capture_operational = False
        if stop_errors:
            detail = "capture service shutdown errors: " + "; ".join(stop_errors)
            self.receipt = replace(
                self.receipt,
                completeness_reasons=(
                    *self.receipt.completeness_reasons,
                    detail,
                ),
            )
            raise CaptureNotReady(detail)
        return mitm_receipt

    # -- injection surface --------------------------------------------------------

    @property
    def openai_base_url(self) -> str:
        """Value to inject as ``OPENAI_BASE_URL`` for an OpenAI-compatible client."""

        return self.proxy.openai_base_url

    @property
    def chat_completions_url(self) -> str:
        """Full chat-completions URL for clients that take an absolute endpoint."""

        return self.proxy.chat_completions_url

    def container_base_url(self, host: str) -> str:
        """The proxy base URL as seen from another network namespace, e.g. Docker."""

        return f"http://{host}:{self.proxy.port}/v1"

    def environment(
        self,
        container_host: str | None = None,
        *,
        binding_path: str | None = None,
        output_dir: str | None = None,
    ) -> dict[str, str]:
        """Return the complete secret-bearing launch environment for a child workload.

        The result contains the collector bearer token. Inject it directly into the
        child process; never serialize or log the mapping wholesale.
        """

        if (
            str(self.binding.capture.mode) == CaptureMode.DISABLED
            or not self._capture_operational
        ):
            return {}
        host = container_host or self.config.collector_container_host
        if host:
            collector_url = f"http://{host}:{self.collector_server.port}"
            openai_url = self.container_base_url(host)
            visible_binding = binding_path or self.config.container_binding_path
            visible_output = output_dir or self.config.container_output_dir
        else:
            collector_url = self.collector_server.base_url
            openai_url = self.openai_base_url
            visible_binding = binding_path or str(self.binding_path)
            visible_output = output_dir or str(self.bundle.root)
        context = self.binding.context_for_child(
            collector_url=collector_url,
            binding_path=visible_binding,
            output_dir=visible_output,
        )
        values = context.to_environment()
        values.update(
            {
                "SYNTH_TRACE_COLLECTOR_TOKEN": self._collector_token,
                "SYNTH_TRACE_CONTEXT_TOKEN": self._collector_token,
            }
        )
        interception = str(self.binding.capture.interception)
        if interception in {Interception.PROVIDER_PROXY, Interception.BOTH}:
            values["OPENAI_BASE_URL"] = openai_url
        if (
            self.websocket_server is not None
            and interception in {Interception.PROVIDER_PROXY, Interception.BOTH}
        ):
            ws_host = host or self.config.websocket_host
            values["OPENAI_RESPONSES_WEBSOCKET_URL"] = (
                f"ws://{ws_host}:{self.websocket_server.port}/v1/responses"
                f"?synth_trace_token={self._websocket_context_token}"
            )
        if (
            interception in {Interception.PROVIDER_PROXY, Interception.BOTH}
            and (
                self.config.anthropic_base_url
                or any(
                    item.route == ANTHROPIC_MESSAGES_PATH
                    for item in self._endpoints
                )
            )
        ):
            values["ANTHROPIC_BASE_URL"] = (
                f"http://{host}:{self.proxy.port}" if host else self.proxy.base_url
            )
        if self.mitm is not None:
            visible_ca_path = (
                self.config.mitm_container_ca_path if host else str(self.mitm.public_ca_path)
            )
            if not visible_ca_path:
                raise CaptureNotReady(
                    "container MITM capture requires mitm_container_ca_path for the "
                    "read-only public CA mount"
                )
            bypass_hosts = tuple(
                dict.fromkeys(
                    (
                        "127.0.0.1",
                        "localhost",
                        *(item for item in (host,) if item),
                    )
                )
            )
            values = mitm_environment(
                proxy_url=self.mitm.proxy_url_for(host),
                ca_bundle_path=visible_ca_path,
                base=values,
                no_proxy_hosts=bypass_hosts,
            )
        return values

    def provider_headers(
        self,
        context: TraceContextV1 | None = None,
    ) -> dict[str, str]:
        """Headers that attribute a proxied provider call to an authorized context."""

        resolved = context or self.binding.context_for_child()
        registered = self._contexts.get(resolved.capture_id)
        if registered != resolved:
            raise ValueError("provider trace context is not registered with this capture")
        return {
            "x-synth-trace-id": resolved.trace_id,
            "x-synth-capture-id": resolved.capture_id,
            "x-synth-actor-id": resolved.actor_id,
            "x-synth-session-id": resolved.actor_session_id,
            "x-synth-context-token": self.collector_server.context_token(
                resolved.capture_id
            ),
        }

    def context_token(self, context: TraceContextV1 | str) -> str:
        """Return one registered context's ephemeral, non-persisted capability."""

        capture_id = (
            context.capture_id
            if isinstance(context, TraceContextV1)
            else context
        )
        return self.collector_server.context_token(capture_id)

    def environment_descriptor(self, container_host: str | None = None) -> dict[str, Any]:
        """Return a log-safe description of launch injection without secret values."""

        values = self.environment(container_host)
        return {
            "variables": tuple(sorted(values)),
            "collector_url": values.get("SYNTH_TRACE_COLLECTOR_URL"),
            "openai_base_url": values.get("OPENAI_BASE_URL"),
            "anthropic_base_url": values.get("ANTHROPIC_BASE_URL"),
            "mitm_proxy_url": values.get("HTTPS_PROXY"),
            "mitm_public_ca_path": values.get("SSL_CERT_FILE"),
            "mitm_allowlist": self.mitm.allowed_hosts if self.mitm is not None else (),
            "binding_path": values.get("SYNTH_TRACE_BINDING_PATH"),
            "output_dir": values.get("SYNTH_TRACE_OUTPUT_DIR"),
            "collector_token": (
                "present"
                if values.get("SYNTH_TRACE_COLLECTOR_TOKEN")
                else "absent"
            ),
            "capture_operational": self._capture_operational,
        }

    def mitm_trust_mount(self) -> dict[str, Any]:
        """Return the one read-only public CA mount required by a container child."""

        if self.mitm is None:
            raise ValueError("this capture does not use TLS MITM interception")
        if not self.config.mitm_container_ca_path:
            raise ValueError("mitm_container_ca_path is required for a container mount")
        return self.mitm.trust_mount(self.config.mitm_container_ca_path)

    def declare_actor(self, actor: ActorV5, session: SessionV5 | None = None) -> None:
        """Add a non-root actor observed by the workload, such as the environment."""

        with self._lifecycle_lock:
            self._assert_capture_mutable()
            actor = actor.sealed()
            if actor.actor_id == self.binding.workload.root_actor_id:
                raise ValueError("child actor id collides with the root actor")
            if actor.parent_actor_id == actor.actor_id:
                raise ValueError("child actor cannot be its own parent")
            if session is None:
                existing = next(
                    (
                        item
                        for item in self._extra_actors
                        if item.actor_id == actor.actor_id
                    ),
                    None,
                )
                if (
                    existing is not None
                    and existing.content_digest != actor.content_digest
                ):
                    raise ValueError("actor id is already declared with different facts")
                if existing is None:
                    self.session.append(
                        RawRecordType.ACTOR_DECLARED,
                        payload={"actor": actor.to_dict()},
                        producer_version="synth-trace-supervisor/1",
                    )
                    self._extra_actors.append(actor)
                return
            actor, session = self._normalize_declared_child(actor, session)
            self._install_child_identity(
                actor,
                session,
                context=None,
                persist=True,
            )

    def _normalize_declared_child(
        self,
        actor: ActorV5,
        session: SessionV5,
    ) -> tuple[ActorV5, SessionV5]:
        if actor.actor_id == self.binding.workload.root_actor_id:
            raise ValueError("child actor id collides with the root actor")
        if actor.parent_actor_id == actor.actor_id:
            raise ValueError("child actor cannot be its own parent")
        if session.actor_id != actor.actor_id:
            raise ValueError("declared session must belong to declared actor")
        if session.session_id == self.binding.workload.actor_session_id:
            raise ValueError("child session id collides with the root session")
        if session.parent_session_id == session.session_id:
            raise ValueError("child session cannot be its own parent")
        if (
            str(session.status) != str(SessionStatus.RUNNING)
            or session.ended_at is not None
        ):
            raise ValueError("declared child session must start in running state")
        if session.parent_session_id:
            parent_session = next(
                (
                    item
                    for item in self._extra_sessions
                    if item.session_id == session.parent_session_id
                ),
                None,
            )
            parent_actor_id = (
                self.binding.workload.root_actor_id
                if session.parent_session_id
                == self.binding.workload.actor_session_id
                else (
                    parent_session.actor_id
                    if parent_session is not None
                    else None
                )
            )
            if parent_actor_id is None:
                raise ValueError(
                    "declared child session references an unknown parent"
                )
            if parent_session is not None and (
                parent_session.session_id in self._declared_session_terminals
                or (
                    parent_session.capture_id
                    and self.collector_server.is_context_terminal(
                        parent_session.capture_id
                    )
                )
            ):
                raise ValueError(
                    "terminal child session cannot declare another child"
                )
            if actor.parent_actor_id not in {None, parent_actor_id}:
                raise ValueError(
                    "declared child actor parent disagrees with session parent"
                )
            actor = replace(
                actor,
                parent_actor_id=parent_actor_id,
                content_digest="",
            ).sealed()
        return actor.sealed(), session.sealed()

    def declare_alias(self, alias: AliasV1) -> None:
        with self._lifecycle_lock:
            self._assert_capture_mutable()
            existing = next(
                (
                    item
                    for item in self._aliases
                    if (
                        str(item.namespace),
                        item.value,
                        item.target_kind,
                    )
                    == (
                        str(alias.namespace),
                        alias.value,
                        alias.target_kind,
                    )
                ),
                None,
            )
            if existing is not None:
                if existing != alias:
                    raise ValueError(
                        "alias identity is already declared with different facts"
                    )
                return
            self.session.append(
                RawRecordType.ALIAS_DECLARED,
                payload={"alias": alias.to_dict()},
                producer_version="synth-trace-supervisor/1",
            )
            self._aliases.append(alias)

    def register_child_context(
        self,
        context: TraceContextV1,
        *,
        actor: ActorV5,
        session: SessionV5,
    ) -> str:
        """Authorize one delegated child capture and include its identity in sealing."""

        with self._lifecycle_lock:
            self._assert_capture_mutable()
            actor, session, context = self._normalize_registered_child(
                actor,
                session,
                context,
            )
            return self._install_child_identity(
                actor,
                session,
                context=context,
                persist=True,
            )

    def _normalize_registered_child(
        self,
        actor: ActorV5,
        session: SessionV5,
        context: TraceContextV1,
    ) -> tuple[ActorV5, SessionV5, TraceContextV1]:
        if (
            actor.actor_id != context.actor_id
            or session.session_id != context.actor_session_id
        ):
            raise ValueError("child actor/session do not match trace context")
        if session.actor_id != actor.actor_id:
            raise ValueError("child session must belong to child actor")
        if context.trace_id != self.binding.trace_id:
            raise ValueError("child context must join the supervisor trace")
        if context.capture_id == self.binding.capture_id:
            raise ValueError("child capture id collides with the root capture")
        if actor.actor_id == self.binding.workload.root_actor_id:
            raise ValueError("child actor id collides with the root actor")
        if session.session_id == self.binding.workload.actor_session_id:
            raise ValueError("child session id collides with the root session")
        if context.actor_id == context.parent_actor_id:
            raise ValueError("child actor cannot be its own parent")
        if (
            context.parent_actor_session_id is not None
            and context.actor_session_id == context.parent_actor_session_id
        ):
            raise ValueError("child session cannot be its own parent")
        parent_contexts = [
            item
            for item in self._contexts.values()
            if item.actor_id == context.parent_actor_id
            and context.parent_actor_session_id
            in {None, item.actor_session_id}
        ]
        if len(parent_contexts) != 1:
            raise ValueError(
                "child context must identify one registered parent session"
            )
        parent_context = parent_contexts[0]
        if self.collector_server.is_context_terminal(parent_context.capture_id):
            raise ValueError("terminal child session cannot register another child")
        if actor.parent_actor_id not in {None, parent_context.actor_id}:
            raise ValueError("child actor parent does not match trace context")
        if session.parent_session_id not in {
            None,
            parent_context.actor_session_id,
        }:
            raise ValueError("child session parent does not match trace context")
        if (
            str(session.status) != str(SessionStatus.RUNNING)
            or session.ended_at is not None
        ):
            raise ValueError("registered child session must start in running state")
        if session.capture_id not in {None, context.capture_id}:
            raise ValueError("child session capture_id must match trace context")
        context = replace(
            context,
            parent_actor_id=parent_context.actor_id,
            parent_actor_session_id=parent_context.actor_session_id,
        )
        actor = replace(
            actor,
            parent_actor_id=parent_context.actor_id,
            content_digest="",
        ).sealed()
        session = replace(
            session,
            capture_id=context.capture_id,
            parent_session_id=parent_context.actor_session_id,
            content_digest="",
        ).sealed()
        return actor, session, context

    def finish_child_session(
        self,
        capture_or_session_id: str,
        *,
        status: SessionStatus | str = SessionStatus.COMPLETED,
        ended_at: str | None = None,
    ) -> str:
        """Durably finish one registered or declaratively observed child session."""

        with self._lifecycle_lock:
            self._assert_capture_mutable()
            if capture_or_session_id in {
                self.binding.capture_id,
                self.binding.workload.actor_session_id,
            }:
                raise ValueError(
                    "root session lifecycle is owned by CaptureSupervisor.finalize"
                )
            matching_contexts = [
                context
                for context in self._contexts.values()
                if capture_or_session_id
                in {context.capture_id, context.actor_session_id}
                and context.capture_id != self.binding.capture_id
            ]
            if len(matching_contexts) > 1:
                raise ValueError("child session identifier is ambiguous")
            if matching_contexts:
                envelope_id, _, _ = self._finish_registered_context(
                    matching_contexts[0],
                    status=status,
                    ended_at=ended_at,
                )
                return envelope_id

            matching_sessions = [
                session
                for session in self._extra_sessions
                if capture_or_session_id
                in {session.session_id, session.capture_id}
            ]
            if len(matching_sessions) != 1:
                raise ValueError("child session is not registered or declared")
            session = matching_sessions[0]
            normalized = str(status)
            if normalized not in {
                str(SessionStatus.COMPLETED),
                str(SessionStatus.FAILED),
                str(SessionStatus.INTERRUPTED),
            }:
                raise ValueError("child session status must be terminal")
            existing = self._declared_session_terminals.get(session.session_id)
            if existing is not None:
                prior_status, prior_ended_at, envelope_id = existing
                if prior_status != normalized or (
                    ended_at is not None and prior_ended_at != ended_at
                ):
                    raise ValueError(
                        "child session is already finished with a conflicting "
                        "terminal fact"
                    )
                return envelope_id
            terminal_at = ended_at or utc_now()
            self._validate_declared_terminal_time(session, terminal_at)
            envelope_id, terminal_at = self.collector.finish_session(
                status=normalized,
                actor_id=session.actor_id,
                session_id=session.session_id,
                ended_at=terminal_at,
            )
            self._declared_session_terminals[session.session_id] = (
                normalized,
                terminal_at,
                envelope_id,
            )
            return envelope_id

    def _finish_remote_context(
        self,
        context: TraceContextV1,
        status: SessionStatus | str,
        ended_at: str | None,
    ) -> tuple[str, str, str]:
        """Finish an HTTP child under the supervisor's complete topology lock."""

        with self._lifecycle_lock:
            try:
                self._assert_capture_mutable()
            except RuntimeError as exc:
                raise SessionActivityError(str(exc)) from exc
            return self._finish_registered_context(
                context,
                status=status,
                ended_at=ended_at,
            )

    def _finish_registered_context(
        self,
        context: TraceContextV1,
        *,
        status: SessionStatus | str,
        ended_at: str | None,
    ) -> tuple[str, str, str]:
        if context.capture_id == self.binding.capture_id:
            raise ValueError(
                "root session lifecycle is owned by CaptureSupervisor.finalize"
            )
        if self._contexts.get(context.capture_id) != context:
            raise ValueError("child context is not registered")
        session = next(
            (
                item
                for item in self._extra_sessions
                if item.session_id == context.actor_session_id
            ),
            None,
        )
        if session is None:
            raise ValueError("registered child session identity is missing")
        normalized = str(status)
        if normalized not in {
            str(SessionStatus.COMPLETED),
            str(SessionStatus.FAILED),
            str(SessionStatus.INTERRUPTED),
        }:
            raise ValueError("child session status must be terminal")
        existing = self.collector_server.terminal_context_fact(
            context.capture_id
        )
        if existing is not None:
            prior_status, prior_ended_at, envelope_id = existing
            if prior_status != normalized or (
                ended_at is not None and prior_ended_at != ended_at
            ):
                raise ValueError(
                    "child session is already finished with a conflicting "
                    "terminal fact"
                )
            return envelope_id, prior_status, prior_ended_at
        terminal_at = ended_at or utc_now()
        self._validate_declared_terminal_time(session, terminal_at)
        return self.collector_server.finish_context(
            context,
            status=normalized,
            ended_at=terminal_at,
        )

    def _install_child_identity(
        self,
        actor: ActorV5,
        session: SessionV5,
        *,
        context: TraceContextV1 | None,
        persist: bool,
    ) -> str:
        existing_actor = next(
            (item for item in self._extra_actors if item.actor_id == actor.actor_id),
            None,
        )
        if (
            existing_actor is not None
            and existing_actor.content_digest != actor.content_digest
        ):
            raise ValueError("child actor id is already registered with different facts")
        existing_session = next(
            (
                item
                for item in self._extra_sessions
                if item.session_id == session.session_id
            ),
            None,
        )
        if (
            existing_session is not None
            and existing_session.content_digest != session.content_digest
        ):
            raise ValueError(
                "child session id is already registered with different facts"
            )
        existing_context = (
            self._contexts.get(context.capture_id)
            if context is not None
            else None
        )
        if existing_context is not None and existing_context != context:
            raise ValueError(
                "child capture id is already registered with different facts"
            )
        if existing_session is not None:
            if context is not None and existing_context is None:
                raise ValueError(
                    "child session is already declared without this capture context"
                )
            return (
                self.collector_server.context_token(context.capture_id)
                if context is not None
                else ""
            )

        capability = ""
        newly_authorized = False
        if context is not None:
            capability = self.collector_server.register_context(
                context,
                started_at=session.started_at,
            )
            newly_authorized = existing_context is None
        try:
            if persist:
                self.collector.register_child(
                    actor=actor,
                    session=session,
                    context=context,
                )
        except BaseException:
            if context is not None and newly_authorized:
                self.collector_server.unregister_context(context)
            raise
        if existing_actor is None:
            self._extra_actors.append(actor)
        self._extra_sessions.append(session)
        if context is not None:
            self._contexts[context.capture_id] = context
        return capability

    def _unterminated_declared_descendants(
        self,
        session_id: str,
    ) -> tuple[str, ...]:
        descendants = self._declared_descendants(session_id)
        return tuple(
            child.session_id
            for child in descendants
            if str(child.status) == str(SessionStatus.RUNNING)
            and child.session_id not in self._declared_session_terminals
            and not (
                child.capture_id
                and self.collector_server.is_context_terminal(child.capture_id)
            )
        )

    def _declared_descendants(
        self,
        session_id: str,
    ) -> tuple[SessionV5, ...]:
        children_by_parent: dict[str, list[SessionV5]] = {}
        for session in self._extra_sessions:
            if session.parent_session_id:
                children_by_parent.setdefault(session.parent_session_id, []).append(
                    session
                )
        descendants: list[SessionV5] = []
        frontier = list(children_by_parent.get(session_id, ()))
        seen: set[str] = set()
        while frontier:
            child = frontier.pop()
            if child.session_id in seen:
                raise ValueError("declared child session topology contains a cycle")
            seen.add(child.session_id)
            descendants.append(child)
            frontier.extend(children_by_parent.get(child.session_id, ()))
        return tuple(descendants)

    def _validate_declared_terminal_time(
        self,
        session: SessionV5,
        ended_at: str,
    ) -> None:
        terminal_moment = _parse_session_timestamp(
            ended_at,
            field=f"session {session.session_id} ended_at",
        )
        started_moment = _parse_session_timestamp(
            session.started_at,
            field=f"session {session.session_id} started_at",
        )
        if terminal_moment < started_moment:
            raise ValueError("child session ended_at precedes started_at")
        unfinished = self._unterminated_declared_descendants(session.session_id)
        if unfinished:
            raise ValueError(
                "child session has unterminated descendants: "
                + ", ".join(sorted(unfinished))
            )
        for descendant in self._declared_descendants(session.session_id):
            descendant_ended_at: str | None = None
            declared_fact = self._declared_session_terminals.get(
                descendant.session_id
            )
            if declared_fact is not None:
                descendant_ended_at = declared_fact[1]
            elif descendant.capture_id:
                captured_fact = self.collector_server.terminal_context_fact(
                    descendant.capture_id
                )
                if captured_fact is not None:
                    descendant_ended_at = captured_fact[1]
            if descendant_ended_at is None:
                descendant_ended_at = descendant.ended_at
            if (
                descendant_ended_at is not None
                and _parse_session_timestamp(
                    descendant_ended_at,
                    field=f"descendant session {descendant.session_id} ended_at",
                )
                > terminal_moment
            ):
                raise ValueError(
                    "child session ended_at precedes a descendant terminal fact"
                )

    def _assert_capture_mutable(self) -> None:
        if (
            self.sealed is not None
            or self._finalization_started
            or self._terminal_resume_status is not None
        ):
            raise RuntimeError("capture is immutable after finalization begins")

    # -- sealing ------------------------------------------------------------------

    def finalize(
        self,
        *,
        status: TraceStatus | str = TraceStatus.COMPLETED,
        termination: TerminationV5 | None = None,
        child_exit_code: int | None = None,
    ) -> SealedCapture:
        if self._terminal_finalization is not None:
            normalized_status = str(self._terminal_finalization.status)
            termination = self._terminal_finalization.termination
            child_exit_code = self._terminal_finalization.child_exit_code
        else:
            normalized_status = str(status)
        if normalized_status not in {
            str(TraceStatus.COMPLETED),
            str(TraceStatus.FAILED),
            str(TraceStatus.INTERRUPTED),
        }:
            raise ValueError("finalized trace status must be terminal")
        request = (normalized_status, termination, child_exit_code)
        with self._lifecycle_lock:
            if self.sealed is not None:
                if self._finalization_error is not None:
                    raise CaptureNotReady(self._finalization_error)
                return self.sealed
            if self._finalizing:
                raise RuntimeError("capture finalization is already in progress")
            if (
                self._finalization_request is not None
                and self._finalization_request != request
            ):
                raise ValueError(
                    "finalization retry must use the original lifecycle inputs"
                )
            if not self._finalization_started:
                self._finalization_started = True
                self._finalization_request = request
                self.collector_server.freeze()
                self.proxy.freeze()
                self.collector.freeze()
            self._finalizing = True
        try:
            if not self._finalization_services_stopped:
                mitm_receipt = self._stop_capture_services(
                    reason=normalized_status
                )
                self._finalization_mitm_receipt = mitm_receipt
                self._finalization_mitm_failure = _mitm_lifecycle_failure(
                    mitm_receipt
                )
                self._finalization_services_stopped = True
            if (
                not self._finalization_egress_evaluated
                and self.config.egress_assertion is not None
            ):
                assertion: EgressAssertion | None = None
                egress_failure: str | None = None
                try:
                    assertion = self.config.egress_assertion()
                    if not assertion.passed:
                        egress_failure = (
                            "egress assertion failed: "
                            + ", ".join(assertion.violations)
                        )
                except BaseException as exc:
                    egress_failure = (
                        f"egress assertion raised {type(exc).__name__}: {exc}"
                    )
                self._finalization_assertion = assertion
                self._finalization_egress_failure = egress_failure
                self._finalization_egress_evaluated = True

            # The typed terminal record is the final raw authority. It is written
            # only after every transport has drained and the egress result is known.
            # Publication retries and fresh-process resumes reuse these exact
            # lifecycle, timestamp, and pre-finalizer coverage inputs.
            if self._finalization_captured_at is None:
                self._finalization_captured_at = utc_now()
            if self._finalization_receipt is None:
                receipt = self.proxy.apply_to_receipt(self.receipt)
                receipt = replace(
                    receipt,
                    direct_egress_asserted=bool(
                        self._finalization_assertion
                        and self._finalization_assertion.passed
                    ),
                    child_exit_code=child_exit_code,
                    completeness_reasons=tuple(
                        dict.fromkeys(
                            (
                                *receipt.completeness_reasons,
                                *(
                                    item
                                    for item in (
                                        self._finalization_mitm_failure,
                                        self._finalization_egress_failure,
                                    )
                                    if item
                                ),
                            )
                        )
                    ),
                    finalization_status="captured",
                    ended_at=self._finalization_captured_at,
                    content_digest="",
                )
                self._finalization_receipt = receipt.sealed()
            if self._terminal_finalization is None:
                finalization = CaptureFinalizationV1(
                    status=normalized_status,
                    termination=termination,
                    child_exit_code=child_exit_code,
                    captured_at=self._finalization_captured_at,
                    coverage_seed=self._finalization_receipt,
                    provenance=replace(
                        self.config.provenance,
                        captured_at=self._finalization_captured_at,
                    ),
                    identity=self.config.identity,
                    root_actor_name=self.config.root_actor_name,
                    root_actor_kind=str(self.config.root_actor_kind),
                    finalizer_name=FINALIZER_NAME,
                    finalizer_version=FINALIZER_VERSION,
                    aliases=tuple(self._aliases),
                    egress_assertion=(
                        _egress_assertion_payload(self._finalization_assertion)
                        if self._finalization_assertion is not None
                        else None
                    ),
                    mitm_lifecycle=(
                        self._finalization_mitm_receipt.to_dict()
                        if self._finalization_mitm_receipt is not None
                        else None
                    ),
                    egress_failure=self._finalization_egress_failure,
                    mitm_failure=self._finalization_mitm_failure,
                ).sealed()
                self.session.append(
                    RawRecordType.CAPTURE_FINISHED,
                    payload=finalization.to_dict(),
                    occurred_at=self._finalization_captured_at,
                    producer_version="synth-trace-supervisor/1",
                )
                self._terminal_finalization = finalization
                self._terminal_resume_status = normalized_status
            self.session.close()

            if (
                not self._finalization_mitm_checked
                and self._finalization_mitm_receipt is not None
            ):
                self.bundle.write_receipt(
                    "mitm-lifecycle",
                    self._finalization_mitm_receipt,
                )
                self._finalization_mitm_checked = True
            elif self._finalization_mitm_receipt is None:
                self._finalization_mitm_checked = True
            if (
                not self._finalization_egress_checked
                and (
                    self._finalization_assertion is not None
                    or self._finalization_egress_failure is not None
                )
            ):
                if self._finalization_assertion is not None:
                    self.bundle.write_receipt(
                        "egress-assertion",
                        self._finalization_assertion,
                    )
                else:
                    self.bundle.write_receipt(
                        "egress-assertion",
                        {
                            "passed": False,
                            "error": self._finalization_egress_failure,
                        },
                    )
                self._finalization_egress_checked = True
            elif (
                self._finalization_assertion is None
                and self._finalization_egress_failure is None
            ):
                self._finalization_egress_checked = True

            if (
                self._finalization_egress_failure
                and str(self.binding.capture.mode)
                == CaptureMode.REQUIRED_EGRESS_ASSERTED
            ):
                self._finalization_error = self._finalization_egress_failure
            elif (
                self._finalization_mitm_failure
                and self.receipt.readiness_ok
                and str(self.binding.capture.mode)
                in {CaptureMode.REQUIRED, CaptureMode.REQUIRED_EGRESS_ASSERTED}
            ):
                self._finalization_error = self._finalization_mitm_failure

            terminal_finalization = self._terminal_finalization
            if terminal_finalization is None:
                raise RuntimeError("capture finalization authority was not persisted")
            finalizer = TraceFinalizer(
                binding=self.binding,
                spool_root=self.bundle.trace_root(self.binding.trace_id),
                segments=self.spool.segments,
                provenance=terminal_finalization.provenance,
                identity=terminal_finalization.identity,
                root_actor_name=terminal_finalization.root_actor_name,
                root_actor_kind=terminal_finalization.root_actor_kind,
            )
            candidate = finalizer.seal(
                coverage=self._finalization_receipt,
                status=normalized_status,
                termination=termination,
                extra_actors=tuple(self._extra_actors),
                extra_sessions=tuple(self._extra_sessions),
                aliases=terminal_finalization.aliases,
            )
            self.bundle.write_trace(
                candidate.document,
                binding=self.binding,
                segments=candidate.segments,
            )
            self.bundle.write_receipt("capture-coverage", candidate.coverage)
            self.bundle.write_manifest()
            with self._lifecycle_lock:
                self.sealed = candidate
            if self._finalization_error is not None:
                raise CaptureNotReady(self._finalization_error)
            return candidate
        finally:
            with self._lifecycle_lock:
                self._finalizing = False

    def materialize_projection(self, kind: str) -> dict[str, Any]:
        """Publish a bound projection receipt without mutating sealed coverage."""

        if self.sealed is None:
            raise RuntimeError("capture must be finalized before projection")
        from ..adapters.atif import export_atif
        from ..models.projection import (
            ProjectionLossV1,
            ProjectionManifestV1,
            bind_projection_manifest,
        )
        from ..projections.v4 import project_v4
        from ..canonical import record_id

        normalized = kind.lower().strip()
        if normalized == "v4":
            payload, manifest = project_v4(self.sealed.document)
            projection_payload: Any = payload.to_dict()
        elif normalized == "atif":
            projection_payload = export_atif(self.sealed.document)
            losses = tuple(
                ProjectionLossV1(
                    field_path="*",
                    reason=str(item),
                    record_count=0,
                )
                for item in (
                    (projection_payload.get("extra") or {}).get("projection_losses")
                    or ()
                )
            )
            manifest = ProjectionManifestV1(
                projection_id=record_id(
                    "proj",
                    kind="atif",
                    scope=(self.sealed.document.trace_id,),
                    key=self.sealed.document.content_digest,
                ),
                format="atif",
                source_trace_id=self.sealed.document.trace_id,
                source_trace_digest=self.sealed.document.content_digest,
                producer="synth_containers.tracing.capture.supervisor",
                producer_version="1",
                created_at=utc_now(),
                losses=losses,
            )
        else:
            raise ValueError(f"unsupported supervisor projection {kind!r}")
        manifest = bind_projection_manifest(manifest, self.binding)
        path, sealed_manifest = self.bundle.write_projection(
            manifest,
            projection_payload,
            kind=normalized,
        )
        self.bundle.write_receipt(f"projection-{normalized}", sealed_manifest)
        self.bundle.write_manifest()
        return {
            "kind": normalized,
            "path": str(path),
            "manifest": sealed_manifest.to_dict(),
        }


def _actor_from_payload(payload: Mapping[str, Any]) -> ActorV5:
    values = dict(payload)
    values["external_trace_refs"] = tuple(
        values.get("external_trace_refs") or ()
    )
    values["aliases"] = tuple(
        AliasV1(**item) if isinstance(item, Mapping) else item
        for item in values.get("aliases") or ()
    )
    return replace(
        ActorV5(**values),
        content_digest="",
    ).sealed()


def _session_from_payload(payload: Mapping[str, Any]) -> SessionV5:
    values = dict(payload)
    coverage = values.get("coverage")
    if isinstance(coverage, Mapping):
        coverage_values = dict(coverage)
        coverage_values["reasons"] = tuple(coverage_values.get("reasons") or ())
        values["coverage"] = SessionCoverageV5(**coverage_values)
    values["aliases"] = tuple(
        AliasV1(**item) if isinstance(item, Mapping) else item
        for item in values.get("aliases") or ()
    )
    return replace(
        SessionV5(**values),
        content_digest="",
    ).sealed()


def _parse_session_timestamp(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an RFC3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed


def _default_provider_endpoints(
    config: SupervisorConfig,
) -> tuple[ProviderEndpointConfig, ...]:
    openai_auth = (
        UpstreamAuthKind.BEARER
        if config.upstream_api_key
        else UpstreamAuthKind.PASSTHROUGH
    )
    endpoints = [
        ProviderEndpointConfig(
            route=CHAT_COMPLETIONS_PATH,
            adapter_name="openai_chat_completions",
            upstream_base_url=config.upstream_base_url,
            upstream_path="/chat/completions",
            auth_kind=openai_auth,
            api_key=config.upstream_api_key,
        ),
        ProviderEndpointConfig(
            route=RESPONSES_PATH,
            adapter_name="openai_responses",
            upstream_base_url=config.upstream_base_url,
            upstream_path="/responses",
            auth_kind=openai_auth,
            api_key=config.upstream_api_key,
        ),
        ProviderEndpointConfig(
            route=RESPONSES_COMPACT_PATH,
            adapter_name="openai_responses",
            upstream_base_url=config.upstream_base_url,
            upstream_path="/responses/compact",
            auth_kind=openai_auth,
            api_key=config.upstream_api_key,
        ),
    ]
    if config.anthropic_base_url:
        endpoints.append(
            ProviderEndpointConfig(
                route=ANTHROPIC_MESSAGES_PATH,
                adapter_name="anthropic_messages",
                upstream_base_url=config.anthropic_base_url,
                auth_kind=(
                    UpstreamAuthKind.HEADER
                    if config.anthropic_api_key
                    else UpstreamAuthKind.PASSTHROUGH
                ),
                auth_header="x-api-key",
                auth_scheme="",
                api_key=config.anthropic_api_key,
            )
        )
    return tuple(endpoints)


def _terminal_resume_provider_endpoints() -> tuple[ProviderEndpointConfig, ...]:
    """Inert endpoint metadata for resealing an already terminal raw authority."""

    return (
        ProviderEndpointConfig(
            route=CHAT_COMPLETIONS_PATH,
            adapter_name="openai_chat_completions",
            upstream_base_url="https://terminal-resume.invalid/v1",
            upstream_path="/chat/completions",
            auth_kind=UpstreamAuthKind.NONE,
        ),
    )


def _websocket_url(http_url: str) -> str:
    parsed = urlsplit(http_url)
    schemes = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}
    scheme = schemes.get(parsed.scheme.lower())
    if scheme is None:
        raise ValueError(
            f"Responses WebSocket upstream requires HTTP(S) or WS(S), got {http_url!r}"
        )
    return urlunsplit(
        (scheme, parsed.netloc, parsed.path, parsed.query, parsed.fragment)
    )


def _mitm_lifecycle_failure(
    receipt: MitmLifecycleReceiptV1 | None,
) -> str | None:
    if receipt is None:
        return None
    reasons: list[str] = []
    if not receipt.readiness_ok:
        reasons.append("scoped MITM never proved readiness")
    if receipt.failure:
        reasons.append(receipt.failure)
    if receipt.process_exited_before_stop:
        reasons.append("mitmdump exited before capture finalization")
    if not receipt.process_stopped:
        reasons.append("mitmdump did not stop during capture finalization")
    if receipt.unmapped_provider_requests:
        reasons.append(
            f"{receipt.unmapped_provider_requests} allowlisted provider request(s) "
            "had no capture route"
        )
    if receipt.unexpected_tls_hosts:
        reasons.append(
            f"{receipt.unexpected_tls_hosts} undeclared TLS host(s) reached the addon"
        )
    if receipt.malformed_addon_events:
        reasons.append(
            f"{receipt.malformed_addon_events} malformed MITM lifecycle event(s)"
        )
    if not receipt.private_key_destroyed or not receipt.confdir_destroyed:
        reasons.append("ephemeral MITM private CA material was not destroyed")
    if not reasons:
        return None
    return "scoped MITM lifecycle incomplete: " + "; ".join(dict.fromkeys(reasons))


def _egress_assertion_from_payload(payload: Mapping[str, Any]) -> EgressAssertion:
    assertion = EgressAssertion(
        allowed_hosts=tuple(payload.get("allowed_hosts") or ()),
        observed_hosts=tuple(payload.get("observed_hosts") or ()),
        violations=tuple(payload.get("violations") or ()),
    )
    if "passed" not in payload or bool(payload.get("passed")) != assertion.passed:
        raise ValueError("egress assertion passed flag is inconsistent")
    return assertion


def _egress_assertion_payload(assertion: EgressAssertion) -> dict[str, Any]:
    return {
        "allowed_hosts": list(assertion.allowed_hosts),
        "observed_hosts": list(assertion.observed_hosts),
        "violations": list(assertion.violations),
        "passed": assertion.passed,
    }


def _mitm_receipt_from_payload(
    payload: Mapping[str, Any],
) -> MitmLifecycleReceiptV1:
    values = dict(payload)
    for name in (
        "allowed_hosts",
        "allowed_authorities",
        "private_key_names",
    ):
        values[name] = tuple(values.get(name) or ())
    receipt = MitmLifecycleReceiptV1(**values)
    if receipt.schema_version != MITM_LIFECYCLE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported MITM lifecycle schema: {receipt.schema_version}"
        )
    if receipt.content_digest != content_digest(receipt):
        raise ValueError("MITM lifecycle receipt digest mismatch")
    return receipt


def _load_resume_binding(
    bundle: LocalTraceBundle,
    *,
    capture_id: str,
    trace_id: str | None,
) -> tuple[TraceCaptureBindingV1, Path]:
    """Load and re-digest the one existing binding selected for resume."""

    from ..validation.rehydrate import build

    candidates = (
        [bundle.trace_root(trace_id) / "binding.json"]
        if trace_id
        else sorted((bundle.root / "traces").glob("*/binding.json"))
    )
    matches: list[tuple[TraceCaptureBindingV1, Path]] = []
    for path in candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        binding = build(TraceCaptureBindingV1, payload)
        if not isinstance(binding, TraceCaptureBindingV1):
            continue
        if binding.capture_id != capture_id:
            continue
        if binding.content_digest != content_digest(binding):
            raise ValueError(f"resume binding digest mismatch at {path}")
        matches.append((binding, path))
    if not matches:
        raise FileNotFoundError(
            f"no existing capture binding for capture_id {capture_id!r}"
        )
    if len(matches) != 1:
        raise ValueError(f"capture_id {capture_id!r} selects multiple bindings")
    return matches[0]


def _load_external_binding(path: Path) -> TraceCaptureBindingV1:
    """Load one pre-launch binding and prove its declared digest."""

    from ..validation.rehydrate import build

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    binding = build(TraceCaptureBindingV1, payload)
    if not isinstance(binding, TraceCaptureBindingV1):
        raise ValueError(f"binding at {path} has the wrong schema")
    if not binding.content_digest:
        raise ValueError(f"binding at {path} is not sealed")
    if binding.content_digest != content_digest(binding):
        raise ValueError(f"binding digest mismatch at {path}")
    return binding


__all__ = ["CaptureNotReady", "CaptureSupervisor", "SupervisorConfig"]
