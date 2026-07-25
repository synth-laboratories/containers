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
import json
from pathlib import Path
import secrets
from types import TracebackType
from typing import Any
from collections.abc import Callable
import uuid

import httpx

from ..canonical import content_digest, utc_now
from ..models.actors import ActorKind, ActorV5, SessionCoverageV5, SessionV5
from ..models.completeness import TerminationV5, TraceStatus
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
from .collector_server import CollectorServer
from .coverage import new_coverage_receipt
from .finalizer import SealedCapture, TraceFinalizer
from .proxy import CaptureProxy
from .proxy import (
    ANTHROPIC_MESSAGES_PATH,
    CHAT_COMPLETIONS_PATH,
    RESPONSES_COMPACT_PATH,
    RESPONSES_PATH,
)
from .routes import ProviderEndpointConfig, UpstreamAuthKind
from .egress import EgressAssertion, mitm_environment
from .mitm import MitmLifecycleReceiptV1, MitmStartupError, ScopedMitmProxy
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
        if str(self.binding.capture.mode) == CaptureMode.OBSERVE_AND_TRANSFORM:
            raise ValueError(
                "observe_and_transform capture requires an explicit versioned "
                "transformation specification; transformation is not implemented"
            )
        if (
            str(self.binding.capture.mode) == CaptureMode.REQUIRED_EGRESS_ASSERTED
            and config.egress_assertion is None
        ):
            raise ValueError(
                "required_egress_asserted mode requires an egress_assertion callback"
            )
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
        self._endpoints = config.provider_endpoints or _default_provider_endpoints(config)
        self._collector_token = secrets.token_urlsafe(32)
        root_context = self.binding.context_for_child()
        self._contexts: dict[str, TraceContextV1] = {
            root_context.capture_id: root_context
        }
        self.proxy = CaptureProxy(
            self.session,
            upstream_base_url=config.upstream_base_url,
            upstream_api_key=config.upstream_api_key,
            provider_endpoints=self._endpoints,
            host=config.proxy_host,
            port=config.proxy_port,
            context_resolver=self._resolve_provider_context,
        )
        self.collector = LocalCollector(self.session)
        self.websocket_server = (
            ResponsesWebSocketServer(
                ResponsesWebSocketRelay(
                    self.session,
                    authorization=(
                        f"Bearer {config.upstream_api_key}"
                        if config.upstream_api_key
                        else None
                    ),
                ),
                host=config.websocket_host,
                port=config.websocket_port,
                context_resolver=self._resolve_provider_context,
            )
            if config.responses_websocket
            else None
        )
        self.collector_server = CollectorServer(
            self.collector,
            host=config.collector_host,
            port=config.collector_port,
            collector_token=self._collector_token,
            on_register_context=self._register_remote_context,
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
        )
        self.sealed: SealedCapture | None = None
        self._extra_actors: list[ActorV5] = []
        self._extra_sessions: list[SessionV5] = []
        self._aliases: list[AliasV1] = []
        self._capture_operational = False
        self._startup_failure: str | None = None

    def _resolve_provider_context(
        self,
        headers: Any,
    ) -> TraceContextV1 | None:
        normalized = {str(name).lower(): str(value) for name, value in headers.items()}
        capture_id = normalized.get("x-synth-capture-id")
        if not capture_id:
            return None
        token = normalized.get("x-synth-context-token") or ""
        if not secrets.compare_digest(token, self._collector_token):
            return None
        context = self._contexts.get(capture_id)
        if context is None:
            return None
        expected = {
            "x-synth-trace-id": context.trace_id,
            "x-synth-actor-id": context.actor_id,
            "x-synth-session-id": context.actor_session_id,
        }
        if any(normalized.get(name) != value for name, value in expected.items()):
            return None
        return context

    def _register_remote_context(
        self,
        context: TraceContextV1,
        actor_payload: dict[str, Any],
        session_payload: dict[str, Any],
    ) -> None:
        actor_values = dict(actor_payload)
        actor_values["aliases"] = tuple(
            AliasV1(**item) if isinstance(item, dict) else item
            for item in actor_values.get("aliases") or ()
        )
        session_values = dict(session_payload)
        coverage = session_values.get("coverage")
        if isinstance(coverage, dict):
            session_values["coverage"] = SessionCoverageV5(**coverage)
        session_values["aliases"] = tuple(
            AliasV1(**item) if isinstance(item, dict) else item
            for item in session_values.get("aliases") or ()
        )
        self.register_child_context(
            context,
            actor=ActorV5(**actor_values),
            session=SessionV5(**session_values),
        )

    # -- lifecycle ---------------------------------------------------------------

    def __enter__(self) -> "CaptureSupervisor":
        return self.start_capture()

    def start_capture(self) -> "CaptureSupervisor":
        """Start the selected capture mechanisms and enforce mode semantics."""

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
        if mitm_receipt is not None:
            self.bundle.write_receipt("mitm-lifecycle", mitm_receipt)
        if stop_errors:
            detail = "capture service shutdown errors: " + "; ".join(stop_errors)
            self.receipt = replace(
                self.receipt,
                completeness_reasons=(
                    *self.receipt.completeness_reasons,
                    detail,
                ),
            )
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
            "x-synth-context-token": self._collector_token,
        }

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

        self._extra_actors.append(actor.sealed())
        if session is not None:
            self._extra_sessions.append(session.sealed())

    def declare_alias(self, alias: AliasV1) -> None:
        self._aliases.append(alias)

    def register_child_context(
        self,
        context: TraceContextV1,
        *,
        actor: ActorV5,
        session: SessionV5,
    ) -> None:
        """Authorize one delegated child capture and include its identity in sealing."""

        if actor.actor_id != context.actor_id or session.session_id != context.actor_session_id:
            raise ValueError("child actor/session do not match trace context")
        if session.actor_id != actor.actor_id:
            raise ValueError("child session must belong to child actor")
        self._contexts[context.capture_id] = context
        self.collector_server.register_context(context)
        if not any(item.actor_id == actor.actor_id for item in self._extra_actors):
            self.declare_actor(actor, session)
        elif not any(item.session_id == session.session_id for item in self._extra_sessions):
            self._extra_sessions.append(session.sealed())

    # -- sealing ------------------------------------------------------------------

    def finalize(
        self,
        *,
        status: TraceStatus | str = TraceStatus.COMPLETED,
        termination: TerminationV5 | None = None,
        child_exit_code: int | None = None,
    ) -> SealedCapture:
        if self.sealed is not None:
            return self.sealed
        mitm_receipt = self._stop_capture_services(reason=str(status))
        mitm_failure = _mitm_lifecycle_failure(mitm_receipt)
        egress_failure: str | None = None
        assertion: EgressAssertion | None = None
        if self.config.egress_assertion is not None:
            try:
                assertion = self.config.egress_assertion()
                self.bundle.write_receipt("egress-assertion", assertion)
                if not assertion.passed:
                    egress_failure = (
                        "egress assertion failed: "
                        + ", ".join(assertion.violations)
                    )
            except BaseException as exc:
                egress_failure = f"egress assertion raised {type(exc).__name__}: {exc}"
                self.bundle.write_receipt(
                    "egress-assertion",
                    {
                        "passed": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
        # Seal the raw authority regardless of the assertion outcome. Required
        # mode reports failure only after the interrupted/partial evidence packet
        # has been durably written.
        self.spool.close()
        receipt = self.proxy.apply_to_receipt(self.receipt)
        receipt = replace(
            receipt,
            direct_egress_asserted=bool(assertion and assertion.passed),
            child_exit_code=child_exit_code,
            completeness_reasons=tuple(
                dict.fromkeys(
                    (
                        *receipt.completeness_reasons,
                        *(item for item in (mitm_failure, egress_failure) if item),
                    )
                )
            ),
        )
        finalizer = TraceFinalizer(
            binding=self.binding,
            spool_root=self.bundle.trace_root(self.binding.trace_id),
            segments=self.spool.segments,
            provenance=replace(self.config.provenance, captured_at=utc_now()),
            identity=self.config.identity,
            root_actor_name=self.config.root_actor_name,
            root_actor_kind=self.config.root_actor_kind,
        )
        self.sealed = finalizer.seal(
            coverage=receipt,
            status=status,
            termination=termination,
            extra_actors=tuple(self._extra_actors),
            extra_sessions=tuple(self._extra_sessions),
            aliases=tuple(self._aliases),
        )
        self.bundle.write_trace(
            self.sealed.document,
            binding=self.binding,
            segments=self.sealed.segments,
        )
        self.bundle.write_receipt("capture-coverage", self.sealed.coverage)
        self.bundle.write_manifest()
        if (
            egress_failure
            and str(self.binding.capture.mode) == CaptureMode.REQUIRED_EGRESS_ASSERTED
        ):
            raise CaptureNotReady(egress_failure)
        if (
            mitm_failure
            and self.receipt.readiness_ok
            and str(self.binding.capture.mode)
            in {CaptureMode.REQUIRED, CaptureMode.REQUIRED_EGRESS_ASSERTED}
        ):
            raise CaptureNotReady(mitm_failure)
        return self.sealed

    def materialize_projection(self, kind: str) -> dict[str, Any]:
        """Write a declared compatibility projection and link it from coverage."""

        if self.sealed is None:
            raise RuntimeError("capture must be finalized before projection")
        from ..adapters.atif import export_atif
        from ..models.projection import ProjectionLossV1, ProjectionManifestV1
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
        path, sealed_manifest = self.bundle.write_projection(
            manifest,
            projection_payload,
            kind=normalized,
        )
        self.bundle.write_receipt(f"projection-{normalized}", sealed_manifest)
        updated_coverage = replace(
            self.sealed.coverage,
            projection_refs=tuple(
                sorted(
                    {
                        *self.sealed.coverage.projection_refs,
                        sealed_manifest.content_digest,
                    }
                )
            ),
            content_digest="",
        ).sealed()
        self.sealed = replace(self.sealed, coverage=updated_coverage)
        self.bundle.write_receipt("capture-coverage", updated_coverage)
        self.bundle.write_manifest()
        return {
            "kind": normalized,
            "path": str(path),
            "manifest": sealed_manifest.to_dict(),
        }


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
