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
from pathlib import Path
from types import TracebackType
from typing import Any

import httpx

from ..canonical import utc_now
from ..models.actors import ActorKind, ActorV5, SessionV5
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
from .coverage import new_coverage_receipt
from .finalizer import SealedCapture, TraceFinalizer
from .proxy import CaptureProxy
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


class CaptureSupervisor:
    """Context manager that runs one capture session and seals one bundle."""

    def __init__(self, config: SupervisorConfig) -> None:
        self.config = config
        self.bundle = LocalTraceBundle(config.bundle_root)
        trace_id = mint_trace_id(kind=str(config.trace_kind), key=config.trace_key)
        capture_id = mint_capture_id(trace_id=trace_id, key=config.trace_key)
        actor_id = mint_actor_id(trace_id=trace_id, name=config.root_actor_name)
        session_id = mint_session_id(trace_id=trace_id, actor_id=actor_id)
        self.binding: TraceCaptureBindingV1 = mint_binding(
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
        capture_root = self.bundle.capture_root(trace_id)
        self.binding_path = self.bundle.write_binding(self.binding)
        self.spool = RawSpool(
            capture_root,
            capture_id=capture_id,
            max_segment_records=config.policy.max_segment_records,
        )
        self.session = CaptureSession(
            binding=self.binding,
            spool=self.spool,
            blob_root=self.bundle.root / "blobs",
        )
        self.proxy = CaptureProxy(
            self.session,
            upstream_base_url=config.upstream_base_url,
            upstream_api_key=config.upstream_api_key,
            host=config.proxy_host,
            port=config.proxy_port,
        )
        self.collector = LocalCollector(self.session)
        self.receipt = new_coverage_receipt(
            binding_id=self.binding.binding_id,
            binding_digest=self.binding.content_digest,
            capture_id=capture_id,
            scope="model_calls_and_application",
            requested_mode=str(config.mode),
            resolved_mode=str(config.mode),
            interception=str(config.interception),
            proxy_config_digest=self.binding.capture.proxy_config_digest or "",
        )
        self.sealed: SealedCapture | None = None
        self._extra_actors: list[ActorV5] = []
        self._extra_sessions: list[SessionV5] = []
        self._aliases: list[AliasV1] = []

    # -- lifecycle ---------------------------------------------------------------

    def __enter__(self) -> "CaptureSupervisor":
        return self.start_capture()

    def start_capture(self) -> "CaptureSupervisor":
        """Start the proxy and prove readiness. Required modes fail closed here."""

        self.proxy.start()
        readiness = self._probe()
        self.receipt = replace(
            self.receipt,
            registration_ok=True,
            readiness_ok=readiness,
            reachability_detail=(
                "proxy /healthz answered from the launching namespace"
                if readiness
                else "proxy /healthz did not answer"
            ),
            injected_variables=("OPENAI_BASE_URL",),
        )
        if not readiness and str(self.config.mode) in {
            CaptureMode.REQUIRED,
            CaptureMode.REQUIRED_EGRESS_ASSERTED,
        }:
            self.proxy.stop(reason="readiness_failed")
            raise CaptureNotReady(
                f"capture mode {self.config.mode} requires a reachable proxy at "
                f"{self.proxy.base_url}"
            )
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

    def _probe(self) -> bool:
        try:
            response = httpx.get(f"{self.proxy.base_url}/healthz", timeout=10.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

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

    def declare_actor(self, actor: ActorV5, session: SessionV5 | None = None) -> None:
        """Add a non-root actor observed by the workload, such as the environment."""

        self._extra_actors.append(actor.sealed())
        if session is not None:
            self._extra_sessions.append(session.sealed())

    def declare_alias(self, alias: AliasV1) -> None:
        self._aliases.append(alias)

    # -- sealing ------------------------------------------------------------------

    def finalize(
        self,
        *,
        status: TraceStatus | str = TraceStatus.COMPLETED,
        termination: TerminationV5 | None = None,
    ) -> SealedCapture:
        if self.sealed is not None:
            return self.sealed
        self.proxy.stop(reason=str(status))
        self.spool.close()
        receipt = self.proxy.apply_to_receipt(self.receipt)
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
        return self.sealed


__all__ = ["CaptureNotReady", "CaptureSupervisor", "SupervisorConfig"]
