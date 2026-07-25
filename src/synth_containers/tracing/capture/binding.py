"""``TraceCaptureBindingV1`` — the immutable capture identity minted before launch.

The binding is what proves a model request, a local application event, and an output
artifact came from the same process and capture session. It is minted locally, needs
no backend, and cannot be mutated by the workload that reads it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from synth_containers.serde import JsonDataclassMixin

from ..canonical import canonical_text, content_digest, record_id, utc_now
from ..models.identity import TraceContextV1, TraceKind


BINDING_SCHEMA_VERSION = "synth.trace-capture-binding.v1"
BINDING_FILE_NAME = "binding.json"


class CaptureMode(StrEnum):
    """How strictly capture is enforced. Strictness is separate from mechanism."""

    DISABLED = "disabled"
    BEST_EFFORT = "best_effort"
    REQUIRED = "required"
    REQUIRED_EGRESS_ASSERTED = "required_egress_asserted"
    OBSERVE_AND_TRANSFORM = "observe_and_transform"


class Interception(StrEnum):
    PROVIDER_PROXY = "provider_proxy"
    TLS_MITM = "tls_mitm"
    BOTH = "both"


class WorkloadKind(StrEnum):
    REACT = "react"
    CODEX = "codex"
    JESTERKY = "jesterky"
    EVALUATOR = "evaluator"
    OTHER = "other"


class TokenCaptureLevel(StrEnum):
    NONE = "none"
    USAGE_ONLY = "usage_only"
    COMPLETION = "completion"
    FULL_TRAINING = "full_training"


@dataclass(frozen=True, slots=True)
class CapturePolicyV1(JsonDataclassMixin):
    """Resolved capture feature policy. The profile name alone is never sufficient."""

    profile: str = "eval_replayable"
    canonical_capture: bool = True
    raw_capture: str = "artifact"
    reasoning_policy: str = "capture_when_exposed"
    tool_output_policy: str = "artifact_over_limit"
    token_level: TokenCaptureLevel | str = TokenCaptureLevel.USAGE_ONLY
    logit_policy: str = "none"
    redaction_profile: str = "strict_headers_and_secrets"
    classification: str = "private"
    retention_class: str = "local_only"
    max_inline_bytes: int = 65536
    max_segment_records: int = 512
    compression: str = "none"

    def digest(self) -> str:
        return content_digest(self)


@dataclass(frozen=True, slots=True)
class BindingContainerV1(JsonDataclassMixin):
    container_definition_id: str | None = None
    contract_version: str | None = None
    contract_digest: str | None = None
    image_ref: str | None = None
    image_digest: str | None = None
    build_source_revision: str | None = None
    runtime_id: str | None = None
    runtime_instance_id: str | None = None


@dataclass(frozen=True, slots=True)
class BindingWorkloadV1(JsonDataclassMixin):
    kind: WorkloadKind | str
    root_actor_id: str
    actor_session_id: str
    parent_actor_id: str | None = None
    parent_actor_session_id: str | None = None
    parent_span_id: str | None = None
    delegation_id: str | None = None
    process_id: str | None = None
    run_id: str | None = None
    rollout_id: str | None = None
    session_id: str | None = None
    workflow_id: str | None = None
    workflow_address: str | None = None


@dataclass(frozen=True, slots=True)
class BindingContextV1(JsonDataclassMixin):
    task_id: str | None = None
    task_instance_id: str | None = None
    benchmark: str | None = None
    seed: int | None = None
    external_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BindingCaptureV1(JsonDataclassMixin):
    interception: Interception | str = Interception.PROVIDER_PROXY
    mode: CaptureMode | str = CaptureMode.REQUIRED
    proxy_profile: str = "openai_chat_completions"
    proxy_config_digest: str | None = None
    registration_id: str | None = None
    registration_expires_at: str | None = None
    output_artifact_root: str = ""


@dataclass(frozen=True, slots=True)
class TraceCaptureBindingV1(JsonDataclassMixin):
    binding_id: str
    trace_id: str
    capture_id: str
    trace_kind: TraceKind | str
    policy: CapturePolicyV1
    workload: BindingWorkloadV1
    capture: BindingCaptureV1
    created_at: str
    container: BindingContainerV1 = field(default_factory=BindingContainerV1)
    context: BindingContextV1 = field(default_factory=BindingContextV1)
    schema_target: str = "synth.trace.v5"
    schema_version: str = BINDING_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    content_digest: str = ""

    def sealed(self) -> "TraceCaptureBindingV1":
        return replace(self, content_digest=content_digest(self))

    def context_for_child(
        self,
        *,
        collector_url: str | None = None,
        binding_path: str | None = None,
        output_dir: str | None = None,
    ) -> TraceContextV1:
        return TraceContextV1(
            trace_id=self.trace_id,
            capture_id=self.capture_id,
            actor_id=self.workload.root_actor_id,
            actor_session_id=self.workload.actor_session_id,
            parent_actor_id=self.workload.parent_actor_id,
            parent_actor_session_id=self.workload.parent_actor_session_id,
            parent_span_id=self.workload.parent_span_id,
            delegation_id=self.workload.delegation_id,
            workflow_address=self.workload.workflow_address,
            binding_path=binding_path,
            collector_url=collector_url,
            output_dir=output_dir,
        )

    def write(self, directory: Path) -> Path:
        """Materialize the binding as a read-only file the workload may read."""

        directory.mkdir(parents=True, exist_ok=True)
        path = directory / BINDING_FILE_NAME
        path.write_text(canonical_text(self) + "\n", encoding="utf-8")
        path.chmod(0o444)
        return path

    @classmethod
    def read(cls, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))


def mint_binding(
    *,
    trace_id: str,
    capture_id: str,
    workload: BindingWorkloadV1,
    capture: BindingCaptureV1,
    policy: CapturePolicyV1 | None = None,
    container: BindingContainerV1 | None = None,
    context: BindingContextV1 | None = None,
    trace_kind: TraceKind | str = TraceKind.AGENT_ROLLOUT,
    metadata: dict[str, Any] | None = None,
) -> TraceCaptureBindingV1:
    """Mint and seal a capture binding. No secrets ever enter a binding."""

    resolved_policy = policy or CapturePolicyV1()
    resolved_capture = replace(capture, proxy_config_digest=resolved_policy.digest())
    binding_id = record_id(
        "bind",
        kind="capture_binding",
        scope=(trace_id, capture_id),
        key={
            "workload": workload.to_dict(),
            "capture": resolved_capture.to_dict(),
            "policy": resolved_policy.to_dict(),
        },
    )
    binding = TraceCaptureBindingV1(
        binding_id=binding_id,
        trace_id=trace_id,
        capture_id=capture_id,
        trace_kind=trace_kind,
        policy=resolved_policy,
        workload=workload,
        capture=resolved_capture,
        created_at=utc_now(),
        container=container or BindingContainerV1(),
        context=context or BindingContextV1(),
        metadata=dict(metadata or {}),
    )
    return binding.sealed()


__all__ = [
    "BINDING_FILE_NAME",
    "BINDING_SCHEMA_VERSION",
    "BindingCaptureV1",
    "BindingContainerV1",
    "BindingContextV1",
    "BindingWorkloadV1",
    "CaptureMode",
    "CapturePolicyV1",
    "Interception",
    "TokenCaptureLevel",
    "TraceCaptureBindingV1",
    "WorkloadKind",
    "mint_binding",
]
