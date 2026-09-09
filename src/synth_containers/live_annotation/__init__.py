"""Live incremental annotation protocols.

A protocol is caller-supplied, digest-pinned code that runs *beside* a rollout
in its own process. It consumes the rollout's durable event stream as the
stream grows and emits a second durable stream of provisional annotations:
achievements as they unlock, milestones as they are reached, failure modes as
they are detected, plus bounded model-assisted judgments.

The protocol never sees the acting policy and the policy never sees the
protocol. Its output is observe-only evidence: it cannot change reward,
achievements, terminal status, or the rollout trace, and it is labelled
``provisional`` until a post-hoc annotator confirms it against the sealed
trace. See ``docs/specs/live-annotation-protocol-v1.md``.
"""

from .contract import (
    PROTOCOL_SCHEMA,
    STREAM_KINDS,
    ProtocolRevision,
    protocol_revision_id,
)
from .process import IsolatedProtocolProcess
from .runner import LiveAnnotationRunner, RunnerLimits
from .service import LiveAnnotationService

__all__ = [
    "PROTOCOL_SCHEMA",
    "STREAM_KINDS",
    "IsolatedProtocolProcess",
    "LiveAnnotationRunner",
    "LiveAnnotationService",
    "ProtocolRevision",
    "RunnerLimits",
    "protocol_revision_id",
]
