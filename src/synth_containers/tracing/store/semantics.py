"""Attach verified interpretation sidecars to existing portable bundles."""

from __future__ import annotations

from ..canonical import content_digest
from ..profiles import TraceSemanticsV1
from .bundle import LocalTraceBundle


def write_trace_semantics(bundle: LocalTraceBundle, profile: TraceSemanticsV1) -> str:
    profile = TraceSemanticsV1.model_validate(profile.model_dump(mode="python"))
    trace = bundle.read_trace(profile.trace_digest)
    events = {item["event_id"] for item in trace.get("events") or ()}
    if any(item.event_id not in events for item in profile.timings):
        raise ValueError("timestamp sidecar references a foreign event")
    artifacts = {item["artifact_id"] for item in trace.get("artifacts") or ()}
    for body in profile.bodies:
        if body.artifact_id not in artifacts:
            raise ValueError("body sidecar references a foreign artifact")
        if body.retrievable:
            assert body.stored_digest is not None
            content = bundle.blobs.get(body.stored_digest)
            if body.retained_bytes is None or len(content) != body.retained_bytes:
                raise ValueError("body retained-byte claim does not match verified bytes")
            if body.disposition == "complete" and body.expected_bytes is not None and len(content) != body.expected_bytes:
                raise ValueError("complete body contradicts expected bytes")
            if body.disposition == "encrypted" and body.readable:
                raise ValueError("encrypted readability requires an owner-side authorized decryption receipt")
    value = profile.model_dump(mode="json")
    digest = content_digest(value)
    bundle.write_receipt(f"trace-semantics-{digest[7:]}", value)
    return digest
