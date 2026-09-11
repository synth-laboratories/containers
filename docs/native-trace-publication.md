# Native Trace V5 publication adapter

`native_trace_publication.publish_native_trace_evidence` is a composable async
publication callback for native lifecycle owners. It accepts an existing sealed
bundle and the public SDK `AsyncFactoryTraceStoreAPI` object. It uses that object's
existing prepare/upload/finalize protocol; it does not receive provider storage
credentials or duplicate Artifact Platform authorization.

```python
receipt = await publish_native_trace_evidence(
    captured_bundle, publication_output,
    store=factory_trace_store,
    run_id=run_id,
    expected_manifest_digest=manifest_digest,
    expected_trace_digests=(trace_digest,),
    required_visual_digests=required_frame_digests,
    max_bytes=64 * 1024 * 1024,
    timeout_seconds=120,
)
```

The caller supplies exact manifest/trace identities from its capture authority.
The adapter collects only declared immutable objects into a bounded exclusive
snapshot, preserving folder layout, then validates the copied bundle through
existing Trace V5 inspection. It refuses mismatched identities, non-self-contained
objects, secret-bearing sealed documents, incomplete capture on the normal path,
and missing required visual attachments. Native PNG events must resolve to
embedded artifacts. Required image/video digests must resolve to bundle blobs
with matching declared size. No operator event or evaluator summary is converted
into a fictitious trace; no sealed digest is silently rewritten for redaction.

The existing `platform.trace_bundle.materialize_harbor_trace_bundle` remains the
producer for durable native Harbor rollout logs and their actual PNG frames. Pass
its resulting extracted, verified bundle into this adapter. Traces captured by
other native harnesses use their existing native capture authority. An interrupted
bundle may be published only with explicit `allow_interrupted=True`; its receipt
preserves incomplete trace identities and never upgrades capture status.

The output contains an exclusive publication owner lock, immutable input claim,
frozen bundle/collection receipt and final publication receipt. The input claim
freezes the original transfer expiry; retrying does not replenish that deadline.
The SDK's manifest-based idempotency key reconciles ambiguous upload/finalize
outcomes. A receipt is committed locally only after the trace store returns the
matching manifest/trace identities and a committed promotion receipt. A committed
receipt may be reread after expiry without another upload.

This receipt is **trace custody**, not a complete scientific result publication,
cleanup confirmation, monetary settlement, or catalog availability guarantee.
Native result manifests must independently reference this publication alongside
immutable grading and cleanup/accounting receipts. Partial freeze/upload failures
remain evidence for reconciliation and cannot be labeled saved results. The SDK
owns network transfer and scoped upload destinations; no signed URL is retained
in local summary custody.

Development validation: scoped Ruff and Python compilation only. Qualification,
real cloud transfers, interrupted-finalize recovery, visual rendering and the
32-attempt/two-viewer latency gate belong to the receiving engineer.
