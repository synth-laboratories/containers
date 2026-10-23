# Trace V5 partial accounting and producer adapters

This v0.24 implementation slice fixes producer/aggregation behavior without adding
fields to sealed Trace V5 dataclasses or rewriting existing trace/evidence bytes.
It does not complete the broader partial-publication/timestamp profile in SYN-4918.

## Usage reductions

`UsageV5.merged` sums **disjoint, already deduplicated** observations. It is not a
snapshot/delta reducer. Known zero stays zero. If a field is unknown on either
side, its aggregate stays `None`; a known/unknown mismatch is listed in
`unavailable_fields`. Explicit field unavailability takes precedence over a supplied
number. An input marked `partial` or `unavailable`, an explicit unavailable field,
or an asymmetric missing field makes the aggregate `partial`, never `derived`.
Optional fields absent on both sides remain absent; this does not alone change the
provenance of otherwise known fields. Source refs retain their sorted union.

Reductions in capture, V4 import and ATIF import/export start with the first real
observation. An empty object or an invented zero is not an additive identity for
unknown usage. Original per-call observations remain authoritative; an aggregate
with a missing component does not expose a misleading known subtotal as a total.
Wall time summation remains work accounting, not parallel elapsed/critical-path
latency. Provider token categories, estimated costs, validation of historical
numeric values, and snapshot/delta authority need their separate semantic profiles.

Responses normalization now preserves zero-valued input/output/cache counts even
when a legacy alias is also present. V4 cannot represent nullable token counts:
projection version 2 declares each affected field in the loss manifest rather than
silently presenting its required integer default as observed usage.

## Codex source time

Codex stdout importer version 3 projects `timestamp` or `occurred_at` from the
outer record, then the nested `msg` object, in that precedence order. It accepts
timezone-qualified RFC3339 with `T`, `Z`/numeric offsets and fractional seconds,
retaining the original spelling. A selected invalid value does not fall through
to another clock. File/stream order remains the event sequence even when source
time repeats or regresses.

Each canonical event payload includes `_synth_importer` and a versioned
`_synth_import_time` object (`synth.codex-source-time.v1`): producer/unavailable
basis, selected field, valid producer occurrence if available, and an absent/invalid
reason otherwise. V5 currently requires `occurred_at`; when source time is unknown
the legacy epoch fallback remains there **with an explicit `canonical_fallback`
label**. It is not a measured producer, collector or ingestion time. The existing
serializer may omit null values; missing optional fields in this profile mean
unknown, never zero. Safe source bytes remain retained in the bundle.

Root import/capture timestamps still use legacy deterministic import values. This
slice does not establish independent observed, committed or indexed clocks and does
not remove the required-time limitation of V5. Consumers needing those guarantees
must wait for the accepted version/extension and storage conformance work.

## Responses terminal outcomes

Responses adapter version 2 recognizes `completed`, `failed`, `cancelled` and
`incomplete` in unary, SSE and WebSocket paths. Terminal observation is separate
from success. Normalized results retain the terminal status plus exact
`incomplete_details`/`error`; raw stream events remain available. When a terminal
payload omits output, reconstructed output and terminal usage/details are retained.
An interrupted stream without a terminal event remains nonterminal.

Capture stores provider status/details on model-call spans. An HTTP-successful
incomplete response becomes a `truncated` span/event; failed/cancelled outcomes
remain error/canceled. The WebSocket relay closes the current call on all terminal
events and retains usage; incomplete/cancelled are not counted as completed work.
HTTP transport errors retain error status. This does not change scientific verdicts
or prove that capture coverage itself is incomplete whenever model output is short.

Previously sealed documents remain readable with their original digests. New raw
captures identify adapter version 2. Finalization still refuses mismatched adapter
versions; unfinished version-1 raw captures require their matching older runtime.
No silent reinterpretation/migration of old raw captures is introduced. Full stream
replay deduplication, multi-item/part reconstruction and the broader hosted
publication/index/crash-prefix tests remain pending in SYN-4918/SYN-4648.
