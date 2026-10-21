# Scoped journal replay and SSE watermarks

This v0.24 slice builds on `synth.trace-stream-event.v1`, the existing envelope
digest/chain and the local rollout journal. It preserves native event bytes,
sequence identities and digest contracts. It adds scoped read positions and
transport watermarks, rather than a second event store or execution service.

## Polling and replay scope

The existing sequence `after`, `limit`, `ack` and `wait_ms` inputs remain available.
`/rollouts/{id}/events` also accepts repeated `kinds` filters and a `cursor` token.
`/rollouts/{id}/reward/events` uses the fixed reward-view filter. A token conflicts
with a nonzero explicit `after`. Invalid or ahead-of-history positions now refuse;
they never silently reset or clamp to the available prefix.

Every page's envelopes, source high-water, native chain head and closed flag come
from one atomic snapshot. These fields do not move when a writer appends or closes
while that page is serialized. Public envelopes are detached copies, so mutating
an append/read result cannot change retained history.

The additive cursor fields are:

| Field | Meaning |
| --- | --- |
| `next` | Legacy last delivered matching sequence, or requested `after`. |
| `scanned_through` | Source prefix inspected by this view. It may include deliberately excluded kinds. |
| `resume` | `synth.stream-cursor.v1`, encoded as an unsigned `sc1.` base64url JSON token. |
| `scope` | Stream, subject kind/identity and canonical filter digest. |
| `chain_head` | Native chain head over the entire snapshot high-water, not necessarily `next`. |
| `durability` | Configured local strategy: `memory` or `local_fsync`. No hosted publication/index claim. |
| `writer_state` | `available` or `recovery_required`. This is independent of source closure. |

A resume token binds its inspected position to that prefix's native chain head.
Reusing it for another stream, subject, view/filter or changed history refuses.
The chain retains the existing digest definition; it does not add timestamp or
metadata coverage absent from that native contract. Runtime owners must retain
immutable run/attempt identities rather than reuse them for unrelated execution.

Filters use sorted, deduplicated kinds; omitted and empty filters differ. A page
that has more matching events advances only through its last delivered match.
When no further match exists, its scoped position advances through the inspected
source tail. This is filtered read progress, not delivery of excluded evidence.
Tokens are bounded at 4096 characters and reject malformed encoding, duplicate
JSON keys, absent/unknown schema versions and invalid field types. Filters are
bounded at 256 inputs and 512 characters per name.

The filter digest is SHA-256 over compact JSON `{"kinds":...}` encoded as UTF-8:
null for an omitted filter; otherwise distinct names sorted by Unicode code point.
Use ASCII JSON escapes (lowercase `\u` hex and UTF-16 surrogate pairs outside the
BMP), no whitespace, and the standard JSON escapes for control characters. The
token JSON itself need not have a particular key order when decoded. Frozen
examples, including empty/unfiltered and Unicode/duplicate filters, are in
`contracts/fixtures/stream/cursor-v1.json`. These are portable byte fixtures;
cross-repository/foreign-language decoder qualification remains SYN-4648.

Tokens are transparent positions, **not** signatures, credentials, admission or
permission grants. Every hosted attach/read must independently enforce current
view/org/actor authority. A filtered consumer cannot verify the full source chain
from only its returned subset; source receipts and retained authority stay separate.

Python exports `StreamCursorV1`, `StreamScopeV1`, `StreamCursorError` and
`StreamWatermarkV1`; `event_log.poll_payload(..., cursor=...)` uses the same contract.

## SSE profiles

The default SSE profile retains numeric semantic sequence IDs. Unsequenced control
events now omit `id:` so they cannot reset a browser's Last-Event-ID. Heartbeats
remain comments with no ID. Malformed, negative or ahead Last-Event-ID values
refuse before streaming headers instead of resetting to zero.

Opt in with `GET /rollouts/{id}/stream?scoped=true`, or supply a scoped `cursor`.
Repeated `kinds` must match the token's view. The fixed reward stream uses its own
scope. Semantic SSE IDs then contain scoped tokens, while each JSON event's native
`event_id`, sequence, payload and digest remain unchanged. Last-Event-ID overrides
the initial URL cursor on reconnect and must match the requested view/history.

`event: stream.watermark` carries `synth.stream-watermark.v1`: scoped cursor,
snapshot high-water, source journal closure, local durability strategy and writer
state. It is a transport read model, not a persisted native event or a capture seal.
It accounts for intentionally excluded tail records and lets poll/SSE consumers
exchange resume positions. Heartbeats never create watermarks or advance evidence.

Source closure is read from the same snapshot as the replayed records, so a writer
closing during replay cannot make SSE exit before draining its final tail. A
fenced scoped stream reports `recovery_required` with its known prefix and ends;
it does not invent closure. Transport EOF, disconnect and source journal closure
do not establish benchmark completion, successful execution, verification, artifact
publication, usage settlement or cleanup. Required capture seals remain their own
native facts/authorities.

New producers refuse control characters in event kinds; SSE also refuses native
field overrides through transport metadata. Historical retained data remains
inspectable through JSON without injecting its kind into SSE framing.

## Local persistence and recovery

These journals have one owning writer. Successful writes fsync the file and its
direct parent directory before publication; the owner must provision and durably
retain the storage root. These are filesystem-local guarantees, not replica,
upload, manifest-commit or index acknowledgments.

An uncertain append or ack persistence failure fences the owner. It cannot reuse
a possibly written sequence. A fresh owner must recover/reconcile the retained
prefix; it may discover a complete record whose acknowledgment was lost. Recovery
does not repeat environment/model compute. An unterminated tail refuses even if
its JSON happens to parse. Corrupt middle records, gaps or mismatched heads refuse;
no automatic repair, truncation or deletion is introduced.

Memory-only closed logs retain their full data when compaction/release is asked
for. Durable compacted logs reload and verify their original high-water/head/closed
identity. Missing or corrupt retained history is explicit rather than an empty
successful page. Legacy `ack` remains stream-wide, not a per-client/per-view
checkpoint, and is published only after its own persistence succeeds.

Poll errors use `error` plus status: 422 invalid input/filter/cursor, 409 wrong
scope/ahead/changed/corrupt history, 410 missing retained history, and 503 unavailable
ack/storage or a fenced ack writer. SSE validation uses the same status with the
code in `detail`. Reconciliation requires retained authority; an error does not
promise a new snapshot or resurrect expired evidence.

## Remaining work

This is not the full SYN-4918/4919/4841/4842 contract. Aggregate/batch cursors,
service-generation fencing, authenticated org views, hosted commit/index clocks,
chunk/output byte coverage, bounded slow-reader/storage policies, durable per-client
checkpoints, terminal reducers/ten-row rendering and limits/usage progress remain
open. Native schemas and other owners' operator feeds require explicit adapters;
this cursor does not merge those authorities. No hosted readiness, provider or
benchmark qualification follows from these local replay checks.
