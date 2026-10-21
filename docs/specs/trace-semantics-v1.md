# Trace semantics and publication v1

The optional `synth.trace-semantics.v1` receipt binds to an exact sealed V5
digest. It never edits a V5 document, raw segment, judgment, or historical digest.
The explicit timestamp extension is **nullable-sidecar-v1**. Required V5
`occurred_at` remains a legacy display field; an importer fallback is not producer
occurrence time. Readers that ignore this profile retain their historical bytes
and explicitly lose the additional semantics.

## Independent facts

Execution outcome, live/sealed lifecycle, capture completeness, body availability,
publication commit and index visibility are independent. Failed execution with
complete capture is valid; sealed partial evidence is valid. A sealed trace is
not a successful task, complete capture, published result or settled bill.

Capture classes declare requiredness, policy digest, nullable expected/retained
counts/bytes, retained ranges and a reason for partial/missing/unsupported/omitted
coverage. Unknown counts, usage, reward and content remain unknown. Completeness
is relative to those declared classes. Body disposition distinguishes complete,
reference-only, redacted, encrypted, digest-only, omitted, missing and truncated.
Readability is stronger than retrievability. Encryption never implies available
decryption authority. Local sidecar attachment verifies retained blob digests and
sizes and refuses foreign entities and unsupported encrypted-readability claims.
Required complete content must be readable and untransformed. Retention roots are
references to owner-maintained custody, not new grants created by a receipt.

## Times and ordering

Each event can retain producer occurrence, collector observation, ingestion and
commit time separately, with availability, basis, clock domain, precision, RFC3339
units and explicit timezone offset. Derived facts cite their sources. Missing time
does not acquire an epoch or collector time in this profile. Source sequence,
producer/stream/restart identity and causality remain authoritative through equal
timestamps, skew and clock regression.

Time filters explicitly select occurred/observed/ingested/committed basis and use
half-open `[start,end)` bounds. Unknown values are excluded, not invented. The
filtered frozen result orders by producer/stream/restart/sequence/event identity;
pagination uses the scoped stream protocol, not timestamp cursors. A live query
needs an owner-selected immutable cut before stable pagination is possible.

Durations use declared monotonic nanosecond clock domains. Open intervals stay
unknown; different domains refuse comparison. Sum of work and union of elapsed
intervals differ for parallel work. Critical-path latency remains unknown without
a causal execution graph; summing parallel spans is not elapsed latency.

## Accounting and corrections

`synth.usage-observation.v1` uses stable observation ID, call ID, authority,
revision, snapshot/delta mode and retained source references. Inclusive input and
output totals include cache and reasoning subsets. Cache read/write fields are
separate subsets; this profile does not assume they are mutually exclusive.
Total tokens, when all known, equal inclusive input plus output. Counts are strict
nonnegative integers: booleans, strings and fractional values refuse.

Replay of an identical ID is idempotent. Conflicting bytes for an ID refuse.
Snapshots select the greatest declared revision within one authority/call; equal
revision conflicts refuse. Deltas sum disjoint observations; corrections explicitly
name a compatible predecessor with a lower revision. Branching corrections and
mixed snapshot/delta or competing authority interpretations require explicit
selection. Missing fields in any selected observation remain missing in totals;
separate known subtotals retain their own partial coverage. Empty input is unknown.

USD cost preserves exact decimal strings, model, authority and source identity.
Estimates additionally require the exact rate-card digest; provider-supplied and
estimated costs do not replace one another implicitly. Retain selected original
observations beside reductions. A legacy float projection can lose decimal
precision and accounting identity. The strict Responses snapshot adapter is
optional; legacy normalization and sealed bytes remain unchanged.

Reward/annotation/rubric/producer/exact-selector/revision structures continue to
use existing V5 evidence standards. `synth.metadata-fact.v1` adds exact digest-bound
trace/session/span/artifact/rollout/experiment/external subjects, definition digest,
authority, units, revision, explicit supersession or coexistence and inspected
evidence. Authority selection is explicit; latest wall-clock time is never policy.
Alternative definitions/reviewers coexist. Corrections cannot change subject,
definition or authority silently.

## Durability and publication

Observed, local durable, uploaded, manifest committed and indexed positions are
separate. Positions are scoped to producer, stream and restart. Each stage cannot
exceed its prerequisite. A contiguous durable high-water starts at ordinal zero
and stops before a gap even when later ranges are retained. Transport heartbeats
are not evidence and do not advance these positions.

Raw spools fsync each append. Immutable bodies, segments, manifests and filesystem
blobs use unique pending files, fsync before hard-link publication and fsync the
directory. Mutable pointers require the existing single owning writer; this is
not multi-process lease authority. An uncertain write fences the in-memory owner.
Reopening verifies committed segments and rejects orphan or missing history.
Rotation retains the partial until the immutable manifest pointer is committed.

Explicit `repair` only discards an unterminated trailing record. It first validates
the committed prefix and every newline-terminated record. A malformed, forged,
out-of-order or corrupt complete record refuses without mutating the partial.
Recovery retains original exact bytes, original/tail/prefix digests and a repair
receipt under the capture's retained bundle objects before removing the partial.
A lost commit reply can be retried without duplicating a segment or generation.
No automatic destructive repair is performed by ordinary reads.

The publication outbox freezes an existing verified self-contained bundle cut,
attribution and operation identity. Its archive is generic immutable file custody;
it is not another hosted trace database. Upload verifies byte/size readback,
then publishes an immutable commit root, then calls the owner's idempotent index
adapter keyed by commit digest. Each stage returns a durable receipt. Failures
retain the outbox and report a conservative blocked stage; exception text is not
retained because remote errors may contain credentials. An uncertain write is
resolved by digest-verified retry. Index failure cannot erase a commit. None of
these paths can invoke original rollout or grader compute.

Readers require an exact commit root and verify archive closure before read/export.
The outbox survives original sandbox/source cleanup. Rhodes still owns execution,
admission, authenticated publication and discovery; Mnemosyne owns hosted evidence
index/query custody; Artifact Platform owns generic bytes. Those services must
register retention roots and implement authorization/fencing before claiming
hosted availability. A portable receipt is not a retention lease or access grant.
