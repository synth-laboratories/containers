# Rhodes client lifecycle

`synth_containers.rhodes_client.RhodesClient` extends the existing `PoolClient`.
It registers content-addressed environment manifests referencing committed
Artifact Platform inputs, lists/inspects/archives versions, checks placement
compatibility, acquires/renews/releases server leases, and follows committed
operator events. Existing submit/get/results/usage/cancel operations remain on
the same client and service authority.

An imported manifest is not a validated release. An acquired legacy lease is
not proof of observed provider readiness. Inspect server evidence separately;
unsupported profiles and missing qualification must not trigger provider or
client-side execution fallback.

`event_page` explicitly selects JSON. `follow_events` consumes bounded SSE
windows with `Last-Event-ID` and reconnects from the last delivered sequence.
The parser bounds each frame, checks exact event IDs and refuses gaps and
unterminated frames. Snapshot high-water marks never advance the evidence
cursor. Timeout/disconnect detaches; explicit `cancel` owns cancellation.
Terminal completion includes publication, inference and cleanup pending axes.
These are operator events, not replacements for native Trace V5 envelopes.

Rollout submission requires a stable `idempotency_key`. A batch validates every
key before sending, uses the service's `requests`/`items` wire, and does not
automatically retry a partially accepted batch. Mutations without a replay
identity are attempted once. HTTP refusals retain status, typed code and
`Retry-After` in `RhodesRequestError`; a caller can reconcile the original IDs
before any deliberate retry.

This interface is in implementation qualification. Native benchmark/provider,
ready-capacity, org funding and Swarms/optimizer adoption evidence remain
separate acceptance requirements in the v0.24 Rhodes tickets.
