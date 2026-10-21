# Rhodes client lifecycle

`synth_containers.rhodes_client.RhodesClient` extends the existing `PoolClient`.
It registers content-addressed environment manifests referencing committed
Artifact Platform inputs, lists/inspects/archives versions, checks placement
compatibility, acquires/renews/releases server leases, and follows committed
operator events. Existing submit/get/results/usage/cancel operations remain on
the same client and service authority.

`upload_environment_source(path, operation_id=...)` captures at most 64 MiB,
prepares one private Artifact Platform publication, uploads create-only bytes
through a separate transport without the Synth bearer token, and finalizes the
same original declaration. It returns the source reference consumed by the
release manifest. A lost response can be reconciled with the same operation;
changed content under that operation conflicts. Lower-level prepare/finalize
methods support callers that already manage the canonical upload targets.

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

## Named definitions

`bind_definition(name, release_id, expected_revision=0, operation_id=...)`
creates a named definition. Use the inspected revision for later bindings.
Names are lowercase letters, digits, underscores, dots or hyphens, starting
with a letter or digit. A binding points to an already imported immutable
release; it does not mutate that release or start compute.

`get_definition`, `list_definitions`, `definition_history` and
`definition_operation` expose the current head, bounded pages and retained
receipts. `archive_definition` requires the current revision and a stable
operation ID. Archive preserves history and executable versions. An explicit
later bind at the archived revision reactivates the name as a new revision.

Bind/archive retries preserve the original operation and expected revision.
The server returns the original receipt even after intervening updates; changed
requests conflict. HTTP 409 revision conflicts require inspection and a new
deliberate operation, rather than automatic rebasing of a write.

## Observed local capacity

`task_readiness(pool_id, task_id)` reads exact provider observations without
reserving capacity. The observed claim returns flat lease coordinates including
`lease_id`, `container_id` and `reused`; malformed service envelopes refuse.
`acquire_observed_lease` requests the optional single-slot,
fresh-instance local Docker HTTP profile. It differs from `acquire_lease`, which
retains the legacy logical reservation contract. Warm reset and hosted placement
require their own supported profiles; the client never substitutes one.

The observed claim excludes competing pool use and survives TTL expiry until
closed or reconciled. Pending interactive operations prevent close. Closing
this borrowed slot retains the provider deployment under pool custody and does
not confirm provider shutdown. Historical use prevents reset-free reassignment.
A rollout can opt into this profile by passing `capacity_lease_id` in its
existing `submit` payload together with its stable `idempotency_key`. The server
consumes that exact lease for one evaluation, pins task/instance/image/expiry,
reobserves the provider before execution and narrows the execution deadline to
the original lease expiry. The executor records the admitted lease ID. Another
rollout or interactive writer cannot reuse the consumed instance without a
qualified reset. Closing an attached live or unresolved rollout refuses.

The optional attachment is a local Docker HTTP profile. Native harness
validation, hosted placement, funding guarantees and MCP/optimizer qualification
remain separate; unsupported profiles refuse explicitly.
