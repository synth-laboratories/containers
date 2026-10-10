# Shared connector runtime

One `SynthTunnelConnectorAgent` owns a physical attachment. Its local target
registry resolves data keys to immutable origin snapshots and explicitly named
secret references. A worker provides a lease and approved route binding; it
cannot supply the origin address. Origin auth is added locally after worker,
backend and hop credentials have been removed.

`HttpConnectorAttachmentProvider` reacquires an attachment after transport or
actor restart, with stable operation identity until acknowledgement. The
acknowledgement must match prepared connector/session/generation lineage.
Readiness clears on disconnection. A newer attachment generation retires the
previous transport; the logical actor session persists. Canonical grants and
revocations are never restored from historical transport or checkpoint state.

The backend's v2 API owns connector, route and lease policy. Borrowed job handles
own lease authorization and credential refresh; they never stop the agent or
remove configured routes. A returned close receipt is accepted policy intent,
not proof that arbitrary origin execution stopped. Cleanup and outcome remain
separately observed.

The new physical mode requires `synth-tunnel.v2`; the explicit legacy lease
mode continues to support its existing negotiated binary codec. V2 connector
WebSocket messages are limited to 2 MiB. HTTP uses bounded 64 KiB data frames,
50 MB request limit and slot-credit streaming uploads. No payload chunks enter
actor state or durable control rows.

The host response byte budget is shared across HTTP, queued WebSocket messages
and incomplete reassembly. Job concurrency remains separate from host limits.
Actor owner permits expire within five seconds when renewal fails. Policies are
reapplied under a fresh generation through a canonical reconciliation command.

## Foreground service

`python -m synth_containers tunnel serve --profile profile.json` runs an owned
attachment; its existing host supervisor owns process restart. `status` and
`doctor` observe scoped policy and local secret-reference resolution.
`close-lease` commits only that job's revocation.

Profiles contain connector/client UUIDs, backend/gateway URLs, a backend
credential environment-variable name, and up to 64 named local targets. Each
target declares an address, service identity and optional header/secret-name
bindings. Profiles contain no raw credential values. UUIDs persist in data;
product route names are never compiled into the tunnel runtime.

The service uses the v2 `SynthClient().tunnels` facade. Its local package version
must include that feature before activation; package publication is a separate
release gate. No credential registry or Keychain is consulted by this service.
On attachment failure, the agent clears readiness, closes its owned origin
uploads/WebSockets and releases stale stream custody before reconnect backoff.
Reconnect pacing cannot extend origin forwarding or stale readiness. It does
not prove arbitrary application/provider work stopped.


## Application checkpoint custody

The checkpoint package separates immutable artifact storage, participant cuts,
origin restore receipts and logical session authority. `FileArtifactStore`
atomically publishes content-addressed blobs; `S3ArtifactStore` uses an injected
S3 client, conditional creation and digest-checked reads. A caller supplies its
already-authorized client credentials; this package consults no secret registry.
Namespace/key validation precedes vendor I/O and preserves
`checkpoint_artifact_scope`. A lost write acknowledgement remains
`checkpoint_blob_write_in_doubt`; retry conditionally creates or verifies the
existing object. An existing object's metadata cannot replace verification of
its actual length and digest, including on duplicate publication. The native
MinIO qualification covers sixteen concurrent duplicate writes, corrupted and
short objects, committed response loss and fresh-client readback after SIGKILL.
This does not qualify Wasabi bucket policies or production retention.

`ApplicationCheckpointCoordinator` waits for a retired serving generation,
verified quiescence and permit expiry, then stores participant blobs and one
joint manifest before sealing the session checkpoint. A local fsynced journal
retains uncertain custody. Completed replay reads the immutable result; it
never recaptures an origin. The participant must reject new work while quiesced
and must reconcile outstanding provider effects before declaring a clean cut.

`ApplicationRestoreCoordinator` requires a matching manifest, service identity,
participant set and fresh origin incarnations. It checks every blob before
starting restore. The logical session remains restoring until all participant
receipts are verified and completion is acknowledged. Lost origin replies are
looked up, never replayed. Partial restore remains fenced until every participant
proves its exact durable receipt. Reconciliation can then complete the idempotent
logical control operation under a fresh actor-wake generation;
opening a socket cannot substitute for proof of origin completion.

`HttpSessionAuthority` connects the async public SDK's canonical session-operation
API to capability-fenced gateway administration. The backend journals typed
intent before an actor effect; the gateway records a single claim and settles
it against durable Rivet facts. Uncertain claims do not dispatch another POST.
The adapter holds no engine/service credential and sends no origin work. Its
metadata response is bounded at 64 KiB; all commands share the caller's original
deadline. Control capabilities cannot be replaced by forwarding credentials.

A manifest covers only its declared, qualified application participants.
Generic process restart, transport reconnection, a logical session checkpoint
or a profile declaration alone does not prove environment snapshot continuity.
Production participant enrollment, retention, deployed infrastructure and
trainer-specific qualification remain release gates.


### Monotonic pause barrier

A durable pause timestamp is receipt metadata. The application capture
coordinator waits the supported five-second owner-permit envelope on its local
monotonic clock, starting after the pause acknowledgement and overlapping actual
participant quiescence. The actor independently refuses a new seal until its
monotonic barrier expires. Forward wall jumps cannot seal early; wall rollback
cannot extend this barrier. Duplicate pause does not restart it. Actor wake
while quiescing conservatively starts a fresh five-second barrier and advances
the generation; stale cut commands still refuse. A cut interrupted before
sealing remains unresolved and cannot be silently recaptured. Participant
quiescence and origin-effect reconciliation remain required independently.

This adds up to five seconds to capture with no outstanding work; it keeps
absolute catalog/JWT/deadline clock requirements unchanged. Actor barrier tests
include an isolated old-wall-predicate negative control, and the real joint MQ
capture records acknowledgement-to-capture elapsed time.


### Completed checkpoint history

The file journal now lives in `checkpoints/journal.py`. One advisory admission
lock covers state transitions and archive rename/read boundaries across
processes. Completed receipts with an immutable artifact move atomically into
UUID-sharded history after state fsync; both rename parents are fsynced. Reads
and duplicate admission consult active custody and archive. Interrupted and
uncertain entries stay active, with the existing 4096-operation bound. Old
completed active entries migrate conservatively at admission or replay.

Real file regressions complete 4097 operations, restart the journal, reject
identity/artifact conflicts and preserve the oldest immutable artifact. An
owned child is SIGKILLed after completed-state fsync but before archival; a
new journal reads its completion and archives it without replay, while a
second uncertain operation remains active. Completed SDK capture contexts leave
memory only after canonical seal confirmation; unresolved contexts remain
bounded. This archive does not delete blobs, change leases, or claim production
retention/backup policy. Canonical actor-cache/import qualification is separate.

### Callback cancellation evidence

The callable-authoring `Container.fastapi()` surface records cancellation intent
before signalling its owned coroutine. It retains that task while cleanup runs
and does not repeatedly cancel it. A normal completed/failed origin result wins
the cancellation race. An already terminal result remains immutable.

If the coroutine exits with `CancelledError` without an origin result, status is
`in_doubt`, with separate cancelled-callback, unverified-origin and unverified
resource-cleanup fields. An awaiter stopping cannot prove a thread, subprocess,
remote provider or committed side effect stopped. This ephemeral SDK surface is
not a durable managed-origin operation ledger or an executor cleanup grant.

Both blocking and asynchronous submissions give the callback independent task
ownership. Cancelling a blocking observer leaves that callback queryable.
Within one origin process, an exact rollout ID/input replay joins existing work
or returns its retained result. Reusing that ID with different input returns
409 before dispatch. Canonical JSON rejects non-finite values. These identities
and results are in memory: process restart does not establish replay safety.

The thin compatibility adapter has no origin termination or authoritative
outcome port. Its terminate endpoint records intent as `in_doubt` with
`origin_termination_unsupported`, leaving actual origin work untouched. It
preserves completed/failed/cancelled/terminated/stopped results. A platform
adapter that can prove cancellation supplies its own termination implementation.

### Persistent callable-origin operations

`Container.fastapi(operation_directory=Path(...))` owns a local SQLite origin
journal. Set `Container(..., operation_directory=Path(...))` to use the same
journal through `serve()` or the default `fastapi()`; an explicit `fastapi`
directory overrides that constructor setting. Persist that private directory
with the service. Its advisory lifetime
lock refuses another live owner; the stored runtime identity refuses directory
reuse by a different service. Admission commits a stable rollout ID and canonical
input digest before callback dispatch. A matching replay observes the retained
operation; conflicting input returns 409 without invoking the callback.

Origin results and cancellation intent commit with SQLite FULL synchronization
before acknowledgement. On process restart, running/cancelling operations become
`in_doubt`, retaining their input identity and cancellation intent. They never
dispatch again automatically. Completed results are immutable and remain
queryable. This establishes durable origin receipts, not atomicity with remote
provider effects: an effect committed before a crash without a receipt remains
uncertain until an authoritative origin/provider reconciliation supplies proof.

Reads address individual indexed records rather than loading all history into
memory. The admission bound covers 4096 unresolved operations, not completed
history; one record is limited to 1 MiB. The selected real tests retain 4097
completed operations, verify capacity/identity/owner refusals and SIGKILL an
actual HTTP origin after completion and after an effect without a result.
The prior SDK loses both completed and unresolved custody after restart.

The compatibility default remains explicitly ephemeral when no directory is
provided; discovery reports `durable_operation_lookup=false`. This journal
requires local filesystem locking/durability semantics. Blob retention, backup
restore, external outcome reconciliation and separately scoped cleanup grants
have their own contracts and qualification.

The journal now retains a non-nil origin incarnation UUID alongside its runtime
identity. Migration from the initial journal adds that identity atomically;
restart keeps it, and malformed persisted identities refuse startup. Metadata
advertises the durable identity and `origin_operation_contract=rollout.v1`.

`GET /operations/rollouts/{rollout_id}` requires the expected `origin_instance`
and canonical `input_sha256` as query fields. It returns those immutable bindings
with the retained record. `POST /operations/rollouts/{rollout_id}/cancel` accepts
the same strict binding in its body and validates it before recording intent or
cancelling a callback. Another incarnation or input refuses with 409; an absent
operation returns 404, and an ephemeral origin reports unsupported with 503.
Cancellation remains intent: interrupted external work and cleanup stay
unverified. These origin ports do not grant gateway access; separately scoped
canonical cleanup capabilities and consumer enrollment are still being added.
