# Native lifecycle development candidate — 2026-09-11

This is implementation documentation. No new tests, live provider runs, acceptance,
or release qualification were performed in this development change.

## Shared coordinator

`execute_native_lifecycle(output, run_id=..., limits=LifecycleLimits(...),
operations=NativeLifecycleOperations(...))` runs async zero-argument setup, work,
verifier and publication callbacks and always invokes cleanup. Callbacks share
caller state. Cleanup must return independent provider custody with
`cleanup_status: confirmed`; a successful CLI exit is not an absence receipt.
The coordinator returns each phase's value separately. Scientific score, execution
error and cleanup outcome must remain independent when adapting results.

The output directory retains `limit-claim.json`, `limit-events.jsonl` and an
exclusive `limit-owner.lock`. Claims contain absolute original expiry and exact
limits; reopening cannot replenish allowances. Recovery never blindly replays a
started phase. It refuses work and invokes cleanup so the owner can reconcile
before admitting a distinct attempt. Phase deadlines use both wall and monotonic
remaining time. Cleanup has reserved grace. Recovery after original expiry gets
bounded emergency cleanup, explicitly outside the expired lifetime.

`run_bounded_process(argv, output=..., max_output_bytes=...)` can implement a work
callback. It creates a POSIX process group, merges stdout/stderr, admits every
write before retention, and terminates the group on exit/cancellation/overflow.
Output is exclusively created so reopening cannot reset that file's allowance.
It does not quota arbitrary child filesystem writes or terminate remote sandboxes.

The coordinator is cooperative asyncio supervision, not a daemon surviving worker
loss. Remote actors that ignore cancellation require the provider's expiry and
fencing mechanisms. A receipt persistence failure does not suppress cleanup;
receipt gaps are retained in subsequent stop custody when possible. Monetary
settlement remains the external budget authority's responsibility.

## Docker workspace actuator

`ObservedDockerEnvironment(workspace_tmpfs_bytes=N, ...)` adds a generated Compose
overlay mounting an **initially empty** `/workspace` tmpfs with an explicit byte
limit. It drops SYS_ADMIN and enables no-new-privileges. Before returning from
startup, it inspects the primary container and requires the requested tmpfs size
and absence of alternate mounts beneath `/workspace`.

This option hides any image content originally at `/workspace`. It is suitable
only for a task that deliberately initializes that empty workspace. It enforces
that mount's bytes, not all disk writes, bind-mounted logs, inode count or root
filesystem storage. Consumers request it through Harbor environment kwargs, e.g.
`--ek workspace_tmpfs_bytes=1073741824`. The default preserves prior task behavior.

## Daytona prepared artifact and expiry

Build completion writes `build-artifact.json`. It binds source/context digests,
private snapshot id/name, admitted architecture and owner expiry. It never claims
that a provider snapshot name is an OCI image digest.

`DaytonaSnapshotProvider(..., qualified_architecture="amd64")` requires an
explicit previously qualified region architecture. It does not imply that the
provider supports a cross-architecture builder switch. `DaytonaSnapshotBuild`
requires that architecture to match provider admission and rejects unsupported
`required_capabilities` before uploading any context. The inspected provider seam
exposes no builder resource/charge ceiling or remote-context expiry capability;
those hard requests fail closed. Sandbox CPU/memory/disk fields are not builder
limits. No provider capability is inferred from operator estimates.

Snapshot retention defaults to 86400 seconds, configurable through
`SnapshotBuildLimits.retention_seconds` (maximum 604800). Recovery can delete a
completed expired snapshot after acquiring its owner lock. This is owner-driven
reconciliation: a hosted scheduler must call `recover_daytona_snapshot_build`.
It is not native provider storage expiry. Legacy claims without an expiry retain
their explicit-release policy. Remote uploaded context cleanup remains unconfirmed
and is recorded separately; deleting a snapshot does not assert context release.

Native `BoundedDaytonaEnvironment` accepts `prepared_snapshot_artifact`,
`expected_source_package_digest` and `expected_architecture` kwargs. Before
creation it checks expiry and source binding, then freshly reads the snapshot and
checks provider id/name/active state and exact admitted resource shape. Creation passes the immutable provider snapshot ID, not the mutable name. The
pinned Daytona 0.210.0 CreateSandbox schema explicitly accepts ID or name. This
binds the provider snapshot identity, without claiming an independently verified
OCI image digest. The adapter retains normal finite sandbox TTL and independent
deletion/absence proof.

## Remaining implementation and qualification boundaries

These changes do not finish all of D2/D4. Integration must wire actual native
consumer callbacks, custody and budget settlement. Whole-filesystem/output-artifact
quotas beyond the explicit mounts/stream require additional substrate actuators.
Daytona hard builder spend/resource control and uploaded-context storage expiry
remain unsupported by the inspected provider seam; obtaining those guarantees
requires a provider capability or a bounded external builder/storage adapter.
Hosted scheduling and multi-worker build deduplication remain owner-service work.
The receiving engineer owns callback integration qualification, timeout/cancel
faults, mount compatibility, resource absence and all provider runs.

## Additional actuators and recovery

`ObservedDockerEnvironment(writable_layer_bytes=N, ...)` asks Docker for the
storage driver's writable-layer `storage_opt.size` quota (1 MiB through 1 TiB).
Docker must accept the setting; unsupported backing filesystems fail creation
without unlimited fallback. Startup inspects the primary container's StorageOpt
before work. Bind mounts and tmpfs are separate resources and require their own
allowances; this option is not a total host-disk guarantee.

`collect_bounded_artifacts(sources, destination, max_bytes=..., max_files=...)`
admits regular-file payloads before each retained write, refuses symlink final
components and changed source metadata, and commits a hashed manifest last.
A failed collection leaves bounded partial evidence without a completion manifest.
The exclusively created destination prevents retries from resetting its allowance.
This controls collection custody, not files produced elsewhere by the worker.

`recover_native_lifecycle(output, cleanup=...)` loads original persisted limits,
acquires the exclusive owner lock and refuses recovery before expiry. The caller
supplies the existing provider reconciler, which independently checks ownership.
No work callback or provider creation occurs during recovery.

`run_bounded_process(..., redact=secrets)` redacts UTF-8 byte sequences before
persistence, including matches split across reads. It bounds both raw admission
and redacted retained bytes. On cancellation/overflow the withheld unfinished
suffix is omitted. Redaction allows at most 128 patterns, each at most 4096 bytes.

## Inspected Daytona 0.210.0 contract evidence

The exact `daytona==0.210.0` and `daytona-api-client-async==0.210.0` wheels
were installed without dependencies for source inspection at
`/Users/joshuapurtell/GitHub/artifacts/native-dev-daytona-sdk-20260911`.
No provider client was created or API called.

- `daytona_api_client_async/models/create_sandbox.py` declares snapshot as ID or name.
- `daytona/common/snapshot.py` documents snapshot memory as `mem`, not `memory`.
- `daytona_api_client_async/models/create_snapshot.py` explicitly describes CPU,
  memory and disk as resources of the **resulting sandbox**; it has no builder
  budget, timeout, CPU or retained-context expiry field.
- `daytona_api_client_async/api/object_storage_api.py` exposes only push-access
  operations; there is no uploaded-context deletion/expiry operation in this seam.
- SnapshotDto contains provider `ref` and `image_name` strings but no independently
  verified OCI digest/architecture attestation. Those strings remain outside the
  immutable-image gate.

## Integrated staged Harbor execution

`staged_harbor_execution.execute_staged_harbor` is wired into evals' native CLI
translator and HTTP task adapter. Setup returns a registrar receipt; the shared
coordinator rechecks task digest, image, custody adapter and frozen inner agent /
verifier limits before building a single-attempt/no-retry command. It supervises
setup, bounded CLI execution, verifier-result decoding, evidence publication and
independent provider cleanup. `execution-plan.json` explicitly labels outer work
as combined CLI execution and outer verifier as receipt decoding; the real agent
and verifier deadlines remain Harbor's frozen inner timers.

Interrupted execution may enter the existing bounded publication allowance to
retain already-owned evidence; it cannot admit new work. Publication receives a
declared redacted representation while original native result custody remains
separate. `cleanup_staged_harbor_jobs` reads exact resource custody and, if pending,
invokes the existing age/ownership-guarded provider reconciler in a bounded
subprocess. Early or ambiguous cleanup stays pending. It never relaxes recovery
age guards to fabricate an immediate absence guarantee.

## Whole-attempt hard-limit admission

`required_limit_capabilities` is now accepted and checked before any native task
staging, shared execution, or Docker/Daytona environment creation. It uses the
existing `LimitCapability` contract (typed instances or serialized entries).
Requirements are rechecked from the staged receipt before launch. This native
composition currently advertises **no whole-attempt guaranteed capabilities**:
its narrower timer/mount/collection receipts do not become such guarantees.
A request for `workspace_bytes/native_control`, including writable bind mounts,
or total `output_bytes/native_control` therefore fails before provisioning.
This is deliberate fail-closed admission, not implementation of host filesystem
quotas. Host bind mounts need independently qualified backing-filesystem quota
custody before a future adapter can advertise those capabilities.

The shared executor also accepts `model=None` for oracle/nop agents, narrowly
allowlisted `extra_cli_args` (`--agent-kwarg`, `--agent-env`, `--verifier-env`,
`--yes`) and a synchronous/asynchronous boolean `should_cancel` callback.
Cancellation is checked before creation and every 250 ms during execution;
it cancels/reaps the owned process group and proceeds through independent provider
cleanup. These arguments cannot override task path, retries, concurrency or phase
multiplier ceilings.
