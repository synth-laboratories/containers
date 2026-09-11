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
checks provider id/name/active state and exact admitted resource shape. Creation
still uses the provider's snapshot-name API. Therefore concurrent external
replacement of that name is not excluded by a post-creation OCI digest receipt;
this path requires private owner isolation and provider qualification. The
adapter retains normal finite sandbox TTL and independent deletion/absence proof.

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
