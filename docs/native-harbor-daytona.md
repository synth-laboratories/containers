# Bounded native Harbor on Daytona

This opt-in extension is the first shared native-Daytona provider slice. It uses
Harbor's supported environment import interface and reuses Harbor's task, agent,
file transfer and verifier implementation. It does not edit installed Harbor or
Daytona packages and does not replace the public optimizer recipe layer.

Install `synth-containers[native-harbor-daytona]` with Python 3.12 or newer. This
extension requires the qualified Harbor 0.22.0 and Daytona 0.210.0 pair. The normal
containers package does not require either dependency. Existing backend Harbor
adapters and the REB broker's old pin remain separate until their migration gates
pass.

Use `--env synth_containers.harbor_daytona:BoundedDaytonaEnvironment` in a native
Harbor command. The task must declare a prebuilt digest-pinned image and explicit
CPU, memory and disk requests. This slice excludes task image builds, generated
snapshots and Compose. The defaults admit one CPU, one GiB of memory and one GiB
of disk. `--ek maximum_cpu=...`, `maximum_memory_gib` and `maximum_disk_gib` narrow
or raise those ceilings within fixed platform bounds; the task still declares
its actual request. These limits are resource bounds, not a currency reservation.

`--ek resource_ttl_minutes=5` sets the absolute provider lifetime (default five
minutes; one through 360 supported). The provider destroys the sandbox after
that lifetime even if it remains active, stopped, paused or archived. The
extension verifies the returned allocation and provider destruction timestamp
before allowing the environment to start. Unexpected or missing confirmation
fails the trial and leaves the handle available for cleanup. Idle stop and
immediate deletion after stop provide additional cleanup paths.

Each environment can make one creation request. A durable exclusive claim in the
trial directory refuses a duplicate after a process restart; an ambiguous
request requires reconciliation or a new explicitly admitted trial. The
version-qualified integration calls Harbor's undecorated creation method to
retain cancellation shielding and handle capture without its creation retry
loop. The cloud resource carries a unique `ai.synth.harbor.owner` label.

`resource-events.jsonl` reuses the shared append-and-fsync `OperatorJournal`.
It records creation intent before the provider call, returned resource identity,
observed allocation/lifetime and cleanup outcomes. Events contain sequence,
owner, timestamp and typed facts. Error messages and credentials are excluded.
This is a local operator journal, not sealed Trace V5 evidence or the hosted
Rhodes event service. The caller must retain/upload the trial directory.

The CLI can replay that custody without provider credentials:

```sh
synth-containers journal /path/to/trial/resource-events.jsonl --limit 100
synth-containers journal /path/to/trial/resource-events.jsonl --after-sequence 3
synth-containers journal /path/to/trial/resource-events.jsonl --follow --timeout-seconds 300
```

Replay returns `events`, `next_sequence`, `high_water` and `has_more`; save the
last processed sequence for reconnect. Follow emits one JSON event per line,
flushes each event, and stops at the observation timeout without stopping the
sandbox. It pins the first observed run identity and retries only writer-lock
contention. Missing files, corrupt/partial history and cursors ahead of durable
history are explicit errors. Reading is capped at 64 MiB; larger journals need
indexed/hosted custody. This CLI does not imply Workshop upload or cloud retention.

Cleanup issues one delete request, then requires a fresh typed provider absence.
Deletion errors and ambiguous reads emit `resource.cleanup_pending` and retain
the handle. Keeping a sandbox indefinitely with `delete=False` is refused.
The provider TTL remains armed regardless of client/viewer survival.

For a worker that exits before cleanup, retain the complete trial directory and
run `synth-containers harbor-daytona-reconcile /path/to/trial` with the same
provider account's environment credentials. Recovery never creates or renews a
sandbox. It waits until the recorded creation allowance plus provider lifetime
and a one-minute margin have expired; there is no force override. Older journals
use their qualified 300-second creation ceiling. A file lock excludes concurrent
recovery in the same directory.

The reconciler combines the durable provider ID with an exact owner-label lookup,
refuses multiple resources or mismatched ownership, records discovered identity
before deleting, and requires both fresh typed absence and an empty owned listing.
An empty listing alone cannot settle an ambiguous create with no known ID. Errors
append cleanup-pending facts while preserving custody; retrying recovery never
retries creation. Provider operations have individual timeouts inside a 90-second
reconciliation deadline. This is explicit local recovery, not a cloud reaper or
permission to delete another trial's resource.

The qualification scope is one deterministic oracle task with shared verification.
This extension does not yet establish full Codex capture, isolated verifier parity,
cloud journal custody, pre-provision database recovery, experiment dollar budgets,
or full Docker/direct-Daytona/hosted executor parity. Do not advertise those
capabilities from this extension's resource bounds.


## Prebuilt task migration

`native-harbor` installs the qualified Harbor dependency for Docker task staging;
`native-harbor-daytona` additionally installs the qualified Daytona SDK. Both
optional integrations need Python 3.12+. The core package remains usable without
Harbor or provider credentials.

```sh
synth-containers harbor-stage /path/to/source-task /path/to/new-stage \
  --provider daytona --image "$PINNED_TASK_IMAGE" --creation-timeout-seconds 300
```

Staging reuses `HarborEnvironmentDraft` / `HarborEnvironmentRelease`. It checks
source freshness, copies a bounded file inventory, preserves task instructions,
verifier files and task configuration values, then changes only the prebuilt
image and creation timeout and removes the unused Dockerfile. The original
`task.toml` and a receipt bind source digest, staged digest, image, provider and
release identity. The receipt is committed after copied data is flushed. An
existing or partial destination is never reused.

The image is selected by the operator. This binding is **not build-provenance
verification** and does not prove the image contains the task runtime. Build
receipts and scientific qualification must establish that before promotion.
Separate verifier images, Compose and host mounts are refused in this slice;
staging never converts a separate verifier into a shared one.

Static package inspection bounds a file at 64 MiB, a source tree at 256 MiB and
10,000 entries, and task TOML at 1 MiB. Hashing streams file bytes and preserves
existing digest framing. Copying rechecks the bounds, and mutations during
inspection/staging refuse a launch receipt. Fractional/string integer limits,
boolean/nonfinite phase durations, special files and symlinks are refused.
Missing network declarations remain `unspecified`; inspection cannot claim
network isolation on their behalf. Artifact-object declarations need an explicit
contract adapter and are refused rather than converted into strings.

Task metadata inspection and staging share a bounded reader: at most 1 MiB,
regular files only, no final symlink following, and no FIFO blocking. Staging
rechecks the bound at the actual read rather than relying on a prior inventory.

`harbor-stage --resource-ttl-minutes N` now freezes the qualified native backend
arguments in `stage-receipt.json`. Python consumers use
`native_harbor_environment_flags` from `harbor_task_stage`; benchmark wrappers do
not maintain their own Daytona ceiling/rounding policy. Docker receipts select
native Docker without claiming the Daytona lifetime guarantee.

When a legacy task omits resource requests, operators may provide an explicit,
complete `--cpus`, `--memory-mb`, `--storage-mb` override to `harbor-stage` (or a
`HarborResourceRequest` to the Python API). Partial, non-integer or above-ceiling
requests fail before materialization. The original package remains unchanged;
the staged task and receipt record both source and resolved requests. These are
allocation requests, not proof of every storage/quota enforcement capability or
of benchmark qualification. No resource default is silently invented for Daytona.

Docker prebuilt tasks also accept immutable local `sha256:<64 lowercase hex>`
image IDs. These references are scoped to the Docker daemon that owns the
image; they are not pullable registry manifest digests and Daytona rejects them.
Stage receipts distinguish `docker_local_image_id` from
`registry_manifest_digest`. Neither reference type by itself proves the image
was built from the bound task package: build provenance remains explicitly
unverified until a build receipt establishes that relationship.

## Native Docker resource custody (opt-in)

`--env synth_containers.harbor_docker:ObservedDockerEnvironment` selects the
Harbor 0.22 Docker extension from the `native-harbor-docker` extra. It records
an exclusive, fsynced creation claim and resource events in the trial directory,
uses a fresh Compose project owner, saves observed container/network/volume
handles before cleanup, and requires typed Docker absence plus an empty exact
owner query before confirming cleanup. Harbor's best-effort cleanup return does
not suffice. Ambiguous creation without an observed primary container remains
pending, including when a later list is empty.

Creation is bounded by the staged task's positive timeout (at most 300 seconds);
cleanup has a 60-second deadline after bounded discovery. It accepts only
prebuilt Linux single-container tasks, requires deletion, and refuses implicit
sidecar image builds. Tasks needing egress control must provide an immutable,
already prepared `egress_control_image` environment argument. Docker provider
TTL and hard workspace quota remain unsupported and explicitly reported as such.
Set the process `TMPDIR` to a run-owned directory for Harbor's temporary Compose
files. This extension is opt-in while benchmark front doors retain their pinned
runtime behavior; there is no automatic legacy lane migration.

Prebuilt staging distinguishes build context from runtime upload.
`environment_transfer="image_only"` (CLI `--environment-transfer image_only`)
retains an empty `environment/` directory after verifying the complete source
copy. It records every omitted build-context file in the receipt and preserves
the task, instruction and verifier files. This is explicit: the default
`preserve` behavior still uploads non-Dockerfile environment files. Image-only
mode asserts that the selected image already supplies its required runtime
files; it does not manufacture build provenance. CyberneticsBench uses this mode
because its image recipe bakes the environment payload and its verifier must not
find build-only/scorer-source markers copied into the agent workspace.
