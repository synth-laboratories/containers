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

The qualification scope is one deterministic oracle task with shared verification.
This extension does not yet establish full Codex capture, isolated verifier parity,
cloud journal custody, pre-provision database recovery, experiment dollar budgets,
or full Docker/direct-Daytona/hosted executor parity. Do not advertise those
capabilities from this extension's resource bounds.
