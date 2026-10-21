# Portable evaluation CLI v1

The shared contract is `synth.evaluation-plan.v1`, a digest-bound run declaration,
`evals.event-stream.v1` events and `synth.evaluation-result.v1` retained results.
The plan accepts TOML/JSON, repeatable `--candidate LABEL=PATH`, repo-relative
candidate presets, exact effective configuration and immutable source/image/
dependency/scorer identities. Labels are presentation. Describe reports actual
scripted provider/model/effort, seeds/replicates, parallelism, resource bounds,
capture/annotation policy, capabilities and aggregate provider maximum before any
compute. These initial profiles use no credentials or paid calls; a credential
or unknown required provider capability refuses. No annotation model runs hidden
behind capture. Hosted identity, per-org grants and secret resolution belong in
the Rhodes/SDK adapter, not a second CLI scheduler.

## Supported profiles

| Adapter | Execution | Continuation | Boundary |
| --- | --- | --- | --- |
| `local-gold-scripted-v1` | Real pinned Gold HTTP engine image, scripted actions | Native world/RNG + scripted history/queued-action checkpoints | Craftax qualification; other images must implement this exact protocol and be explicitly admitted as another benchmark profile |
| `retained-grading-v1` | Frozen grading plan and retained bundle | Completed-result/publication retry; no agent continuation | Exact-comparison, captured CardCodeBench native rubric and captured Craftax metrics |
| `rhodes-v1` | Refuses locally until the service adapter is installed | Service-owned | Declaration never grants hosted capacity or permissions |

This does not provide arbitrary process exec, model-policy support, detached local
jobs, generic untrusted container execution or universal benchmark validity. Those
capabilities must be declared by an adapter, with preflight refusal otherwise.
Pinned local image IDs are local configuration identities, not registry manifest
digests. Docker starts the supplied image without building it. Candidate code is
never compiled/executed on the host. The retained-grading profile is independently
bounded and cannot become original-agent generation.

## Lifecycle and operator views

```
synth-containers eval example --image sha256:... --directory example
synth-containers eval describe example/evaluation.toml --candidate scripted=scripted.toml
synth-containers eval run example/evaluation.toml --candidate scripted=scripted.toml --root runs/run-1
synth-containers eval watch runs/run-1 --follow
synth-containers eval results runs/run-1
synth-containers eval cancel runs/run-1
synth-containers eval resume runs/run-1
synth-containers eval fork runs/run-1 --destination runs/child --run-id child --child-budget child-budget.json
synth-containers eval export runs/run-1 --destination archives/run-1
```

`eval identities` reports the installed execution-code and dependency digests.
`eval schema` returns the machine-readable contract. The example is generated from
the installed package and contains no dated owner-local paths or private sibling
repository assumptions. A retained-grading plan locates its grading inputs
explicitly; it does not discover private assets from ambient sibling paths.

Foreground execution owns an OS-locked run directory and a separate engine guard.
Only the controller holds the guard's stdin write end. Controller exit, Ctrl-C or
process death closes it, and the guard removes that exact owned container. The
guard records cleanup; missing cleanup confirmation refuses. Host/engine failure
still requires reconciliation and does not become proof of cleanup. Foreign jobs
and shared engines are untouched. A detached local request refuses before launch.
Cancellation is a durable request observed at safe native boundaries, distinct
from confirmed cancellation/cleanup. Watch is read-only and never controls lifetime.

The terminal renders at most ten rollouts with aligned status/progress/latest
columns. Short terminals show fewer; visibility never limits execution concurrency.
Plain and JSON views consume the same durable events. JSON contains no ANSI bytes;
machine results live separately in `summary.json` and per-lane `job_result.json`.
Observer pipe failure detaches that view without cancelling computation. Events
append/fsync before observers receive them. Native action records and exact bytes
remain retained independently; V5 traces explicitly declare null-field projection
loss and embed the original journal body.

Exit codes are 0 completed execution, 2 preflight/refusal, 3 recorded execution
error, and 130 interruption. Benchmark score/validity remains distinct: successful
execution may produce a failed scientific result or partial/unscored evidence.
`matrix_run_manifest.json`, `events.jsonl` and per-lane `job_result.json` retain the
existing Evals folder/index contract. The thin Evals consumer uses LaneProjector
and best-effort result indexing. It does not write another result authority.

## Strict continuation and forks

Each native action has a durable dispatch intent and an action/state/reward/native
event record with a chained digest. Periodic joint checkpoints use the existing
ApplicationCheckpoint and portable manifest contracts: exact world/RNG/native
events, policy history/queued actions/cursor, source/image/dependencies/config/
scorer/protocol, cadence, time policy and cumulative consumed/outstanding usage.
The foreground local owner commits blobs and exported manifest before its separate
commit anchor. Those local anchors are not hosted fences or execution grants.

Strict recovery selects the greatest committed nonterminal position and refuses
ambiguous, missing, corrupt or incompatible history. It verifies and restores a
fresh engine, checks policy history and reconciles each committed journal-suffix
action against its exact native state/reward digest before new dispatch. Replay
reconstructs a fresh private deterministic world; it is not a promise that arbitrary
external effects can be repeated. An intent without a committed action stays
counted as a consumed dispatch. Reconstructing a lost deterministic action can
require another physical dispatch. Provider billing is never called exactly-once.

Active-runtime and absolute-deadline policies differ. Strict recovery preserves
limits, configuration and cadence. Known time/calls never reset. An abruptly dead
attempt's unobserved active time stays unknown: the active-runtime profile refuses
until owner reconciliation. Absolute-deadline recovery retains its original expiry
and may proceed only before it; downtime never renews admission. Native proof uses
that explicit profile. A terminal snapshot cannot become resumable.

Forks select committed parent cuts, retain parent lineage, allocate new run/lineage
and an explicit child budget authority/reference, and copy complete checkpoint
closure into isolated mutable child custody. Same-policy and edited-policy forks
are explicit. Only policy identity may change in an edited-policy fork; world,
scorer, source, protocol, limits and cadence stay compatible. Child funding does
not erase inherited usage. Local owner references do not authorize hosted spend.
Future restores from a child-owned checkpoint use strict continuation semantics.

Legacy reward/frontier checkpoint search remains separate. Legacy platform ring
pruning respects registered continuation/fork/archive roots and the immediate
resume parent. Root registration survives process restart, and release requires
the exact owning reference. Portable native exports are self-contained; this CLI
does not prune them. Hosted custody owners must protect equivalent roots before
performing hosted garbage collection.
