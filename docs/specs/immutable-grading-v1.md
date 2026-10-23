# Immutable grading v1

`synth.grading-plan.v1` pins an operation, exact sealed trace and optional frozen
submission, task/world/dataset/split, scorer program and optional image,
dependencies, rubric, aggregation, judge configuration/seed, permissions and
wall/CPU/memory/input/output/network/provider-cost limits. Plans accept JSON or
TOML. Labels and source folder names are not authority. Inputs declare exact
digest, size, media type, requiredness, visibility and retention root. Exact V5
selectors must resolve before compute. Missing/corrupt/inaccessible state, media,
files or rubric refuse; they never produce a zero score.

The first implementation is `trusted-offline-rubric-v1`: pinned owner-trusted pure
rubrics in a subprocess with sanitized environment, CPU/wall/RSS/output bounds,
private retained workspace and owned cleanup. It is not an untrusted-code or
network security sandbox. It never compiles or executes a submission. A container
image, arbitrary scorer, model judge, provider budget or untrusted program requires
`admitted-external-grader-v1` and an owner-provided isolated/admitted runner; the
local profile refuses that combination. Native frozen Python scorer source also
requires the owner's exact `--trust-source-digest`. A plan cannot grant its own
trust or credentials. No private sibling repository is required by the package.

Supported pure rubrics are exact JSON comparisons, captured Craftax trajectory
metrics and CardCodeBench's native captured-observation rules. The CardCodeBench
adapter verifies the frozen submission's native tree identity against the original
verification receipt, task identity, native observation bytes, exact planned case
coverage and a complete trusted native scorer source bundle. Private expected
authority never enters candidate execution. Regrading captured observations does
not establish fresh compilation, a new engine run, fresh heldout validity or
benchmark completeness. Existing native compilation/verifier adapters retain that
responsibility. The Craftax metric adapter refuses absent action prefixes, required
state/achievement facts or rewards. Different native containers can add external
admitted execution without weakening these guarantees.

## Identity and recovery

The operation's immutable plan and retained input closure are committed before
dispatch. A filesystem lock protects the local owning invocation; this is not
another scheduler. Each physical attempt has a new UUID and start receipt.
The worker commits its completed output before writing its pipe response. The
controller retains bounded stdout/stderr, elapsed time and completion facts before
sealing judgments. A completed operation is returned on replay; conflicting plans
under an operation ID refuse. A lost result/publication reply reuses committed
worker output. Unknown in-flight attempts refuse implicit recompute; an explicit
offline retry creates a new attempt. Provider judges need their own reconciliation
and admission and are not silently retried here.

If only worker completion survived, recovery keeps elapsed time and missing log
coverage unknown and records that worker exit was not observed. It never invents
an empty log or claims cleanup from a file's presence. Live owned workers stop on
Ctrl-C/timeout/output or memory limits; a worker's owner watchdog exits if the
controller disappears. CLI interruption retains inputs and partial captured logs.

Judgments, rubric/verifier definitions and execution results reuse V5 standards,
validation and rubric aggregation. Canonical trace bytes and prior judgments are
unchanged. New intentional operations coexist. Corrections require retained prior
evidence with the same subject and definition and explicitly supersede its verifier
result through the existing append-only evidence API. Changing rubric/reviewer
identity is an alternative interpretation. Selection names exact authoritative
result IDs; it never defaults to latest timestamp.

Raw named case scores remain distinct from the rubric-normalized aggregate. Reports
retain planned/completed/scored counts and missingness. Comparisons require matching
trace/submission/task/world/dataset/split identities, report common scored cases and
case deltas, and expose left/right-only or missing cases separately. Original
execution status and historical spend remain unchanged. Offline grading reports
zero provider calls; its CPU/time is a new grading attempt, not original rollout
spend.

## CLI

```
synth-containers regrade describe grading.toml --bundle retained-bundle
synth-containers regrade run grading.toml --bundle retained-bundle --state-root grades
synth-containers regrade inspect OPERATION --state-root grades
synth-containers regrade publish OPERATION --state-root grades --bundle retained-bundle
synth-containers regrade export OPERATION --state-root grades --destination fresh-bundle
```

Describe is provider-free and returns exact pins, access/sufficiency blockers and
the cost ceiling. Run allocates durable operation/attempt state. Inspect, publish
and export cannot call a grader or agent. Publication retries append the same
immutable evidence, copy digest-verified private inputs/logs and retain a new
manifest cut. Hosted publication/indexing uses the existing owner adapters and
trace publication outbox. A retained original bundle permits export after sandbox
cleanup; source paths are not needed. Exported grader assets remain private and
must receive the same owner-side retention/access policy as original evidence.

Exit codes: 0 completed/readable, 2 refused/preflight failure, 3 recorded grading
failure, 130 interrupted. JSON results and errors use separate stdout/stderr.
Grading success means the rubric executed; scores and scientific validity remain
separate facts.

## Observed Craftax trajectory profile

`observed_craftax_trajectory(document)` derives
`synth.craftax-observed-trajectory.v1` from a sealed V5 trace. It requires one
closed episode with an exact contiguous action prefix, an environment-authority
reward and post-action readout for every step, and the initial observation.
Duplicate steps, ambiguous event IDs, missing readouts, nonfinite rewards and
unknown prefixes refuse. Records retain the source trace digest and exact action,
reward and observation event IDs; readouts are copied without changing capture.

A grading plan can require `trace_bound_observation`. Preflight derives this
profile again from the original trace and compares the entire input, including
citations. Renaming an input or changing a value cannot establish lineage.
Legacy frozen trajectory inputs remain supported without this stronger claim.
Unknown required facts refuse. `full_state` remains unqualified: symbolic readouts
and achievement lists do not prove simulator, RNG or full world state. A future
qualified snapshot profile must define that evidence explicitly.

Raw reward can be negative. A rubric must declare a suitable score range; the
named action, achievement and reward scores remain separate from its normalized
aggregate. Different normalizations produce different rubric identities.
