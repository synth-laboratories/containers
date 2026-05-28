# Implementation Handoff From Submission Review

This historical handoff seeded the public `synth-containers` package from three
private submission snapshots. Active development now lives in
`packages/synth-containers` in `synth-cookbooks-public`; keep this document as
design provenance, not as the current development runbook.

The original review compared these submissions:

- [containers_repo](containers_repo)
- [containers_repo-2](containers_repo-2)
- [implemented_synth_containers](implemented_synth_containers)

## Executive Summary

Use [containers_repo](containers_repo) as the base.

Cherry-pick:

- the stronger consumer-target framing, reference runtime, and fuller OpenAPI
  shape from [containers_repo-2](containers_repo-2)
- the compact compatibility-report and some task/catalog convenience patterns
  from [implemented_synth_containers](implemented_synth_containers)

Do not try to merge all three wholesale. `containers_repo` is the best starting
point because it already combines:

- ontology + protocols + profiles
- capability and tool-runtime surfaces
- Go-Explore / MIPRO contract-aware types
- a real reference HTTP adapter
- a real HTTP client
- framework descriptor surfaces
- a test shape that matches actual workflow semantics

## Final Direction

The local repo should become the canonical implementation of:

1. shared ontology nouns
2. primitive protocols
3. composed execution profiles
4. capability and metadata declarations
5. normalized wire contracts
6. reference runtime + reference HTTP adapter
7. consumer compatibility surfaces
8. framework adapter surfaces

The implementation should stay faithful to the design stance already written in:

- [ontology.md](ontology.md)
- [framework_comparison.md](framework_comparison.md)
- [algorithm_support.md](algorithm_support.md)

Core rule:

- semantics first
- protocols second
- transport third

## Recommendation By Submission

### 1. Base: `containers_repo`

Use this as the implementation base.

Best parts:

- strongest overall package breadth in
  [README.md](containers_repo/README.md:28)
- best reference HTTP surface, including:
  - `/metadata`
  - `/task_catalog`
  - rollout lifecycle
  - artifacts / events / trace
  - pause / checkpoint / resume / terminate
  in [http_adapter.py](containers_repo/src/synth_containers/http_adapter.py:119)
- strongest end-to-end workflow test in
  [test_http_reference_adapter.py](containers_repo/tests/test_http_reference_adapter.py:17)
- best Go-Explore / MIPRO-specific contract and recovery modeling in
  [test_go_ex_and_mipro_contracts.py](containers_repo/tests/test_go_ex_and_mipro_contracts.py:18)

Weaknesses to correct while implementing:

- slightly sprawling and still somewhat internal-Synth-shaped
- some package hygiene noise in the submission snapshot
- less crisp consumer-target reporting than submission 2/3

### 2. Cherry-pick: `containers_repo-2`

Use this as the source for conceptual refinements.

Best parts:

- best consumer-target assertions in
  [test_consumers_and_adapters.py](containers_repo-2/tests/test_consumers_and_adapters.py:29)
- best honest “not yet supported” treatment for token-level NanoLong RL in
  [test_consumers_and_adapters.py](containers_repo-2/tests/test_consumers_and_adapters.py:50)
- best self-contained reference runtime in
  [reference_runtime.py](containers_repo-2/src/synth_containers/reference_runtime.py:53)
- best async executor proving surface in
  [test_reference_runtime.py](containers_repo-2/tests/test_reference_runtime.py:46)
- best OpenAPI starting point in
  [container-contract-v1.yaml](containers_repo-2/openapi/container-contract-v1.yaml:1)

Weaknesses to avoid carrying over directly:

- README talks about `/metadata`, but OpenAPI only exposes `/info`
- no real reference FastAPI implementation to match the OpenAPI
- package is conceptually strong but less operationally complete than
  `containers_repo`

### 3. Cherry-pick selectively: `implemented_synth_containers`

Use this for some utility patterns, not as the base.

Best parts:

- compact compatibility-report API in
  [compatibility.py](implemented_synth_containers/src/synth_containers/compatibility.py:12)
- decent HTTP client shape in
  [client.py](implemented_synth_containers/src/synth_containers/client.py:1)
- some practical task/catalog convenience in
  [tasks.py](implemented_synth_containers/src/synth_containers/tasks.py:32)

Weaknesses:

- much weaker OpenAPI / wire completeness in
  [container-runtime-v0.1.yaml](implemented_synth_containers/openapi/container-runtime-v0.1.yaml:9)
- more “compatibility utility library” than full substrate
- ontology is a bit looser / less aligned with the current direction

## Implementation Plan For Local Repo

Implement in `packages/synth-containers`.

### Phase 1: Adopt the base package shape from `containers_repo`

Create local modules modeled primarily on:

- `src/synth_containers/ontology.py`
- `src/synth_containers/nouns.py`
- `src/synth_containers/protocols.py`
- `src/synth_containers/profiles.py`
- `src/synth_containers/capabilities.py`
- `src/synth_containers/tool_runtime.py`
- `src/synth_containers/proxying.py`
- `src/synth_containers/contracts.py`
- `src/synth_containers/recovery.py`
- `src/synth_containers/formats.py`
- `src/synth_containers/http_adapter.py`
- `src/synth_containers/http_client.py`
- `src/synth_containers/adapters.py`

Use the module boundaries from `containers_repo`, but clean them up where the
split is too Synth-internal.

### Phase 2: Replace the synthetic in-memory runtime with the better reference runtime pattern from `containers_repo-2`

Use the `CounterRuntime` and async executor ideas from:

- [reference_runtime.py](containers_repo-2/src/synth_containers/reference_runtime.py:53)

Bring over:

- reset / step / observe / read_state
- checkpoint / restore / fork
- trace / outcome
- tool surface
- multi-actor stubs
- async queued executor

But keep the HTTP adapter surface from `containers_repo`, not the thinner wire
surface from `containers_repo-2`.

### Phase 3: Use the OpenAPI from `containers_repo-2` as the spec base, then align the reference adapter to it

Use:

- [container-contract-v1.yaml](containers_repo-2/openapi/container-contract-v1.yaml:1)

Then explicitly reconcile it with the stronger reference HTTP adapter from
`containers_repo`.

Required outcome:

- the OpenAPI and the reference server must describe the same routes
- `/metadata` must be first-class, not just implied by `/info`
- `/task_info` and `/task_catalog` need to remain explicit
- checkpoint, resume, pause, terminate, artifacts, events, trace, summary, and
  usage routes must all be represented clearly

### Phase 4: Lift the consumer compatibility surface from submissions 2 and 3

Use:

- the target vocabulary and assertion style from
  [test_consumers_and_adapters.py](containers_repo-2/tests/test_consumers_and_adapters.py:29)
- the reporting ergonomics from
  [compatibility.py](implemented_synth_containers/src/synth_containers/compatibility.py:12)

Desired final shape:

- one canonical `ConsumerTarget` vocabulary
- one `CompatibilityReport`
- one consumer evaluation function
- one assertion helper

The system must be able to say both:

- `supported`
- `unsupported`

and explain why, including explicit missing features. Preserve the honest
reporting pattern for token-level RL support.

### Phase 5: Keep the stronger Go-Explore / MIPRO contract surfaces from `containers_repo`

Prefer the more explicit contract-aware pieces from:

- [test_go_ex_and_mipro_contracts.py](containers_repo/tests/test_go_ex_and_mipro_contracts.py:18)

The final local repo should keep first-class types for:

- Go-Explore-style container execution contract projection
- checkpoint/resume semantics
- artifact paths
- recovery projection
- task/runtime metadata that can be derived from existing Go-Ex and MIPRO
  surfaces

Do not reduce the repo to only an abstract environment ontology. It needs to
stay usable for those real consumers.

### Phase 6: Prefer the `containers_repo` HTTP client, but incorporate the ergonomic bits from `implemented_synth_containers`

Use `containers_repo` as the base because it matches the richer route set.

Cherry-pick from `implemented_synth_containers`:

- retry/backoff shape
- optional route handling ergonomics
- some method naming cleanup if it improves clarity

But do not regress the richer route coverage from repo 1.

### Phase 7: Keep task abstractions explicit and ready for future SQLite-backed evolution

Use the task/catalog convenience ideas from
[implemented_synth_containers/tasks.py](implemented_synth_containers/src/synth_containers/tasks.py:32),
but keep them aligned with the ontology direction already documented for the new
repo.

Important:

- do not lock the implementation to in-memory catalogs forever
- keep the shape evolvable toward a SQLite-backed task catalog
- keep task/data abstractions separate from runtime/execution abstractions

## Target Local Module Layout

This is the intended local package layout:

```text
src/synth_containers/
├── __init__.py
├── ontology.py
├── nouns.py
├── protocols.py
├── profiles.py
├── capabilities.py
├── tool_runtime.py
├── proxying.py
├── contracts.py
├── recovery.py
├── tasks.py
├── compatibility.py
├── wire.py
├── http_adapter.py
├── http_client.py
├── reference_runtime.py
└── adapters.py
```

Notes:

- `formats.py` from submission 1 can either remain separate or be folded into
  `wire.py` if that produces a cleaner API.
- if `models.py` is needed, keep it limited and avoid duplicating nouns
  unnecessarily.
- do not split files just because one submission did; split only when the
  boundary is conceptually real.

## What To Preserve Semantically

The final local implementation must preserve these ideas:

- ontology is a hierarchy, not one flat API
- fidelity matters:
  - native
  - derived
  - approximate
  - unsupported
- framework adapters must be honest about lossiness
- `/metadata` is for runtime/container capabilities
- `/task_info` is for task-scoped metadata
- `/task_catalog` remains explicit for catalog-backed systems
- checkpoints must declare semantics, not just existence
- RL support must distinguish:
  - no token traces
  - token ids only
  - token ids + logprobs
  - token ids + logits

## What To Avoid

Do not:

- flatten everything into one generic “container” object
- treat all checkpoints as equivalent
- treat all rewards as equivalent
- drop task/catalog abstractions just because some runtimes do not use them
- overfit the package to one current consumer
- make the OpenAPI less complete than the reference adapter

## Concrete Merge Recommendation

If implementing incrementally, proceed in this order:

1. start from `containers_repo` module boundaries and HTTP adapter
2. swap in `containers_repo-2` reference runtime and OpenAPI draft
3. add `implemented_synth_containers` compatibility-report ergonomics
4. reconcile naming and reduce duplication
5. only then refine docs and examples

## Success Criteria

The local repo should be considered successful when it can honestly express and
prove support for:

- Go-Explore
- MIPROv2
- standard eval harnesses
- pipeline RL without token traces
- pipeline RL with token ids / logprobs / logits
- proxying Harbor
- proxying OpenEnv
- proxying Archipelago

And when it contains:

- a clear ontology
- a clear capability vocabulary
- a clear wire contract
- a working reference runtime
- a working reference HTTP adapter
- a working client
- consumer compatibility reporting
- framework adapter reporting

## Quick Build Heuristic

When a decision is ambiguous, prefer the option that:

1. preserves semantics
2. makes unsupported features explicit
3. helps a real downstream consumer
4. keeps the repo usable as a public semver'd substrate

If a pattern only looks elegant but does not materially help Go-Ex, MIPRO,
evals, RL, or the external framework adapters, do not prioritize it.
