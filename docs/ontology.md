# Ontology Draft

This repo is organized around a layered ontology:

1. core nouns
2. primitive protocols
3. composed profiles
4. capability declarations
5. adapters and transports

The core design rule is:

- semantics first
- protocols second
- transport third

The repo should not start from one flat HTTP API and force every framework into
it. Instead, transports should sit on top of shared ontology and capability
declarations.

## Most foundational shared nouns

The smallest shared semantic layer currently looks like:

- `Runtime`
- `Actor`
- `Action`
- `Observation`
- `State`
- `Execution`
- `Outcome`
- `TaskInstance`

These span:

- `Environments-old`
- `OpenEnv`
- `Harbor`
- `Archipelago`
- future Synth container adapters

### Why these nouns

- `Runtime`
  - the thing being acted on over time
  - environment, sandbox, world, session, harness-backed runtime

- `Actor`
  - the entity taking actions
  - agent, policy-driven runtime actor, human proxy, multi-agent role

- `Action`
  - one intervention applied by an actor

- `Observation`
  - actor-visible output after reset/step/query/interaction

- `State`
  - authoritative runtime state, not necessarily fully visible

- `Execution`
  - the most abstract noun over:
    - rollout
    - session
    - step loop
    - benchmark run

- `Outcome`
  - the most abstract noun over:
    - reward
    - score
    - grade
    - verifier result
    - pass/fail

- `TaskInstance`
  - one concrete runnable unit

## Near-universal but not deepest-core nouns

These should still be first-class in the repo, but are one ring out from the
deepest common substrate:

- `Task`
- `TaskCatalog`
- `Artifact`
- `Trace`
- `Checkpoint`
- `Trajectory`
- `Tool`
- `AgentSession`

## Primitive protocols

These are minimal behavioral interfaces, not products.

- `CatalogBacked`
- `Resettable`
- `Steppable`
- `Observable`
- `StateReadable`
- `Checkpointable`
- `Restorable`
- `Forkable`
- `RolloutRunnable`
- `AsyncRolloutRunnable`
- `TraceEmitting`
- `RewardEmitting`
- `VerifierBacked`
- `ToolCallable`
- `TokenTraceEmitting`
- `MultiActor`
- `ProxiedInferenceBacked`

## Composed profiles

These are the human-meaningful protocol bundles that turn the ontology into a
framework typology.

- `StatelessEvaluator`
- `GymStyleEnvironment`
- `CheckpointableStatefulEnvironment`
- `CheckpointableLongHorizonEnvironment`
- `MultiAgentLongHorizonEnvironment`
- `SandboxedMCPWorld`
- `RLTrajectoryEmitter`
- `TokenLevelRLEnvironment`
- `HarnessManagedBenchmarkEnvironment`

## Capability fidelity

Not every adapter should claim full fidelity. Every mapping should eventually
be marked as one of:

- `native`
- `derived`
- `approximate`
- `unsupported`

That lets the ontology act as both:

- an interface layer
- a typology over frameworks

without pretending that all systems support the same semantics at the same
quality.

## Next ontology questions

- Should `Execution` split into `Rollout`, `Session`, `Episode`, and `EvalRun`
  as sibling nouns under a common parent?
- Should `Outcome` remain abstract, or should `Reward`, `Grade`, and
  `VerifierResult` become stronger first-class top-level refinements?
- Should `ToolCall` be its own noun separate from `Action`?
- Should `Checkpoint` and `StateSnapshot` be separate nouns?
- Should `Actor` and `Policy` be separate nouns?
