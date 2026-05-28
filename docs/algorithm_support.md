# Algorithm Support Targets

This repo exists to support concrete algorithm families and benchmark harnesses.

## Primary algorithm targets

### 1. Go-Explore / `go_ex`

What it needs:

- stateful long-horizon runtimes
- checkpoint creation
- restore/resume/fork semantics
- async rollout execution
- trace and artifact emission
- explicit tool/runtime capability metadata

Most relevant protocol profiles:

- `CheckpointableLongHorizonEnvironment`
- `MultiAgentLongHorizonEnvironment` for future extensions

### 2. MIPROv2

What it needs:

- short-horizon request/response evaluation
- task metadata and candidate evaluation semantics
- verifier/grade outputs
- stable artifacts
- eventually long-horizon/session runtimes as an extension

Most relevant protocol profiles:

- `StatelessEvaluator`
- `HarnessManagedBenchmarkEnvironment`

### 3. Standard eval harnesses

What they need:

- task catalogs
- reproducible task instances
- blocking and async rollout execution
- artifacts, traces, and verifier outputs

Most relevant protocol profiles:

- `StatelessEvaluator`
- `HarnessManagedBenchmarkEnvironment`

### 4. Pipeline RL without token-level traces

What it needs:

- reset/step/state or equivalent rollout semantics
- trajectories
- rewards or outcome signals
- async rollout collection at scale
- reproducible task instances and optionally checkpoints

Most relevant protocol profiles:

- `RLTrajectoryEmitter`
- `CheckpointableLongHorizonEnvironment`

### 5. Pipeline RL with token ids / logprobs / logits

What it needs beyond normal pipeline RL:

- token ids
- logprobs and/or logits
- actor/policy provenance
- possibly proxied inference metadata
- trace semantics rich enough for replay and training

Most relevant protocol profiles:

- `TokenLevelRLEnvironment`

## Primary adapter targets

These are the frameworks we want to proxy or wrap through the shared substrate.

- `Environments-old`
- `OpenEnv`
- `Harbor`
- `Archipelago`

## Success criterion

The design is only good enough if we can actually test these targets and show:

- the ontology is expressive enough
- the capability declarations are honest
- the adapters are not excessively lossy
- the required protocol profiles can be implemented in practice

This means the repo should eventually carry explicit adapter smoke plans for:

- `go_ex`
- `miprov2`
- standard eval
- pipeline RL without token-level traces
- pipeline RL with token-level traces
- proxying `Environments-old`, `OpenEnv`, `Harbor`, and `Archipelago`

