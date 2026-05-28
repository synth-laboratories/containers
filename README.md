# Synth Containers

Shared Synth container and environment runtime substrate. Canonical home for the
`synth-containers` package and reference task containers consumed by
[`synth-laboratories/optimizers`](https://github.com/synth-laboratories/optimizers)
(GEPA and friends) and the broader Synth platform.

This package is the semver-friendly home for the shared ontology, protocols,
profiles, capability vocabulary, reference HTTP transport, and adapter
comparison surface needed to make real container consumers interoperate across:

- Go-Explore / `go_ex`
- MIPROv2
- standard eval harnesses
- pipeline RL without token traces
- pipeline RL with token ids / logprobs / logits
- Harbor / OpenEnv / Archipelago proxy adapters

## Design stance

The package stays layered on purpose:

1. semantics first
2. protocols second
3. transport third

The goal is not one flat API that erases every runtime’s meaning. The goal is a
shared substrate with explicit fidelity, checkpoint semantics, reward semantics,
and runtime/tool/proxy declarations.

## Implemented package surface

`src/synth_containers/` now includes:

- `ontology.py`
  - contract version
  - core nouns
  - primitive protocols
  - composed profiles
  - checkpoint / resume / reward vocabulary
- `nouns.py`
  - task, actor, action, observation, state, checkpoint, artifact, trace,
    verifier, outcome, trajectory, execution records
- `protocols.py`
  - runtime-checkable protocol definitions for the primitive behaviors
- `profiles.py`
  - profile specs, profile inference, and missing-protocol reporting
- `capabilities.py`
  - canonical capability surface, route hints, token emission support, task info,
    task catalog, runtime metadata
- `tool_runtime.py`
  - tool-runtime / schema / output-mode compatibility surface
- `proxying.py`
  - inference target and proxy/runtime routing declarations
- `contracts.py`
  - Go-Explore / long-horizon shared contracts and stable artifact path surface
- `recovery.py`
  - recovery projection for resumable / replayable / restartable runs
- `formats.py`
  - canonical HTTP payload formatting for metadata, task info, rollout, and
    execution state
- `wire.py`
  - rollout lifecycle and submission-mode helpers for stable async semantics
- `http_adapter.py`
  - reference FastAPI adapter over the normalized runtime surface
- `http_client.py`
  - async client with retry/backoff and optional-route handling for the
    reference HTTP contract
- `reference_runtime.py`
  - concrete counter runtime + async queued executor + managed runtime adapter
- `compatibility.py`
  - canonical consumer-target compatibility reporting and assertions
- `tasks.py`
  - task/catalog convenience layer designed to stay evolvable toward persistent
    catalogs
- `adapters.py`
  - framework fidelity descriptors for `Environments-old`, `OpenEnv`,
    `Archipelago`, and `Harbor`

## Core ontology

The deepest shared nouns are:

- `Runtime`
- `Actor`
- `Action`
- `Observation`
- `State`
- `Execution`
- `Outcome`
- `TaskInstance`

First-class surrounding nouns include:

- `Task`
- `TaskCatalog`
- `Artifact`
- `Trace`
- `Checkpoint`
- `Trajectory`
- `Tool`
- `VerifierResult`
- `AgentSession`

## Primitive protocols

The package defines and types the shared protocol vocabulary for:

- catalog-backed
- resettable
- steppable
- observable
- state-readable
- checkpointable
- restorable
- forkable
- rollout-runnable
- async-rollout-runnable
- trace-emitting
- reward-emitting
- verifier-backed
- tool-callable
- token-trace-emitting
- multi-actor
- proxied-inference-backed

## Composed profiles

Implemented profiles:

- stateless evaluator
- gym-style environment
- checkpointable stateful environment
- checkpointable long-horizon environment
- multi-agent long-horizon environment
- sandboxed MCP world
- RL trajectory emitter
- token-level RL environment
- harness-managed benchmark environment

Profiles can be inferred from declared protocol fidelity, and missing protocol
requirements can be reported explicitly.

## Canonical capability surface

The normalized capability surface includes:

- contract version
- runtime kind
- rollout modes
- statefulness tier
- noun / protocol / profile fidelity
- checkpoint semantics
- restore / resume semantics
- branching support
- true-environment-snapshot support
- artifact / trace / reward / verifier support
- tool runtime declarations
- token / logprob / logits support
- route hints for discovery and control operations

## Reference HTTP surface

The reference FastAPI adapter implements:

- `GET /`
- `GET /health`
- `GET /metadata`
- `GET /info`
- `GET /task_info`
- `GET /task_catalog`
- `POST /rollout`
- `POST /rollouts`
- `GET /rollouts/{rollout_id}`
- `GET /rollouts/{rollout_id}/state`
- `GET /rollouts/{rollout_id}/summary`
- `GET /rollouts/{rollout_id}/usage`
- `GET /rollouts/{rollout_id}/artifacts`
- `GET /rollouts/{rollout_id}/events`
- `GET /rollouts/{rollout_id}/trace`
- `POST /rollouts/{rollout_id}/pause`
- `POST /rollouts/{rollout_id}/terminate`
- `GET /rollouts/{rollout_id}/checkpoints`
- `POST /rollouts/{rollout_id}/checkpoints`
- `GET /rollouts/{rollout_id}/checkpoints/{checkpoint_id}`
- `GET /checkpoints`
- `GET /checkpoints/{checkpoint_id}`
- `POST /checkpoints/{checkpoint_id}/labels`
- `POST /rollouts/{rollout_id}/resume`

The rollout formatter is intentionally trainer-friendly:

- `artifact[].turns`
- `trace.event_history`
- `trace.inference.turns`
- `reward_info.outcome_reward`
- per-turn `event_rewards`
- optional token-id / logprob / top-logprob data

## Validation coverage

The package surfaces are structured to be directly verifiable for:

- profile inference and capability validation
- Go-Explore contract/recovery projection
- MIPROv2 evaluator compatibility
- RL payloads with and without token-level traces
- HTTP lifecycle operations including pause/checkpoint/resume/terminate
- framework compatibility for `Environments-old`, `OpenEnv`, `Archipelago`,
  and `Harbor`

## References informing the design

- old Synth environments / Horizons legacy
- current Synth container contracts in `synth-lab`
- current eval long-horizon wrapper surface
- NanoLong long-horizon RL container requirements
- Harbor
- OpenEnv
- Archipelago

See also:

- [Ontology Draft](docs/ontology.md)
- [Framework Comparison](docs/framework_comparison.md)
- [Algorithm Support Targets](docs/algorithm_support.md)
