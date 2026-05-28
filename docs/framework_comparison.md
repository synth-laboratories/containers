# Framework Comparison

This document compares the Synth `synth-containers` package against the main
historical and external references currently shaping the design.

## High-level comparison

| System | Primary abstraction | Strong alignment with Synth containers | Main divergence | Best ideas to borrow |
|---|---|---|---|---|
| `containers` | Layered runtime substrate: `Engine -> Environment -> Task/TaskCatalog -> Container Adapter` | Statefulness, task/data organization, blocking + async rollouts, RL outputs, semver'd package | Broader than the references by design | Capability tiers, checkpoint semantics, task catalog, metadata discovery |
| `Environments-old` | Stateful environment framework | Strong on `Engine + Environment`, checkpointing, reproducibility, typed task instances, metadata | Weak on modern rollout/job semantics and async long-horizon protocols | `Engine` truth vs `Environment` wrapper, `Impetus` vs `Intent`, snapshotting as a primitive |
| `Harbor` | Benchmark/eval/RL harness over container environments | Strong on rollout generation, orchestration, datasets, RL on tasks | Harness-centric rather than substrate-centric | Provider abstraction, registry ergonomics, many-agents/optimizers over one env substrate |
| `Archipelago` | Sandbox + MCP gateway + grading by before/after snapshot | Strong on container sandboxing, MCP, snapshot-based evaluation | Sandbox/world snapshot is the primitive, not engine/task | MCP gateway patterns, sandbox lifecycle, before/after grading |
| `OpenEnv` | Gym-style client/server environment API | Strong on `reset/step/state`, typed env interaction, RL friendliness | Weaker on task catalogs, splits, evaluator intent, rich checkpoint semantics | Client/server env split, typed `Action/Observation/State`, env scaffolding |

## Alignment by idea

| Idea | `containers` | `Environments-old` | `Harbor` | `Archipelago` | `OpenEnv` |
|---|---|---|---|---|---|
| `Engine + Environment` split | Core | Core | Not central | Not central | Partial |
| Typed task instances / metadata / splits | Core | Strong | Moderate | Weak | Weak |
| Stateful execution | Core | Core | Moderate | Moderate | Core |
| Checkpoint / resume / branching semantics | Core | Partial-strong | Moderate | Snapshot-oriented | Partial |
| Blocking short-horizon rollouts | Core | Weak | Strong | Moderate | Strong |
| Async long-horizon rollouts | Core | Weak | Strong | Moderate | Partial |
| Multi-agent support | Planned core capability | Weak | Strong | Moderate | Partial |
| MCP / tool gateway patterns | Supported capability | Weak | Moderate | Strong | Partial |
| RL token ids / logits / action traces | Planned core capability | Weak | Moderate | Weak | Moderate |
| Harness / benchmark orchestration | Downstream consumer layer | Weak | Core | Strong | Weak |

## Design takeaway

The Synth `synth-containers` package should:

- take the bottom-of-stack ideas from `Environments-old`
- take typed environment interaction instincts from `OpenEnv`
- take orchestration/provider ideas from `Harbor`
- take MCP sandbox and snapshot grading ideas from `Archipelago`

It should not copy any one of those systems wholesale.
