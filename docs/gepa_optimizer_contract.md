# GEPA Optimizer Contract

`synth-containers` advertises public optimizer contracts in runtime metadata.
GEPA v1 uses:

```json
{
  "metadata": {
    "optimizer_contracts": {
      "gepa": {
        "version": "synth_optimizers.gepa.v1",
        "program_route": "/program",
        "dataset_route": "/dataset",
        "dataset_rows_route": "/dataset/rows",
        "rollout_route": "/rollout"
      }
    }
  }
}
```

## Contract Nouns

- `PromptProgram`: container-declared prompt program and mutable modules.
- `PromptModule`: prompt field that can appear in a candidate payload.
- `TargetModule`: module that the optimizer is allowed to change.
- `PromptCandidatePayload`: candidate field map.
- `CandidateOverlay`: rollout-time candidate overlay.
- `DatasetRows`: stable row payload fetched through the container.
- `RolloutResult`: standard rollout response with reward, usage, trace, and
  artifacts.

## Boundary Rules

- GEPA core is domain-agnostic. It must not branch on cookbook names, task
  names, labels, tools, achievements, datasets, or environment-specific
  semantics.
- The container owns task execution and scoring. The optimizer consumes typed
  `DatasetRows`, candidate overlays, `RolloutResult` rewards, traces, usage,
  and artifacts.
- The container must expose every mutable prompt field through `/program`.
  Candidate proposals may only rewrite declared mutable `candidate_field`
  values.
- Seed candidates must come from `[seed_candidate]` in the run config or
  `/program.seed_candidate`. Optimizers must not synthesize seeds from module
  content.
- Rollout rewards are required. A missing reward is a contract error, not a
  zero score.
- Domain-specific objectives and verifier judgments must be explicit typed
  outputs from the container or a configured verifier service. They must not be
  inferred from example text or hard-coded into GEPA.

## Routes

### `GET /program`

Returns the prompt program.

Required top-level fields:

- `version`
- `program_id`
- `modules`
- `target_modules`
- `seed_candidate`

### `GET /dataset`

Returns dataset metadata, split names, row counts when known, and filter hints.

### `POST /dataset/rows`

Returns concrete rows by split and seeds. The optimizer never imports the
dataset directly when this route is available. Every returned row must be an
object with a stable `example_id` or `id`; a request for `N` seeds must return
`N` rows or fail.

### `POST /rollout`

Executes one candidate on one task row or seed. The rollout request uses the
normal `synth-containers` request model and may include `candidate`. The result
must include a numeric `reward_info.outcome_reward`, `summary.outcome_reward`,
or `reward`.

## Compatibility Rules

- Existing routes stay valid.
- Containers that do not advertise GEPA are still valid containers.
- Optimizers must fail early when a requested GEPA run points at a container
  without `metadata.optimizer_contracts.gepa.version`.
- Contract version changes must be additive or use a new version string.
