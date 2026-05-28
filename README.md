# containers

Synth task containers — HTTP services that expose tasks, datasets, rollouts, and program overlays for use with `synth-laboratories/optimizers` (GEPA and friends) and the broader Synth platform.

## Status

Initial scaffold. Container sources and task definitions land in subsequent commits.

## Layout (forthcoming)

- `src/synth_containers/` — shared Python package for building a container (FastAPI app, rollout runner, program overlay machinery).
- `tasks/` — one directory per task (MiniGrid, Crafter, HotpotQA ReAct, …), each producing a runnable HTTP container.
- `openapi/` — generated OpenAPI specs for the container HTTP surface.
- `docs/` — usage notes and per-task READMEs.

## Related repos

- [synth-laboratories/optimizers](https://github.com/synth-laboratories/optimizers) — GEPA and other prompt/program optimizers that drive these containers.
