# synth-containers skills

Portable agent skills for building and debugging public task containers with the
`synth-containers` contract in this repo.

- `containers/` — build, upgrade, review, or debug task containers (HTTP rollout
  contract, `/task_info`/`/program`/`/dataset` routes, verifier-backed tasks,
  agent-environment adapters).

Each skill is a folder with a required `SKILL.md` (Agent Skills format). Runnable
cookbooks that use these containers live in
[`synth-cookbooks-public`](https://github.com/synth-laboratories/synth-cookbooks-public).
