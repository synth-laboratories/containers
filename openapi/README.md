# OpenAPI

This folder contains the versioned public wire contract for the reference
runtime surface:

- [container-contract-v1.yaml](container-contract-v1.yaml)

The v1 contract is based on the stronger repo-2 draft, then reconciled against
the repo-1 reference FastAPI adapter so they describe the same first-class
surfaces:

- `/metadata`, `/info`, and `/compatibility`
- `/task_info` and `/task_catalog`
- request-aware `/task_info` selectors for seed, split, family, task id, and
  task instance id
- `resource_refs` for task info, task catalog, tasks, instances, and heavy
  data/runtime/evaluation/config references
- rollout submit/state/summary/usage/events/trace/artifacts
- pause/terminate/checkpoint/resume lifecycle controls
- rollout-scoped and global checkpoint discovery
