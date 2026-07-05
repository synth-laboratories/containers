# Containers TODO

Backlog for cross-container contract and SDK work. Task-specific notes belong in
each container repo; this file tracks general standards for `synth-containers`.

---

## Optional rollout limits and progress on state

**Status:** implemented for Crafter gold (python + rust lanes) and GRPO trainer.

`GET /rollouts/{rollout_id}/state` includes optional `limits` and `progress` when
an agent-hosted client wires them through. Crafter gold accepts `limits` on
`POST /rollouts` and agent counters via `POST /rollouts/{id}/progress`.

**Follow-up work:**

- [x] Crafter gold state + progress endpoint
- [x] GRPO trainer push + state poll during live rollouts
- [x] Watch dashboard `RolloutProgress` events
- [ ] Document canonical keys in OpenAPI (`RolloutStateResponse`)
- [ ] Extend `synth-containers` SDK helpers for rollout limits/progress
- [ ] Capability flag when a container supports polled rollout budgets

**Reference:** `gamebench/tasks/crafter-singleplayer/shared/http_contract.md`
