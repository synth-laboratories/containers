---
name: synth-containers
description: Use when building, upgrading, reviewing, or debugging public Synth task containers using the synth-containers contract, including GEPA-compatible HTTP containers, rollout APIs, task_info/program/dataset routes, verifier-backed tasks, and agent-environment containers such as MiniGrid, TBLite, Crafter, Harbor, OpenEnv, or Archipelago adapters.
---

# Synth Containers Skill

Use this skill when the user asks to build or adapt a task container for public
Synth cookbooks. The goal is a real, inspectable HTTP task boundary that can be
used by optimizers, eval harnesses, and agents without depending on private
services.

## First Files

Read only what is needed for the specific container:

- `README.md` for the canonical contract surface.
- `src/synth_containers/README.md` for package
  ownership boundaries.
- `src/synth_containers/formats.py` when response
  formatting needs to match the shared wire shape.
- `src/synth_containers/http_adapter.py` when using
  the reference FastAPI adapter.
- Existing nearby containers in `synth-cookbooks-public/cookbooks/optimizers/gepa/*_container/` when
  the user is building a GEPA-compatible container.

## Design Boundary

The container owns task semantics. It should load data, call policy models when
needed, execute the environment or verifier, calculate scores/rewards, and
return public-safe evidence. Optimizers and trainers should interact with it
only through HTTP and candidate overlays.

Keep these boundaries explicit:

- Container: task instances, observations, actions, rollouts, rewards,
  verifier results, traces, artifacts, task metadata, retries at the task edge.
- Optimizer/trainer: candidate generation, prompt overlays, scheduling,
  budget accounting, cache policy, acceptance decisions.
- External provider: policy/proposer inference APIs, remote execution services,
  model auth. Do not hide provider-specific semantics in generic route fields.

### Task container vs proposer substrate

These are separate concerns. Do not conflate them in docs, TOML, or skills.

- **Task container** (this repo): HTTP `/rollout`, `/task_info`, `/program`, etc.
  GEPA cookbook configs usually launch it locally:

  ```toml
  [container]
  command = ["uv", "run", "--project", ".", "python", "synth_service_app.py"]
  cwd = "."
  ```

  Dockerizing the task server is optional and independent of GEPA proposer config.
  Preferred eval packaging is a **local** catalog image (no GHCR):

  ```python
  from synth_containers import ContainerRunner

  handle = ContainerRunner(image_id="banking77", catalog="evals/containers/images").serve()
  ```

  CLI: `synth-containers serve --target banking77_classify` (duck) or
  `synth-containers up banking77 --catalog evals/containers/images` (local image).
  `url` / `command` still work. Harbor trial images are not this path.

- **Proposer substrate** (optimizers repo): where the Codex app-server runs during
  reflective prompt optimization — `runtime_substrate = "local"` (default) or
  `"docker"` with `[proposer.docker].image`. Staging, bind mounts, and Codex CLI
  isolation live in `synth_optimizer_platform` agent runtime, not in
  `synth-containers`.

- **Policy credentials**: rollout models read keys from the task container
  environment (`OPENAI_API_KEY`, etc.). That BYOK boundary is unrelated to the
  proposer docker image, which bundles Codex CLI for proposal generation only.

Vocabulary alignment (names only — containers does not launch GEPA proposers):

- optimizers `runtime_substrate` — local host vs docker for Codex proposer process
- optimizers `sandbox_mode` — Codex in-agent filesystem/sandbox policy
- containers capability metadata `sandbox_profile` — task/environment sandbox hints
  for agents reading `/metadata`; not the GEPA proposer launch substrate

For proposer docker quickstarts and acceptance commands, see
`optimizers/skills/gepa/SKILL.md` and `optimizers/README.md` (Authentication and
models → Docker proposer substrate). The proposer image Dockerfile currently lives
in `optimizers/docker/codex-gepa-proposer/`, not in this repo.

## Required HTTP Routes

For GEPA and most public optimizer consumers, implement these routes:

- `GET /health`: process readiness. Return 200 only after heavy assets,
  datasets, and model clients are initialized enough for first use.
- `GET /metadata`: protocol, route inventory, runtime kind, task id, and
  capability hints.
- `GET /task_info`: rich task description for agents/proposers.
- `GET /program`: prompt program, mutable fields, objectives, and candidate
  overlay schema.
- `GET /dataset`: split names, sizes, seed behavior, and sampling notes.
- `POST /dataset/rows`: resolve requested split/seeds to concrete public-safe
  row records.
- `POST /rollout`: execute one blocking rollout and return scores plus evidence.

Optional async/lifecycle routes:

- `POST /rollouts`
- `GET /rollouts/{id}`
- `GET /rollouts/{id}/state`
- `GET /rollouts/{id}/summary`
- `GET /rollouts/{id}/usage`
- `GET /rollouts/{id}/artifacts`
- `GET /rollouts/{id}/events`
- `GET /rollouts/{id}/trace`
- `POST /rollouts/{id}/pause`
- `POST /rollouts/{id}/terminate`
- checkpoint and resume routes when the runtime truly supports them

Do not claim checkpoint/resume/terminate support unless the implementation is
real. Capability metadata should be conservative.

## `/task_info` Content

`/task_info` is what lets a general proposer understand a new task. Include:

- `task_id`, `task_family`, and a short objective.
- Domain description and what the policy sees.
- Output contract: exact labels, answer type, action format, patch format, or
  tool-call schema.
- Evaluation metric and how partial credit works.
- Dataset split and seed semantics.
- Public-safe examples or edge cases when allowed.
- Failure modes that a prompt optimizer should address.
- Proposal guidance: which strategies are valid, which are overfitting, and
  what should not be copied from train examples.
- Constraints: allowed actions, time limits, token limits, sandbox rules,
  verifier limits, safety constraints, and formatting constraints.

For closed-output tasks, include the full output space or a route to it. For
open-output tasks, describe answer procedures and overfitting hazards instead
of leaking literal train targets into reusable prompts.

## `/program` Shape

Return a prompt-program object that makes mutability explicit:

- Stable program id and human-readable name.
- Mutable prompt fields, for example `stage2_system` or
  `react_system_prompt`.
- Seed candidate payload keys that exactly match mutable fields.
- Scoring objectives and primary objective.
- Candidate overlay schema: how an optimizer replaces mutable fields during
  rollout.
- Any policy configuration that is task-relevant but not secret.

Keep mutable fields narrow. Do not make environment constants, verifier code,
or dataset selection mutable unless the optimizer is explicitly meant to change
them.

## `/dataset` And Seeds

Prefer deterministic seed-to-row resolution:

- `GET /dataset` declares train/heldout splits, row counts, and seed policy.
- `POST /dataset/rows` accepts split plus explicit seeds and returns rows in a
  stable order.
- Preserve arbitrary requested seeds when the optimizer uses seed ids as stable
  evidence keys. Avoid modulo-folding externally supplied seeds unless the
  route clearly returns both original and resolved ids.
- Keep train and heldout sampling independent. Report sampler names and shuffle
  seeds in run logs or metadata when using random selection.

Rows should carry enough public-safe context for debugging and proposer
evidence: row id, seed, input text/observation/task spec, expected output or
verifier target when allowed, and lightweight metadata.

## `/rollout` Response

Each rollout should return:

- Candidate id or overlay id when supplied.
- Row id/seed/split.
- Scores keyed by objective name.
- Primary reward/score.
- Prediction, action trace, patch, answer, or final output.
- Expected output or verifier target when public-safe.
- Failure reason and verifier details when failed.
- Usage: tokens, model calls, wall time, external cost when available.
- Trace/events/artifacts references when produced.
- Task-specific evidence that helps a proposer infer why wins/losses differ.

Use real verifiers and real environments. Do not replace the task with string
fixtures unless the cookbook is explicitly a tiny contract smoke test.

## FastAPI Implementation Notes

For blocking live environments, prefer a synchronous FastAPI handler:

```python
from fastapi import Body, FastAPI

app = FastAPI()

@app.post("/rollout")
def rollout(payload: dict | None = Body(default=None)) -> dict:
    request = payload or {}
    return run_rollout(request)
```

Do not put heavy CPU-bound or blocking environment work inside `async def`
handlers unless it is explicitly moved to a thread/process executor. An async
handler that blocks the event loop will serialize apparently concurrent
rollouts and cause client timeouts.

## Dependency And Launch Pattern

Each container should own a local `pyproject.toml` if it has task-specific
dependencies. Keep install scope narrow:

- `banking77_container`: classifier dependencies only.
- `hotpotqa_container`: datasets, FastAPI, OpenAI client.
- `tblite_container`: pytest/verifier dependencies.
- `minigrid_container`: gymnasium/minigrid dependencies.
- `crafter_container`: craftax/jax dependencies.

GEPA cookbook configs usually launch through:

```toml
[container]
command = ["uv", "run", "--project", ".", "python", "synth_service_app.py"]
cwd = "."
startup_timeout_seconds = 120
```

Use environment variables for secrets. Never write API keys into TOML, README,
event logs, traces, or sample artifacts.

## Capability Metadata

Use `synth_containers` vocabulary where possible:

- runtime kind and task family
- rollout modes: `sync`, `async`
- statefulness tier
- checkpoint/restore/branching support
- reward/verifier/trace/artifact support
- tool-call and token-trace support
- route hints
- external resource refs for datasets, code, sandboxes, or assets

When adding new fields, add them first to canonical dataclasses where practical,
then update formatter, OpenAPI/README route inventory, and compatibility
projection. Avoid one-off wire keys that duplicate existing nouns.

## Validation Checklist

Do a narrow validation pass appropriate to the change:

- Syntax: `python -m py_compile synth_service_app.py`.
- Route smoke: start the container and check `/health`, `/metadata`,
  `/task_info`, `/program`, `/dataset`.
- Dataset smoke: request a few train and heldout rows by explicit seed.
- Rollout smoke: run one tiny rollout with a seed candidate overlay.
- Verifier smoke: confirm both a pass and fail path if the verifier is local
  and cheap.
- GEPA smoke: run `bash run_fresh_gepa.sh --profile smoke` when the container
  is intended for GEPA.

Respect repository instructions about tests. Do not add automated test files
unless the user explicitly requested tests.

## Public-Safe Guardrails

- No private endpoints, unredacted local paths in reusable docs, raw secrets,
  private traces, or private datasets.
- Do not depend on a developer's local auth state for public examples.
- Keep run artifacts under ignored run directories unless the artifact is a
  deliberate public sample.
- Cache/replay should prove inspectability, not hide that a container is fake.
- Document any intentionally unsupported route rather than returning a
  misleading success.

