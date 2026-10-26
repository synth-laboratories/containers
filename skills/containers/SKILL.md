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
- `GET /rollouts/{id}/stream` (SSE when implemented)
- `GET /rollouts/{id}/reward`
- `GET /rollouts/{id}/reward/events`
- `GET /rollouts/{id}/reward/stream`
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

## Running Local Rollouts

- `synth-containers up IMAGE --env KEY` forwards `KEY` from the host shell.
  When it only lives in a dotenv file, add `--env-file PATH`; the file supplies
  values for bare `--env KEY` forwards and nothing else.
- Built-in harnesses (`react`, `single_call`, `responses_react`, ...) need only
  `POST /policy-configs`. `PUT /policy` is for code-backed harnesses.
- A rollout takes minutes. Run `synth-containers rollout --config NAME --seeds
  2001 2002 ... --out progress.json` from your own terminal; a short tool-call
  wall clock that drops the `/complete` connection used to mark finished
  rollouts `failed`. Count a rollout by `reward_status == "scored"`.
- A catalog entry with a missing build context is skipped with a warning; ask
  for it by id to see the error.

## Capability Metadata

Use `synth_containers` vocabulary where possible:

- runtime kind and task family
- rollout modes: `blocking`, `async` (`ontology.RolloutMode`)
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


## Repository source of truth (Harbor and streaming)

Use the checked-out `src/synth_containers/` implementation as the API authority;
verify installed-package parity before relying on an installed SDK. This skill
is guidance, not a replacement for current signatures or provider validation.

For native Harbor tasks, read `harbor_environment.py`: inspect with
`inspect_harbor_package`, validate `HarborProviderCompatibility`, then register
separate agent/verifier image identities with `register_harbor_environment`.
Registration is a static content-addressed contract, not execution or proof of
provider isolation. Local Docker `sha256:<64>` image IDs are valid only for
`local-docker`. Remote providers use immutable registry digest references, or
for native Daytona builds the provider-specific
`daytona://snapshots/<uuid>/daytona-<64hex>:daytona` identity. The Daytona
executor must read back the snapshot UUID and content-addressed build ref
before launch. A build ref is not an OCI manifest digest; never relabel it.
Preserve source-package digests, artifact collection and separate-verifier
custody. Provider support must be qualified against the actual Harbor backend;
do not translate a local image tag into a remote compatibility claim.

For rollout streams, read `event_log.py` and the runtime's route implementation.
Use `RolloutEventLog` for durable append-before-publication journals,
`stream_descriptor` (`synth.rollout.stream.v1`), and `LogEnvelope`
(`synth.trace-stream-event.v1`). Poll with `poll_payload`; stream with `iter_sse`
and `SSE_HEADERS`. Consumers validate envelope digests and contiguous sequence
cursors; SSE reconnects use `Last-Event-ID`, polling uses `after`. Controls have
no sequence. Advertise only implemented transports. Reward routes preserve
explicit zero scores and distinguish unscored failure from zero reward.
Journal recovery is not execution resume; interruption must remain explicit.
Checkpoint/branch/cancel capabilities require real runtime implementations.

Parallel Harbor workers must isolate process-wide environment and provider
clients, output directories, verifier credentials and lifecycle ownership.
Share a process-safe aggregate spend ledger, correlate each call to its rollout,
and retain provider cleanup receipts. Qualify Docker and remote providers
separately; a Docker pass does not prove Daytona execution. Explicitly label
retained replay and never present it as a new paid or remote run.

Native Daytona builds may return an empty `image_name` while exposing `ref` and
`build_info.snapshot_ref`. Its private build registry is not necessarily
pullable as a downstream `FROM`. Self-contained Dockerfiles with shared cached
prefixes avoid inventing registry access or digest evidence. Verify this against
the installed Daytona SDK and provider readback rather than treating a registry
tag as immutable.

When bootstrapping a long-lived relay in native Harbor Daytona, its command
session can wait on background descendants. An SDK bootstrap that spawns a
process with a new session, closed inherited descriptors and redirected stdio
avoids that wait. This is transport setup; keep the policy's native agent loop.
Retain separate evidence for failed startup attempts and successful provider
qualification, including network policy readback and exact sandbox deletion.

Boat v1 remote Docker adapters register images as
`boat://sandboxes/<seed-id>/images/sha256:<64hex>` with provider `boat`.
This is a seed-scoped OCI configuration ID, not a registry manifest digest.
Fork a credential-free seed, then inspect the exact Docker image ID in the fork
before running the task container. Boat's VM-ready state can precede Docker's
restored image inventory; wait for that concrete image instead of rebuilding or
substituting a tag. Keep task containers at `--network none`, pass no provider
credentials or Docker socket, and preserve separate verifier custody. Use the
file API for binary/log transfers: detached-command status output is a tail.
Boat lifecycle/API details belong in the adapter; verify current provider limits
and distinguish a stopped reusable seed from deleted per-rollout VMs. Do not
claim catalog or parallel qualification from an underlying sandbox smoke alone.

Modal sandbox adapters register immutable provider image IDs as
`modal://images/im-<id>` with provider `modal`; do not substitute mutable registry
tags or pretend these are OCI digests. Resolve the exact ID before launch, block
outbound networking, pass no secrets/volumes/OIDC identity, and drop privileges
for policy execution. Keep credentials on the host relay. Verify termination by
polling the owned sandbox ID; a terminate request alone is not cleanup evidence.
Capacity qualification must retain both roles for every requested concurrent task.
Use sandbox-specific non-preemptible pricing, not normal function rates.
