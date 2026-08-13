# Handoff — local Trace V5 capture for long-horizon container rollouts

**Date:** 2026-08-13
**Primary repo:** `containers` @ `josh/aug12-containers-platform` (`synth-containers` 0.4.0.20260730)
**Secondary repo:** `gamebench` @ `feat/deo-bundle-pool-packaging` (Craftax gold service, Rust)
**Goal:** run Luna ReAct on Craftax to **2000 steps** and land one sealed, verifiable
Trace V5 bundle per rollout — messages, tool calls, thinking, and images — stored on
local disk, with capture optional and a per-container disk budget.

## Why this is a containers change, not a gamebench one

Trace V5 schema, transport, normalization and storage are owned by `synth-containers`.
The Craftax workload owns only the facts it observes. Any V5 emitter written inside
gamebench forks a version-pinned schema and will drift — `tracing verify` would stop
accepting it. Build the general capability here; wire Craftax to it.

## What already exists (verified, do not rebuild)

| capability | where |
|---|---|
| Local HTTP collector | `tracing/capture/collector_server.py:31` `CollectorServer` |
| Raw segment spool to disk | `tracing/capture/spool.py:107` `RawSpool` (`root/segments/`) |
| Seal segments → one trace doc | `tracing/capture/finalizer.py:161` `TraceFinalizer.seal(...) -> SealedCapture` |
| Self-contained on-disk bundle | `tracing/store/bundle.py` `LocalTraceBundle` |
| Store backends | `tracing/store/`: `filesystem` · `sqlite_catalog` · `bundle` · `s3` · `projection` |
| **Per-unit-of-work child sessions** | `capture/supervisor.py:472` `register_child_context()`, `:271` `context_for_child()`, `:478` `_restore_child_state()` |
| Language-agnostic emission | `capture/emitter.py` `TraceEmitter` — httpx POST + 4 headers (`x-synth-trace-id`, `-capture-id`, `-actor-id`, `-session-id`) |
| Capture on/off | `CaptureMode` (incl. `REQUIRED`) |
| CLI | `tracing run --output` · `tail` · `verify` |

`TraceFinalizer.seal()` is already bound to a spool + segment set and accepts
`extra_actors` / `extra_sessions`, so **sealing is already per-capture-session, not
per-process.** Child topology, terminal ordering and restore are already modelled.

## The pattern being implemented

**Sidecar collector · per-rollout child session · HTTP emission · seal on completion ·
reference in response · ship later.**

The only structural mismatch today is that the CLI entry point (`tracing run`) scopes a
capture to a *wrapped command*. A container hosting a long-lived multi-tenant service
needs the capture scoped to a *request*. Nothing below the CLI requires the coupling.

---

## Build 1 — detached capture mode

`CaptureSupervisor` currently owns a child process lifecycle. Add a mode that runs
collector + spool + finalizer **without** a workload subprocess, for the container's
lifetime.

- New `SupervisorConfig` flag (e.g. `detached: bool`) or a `DetachedCaptureSupervisor`.
- Starts `CollectorServer` + `RawSpool`, registers a container-lifetime **root** session,
  and stays up.
- Provider-proxy / mitmdump must be optional in this mode — a Rust workload calling
  OpenRouter directly is not intercepted, and `Interception.PROVIDER_PROXY` should not
  be a hard requirement for capture to function.
- Existing restore logic (`_restore_child_state`) must still work after a container
  restart.

## Build 2 — HTTP control surface for open/seal

So a non-Python workload can drive capture without importing the package.

```
POST /captures                    → open a child session under the container root
                                    body: {rollout_id, labels{}, capture_mode}
                                    200:  {trace_id, capture_id, actor_id, session_id}
POST /captures/{id}/events        → emit (same shape TraceEmitter posts today)
POST /captures/{id}/seal          → finalize
                                    body: {status: completed|failed|interrupted, termination}
                                    200:  {trace_v5_digest, bundle_path, bytes, event_count}
GET  /captures/{id}               → status / manifest
DELETE /captures/{id}             → abandon without sealing (must not leak spool)
```

Requirements:

- `seal` returns the **`trace_v5_digest`**. Downstream curation hard-rejects any
  trajectory without one, so this is the contract that matters.
- Concurrency: N open captures at once, independently sealed, no cross-talk. This is
  the whole reason for per-request sessions.
- An unsealed capture whose container dies must be recoverable or explicitly marked
  `interrupted` on restart — never silently lost.

## Build 3 — per-container disk budget

Traces with images are large. A 2000-step image rollout is on the order of hundreds of
MB, and ten concurrent rollouts will fill a volume.

- Config: `capture_disk_budget_bytes` (per container) + `capture_disk_reserve_bytes`
  (headroom never consumed).
- Accounting across spool segments **and** sealed bundles.
- On budget exceeded, policy is explicit and configurable — at minimum:
  - `refuse` — new `POST /captures` returns 507; **rollouts keep running uncaptured**
    rather than failing;
  - `evict_oldest_sealed` — drop sealed bundles oldest-first, never an open spool.
- **Never silently drop events from a live capture.** A capture that cannot be
  completed is sealed `interrupted` with a reason, or refused up front. A partial trace
  presented as complete is worse than no trace.
- Report `disk_used_bytes` / `budget_bytes` on `GET /captures` and in container health.

## Build 4 — Craftax wiring (gamebench)

`tasks/craftax-singleplayer/gold_rust/src/bin/craftax_gold.rs`,
`run_optimizer_rollout`.

Mirror the NEV spool already committed there (`96cf236`) — the shape is proven:
work goes to disk, the response carries a **reference**, retrieval is a route.

1. `POST /captures` at rollout start when capture is requested; hold the context.
2. Emit per turn (see *What to record*).
3. `POST /captures/{id}/seal` on terminal.
4. Put `trace_v5_digest`, `bundle_path` and `event_count` in `summary.trace`, next to
   the existing `nev` ref. **Do not inline the trace.**
5. `capture: "off" | "best_effort" | "required"` in `policy.config`, defaulting to
   `off` so existing runs are unchanged. `required` fails the rollout if capture
   cannot open; `best_effort` proceeds uncaptured and says so in the record.

### Prerequisite blocker

`max_llm_turns` clamps to **128** (`craftax_gold.rs:595`). At the measured **7.56
steps/turn** that ceilings any rollout at ~968 steps, so **2000 steps is currently
unreachable.** Raise the clamp (≥1024) as part of this work.

This also corrects a claim in `image_input.md`: the long-run survivors that finished at
894–1000 steps and 127–128 turns were **turn-capped by this clamp**, not step-capped.
A `max_llm_turns: 150` request was silently reduced to 128.

## What to record

Per turn, one span with these facts. All of it already exists in the `turns` array the
Rust loop builds — this is a routing job, not a collection job.

| fact | source |
|---|---|
| messages sent | the assembled conversation (system / user / assistant tool_calls / tool results) |
| tool calls | `used_tool_call`, function name, arguments |
| tool results | the observation string returned |
| thinking | assistant `content` — Luna returns reasoning there alongside `tool_calls` |
| images | the frame when `observation_mode` is `image`/`both` — **as a blob reference, not inline base64** |
| usage | `prompt_tokens`, `completion_tokens` per call |
| env transition | actions executed, reward delta, achievements delta, step index |
| compaction | existing `context.compactions[]` — trigger, tokens before, messages before/after |

**Images must go to the bundle's blob store** (`store/filesystem.py`
`FilesystemBlobStore`) and be referenced by digest. Inlining base64 into events is what
made a single rollout hit 1M tokens; do not repeat it in the trace.

Redaction: `capture/redaction.py` already exists — the `Authorization: Bearer` header on
provider calls must never reach a segment.

---

## Acceptance tests

Each must be runnable and must fail loudly, not warn.

### A. Detached capture, no workload subprocess
1. Start a detached supervisor with `--output <dir>`; no child command.
2. `POST /captures` → 200 with a full context.
3. Post 100 events; `seal` → `trace_v5_digest` + `bundle_path`.
4. `synth-containers tracing verify <bundle>` **passes**.
5. Kill and restart the supervisor; `GET /captures/{id}` still resolves the sealed bundle.

### B. Concurrency
1. Open 10 captures concurrently, interleave events across them, seal in a different
   order than opened.
2. All 10 verify; **no event appears in the wrong bundle** (assert per-bundle event ids).

### C. Disk budget
1. Budget 50 MB. Emit until exceeded.
2. `refuse`: `POST /captures` → 507; **in-flight captures still seal successfully**;
   an uncaptured rollout still completes.
3. `evict_oldest_sealed`: oldest sealed bundle removed, **no open spool touched**,
   surviving bundles still verify.
4. Assert `disk_used_bytes` never exceeds `budget - reserve`.

### D. Interrupted capture
1. Open a capture, emit, `SIGKILL` the container.
2. On restart the capture is sealed `interrupted` with a reason, or explicitly
   recoverable. **It is never reported `completed`, and never silently vanishes.**

### E. Craftax end-to-end, long horizon
1. `max_steps: 2000`, `max_llm_turns: 400`, `observation_mode: both`, `capture: required`.
2. Rollout reaches **>1500 env steps** (proves the clamp fix).
3. Response `summary.trace` carries `trace_v5_digest` + `bundle_path`; the trace is
   **not** inlined.
4. Bundle verifies. It contains, for a sampled turn: the full message list, the tool
   call with arguments, the tool result, the assistant thinking text, and an **image
   referenced by digest** that resolves in the blob store.
5. Secret scan over the bundle: no `Authorization`, no `sk-`, no `OPENROUTER_API_KEY`.

### F. Capture off
1. Identical rollout with `capture: "off"`.
2. No capture opened, no bytes written, rollout result **identical in reward, steps and
   achievements** to a captured run on the same seed.
3. Wall-clock overhead of capture measured and reported — capture must not silently
   halve throughput.

## Definition of done

- Detached mode + control surface merged in `synth-containers`, with A–D green.
- Craftax emits and seals per rollout; E and F green.
- Turn clamp raised; a 2000-step rollout demonstrably reaches >1500 steps.
- Disk budget enforced with an explicit, configurable policy and no silent event loss.
- `image_input.md` corrected re: turn-capping.
- Every curated trajectory can present a `trace_v5_digest` — this is what unblocks
  §12 dry test 2 in `HANDOFF_CRAFTAX_SFT_UPLIFT_2026-08-13.md`, whose curator hard-
  rejects any candidate lacking `sealed` + `trace_v5_digest`.

## Do not

- **Do not reimplement the V5 schema in Rust.** Emit over HTTP; the schema stays owned
  here. This is the single most important constraint in this document.
- Do not inline images or full traces into rollout responses. Reference by digest.
- Do not make capture mandatory. `off` must stay the default and must cost nothing.
- Do not drop events from a live capture to stay under budget. Refuse or seal
  `interrupted`; never present a partial trace as complete.
- Do not couple capture to mitmproxy/provider-proxy. A workload that calls a provider
  directly must still be capturable.
