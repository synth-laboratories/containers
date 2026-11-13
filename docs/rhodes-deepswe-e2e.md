# DeepSWE → Rhodes → Codex Luna 6 low

**Live validation, 2026-11-13: not accepted on current hosted Rhodes.** The first
run reached the legacy staging API, built in 104.8 seconds and remained queued
behind a failed/absent legacy worker. It was cancelled before execution and its
pool archived; independent provider readback confirmed its exact build snapshot
absent. The current extracted Rhodes API passed identity/readiness checks but
refused `POST /v1/pools` with `rhodes_customer_route_refused` at
`public_origin_guard`, before allocating compute. Hosted compute intentionally
requires the Core-owned execution path. This standalone launcher needs a
Core-owned Harbor adapter before it can claim hosted end-to-end acceptance; no
model-solving result has been obtained. The [API versioning contract](rhodes-api-versioning-proposal.md)
describes the implemented source handshake (not yet deployed). Do not reopen the guarded hosted
writes or fall back to the legacy host to make the test pass.

`synth-deepswe-e2e` uploads and builds one DeepSWE task, runs one Codex attempt
through Rhodes on Daytona, checks the task's native isolated verifier, and
archives the pool through Rhodes. It times preparation, upload/build, execution,
verification, teardown and total elapsed time. It does not deploy the Rhodes
control plane itself.

Those lifecycle operations require a compatible API that authorizes direct pool
writes. The current hosted public origin does not. The runner first negotiates
stable `>=2.0.0,<3.0.0` through credential-free discovery and requires both
`customer.pool.create` and `customer.rollout.create`. It refuses legacy discovery
or missing/false capabilities before provider inventory, pool creation or upload.
It then checks service/build identity and `/readyz`, retains the contract profile,
and bounds queue wait separately at 120 seconds (`--queue-seconds`). Future SDK
requests pin `Synth-API-Major: 2`. Worker readiness remains separately unobserved.

The script fixes the agent to `codex`, the model route to `gpt-6-luna`, and
`kwargs.reasoning_effort` to `low`. The backend must admit that route and provide
the managed inference gateway. Model/provider keys remain under their existing
backend authority. The task image and Codex build layer come from the existing
DeepSWE producer; agent execution has no general internet and grading gets only
`model.patch` in a fresh verifier sandbox.

## Inputs and invocation

Use Python 3.12+ with this checkout installed and its existing
`native-harbor-daytona` extra, including Daytona **0.210.0**. The dependency is
loaded only for a live invocation. The live environment needs:

- `SYNTH_BACKEND_URL`: an existing Rhodes backend configured for Daytona.
- `SYNTH_API_KEY`: an authorized Synth API key.
- `DAYTONA_API_KEY`: the same provider account used by that backend, for
  **read-only** cleanup verification. `DAYTONA_API_URL` is optional.
- `SYNTH_PROJECT_ID`: optional project attribution, also accepted as `--project-id`.

Use existing project `.env`/secrets-proxy credentials. The script neither accesses
Keychain nor imports credentials into a registry. For example, from this checkout,
with an existing local environment file:

```sh
uv run --env-file /absolute/path/to/project.env --extra native-harbor-daytona \
  synth-deepswe-e2e \
  --deepswe-root /absolute/path/to/deep-swe \
  --pins /absolute/path/to/harbor-deepswe/pins.toml \
  --task anko-default-function-arguments
```

The corpus must contain `tasks/<slug>/task.toml`, `instruction.md`, `tests/`,
and `solution/solution.patch`, as expected by the existing DeepSWE loader. The
reference patch is not uploaded into the candidate's build context. The pin file
must contain `["deepswe/<slug>"]` with an `agent = "...@sha256:<64 hex>"` image.

Validate the corpus, pin and execution plan without credentials, provider imports,
network requests or billable execution:

```sh
PYTHONPATH=src python3 -m synth_containers.rhodes_deepswe_e2e \
  --deepswe-root /absolute/path/to/deep-swe \
  --pins /absolute/path/to/harbor-deepswe/pins.toml \
  --task anko-default-function-arguments --dry-run
```

Live invocation uses paid image-build, sandbox and model services. Use the
experiment's authorized spending tranche and backend financial limits; the script's
time bounds do not establish a dollar cap or a cost estimate. The script has no
automatic scientific retry or model/provider fallback.

## Acceptance and output

A normal invocation prints a timed stage log, then `E2E PASS` or `E2E FAIL` and
every individual check. PASS requires:

- The uploaded task retains Codex/Luna 6/low and the DeepSWE lifecycle profile.
- The backend records Codex/Luna 6, and committed inference-accounting history
  contains a settled invocation. Requested effort is a task-configuration check;
  it is not a claim of independent provider-side effort attestation.
- The original rollout completes; agent and verifier exit zero; a finite native
  reward is available; the verifier is isolated; both phases and both Daytona
  launches are recorded.
- Agent/verifier cleanup receipts name distinct owners and confirm absence.
- Replaying the identical submission returns the original rollout ID.
- Pool archival/provider cleanup completes, owned sandbox labels are absent,
  and the exact newly used build snapshot disappears. Pre-existing shared
  snapshots and unrelated concurrent jobs are preserved. Direct image launches
  do not require a snapshot deletion.

Reward `0` can pass infrastructure acceptance and is printed as
`task_solved=False`. Add `--require-solved` to require reward `1` as well. Missing
reward, accounting, provenance or cleanup evidence always fails; an older backend
without these facts cannot obtain PASS by omission.

Private, redacted JSON receipts are retained under
`.local/rhodes-deepswe-e2e/<run-id>/` (`--out` overrides the parent):

- `report.json`: original IDs, checks, errors and timings, saved before effects.
- `result.json`: result metrics, execution provenance and requested agent.
- `events.json`: contiguous committed operator event history.
- `usage.json`: the backend's reported usage, with unknowns preserved.

Exit codes: **0** all requested checks passed; **1** execution/acceptance/cleanup
failed; **2** invalid local input or missing prerequisites. No credentials or raw
exception bodies are printed. The script does not print native task/test bytes.

## Failure and cleanup behavior

The pool ID is chosen and saved before creation. A lost create response therefore
still permits exact pool lookup and an ownership check before deletion. Submission
uses both a fixed idempotency key and logical intent; a lost acknowledgement is
recovered using that same identity and explicitly cancelled during cleanup.

Timeout, Ctrl+C and SIGTERM attempt bounded teardown. Active execution is
explicitly cancelled; pool-delete conflicts retry only that pool while Rhodes
reconciles held capacity. Default cleanup allowance is 300 seconds
(`--cleanup-seconds`). The default observation allowance covers the corpus's
agent/verifier ceilings plus queue/collection margin; `--rollout-seconds` can
override it. Image-build/API requests default to 1800 seconds
(`--request-seconds`). A short observer timeout does not change native phase
budgets; it triggers explicit cancellation instead.

A hard kill, unavailable backend/provider, unidentifiable failed build, or
exhausted cleanup allowance cannot be certified clean. The script exits with a
failure receipt containing the exact pool/rollout IDs and unresolved resources.
Rhodes retains provider cleanup and snapshot-ledger authority; the script never
deletes provider resources directly or sweeps another run's resources. There is
no detached client-side watchdog after the script exits.
