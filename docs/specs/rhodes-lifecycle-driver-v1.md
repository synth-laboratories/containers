# Rhodes lifecycle driver v1

`scripts/stream-rhodes.sh` (module `synth_containers.rhodes_lifecycle`) drives one
complete Rhodes lifecycle through the public API and reports only what it
observed. It is an operator tool, not an acceptance claim.

## Sequence

1. **Registry (optional).** `--import <declaration>` registers a release, or
   `--release-id` selects one. Either requires a Docker compatibility verdict
   before any pool exists. `unsupported`/`blocked` never run. `unqualified` runs
   only with `--allow-unqualified`, and the report keeps `release_support:
   "unqualified"`. Without either flag the driver deploys `--image` (a pinned
   `sha256:` digest, default the cached native Craftax runtime). Source upload
   stays a separate step (`synth-containers rhodes upload-source`) because the
   import declaration must reference the author's own manifest. `--validate`
   first runs one bounded server validation (`--validation-seed`, same harness
   and `--max-steps`), waits up to `--validation-seconds` and records the
   operation in `registry.validation`. A pass makes the verdict `supported`;
   `--validate` without a release refuses.
2. **Lanes.** `--parallel N` creates N project-bound `arbitrary` Docker pools.
   The observed-capacity profile holds a whole pool per lease and never reuses a
   used instance, so seeds within a lane run in order, each on a fresh deployment.
3. **Per seed:** server-provisioned deployment create → readiness polling (409/503
   are "not ready yet", anything else refuses) → `synth.rhodes-observed-capacity.v1`
   lease → rollout admitted with `capacity_lease_id`, `seed`, `max_steps` and
   `policy_ref` → committed event follow → result and usage reads.
4. **Teardown (always):** cancel a non-terminal owned rollout, release the lease,
   delete the deployment at its current revision, read back 404, delete the pool.

Every mutation carries a stable key `rhodes-lifecycle:<run>:<step>:<seed>`, so a
retry replays the server receipt instead of repeating an effect. Pool creation
has no replay identity and is never retried.

## Output and exit

`<out>/<stamp>-<id>/` holds `stream.log`, `events.jsonl` (original committed
events), `result-seed*.json`, `report.json` and `summary.txt`. Checks: every seed
completed and scored, every lease released, every deployment deleted and absent,
every pool deleted, observed rollout overlap reaches the lane count, and no
interrupt. Exit 0 only when all pass; 1 a check failed; 2 refused before running;
130 interrupted. The first Ctrl-C cancels only this run's active rollouts and then
completes teardown.

The default policy is the scripted harness (`scripted_react`/`engine_acceptance`):
no model or paid provider calls.

## CLI verbs added with this driver

`pools create|get|delete`, `deployments create|get|delete|operation`,
`readiness`, `leases acquire|get|release`,
`environments validate|validation` (`validate --wait` exits 0 only on
`passed`), alongside the existing registry,
`exec`, `watch` and result verbs.

## Known limits

- **Slots.** Readiness and lease observation probe `http://127.0.0.1:<published
  port>` from the API process. Inside a slot's bridge-networked `backend-api`
  container that address is the container itself, so this profile needs a slot
  networking decision (SYN-4840) before it can pass there. Host-process backends
  are unaffected.
- **Validation.** Server validation (SYN-4839) runs in the API process. On
  slots it shares the loopback limit above; it also inspects trace bytes
  without retaining them yet.
- **Not covered.** Hosted providers, warm reset, money settlement, Swarms or
  optimizer adoption and the G0–G5 matrices.


### CardCodeBench process and workspace custody

The producer builds only public workspace files into the task image. It
provisions the declared named non-root agent account, grants that account the
public workspace, and leaves the trusted native phase supervisor as root so it
can drop the candidate's privilege and run the separate isolated verifier.
Private grading files are transferred only to the verifier phase. Task limits
request `pids_limit=128`; the backend must preserve that finite process bound
through resource normalization and Docker cgroup creation. Bases need `id`,
`useradd`, and `chown`; failure to provision the named account fails the build.
