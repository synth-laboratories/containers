# Hosted deployment operations

The public `PoolClient` and `synth-containers` CLI use backend project-bound
operation intents. The backend owns provisioning, revision checks, resource
cleanup and metering observations. A client disconnect does not undo acceptance.
Use `SYNTH_BACKEND_URL` and `SYNTH_API_KEY` from an authorized environment.

Commands:

- `deployment-create POOL TASK REQUEST.json --project-id PROJECT --idempotency-key KEY`
- `deployment-get POOL TASK --project-id PROJECT`
- `deployment-update POOL TASK REQUEST.json --project-id PROJECT --idempotency-key KEY --expected-revision REVISION`
- `deployment-delete POOL TASK --project-id PROJECT --idempotency-key KEY --expected-revision REVISION`
- `deployment-lookup POOL TASK --project-id PROJECT --idempotency-key KEY`
- `deployment-operation POOL OPERATION_ID --project-id PROJECT`

Read a revision immediately before update/delete. Reusing an accepted key with
changed content conflicts. Mutations do not automatically retry transport errors.
If a response is lost, use deployment-lookup with the original key. Pending or
recovery-required means reconcile the existing intent; a new key is not a repair.
A completed lookup returns the persisted receipt. A 404 means no visible intent
at observation time; it is not proof that an in-flight request has finished.

The request file contains the deployment configuration (provider, pinned image,
interface, limits and service settings). Keep files containing credentials
private. CLI output is the backend's redacted receipt, not a copy of the request.
Bodies are bounded to 1 MiB and finite JSON. Public operations bind a project;
only authenticated internal runtime context can supply an SMR run binding.

Use existing `submit`, `watch`, `get` and `cancel` for evaluations. A terminal
scientific result does not establish provider cleanup or settled cost. Resource
observations report their own status; unknown cost stays null. Hosted execution
limits and replay support depend on the deployed backend version, not merely
this client being installed. Direct Docker catalog commands keep local ownership.

On backends exposing `cleanup_pending`, `watch` stays attached after scientific
termination until recovery and capacity release are confirmed. Its timeout bounds
observation only; disconnecting does not cancel execution or recovery. Resume
with `--after-sequence` from the last persisted event. Older backends without the
field retain scientific-terminal behavior.

Backends exposing `publication_pending` keep watchers attached until the queued
result snapshot has a terminal publication receipt as well. Cleanup and result
publication are independent; neither field certifies the full trace bundle or
settled costs. Publication failures remain visible and may be retried by recovery.

`synth-containers result ROLLOUT` (or `PoolClient.get_result_snapshot`) retrieves
the committed `result.json` through the authenticated Artifact Platform. It
refuses pending publication and verifies the bounded bytes against the receipt's
size, SHA-256, rollout, pool and tenant-derived publication identity. Redirects
are refused; target-provided storage URLs are never followed. The result preserves
null rewards and the original scientific verdict. The download is bounded to
1 MiB and 60 seconds, and can be redirected to a local JSON file. This snapshot
has `custody_scope: result_snapshot_only`; full trace custody has its own receipt.

When `inference_pending` is exposed, watchers also retain observation while a
managed invocation is unfinished or awaiting accounting reconciliation. This is
independent of scientific completion, provider cleanup and result publication.
The existing watcher deadline still bounds the wait; a timeout preserves the
resume cursor and never starts another evaluation. A false value means no
unfinished managed invocation was reported, not that every external cost is known.
