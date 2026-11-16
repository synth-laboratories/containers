# Container services and managed Harbor Serve

A package describes its HTTP interfaces, durable state and execution protocol. Rhodes binds that declaration to an owned deployment and publishes authenticated endpoints. Framework adapters keep their native API and scheduler. The Synth operation facade delegates to that same scheduler.

## Package and protocol

`synth.container-package.v2` adds `services` to the v1 manifest. Build plans and artifact descriptors become v2 and hash the complete service declaration. V1 packages remain unchanged and reject v2 fields. `synth.container-services.v1` describes named interfaces (`protocol`, `version`, `port`, `base_path`, `transport`), optional single-writer state mounts, and an optional execution interface. Compatibility reports match declarations; they do not certify readiness, admission or provider support.

The executable interface uses `synth.service-operations`, version `1`:

- `GET /.well-known/synth-service` returns the declaration digest and persistent origin UUID.
- `POST /operations` accepts `{operation_id, input, expected_origin_instance}` and returns 202 with an identity-bound receipt.
- `GET /operations/{id}?origin_instance=…&input_sha256=…` observes the same native operation.
- `POST /operations/{id}/cancel` uses those same query bindings and records cancellation intent before requesting native cancellation.

`DurableServiceRuntime` owns the existing `RolloutJournal`, including its single-writer lock, input hashing and durable origin identity. Implement `OperationDriver.submit`, `recover`, `observe` and `cancel` to attach an existing scheduler. Recovery never calls `submit`. A lost reply remains `in_doubt` until native identity and resource cleanup can be established. A terminal result without confirmed cleanup does not release custody. Closing a client connection does not cancel accepted work.

`ServiceOperationClient` validates receipts against the submitted operation, digest and origin. Neither transport retries nor a new controller address establish permission to run the work again. Native results remain verbatim; missing scores and token measurements remain unknown.

## Rhodes binding

Use `service_auth_mode="managed"`. Rhodes mints the private origin bearer and a deployment generation, injects `ENVIRONMENT_API_KEY`, and returns named endpoints from `GET /v1/pools/{pool}/deployments/{task}/services`. Public callers use their Synth credential at those URLs. Provider origins and private bearers are not public endpoint credentials.

Rhodes observes the provider-owned instance before forwarding requests. The gateway supports HTTP and Streamable HTTP MCP, including native docs, OpenAPI, query/download routes and cancellation. A declared execution service can also run through the existing Rhodes rollout path with the native request in `payload`. The original execution deadline, deployment digest, provider identity, input digest and origin UUID remain bound through recovery.

Resource callbacks use durable `ServiceAttempt` admission. Every trial/retry is a separate attempt. The deployment policy bounds CPU, memory, storage, timeout, queue size and concurrency. Child resources use deterministic identities. Start uncertainty retains capacity; cleanup and billing settlement are independently retried. Deployment deletion drains child resources before removing the controller.

The initial persistence/child-provider implementation is Docker. Live-provider acceptance is separate from the protocol and source tests. Controller state uses a named volume scoped to the local deployment, organization, pool, task and mount. Update/recreation reattaches that volume; deletion retains it. Native job history and downloadable files remain on that volume. This is deployment-local retention, not publication to Synth Artifact Platform. Export native artifacts before retiring their deployment or reattach its retained state. Stateless generic HTTP services can use the existing Docker/Daytona controller profiles. Persistent controllers and managed child execution on other providers fail explicitly until their storage and custody adapters are qualified.

## Harbor 0.26

Install `synth-containers[harbor-serve]` under Python 3.12+. This extra pins Harbor 0.26.0 and conflicts with the separate Harbor 0.22 native adapter extras. `synth-harbor-serve` launches Harbor's own `Service`, SQLite store, executor, REST API and MCP server, plus the Synth facade at `/synth`. There is one native job/trial scheduler.

`harbor_services(concurrency=4, port=8390)` produces the matching declaration. Launch with `--task-root /tasks --max-concurrent 4 --port 8390`. Rhodes supplies `HARBOR_API_KEY` and `SYNTH_SERVICE_AUTHORITY_URL`; package them as deployment-owned values, never caller-provided credentials. Data defaults to `/var/lib/harbor`.

Tasks must be packaged beneath approved task roots and use digest-pinned sandbox images. Server filesystem imports, custom agent/verifier imports, caller credentials, external model URLs, arbitrary agent kwargs, MCP servers, host mounts and remote task imports are refused. Task assets transfer through bounded archives; sandbox command execution and cleanup stay with Rhodes. Oracle trials need no inference. Model-based agents require a declared model list and per-trial ceiling, supplied with `--model MODEL --inference-ceiling-micros N`; the exact same policy must appear in the package declaration. Rhodes uses its existing Core inference hold/session/outbox path, and each trial receives only a scoped gateway token. Responses and Chat Completions wires are supported for configurable built-in agents.

Native pause/resume controls queued work. Running trials interrupted by restart become interrupted native results after resource reconciliation; this is not computational checkpoint/resume. Retries receive new native IDs, re-admission and their own holds, after prior cleanup is confirmed.

A service package uses the explicit `service_handshake` validation case in Rhodes. That validates the observed image, operation protocol, declaration and origin, then removes the validation instance. Its receipt says `service_handshake_only`; it does not claim a benchmark reward or paid-provider qualification.

## Verification

`tests/test_service_package.py` covers v2 digest binding and v1 refusal. `tests/test_service_runtime.py` covers durable lost-reply recovery, replay conflicts, cancellation custody and writer exclusion. `tests/test_managed_harbor_serve.py` runs the real Harbor scheduler, REST and MCP with an explicit resource-authority fixture. Rhodes adds SQL lifecycle tests and an HTTP integration test connecting all three layers. Provider fixtures prove protocol behavior, not installed provider behavior.

Docker managed attempts must request limits that the Docker adapter can represent
exactly. Rhodes rejects requests that would be raised to a provider minimum or
reduced to a platform ceiling before reserving a hold. No-network child sandboxes
use Docker's `none` network; they do not allocate a bridge per trial. Retained
controller state uses the authenticated organization captured before worker-thread
I/O, together with deployment scope, pool, task and state name.
