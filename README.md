<h1 align="center">synth-containers</h1>
<p align="center">The task contract for Synth optimizers and evals — wrap any task as a small HTTP service that optimizers can target without touching your code.</p>
<p align="center">
<a href="https://pypi.org/project/synth-containers/">PyPI</a> ·
<a href="https://github.com/synth-laboratories/optimizers">Optimizers</a> ·
<a href="https://github.com/synth-laboratories/synth-cookbooks-public">Cookbooks</a>
</p>

A Synth container is a small HTTP service around a task. It owns the dataset, the
scoring/verifier logic, the mutable prompt or policy fields, and the policy-model
credential boundary. An optimizer like
[`synth-optimizers`](https://github.com/synth-laboratories/optimizers) GEPA sees one URL
and a typed rollout contract — it never imports your task package or reads private
evaluator state. The same contract works whether the task is a classifier, a coding
agent, or a live game environment, and in Python, Rust, or TypeScript.

## Install

```bash
pip install synth-containers
# or
uv add synth-containers
```

## Local Synth development

Register the current checkout once after changing Containers or its package version:

```bash
./scripts/register-local-dev-build.sh
```

The command builds a wheel, installs it into an immutable machine-local directory,
verifies the installed version, and atomically selects it for local Workshop builds.
No launch flags or environment variables are required. Re-running it reuses an
identical registered wheel.

Sibling Optimizers development remains editable through its checked-in uv source:

```bash
cd ../optimizers
uv sync --group dev
```

## The contract

| Route | Method | Purpose |
| --- | --- | --- |
| `/metadata` | GET | contract version + capabilities |
| `/program` | GET | mutable prompt fields + seed candidate |
| `/dataset` | GET | split names + row counts |
| `/dataset/rows` | POST | rows for a requested seed list |
| `/rollout` | POST | run a candidate on a row → reward + usage |
| `/health` | GET | liveness |

The Python SDK also provides `Container`, `Container.serve()`,
`ContainerHandle`, `ContainerConnection`, and `ContainerRunner` (url, command,
in-process app, or local `image_id`). CLI: `synth-containers serve` and
`synth-containers up`.

## Example

```python
from fastapi import Body, FastAPI
from synth_containers import GEPA_OPTIMIZER_CONTRACT_VERSION

app = FastAPI()

@app.post("/rollout")
def rollout(payload: dict = Body(...)) -> dict:
    candidate, row = payload["candidate"], payload["row"]
    # run the task with the candidate's mutable fields, score it with a real verifier
    return {"reward": ..., "usage": ...}
```

See the
[cookbooks](https://github.com/synth-laboratories/synth-cookbooks-public/tree/main/cookbooks/optimizers/gepa)
for complete containers: Banking77, HotpotQA, MiniGrid, TBLite, and Crafter.

## Links

- [Optimizers](https://github.com/synth-laboratories/optimizers) — GEPA on this contract
- [Cookbooks](https://github.com/synth-laboratories/synth-cookbooks-public) — runnable containers
- [Agent skill](skills/containers/SKILL.md) — drop into a coding agent to build and debug containers
- [Contract OpenAPI](openapi/container-contract-v1.yaml)

## License

MIT

### Native Harbor result decoding

`synth_containers.harbor_results` provides a bounded native result reader and
staged structural validation through `HarborTrialRecord`. Use
`read_result_object(path, label=...)`, then `HarborTrialRecord.from_mapping(...)`.
The reader caps each JSON result at 16 MiB. Explicit accessors validate exception
information, agent context and verifier reward; missing/non-finite rewards are
errors, while a genuine numeric zero remains zero. Benchmark-specific trusted
scoring, exception harvesting and trace custody remain the caller's authority.
This decoder is shared infrastructure, not a Docker/Daytona execution-parity claim.

### Required execution-limit capabilities

`oci_trial.TrialRunRequest.required_limit_capabilities` optionally requires typed
`LimitCapability` entries from `synth_containers.limit_capabilities`. The executor
checks them before creating output directories or starting a provider process.
Read `OciTrialExecutor.limit_capabilities` during planning; declarations are not
receipts that a particular container's actuators have been armed.

The current OCI executor advertises sampled work-time and output-byte thresholds,
and native CPU allocation and memory controls. Native allocation controls survive
the Python supervisor; its sampled deadline and output checks do not. A required
hard output/workspace quota, provider-spend reservation, or deadline surviving
supervisor loss is refused. CPU allocation is not CPU-time accounting. Native
memory controls describe the container memory setting, not a total swap budget.

Requirements specify mechanisms, separately from `TrialExecutionLimits` values.
The resolver does not silently rank native, sampled and reserved mechanisms as
interchangeable. Call `unsupported_limit_capabilities` for optional planning
requirements and report the returned entries; do not silently drop mandatory
ones. This additive SDK surface does not yet establish hosted Rhodes or native
Harbor capability parity.

### Project container leases

`PoolClient.assign_lease` requires a project, image capability, explicit substrate
and stable idempotency key. It verifies the returned server-resolved substrate,
project, image capability and active status. An older response that only repeats
the requested provider in metadata is refused. A refusal after a response may
refer to an allocated lease: inspect the reported lease ID and reconcile before
changing the request. Mutating transport failures are never retried automatically.

The CLI exposes the same client methods:

```sh
synth-containers lease-assign PROJECT_UUID --image-kind synth_sdk --substrate docker --idempotency-key stable --ttl-seconds 300
synth-containers lease-get LEASE_ID
synth-containers lease-renew LEASE_ID --ttl-seconds 300
synth-containers lease-release LEASE_ID
```

Use `daytona` for an explicitly Daytona-backed pool. Renewal requests accept
60–3600 seconds. Releasing a lease releases its admission claim; deployment
deletion remains a separate revision-checked operation. These methods do not
bypass Rhodes rollout admission by calling the returned container URL directly.
