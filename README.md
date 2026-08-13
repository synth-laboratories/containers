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
`ContainerHandle`, and `ContainerConnection` for URL-only optimizer handoff.
Use generic route hints and capability metadata here; optimizer-specific GEPA
settings belong in `synth-optimizers`.

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
