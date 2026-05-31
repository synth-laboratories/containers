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

Latest daily dev build:

```bash
pip install --pre synth-containers==0.2.0.dev202605312141
uv add --prerelease allow synth-containers==0.2.0.dev202605312141
```

## Local Better SDK Dev

This branch pair expects:

- `containers`: package `synth-containers==0.2.0.dev202605312141`
- `optimizers`: package `synth-optimizers==0.2.0.dev202605312141`

Install both editable checkouts with `uv` (sibling repos under your workspace):

```bash
cd optimizers
uv sync --group dev
uv pip install -e ../containers
uv pip install -e .
```

Verify import paths and versions:

```bash
uv run python -c "import importlib.metadata as m, synth_containers, synth_optimizers; print(synth_containers.__file__); print(m.version('synth-containers')); print(synth_optimizers.__version__)"
```

The SDK validation examples (Banking77, TBLite, Crafter, MiniGrid) live in local
`optimizers/dev_examples/` (gitignored — not shipped in the repo).

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
