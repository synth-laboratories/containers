# Local target standard v1

**Locked in this repository.** A Synth eval/optimizer target is **code + Dockerfile + Compose**. Never GHCR. Never a registry URL in the catalog. A digest is provenance after a local build, not how the target is found.

This is the packaging contract for Banking77, HealthBench, and Craftax. Workshop and Optimizers register the **loopback URL Compose publishes**. They do not pull images and they do not attach a raw GameBench engine.

## Layout

```text
  targets/
    catalog.toml                 # ids only; no ghcr.io, pull = false
    banking77/Dockerfile + compose.yaml
    healthbench/Dockerfile + compose.yaml
    craftax/Dockerfile + compose.yaml
    _runtime/Dockerfile          # shared copy
```

Each `compose.yaml` is the operator interface:

```bash
docker compose -f targets/banking77/compose.yaml up --build
```

That must:

- build from the Containers repo root (this tree)
- set `pull_policy` so Compose does not fetch a GHCR image
- publish **only** `127.0.0.1:<port>:<container-port>`
- start `targets/_runtime/serve.py` with `SYNTH_CONTAINER_TARGET`

Base image may be `python:3.12-slim` from Docker Hub. That is not GHCR. No `image:` line may contain `ghcr.io`.

## Wire contract (already implemented here)

All three targets are served by `create_compat_app` (`src/synth_containers/platform/app.py`). The façade already speaks:

| Route | Why it exists |
|---|---|
| `GET /health`, `GET /info` / `/metadata` | identity + advertised operations |
| `POST /rollouts/prepare` | allocate stream; emit `stream.subscribed` |
| poll / SSE on the declared descriptor | Workshop waits for ready |
| `POST /rollouts` | start after subscribe |
| `GET /reward` | authored reward; missing ≠ 0 |
| GEPA v2 `/program`, `/taskset`, `/dataset` | Banking77 / HealthBench search |

`capabilities.operations.prepare|start|get|poll` must be `true`. Workshop fail-closes if they are `unknown`. Native GameBench HTTP (`POST /run_scenario`, PNG, SSE) is the **engine**, not this contract. Craftax Compose serves `craftax_engine` (in-process fixture) or `craftax_react` (gold HTTP on the compose network). It does not publish raw `:18098` as the Workshop eval URL.

## The three targets

| Catalog id | `SYNTH_CONTAINER_TARGET` | Loopback | Proof in this repo | Live extras |
|---|---|---|---|---|
| `banking77` | `banking77_classify` | `127.0.0.1:8765` | `tests/test_banking77_platform.py` (prepare → subscribe → start) | cookbook GEPA v2 is the same contract |
| `healthbench` | `healthbench_chat` | `127.0.0.1:8114` | `tests/test_healthbench_platform.py` | grader needs `OPENAI_API_KEY`; compose passes it through, never bakes it |
| `craftax` | `craftax_engine` | `127.0.0.1:8097` | platform Craftax tests + this standard’s prepare smoke | gold overlay: set engine URL; still this façade, not GameBench native |

## What other repos may do

| Repo | Allowed | Forbidden |
|---|---|---|
| **containers** (this) | Dockerfile, compose, façade, catalog | GHCR names, `pull = true` |
| **cookbooks** | GEPA configs that point at the loopback URL | shipping a second image pin |
| **gamebench** | gold engine as a **compose sibling service** | being registered as the Workshop eval target |
| **optimizers catalog** | recipe id, seeds, pointer `compose: containers/targets/<id>/compose.yaml` | `image = "ghcr.io/…"` |
| **workshop** | register the published loopback URL | GHCR pull, shimming native engines |

## SHA

Record `git rev-parse HEAD` of this repo (and GameBench if gold is in the compose graph) on the receipt. Do not put `image_digest` in the catalog as a fetch key.
