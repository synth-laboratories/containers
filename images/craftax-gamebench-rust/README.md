# Craftax GameBench rust HTTP task

Rust GameBench gold + the synth-containers façade. Catalog id:
`craftax-gamebench-rust`. `synth-containers` does not name Craftax; this
directory is the particular container.

Do not use `nonsensitive/craftax_go_explore` (JAX). Do not serve a fixture world.

## Up

Needs the rust binary (or a crate it can build): sibling
`craftax-runtime/gold_rust`, or pin
`SYNTH_CRAFTAX_GOLD_BIN` / `SYNTH_CRAFTAX_GOLD_ROOT`.

Prefer the module so Ctrl+C stops **both** rust and the façade. Catalog `up`
runs the same command as a subprocess; a wrapper SIGTERM can leave rust
listening.

```bash
export SYNTH_CONTAINER_IMAGE_CATALOG=/path/to/containers/images/craftax-gamebench-rust
# optional: skip cargo if you already built
# export SYNTH_CRAFTAX_GOLD_BIN=/path/to/gold_rust/target/release/craftax_gold

cd containers/images/craftax-gamebench-rust
PYTHONPATH=. python -m craftax_gold --port 8080
```

Same stack through the catalog:

```bash
synth-containers up craftax-gamebench-rust \
  --catalog /path/to/containers/images/craftax-gamebench-rust \
  --port 8080
```

Python:

```python
from craftax_gold import serve

with serve(port=8080) as handle:
    handle.health()
```

`/health` is 200 only when rust gold is up (`gold_ok: true`). Otherwise 503.
Do not set `SYNTH_CRAFTAX_URL` unless you are attaching to gold you already
started.

Other players, same gold: `--goex` (true checkpoints), `--code-policy`
(`PUT /policy`; `engine_generation` stays 1), or `--target craftax_nanohorizon`
(PUT `policy.py` **and** bind a sampler config; contest speedrun). Default up
stays `craftax_react`.

`synth-containers serve --target craftax_react` is the wrong path (public
`TARGETS` has no Craftax).

The runtime image consumes two explicit frozen contexts: `CRAFTAX_ENGINE_ARTIFACTS`
and `CONTAINERS_WHEEL_CONTEXT`. It never compiles Rust during image assembly or
startup. Prepare the engine from a clean, pinned `craftax-runtime` checkout:

```bash
python scripts/prepare_craftax_runtime.py \
  --source-root /path/to/clean/craftax-runtime \
  --output-root "$PWD/.local/craftax-runtime-builds" \
  --launcher /path/to/synth-dev/local_dev/scripts/core_dev_build.py \
  --storage-guard /path/to/synth-dev/local_dev/scripts/storage_target_guard.py \
  --mbx-linux-dir /path/to/qualified/linux-arm64-mbx
```

The command reports its frozen runtime directory and retains source/compiler
hashes, test/build logs and cleanup state. The compiler has a bounded lifetime,
two CPUs, 8 GiB memory, the normal shared admission lane and an 8 GiB owned target
guard. Source is read-only at the same path used by runtime data lookup. Existing
shared mbx caches retain their 80 GiB policy. A retained uncertain operation is
reconciled before replacement; no raw Cargo fallback is available.

Set `CRAFTAX_ENGINE_ARTIFACTS` to that qualified runtime directory and
`CONTAINERS_WHEEL_CONTEXT` to a directory containing exactly one qualified
`synth-containers` wheel. The image verifies binary/data closure and compiler
identity, installs dependencies from the hash-locked requirements, then installs
that wheel without resolving another SDK version. These local ARM64 inputs do
not establish hosted/AMD64 placement qualification.

Use this directory's dedicated `catalog.toml` for NanoHorizon and live evals.
The parent `containers/images/catalog.toml` is a development inventory and may
name unrelated images that are not present in a clean Craftax checkout.

NanoHorizon public GEPA (`craftax_nanohorizon`): `GET /program` and
`POST /rollout` overlay `system_prompt`. Contest eval still PUTs `policy.py`
and uses `POST /rollouts`. Rebuild this image after pulling those routes.
Do not point GEPA at cookbook JAX Crafter.

## ReAct Luna low rollout

Paid planner reads `OPENROUTER_API_KEY` at request time (evals `.env` is fine).
Default episode pin is 120 env steps; cap it unless you want a long Luna run.

```bash
export OPENROUTER_API_KEY  # already in the server env if you sourced .env before up
export SYNTH_CRAFTAX_MAX_STEPS=12   # set on the server process before up

curl -sS http://127.0.0.1:8080/health

curl -sS -X POST http://127.0.0.1:8080/policy-configs \
  -H 'Content-Type: application/json' \
  -d '{
    "config_id": "luna_low",
    "harness": "react",
    "config": {
      "provider": "openrouter",
      "model": "openai/gpt-5.6-luna",
      "effort": "low",
      "api_key_env": "OPENROUTER_API_KEY",
      "max_tokens": 2048,
      "parse_retries": 2
    }
  }'

curl -sS -X POST http://127.0.0.1:8080/rollouts \
  -H 'Content-Type: application/json' \
  -d '{
    "telemetry": {"enabled": true, "transport": "sse"},
    "task_instance_id": "seed:0",
    "policy_ref": {"harness": "react", "config": "luna_low"}
  }'

# use rollout_id from the start response
curl -sS -X POST http://127.0.0.1:8080/reward \
  -H 'Content-Type: application/json' \
  -d '{"rollout_id":"<id>","mode":"terminal"}'
```

Env-sum is `eval:craftax.env_sum`. Stream: `GET /rollouts/<id>/events?after=0`.
Policy session should show `model: openai/gpt-5.6-luna`, `reasoning_effort: low`.

## Who attaches this URL

Evals workshop cases, Desktop Inventory, SFT/GELO recipes, and `gate --craftax-url` all talk to **this** façade (`:8080`), not rust gold `:8098` and not `run_service.py`. Harbor GameBench DEO stays on Harbor.
