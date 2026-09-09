# GSM8K loopback mechanism receipt — 2026-08-20

A mechanism receipt, not a score: one seed, greedy, base model. It shows the
pinned GSM8K world reaching a **real** local `synth-mlx-rl` service (v0.7
worktree, real MLX backend, `Qwen/Qwen3.5-0.8B` offline, `HF_HUB_OFFLINE=1`)
through both container-side sampling paths, and the token join resolving.

| File | What it is |
| --- | --- |
| `receipt.json` | summary: dataset pin, both rollouts, the join |
| `metadata.json` | the target's `/metadata` (includes `dataset` manifest) + `/training/capabilities` |
| `service.json` | service `/healthz`, `/v1/synth/capability`, the published snapshot |
| `training_rollout.json` | `POST /training/rollouts` request (bearer redacted) + summary |
| `training_rollout_events.json` | that rollout's sealed event log |
| `provider_rollout.json` / `provider_rollout_events.json` | `POST /rollouts` with the `synth_mlx_rl` provider; the event log carries `token_capture` |
| `join_check.json` | `token_capture.proxy_request_ids[0]` resolved against `/v1/synth/rollouts/{prid}` |

Dataset: `openai/gsm8k` `main` @ `740312add88f781978c0658806c59bc2815b9866`,
profile `hf` (declared in code), train `7473` rows
`sha256:dca44988…`, heldout(`test`) `1319` rows `sha256:32c548f0…`,
shuffle seed `20260820`.

Reproduce: start the service (see synth-mlx-rl `scripts/mlx_guard.py` for the
admission/watchdog discipline), then

```bash
uv run --with datasets python scripts/gsm8k_loopback_receipt.py \
    --base-url http://127.0.0.1:8791 --out docs/receipts/<date>/gsm8k-loopback
```
