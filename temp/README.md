# temp — Craftax 10 seeds through synth-containers HTTP

Scratch eval. **This is the containers platform**, not `evals/suites/nonproduct/craftax`.

```bash
bash temp/run.sh              # craftax_react + gold rust + OpenRouter Luna medium
bash temp/run.sh --paid       # same as default
bash temp/run.sh --scripted   # craftax_engine, ScriptedReAct, in-process world
```

Passing a `.toml` is refused. The old `luna_med_10x.toml` drove the private Evals harness against GameBench rust; that path does not implement C1-08.

Default fails closed without `OPENROUTER_API_KEY`. The key is never written to the event log.

## Contract (logged on every run)

```text
POST /rollouts/prepare
GET  {declared transports.poll.url}?after=0   until stream.subscribed ready=true
POST /rollouts   slot=stream  telemetry.transport=sse  (auto forbidden)
POST /reward     mode=terminal   missing stays null
```

| Mode | Target | World | Policy | Honest claim |
| --- | --- | --- | --- | --- |
| default / `--paid` | `craftax_react` | `GoldCraftaxWorld` at `SYNTH_CRAFTAX_URL` (default `:18100`) | OpenRouter `gpt-5.6-luna` medium | Needs gold rust + `OPENROUTER_API_KEY`. Headless C1-08. **Not** A1 Desktop |
| `--scripted` | `craftax_engine` | in-process `CraftaxWorld` | `ScriptedReAct` | Headless C1-08 / C3. **Not** a Luna eval |

Default `SYNTH_CRAFTAX_MAX_STEPS=120` on Luna (gold's silent world default). `--scripted` stays at 8. Occupancy is sequential through one façade (`scale_leases=10`).

Outputs under `_runs/luna_med_10x/`: `run.log`, `summary.json`, per-seed `events/*.jsonl`, platform `event_logs/` + `seals/` when the façade persists them.
