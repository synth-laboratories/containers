# Stock Craftax image under `synth-containers eval` with a plan-declared engine command/port

Date: 2026-10-27. Source: this branch at `2e7158f` (source digest null).

Image: `evals-craftax-gamebench-rust:local` (sha256:77a13963…), whose own CMD serves the façade on 8080. The plan declares
`engine_command = [".../craftax_gold", "--host", "0.0.0.0", "--port", "8098"]` and `engine_port = 8098`.

| Step | Result |
|---|---|
| `eval describe` | supported, no blockers, `command_source: plan` |
| `eval run` | rc 0 in 38 s; `execution: completed`, `cleanup: owned-engine-stopped` |
| lane `scripted-s0-r0` | `completed`, `stop_reason: step_limit`, 8 steps / 8 calls |
| engine receipt | `cleanup: stopped`, `create_seconds: 36.965` |
| leftovers (`ai.synth.eval.owner` label) | 0 |

Earlier attempts on the same host were refused with `native_engine_startup_failed` because the daemon took 38–43 s per container create (plain `alpine:3` included) against the previous 30 s wall; the guard now uses a 180 s wall and records the measured create duration. Files: `describe.json`, `run-stdout.json`, `run/summary.json`, `run/engine-receipts/`, the lane `job_result.json` and `actions.jsonl`.
