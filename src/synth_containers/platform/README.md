# Platform façade

Pins, leases, durable sequence logs, `/reward`, occupancy, and the `stream` slot.

Normative stream lifecycle and envelope: [`docs/trace_stream_protocol_v1.md`](../../../docs/trace_stream_protocol_v1.md).

See: `backend/notes/specifications/tanha/references/synthstyle.md` — general foundations, targeted affordances; one umbrella layer; hierarchies of clear nouns.

## Problem

`CompatPlatform` was blending policy, persistence, env loops, and content families in one file (`if craftax / harbor / echo / digbench`). That is blended spaghetti: every new target rewires the façade.

## Approach

One umbrella: `TargetRuntime`. Callers say “run this target.” Children live under `runtimes/`. The façade does not name content families in control flow.

## Variants now

| Kind | Module | What it owns |
| ---- | ------ | ------------ |
| `external` | `TargetSpec.runtime` | Image-owned world (not in this package). |
| `harbor` | `runtimes/harbor.py` | Trial/verifier fold. Fixture keeps verifier on the parent log. `env:harbor_docker` runs agent vs verifier as distinct `docker run`s; native `reward.txt` ≡ `/reward`. |
| `digbench` | `runtimes/digbench.py` | Mock dungeon (`env:digbench_mock`) or live Agent API (`env:digbench_relay`). No frames. |
| `openenv` | `runtimes/openenv.py` | Echo-shaped gym wrap (`env:echo` via `echo_world.py`). Observation / action / env reward. Not a fold. Not an unmodified image. |
| `banking77` | `runtimes/banking77.py` | One-shot classify. Content, not a fold. Gold private. Classify, Tinker, or scoped Responses policy. |
| `gsm8k` | `runtimes/gsm8k.py` | One-turn grade-school math. Content, not a fold. Reference answer private. Exact match on the parsed number; an unparseable completion is null, not zero. |
| `healthbench` | `runtimes/healthbench.py` | Open-text physician-rubric chat. Policy and scorer are independent paid roles. |

Dispatch is `_BY_FAMILY` in `runtimes/__init__.py` — the one umbrella layer.

## Planned variants (plug in here, do not rewire pins/logs/`/reward`)

- Unmodified OpenEnv Echo image (A7)
- Harbor-packaged GameBench task image (A2 Desktop register; Docker alpine fixture is the fold wiring)

Trace Streaming Profile kit: `docs/specs/trace-streaming-profile-v1.md` + `tests/conformance/trace_stream/`.

## Non-goals

- Echo as a fold or `compat/echo.py`
- Harbor-wrapping or OpenEnv-wrapping dig.bench or Banking77
- Inventing frames for text games
- Filling missing reward with 0
- Deep stacks of generic-of-generic facades

## Entrypoints

- `CompatPlatform` — pins, leases, logs, `/reward`
- `RolloutEventLog` — fsync journal before a control or semantic event becomes visible
- `runtime_for(spec)` — umbrella
- `create_compat_app(target)` — HTTP edge
- `project_envelopes` — honest headless projection
- `project_harbor_atif` — Harbor-only overlay of the log
- `examples/serve_banking77.py` — loopback `banking77_classify` (default `:8099`, optional `--storage-root`)
- `examples/serve_healthbench.py` — loopback `healthbench_chat` (default `:8114`)
- `examples/deo_nested_reward.py`, `examples/banking77_datagen.py` — headless C4-06 / Banking77 data gen (no Desktop, no `--paid`)

## Live-stream order

Every authoritative run uses the same order: prepare, consume the non-advancing
`stream.subscribed` record from the declared poll URL, then start. Poll, SSE, and
WebSocket read the same journaled envelopes and use semantic `sequence` as the
consumer cursor. The façade fsyncs each envelope before adding it to the
publishable in-memory index; recovery rejects corrupt digests and sequence gaps.
