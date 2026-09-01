# Live Annotation Protocol v1

**Status:** implemented in `synth_containers.live_annotation`; proven over the compat façade (`tests/test_live_annotation_*.py`) and over a real Craftax seal replayed through the isolated host (evals `tests/test_craftax_live_protocol.py`).
**Relation to post-hoc annotation:** additive. The sealed `tracing/annotation` path stays the evidence authority; this lane only adds *provisional* findings while a rollout is still running.
**Workshop consumer:** `workshop/docs/HANDOFF_LIVE_ANNOTATION_PROTOCOLS_2026-09-01.md`.

## Product sentence

A live annotation protocol is caller-supplied, digest-pinned code that runs beside a rollout in its own process, consumes the rollout's durable event stream as it grows, and emits a second durable stream of provisional annotations: achievements as they unlock, milestones as they are reached, failure modes as they are detected, plus bounded model-assisted judgments. Twenty parallel Craftax rollouts are unreadable as raw streams; their annotation streams are the middle ground.

## Causal boundary

The protocol is **observe-only**. It:

- reads the rollout log through the same `after(sequence)` cursor a remote consumer uses;
- has no handle on the world, the policy, the pin, or the runtime;
- runs with `python -I -S` in a scrubbed environment: stdlib only, no site-packages, no network of its own;
- cannot change reward, achievements, terminal status, or the rollout's own log;
- is not visible to the acting policy, and the policy is not visible to it.

Its output is labelled `provisional` end to end. Nothing on the annotation stream is ever promoted into `synth.trace.v5` or an evidence head by this lane; the post-hoc annotators reconcile against the sealed trace and remain the authority.

## Protocol code contract (`synth.live-annotation-protocol.v1`)

A protocol is one Python module, stdlib only:

```python
PROTOCOL = "synth.live-annotation-protocol.v1"   # required marker
PROTOCOL_ID = "craftax.live.v1"                  # optional; must match the install's protocol_id

class Protocol:
    def __init__(self, config: dict): ...
    def on_event(self, event: dict) -> list[dict] | dict | None: ...           # required
    def on_model_result(self, request_id: str, result: dict | None, error: str | None) -> list[dict] | None: ...  # optional
    def on_close(self) -> list[dict] | None: ...                                # optional
```

`event` is `{"kind", "sequence", "ts", "payload"}` for every semantic envelope of the rollout stream, in sequence order; control records are not delivered. `on_close` runs once the rollout log closed (or the runtime returned) and everything durable was consumed.

### Emissions

| `op` | Fields | Published as |
| --- | --- | --- |
| `finding` | `finding_id`, `kind`, `label`, `step?`, `confidence? (0..1)`, `evidence: {sequences: [...]}`, `supersedes?`, `detail?` | `annotation.finding` with `status: "provisional"` |
| `retract` | `finding_id`, `reason` | `annotation.finding.retracted` |
| `metric` | `name`, `value`, `step?` | `annotation.metric` |
| `model_request` | `request_id`, `instructions`, `context`, `schema?`, `max_output_tokens?` | `annotation.model.requested`, then `annotation.model.completed` / `annotation.model.failed`, then `on_model_result` in the child |

Rules the runner enforces (a breach is recorded as `annotation.protocol.error` and the emission is dropped; the rollout is unaffected):

- `finding_id` is unique per rollout; `supersedes` must name an active finding; `retract` must name an active finding;
- evidence sequences are positive integers on the source stream (`evidence.stream_id` is filled in by the runner);
- `detail` ≤ 16 KiB, no credential-shaped keys; instructions ≤ 32 KiB; context ≤ 128 KiB;
- model calls are counted against `configuration.model.max_calls` and the runner's ceiling; output tokens are capped;
- a protocol exception ends the stream with outcome `protocol_failed`; the stream still seals.

### Model requests

The child has no network and no credentials. It emits `model_request`; the container executes it with the settings in the installed `configuration.model` block (`model`, `base_url`, `api_key_env`, `max_calls`, `max_output_tokens`, `timeout_seconds`), reading the key from the environment at call time, and hands `{"text", "parsed"}` back through `on_model_result`. Prompt bodies are never persisted; the stream carries their digests, sizes, usage and provider request id.

## Wire contract

### Install and read

```
PUT  /annotation-protocol   {code, protocol_id, configuration?, source_revision?}
GET  /annotation-protocol   -> synth.container-annotation-protocol.v1
```

Revision id: `anprev_<sha256[:16]>` over `{code_sha256, protocol_id, configuration, source_revision}`. Re-installing identical inputs is idempotent (`idempotent: true`). The install boots the code once in isolation and refuses (`422 protocol_boot_failed`) anything that cannot start; a declared `PROTOCOL_ID` that disagrees with `protocol_id` is `422 protocol_identity_mismatch`; credential-shaped configuration keys are `422 protocol_credential_forbidden`. All installed revisions stay available; `current` is only the default identity reported by `GET`.

`configuration_digest` is `"sha256:" + sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":")))`, the same form as `PUT /policy`, so Workshop's canonical-JSON digest round-trips.

### Bind to a rollout

`POST /rollouts/prepare` and `POST /rollouts` accept `annotation_protocol_revision_id`. Unknown → `404 annotation_protocol_unknown`. The pin is part of rollout identity: a replay with a different pin is `409 rollout_identity_conflict`. The pin persists in the completed-rollout manifest and is echoed on every rollout record.

When bound, the stream descriptor declares the sibling channel (never guessed):

```json
"annotation": {
  "schema": "synth.live-annotation-stream.v1",
  "id": "stream:<rollout_id>:annotations",
  "events": "/rollouts/<rollout_id>/annotations/events",
  "stream": "/rollouts/<rollout_id>/annotations/stream",
  "cursor": {"kind": "sequence"},
  "kinds": ["annotation.protocol.bound", ...],
  "status": "provisional"
}
```

`annotation` is `null` when no protocol is bound.

### The annotation stream

Same envelope (`synth.trace-stream-event.v1`), same durable journal, same `poll_payload` / `iter_sse`, own sequence space. Every row carries `rollout_id` and `stream_id` so a viewer can fold it with the rollout stream by rollout and de-duplicate by `(stream_id, sequence)`.

Order:

1. control `stream.subscribed`;
2. `annotation.protocol.bound` — revision, protocol id, configuration digest, judge model, isolation receipt;
3. any of `annotation.finding`, `annotation.finding.retracted`, `annotation.metric`, `annotation.model.requested|completed|failed`, `annotation.protocol.error`, each with `source_sequence` (the rollout sequence the protocol had consumed) and `protocol_revision_id`;
4. `annotation.closed` — outcome (`completed`, `protocol_failed`, `spawn_failed`, `runner_failed`), counters, `source_closed`;
5. `capture.high_water`, `capture.closed`, then the journal closes.

The stream seals after the rollout journal: the runner drains everything durable, waits (bounded by `drain_timeout_seconds`, default 30 s) for in-flight judgments, flushes `on_close`, and closes. Consumers should expect `capture.closed` on the annotation stream shortly after the rollout's own.

Poll pages add `schema: synth.live-annotation-stream.v1` and a `summary` of the runner while it is in memory.

### Advertisement

`GET /info` → `capabilities.operations["annotation.live"] = true`, `capabilities.operations["annotation.protocol.put"] = true`, and a `live_annotation` block (`mode: observe_only`, `findings_status: provisional`, kinds, `installed`).

## Durable layout

```
<storage_root>/live_annotation/protocols/<anprev_…>.json   immutable revisions (code + identity)
<storage_root>/live_annotation/protocols/current.json
<storage_root>/live_annotation/events/<sha256(rollout_id)>.jsonl
```

Both recover on restart; a closed annotation stream is served with the producer gone.

## Selectors and post-hoc reconciliation

Provisional evidence cites `(stream_id, sequences)` on the rollout stream. In the sealed `synth.trace.v5` document those sequences are the events' `event_id` / `order.chronological_sequence`, so a post-hoc annotator can resolve every provisional citation against the sealed trace and either confirm (emit a sealed finding with an exact selector) or drop it. This lane never writes into evidence heads itself.

## Source map

- `src/synth_containers/live_annotation/contract.py` — code and emission contracts, kinds, revision id
- `src/synth_containers/live_annotation/process.py` — `IsolatedProtocolProcess` (`-I -S`, JSONL)
- `src/synth_containers/live_annotation/runner.py` — per-rollout tail → protocol → stream, ceilings, sealing
- `src/synth_containers/live_annotation/model.py` — parent-side model caller
- `src/synth_containers/live_annotation/service.py` — revisions, streams, attach/detach, recovery
- `src/synth_containers/live_annotation/api.py` — routes
- `src/synth_containers/platform/state.py` — pin, descriptor channel, attach around `simulate`, advertisement
- `src/synth_containers/event_log.py` — `wait_for_change` / `wake_readers`

First protocol: `evals/domains/craftax/annotations/live_protocol.py` (`craftax.live.v1`).
