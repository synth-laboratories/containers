# Synth trace-stream protocol v1

**Status:** implemented normative Containers profile  
**Envelope:** `synth.trace-stream-event.v1`  
**Prepared descriptor:** `synth.rollout.stream.v1`  
**Terminal evidence:** `synth.trace.v5`

This is the opinionated event protocol shared by native Containers targets and
compatibility folds. It is a small lifecycle protocol—prepare, subscribe,
start, sustain, close, seal—rather than a bag of benchmark-specific JSON.
Harbor is the only first-class compatibility fold in v1. GameBench is task or
dataset identity, never a source format.

## 1. Lifecycle

1. `POST /rollouts/prepare` allocates `rollout_id`, the stream `id`, retention,
   cursor semantics, reward URL, and the exact allowed transport URLs. It does
   not start the environment or policy.
2. The consumer connects to a declared poll, SSE, or WebSocket URL and requires
   the non-evidence control envelope `stream.subscribed`.
3. Only after that barrier may `POST /rollouts` mutate the prepared rollout.
4. The producer fsyncs each envelope before exposing it through any transport.
5. Partial spans open, emit zero or more data/plan records, and close on the
   same monotonic rollout sequence as environment events.
6. Authoritative terminal status closes the append-only journal. `/reward`
   remains missing/incomplete until its evaluation authority can score it.
7. The sealed trace must match the live high-water and exact ordered event
   digest. Consumers can reopen it after the environment/policy service exits.

Starting without `stream.subscribed`, switching to an unadvertised transport,
guessing `/events`, or changing retention after prepare fails closed.

The normalized HTTP surface is plural and rollout-scoped:

```text
POST /rollouts/prepare
POST /rollouts
GET  /rollouts/{rollout_id}
GET  /rollouts/{rollout_id}/events?after={sequence}&limit={evidence_count}
GET  /rollouts/{rollout_id}/stream
WS   /rollouts/{rollout_id}/ws
GET  /rollouts/{rollout_id}/reward
POST /reward
```

Wire JSON uses `snake_case`. The stream descriptor uses `id`, nested
`transports.{poll,sse,websocket}.url`, `cursor`, `reward.url`, `auth`, and
`retention`. Flat aliases such as `poll_url`, `sse_url`, `stream.id`, and the
singular `/rollout` route are not part of the normalized API. A compatibility
adapter may accept a legacy route privately, but must advertise and emit only
the normalized surface.

### 1.1 Timeout and idempotency

`rollout_id` is the mutation idempotency key and is allocated by the caller
before its first request. A timeout is an unknown transport outcome, not proof
that the mutation did not land.

- Repeating prepare with the same `rollout_id`, transport, and retention
  returns the existing descriptor with `replayed: true`; changed bindings are
  `409 rollout_prepare_identity_conflict`.
- Repeating start with the same `rollout_id`, task identity, policy reference
  (including code), transport, and retention returns current state with
  `replayed: true`; changed identity is `409 rollout_identity_conflict`.
- `GET /rollouts/{rollout_id}` distinguishes prepared, running, and terminal
  state after an ambiguous disconnect.
- Recovery reads the descriptor-declared poll URL with
  `after=<last durable sequence>`. The returned `cursor.high_water` is the next
  checkpoint and `cursor.closed` is authoritative terminal delivery state.
- Poll pages return at most `limit` evidence envelopes plus any non-advancing
  subscription control record. `cursor.next` is the last evidence sequence in
  the page and `cursor.has_more` requires the consumer to continue from that
  exact cursor. A page that claims `has_more` without advancing `next` is
  invalid and must fail closed.

SSE reconnect uses `Last-Event-ID`. WebSocket consumers backfill through poll
before reattaching. No reconnect path allocates a new rollout ID or infers that
missing events mean the rollout should be rerun.

## 2. Event envelope

```json
{
  "schema": "synth.trace-stream-event.v1",
  "kind": "span.policy.data",
  "ts": "2026-08-12T19:04:53.092744Z",
  "control": false,
  "sequence": 5,
  "event_id": "5",
  "digest_schema": "synth.envelope-digest.v2",
  "digest": "1f0bb1d82f55e24c",
  "payload": {}
}
```

- Evidence `sequence` is an integer beginning at 1 and increasing by exactly
  one within a rollout. A consumer cursor means “events with sequence greater
  than this value.”
- Control records have `sequence = null`, `control = true`, and do not advance
  evidence high-water. In JSON the sequence field is omitted.
- `event_id` is the decimal sequence for evidence and the kind for control.
- `digest_schema` names the byte contract for `digest`. New envelopes use
  `synth.envelope-digest.v2`, whose tagged encoding is specified in
  `docs/specs/envelope-digest-v2.md`; legacy persisted envelopes may omit it.
- `digest` is the first 16 hexadecimal characters of SHA-256 over the bytes
  selected by `digest_schema`. Timestamps are provenance but do not change
  semantic identity.
- `payload` is a JSON object. Secret-bearing keys/values are rejected before
  persistence.
- Transport wrappers may add `rollout_id`; they may not rewrite the envelope.

The journal persists `{record:"envelope", envelope}` lines plus one terminal
`{record:"closed", high_water}` line. Recovery refuses malformed JSON,
unknown records, digest drift, sequence gaps, a mismatched close high-water, or
records after close.

## 3. Span partials

Spans use ordered event kinds rather than mutable snapshot replacement:

```text
trace.opened
  span.policy.opened
    span.policy.data       zero or more provider/model/usage/retry facts
    span.policy.plan       selected action plan
  span.policy.closed
  observation → frame → action → reward_signal  repeated as applicable
status                     authoritative terminal state
trace sealed               same exact high-water
```

`opened` allocates identity and declares the call/harness. `data` sustains the
span with observed provider/model output, nullable usage/cost, parse attempts,
and action authority. `plan` records the bounded actions selected for the
environment. `closed` records completion; it does not infer data that never
arrived. A failed or cancelled span still closes with explicit error status.
Private chain-of-thought is not a protocol field. Producers may emit a concise
policy rationale only when the provider/harness explicitly returns one for
display.

## 4. Target profiles

The envelope is shared; event vocabularies are profile-specific:

| Profile | Required/allowed facts | Forbidden claims |
| --- | --- | --- |
| Native Craftax | observation, immutable frame URL, action, RewardSignal, achievements, policy spans, status | invented map/frame, GameBench as source format |
| Harbor fold | trial lifecycle, tools/stdout, verifier/reward evidence, child resource refs | parent claiming child Craftax frames |
| OpenEnv compatibility | observation/action/env reward and only advertised checkpoint semantics | inferring restore from reset |
| dig.bench relay | session, text observation, legal actions, stats, action/invalid action, status | frames, Harbor/OpenEnv wrapper |

Unknown task-specific payload fields are preserved as data, but they do not
grant an affordance. The prepared target metadata is the authority for
`require | prefer | unused` capability negotiation.

## 5. Environment, policy, and evaluator ownership

- The **environment service** owns state, accepted actions, observations,
  frames, native rewards, achievements, and terminal status.
- The **policy service** owns model/provider/config identity, policy spans,
  proposed actions, usage, and independently restartable harness state.
- The **evaluation plan** declares reward nodes and authority. Environment
  reward may be an online sum; Harbor/script/verifier scoring is usually
  terminal or asynchronous. Missing reward is `null`/omitted, never zero.
- A policy may restart between rollouts while the environment/world stays
  alive. An environment may restart while immutable policy code/config remains
  pinned. Neither operation may mutate an in-flight rollout pin.

Child evaluations link through `synth.resource-ref.v1` with slot `stream` and
their own rollout/reward identities. Optimizers emit `optimizer_event.v1` for
search/training state and link these children; they do not copy environment
frames into optimizer events.

## 6. Transport equivalence

Poll, SSE, and WebSocket expose the same persisted envelopes and sequence
cursor. SSE uses `id: <sequence>` and accepts `Last-Event-ID`; a heartbeat is a
comment and never advances sequence. WebSocket is intended for interactive
control or binary delivery but remains journal-equivalent. Poll is the bounded
backfill and recovery baseline. A relay may derive SSE from poll only when it
preserves identities and persist-before-publish order.

## 7. Acceptance floor

The conformance suite proves:

- prepare/subscribed-before-start and frozen transport/retention;
- idempotent prepare/start, immutable identity conflicts, and authoritative
  status recovery after ambiguous disconnect;
- exact monotonic cursors, replay de-duplication, and no evidence gaps;
- fsync-before-publish plus fail-closed corruption/secret recovery;
- transport equivalence and declared-URL-only clients;
- reward incompleteness, async evaluation, and missing-not-zero projection;
- concurrent rollout isolation and immutable policy pins;
- terminal high-water/digest reconciliation and producer-independent reopen;
- restart-safe Trace V5 capture spools without reopening durable terminal
  captures.

Run the repository acceptance with:

```bash
uv run --with pytest pytest -q
```

The bounded real Craftax acceptance is
`examples/craftax_muse_ten_seeds.py`. It must connect all ten streams before
start and reject any fallback policy evidence; it is paid and therefore not a
normal PR test.
