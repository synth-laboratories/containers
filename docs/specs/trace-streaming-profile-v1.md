# Trace Streaming Profile v1

**Status:** Containers-owned kit. Floor already has a durable sequence log.  
**Normative suite table:** `workshop/docs/live_evals.md` (TS-A…E).  
**Wire protocol:** `docs/trace_stream_protocol_v1.md`.

This file is the BCP 14 profile for that suite. It does not invent a second
event schema. Semantic events are `synth.trace-stream-event.v1`. Prepared
descriptors are `synth.rollout.stream.v1`. Sealed evidence is `synth.trace.v5`.

## MUST

1. Discovery returns one versioned stream descriptor with stable
   `rollout_id` / `stream.id` and declared transports. Absent transports are
   `null`, not guessed URLs.
2. The first semantic event is `trace.opened`. Exactly one `capture.closed`
   occurs. Heartbeats and `stream.subscribed` are control and MUST NOT advance
   the evidence cursor.
3. Nested session/span subjects open parent-before-child and close
   child-before-parent.
4. Missing reward, usage, sequence, and score stay unavailable. They MUST NOT
   be normalized to `0`, empty success, or `completed`.
5. Secrets (headers, tokens, nested credentials) MUST NOT be persisted.
6. Poll, SSE, and any advertised WebSocket yield the same ordered envelope
   IDs and digests.
7. `capture.closed` then a verified seal digest. Stream EOF is not completeness.

## SHOULD

Unknown namespaced data kinds (`x.*`, task-family NEV kinds) survive relay
without being rewritten as core kinds.

## MUST NOT

- Copy producer cursors (`nev_cursor`) onto the consumer stream.
- Treat Workshop TS-E as passed from this kit. TS-E is a Desktop consumer
  gate (A1 / A5). Containers proves the producer profile.

## Conformance

Runnable tests live at `tests/conformance/trace_stream/`. IDs match
`live_evals.md`. Gate E is skipped here on purpose.
