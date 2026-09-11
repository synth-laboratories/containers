# Hosted native journal custody

`synth-containers publish-native-journal` uploads a bounded sanitized native
limit journal to an already accepted hosted rollout. It uses the existing
`PoolClient` transport and authorized `SYNTH_API_KEY` / `SYNTH_BACKEND_URL`
configuration. The worker must supply its current execution epoch; this command
does not accept work, reserve spend, acquire ownership, or run an evaluation.

```sh
synth-containers publish-native-journal /run/limit-events.jsonl \
  --run-id RUN --pool-id POOL --rollout-id ROLLOUT \
  --execution-epoch EPOCH --producer-id LIMITS \
  --checkpoint /run/custody-receipts.jsonl --complete
```

The backend accepts the typed `synth.rollout-limit-event.v1` summary envelope,
not arbitrary logs, prompts, credentials, or optimizer events. Unsupported
journals need a declared projection with separate provenance. No source event
is silently redacted or rewritten by this uploader.

Pages contain at most 128 records and 256 KiB. A per-checkpoint process lock
prevents two uploaders from advancing the same local cursor. Append-and-fsync
receipts advance only after an identity-checked backend acknowledgment. Lost
responses replay the same IDs, and changed content conflicts at ingest. On
restart the acknowledged source prefix digest is checked. A server cursor ahead
of the submitted page never causes later local events to be skipped.

`--complete` is valid only after the producer stops appending. Exit 2 with
`has_more: true` means the bounded page budget was exhausted; invoke again with
the same checkpoint. Local readers cap a journal at 64 MiB; larger retained
histories require an indexed reader or explicit segmented producers.

A custody receipt confirms an operator journal in the hosted database. It does
not mean the scientific result or attachments were published and does not seal
Trace V5. The Workshop projection keeps these outcomes separate. Provider runs
and restart/duplication qualification belong to the receiving testing engineer.

## Hosted process capture recovery

The canonical `PoolClient.read_native_agent_capture(pool_id, rollout_id, capture_id)`
reads the original backend-admitted capture and coverage.
`publish_native_agent_capture(...)` retries publication of those original bytes;
it never reruns the agent. Open captures require backend epoch fencing before
interrupted recovery. These methods first ship in the unpublished dev29 candidate.
They do not establish whole-attempt, visual, or lossless live capture.
