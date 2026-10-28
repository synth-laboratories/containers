# Engine guard progress-bound proof (real Docker, loaded host), 2026-10-28

**Evidence correction:** The historical restart driver recorded
`attempts_for_op: []` because it looked for an `attempts` field in an
operations-list response, then accepted `len <= 1`. Its signal timestamp was
written after the process-wait loop. The retained completion and removal are
real observations, but that flag does not establish exact-once dispatch or the
signal-time state. The fresh local restart proof in the shared workspace is
`artifacts/codex-gate-a-20261028/RESTART-PROOF.md`. The capacity receipt refused
the fifth operation after four admissions, not a second operation.

Retained for SYN-5206 from the mloky service-qualification run
(`temper_runtime/artifacts/service-qualification-20261028/real-wave4/`, mloky main 3a394c9d via #260).
Guard under test: containers dev 5d098d32 (#51 progress-bound waits + hello handshake, #52 typed refusal payload).

| Proof | Window (UTC) | Host load1 | Engine receipt | Outcome |
|---|---|---|---|---|
| cancel | 16:06:32–16:08 | 50–58 | `cancel/20261028T160632Z/receipts/` | cancelled mid-run; guard reported `removed`; no owner-labelled container left |
| restart | 16:06:52–16:09 | 50–58 | `restart/20261028T160652Z/receipts/` | service restarted while the engine ran; guard kept custody and reported `removed` |
| capacity | 16:08:25–16:10 | 50–58 | `capacity/20261028T160825Z/receipts/` | fifth run refused at capacity with a typed refusal; no container created for that operation |

Acceptance mapping:
- No container with the owner label remains after cleanup: every engine receipt here says `cleanup: removed`
  (the guard runs `docker create`/`start`/`rm -f` to completion and never kills the CLI, so a slow create
  cannot materialize a container after cleanup; a guard that goes silent leaves `cleanup: unconfirmed`
  with `recorded_by: parent` for the sweep, see `native_engine._parent_receipt`).
- Caller wait and guard wall agree: both fixed waits are gone; the parent is bound only by guard reporting
  liveness (heartbeat every 1 s, 120 s silence after `hello`).
- Slow-daemon test: `tests/test_engine_guard_progress.py` (fake Docker CLI with slow create/start/stop and a guard
  killed mid-way), 26 passed three times at host load 37–59.
- An earlier attempt the same day (`cancel/20261028T152500Z` in the mloky tree) failed on a pruned engine image;
  the guard reported `failed` then `removed` correctly. `MLOKY-SUMMARY.md` is the full driver summary;
  `load-gate.txt` the load log.
