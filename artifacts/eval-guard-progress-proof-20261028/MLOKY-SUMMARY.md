# SYN-5165 / SYN-5206 wave-4 real-Docker qualification (2026-10-28, lane D)

**Evidence correction:** The historical restart receipt below recorded
`attempts_for_op: []`; its driver read a nonexistent `attempts` field from
`local_eval_list` and accepted zero as at most one. Its signal receipt was
timestamped after the process-wait loop, not at the signal. The retained
service status reports a completed initial attempt and cleanup, but the old
`proved` flag does not establish exactly-once dispatch or signal-time state.
SYN-5224 hardens the driver; a fresh real-daemon requalification is still
required. The original receipts and historical account below are preserved.

Local proof only; no hosted Rhodes/Swarm acceptance, no provider calls (scripted noop candidate).

## PRs

- Containers SYN-5206 engine guard: https://github.com/synth-laboratories/containers/pull/51
  (branch `claude/syn-5206-engine-guard-20261028`, base `dev` 2a8975dc)
- mloky SYN-5165 receipts/docs: https://github.com/synth-laboratories/mloky/pull/242
- Containers SYN-5165 refusal fields: https://github.com/synth-laboratories/containers/pull/52 (merged 5d098d32)

## SYN-5206 (Containers) — fixed, unit-proven

The guard no longer has fixed waits (45 s startup, 20 s cleanup, 180 s create wall).
`docker create`/`start`/`rm -f` run to completion and are never killed (a killed
`docker create` was the leak: the daemon still created the container after the guard
had already run `rm` and written `cleanup: stopped, container: null`). The guard
heartbeats every 1 s and reports transitions with timestamps and daemon-observed status;
removal is confirmed by observing the `ai.synth.eval.allocation=<name>` label absent.
The parent fails closed only on a terminal state, guard exit, or 120 s with no report,
and never terminates the guard. Receipt: `cleanup: removed|unconfirmed`,
`last_observed_state`, `transitions`, `name`, `label ai.synth.eval.owner=<run>`,
`recorded_by: guard|parent`. Guard death → parent writes `unconfirmed` with the name.

Tests (fake Docker CLI driver, parent silence bound 0.8 s vs 2.5 s slow phases, so a pass
proves progress-bound waiting): `tests/test_engine_guard_progress.py` slow create, slow
start, slow stop, guard killed mid-way; plus `tests/test_eval_engine_override.py`.
`.venv/bin/python -m pytest -p no:cacheprovider -q tests/test_engine_guard_progress.py tests/test_eval_engine_override.py` → 24 passed.

## Real-Docker proofs (SYN-5165)

Driver: `proofs.py <cancel|restart|capacity>` (adapted from `wave3-gate-20261028/journey.py`;
dev MCP server over stdio with `--eval-service autostart`, catalog-bound scripted noop
policy, stock image `evals-craftax-gamebench-rust:local` with plan-declared
`engine_command`/`engine_port`, digests re-pinned from `synth-containers eval identities`).
`run-gated.sh` runs each proof only when the 1-minute load is below 20 (15 s poll, at most
60 polls per proof); every poll is in `load-gate.log`.

## Outcome: all three proofs PASS on real Docker (attempt 2, 16:06–16:10Z, load 50–58)

Attempt 1 (`cancel/20261028T152500Z`, 15:25Z, load 37) did not reach `running`: the plan's
pinned engine image `sha256:77a13963…` (`evals-craftax-gamebench-rust:local`) had been removed
by the unpinned-image prune, so the guard reported `native_engine_create_failed: No such image`
and cleaned up correctly (`failed → stopping → removed`, `cleanup: removed`, 3 s), the run exited 2,
and the driver's `until()` then polled for `running` for 20 minutes because it ignored terminal
phases. Three fixes, all committed here or in Containers: `proofs.py` resolves the image id from
the tag at run time and stops when every operation is terminal before the awaited condition;
the image was rebuilt from committed sources (`evals/containers/images/craftax-gamebench-rust/
build_nanohorizon_image.sh`: evals 05fe954f, containers ca2f7995, gamebench bfad00f4; base image
`d82b2771566e`, tagged `:local`) and the tag is now a durable pin in
`synth-dev/config/pinned_images.txt`; Containers PR #52 makes the public refusal keep the typed
engine error's code/stage/name/last_state/cause instead of collapsing to `NativeEngineError`.

The load gate (< 20) was overridden by the coordinator for attempt 2: the load was other
sessions' containers and workers, and the point of the progress-bound guard (containers#51) is
that proofs must hold under load. Containers witness checkout at containers#52 head (2dd1a29,
includes #51 ea76769). Every receipt is under `<mode>/<stamp>/receipts/`; the final one carries
`result.proved` and `docker_ps_owner_label_final`. Runtime state under `<mode>/<stamp>/root/`
(SQLite stores, service socket records) is not committed.

| Proof | Real Docker (attempt 2) | Fake executable (earlier, `../fake/`) |
|---|---|---|
| cancel (running run → `cancelled`, container gone) | PASS `cancel/20261028T160632Z`: 18 s, final phase `cancelled` (source `service`, attempt `cancelled`), owned containers after: none | PASS (scenario e) |
| restart (SIGKILL serve mid-run → up → true terminal state, `source: service`, one attempt) | PASS `restart/20261028T160652Z`: 91 s, serve pid 78639 killed, new pid 88325, final phase `completed` from the service with one dispatch record set, owned containers after: none | PASS (scenario d) |
| capacity (4 running, 5th refused `concurrency_exhausted`, owned running ≤ 4) | PASS `capacity/20261028T160825Z`: 68 s, cap-1..4 admitted, cap-5 refused `{code: concurrency_exhausted, service: synth-local-eval, stage: admission, cause: "4 active runs at the bound of 4; nothing was queued", detail: {active: 4, bound: 4}}`, max owned running observed 2 (sampled every 2 s), all four then cancelled, owned containers after: none | PASS (scenario f) |
| SYN-5206 guard: no Created leak when create/start outlasts a wait | observed under a real daemon at load 50–58 in all three proofs (every receipt `cleanup: removed`; `docker ps -a` by owner label empty after each) and in attempt 1's missing-image failure | n/a |

The service does not queue: per `docs/local-eval-service.md` the N+1 submission is
refused with `concurrency_exhausted`, and the capacity proof asserts exactly that.
Restart confirmed the open question below: the dispatched `synth-containers eval run` survived
serve death and the service reconciled `running → completed` (not `unknown`).

## Steps as run (kept for reruns)

1. On a calm host (1-minute load < 20), from `temper_runtime/`:
   `export TMPDIR=/Users/joshuapurtell/GitHub/.tmp MLOK_RUNTIME_DIR=/Users/joshuapurtell/GitHub/.tmp/mlok-sockets`
   and set `PROOF_CONTAINERS` to the selected pinned checkout's executable;
   then `artifacts/service-qualification-20261028/real-wave4/run-gated.sh cancel restart capacity`.
   Containers must be at containers#51's head in the witness checkout (editable install).
2. Each run writes `real-wave4/<mode>/<stamp>/receipts/NN-*.json` and `<mode>-run.log`;
   `NN-summary.json` carries `result.proved` and `docker_ps_owner_label_final` (must be `[]`).
   The driver deletes `root/**/.service/token` before exit; re-check before committing.
3. The driver has not executed end to end yet (written this session, never run against
   the live MCP server). Expect to fix field names on first run: `phase`/`source` lookups
   are copied from `journey.py`; `local_eval_list` attempt rows are assumed to carry
   `operation_id`. Default `steps` is 3000 so a run stays `running` long enough to cancel,
   kill serve and fill capacity; pass a second argument to change it.
4. Record results in the table above and in `../QUALIFICATION.md`, then mark the mloky PR
   ready.

## Open questions

- Restart: the dispatched `synth-containers eval run` lives in its own session and
  survives serve death (fake scenario d observed `running → completed`), so the real
  restart proof will show reconciliation `running → <terminal>`, not `unknown`.
- Should real-Docker qualification get a reserved calm-host window? At load 35 to 77 the
  engine create alone took 64 s to 187 s in earlier waves.
