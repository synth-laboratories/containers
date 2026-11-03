# Containers candidate integration — 2026-11-03

Integrated source `e24710045e695976aa39f59eea7a61c99ff85dde` onto current
`origin/dev` `037f2e5e49ca8d7ce561d55e617eb757c1c9602c`; Git preserved the
upstream Harbor evidence changes without conflicts. Candidate merge is
`730ebce699ab22d31060f0e1b304f916a883248a`.

All commands, exit codes, full failure logs and final successful logs are
retained beside this file. The candidate owns its Python 3.13 environment;
`uv sync --locked --no-python-downloads --no-default-groups --group dev
--extra finite-recorder` installed the locked dependencies without changing
foreign environments. No provider, paid compute, production service or
foreign container was invoked or changed.

Executed checks:

- `sdk-catalog.result.json`: 104 SDK/CLI/catalogue/build-operation tests passed.
- `trace-api-regression.json`: 114 pool/API/metadata/trace/Harbor regressions
  passed. Two existing FastAPI/Starlette deprecation warnings were retained.
- `finite-unit-final.json`: 9 recorder tests passed, 4 real Docker cases
  deliberately deselected for that unit-only batch.
- `finite-terminal-fixed.json`: real cached native BusyBox execution, exact
  stdin/stdout, signed terminal archive, confirmed cleanup, same-archive
  recovery and refusal of second execution passed.
- `finite-ack-seal-fixed.json`: 11 tests passed, 1 already-covered case
  deselected. Includes real lost-create ACK, lost-terminal ACK, and crash
  before first archive seal followed by signed-original-only recovery and
  independent Trace V5 validation. No original workload was re-executed.

The cached test image was
`sha256:a46257e06c82ad85242d0d38077d0a52d9896389ca60a1a59d3463db586e5fa4`
(Linux arm64). Test signing keys were generated solely inside owned test
scratch and removed by the fixture; operator keys were never read. Actual
Docker cases require explicit `RHODES_FINITE_TEST_IMAGE` and never pull an
image. Current final inventory reports zero finite test allocations.

The first actual test exposed an exact Docker CLI compatibility defect:
absence is reported as lowercase `error: no such object: <exact locator>`.
The source now recognizes only this additional exact form while preserving
responsive same-daemon/double-read requirements and transport-error refusal.
Create now explicitly attaches all three streams and verifies actual
OpenStdin/StdinOnce/Attach/Tty state before execution.

An initial owned cat execution timed out, and its inspect/rm briefly blocked
despite healthy daemon information. Its delayed inspection showed StdinOnce
and all Attach flags already true, so missing attach flags are **not a proven
cause** of that initial timeout. Later positives passed, but the initial hang
cause remains unknown and is retained in the failure logs. Its exact original
ID and all late-created original intents were subsequently reconciled and
removed with matching ownership labels; cleanup receipts are retained.
Pressure papercut `42516308-52ed-4edc-8041-25b62e0419df` records both the
observed defect and this corrected interpretation, and remains open.

These checks qualify the tested local recorder/SDK paths. They do not prove
installed Artifact/Mnemosyne custody, protected operator key installation,
hosted services, owner revocation under live load, or the later study gates.
The paired backend wheel must use the final candidate source after these
changes; root owns packaging and backend dependency pins.
