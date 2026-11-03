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

## Installed-wheel follow-up and EOF correction

The installed `0.4.7.dev2026110302` wheel repeated the real execution timeout
under backend Python 3.11 after its nine selected unit cases passed. This is
retained in backend `rhodes-installed-finite-wheel-retry.log`; it supersedes
any interpretation that the earlier timeout was purely transient or that
explicit attach flags alone qualified the path.

The final input write previously closed stdin only on a second write-ready
event. The source now delivers EOF immediately after the final successful
write. A real OS-pipe regression proves exact bytes and EOF after a single
event, without relying on a second readiness notification. Timed-out local
control clients are also killed/reaped within a bounded wait; this never proves
their original provider effects absent or cancelled. Original create_pending
recovery remains in doubt until its real owner proves the outcome.

`finite-eof-control-regressions.json` reports 11 unit tests passed. An immediate
real rerun encountered a Docker create timeout before execution; that original
intent remains explicitly uncertain, with exact names retained for owner
reconciliation. Further real/wheel qualification follows in the retained
receipts, rather than being inferred from the source correction.

The installed dev03 wheel passed eleven unit cases but its real test timed out
at Docker create before starting the child. The same source passed a quiet
editable real run without increasing its original 60-second budget. Inspection
found that noninteractive control clients inherited ambient worker stdin; the
source now binds those control clients to DEVNULL. The attached workload still
receives only its separate exact declared input pipe. An actual OS-pipe test
keeps ambient stdin open with a sentinel and proves the control subprocess sees
immediate EOF and no sentinel bytes. `finite-control-stdin-isolation.json`
reports twelve unit tests passed. Installed real qualification of this final
control boundary follows in subsequent receipts; its runtime causal effect is
not inferred from the unit result.

## Final installed and original-identity checks

The installed dev04 wheel from `1c29152` passed all twelve unit cases. Its
real lifecycle case, run alone, still timed out during Docker create at the
15-second control guard, before the workload started. A separate fresh owned
create-only probe also timed out after 30 seconds with no stdout or stderr.
Inherited caller, stripped runner and SDK control environments resolve the
same Docker executable, context, socket and daemon ID; all three DOCKER_HOST,
DOCKER_CONTEXT and DOCKER_CONFIG variables were absent. The input isolation
fix is therefore a tested boundary correction, not an established explanation
for this unresolved Docker behavior. No timeout guard or original execution
budget was increased. Final installed real qualification remains blocked.

Static recovery review then found that a signed terminal record was checked
against the original epoch and daemon but not the signed allocation's immutable
container ID. Recovery now requires the terminal schema and exact container ID
from the signed creation/cleanup lifecycle. Two signed-spool negative cases
exercise conflicting IDs and a lifecycle-shaped terminal body. The final
`finite-terminal-identity-regressions` batch passed all fourteen unit cases;
root owns the matching final wheel and installed-source verification.

The dev03 late allocation was removed only after its actual ownership label
matched its retained original scope. The two dev04/probe create intents remain
explicitly uncertain: bounded inspect currently reports exact no-such-object,
which does not disprove a delayed server commit. Exact names, scopes, retained
spool locations and reconciliation receipts are in the backend handoff. Root
coordinates their final owned-only cleanup; these pending intents never
authorize a second original execution.

At 20:29 UTC both retained late creates appeared in `created` state. Final
backend reconciliation matched their original scopes and daemon, removed only
their immutable IDs, and confirmed exact ID/name absence. No workload was
started. Both observed original effects are reconciled; the failed real gate
remains unqualified. Root then communicated a shared-host recovery hold: all
this lane's jobs have drained, and no new DB/Docker/build/test work is submitted
until that hold is explicitly cleared.

## Final paired installed qualification

After the recovery owner explicitly released the host hold, root built immutable
recorder06 from source `3951e33` and installed it alongside SDK05. All 21 recorder
cases passed, including the four real cached-image Docker cases, lost create and
terminal ACK recovery, first archive seal and the signed no-dispatch fence.
The exact installed module SHA256 was
`3d725b3427039c186c86d0e5a425521eb62573e315ef7a548e9541c809b5250f`.
Backend receipts `final-installed-finite06-all` and
`final-installed-finite06-inventory.json` retain the installed path/version and
zero remaining finite allocations. All 324 Python modules match the committed
wheel export byte for byte. This supersedes the earlier installed-runtime block
for the tested local profile; it does not establish a source cause for the
earlier shared Docker timeout, or installed Artifact/Mnemosyne owner custody.
