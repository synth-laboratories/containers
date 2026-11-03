

### Finite isolated stdio capture source (2026-11-03)

The opt-in `attempt_capture_export.finite_contract` supports only
`isolated_stdio_v1`. It binds one actual succeeded Docker source build and the
selected qualified release to authorized retained source TAR bytes and a
single-platform OCI layout TAR. The verifier checks exact package files and
executable bits, grader/policy source paths, image/config/layer hashes, bounded
uncompressed layer diff IDs, and the original build/source/image association.
Independent source and image declarations cannot establish that association.
The actual provider creates a separate, named, labelled original container:
read-only rootfs, network none, nonroot, no capabilities, no new privileges,
fixed PID budget, exact CPU/memory limits and only an empty read-only shm bind.
Candidate callbacks and host credential injection are excluded by this profile.

The original input, sanitized effective configuration, source closures and
policy are held before dispatch. Exact stdin/stdout/stderr and host-observed
stream journal are recorded into a fsynced owner-only original spool. A protected
operator signer binds terminal exit, drained streams, actual image, command,
resources, original epoch and closure hashes. It never runs in the child.
A second immutable `terminal_publication` retains all actual output objects;
`terminal_native` is the concrete Mnemosyne owner ExactReference. Only after
terminal proof, cleanup, self-contained native archive verification, current
Artifact holds and actual native owner retention can full verification become
true. Pre-execution custody always retains `terminal_capture_pending`.

Installation supplies Containers `finite-recorder` extra (cryptography), Docker,
an absolute `RHODES_FINITE_RECORDER_PRIVATE_KEY_FILE` (regular file owned by the
worker UID, mode 0600), absolute `RHODES_FINITE_RECORDER_SPOOL_DIR` (owned directory,
mode 0700), and `RHODES_FINITE_RECORDER_COHORT_JSON` with schema_version
`synth.rhodes-recorder-cohort.v1`, cohort_id, key_id, exact recorder module_sha256
and public_key_pem. No key was read/generated/enrolled during this source work.
The operator must pin the installed recorder source after building the cohort.
Current original-user native-use consent, project/source READ+EXPORT rights,
sufficient original Artifact/Mnemosyne quota and genuine owner modules remain
mandatory. Logical owner quota is never physical storage assurance.

Recovery never launches work: it inspects only the original deterministic name
and matching rollout/epoch label, verifies exact daemon No-such response plus
responsive fresh daemon read before absence, and rereads protected fsynced
terminal spool bytes. An admitted create without an ACK remains in doubt when
absent. Missing terminal files preserve partial evidence without full promotion.
Spool retirement is deliberately left to an operator after durable custody.
Unsupported remote/Harbor/post-handoff/grader/image encodings still refuse full
required capture before dispatch. This source has not been imported, compiled,
tested or executed. Root completed dependency metadata lock resolution with
builds, installs and Python downloads disabled; the finite-recorder extra is
now in uv.lock. Root owns immutable wheel packaging. Operator installation and
all positive/negative crash/runtime/custody checks remain deferred to
integration/testing under the user's explicit hold.

Finite admission requires enough original reservation for source/image bytes,
twice the declared preexecution archive bound, up to 32 MiB terminal publication
bytes and at least 16 Artifact objects. A separate concrete Mnemosyne reservation
with only required trace kind `trace` and at least 8 MiB is protected before
dispatch. Actual consumption can still refuse; it never overrides quota.
Original signer unavailability is a typed predispatch refusal. The finite
container never borrows the shared task deployment handle during cleanup.


### Source cross-lane audit follow-up (2026-11-03)

The producer now calls the concrete owner `lock_current_closure` in its final
transaction before promoting full capture. The owner locks live original
reservations, original credential version and current exact object/native
holds/grants through commit. The producer checks the public receipts did not
change and validates full receipt coherence: complete role digests, no opaque
credential references, same-epoch committed terminal objects, and actual
Mnemosyne native reference matching the terminal native archive digest. This
is an observation under the then-current retained owners, not a promise that
an explicit later owner release cannot withdraw custody.

The source audit identified and sent actual owner fixes for inventory.json
authorization, private native trace read-before-grant, and exact final owner
fencing. Root corrected source proxy selection for native task source pins
without requiring duplicate original.inputs and tightened Forge full guards.
The maximum source-ref tuple is 130 (128 policy refs plus finite source/image,
with identical registered source deduplicated). No runtime claim follows from
these source inspections.

Recovery first checks the exact signed original input and stream digests and
rebuilds stdout/stderr from the fsynced chunk journal. If no archive exists AND
no terminal/native publication identity was pinned, it may seal the first
Trace V5 from these actual original observations. Provenance explicitly records
`signed-original-stdio-recovery`, the original proof digest and materializer
module digest. O_EXCL/fsync prevents replacement of an existing archive; a
partial or conflicting original archive remains a refusal. No process or
workload is run by recovery. Missing unsigned/unfinished terminal observations
stay partial.

Before actual create, the recorder durably transitions `preflight` to
`create_pending` under a live original ownership check. Recovery can close a
proven no-effect preflight; create_pending or legacy admitted without an ACK
remains uncertain. The original execution fact cannot be overwritten for a
second dispatch. Healthchecks and daemon logging are disabled and checked;
actual entrypoint/argv/no-new-privileges and read-only shm+mqueue mounts are
verified. Stdio output and journal have independent strict byte bounds.

All tests, lint, type checking, compilation, builds, imports, provider/Docker
execution, source/registry fetch and signer access/enrollment remain unrun under
the user's explicit testing hold. Integrate the matching
owner helper/receipt fixes and current-key-lock migration before qualification.

The finite image closure additionally records `image_platform` as the canonical
SHA256 of OCI os/architecture. Runtime independently inspects the actual local
image and Docker daemon and requires native Linux amd64/arm64 agreement; implicit
QEMU/emulation is unsupported because that dependency is outside retained image
layers. No image inspection or Docker command was run during implementation.

### Source completion pass (2026-11-03)

The recorder now pins the actual Docker daemon ID before original create and
the immutable container ID after its ACK. Signed, fsynced lifecycle originals
record owner binding, create, start and confirmed cleanup independently of DB
ACKs. Recovery requires the same daemon and original signed lifecycle; an
unacknowledged create with no concrete terminal/cleanup proof stays in doubt.
An unavailable or changed daemon cannot prove absence. Cleanup targets the
observed immutable ID and original label, including after container rename.

The original deadline now includes preflight and owner acknowledgments. Start
checks deadline/cancellation again after provider validation. Local attach
clients/pipes are reaped separately; their exit never proves provider cleanup.
No project import, test, compiler, lint, Docker or provider command was run.
The matching backend requires this source revision's on_owner and
deadline_timestamp inputs, and passes the retained daemon ID during recovery.

### Review follow-up

The recorder rechecks cancellation and the original deadline after the durable
`create_pending` acknowledgment. Expiry, cancellation or acknowledgment failure
before provider submission retains a signed, fsynced `create_not_dispatched`
original. Recovery distinguishes this fact from an ambiguous Docker create and
refuses any conflicting allocation/lifecycle. The source regression suite passed
17 checks with real Docker cases deselected; this does not qualify an installed
container runtime. This source changes the recorder module digest and therefore
requires a freshly paired wheel and explicit cohort pin before installation.
