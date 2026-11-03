

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
tested, built or executed. Lock resolution, package/wheel generation, operator
installation and all positive/negative crash/runtime/custody checks are deferred
to integration/testing under the user's explicit hold.

Finite admission requires enough original reservation for source/image bytes,
twice the declared preexecution archive bound, up to 32 MiB terminal publication
bytes and at least 16 Artifact objects. A separate concrete Mnemosyne reservation
with only required trace kind `trace` and at least 8 MiB is protected before
dispatch. Actual consumption can still refuse; it never overrides quota.
Original signer unavailability is a typed predispatch refusal. The finite
container never borrows the shared task deployment handle during cleanup.
