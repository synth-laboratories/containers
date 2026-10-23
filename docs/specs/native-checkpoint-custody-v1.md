# Process-owned native checkpoint custody

`CompatPlatform.record_checkpoint` acknowledges native JSON cuts under the
existing `synth.containers.checkpoint.v1` schema. This is process-owned retained
state, distinct from portable-checkpoint-v1, Artifact Platform custody, runtime
snapshot management, restore admission and child funding.

The producer supplies a nonempty exact string checkpoint ID, at most1024 UTF-8
bytes, and native facts. Leading/trailing whitespace and producer-supplied
schema_version, target_id or content_digest refuse before writes. The platform
authors those authority fields. Facts are frozen through the existing JSON
codec once; tuples become arrays and the existing default=str conversion is
retained. Nonfinite values refuse. This does not qualify arbitrary object
serialization or cross-version continuation.

The acknowledged value, cached value and restart value have the same JSON
shape. Caller input and returned acknowledgments share no mutable nested
objects with the retained cache. Runtime code with direct access to the process
is not a security isolation boundary.

Capture, retention-root operations and deletion serialize under the process
state lock. A conflicting concurrent capture cannot replace the original cut.
Atomic write uses the existing file/fsync/replace/directory-fsync barrier. If
the write committed before its acknowledgment failed, replay reads and verifies
the original file, confirms its directory barrier and acknowledges without
rewriting. Changed facts, corrupt bytes, denied reads or a different target's
file never become permission to overwrite. Only FileNotFoundError means absent.
Existing v1 sealed bytes/digests are not migrated or rewritten.

Deletion verifies the current target/file and retention roots, unlinks and
fsyncs the directory before forgetting memory. A failed unlink or durability
barrier keeps the original cached identity. A retry can finish the barrier
after the file is already absent. Retention requires matching readable durable
bytes, so a failed deletion cannot produce a new root over a missing file.

One active CompatPlatform owns its storage root. These locks do not implement
cross-process writers, permanent deleted-ID tombstones, portable checkpoints,
remote checkpoint authorization or provider snapshot cleanup. Those need the
explicit service-owned custody/retention and restore contracts. A local native
checkpoint acknowledgment must not be promoted into any of those claims.
