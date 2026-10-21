# Portable checkpoint metadata and preflight v1

Containers adds `synth.portable-checkpoint.v1` around the existing immutable
`synth.application-checkpoint.v1` joint cut. The original capture/restore
coordinators, participant receipts and tunnel generation fences remain the
execution substrate. Benchmark adapters still own codecs and the meaning of
their captured state. This profile provides complete local byte export,
provider-free inspection, explicit restore requirements and strict-position
selection. It does not implement hosted admission or a new recovery scheduler.

## Manifest

`PortableCheckpointManifestV1` pins:

- Original application checkpoint reference and checkpoint UUID; source run,
  lineage, committed event sequence and chain head.
- Source, image, dependency, task, world, scorer and protocol digests. The image
  identity is an immutable digest supplied by the adapter, not a mutable tag.
- Two content-addressed configuration artifacts: `configuration` contains all
  effective non-policy settings, including cadence, original limits and clock
  semantics; `policies` contains every participating policy's effective prompt,
  code/model/effort/temperature and other policy settings. Use credential
  references, never credential values. Adapters must define deterministic
  serialization and distinguish omitted values from explicit defaults.
- Every joint-cut participant's exact codec digest and named component coverage.
  Environment/RNG, policy history/memory/RNG, queued messages and pending plans
  belong to their benchmark's codec. An empty queue is a captured component;
  an uncaptured queue is missing coverage.
- Named action/observation/reward/event or benchmark-specific cursor positions
  and chain heads. The required `event` cursor must match the manifest's primary
  sequence/head. This consistency check alone never proves a coherent joint cut.
- Cumulative resource liabilities, clock policy, terminal status, unresolved
  effects and advertised modes. Resource/provider/unit tuples are unique;
  consumed/outstanding amounts use exact nonnegative decimal strings with up to
  18 integral and 12 fractional digits. `null` means unknown. Missing resources
  do not imply zero. Accounting must include active time and the adapter's
  relevant calls/tokens/spend; authoritative current liabilities can be newer.

The application checkpoint must have the same checkpoint identity, contain
exactly the declared participant set and satisfy its existing joint-cut
invariants. Configuration and participant blobs must be fully accessible in the
same artifact namespace. The v1 local export embeds this entire closure; it
does not dereference arbitrary URLs, private dependency registries or external
blob locators. Image/dependency availability remains a target-runtime check.

## Byte custody and local export

SDK imports are available from `synth_containers.checkpoints`:

```python
inspection = export_checkpoint_bundle(manifest, artifact_reader, output_directory)
inspection = load_checkpoint_bundle(output_directory)
report = check_checkpoint_restore(manifest, artifact_reader, request, runtime_support)
selected = select_strict_checkpoint(candidate_reports)
```

`ArtifactReader` requires only `get(reference)`; existing file/S3 stores also
implement it. `CheckpointArtifactReader` reads an existing local CAS without
creating anything. `CheckpointBundleStore` reads an exported bundle. Both
bounded local readers refuse symlinks below their configured root, non-regular
files, absent bytes, namespace mismatch and wrong size/digest. The caller owns
the configured root and its ancestors. Construction and reads have no provider
or network side effects and never decode participant state.

Each artifact is limited to 64 MiB, the complete unique closure to 256 MiB, and
JSON metadata to 8 MiB. JSON duplicate keys and unknown schema fields refuse.
Inspect/export verify every referenced digest before claiming closure, without
using absolute source paths from the manifest. Input changes during export
refuse on the subsequent verified read.

Export creates a new private directory and never overwrites an existing target.
Its layout is:

```text
bundle/
  manifest.json
  blobs/sha256/<hex-digest>
```

Blobs are published/fsynced first through the existing immutable file store.
A complete fsynced temporary manifest is linked into `manifest.json` last;
the bundle and parent directories are fsynced before success. Files have mode
0600 inside a mode-0700 bundle root. A failed export retains its owned partial
output. Without `manifest.json`, it is not discoverable as a complete bundle.
An I/O failure after the marker link has an uncertain publication outcome:
reopen/inspect it; never infer authority from either the error or a directory's
presence. Export is a portable copy, not an owning journal's checkpoint commit.

## Read-only compatibility and explicit modes

`CheckpointRestoreRequestV1` binds source/target identities, target execution,
mode and the owning journal's supplied `CheckpointCommitAnchorV1` (manifest
digest plus committed sequence/head). The anchor must come from the caller's
authoritative committed-journal read. An unsigned bundle, request or report
cannot authenticate its own publication, grant access or authorize execution.

`CheckpointRuntimeSupportV1` is an owner-selected target-runtime policy: exact
participant/codec set, required component and cursor coverage, supported modes,
trusted decoder digests and clock policy. Never accept this policy from an
untrusted workload as a grant of decoder trust. Codec byte integrity and decoder
safety are separate checks. Named coverage is an adapter declaration; only
native codec/trajectory qualification proves its semantic completeness.

| Mode | Compatibility behavior |
| --- | --- |
| `inspect` | Verify byte closure, source/commit binding and declared support; decode no state. Terminal evidence may be inspected without decoder trust. |
| `replay` | Also require exact participant/codec support, coverage and decoder trust; terminal evidence can be replayed. |
| `strict_continue` | Require nonterminal cut, no unresolved effects, exact full execution/configuration/policy identity and compatible clock. Keep original run/lineage; no child funding artifact is accepted. |
| `same_policy_fork` | Same execution/configuration/policy checks, plus a distinct child run and lineage and an explicit child budget reference. |
| `edited_policy_fork` | May change only the policies artifact; all other pinned execution/configuration identities stay equal. Require distinct child run/lineage and explicit child budget. |

Changing a world/scorer/protocol/source image or checkpoint cadence requires a
separately qualified migration/profile; it cannot silently pass this v1 fork
check. A world-only cut fails a runtime policy requiring policy participants.
Terminal snapshots and unresolved in-flight effects cannot pass restore modes.

`declaration_supported=true` means these local declaration/byte checks passed.
Every report explicitly retains `admission=not_checked`,
`suffix_reconciliation=required`, `liability_reconciliation=required`,
`target_inputs=not_checked` and `decoder_execution=not_run`. It preserves the
checkpoint's exact liabilities, including unknown reservations.

The owning runtime must still authenticate the commit, validate target assets,
decode/check native state safely, obtain current admission and live fences,
reconcile all committed suffix effects/cursors, carry forward subsequent
charges/holds and current remaining allowance, and apply the original deadline
or active-runtime semantics. Downtime never renews a lease, absolute deadline or
budget. Ambiguous provider calls remain explicit; exactly-once billing is not
promised. An edited fork needs separately admitted child funding and native
child-state isolation. The CLI currently performs no restore or provider work.

## Strict selection versus search

`select_strict_checkpoint` accepts local preflight reports for one identical
strict-continuation request/runtime policy, apart from each candidate's supplied
commit anchor. It chooses the highest compatible committed sequence, ignores
blocked candidates and refuses conflicting manifests at that sequence. It
refuses mixed run/lineage/configuration/runtime scopes and fork/replay reports.
No compatible candidate returns `None`, never a fresh-run recommendation.

These reports are local values, not signatures or execution grants. Selection
does not discover an owning run's terminal state or reconcile a durable suffix;
those checks still precede resumed dispatch.

The existing `derive_run_recovery_projection` retains its reward/frontier search
ordering for compatibility. Its metadata now explicitly says
`selection_policy=reward_frontier_search`. Its historical `supports_true_resume`
flag describes the legacy capability heuristic; it is not strict-continuation
qualification. Optimizer callers must choose the intended mode explicitly when
adopting this new interface.

## CLI

```sh
synth-containers checkpoint schema manifest
synth-containers checkpoint export checkpoint.json --store ./source-cas --output ./bundle
synth-containers checkpoint inspect ./bundle
synth-containers checkpoint check ./bundle --request restore.json --runtime-support runtime.json
```

All commands emit JSON. `check` exits 0 for matching declarations, 2 for
unsupported/invalid input. `inspect` refuses incomplete/corrupt/inaccessible
closure. Error output uses stable codes without payloads or filesystem details.
No `resume` command is advertised until the owning runtime/adapters qualify the
execution and accounting gates above.

## Qualification and remaining adoption

`tests/test_portable_checkpoint.py` exercises portable export after source store
deletion, malformed/corrupt/missing/private evidence, failed publication,
terminal/unsupported/untrusted cuts, coverage and clock mismatches, strict
identity preservation, explicit forks and position selection. A fresh process
uses the existing joint restore coordinator with three scripted participants
(world plus two policies); seven continuation actions/rewards/final states match
the uninterrupted suffix, including RNG, pending plans, histories and queued
messages. Repeating the same completed restore reads its receipt without
another participant call.

This is deterministic local conformance. It is not Craftax/GELO/MAPO adapter
adoption or hosted Rhodes qualification. Kill-at-every-domain-boundary tests,
native encoder/decoder safety, committed suffix and post-checkpoint liability
reconciliation, retention roots, hosted export/import custody and actual
resume/fork commands remain open under SYN-4922/SYN-4919 and benchmark owners.
