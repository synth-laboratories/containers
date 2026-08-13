# Trace V5 storage and compatibility policy

**Status:** Proposed policy for the public `synth-containers` Trace V5 surface.

## Compatibility promise

Trace V5 is an immutable evidence format. A package upgrade must not change the
meaning or digest of a previously sealed `synth.trace.v5` document.

The following are frozen for V5:

- `synth.canonical-json.v1`
- `sha256:<hex>` semantic digest syntax
- exclusion of a record's own top-level `content_digest` while sealing
- null removal, preservation of empty arrays/objects, sorted keys, compact UTF-8
  JSON, and rejection of non-finite numbers
- sealed trace identity as the subject of evidence, selectors, annotations, and
  visual projections

A change to those rules requires a new canonical profile and trace schema, not a
reinterpretation of `synth.trace.v5`.

## Supported portable inputs

New readers should preserve support for:

1. Current `synth.trace-bundle-manifest-pointer.v1` bundles whose immutable
   `synth.trace-bundle.v1` manifest contains an `objects` inventory.
2. Push 1 inline `synth.trace-bundle.v1` manifests without `objects`, using the
   existing backward-compatible verification path.
3. Standalone sealed `synth.trace.v5` documents, wrapped without modifying the
   sealed document.
4. Released legacy formats with explicit versioned adapters.
5. Unknown future schemas as opaque bytes when their transport integrity can be
   established; readers must not guess at their semantics.

Manifest additions must be optional. Old readers may ignore unknown members,
but importers must preserve their original bytes. New V5 trace semantics belong
under the existing `extensions` field. Breaking trace semantics use a new trace
schema version.

## Three different digests

Do not collapse these identities:

| Digest | Definition |
| --- | --- |
| Trace digest | Semantic digest of canonical sealed `synth.trace.v5` |
| Bundle digest | Semantic digest of one immutable bundle generation manifest |
| Archive digest | Byte digest of the deterministic portable ZIP |

One trace can occur in multiple bundles. A bundle can contain multiple traces.
Transport/archive identity is not trace identity.

## Migration policy

Migration is append-only and source-preserving:

```text
source bytes
  -> source byte digest
  -> versioned adapter
  -> new sealed V5 trace
  -> migration receipt
```

The original bytes remain an immutable artifact. The converted trace receives a
new V5 semantic digest and records:

- `provenance.source_format`
- adapter producer and version
- `provenance.transformation_chain`
- native identities through aliases
- losses, omissions, and uncertainty through completeness/extensions
- a receipt containing source digest, adapter version, output digest, warnings,
  and losses

Adapters must not reuse a legacy digest as V5 identity or rewrite legacy files
in place. With identical input bytes and adapter version they must produce the
same sealed trace digest. Operational import time belongs in the consumer's
catalog; it must not introduce nondeterminism into the converted trace.

If a wrapper bundle records packaging time, repeated packaging may produce a
new bundle digest while still containing the same sealed trace digest. Consumers
must deduplicate the trace independently from bundle membership.

## Compatibility result

The stable inspection API should classify each input as:

- `native`: verified and projectable current V5
- `legacy_native`: verified older supported bundle/trace, read without rewrite
- `migrated`: source preserved and a derived V5 trace produced
- `opaque`: bytes retained but semantic schema unsupported
- `partial`: declared objects unavailable; retain as quarantined input, not a
  trusted self-contained bundle
- `invalid`: integrity, schema, path, or archive-safety failure

Invalid input is never published into trusted content-addressed storage.

## Conformance requirements

Permanent fixtures should cover:

- Push 1 inline bundle manifest
- current pointer/object-inventory bundle
- standalone V5 trace
- Trace V4 migration
- Harbor ATIF/native trajectory migration
- legacy rollout-step projection
- missing media/partial bundle
- byte corruption and semantic-digest corruption
- path traversal, symlink, collision, and archive expansion attacks
- unknown future schema

Every release validates all historical fixtures and asserts that previously
sealed trace, bundle, object, and deterministic archive digests remain unchanged.
Fixture bytes and expected digests are release artifacts and must not be updated
merely to make a new implementation pass.

## Shared consumer boundary

Consumers should call a stable inspection/projection API instead of importing
bundle internals. The first common viewer projection should be independently
versioned as:

```text
synth.trace-projection.rollout-inspector.v1
```

Every projected entry carries the sealed trace digest and a stable
`synth.trace-selector.v1`. Projections are derived and replaceable; they never
become evidence authority.

The implemented Python boundary is:

```python
from synth_containers.tracing import inspect_trace_input

inspection = inspect_trace_input(source, archive_output=optional_zip_path)
```

The equivalent process boundary is `synth-trace inspect-input SOURCE
[--archive-output PATH]`. Both emit `synth.trace-inspection.v1`, including the
compatibility classification, validation issues, qualified bundle/archive/trace
digests, and trace, asset, and projection inventories. An archive output is only
written after the source is verified as trusted and self-contained.

The first common viewer packet is built with
`rollout_inspector_from_sealed(trace, evidence)` and carries schema version
`synth.trace-projection.rollout-inspector.v1`. It composes the existing sealed
visual lanes/items, whose entries already retain stable Trace V5 selectors.
