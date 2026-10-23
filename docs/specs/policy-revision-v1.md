# Policy revision protocol v1

Status: review candidate, 2026-10-23. Producer: Containers. The initial reference
implementation supplies process-local next-rollout configuration, opt-in prompt
programs, and single-file Python observation/action policies. It does not supply
live activation, atomic rosters, weight loading, Rust compilation or recovery.
Implementations advertise only their registered adapters and profiles.

## Resource and operations

Use one immutable `synth.policy-revision.v1` manifest. A candidate is optimizer
lineage pointing to a policy revision, not a second runtime resource. The existing
portable `synth.candidate.v1` registration binds scientific candidate/task/input
evidence; it is not a second policy installation API. Retain its identities and
refer to the policy revision from its content instead of changing historical
candidate bytes. Do not replace the existing rollout, checkpoint or artifact
authorities.

| Operation | HTTP | Effect |
| --- | --- | --- |
| Publish | `POST /policy-revisions` | Validate and register immutable content; return revision ID. No selection or execution. |
| Publish with expected ID | `PUT /policy-revisions/{id}` | Same operation, refusing a body whose derived ID differs with 409. |
| Prepare | `POST /policy-revisions/{id}/prepare` | Validate the adapter and materialize its declared readiness; return an operation receipt. |
| Bind next rollout | Existing `POST /rollouts` (legacy alias `/rollout`) | Admit `policy_revision_id`, resolve prepared content, and freeze the binding for that rollout. |
| Inspect | `GET /policy-revisions/{id}` and existing rollout readback | Return publication/preparation state and the admitted actual binding. No source bytes or credentials. |

There is no redundant `bind` network call for next-rollout execution. Admission
of the rollout is the binding operation. Live activation will require a separate
versioned operation, expected prior epoch, state plan and operation identity.
An ordinary PUT or process restart must never be interpreted as live activation.

Example manifest:

```json
{
  "schema_version": "synth.policy-revision.v1",
  "namespace": "optimizers",
  "name": "prompt",
  "harness": "prompt_program",
  "components": {
    "prompt_fields": {"stage2_system": "Classify this request."},
    "config": {"policy": {"provider": "openrouter", "model": "openai/gpt-oss-20b"}}
  },
  "source": {"revision": "git:<source-sha>"}
}
```

The exported JSON Schema is `openapi/policy-revision-v1.json`; the canonical HTTP
mapping is also included in `openapi/container-contract-v1.yaml`. Unknown request
fields and wrong types refuse. Optional future components require a new schema
or explicit negotiated extension. Unknown response fields may be ignored; unknown
required profiles/schema versions must refuse before effects.

## Components and identity

`prompt_fields` is a map of declared mutable field IDs to strings.
`module_demos` maps module IDs to ordered training-example IDs. The task adapter
must validate pool membership, split role, duplicates and heldout exclusion;
generic prompt adapters do not advertise this component. Demo selection is not
an untyped JSON string inside `prompt_fields` in the new protocol.

`code_bundle` specifies language, canonical relative file paths, entrypoint,
serve-contract version and optional dependency digest. The current executable
adapter accepts one Python file, no external dependencies, and
`process_observation_action.v1`: load a callable, acknowledge readiness over
JSONL, answer observation/action requests, and close. Startup has a bounded
deadline and failed/hanging children are reaped. This is a process boundary,
not a general hostile-code sandbox. Rust and multi-file/dependency builds require
an admitted isolated build adapter, pinned build inputs, mbx and build receipts;
they currently refuse rather than invoking an arbitrary host command.

`config` is immutable adapter-specific data. `model` is identity only.
`weights_ref`, `renderer_ref`, `sampler_ref` and `protocol` use digest-pinned
artifact references. A URL, trainer save acknowledgement or config entry does
not prove that weights are loaded. Unsupported components refuse publication.
Manifest references do not transfer artifact custody or grant fetch authority.
Raw credential fields and signed/userinfo-bearing artifact URLs refuse.

The producer normalizes the validated manifest, excluding absent optional fields
and including schema defaults, then applies the existing Containers
`package_models.content_digest` JSON serialization (sorted keys, compact JSON,
ASCII escapes, finite numbers only). IDs are `polrev_` plus the **full** SHA-256
hex digest. Clients use the producer-returned ID; they must not implement a
different JSON number canonicalizer. Shared fixtures freeze exact bytes for
Python/Rust decoders. Existing `polrev_<16hex>` IDs retain their legacy meaning.

`namespace`/`name` are descriptive identity, not mutable aliases. Identical
publication is idempotent. A changed manifest is a new revision. A mismatched
expected ID returns `policy_revision_conflict`. `parent_revision_id` expresses
policy lineage; changing a world checkpoint does not by itself mint a policy
revision. A fork records both checkpoint and selected policy IDs separately.

## Preparation and binding receipts

Prepare takes `operation_id`, `profile` and `timeout_s` (1–30 seconds here).
States are `published`, `preparing`, `ready`, `failed`. Repeating an operation ID
with identical arguments returns its retained result; changing its revision or
arguments returns 409. Retrying a failed preparation requires a new operation ID.
Failure does not select a default policy or retire an existing actor. Configuration
readiness is labelled `validated_configuration`; executable readiness identifies
the actual probe process/serve contract. Neither is labelled loaded model weights.

Preparation receipts include revision/digest, state, durability, operation/profile,
actual binding or a stable failure code. Readiness and active execution remain
separate. On next-rollout admission the resolver freezes the exact configuration
and source. A newer publication/preparation cannot alter an in-flight rollout.
Rollout responses/readback retain `policy_revision_id`; a caller must check it.
Supplying a revision and conflicting inline candidate, code or configuration
returns a typed refusal before rollout effects. Legacy v2 callers may continue
their existing request shape during the explicitly versioned compatibility
window; migrated clients publish and pin. Negotiated-path failure is not a
reason to fall back to the old inline shape.

The reference registry is **process_local**. A new process must republish and
prepare retained manifests; clients cannot assume an old preparation exists.
Republishing gives the same ID, not a recovered active actor. Durable manifest
bytes belong in existing Artifact Platform custody; execution/activation intents
and receipts belong to their actual runtime owner. A recoverable profile needs
fresh-process reconciliation qualification before advertisement.

## Profiles

| Profile | Required semantics | Initial implementation |
| --- | --- | --- |
| `next_rollout_update.v1` | Immutable resolution, prepared admission, fixed in-flight bindings, actual rollout readback. | Implemented for registered config/prompt/Python adapters. |
| `policy_set_update.v1` | A revision containing a complete seat→revision roster, all dependencies ready, one atomic visibility point. | Proposed; absent from capabilities. Reuse CISPO PolicySetRevision semantics; no partially published roster. |
| `checkpoint_fork_with_policy.v1` | Explicit parent checkpoint and target policy/set, compatible participant closure, new child identity, preserved budgets/cursors. | Proposed composition with portable checkpoint v1; absent from capabilities. Never substitute fresh execution. |
| `live_actor_update.v1` | Prepare, quiesce, explicit reset/transfer/migration, ready replacement, expected prior binding/epoch CAS, effect fencing, retire old. | Proposed; absent from capabilities. OHCO actor/world supervisor is the reference, not beta context-search OHCO. |
| `recoverable_*` | Durable intents/outcomes, response-loss reconciliation and crashes before/after commit. | Deferred; absent from capabilities. |

Rollback is a new activation of old artifact bytes with a newer epoch and a
compatible state plan. It does not rewind world actions, spend or action receipts.
After a committed live cut, old-process cleanup is an independent outcome.
Before commit the prior binding remains authoritative. Unknown commit outcomes
must reconcile by operation ID, not issue a replacement operation implicitly.

Capability metadata is `policy_revision`: schema, supported profiles, process
durability, routes and adapter→component sets. It derives from installed adapters.
Runtime declaration and adapter qualification remain distinct: a boolean alone
cannot certify a live swap or weight load. Generic reference runtimes explicitly
opt into the prompt resolver only after supporting resolved candidates.

Authorization and current capacity/funding are revalidated by the existing
runtime/admission owner. Capability discovery does not grant publication,
execution, network, artifact or training authority. Local endpoints inherit their
hosting app's authorization boundary; this module creates no new credential path.

## Refusals and qualification

422: malformed/unknown manifest fields, credential fields, illegal relative paths,
oversized bundle, inline candidate alongside a revision. 404: unknown revision.
409: mismatched expected ID, operation reuse with changed inputs, unsupported
profile/component, not-ready revision, conflicting binding. Preparation failures
are retained results with `state=failed`, including `policy_startup_failed`,
`policy_startup_timeout`, `policy_code_contract_unsupported`,
`policy_prompt_field_forbidden` or `policy_preparation_failed`. Callers inspect
state; HTTP success alone is not readiness. Existing authorization/busy refusals
retain their host protocol's status and authority.

The local corpus covers repeat/conflict, schema/credential/path rejection,
unsupported profile with no effects, unready refusal, failure/hanging startup,
per-rollout A/B/A isolation, readback, inline override rejection and legacy
decoder behavior. Rust transport tests cover publication order, cache reuse,
failed/substituted readiness and no inline policy on the negotiated path.
Installed-wheel and cross-language HTTP receipts identify their exact source,
artifact and adapter. No live/joint/recoverable/provider conformance is inferred
from these local cases.

## Downstream mapping

GEPA's shared ContainerClient publishes prompt fields/inference configuration
after metadata negotiation, prepares once per candidate/configuration and sends
only the ID in the actual rollout wire. The Banking77 producer resolves it before
constructing the system prompt and retains it in queued/terminal readback. MIPRO
uses the same transport plus a task-qualified `module_demos` adapter; its unmerged
primary source and demo-pool rules require a separate source selection.

GELO fresh native evaluation publishes config and pins; continuation must require
the combined fork profile before proposer/trainer/rollout effects. SFT's evaluator
publishes checkpoint configuration and checks rollout binding; this does not
qualify a weights loader. CISPO/MAPO need atomic roster and participant closure,
not a loop of single-policy prepares. Preserve MAPO's independent prompt/protocol
stage constraints and check them both optimizer-side and at authoritative prepare.
OHCO actor code uses the live profile; context-search OHCO can use next-rollout
variants independently. VZPO/optimization-data retain bundle/artifact lineage.

For REB 013, `miprov2.rs::optimize` is the optimizer submission, **not** the rollout
policy. It remains a separately built benchmark artifact. Publish each generated
instruction and module demo selection as a policy revision; score selection rows
by ID. Keep the current evaluator-owned proposer, scoped credentials, independent
budgets, deterministic five draws, hidden heldout boundary and audit semantics.
The current `/v1/cache_policy` toggles evaluator caching; it is not publication.
Changing task interfaces requires a task-version increment and three new cold
OpenRouter golds before readiness can advance. The current pending readiness must
not be changed on the strength of a protocol fixture.
