# Container execution requirements v1

`synth_containers.execution_requirements` owns this portable contract. It does
not change the historical `synth.container-package.v1` manifest or its digests.
Reliability tier: 1, because its refusals precede provider effects.

`ExecutionRequirements` requires every field:

- `schema_version`: `synth.container-execution-requirements.v1`.
- `platform`: `linux/amd64` or `linux/arm64`.
- `resources`: integer `cpu_millis`, `memory_bytes`, `disk_bytes`, `gpu_kind`
  (`none`, `t4`, `l4`, `h100`) and integer `gpu_count`.
- `max_wall_seconds`.
- `network`: `mode` (`none` or `allowlist`) and sorted, distinct lowercase
  hostname `hosts`. No-network requires an empty tuple. An allow-list requires
  explicit hosts. Wildcards, normalization and unrestricted networking are absent.
- `privileged`: a boolean declaration, never an implicit provider permission.
- `verifier_isolation`: `self_reported`, `shared` or `isolated`. A Harbor translator
  maps an accepted source `separate` declaration to `isolated`; the canonical
  contract rejects `separate`. It never translates isolation to shared execution.

The digest uses existing `package_models.content_digest(model_dump(mode="json"))`:
SHA256 of sorted compact JSON, UTF-8, `allow_nan=False`, no trailing newline.
The existing helper uses `ensure_ascii=True`. Bounds live in `execution_limits.py`.

`preflight_execution` composes three inputs:

1. Requirements and their retained expected digest.
2. An `ExecutionArtifact` with exact OCI identity, actual inspected platform,
   inspection state and inspection-receipt digest. An uninspected artifact stays
   blocked; an arm64 artifact cannot satisfy an amd64 declaration.
3. `ScoringOfferCapabilities` from the authenticated placement owner. It binds
   provider, exact `option_id` and `offer_id`, catalog/descriptor/tariff digests,
   adapter revision, qualification, environment release admission, enforced
   platforms/resources/wall time/network modes, environment host-policy digest
   and allowed hosts, privilege support and verifier modes.

Packages and caller input cannot author offer qualification, release admission,
environment host permissions, adapter capabilities or artifact inspection facts.
The owner must obtain these facts from trusted current sources. Every requested
host must be authorized and enforced; a provider's general ability to network
does not establish an allow-list. The checker performs no provider I/O.

The result is `supported`, `unsupported`, `unqualified` or `blocked`, with a
stable code, stage, cause, original requirements identities, exact offer and
adapter identities, host-policy digest and image inspection identity. Missing
qualification is not a pass. The owner emits one structured refusal log with
its run/org/operation identity, retains the receipt before dispatch and rechecks
it against the original request on redelivery.

This receipt proves capability compatibility only. A financial hold, lifecycle
ownership, create receipt, supplier meter, terminal outcome and independent
cleanup remain separate obligations. No offer is promoted by constructing a
model. No provider, image architecture, GPU, verifier mode or network permission
is substituted. Build/runtime integrations must preserve the receipt identities
through dispatch and settlement.

Source qualification: `tests/test_execution_requirements.py` exercises typed
refusals after qualified fixture admission, including off-list hosts and verifier
isolation. Fixture authority does not qualify a live provider. Actual adapter,
environment and image qualification remain required before installed acceptance.
