# Rhodes API semantic versioning

Status: implemented and qualified in source on 2026-11-13 in the paired backend
`lane/rhodes-api-semver-20261113` and Containers
`codex/rhodes-deepswe-luna6-e2e-20261113` worktrees. This does not describe an
installed deployment or a working hosted Harbor adapter.

## Observed incompatibility

The legacy `api-dev.usesynth.ai` accepted `/v1/pools` writes while its Rhodes
worker had no active deployment. The extracted hosted `rhodes-api` has a healthy
API/worker but intentionally refuses those same public writes with
`403 rhodes_customer_route_refused`, stage `public_origin_guard`, cause
`route_not_served_on_hosted_origin`. Its supported compute boundary is Core-owned
execution. A healthy HTTP socket, an accepted read, or a shared `/v1` prefix does
not establish the standalone launcher's required contract.

Removing formerly supported hosted customer mutations is a breaking public
contract change, even when the URLs and read schemas remain present. Treat it as
a major version change. A version number does not confer an identity, grant or
budget, and must not be used to bypass the hosted origin guard.

## Implemented contract

The backend publishes a small credential-free `GET /.well-known/synth-api` response, explicitly
allowed by the public origin guard. Hosted discovery fields (build identity varies):

```json
{
  "schema": "synth.api-discovery.v1",
  "service": "rhodes-api",
  "api": {"name": "synth.rhodes", "version": "2.0.0", "supported_versions": ["2.0.0"]},
  "authority": "synth-backend.core",
  "execution_profile": "hosted-core-owned",
  "capabilities": {
    "customer.pool.create": false,
    "customer.rollout.create": false,
    "core-owned.execution": true,
    "harbor.isolated-verifier": true
  },
  "build": {"revision": "<source SHA>", "image_digest": "sha256:<digest>"}
}
```

- **Major:** remove/change an operation, required field, response meaning,
  credential authority, execution ownership, verifier isolation, or idempotency
  guarantee incompatibly.
- **Minor:** add backward-compatible operations, optional fields or capabilities.
- **Patch:** compatible fixes that preserve the promised semantics.

Keep API semver independent of package version, git SHA, image digest, database
migration number and worker protocol version. Pin the actual build separately.
Internal Core→Rhodes owner protocols also need explicit versions, rather than
relying on a public API version to imply cross-service compatibility.

An SDK declares a supported major/minor range and required capabilities, e.g.
`>=2.0.0,<3.0.0` plus its required operations. A standalone direct-pool client
requires `customer.pool.create` and `customer.rollout.create`; it must refuse
this example before uploading or requesting a hold. It must not automatically
fall back to the legacy service or change provider/harness.

Request negotiation accepts `Synth-API-Major: 2` or the versioned media type
`Accept: application/vnd.synth.rhodes.v2+json`. The server returns the selected API semver in
`Synth-API-Version` response metadata. An unsupported requested major returns a typed `406
api_version_unsupported`, with the supported versions. Do not silently ignore a
version request or disguise an unsupported contract as an authentication failure.

## Readiness and timing remain separate

Worker progress readiness (last progress time and stale threshold) remains future
work. A compatible API whose worker is absent is not launch-ready. Queue,
provider startup, agent, verifier, publication, accounting and cleanup need their
own observations; compatibility preserves these unknowns.

The launcher first requests credential-free discovery, accepts stable 2.x versions
(including build metadata), requires the direct pool/rollout creation capabilities,
then checks service/build identity and control-plane readiness. The SDK refuses
missing discovery, redirects, malformed or oversized documents, contradictory
version headers, incompatible/prerelease versions and missing/false capabilities.
Discovery is bounded at 64 KiB and ten seconds, with no credential headers or cookies.
It pins future requests to major 2, preserving each operation's Accept header.
No legacy fallback is attempted. Queue wait remains independently bounded at
120 seconds; discovery explicitly leaves worker readiness `not_observed`.

## Delivery and acceptance

Server negotiation and consumer validation are implemented together. Source tests
cover positive minor/patch versions, unsupported and malformed selectors, false
capabilities, hosted guard preservation, legacy discovery refusal and refusal
before mutations/provider inventory. A cross-worktree ASGI smoke connects the
actual SDK to the composed Rhodes app without credentials or paid execution.

A Core-owned Harbor adapter and installed-build qualification remain required
for the original hosted end-to-end launch. Do not reopen hosted pool writes.
Worker progress, idempotent execution, model result and provider teardown still
need their own retained live evidence; source API compatibility cannot prove them.

Evidence: `/Users/joshuapurtell/GitHub/artifacts/rhodes-api-semver-20261113/RECEIPT.md`.
