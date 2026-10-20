# Portable native execution

The native execution cohort provides bounded local coordination and provider
adapters. Rhodes remains the hosted execution owner. Importing this package
does not launch a provider, create a hosted scheduler, or qualify benchmark
scores.

Install the explicit `native-harbor`, `native-harbor-docker`, or
`native-harbor-daytona` extra on Python 3.12 or newer. The adapters require
Harbor 0.22.0, Docker SDK 7.1.0, and Daytona 0.210.0 as applicable. These seams
are intentionally pinned; a different SDK is a refusal, not assumed compatible.

The local CLI exposes `journal`, `harbor-image-context`, `harbor-stage`,
`harbor-docker-reconcile`, and `harbor-daytona-reconcile`. Journal replay and
image-context freezing need no provider credentials. Reconciliation requires
retained resource identity and provider ownership; it does not replay agent work.
The existing catalog and tunnel commands retain their dispatch behavior.

`LifecycleLimits` and `DurableRolloutSupervisor` retain a run's absolute deadline,
exclusive owner, phase ceilings, stop decisions, and cleanup receipts. Recovery
cannot renew allowances or repeat completed work. Cancellation of a Python task
does not prove a remote resource stopped. Confirmed cleanup requires the adapter's
independent absence receipt. Host deadlines alone do not survive supervisor loss;
provider TTL and fencing remain separate requirements.

`LimitCapability` describes the actual mechanism and whether it survives
supervisor loss. A sampled threshold, native allocation control, and reserved
allowance are distinct guarantees. Unsupported requirements fail admission.

`collect_bounded_artifacts` retains partial bytes after failure without committing
a result manifest. `freeze_native_trace_evidence` preserves sealed Trace V5
identities and required visual attachments. Interrupted capture requires explicit
opt-in and remains incomplete. Publication receipts, cleanup confirmation,
execution status, and benchmark score remain separate facts.

The archived hosted native-journal uploader is deliberately excluded: its
`PoolClient.publish_native_events` protocol is absent from current `dev`.
Integrate that protocol with its hosted owner before exposing the uploader.
The public trace-store adapter is available through `NativeTraceStore`; mocked
and local custody checks do not qualify a deployed trace-store service.

This is a selected port from archive commit
`c7ec3f7bf88f7a389840142637a9da760a7219b1`, not a merge of its historical stack.
The OCI executor and its associated tests retain Apache-2.0 provenance in
`NOTICE` and `LICENSES/Apache-2.0.txt`.
