# Shared OCI trial execution

`synth_containers.oci_trial` owns the existing `eval.target.v1` file adapter:
read-only input and candidate mounts, writable output, native CPU/memory/PID
controls, observed time/output limits, explicit credential injection, and bounded
stop attempts. `TrialRunRequest` accepts structural limits, so containers imports
neither optimizers nor private evals. Public optimizers retains its recipe,
candidate, scoring and budget contracts and adapts its existing executor imports.

This is a Docker/Podman adapter. It does not claim Daytona file isolation, hosted
artifact custody, a hard disk quota, or recovery after the local supervisor dies.
Optional target JSONL remains observational; the worker's durable journal is a
separate contract. Preserve target evidence protocols when adding other providers.

The extracted implementation and associated checks retain Apache-2.0 licensing;
package distribution metadata and NOTICE identify the mixed-license source.
