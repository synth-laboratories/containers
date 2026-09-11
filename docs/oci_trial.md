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
separate contract. Once a target event file appears, malformed/nonfinite JSON,
non-object records, replacement, truncation, disappearance and read failures
fail observation explicitly. Reads cap each record at 1 MiB and page at 256
records. Partial writes wait while live; incomplete final records fail. Reader
and callback failures reach the execution owner, which stops running resources.
A final drain exceeding two seconds fails rather than claiming complete evidence. Preserve target evidence protocols when adding other providers.

The extracted implementation and associated checks retain Apache-2.0 licensing;
package distribution metadata and NOTICE identify the mixed-license source.

Direct callers must supply a SHA256-pinned image, a supported network mode and
finite positive resource controls. Validation precedes filesystem/provider work.
Mount coordinates are absolute and comma-free; symlink roots are refused.
The stderr sink refuses symlinks and special files rather than following them.
