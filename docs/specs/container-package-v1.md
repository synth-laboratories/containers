# Portable container packages and build output

This is the initial v0.24 package/build contract. It adds a common authoring and
input/output boundary alongside the existing image catalog. The catalog launcher
continues to own Docker builds and execution; package scripts can delegate to it
or to a maintained external builder. No provider, hosted registry, permission,
startup or algorithm-compatibility claim follows from validating a descriptor.

## Author a package

`container.toml` (or the equivalent strict JSON object) declares an executable
build/resolve script, package-relative inputs, target platform, startup/readiness,
optional versioned protocol declarations, dependencies and credential references.
Unknown fields and unknown schema versions refuse.

```toml
schema_version = "synth.container-package.v1"
package_id = "my-environment"

[build]
entrypoint = "build.sh"
inputs = ["Dockerfile", "src", "requirements.txt"]
platform = "linux/amd64"

[runtime]
profile = "command"
command = ["python", "-m", "my_environment"]
working_directory = "/app"

[runtime.readiness]
kind = "process"

[[contracts]]
name = "my-environment.eval"
version = "v1"

[[credentials]]
name = "MODEL_API_KEY"
reference = "secretref://org/inference"
```

Generic command images need no Synth HTTP server. HTTP readiness declares
`kind = "http"`, `path` and `port`; native readiness declares `kind = "adapter"`
and an adapter identity. These are declarations for later runtime checks.
`contracts` may be empty. A protocol declaration is not a negotiated or qualified
optimizer capability. Credentials are references for later authorized binding;
this local build command does not resolve them or forward provider credentials.

Dependencies declare `name`, `reference` and exact SHA-256 `digest`. Include all
consumed toolchain/base-image/bundle identities. Declared inputs and dependencies
are the builder's responsibility; this protocol does not establish hermeticity.

## Plan, build and inspect

```sh
synth-containers package schema manifest
synth-containers package plan ./container.toml
synth-containers package build ./container.toml --output ./build-receipt
```

Planning executes no package code. It hashes the manifest bytes, parsed contract,
build script, declared regular files, executable bits and directory names
(including empty directories). Directory inputs expand recursively. Source paths
are canonical package-relative names; parent traversal, symlinks, special files,
missing inputs, nonexecutable entrypoints and excessive input size/count refuse.
Input snapshots are bounded at 64 MiB/file, 256 MiB/package and 10,000 entries.
The build-plan digest includes platform, startup, credentials/dependency refs and
protocol declarations; it contains no machine-specific source root or timestamps.
It identifies declared input equality, not an automatically reusable build cache.

Building copies and hash-checks only acknowledged inputs into an ephemeral tree,
then executes the script's argument vector without a shell wrapper. Inputs which
changed after planning refuse before execution. The script receives:

| Variable | Meaning |
| --- | --- |
| `SYNTH_CONTAINER_BUILD_PLAN` | Frozen `synth.container-build-plan.v1` JSON path. |
| `SYNTH_CONTAINER_BUILD_OUTPUT` | Path at which to write the build-output JSON. |
| `SYNTH_CONTAINER_PACKAGE_ROOT` | Copied input root and working directory. |

Build scripts are trusted local code, **not sandboxed**. They retain ordinary
host authority; frozen inputs and the small environment allowlist are not security
isolation. Use managed mbx/admission/storage launchers for Rust. Native builds
and provider effects require their existing limits and authorization. The timeout
and merged log bound do not impose a filesystem quota on files a builder writes.
Uploaded/untrusted builds on Rhodes must execute as admitted isolated build
workloads, rather than invoking this local helper in the API/control-plane
process. The pure plan/descriptor contracts can be shared across those adapters.

## Build output and immutable artifact descriptor

The script exits zero and writes an object such as:

```json
{
  "schema_version": "synth.container-build-output.v1",
  "action": "resolve",
  "input_digest": "sha256:<the plan's 64 lowercase hex digits>",
  "artifact": {
    "kind": "oci_image",
    "digest_kind": "oci_manifest",
    "reference": "registry.example/environment@sha256:<64 lowercase hex digits>",
    "digest": "sha256:<the same 64 lowercase hex digits>",
    "platform": "linux/amd64"
  }
}
```

The abbreviated digests above are placeholders, not runnable image identities.
Use `action = "build"` for a newly built image and `"resolve"` for a prebuilt
artifact. Registry manifest/index identity requires an exact matching pinned
reference. A Docker daemon config identity uses `digest_kind = "oci_config"`
and its bare `sha256:…` reference. A local config digest is not a registry
manifest or an uploadable registry reference.

`synth.container-artifact.v1` binds this output to the complete plan, rejects a
different input digest/platform and explicitly records
`artifact_verification = "not_checked"` and `runtime_validation = "not_run"`.
Consumers must independently verify artifact availability/identity and negotiate
actual runtime capabilities, provider support, current grants, budgets and ready
capacity. Build/resolve completion is not runtime readiness or benchmark success.

The exclusive output directory retains `plan.json`, bounded `build.log`, the
builder's `build-output.json` and validated `artifact.json`. Failures retain a
typed `failure.json`; retry uses a new directory rather than overwriting evidence.
Receipt files are published atomically after file fsync and directory fsync;
an existing receipt is never replaced.
Cancellation fences the local process group and retains its disposition. It does
not establish absence of resources a builder created through external providers.
Machine stdout contains only the final JSON object; builder progress stays in
the retained log.

Python consumers use `plan_container_package`, async `build_container_package`
and `validate_container_build_output`. Models and schemas are public through
`synth_containers`; `package schema` also exposes plan/build-output/artifact
schemas. Hosted operation IDs, resumable uploads, runtime launch, catalog-wide
adoption and optimizer recipe migration are subsequent accepted work.
