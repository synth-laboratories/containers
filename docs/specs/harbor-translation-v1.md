# Harbor bundle translation v1 (Reading B authoring adapter)

`synth_containers.harbor_translation` statically turns a Harbor-shaped bundle
into a `synth.container-package.v1` package with an
`execution-requirements.toml` sidecar. It is the optional Harbor authoring
adapter from the accepted Reading B ADR: Harbor concepts (phases, verifier
separation, result files, static inspection) normalize onto the same package
and canonical execution preflight as hand-written `container.toml` packages.

```sh
synth-containers package translate-harbor BUNDLE --policy POLICY.toml --output OUT
synth-containers package execution-plan OUT/container.toml
```

## Static only

Translation never builds an image, executes the Dockerfile, imports bundle code,
runs shell commands or reads credentials. Inputs are snapshotted with the
package file bounds; symlinks, special files and bound overflows refuse.

## Dialects

| Dialect | Detected by | Notes |
| --- | --- | --- |
| `harbor-task-1.0` | `version = "1.0"` and `[verifier]` | Resources from `[environment]`; agent wall time from `[agent].timeout_sec`; `environment_mode` must be declared. |
| `reb-task-metadata` | `[task]` and `[splits]` | No resources or verifier mode in source. Hidden splits force an `isolated` verifier; wall time is `[budgets].agent_wall_time_seconds`. |

Anything else is `harbor_translation_dialect_unknown`.

## What is packaged

Only `environment/` (copied to `context/`) and `instruction.md`. `tests/` and
`solution/` never enter the package; the receipt lists each excluded file's
digest so a later trusted verifier can bind the exact hidden bytes. Alternate
recipes (`environment/Dockerfile.*`) are excluded. The source Dockerfile is
replaced by a translated root `Dockerfile`.

A visible file is refused (`harbor_translation_hidden_input_visible`) when its
bytes equal a hidden data file (Harbor 1.0: any `tests/` file; REB: the
evaluator `heldout/` tree and `heldout_eval.json`), or when a path component
names a declared hidden split or `heldout`.

## Verifier and phases

Phases are `build` (`build.sh`), `agent` (`instruction.md`) and `verifier`
(`tests/test.sh`, never packaged). Harbor `separate` maps to `isolated`,
`shared` to `shared`, and the source spelling is retained. Absent mode refuses;
there is no shared fallback. The REB dialect has no mode: declared hidden
splits make its trusted held-out scorer `isolated`.

Each phase gets its own `ExecutionRequirements` and is preflighted separately:

- The packaged training container (`execution-requirements.toml`) uses the
  agent wall time and the policy network. Its `verifier_isolation` is
  `self_reported`: training scores from that container are unattested.
- The verifier phase (`verifier_requirements` in the receipt, not a package
  input) uses the verifier timeout, verifier network and the verifier mode.
  Harbor `network_mode = "no-network"` maps to `none`; any other mode refuses
  until an exact allow-list exists. A Harbor verifier inherits
  `[environment]` resources.

## Requirements and policy

`synth.harbor-translation-policy.v1` supplies only what bundles do not
declare: `platform`, the training-container `network`, `resources` (when the
source has none), `runtime_command` (when the Dockerfile has no exec-form
`CMD`) and `[verifier]` `network`/`max_wall_seconds`/`resources`. A policy
value that differs from a source declaration refuses with
`harbor_translation_policy_conflicts_source`; a missing verifier fact refuses
with `verifier_requirement_undeclared`. `[metadata]` is recorded verbatim and
never grants resources (ResinBench `gpu = "H100"` describes the owner
evaluator, not the training container). `privileged` is always false. The
receipt names the provenance of every field for both phases.

## Raw cargo

A Dockerfile `RUN` that invokes cargo is rewritten. Supported chains are
`cd <dir>`, plain commands, `cargo fetch` and `cargo build` joined by `&&`.
`cargo fetch` becomes a build-host `mbx fetch` into `context/.synth-cargo-home`
(plus the generated `Cargo.lock` beside the copied manifest), copied into the
image's `CARGO_HOME` registry. `cargo build` was a compile warm-up and is
removed. Other cargo subcommands, any other cargo mention (absolute paths,
`VAR=… cargo`, exec-form `RUN`), shell control flow (`;`, `|`, `||`, `(…)`,
`$(…)`, redirects), unbalanced quoting, or a fetch directory that is not a
copied context directory refuse. A cargo token that survives in an emitted
`RUN` refuses as a post-condition.

## Outputs

`container.toml`, `execution-requirements.toml`, `Dockerfile`, `build.sh`,
`docker_build_receipt.py`, `context/`, `instruction.md` and
`translation-receipt.json`. The translation receipt is outside the build inputs.
The emitted package is re-planned with `plan_execution_package`; its refusal is
`harbor_translation_package_refused`. Translation is deterministic.

## Local image identity

The generated entrypoint streams Docker's OCI-layout export through the receipt
writer. The layout must name one image root. For an image index, the writer
selects the unique image manifest for the requested platform. It verifies
manifest and config descriptor kinds, sizes, SHA-256 digests and the config
platform before emitting an `oci_config` identity. The build log retains the
Docker lookup ID and selected manifest digest separately. Legacy archives
without an OCI descriptor layout refuse with `docker_image_receipt_invalid`.

A registry release needs an independently read-back, digest-pinned single-image
manifest. Changing the receipt writer changes the frozen package input digest.

## What this does not prove

A supported `preflight_execution` result against a fixture offer shows that the
requirements fit an offer of that shape. It does not qualify a live supplier,
adapter, image or egress policy. `scripts/harbor_translation_proof.py` writes
labelled fixture receipts; see `receipts/syn-5647/`.
