# Container execution package v1

`execution_package.plan_execution_package` composes historical package planning
with the canonical `ExecutionRequirements` contract. It executes no build,
package code or provider operation. It does not change historical manifest or
build-plan schemas, default behavior or content digests.

An execution-enabled package explicitly lists `execution-requirements.toml` in
its build inputs. The sidecar contains every field of
`synth.container-execution-requirements.v1`, including verifier isolation. The
existing package reader rejects symlinks, special files, path escapes and input
limits. Execution planning then rereads the exact sidecar with a 64 KiB bound,
compares its size and SHA256 with the frozen input row, validates its TOML and
canonical requirements, and refuses disagreement with `build.platform`.

The returned `synth.container-execution-package.v1` retains:

- the original `PackageBuildPlan`;
- the exact requirements file identity already in that plan;
- the validated canonical `execution_requirements`;
- its `execution_requirements_digest`, using the canonical contract's existing
  sorted compact JSON digest method.

Byte identity and semantic identity serve different purposes. Changing even a
comment after the input snapshot is refused. Semantically equivalent sidecar
formatting may share a requirements digest but changes the build input digest.
A wire plan is not an authorization receipt: the trusted owner must plan from
its immutable package inputs or verify those inputs against the retained plan.
Never accept caller-authored source identities, image inspection or offer facts
as authority merely because their digests are syntactically valid.

The CLI path is `synth-containers package execution-plan container.toml`. A
requirements refusal emits one JSON line with code, stage, cause, package ID and
sidecar path. It contains neither source contents nor credentials. The exception
retains its chained parser/reader cause; its public refusal payload is immutable.
Historical `package plan` still permits packages without a sidecar.

This source seal is a prerequisite for platform and network preflight. It is not
an image inspection, an amd64 build, adapter qualification, supplier funding,
execution admission or verifier attestation. The placement owner still supplies
the exact inspected `ExecutionArtifact` and authenticated
`ScoringOfferCapabilities` to canonical preflight. A platform declaration cannot
relabel an arm64 image as amd64. The admitted requirements/digest and the resulting
preflight receipt must survive deployment dispatch and retries.

No Core runtime/wheel integration or historical package migration is performed
by this module. Core consumers must first use a wheel from the W02-integrated
Containers cohort. Tests cover frozen sidecar custody, changed bytes, invalid
encoding/TOML/schema, platform disagreement, compound-plan tampering, CLI refusal
shape and unchanged historical planning.
