"""Harbor bundles translate statically into a frozen, preflightable container package.

Fixture bundles mirror the exact shapes of REB 013 (REB task-metadata dialect,
hidden held-out scorer, raw cargo in the Dockerfile) and the ResinBench
coding-agent task (Harbor 1.0, separate/no-network verifier, H100 metadata).
Fixture offers exercise the canonical checker only; no supplier is qualified.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_containers.execution_package import plan_execution_package
from synth_containers.execution_requirements import (
    ArtifactVerification,
    ExecutionArtifact,
    ExecutionPlatform,
    ExecutionProvider,
    GPUKind,
    NetworkMode,
    OfferQualification,
    PreflightCode,
    PreflightResult,
    ScoringOfferCapabilities,
    VerifierIsolation,
    preflight_execution,
)
from synth_containers.harbor_translation import (
    HarborDialect,
    HarborTranslationCause,
    HarborTranslationCode,
    HarborTranslationError,
    HarborTranslationPolicy,
    translate_harbor_bundle,
)
from synth_containers.package_cli import main as package_main
from synth_containers.package_models import ImageArtifactIdentity

HELDOUT_BYTES = b'{"split": "healthbench", "rows": ["hidden-row-1"]}\n'
REB_DOCKERFILE = """FROM rust:1.85-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \\
      bash ca-certificates python3 \\
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY workspace/ /app/workspace/
COPY harness/ /app/harness/
# Prefetch/build harness crates so verifier compile is mostly offline.
RUN cd /app/harness \\
    && cp /app/workspace/miprov2.rs /app/harness/src/submission.rs \\
    && cargo fetch \\
    && cargo build --release --quiet
ENV TASK_ROOT=/app
CMD ["bash"]
"""


RESOURCES = {
    "cpu_millis": 4000,
    "memory_bytes": 8 << 30,
    "disk_bytes": 20 << 30,
    "gpu_kind": "none",
    "gpu_count": 0,
}


def write(path: Path, content: str | bytes, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)
    if executable:
        path.chmod(0o755)


def reb_bundle(root: Path, dockerfile: str = REB_DOCKERFILE) -> Path:
    write(
        root / "task.toml",
        """[task]
id = "013-rust-miprov2"
language = "rust"
[splits]
visible = ["banking77_train"]
hidden = ["heart_disease", "healthbench"]
[paths]
evaluator = "evaluator"
heldout_eval = "evaluator/heldout_eval.json"
[policy]
model = "openai/gpt-oss-20b"
[budgets]
agent_wall_time_seconds = 1800
""",
    )
    write(root / "instruction.md", "Optimize miprov2.rs.\n")
    write(root / "environment/Dockerfile", dockerfile)
    write(root / "environment/Dockerfile.local", "FROM scratch\nRUN cargo build\n")
    write(root / "environment/workspace/miprov2.rs", "pub fn run() {}\n")
    write(root / "environment/workspace/train/banking77.json", "[]\n")
    write(root / "environment/harness/Cargo.toml", '[package]\nname = "h"\nversion = "0.1.0"\n')
    write(root / "environment/harness/src/main.rs", "fn main() {}\n")
    write(root / "tests/test.sh", "#!/usr/bin/env bash\npython3 /tests/evaluator/score.py\n", True)
    write(root / "tests/evaluator/score.py", "print('trusted held-out scorer')\n")
    write(root / "tests/evaluator/heldout/healthbench.json", HELDOUT_BYTES)
    write(root / "tests/evaluator/heldout_eval.json", '{"seeds": [1, 2, 3]}\n')
    write(root / "fixtures/banking77.json", "[]\n")
    write(root / "solution/solve.sh", "#!/bin/sh\ncp ref.rs /app/workspace/miprov2.rs\n", True)
    return root


def resin_bundle(root: Path, verifier: str = 'environment_mode = "separate"\n') -> Path:
    write(
        root / "task.toml",
        f"""version = "1.0"
[metadata]
gpu = "H100"
[agent]
timeout_sec = 600.0
[verifier]
timeout_sec = 1500.0
{verifier}network_mode = "no-network"
[environment]
build_timeout_sec = 600.0
cpus = 2
memory_mb = 4096
storage_mb = 10240
""",
    )
    write(root / "instruction.md", "Make the candidate faster.\n")
    write(
        root / "environment/Dockerfile",
        "FROM python:3.12-slim\nWORKDIR /app\nCOPY candidate /app/candidate\n"
        "COPY bench.py /usr/local/bin/bench\nRUN chmod +x /usr/local/bin/bench\n",
    )
    write(root / "environment/candidate/qwen3/model.py", "MODEL = 'candidate'\n")
    write(root / "environment/bench.py", "print('development bench')\n")
    write(root / "tests/test.sh", "#!/bin/sh\npython3 /tests/verifier.py\n", True)
    write(root / "tests/verifier.py", "print('owner evaluator client')\n")
    write(root / "tests/evaluator.json", '{"url": "", "token": ""}\n')
    write(root / "solution/reference/qwen3/model.py", "MODEL = 'reference'\n")
    return root


def policy(**changes: object) -> HarborTranslationPolicy:
    fields: dict[str, object] = {
        "schema_version": "synth.harbor-translation-policy.v1",
        "platform": "linux/amd64",
        "network": {"mode": "none", "hosts": []},
        "resources": RESOURCES,
        "verifier": {
            "network": {"mode": "none", "hosts": []},
            "max_wall_seconds": 3600,
            "resources": RESOURCES,
        },
    }
    fields.update(changes)
    return HarborTranslationPolicy.model_validate(fields)


def offer(**changes: object) -> ScoringOfferCapabilities:
    # Fixture authority tests the checker only; no real supplier is qualified.
    fields: dict[str, object] = {
        "provider": ExecutionProvider.DAYTONA,
        "offer_id": "fixture:4vcpu-8gib",
        "option_id": "sbo1_" + "5" * 16 + "_" + "6" * 64,
        "catalog_digest": "sha256:" + "1" * 64,
        "descriptor_digest": "sha256:" + "2" * 64,
        "tariff_revision": "sha256:" + "7" * 64,
        "adapter_revision": "sha256:" + "3" * 64,
        "qualification": OfferQualification.QUALIFIED,
        "release_admitted": True,
        "platforms": [ExecutionPlatform.LINUX_AMD64],
        "resources": {
            "cpu_millis": 4000,
            "memory_bytes": 8 << 30,
            "disk_bytes": 20 << 30,
            "gpu_kind": "none",
            "gpu_count": 0,
        },
        "max_wall_seconds": 3600,
        "network_modes": [NetworkMode.NONE],
        "privileged": False,
        "network_policy_digest": "sha256:" + "4" * 64,
        "allowed_hosts": [],
        "verifier_modes": [VerifierIsolation.ISOLATED],
    }
    fields.update(changes)
    return ScoringOfferCapabilities.model_validate(fields)


def artifact(platform: str = "linux/amd64") -> ExecutionArtifact:
    digest = "sha256:" + "a" * 64
    return ExecutionArtifact(
        identity=ImageArtifactIdentity(
            kind="oci_image",
            digest_kind="oci_manifest",
            reference="registry.example/reb@" + digest,
            digest=digest,
            platform=platform,
        ),
        verification=ArtifactVerification.VERIFIED,
        inspection_receipt_digest="sha256:" + "b" * 64,
    )


def refusal(callable_, *args):
    with pytest.raises(HarborTranslationError) as caught:
        callable_(*args)
    return caught.value.refusal


def test_reb_bundle_translates_without_hidden_inputs_or_raw_cargo(tmp_path):
    bundle = reb_bundle(tmp_path / "013-rust-miprov2")
    receipt, plan = translate_harbor_bundle(bundle, tmp_path / "out", policy())
    out = tmp_path / "out"

    assert receipt.source_dialect is HarborDialect.REB_TASK_METADATA
    assert receipt.package_id == "013-rust-miprov2"
    packaged = {entry.path for entry in plan.build_plan.files}
    assert not any(path.startswith(("tests", "solution")) for path in packaged)
    assert "context/Dockerfile.local" not in packaged
    for path in packaged:
        assert HELDOUT_BYTES not in (out / path).read_bytes()
    excluded = {entry.path: entry.reason for entry in receipt.excluded_inputs}
    assert excluded["tests/evaluator/heldout/healthbench.json"] == "verifier_only"
    assert excluded["solution/solve.sh"] == "reference_solution"
    assert excluded["environment/Dockerfile"] == "translated_recipe"

    dockerfile = (out / "Dockerfile").read_text()
    assert "cargo fetch" not in dockerfile and "cargo build" not in dockerfile
    assert "RUN cd /app/harness && cp /app/workspace/miprov2.rs" in dockerfile
    assert "COPY .synth-cargo-home/registry/ /usr/local/cargo/registry/" in dockerfile
    build = (out / "build.sh").read_text()
    assert 'mbx fetch --manifest-path "$root/context/harness/Cargo.toml"' in build
    assert "cargo " not in build.replace("synth-cargo-home", "")
    assert (receipt.dockerfile_rewrites[0].action, receipt.dockerfile_rewrites[0].line) == (
        "replaced",
        9,
    )

    requirements = plan.execution_requirements
    # Training scores are self-reported; the held-out verifier is its own isolated phase.
    assert requirements.verifier_isolation is VerifierIsolation.SELF_REPORTED
    assert requirements.max_wall_seconds == 1800
    assert receipt.requirement_provenance["resources"] == "policy"
    assert receipt.verifier_requirements.verifier_isolation is VerifierIsolation.ISOLATED
    assert receipt.verifier_requirements.max_wall_seconds == 3600
    assert receipt.verifier_requirement_provenance["max_wall_seconds"] == "policy:verifier"
    assert excluded["fixtures/banking77.json"] == "outside_build_context"
    verifier = next(phase for phase in receipt.phases if phase.name == "verifier")
    assert (verifier.entrypoint, verifier.packaged) == ("tests/test.sh", False)
    assert receipt.execution_requirements_digest == requirements.digest()
    # Translation is deterministic: a second emission has the same identities.
    again, again_plan = translate_harbor_bundle(bundle, tmp_path / "out2", policy())
    assert again.digest() == receipt.digest()
    assert again_plan.build_plan.input_digest == plan.build_plan.input_digest


def test_reb_package_passes_and_refuses_canonical_preflight(tmp_path):
    receipt, plan = translate_harbor_bundle(
        reb_bundle(tmp_path / "bundle"), tmp_path / "out", policy()
    )
    replanned = plan_execution_package(tmp_path / "out" / "container.toml")
    assert replanned == plan
    requirements = plan.execution_requirements
    digest = receipt.execution_requirements_digest

    training_offer = offer(verifier_modes=[VerifierIsolation.SELF_REPORTED])
    supported = preflight_execution(
        requirements, expected_digest=digest, artifact=artifact(), offer=training_offer
    )
    assert (supported.result, supported.code) == (
        PreflightResult.SUPPORTED,
        PreflightCode.SUPPORTED,
    )
    arm = preflight_execution(
        requirements,
        expected_digest=digest,
        artifact=artifact("linux/arm64"),
        offer=training_offer,
    )
    assert arm.code is PreflightCode.ARTIFACT_PLATFORM_MISMATCH
    verifier = receipt.verifier_requirements
    isolated = preflight_execution(
        verifier,
        expected_digest=receipt.verifier_requirements_digest,
        artifact=artifact(),
        offer=offer(),
    )
    assert isolated.result is PreflightResult.SUPPORTED
    shared_only = preflight_execution(
        verifier,
        expected_digest=receipt.verifier_requirements_digest,
        artifact=artifact(),
        offer=offer(verifier_modes=[VerifierIsolation.SHARED, VerifierIsolation.SELF_REPORTED]),
    )
    assert shared_only.code is PreflightCode.VERIFIER_ISOLATION_UNSUPPORTED


def test_off_policy_egress_is_refused_by_preflight(tmp_path):
    _, plan = translate_harbor_bundle(
        reb_bundle(tmp_path / "bundle"),
        tmp_path / "out",
        policy(network={"mode": "allowlist", "hosts": ["scorer.example"]}),
    )
    result = preflight_execution(
        plan.execution_requirements,
        expected_digest=plan.execution_requirements_digest,
        artifact=artifact(),
        offer=offer(
            network_modes=[NetworkMode.NONE, NetworkMode.ALLOWLIST],
            allowed_hosts=["other.example"],
            verifier_modes=[VerifierIsolation.SELF_REPORTED],
        ),
    )
    assert result.code is PreflightCode.NETWORK_HOST_NOT_ALLOWED


def test_resinbench_bundle_preserves_separate_no_network_and_never_grants_metadata_gpu(tmp_path):
    receipt, plan = translate_harbor_bundle(
        resin_bundle(tmp_path / "coding-agent"),
        tmp_path / "out",
        policy(resources=None, runtime_command=("bash",), verifier={}),
    )
    requirements = plan.execution_requirements
    assert receipt.source_dialect is HarborDialect.HARBOR_TASK_1_0
    assert requirements.verifier_isolation is VerifierIsolation.SELF_REPORTED
    verifier_requirements = receipt.verifier_requirements
    assert verifier_requirements.verifier_isolation is VerifierIsolation.ISOLATED
    assert verifier_requirements.network.mode is NetworkMode.NONE
    assert verifier_requirements.max_wall_seconds == 1500
    assert verifier_requirements.resources == requirements.resources
    assert requirements.resources.gpu_kind is GPUKind.NONE
    assert (requirements.resources.cpu_millis, requirements.max_wall_seconds) == (2000, 600)
    assert requirements.resources.memory_bytes == 4096 << 20
    verifier = next(phase for phase in receipt.phases if phase.name == "verifier")
    assert verifier.source_isolation_spelling == "separate"
    assert (verifier.network_mode, verifier.timeout_seconds) == ("no-network", 1500)
    assert receipt.source_declarations["metadata"]["gpu"] == "H100"
    packaged = {entry.path for entry in plan.build_plan.files}
    assert "context/candidate/qwen3/model.py" in packaged
    assert not any("evaluator.json" in path or "reference" in path for path in packaged)
    result = preflight_execution(
        requirements,
        expected_digest=receipt.execution_requirements_digest,
        artifact=artifact(),
        offer=offer(verifier_modes=[VerifierIsolation.SELF_REPORTED]),
    )
    assert result.result is PreflightResult.SUPPORTED
    # The verifier's 1500 s is checked on its own: a 600 s offer cannot host it.
    short = preflight_execution(
        verifier_requirements,
        expected_digest=receipt.verifier_requirements_digest,
        artifact=artifact(),
        offer=offer(max_wall_seconds=600),
    )
    assert short.code is PreflightCode.WALL_TIME_UNSUPPORTED


def test_cli_translates_and_reports_typed_refusals(tmp_path, capsys):
    bundle = reb_bundle(tmp_path / "bundle")
    policy_path = tmp_path / "policy.toml"
    policy_path.write_text(
        """schema_version = "synth.harbor-translation-policy.v1"
platform = "linux/amd64"
[network]
mode = "none"
hosts = []
[resources]
cpu_millis = 4000
memory_bytes = 8589934592
disk_bytes = 0
gpu_kind = "none"
gpu_count = 0
[verifier]
max_wall_seconds = 3600
[verifier.network]
mode = "none"
hosts = []
[verifier.resources]
cpu_millis = 4000
memory_bytes = 8589934592
disk_bytes = 0
gpu_kind = "none"
gpu_count = 0
"""
    )
    assert (
        package_main(
            [
                "translate-harbor",
                str(bundle),
                "--policy",
                str(policy_path),
                "--output",
                str(tmp_path / "o"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["package_id"] == "013-rust-miprov2"
    assert (
        package_main(
            [
                "translate-harbor",
                str(bundle),
                "--policy",
                str(policy_path),
                "--output",
                str(tmp_path / "o"),
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().out)["code"] == HarborTranslationCode.OUTPUT_EXISTS


@pytest.mark.parametrize(
    ("mutate", "code", "cause"),
    [
        (
            lambda b: (b / "environment/link").symlink_to(b / "tests/evaluator/score.py"),
            HarborTranslationCode.INPUT_INVALID,
            HarborTranslationCause.INPUT_SYMLINK_OR_SPECIAL,
        ),
        (
            lambda b: write(b / "environment/workspace/cached.json", HELDOUT_BYTES),
            HarborTranslationCode.HIDDEN_INPUT_VISIBLE,
            HarborTranslationCause.HIDDEN_CONTENT_DUPLICATED,
        ),
        (
            lambda b: write(b / "environment/fixtures/heart_disease.json", "[]\n"),
            HarborTranslationCode.HIDDEN_INPUT_VISIBLE,
            HarborTranslationCause.HIDDEN_SPLIT_NAMED,
        ),
        (
            lambda b: (b / "tests/test.sh").unlink(),
            HarborTranslationCode.VERIFIER_UNDECLARED,
            HarborTranslationCause.VERIFIER_ENTRYPOINT_MISSING,
        ),
        (
            lambda b: write(b / "task.toml", "[something]\nx = 1\n"),
            HarborTranslationCode.DIALECT_UNKNOWN,
            HarborTranslationCause.DIALECT_UNRECOGNIZED,
        ),
        (
            lambda b: write(
                b / "environment/Dockerfile",
                REB_DOCKERFILE.replace("cargo fetch", "cargo install x"),
            ),
            HarborTranslationCode.RAW_CARGO_UNSUPPORTED,
            HarborTranslationCause.CARGO_SUBCOMMAND_UNSUPPORTED,
        ),
        (
            lambda b: write(
                b / "environment/Dockerfile",
                REB_DOCKERFILE.replace("&& cargo fetch", "; cargo fetch"),
            ),
            HarborTranslationCode.RAW_CARGO_UNSUPPORTED,
            HarborTranslationCause.CARGO_SHELL_UNPARSEABLE,
        ),
        *(
            (
                lambda b, run=run: write(b / "environment/Dockerfile", REB_DOCKERFILE + run + "\n"),
                HarborTranslationCode.RAW_CARGO_UNSUPPORTED,
                cause,
            )
            for run, cause in (
                (
                    "RUN /usr/local/cargo/bin/cargo build",
                    HarborTranslationCause.CARGO_INVOCATION_UNRECOGNIZED,
                ),
                (
                    "RUN CARGO_HOME=/x cargo fetch",
                    HarborTranslationCause.CARGO_INVOCATION_UNRECOGNIZED,
                ),
                (
                    "RUN (cd /app/harness && cargo build)",
                    HarborTranslationCause.CARGO_SHELL_UNPARSEABLE,
                ),
                ('RUN ["cargo", "build"]', HarborTranslationCause.CARGO_SHELL_UNPARSEABLE),
                ("RUN echo 'x && cargo fetch", HarborTranslationCause.CARGO_SHELL_UNPARSEABLE),
            )
        ),
        (
            lambda b: write(
                b / "environment/Dockerfile", REB_DOCKERFILE.replace('CMD ["bash"]', "CMD bash")
            ),
            HarborTranslationCode.DOCKERFILE_UNSUPPORTED,
            HarborTranslationCause.CMD_SHELL_FORM,
        ),
        (
            lambda b: write(b / "environment/Dockerfile", "FROM x\nADD https://e.example/a /a\n"),
            HarborTranslationCode.DOCKERFILE_UNSUPPORTED,
            HarborTranslationCause.INSTRUCTION_UNSUPPORTED,
        ),
    ],
)
def test_reb_negative_cases_refuse_before_emitting(tmp_path, mutate, code, cause):
    bundle = reb_bundle(tmp_path / "bundle")
    mutate(bundle)
    found = refusal(translate_harbor_bundle, bundle, tmp_path / "out", policy())
    assert (found.code, found.cause) == (code, cause)
    assert not (tmp_path / "out").exists()


def test_harbor_verifier_mode_must_be_declared_and_policy_cannot_override_source(tmp_path):
    undeclared = resin_bundle(tmp_path / "a", verifier="")
    found = refusal(
        translate_harbor_bundle, undeclared, tmp_path / "o1", policy(runtime_command=("bash",))
    )
    assert found.cause is HarborTranslationCause.VERIFIER_MODE_MISSING
    declared = resin_bundle(tmp_path / "b")
    found = refusal(
        translate_harbor_bundle, declared, tmp_path / "o2", policy(runtime_command=("bash",))
    )
    assert (found.code, found.cause) == (
        HarborTranslationCode.POLICY_CONFLICT,
        HarborTranslationCause.SOURCE_DECLARATION_DIFFERS,
    )
    found = refusal(
        translate_harbor_bundle, declared, tmp_path / "o3", policy(resources=None, verifier={})
    )
    assert found.cause is HarborTranslationCause.RUNTIME_COMMAND_UNDECLARED


def test_reb_verifier_requirements_must_be_declared_by_policy(tmp_path):
    found = refusal(
        translate_harbor_bundle,
        reb_bundle(tmp_path / "bundle"),
        tmp_path / "out",
        policy(verifier={"network": {"mode": "none", "hosts": []}, "resources": RESOURCES}),
    )
    assert (found.code, found.cause, found.detail) == (
        HarborTranslationCode.REQUIREMENT_MISSING,
        HarborTranslationCause.VERIFIER_REQUIREMENT_UNDECLARED,
        "verifier.max_wall_seconds",
    )
    conflict = refusal(
        translate_harbor_bundle,
        resin_bundle(tmp_path / "resin"),
        tmp_path / "o2",
        policy(resources=None, runtime_command=("bash",), verifier={"max_wall_seconds": 60}),
    )
    assert conflict.code is HarborTranslationCode.POLICY_CONFLICT
