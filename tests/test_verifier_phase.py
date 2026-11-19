"""SYN-5685: the held-out verifier phase is derived from the Harbor receipt, never uploaded freely."""

from __future__ import annotations

import json

import pytest

from synth_containers.package_cli import main, verifier_phase_from_paths
from synth_containers.package_models import ImageArtifactIdentity
from synth_containers.verifier_phase import (
    PHASE_FILENAME,
    VerifierPhaseCode,
    VerifierPhaseError,
    bind_verifier_phase,
    derive_verifier_phase,
)
from tests.test_harbor_translation import policy, reb_bundle
from synth_containers.harbor_translation import translate_harbor_bundle

CANDIDATE = "sha256:" + "a" * 64
VERIFIER = "sha256:" + "b" * 64


def image(digest: str, name: str) -> ImageArtifactIdentity:
    return ImageArtifactIdentity(kind="oci_image", digest_kind="oci_manifest",
                                 reference=f"ghcr.io/synth/{name}@{digest}", digest=digest,
                                 platform="linux/amd64")


def config(base: str = CANDIDATE) -> dict:
    return {"os": "linux", "architecture": "amd64",
            "config": {"Labels": {"org.opencontainers.image.base.digest": base}}}


def translated(tmp_path):
    bundle = reb_bundle(tmp_path / "013-rust-miprov2")
    task = (bundle / "task.toml").read_text()
    task = task.replace('language = "rust"\n', 'language = "rust"\nsubmission_file = "miprov2.rs"\n')
    task = task.replace("[paths]\n", '[paths]\nworkspace = "workspace"\n')
    (bundle / "task.toml").write_text(task)
    return translate_harbor_bundle(bundle, tmp_path / "out", policy())


def derive(receipt, plan, **changes):
    args = dict(candidate=image(CANDIDATE, "candidate"), verifier=image(VERIFIER, "verifier"),
                verifier_config=config())
    args.update(changes)
    return derive_verifier_phase(receipt, plan, **args)


def code(callable_, *args, **kwargs):
    with pytest.raises(VerifierPhaseError) as raised:
        callable_(*args, **kwargs)
    return raised.value.refusal.code


def test_reb_phase_follows_the_task_declared_verifier(tmp_path):
    receipt, plan = translated(tmp_path)
    phase = derive(receipt, plan)
    # Wall time, network and isolation are the receipt's verifier requirements.
    assert phase.requirements == receipt.verifier_requirements
    assert phase.requirements.max_wall_seconds == 3600
    assert phase.requirements.network.mode == "none"
    assert phase.requirements.verifier_isolation == "isolated"
    assert phase.requirements_digest == receipt.verifier_requirements_digest
    assert phase.translation_receipt_digest == receipt.digest()
    assert phase.command == ("/tests/test.sh",)
    workdir = plan.build_plan.manifest.runtime.working_directory
    assert [(f.candidate_path, f.verifier_path) for f in phase.submission] == [
        (f"{workdir}/workspace/miprov2.rs", "/submission/miprov2.rs")
    ]
    assert phase.result_path == "/logs/verifier/reward.txt"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("command", ["/bin/sh", "-c", "echo 1 > /logs/verifier/reward.txt"]),
        ("requirements.max_wall_seconds", 60),
        ("requirements.network", {"mode": "allowlist", "hosts": ["exfil.example"]}),
        ("result_path", "/tmp/forged.txt"),
        ("submission", [{"candidate_path": "/tests/evaluator/heldout/healthbench.json",
                         "verifier_path": "/submission/miprov2.rs", "max_bytes": 64}]),
        ("translation_receipt_digest", "sha256:" + "0" * 64),
    ],
)
def test_an_uploaded_phase_that_differs_from_the_receipt_is_refused(tmp_path, field, value):
    receipt, plan = translated(tmp_path)
    document = json.loads(derive(receipt, plan).model_dump_json())
    head, _, tail = field.partition(".")
    if tail:
        document[head][tail] = value
    else:
        document[head] = value
    raw = json.dumps(document).encode()
    assert code(bind_verifier_phase, raw, receipt, plan, candidate=image(CANDIDATE, "candidate"),
                verifier_config=config()) is VerifierPhaseCode.PHASE_MISMATCH


def test_the_exact_derivation_binds(tmp_path):
    receipt, plan = translated(tmp_path)
    phase = derive(receipt, plan)
    assert bind_verifier_phase(phase.model_dump_json().encode(), receipt, plan,
                               candidate=image(CANDIDATE, "candidate"), verifier_config=config()) == phase


def test_verifier_image_must_be_built_from_the_candidate_and_distinct(tmp_path):
    receipt, plan = translated(tmp_path)
    assert code(derive, receipt, plan, verifier_config=config("sha256:" + "c" * 64)) \
        is VerifierPhaseCode.IMAGE_LINEAGE_MISMATCH
    assert code(derive, receipt, plan, verifier=image(CANDIDATE, "candidate")) \
        is VerifierPhaseCode.NOT_DISTINCT


def test_a_receipt_from_another_package_is_refused(tmp_path):
    receipt, plan = translated(tmp_path)
    other = receipt.model_copy(update={"build_input_digest": "sha256:" + "d" * 64})
    assert code(derive, other, plan) is VerifierPhaseCode.RECEIPT_MISMATCH


def test_a_task_without_a_declared_submission_has_no_phase(tmp_path):
    receipt, plan = translate_harbor_bundle(reb_bundle(tmp_path / "bundle"), tmp_path / "out", policy())
    assert code(derive, receipt, plan) is VerifierPhaseCode.SUBMISSION_UNDECLARED


def test_cli_writes_the_phase_beside_the_package(tmp_path, capsys):
    receipt, plan = translated(tmp_path)
    candidate = tmp_path / "build-output.json"
    candidate.write_text(json.dumps({
        "schema_version": "synth.container-build-output.v1", "action": "resolve",
        "input_digest": plan.build_plan.input_digest,
        "artifact": image(CANDIDATE, "candidate").model_dump(mode="json"),
    }))
    blob = tmp_path / "verifier-config.json"
    blob.write_text(json.dumps(config()))
    ref = "ghcr.io/synth/verifier@" + VERIFIER
    assert main(["verifier-phase", str(tmp_path / "out"), "--candidate-output", str(candidate),
                 "--verifier-image", ref, "--verifier-config", str(blob)]) == 0
    written = (tmp_path / "out" / PHASE_FILENAME).read_bytes()
    assert bind_verifier_phase(written, receipt, plan, candidate=image(CANDIDATE, "candidate"),
                               verifier_config=config()) == verifier_phase_from_paths(
        tmp_path / "out", candidate, ref, blob)
    blob.write_text(json.dumps(config("sha256:" + "e" * 64)))
    assert main(["verifier-phase", str(tmp_path / "out"), "--candidate-output", str(candidate),
                 "--verifier-image", ref, "--verifier-config", str(blob)]) == 2
    assert "execution_verifier_image_lineage_mismatch" in capsys.readouterr().out
