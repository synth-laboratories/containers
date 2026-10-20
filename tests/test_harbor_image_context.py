"""Image preparation must bind a bounded immutable context before provider calls."""

import json

import pytest

from synth_containers.harbor_environment import HarborEnvironmentError, inspect_harbor_package
from synth_containers.harbor_image_context import freeze_harbor_image_context


def draft(tmp_path):
    source = tmp_path / "source"
    (source / "environment/workspace").mkdir(parents=True)
    (source / "tests").mkdir()
    (source / "environment/Dockerfile").write_text(
        "FROM python:3.12-slim\nCOPY workspace /workspace\n"
    )
    (source / "environment/workspace/start.py").write_text("print('start')\n")
    (source / "instruction.md").write_text("Complete the task.\n")
    (source / "task.toml").write_text(
        '[task]\nname="fixture"\n[agent]\ntimeout_sec=30\n[verifier]\ntimeout_sec=30\n[environment]\ncpus=1\nmemory_mb=1024\nstorage_mb=1024\n'
    )
    (source / "tests/test.sh").write_text("exit 0\n")
    return inspect_harbor_package(source)


def test_context_freeze_preserves_dockerfile_and_excludes_verifier(tmp_path):
    source = draft(tmp_path)
    context = freeze_harbor_image_context(source, tmp_path / "frozen")
    assert (context.root / "Dockerfile").read_bytes() == (
        source.root / "environment/Dockerfile"
    ).read_bytes()
    assert not (context.root / "tests").exists()
    assert json.loads((tmp_path / "frozen/context-receipt.json").read_text()) == context.as_dict()
    context.verify()
    with pytest.raises(FileExistsError):
        freeze_harbor_image_context(source, tmp_path / "frozen")


def test_changed_context_cannot_be_submitted(tmp_path):
    context = freeze_harbor_image_context(draft(tmp_path), tmp_path / "frozen")
    (context.root / "workspace/start.py").write_text("changed")
    with pytest.raises(HarborEnvironmentError, match="context_changed"):
        context.verify()


def test_source_change_is_refused_before_destination_creation(tmp_path):
    source = draft(tmp_path)
    (source.root / "tests/test.sh").write_text("exit 1")
    with pytest.raises(HarborEnvironmentError, match="source_changed"):
        freeze_harbor_image_context(source, tmp_path / "frozen")
    assert not (tmp_path / "frozen").exists()


def test_context_cannot_be_frozen_inside_source(tmp_path):
    source = draft(tmp_path)
    with pytest.raises(HarborEnvironmentError, match="cannot_mutate"):
        freeze_harbor_image_context(source, source.root / "nested")


def test_cli_freezes_context_without_loading_provider_sdk(tmp_path, capsys):
    from synth_containers.cli import main

    source = draft(tmp_path)
    assert main(["harbor-image-context", str(source.root), str(tmp_path / "frozen")]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["source_package_digest"] == source.source_package_digest
    assert (tmp_path / "frozen/context/Dockerfile").is_file()
