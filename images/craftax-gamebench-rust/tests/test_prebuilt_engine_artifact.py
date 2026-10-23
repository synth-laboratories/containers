"""Runtime assembly refuses modified, extra and wrongly targeted compiler bytes."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("craftax_engine_verification", ROOT / "verify_engine_artifact.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture(autouse=True)
def arm_runtime(monkeypatch):
    monkeypatch.setattr(module.platform, "machine", lambda: "aarch64")


def artifact(root):
    header = b"\x7fELF\x02\x01" + b"\0" * 12 + (183).to_bytes(2, "little")
    (root / "craftax_gold").write_bytes(header + b"unit fixture only")
    defaults = root / "defaults/worlds.json"
    defaults.parent.mkdir()
    defaults.write_text("{}")
    manifest = dict(schema_version="synth.craftax.compiler-artifact.v1", platform="linux/arm64",
        source_revision="a" * 40,
        mbx_sha256="3b7408c71fae119b09c506ab47daff6cba6476c5e7140111959824bd72e63331",
        rust_image="rust:1.97-slim-bookworm@sha256:2775a09d208ff0d7c1f50490c45b62db929e87ba1dcbc3f2132ac71a704bcdd3",
        binary_sha256=hashlib.sha256((root / "craftax_gold").read_bytes()).hexdigest(),
        source_input_sha256={"defaults/worlds.json":hashlib.sha256(defaults.read_bytes()).hexdigest()})
    (root / "build-inputs.json").write_text(json.dumps(manifest))
    return manifest


def test_exact_artifact_verifies(tmp_path):
    manifest = artifact(tmp_path)
    assert module.verify(tmp_path) == manifest


@pytest.mark.parametrize("change", ["binary", "data", "extra", "symlink", "compiler", "platform"])
def test_changed_or_unowned_artifact_refuses(tmp_path, change):
    manifest = artifact(tmp_path)
    if change == "binary":
        (tmp_path / "craftax_gold").write_bytes(b"changed")
    elif change == "data":
        (tmp_path / "defaults/worlds.json").write_text("changed")
    elif change == "extra":
        (tmp_path / "untracked").write_text("extra")
    elif change == "symlink":
        (tmp_path / "craftax_gold").rename(tmp_path / "other")
        (tmp_path / "craftax_gold").symlink_to(tmp_path / "other")
    elif change == "compiler":
        manifest["mbx_sha256"] = "b" * 64
    else:
        manifest["platform"] = "linux/amd64"
    (tmp_path / "build-inputs.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        module.verify(tmp_path)


def test_runtime_boot_refuses_missing_binary_instead_of_compiling(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    stack = importlib.import_module("craftax_gold.stack")
    monkeypatch.delenv("SYNTH_CRAFTAX_GOLD_BIN", raising=False)
    with pytest.raises(RuntimeError, match="craftax_gold_prebuilt_required"):
        stack.resolve_binary()
    monkeypatch.setenv("SYNTH_CRAFTAX_GOLD_BIN", str(ROOT / "missing-binary"))
    with pytest.raises(RuntimeError, match="craftax_gold_bin_missing"):
        stack.resolve_binary()


@pytest.mark.parametrize("machine", ["x86_64", "amd64", "armv7l", "unknown"])
def test_runtime_architecture_must_match_the_prepared_engine(tmp_path,monkeypatch,machine):
    monkeypatch.setattr(module.platform,"machine",lambda:machine)
    with pytest.raises(ValueError,match="requires an ARM64 runtime image"):
        module.verify(tmp_path)
