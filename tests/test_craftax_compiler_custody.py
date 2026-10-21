"""Compiler cleanup needs an acknowledged ID and exact ownership evidence."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

path = Path(__file__).resolve().parents[1] / "scripts/prepare_craftax_runtime.py"
spec = importlib.util.spec_from_file_location("craftax_compiler", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_known_id_absence_is_confirmed_but_ambiguous_create_is_not(tmp_path):
    cid = tmp_path / "cid"
    def absent(command, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="error: no such object: " + command[-1])
    unknown = module.cleanup_compiler("owned", "op", cid, {}, image_id="image", run=absent)
    assert unknown["status"] == "create_outcome_unknown"
    cid.write_text("exact-id")
    known = module.cleanup_compiler("owned", "op", cid, {}, image_id="image", run=absent)
    assert known == {"status":"independently_absent","instance_id":"exact-id"}


@pytest.mark.parametrize("field", ["label", "image"])
def test_foreign_compiler_is_never_removed(tmp_path, field):
    cid = tmp_path / "cid"
    cid.write_text("exact-id")
    commands = []
    row = {"Id":"exact-id", "Image":"image", "Config":{"Labels":{"synth.rhodes.craftax-compiler":"op"}}}
    if field == "label":
        row["Config"]["Labels"]["synth.rhodes.craftax-compiler"] = "foreign"
    else:
        row["Image"] = "foreign-image"
    def inspect(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps([row]), stderr="")
    result = module.cleanup_compiler("owned", "op", cid, {}, image_id="image", run=inspect)
    assert result["status"] == "foreign_preserved"
    assert all(command[1] == "inspect" for command in commands)


def test_remove_acknowledgement_without_absence_remains_unconfirmed(tmp_path):
    cid = tmp_path / "cid"
    cid.write_text("exact-id")
    row = {"Id":"exact-id", "Image":"image", "Config":{"Labels":{"synth.rhodes.craftax-compiler":"op"}}}
    commands = []
    def respond(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps([row]) if command[1] == "inspect" else "", stderr="")
    result = module.cleanup_compiler("owned", "op", cid, {}, image_id="image", run=respond)
    assert result["status"] == "unconfirmed"
    assert commands[1] == ["docker","rm","-f","-v","exact-id"]


def test_missing_socket_is_not_container_absence(tmp_path):
    cid = tmp_path / "cid"
    cid.write_text("exact-id")
    def missing(command, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="cannot connect: no such file or directory")
    assert module.cleanup_compiler("owned","op",cid,{},image_id="image",run=missing)["status"] == "unknown"
