"""Package input boundaries match Core's retained canonical artifact."""
import json
from pathlib import Path

import pytest

from synth_containers import file_limits_generated as limits
from synth_containers import harbor_environment, package_files
from synth_containers.package_models import portable_path


def test_generated_policy_is_the_canonical_artifact():
    root = Path(__file__).resolve().parents[1]
    policy = json.loads((root / "contracts/file-limits.json").read_text())
    for key, value in policy.items():
        if key != "schema":
            assert getattr(limits, key.upper()) == value
    assert package_files.MAX_FILE_BYTES == harbor_environment._MAX_FILE_BYTES == limits.FILE_MAX_BYTES
    assert package_files.MAX_PACKAGE_BYTES == harbor_environment._MAX_TREE_BYTES == limits.LAUNCH_MAX_BYTES
    assert package_files.MAX_ENTRIES == harbor_environment._MAX_TREE_ENTRIES == limits.LAUNCH_MAX_FILES


def test_file_boundary_accepts_binary_and_refuses_one_extra_byte(tmp_path):
    path = tmp_path / "binary.bin"
    with path.open("wb") as f:
        f.truncate(limits.FILE_MAX_BYTES)
    assert len(package_files.read_package_file(tmp_path, "binary.bin")) == limits.FILE_MAX_BYTES
    with path.open("ab") as f:
        f.write(b"\xff")
    with pytest.raises(package_files.ContainerPackageError, match="too_large"):
        package_files.read_package_file(tmp_path, "binary.bin")
    with pytest.raises(package_files.ContainerPackageError, match="byte_limit"):
        package_files.snapshot_inputs(tmp_path, ("binary.bin",))


def test_package_total_accepts_exact_limit_and_refuses_one_extra_byte(tmp_path):
    paths = []
    for n in range(limits.LAUNCH_MAX_BYTES // limits.FILE_MAX_BYTES):
        name = f"part{n}.bin"
        with (tmp_path / name).open("wb") as f:
            f.truncate(limits.FILE_MAX_BYTES)
        paths.append(name)
    files, _ = package_files.snapshot_inputs(tmp_path, tuple(paths))
    assert sum(f.size_bytes for f in files) == limits.LAUNCH_MAX_BYTES
    (tmp_path / "extra").write_bytes(b"x")
    with pytest.raises(package_files.ContainerPackageError, match="byte_limit"):
        package_files.snapshot_inputs(tmp_path, (*paths, "extra"))


@pytest.mark.parametrize("name", ["x" * 256, "a/../b", "é", "bad name"])
def test_package_names_obey_canonical_policy(name):
    with pytest.raises(ValueError):
        portable_path(name)
