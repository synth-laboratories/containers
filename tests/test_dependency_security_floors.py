"""Keep the September release's reviewed dependency security floors intact."""

from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[1]
FLOORS = {"starlette": "1.3.1", "idna": "3.15", "cryptography": "50.0.0", "urllib3": "2.7.0"}


def test_lock_uses_reviewed_patched_versions():
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    versions = {p["name"]: Version(p["version"]) for p in lock["package"]}
    for name, floor in FLOORS.items():
        assert versions[name] >= Version(floor), name


def test_published_runtime_and_dev_requirements_preserve_security_floors():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    runtime = {r.name: r for raw in project["dependencies"] if (r := Requirement(raw))}
    development = {r.name: r for raw in project["optional-dependencies"]["dev"] if (r := Requirement(raw))}
    for name in ("starlette", "idna"):
        assert str(runtime[name].specifier) == f">={FLOORS[name]}"
    for name in ("cryptography", "urllib3"):
        assert str(development[name].specifier) == f">={FLOORS[name]}"
    assert "tblite" not in runtime
