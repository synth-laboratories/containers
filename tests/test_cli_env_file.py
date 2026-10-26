from pathlib import Path

import pytest

from synth_containers.cli import _env_pairs
from synth_containers.launch import LaunchError


def test_env_file_supplies_bare_forwards_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FIXTURE_KEY", raising=False)
    monkeypatch.delenv("OTHER_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_text('# c\nexport FIXTURE_KEY="abc"\nOTHER_KEY=zzz\n', encoding="utf-8")
    assert _env_pairs(["FIXTURE_KEY", "X=1"], path) == {"FIXTURE_KEY": "abc", "X": "1"}
    with pytest.raises(LaunchError, match="env_missing:FIXTURE_KEY"):
        _env_pairs(["FIXTURE_KEY"])


def test_host_env_wins_over_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIXTURE_KEY", "host")
    path = tmp_path / ".env"
    path.write_text("FIXTURE_KEY=file\n", encoding="utf-8")
    assert _env_pairs(["FIXTURE_KEY"], path) == {"FIXTURE_KEY": "host"}
