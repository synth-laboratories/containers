"""Every operation a Workshop skill names must exist in the operations surface.

The skill files live in the Workshop repo; this test finds them through
``WORKSHOP_SKILLS_DIR`` or the sibling checkout and skips when neither exists.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from synth_containers.tracing.annotation import OPERATION_DESCRIPTORS

SKILLS = ("trace-v5-annotate", "trace-v5-verify", "craftax-trace-analysis", "annotation-review")
NAME = re.compile(r"`((?:annotation|verification)_[a-z_]+)\(")


def _skills_dir() -> Path | None:
    candidates = [os.environ.get("WORKSHOP_SKILLS_DIR")]
    candidates += [str(Path(__file__).resolve().parents[2] / repo / "apps" / "synth_desktop" / "skills") for repo in ("workshop-v08-e2e-refactor", "workshop")]
    for candidate in candidates:
        if candidate and Path(candidate).is_dir() and all((Path(candidate) / name / "SKILL.md").exists() for name in SKILLS):
            return Path(candidate)
    return None


def test_every_operation_named_in_a_skill_exists() -> None:
    skills_dir = _skills_dir()
    if skills_dir is None:
        pytest.skip("Workshop skills directory not found (set WORKSHOP_SKILLS_DIR)")
    known = {item["name"] for item in OPERATION_DESCRIPTORS}
    for name in SKILLS:
        text = (skills_dir / name / "SKILL.md").read_text(encoding="utf-8")
        named = set(NAME.findall(text))
        unknown = sorted(named - known)
        assert not unknown, f"{name} names operations that do not exist: {unknown}"
        assert "Status: experimental" not in text, f"{name} is wired now; the experimental banner must be gone"
        assert "container_id" in text, f"{name} must tell agents to name the sealing container"
        assert "wait=" not in text and "PaidComputeAuthorization" not in text, f"{name} still documents removed semantics"
