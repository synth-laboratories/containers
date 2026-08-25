"""Repo-wide line-count cap on Python source files.

Part of the "Boring Boundaries" post-v0.7 structure programme, item P0-9
(handoff `v0.7-post-release-structure-handoff.md` §3, decision default
D-X-2: 2,000-line cap for Rust/Python files, 600 for renderer TS/TSX).

This is a *lock*, not a style opinion: it stops new drift, it does not fix
existing drift. The ALLOWLIST below is the complete set of files that were
already over the cap when this test was written (containers @ origin/v0.7
8481ff8, 2026-08-21), each with its line count at that time.

Rules enforced:
  * Any tracked-shape `.py` file NOT on the allowlist that exceeds the cap
    fails the test — new oversized files are not permitted.
  * Any allowlisted file that has been trimmed back to (or under) the cap
    ALSO fails the test, with a message to remove it from the allowlist.
    The allowlist may only ever shrink, never grow silently.
  * Adding a *new* entry to the allowlist for a file that is not actually
    over the cap fails immediately (keeps the list honest).
"""

from __future__ import annotations

import pathlib

LINE_CAP = 2000

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Directory names skipped anywhere in the tree (build/VCS/tooling noise,
# never source we own).
_EXCLUDED_DIR_NAMES = {
    ".venv",
    ".git",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".cursor",
}

# path (relative to repo root, posix) -> line count recorded at the time it
# was added to this list. The count is informational (drift in either
# direction is expected as the file continues to change); the enforced
# invariant is `test_allowlist_entries_still_needed` below.
ALLOWLIST: dict[str, int] = {
    "src/synth_containers/platform/state.py": 2044,
    "src/synth_containers/tracing/capture/finalizer.py": 2085,
    "src/synth_containers/tracing/capture/supervisor.py": 2331,
    "src/synth_containers/tracing/validation/validator.py": 4434,
    "tests/test_trace_v5_capture_security_regressions.py": 2549,
}


def _is_excluded(path: pathlib.Path) -> bool:
    rel_parts = path.relative_to(REPO_ROOT).parts[:-1]
    return any(part in _EXCLUDED_DIR_NAMES or part.endswith(".egg-info") for part in rel_parts)


def _iter_python_files():
    for path in sorted(REPO_ROOT.rglob("*.py")):
        if not _is_excluded(path):
            yield path


def _line_count(path: pathlib.Path) -> int:
    with path.open("r", encoding="utf-8", errors="surrogateescape") as fh:
        return sum(1 for _ in fh)


def test_no_new_oversized_python_files():
    violations = []
    for path in _iter_python_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in ALLOWLIST:
            continue
        count = _line_count(path)
        if count > LINE_CAP:
            violations.append(f"{rel}: {count} lines (cap {LINE_CAP})")

    assert not violations, (
        f"The following files exceed the {LINE_CAP}-line cap and are not on "
        "the allowlist in tests/code_quality/test_file_size_cap.py. Split the "
        "file, or if that is genuinely not feasible right now, add it to "
        "ALLOWLIST with its current line count:\n  " + "\n  ".join(violations)
    )


def test_allowlist_entries_still_needed():
    stale = []
    for rel in ALLOWLIST:
        path = REPO_ROOT / rel
        if not path.is_file():
            stale.append(f"{rel}: file no longer exists; remove it from ALLOWLIST")
            continue
        count = _line_count(path)
        if count <= LINE_CAP:
            stale.append(
                f"{rel}: now {count} lines (<= cap {LINE_CAP}); remove it from "
                "ALLOWLIST in tests/code_quality/test_file_size_cap.py — the "
                "list may only shrink"
            )

    assert not stale, "\n".join(stale)


def test_allowlist_entries_are_actually_over_cap():
    # Guards against padding the allowlist with files that were never
    # offenders in the first place.
    under_cap_at_recording = [
        f"{rel}: recorded at {recorded} lines, which is not over the "
        f"{LINE_CAP}-line cap"
        for rel, recorded in ALLOWLIST.items()
        if recorded <= LINE_CAP
    ]
    assert not under_cap_at_recording, "\n".join(under_cap_at_recording)
