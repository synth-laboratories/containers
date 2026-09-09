#!/usr/bin/env python3
"""Remove trial containers that outlived the process that started them.

    uv run python scripts/reap_expired_trials.py          # past their deadline
    uv run python scripts/reap_expired_trials.py --all    # every trial container

`NestedTrial.timeout_seconds` kills `docker run`, not the container. A client
that is SIGKILLed leaves the container grading on the daemon with nobody
watching, and `reap_rollout`'s `finally` never runs. Each trial now carries a
`synth.deadline` label so any later process can judge it; the runtime sweeps
expired ones before each rollout, and this is the same sweep by hand.

`--all` ignores deadlines and removes every labelled trial container. Use it
when a run has been abandoned and its verifiers are still inside their windows
-- a DEO grade is 45 minutes, which is a long time to wait for a box to recover.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from synth_containers.nested import DEADLINE_LABEL, reap_expired  # noqa: E402


def reap_all(binary: str = "docker") -> list[str]:
    listed = subprocess.run(  # noqa: S603
        [binary, "ps", "-aq", "--filter", f"label={DEADLINE_LABEL}"],
        capture_output=True,
        text=True,
        check=False,
    )
    ids = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
    if ids:
        subprocess.run([binary, "rm", "-f", *ids], capture_output=True, check=False)  # noqa: S603
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="remove every labelled trial container, deadline or not",
    )
    args = parser.parse_args()
    removed = reap_all() if args.all else reap_expired()
    if removed:
        print(f"removed {len(removed)}: {' '.join(removed)}")
    else:
        print("nothing to reap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
