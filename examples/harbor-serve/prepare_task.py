"""Materialize the oracle smoke task with an explicitly pinned task image."""

import argparse
import re
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--image", required=True)
args = parser.parse_args()
if not re.fullmatch(r"[^\s]+@sha256:[a-f0-9]{64}", args.image):
    parser.error("--image must be an immutable OCI digest reference")
root = Path(__file__).parent / "tasks" / "example"
(root / "solution").mkdir(parents=True, exist_ok=True)
(root / "tests").mkdir(exist_ok=True)
(root / "instruction.md").write_text("Write the word done to /tmp/answer.\n")
(root / "task.toml").write_text(
    f'[environment]\ndocker_image = "{args.image}"\ncpus = 1\nmemory_mb = 2048\nstorage_mb = 8192\n'
    "[agent]\ntimeout_sec = 30\n[verifier]\ntimeout_sec = 30\n"
)
(root / "solution" / "solve.sh").write_text("#!/bin/sh\nprintf done > /tmp/answer\n")
(root / "tests" / "test.sh").write_text(
    "#!/bin/sh\nmkdir -p /logs/verifier\n"
    'if [ "$(cat /tmp/answer)" = done ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt\n'
)
