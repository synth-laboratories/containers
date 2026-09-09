"""Write digest-addressed PR conformance receipts (in-process, no --paid)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi.testclient import TestClient

from synth_containers.platform import PR_TARGETS, create_compat_app
from tests.conformance.container_compat.run import receipt_from_suite, run_against_client

RECEIPTS_DIR = Path(__file__).resolve().parent / "receipts"


def write_pr_receipts(directory: Path | None = None) -> list[dict]:
    out = directory or RECEIPTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    receipts: list[dict] = []
    for target in PR_TARGETS:
        storage_root = tempfile.mkdtemp(prefix=f"pr-receipt-{target}-")
        with TestClient(create_compat_app(target, storage_root=storage_root)) as client:
            suite = run_against_client(client, target, paid=False)
        receipt = receipt_from_suite(suite)
        path = out / f"{target}.json"
        path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        receipts.append(receipt)
    return receipts


def main() -> int:
    receipts = write_pr_receipts()
    failed = [row["target"] for row in receipts if row.get("failed")]
    json.dump(
        {
            "schema": "synth.container-compat-conformance.v1",
            "written": [row["target"] for row in receipts],
            "failed": failed,
        },
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
