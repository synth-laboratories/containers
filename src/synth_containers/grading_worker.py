"""Standalone worker for owner-trusted pure rubrics; never executes submissions.

The controller sanitizes environment, bounds CPU/wall/RSS/output and retains all
bytes. This is not an untrusted-code or network security sandbox. Such graders
require the separate admitted-external-grader interface.
"""

from __future__ import annotations

import json
from importlib import import_module
import os
from pathlib import Path
import sys
import threading
import time


def evaluate(request: dict) -> dict:
    inputs = request["inputs"]
    scorer = request["scorer"]
    cases = {}
    if scorer == "json-exact-v1":
        for name in request["case_ids"]:
            row = inputs["cases"][name]
            if "actual" not in row or "expected" not in row:
                raise ValueError("missing exact-comparison evidence")
            passed = row["actual"] == row["expected"]
            cases[name] = {"score": float(passed), "passed": passed}
    elif scorer == "craftax-trajectory-metrics-v1":
        records = inputs["trajectory"]["records"]
        for name in request["case_ids"]:
            if name == "actions":
                value = len(records)
            elif name == "achievements":
                value = len(records[-1]["state"]["achievements"])
            elif name == "reward":
                value = sum(row["reward"] for row in records)
            else:
                raise ValueError("unsupported trajectory metric")
            cases[name] = {"score": value, "passed": None}
    elif scorer.startswith("cardcodebench-captured-"):
        package = Path("native")
        for name, source in inputs["scorer_source"]["files"].items():
            target = package / "cardcodebench" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source)
        sys.path.insert(0, str(package.absolute()))
        # This is the deliberately dynamic, frozen native-rubric plugin boundary.
        # Preflight verified every source file and the owner's exact trust digest.
        module = "cardcodebench.panel20" if scorer.endswith("v1") else "cardcodebench.rubric_v2"
        judge = getattr(import_module(module), "judge")
        if not callable(judge):
            raise ValueError("native judge interface unavailable")
        observations = inputs["observations"]
        for name, case, observed in zip(request["case_ids"], observations["cases"], observations["outputs"], strict=True):
            errors = judge(case, observed, inputs["authority"])
            cases[name] = {"score": float(not errors), "passed": not errors,
                           "reason": "native-rule-failure" if errors else "native-rule-pass"}
    else:
        raise ValueError("unsupported scorer")
    return {"cases": cases}


if __name__ == "__main__":
    parent = os.getppid()
    def watch_owner():
        while True:
            time.sleep(0.2)
            if os.getppid() != parent:
                os._exit(74)
    threading.Thread(target=watch_owner, daemon=True).start()
    request = json.load(sys.stdin)
    result = json.dumps(evaluate(request), sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    # Worker completion is committed before pipe output so controller death or
    # a lost publication reply cannot cause a completed rubric to run again.
    temporary = Path(".worker-result.pending")
    with temporary.open("xb") as handle:
        handle.write(result)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, "worker-result.json")
    descriptor = os.open(".", os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.write(sys.stdout.fileno(), result)
