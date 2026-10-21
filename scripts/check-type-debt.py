"""Check every source in compatible, locked environments; see docs/release-typing.md."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
SDK = "synth-ai @ git+ssh://git@github.com/synth-laboratories/synth-ai.git@9ab8088abf81cdb42f0897ecbe22c68e8f47485f"
ADDON = "src/synth_containers/tracing/capture/mitmproxy_addon.py"


def run(argv: list[str], *, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=ROOT, env=environment, text=True, capture_output=True)


def check_profile(profile: str, *, prepare: bool) -> bool:
    target = ROOT / ".local" / ("release-types-" + profile)
    environment = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(target))
    python = target / "bin" / "python"
    lock_digest = hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
    prepared_lock = target / "prepared-lock-sha256.txt"
    if prepare:
        commands = []
        if not python.exists():
            commands.append(["uv", "venv", "--python", "3.13", str(target)])
        commands.append([
            "uv", "sync", "--locked", "--no-default-groups", "--no-install-project", "--no-build",
            *(["--no-install-package", "synth-ai"] if profile == "main" else []),
            *(["--group", "dev", "--group", "release-types"] if profile == "main" else ["--group", "mitm-types"]),
        ])
        if profile == "main":
            # This exact SDK revision has a reviewed pure-Python setuptools
            # build. Every other source dependency is refused by profile sync.
            commands.append(["uv", "pip", "install", "--python", str(python), "--no-deps", SDK])
        for command in commands:
            result = run(command, environment=environment)
            if result.returncode:
                print(result.stdout + result.stderr, end="")
                return False
        prepared_lock.write_text(lock_digest + "\n")
    elif not prepared_lock.exists() or prepared_lock.read_text().strip() != lock_digest:
        print(f"{profile}: prepared environment does not match uv.lock")
        return False
    if not python.exists():
        print(f"{profile}: missing prepared environment")
        return False
    paths = ["src", "--exclude", ADDON] if profile == "main" else [ADDON]
    result = run([
        str(target / "bin" / "ty"), "check", *paths,
        "--python", str(python), "--output-format", "concise",
    ], environment=environment)
    output = result.stdout + result.stderr
    print(f"{profile}: {output}", end="" if output.endswith("\n") else "\n")
    diagnostics = [line for line in output.splitlines() if re.match(r"src/.*:\d+:\d+: (error|warning)\[", line)]
    # The retained baseline is historical evidence, not permission to re-add
    # old errors. Unknown checker failures and warnings both fail this gate.
    return result.returncode == 0 and not diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("all", "main", "mitm"), default="all")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    profiles = ("main", "mitm") if args.profile == "all" else (args.profile,)
    results = [check_profile(profile, prepare=not args.check_only) for profile in profiles]
    if not all(results):
        return 1
    print("Type-debt gate passed: zero diagnostics in every selected locked profile.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
