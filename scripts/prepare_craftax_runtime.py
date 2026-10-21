#!/usr/bin/env python3
"""Qualify a frozen Craftax Linux engine for the runtime-only image stage.

Uses synth-dev's existing mbx/admission/storage-guard pattern. No Rust compiler
runs during Docker image assembly. All compiler inputs and outputs are owned,
byte-pinned and retained; this command does not activate a service.
"""

import argparse
import hashlib
import json
import os
import signal
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

BOUNDARY = Path("/Users/joshuapurtell/GitHub")
SOURCE_MOUNT = "/opt/gamebench/tasks/craftax-singleplayer"
RUST_IMAGE = "rust:1.97-slim-bookworm@sha256:2775a09d208ff0d7c1f50490c45b62db929e87ba1dcbc3f2132ac71a704bcdd3"
MBX_SHA256 = "3b7408c71fae119b09c506ab47daff6cba6476c5e7140111959824bd72e63331"
TARGET_LIMIT = 8 * 1024**3


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def owned(path):
    resolved = path.resolve()
    if not resolved.is_relative_to(BOUNDARY):
        raise ValueError("compiler path outside the authorized GitHub boundary")
    return resolved


def source_inputs(source):
    names = subprocess.check_output(["git", "-C", str(source), "ls-files", "-z"], text=True).split(
        "\0"
    )
    selected = {}
    for name in names:
        if not name or not (
            name.startswith(("gold_rust/src/", "defaults/", "shared/assets/craftax/"))
            or name in {"gold_rust/Cargo.toml", "gold_rust/Cargo.lock"}
        ):
            continue
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("compiler input must be a tracked regular file: " + name)
        selected[name] = digest(path)
    if (
        not {
            "gold_rust/Cargo.toml",
            "gold_rust/Cargo.lock",
            "defaults/worlds.json",
            "shared/assets/craftax/grass.png",
        }
        <= selected.keys()
    ):
        raise ValueError("required Craftax source inputs missing")
    return dict(sorted(selected.items()))


def cleanup_compiler(name, operation, cidfile, environment, *, image_id, run=subprocess.run):
    """A missing inventory row proves absence only for an acknowledged ID."""
    identifier = cidfile.read_text().strip() if cidfile.exists() else ""
    identity = identifier or name

    def probe(identity):
        return run(
            ["docker", "inspect", identity],
            capture_output=True,
            text=True,
            timeout=20,
            env=environment,
        )

    def absent(result, identity):
        text = result.stderr.lower()
        return result.returncode != 0 and any(
            (prefix + identity.lower()) in text
            for prefix in ("no such object: ", "no such container: ")
        )

    try:
        observed = probe(identity)
        if absent(observed, identity):
            return {
                "status": "independently_absent" if identifier else "create_outcome_unknown",
                "instance_id": identifier or None,
            }
        if observed.returncode:
            return {"status": "unknown", "instance_id": identifier or None}
        row = json.loads(observed.stdout)[0]
        if (row.get("Config", {}).get("Labels") or {}).get(
            "synth.rhodes.craftax-compiler"
        ) != operation or row.get("Image") != image_id:
            return {"status": "foreign_preserved", "instance_id": row["Id"]}
        removed = run(
            ["docker", "rm", "-f", "-v", row["Id"]],
            capture_output=True,
            text=True,
            timeout=20,
            env=environment,
        )
        if removed.returncode or not absent(probe(row["Id"]), row["Id"]):
            return {"status": "unconfirmed", "instance_id": row["Id"]}
        return {"status": "independently_absent", "instance_id": row["Id"]}
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, IndexError):
        return {"status": "unknown", "instance_id": identifier or None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--storage-guard", type=Path, required=True)
    parser.add_argument("--mbx-linux-dir", type=Path, required=True)
    parser.add_argument("--operation-id")
    args = parser.parse_args()
    source, output, launcher, guard, mbx = map(
        owned,
        (args.source_root, args.output_root, args.launcher, args.storage_guard, args.mbx_linux_dir),
    )
    if launcher.name != "core_dev_build.py" or guard != launcher.with_name(
        "storage_target_guard.py"
    ):
        raise ValueError("maintained launcher/storage guard pair required")
    if digest(mbx / "mbx") != MBX_SHA256:
        raise ValueError("qualified Linux ARM64 mbx identity mismatch")
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True):
        raise ValueError("clean pinned source required")
    inputs = source_inputs(source)
    operation = args.operation_id or uuid4().hex
    if len(operation) != 32 or any(char not in "0123456789abcdef" for char in operation):
        raise ValueError("canonical operation identity required")
    out = output / operation
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = out / "compiler-artifact.json"
    if receipt_path.exists():
        raise ValueError("retained compiler operation requires reconciliation; do not overwrite it")
    target = out / "target"
    temporary = out / "tmp"
    target.mkdir()
    temporary.mkdir()
    cache = BOUNDARY / "artifacts/linux-mbx-cache/local-image"
    for name in ("cargo-home", "mbx-cache", "apt-lists", "apt-archives"):
        (cache / name).mkdir(parents=True, exist_ok=True)
    receipt = dict(
        schema_version="synth.craftax.compiler-artifact.v1",
        status="create_intent",
        operation_id=operation,
        source_revision=revision,
        source_root=str(source),
        source_input_sha256=inputs,
        platform="linux/arm64",
        rust_image=RUST_IMAGE,
        mbx_sha256=MBX_SHA256,
        launcher_sha256=digest(launcher),
        storage_guard_sha256=digest(guard),
        target_limit_bytes=TARGET_LIMIT,
    )

    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    receipt["preparation_script_sha256"] = digest(Path(__file__))
    save()
    environment_host = os.environ.copy()
    docker_config = out / "docker-config"
    docker_config.mkdir(mode=0o700)
    (docker_config / "config.json").write_text('{"auths":{}}\n')
    endpoint = (
        environment_host.get("DOCKER_HOST")
        or subprocess.check_output(
            ["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"], text=True
        ).strip()
    )
    if not endpoint.startswith("unix://"):
        raise ValueError("local Unix Docker endpoint required")
    environment_host.update(DOCKER_CONFIG=str(docker_config), DOCKER_HOST=endpoint)
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", RUST_IMAGE, "--format", "{{.Id}}"],
        env=environment_host,
        text=True,
        timeout=20,
    ).strip()
    if not image_id.startswith("sha256:") or len(image_id) != 71:
        raise ValueError("exact local compiler image identity required")
    receipt["compiler_image_id"] = image_id
    save()

    def interrupted(signum, frame):
        raise KeyboardInterrupt("owned Craftax compiler interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    name = "rhodes-craftax-compile-" + operation
    cidfile = out / "container-id"
    manifest = SOURCE_MOUNT + "/gold_rust/Cargo.toml"
    # The existing Linux mbx lane shares caches, while target/config/tmp custody
    # remains with this invocation. Source is read-only at the compiled-in path.
    script = f"""set -eu
if ! command -v python3 >/dev/null || ! command -v pkg-config >/dev/null; then
  rm -f /etc/apt/apt.conf.d/docker-clean
  if [ -z "$(find /var/lib/apt/lists -maxdepth 1 -name '*_Packages*' 2>/dev/null)" ]; then apt-get update; fi
  apt-get install -y --no-install-recommends -o APT::Keep-Downloaded-Packages=true python3 pkg-config ca-certificates
fi
/mbx/mbx test --locked --release --manifest-path {manifest} --lib
/mbx/mbx build --locked --release --manifest-path {manifest} --bin craftax_gold
"""
    command = [
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--platform",
        "linux/arm64",
        "--name",
        name,
        "--label",
        "synth.rhodes.craftax-compiler=" + operation,
        "--cidfile",
        str(cidfile),
        "--memory",
        "8g",
        "--cpus",
        "2",
        "--mount",
        f"type=bind,src={source},dst={SOURCE_MOUNT},readonly",
        "--mount",
        f"type=bind,src={out},dst={out}",
        "--mount",
        f"type=bind,src={temporary},dst=/tmp/craftax-mbx",
        "--mount",
        f"type=bind,src={mbx},dst=/mbx,readonly",
        "--mount",
        f"type=bind,src={cache / 'cargo-home'},dst=/var/cache/cargo",
        "--mount",
        f"type=bind,src={cache / 'mbx-cache'},dst=/var/cache/mbx",
        "--mount",
        f"type=bind,src={cache / 'apt-lists'},dst=/var/lib/apt/lists",
        "--mount",
        f"type=bind,src={cache / 'apt-archives'},dst=/var/cache/apt/archives",
        "--workdir",
        SOURCE_MOUNT,
    ]
    environment = {
        "CARGO_TARGET_DIR": str(target),
        "CARGO_HOME": "/var/cache/cargo",
        "MBX_CACHE_DIR": "/var/cache/mbx",
        "CARGO_BUILD_JOBS": "2",
        "MBX_TARGET_VIEWS": "0",
        "RUSTC_WRAPPER": "",
        "CARGO_BUILD_RUSTC_WRAPPER": "",
        "TMPDIR": "/tmp/craftax-mbx",
        "XDG_CONFIG_HOME": str(out / "config"),
        "XDG_CACHE_HOME": str(out / "cache"),
        "XDG_DATA_HOME": str(out / "data"),
        "MBX_GC_MAX_TOTAL_SIZE": "80GiB",
        "MBX_GC_MIN_FREE_SIZE": "20GiB",
        "MBX_GC_INTERVAL": "10m",
    }
    for key, value in environment.items():
        command += ["--env", key + "=" + value]
    command += [
        image_id,
        "timeout",
        "--signal=TERM",
        "--kill-after=10",
        "1500",
        "/bin/sh",
        "-ec",
        script,
    ]
    wrapped = [
        sys.executable,
        str(launcher),
        "--wait-seconds",
        "60",
        "--label",
        name,
        "--job-class",
        "qualification",
        "--",
        sys.executable,
        str(guard),
        "--target",
        str(target),
        "--limit-bytes",
        str(TARGET_LIMIT),
        "--",
        *command,
    ]
    receipt["command"] = wrapped
    save()
    try:
        with (out / "compile.log").open("w") as log:
            result = subprocess.run(
                wrapped,
                cwd=source,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1800,
                env=environment_host,
            )
        receipt["exit_code"] = result.returncode
        if result.returncode:
            raise RuntimeError("guarded Craftax compilation failed; inspect retained compile.log")
        if (
            subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip()
            != revision
            or source_inputs(source) != inputs
            or digest(Path(__file__)) != receipt["preparation_script_sha256"]
            or subprocess.check_output(
                ["git", "-C", str(source), "status", "--porcelain"], text=True
            )
        ):
            raise ValueError("compiler inputs changed during qualification")
        binary = target / "release/craftax_gold"
        with binary.open("rb") as stream:
            header = stream.read(20)
        if (
            len(header) != 20
            or header[:6] != b"\x7fELF\x02\x01"
            or int.from_bytes(header[18:20], "little") != 183
        ):
            raise ValueError("Linux ARM64 engine required")
        runtime = out / "runtime"
        runtime.mkdir()
        shutil.copy2(binary, runtime / "craftax_gold")
        for directory in ("defaults", "shared/assets/craftax"):
            for relative in inputs:
                if relative.startswith(directory + "/"):
                    destination = runtime / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source / relative, destination)
        receipt.update(
            status="compiled",
            binary_sha256=digest(runtime / "craftax_gold"),
            runtime_directory=str(runtime),
        )
    except BaseException as error:
        receipt.update(status="failed", error=type(error).__name__)
        raise
    finally:
        receipt["cleanup"] = cleanup_compiler(
            name, operation, cidfile, environment_host, image_id=image_id
        )
        save()
    if receipt["cleanup"]["status"] != "independently_absent":
        receipt["status"] = "cleanup_pending"
        save()
        raise RuntimeError("compiler cleanup requires reconciliation")
    receipt["status"] = "assembling"
    save()
    (runtime / "build-inputs.json").write_text(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "schema_version",
                    "source_revision",
                    "source_input_sha256",
                    "platform",
                    "rust_image",
                    "mbx_sha256",
                    "binary_sha256",
                )
            },
            sort_keys=True,
        )
        + "\n"
    )
    receipt["status"] = "qualified"
    save()
    print(
        json.dumps(
            {"receipt": str(receipt_path), "runtime_directory": receipt["runtime_directory"]}
        )
    )


if __name__ == "__main__":
    main()
