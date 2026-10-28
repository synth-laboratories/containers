"""Engine egress isolation: internal network plus a two-network forwarding sidecar.

Docker publishes no host port from ``--network none`` or an ``--internal``
network. With declared egress ``none`` the engine therefore runs alone on an
owner-labelled internal network (no route out, no published port) and a small
sidecar, attached to that network and to the default bridge, publishes the
engine port on loopback and forwards each connection to the engine's internal
address. Every resource carries the owner label; the guard removes sidecar,
engine and network in that order and confirms each by observation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import subprocess

EGRESS_UNRESTRICTED = "unrestricted_bridge"
EGRESS_NONE = "none"
SUPPORTED_EGRESS = (EGRESS_UNRESTRICTED, EGRESS_NONE)
OBSERVED_NONE = "none(internal+sidecar)"
BRIDGE_NETWORK = "bridge"
# Present locally on supported hosts; never pulled (``--pull never``). Its
# busybox ``nc -lk -e`` is the forwarder. Absence is detected and reported as
# ``egress_isolation_unsupported``, never assumed.
SIDECAR_IMAGE = "alpine:3.20"
SIDECAR_MEMORY_MB = 64
SIDECAR_CPUS = "0.25"
SIDECAR_PIDS = 64
UNSUPPORTED_CODE = "egress_isolation_unsupported"


class EgressMismatch(RuntimeError):
    """Docker did not place the engine or sidecar as declared; never downgraded."""

    code = "native_engine_limits_mismatch"
    stage = "egress"

    def __init__(self, declared: dict, observed: dict | None, cause: str):
        self.declared, self.observed, self.cause = declared, observed, cause
        super().__init__(f"{self.code}: stage={self.stage} cause={cause}")


class EgressUnsupported(RuntimeError):
    """The host cannot provide the declared egress isolation (detected, e.g. sidecar image missing)."""

    code = UNSUPPORTED_CODE
    stage = "egress"

    def __init__(self, declared: str, reason: str):
        self.declared, self.reason = declared, reason
        super().__init__(f"{self.code}: stage={self.stage} declared={declared} cause={reason}")


@dataclass(frozen=True)
class EgressNames:
    """Derived, owner-scoped resource names for one allocation."""

    engine: str

    @property
    def network(self) -> str:
        return self.engine + "-net"

    @property
    def sidecar(self) -> str:
        return self.engine + "-fwd"


def requested_egress(request: dict) -> str:
    egress = request.get("egress") or EGRESS_UNRESTRICTED
    if egress not in SUPPORTED_EGRESS:
        raise ValueError(f"egress_unsupported: declared={egress} supported={','.join(SUPPORTED_EGRESS)}")
    return egress


def network_create_argv(docker: str, names: EgressNames, owner_label: str, allocation_label: str) -> list[str]:
    return [docker, "network", "create", "--internal", "--label", owner_label,
        "--label", f"{allocation_label}={names.network}", names.network]


def sidecar_create_argv(docker: str, names: EgressNames, owner_label: str, allocation_label: str,
                        port: int, engine_address: str, image: str) -> list[str]:
    """Forwarder on the bridge publishing loopback ``port``; the internal network is connected before start."""
    return [docker, "create", "--pull", "never", "--name", names.sidecar,
        "--label", owner_label, "--label", f"{allocation_label}={names.sidecar}",
        "--read-only", "--memory", f"{SIDECAR_MEMORY_MB}m", "--memory-swap", f"{SIDECAR_MEMORY_MB}m",
        "--cpus", SIDECAR_CPUS, "--pids-limit", str(SIDECAR_PIDS), "--network", BRIDGE_NETWORK,
        "-p", f"127.0.0.1::{port}", image, "nc", "-lk", "-p", str(port), "-e", "nc", engine_address, str(port)]


def network_names(networks) -> list[str]:
    return sorted(networks) if isinstance(networks, dict) else []


def engine_address(networks, network: str) -> str | None:
    entry = networks.get(network) if isinstance(networks, dict) else None
    address = entry.get("IPAddress") if isinstance(entry, dict) else None
    return address or None


def published(bindings, port: int) -> list[dict]:
    entries = bindings.get(f"{port}/tcp") if isinstance(bindings, dict) else None
    return [entry for entry in entries or () if isinstance(entry, dict)]


def verify_isolation(declared: dict, observed: dict, port: int) -> None:
    """Engine only on the internal network with nothing published; sidecar on both, publishing loopback."""
    network = declared["network"]
    if observed.get("network_internal") is not True:
        raise EgressMismatch(declared, observed, f"network_internal declared=true observed={observed.get('network_internal')}")
    if observed.get("engine_networks") != [network]:
        raise EgressMismatch(declared, observed, f"engine_networks declared={network} observed={observed.get('engine_networks')}")
    if observed.get("engine_port_bindings"):
        raise EgressMismatch(declared, observed, "engine_port_bindings declared=none observed=published")
    if not observed.get("engine_address"):
        raise EgressMismatch(declared, observed, "engine_address observed=none")
    if observed.get("sidecar_networks") != sorted((BRIDGE_NETWORK, network)):
        raise EgressMismatch(declared, observed, f"sidecar_networks declared=bridge,{network} observed={observed.get('sidecar_networks')}")
    bindings = published(observed.get("sidecar_port_bindings"), port)
    if not bindings or any(entry.get("HostIp") != "127.0.0.1" for entry in bindings):
        raise EgressMismatch(declared, observed, f"sidecar_port_bindings declared=127.0.0.1::{port} observed={bindings}")


def parse_json(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return None


def egress_capability(docker: str = "docker", image: str = SIDECAR_IMAGE) -> dict:
    """Detected host capability for declared egress ``none``: the sidecar image must be local.

    Never pulls. Network support itself is verified per allocation by the guard.
    """
    try:
        inspected = subprocess.run([docker, "image", "inspect", "-f", "{{.Id}}", image],
            capture_output=True, text=True, check=False)
    except OSError as error:
        inspected, reason = None, f"docker cli unavailable: {type(error).__name__}"
    identifier = inspected.stdout.strip() if inspected is not None else ""
    if inspected is not None and inspected.returncode == 0 and identifier.startswith("sha256:"):
        return {"supported": list(SUPPORTED_EGRESS), "none": {"supported": True, "code": None,
            "sidecar_image": image, "sidecar_image_id": identifier, "observed": OBSERVED_NONE}}
    if inspected is not None:
        reason = f"sidecar image {image} not present locally"
    return {"supported": [EGRESS_UNRESTRICTED], "none": {"supported": False, "code": UNSUPPORTED_CODE,
        "sidecar_image": image, "sidecar_image_id": None, "reason": reason}}
