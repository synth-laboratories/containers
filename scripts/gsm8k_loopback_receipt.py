"""GSM8K through the loopback rollout lane against a real synth-mlx-rl service.

A mechanism receipt, not a score. It proves, once, that the pinned GSM8K world
is reachable from both container-side sampling paths with a real local policy:

1. ``POST /training/rollouts`` — the typed training boundary, sampling through
   the hosted sampler contract at ``/v1/training/sample`` with the policy
   pinned to one published snapshot; the summary carries the token record.
2. ``POST /rollouts`` with the ``synth_mlx_rl`` provider — the OpenAI-shaped
   surface; the event log seals a ``token_capture`` whose ``proxy_request_ids``
   resolve against the service's own rollout record under the same
   ``policy_snapshot_id``.

Everything written here is what the lane observed: the dataset manifest the
container advertised, the rollout summary, the event logs, and the join. Bearer
tokens are redacted before anything is written.

    uv run --with datasets python scripts/gsm8k_loopback_receipt.py \
        --base-url http://127.0.0.1:8791 --out docs/receipts/<date>/gsm8k-loopback

The service must already be running (see synth-mlx-rl ``scripts/mlx_guard.py``
for the admission/watchdog discipline around a resident model); this script
never starts or stops a model.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import os
import platform as host_platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

# The dataset pin is code; the profile is declared here, in code, for this run.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# Admission for a plaintext *loopback* sampler. `SamplerEndpoint.validate` still
# refuses any non-loopback host, so this cannot widen the lane past 127.0.0.1.
os.environ["SYNTH_CONTAINERS_ALLOW_LOOPBACK_SAMPLER"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from synth_containers.platform import create_compat_app  # noqa: E402
from synth_containers.platform import gsm8k_world  # noqa: E402
from synth_containers.training_rollout import ROLLOUT_REQUEST_SCHEMA_VERSION  # noqa: E402

TELEMETRY = {"enabled": True, "transport": "sse", "retention": "run"}
BEARER = "local-dev-token"
REDACTED = "<redacted>"


def _get(url: str, timeout: float = 30.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - loopback only
        return json.loads(response.read())


def _post(url: str, payload: dict[str, Any], timeout: float = 600.0) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read())


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (REDACTED if key in {"bearer_token", "auth_bearer", "api_key"} else _redact(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_redact(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8791")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--profile", choices=("hf", "fixture"), default="hf")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]

    gsm8k_world.declare_profile(args.profile)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # --- service identity -----------------------------------------------------
    health = _get(f"{base}/healthz")
    capability = _get(f"{base}/v1/synth/capability")
    snapshot = _post(f"{base}/v1/synth/snapshots", {"metadata": {"purpose": "gsm8k-loopback-receipt"}})
    snapshot_id = snapshot["policy_snapshot_id"]
    _write(out / "service.json", {"healthz": health, "capability": capability, "snapshot": snapshot})

    client = TestClient(create_compat_app("gsm8k_solve", storage_root=tempfile.mkdtemp(prefix="gsm8k-loopback-receipt-")))
    metadata = client.get("/metadata").json()
    training_caps = client.get("/training/capabilities").json()
    _write(out / "metadata.json", {"metadata": metadata, "training_capabilities": training_caps})
    dataset = metadata["dataset"]
    assert dataset["pinned"] is (args.profile == "hf"), dataset
    assert dataset["revision"] == gsm8k_world.HF_REVISION

    # --- 1. the typed training boundary ----------------------------------------
    request = {
        "schema_version": ROLLOUT_REQUEST_SCHEMA_VERSION,
        "job_id": "gsm8k-loopback-receipt",
        "attempt_id": "attempt-1",
        "rollout_id": f"gsm8k_training_seed{args.seed}",
        "idempotency_key": f"gsm8k-loopback-receipt:{snapshot_id}:{args.seed}",
        "policy_version": snapshot_id,
        "sampler": {
            "url": f"{base}/v1/training/sample",
            "bearer_token": BEARER,
            "connection_mode": "keep_alive",
        },
        "task": {
            "world_ref": "world:gsm8k@heldout",
            "task_instance_id": f"seed:{args.seed}",
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
        },
    }
    t0 = time.time()
    response = client.post("/training/rollouts", json=request)
    training_seconds = round(time.time() - t0, 2)
    assert response.status_code == 200, response.text
    summary = response.json()
    training_events = client.get(
        f"/rollouts/{request['rollout_id']}/events", params={"after": 0}
    ).json()["events"]
    _write(
        out / "training_rollout.json",
        {"request": request, "response": summary, "wall_seconds": training_seconds},
    )
    _write(out / "training_rollout_events.json", training_events)
    action = summary["actions"][0]
    assert action["policy_version"] == snapshot_id
    assert len(action["token_ids"]) == len(action["log_probs"]) > 0
    training_action_event = next(e["payload"] for e in training_events if e["kind"] == "action")

    # --- 2. the synth_mlx_rl provider surface + the join ------------------------
    config_id = "mlx_loopback_chat"
    registered = client.post(
        "/policy-configs",
        json={
            "config_id": config_id,
            "harness": "solve",
            "config": {
                "provider": "synth_mlx_rl",
                "api_family": "chat_completions",
                "model": health.get("model") or "Qwen/Qwen3.5-0.8B",
                "base_url": f"{base}/v1",
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "timeout_seconds": 600,
            },
        },
    )
    assert registered.status_code == 200, registered.text
    rollout_id = f"gsm8k_provider_seed{args.seed}"
    assert client.post("/rollouts/prepare", json={"rollout_id": rollout_id, "telemetry": TELEMETRY}).status_code == 200
    t0 = time.time()
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": rollout_id,
            "telemetry": TELEMETRY,
            "slot": "stream",
            "world_ref": "world:gsm8k@heldout",
            "task_instance_id": f"seed:{args.seed}",
            "policy_ref": {"harness": "solve", "config": config_id},
        },
    )
    provider_seconds = round(time.time() - t0, 2)
    assert started.status_code == 200, started.text
    reward = client.post("/reward", json={"rollout_id": rollout_id, "mode": "terminal"}).json()
    provider_events = client.get(f"/rollouts/{rollout_id}/events", params={"after": 0}).json()["events"]
    _write(
        out / "provider_rollout.json",
        {"started": started.json(), "reward": reward, "wall_seconds": provider_seconds},
    )
    _write(out / "provider_rollout_events.json", provider_events)
    capture = next(e["payload"] for e in provider_events if e["kind"] == "token_capture")
    assert capture["proxy_request_ids"], capture
    assert capture["policy_snapshot_id"], capture
    provider_action = next(e["payload"] for e in provider_events if e["kind"] == "action")

    prid = capture["proxy_request_ids"][0]
    record = _get(f"{base}/v1/synth/rollouts/{prid}")["record"]
    join = {
        "proxy_request_id": prid,
        "container_policy_snapshot_id": capture["policy_snapshot_id"],
        "service_policy_snapshot_id": record["policy_snapshot_id"],
        "snapshot_ids_agree": record["policy_snapshot_id"] == capture["policy_snapshot_id"],
        "service_completion_tokens": len(record["completion_token_ids"]),
        "service_rollout_logprobs": len(record["rollout_logprobs"]),
        "aligned": len(record["completion_token_ids"]) == len(record["rollout_logprobs"]),
        "published_snapshot_id": snapshot_id,
        "provider_sampled_published_snapshot": record["policy_snapshot_id"] == snapshot_id,
    }
    assert join["snapshot_ids_agree"] and join["aligned"], join
    _write(out / "join_check.json", join)

    receipt = {
        "schema_version": "gsm8k.loopback-receipt.v1",
        "kind": "mechanism receipt (not a score)",
        "started_at": started_at,
        "host": {"platform": host_platform.platform(), "python": sys.version.split()[0]},
        "containers_commit": _git(repo, "rev-parse", "HEAD"),
        "service": {
            "base_url": base,
            "version": health.get("version"),
            "model": health.get("model"),
            "published_snapshot_id": snapshot_id,
        },
        "dataset": {
            "dataset": dataset["dataset"],
            "revision": dataset["revision"],
            "profile": dataset["profile"],
            "profile_source": dataset["profile_source"],
            "pinned": dataset["pinned"],
            "splits": dataset["splits"],
            "shuffle_seed": dataset["shuffle_seed"],
        },
        "capabilities_digest": metadata["capabilities_digest"],
        "training_rollout": {
            "rollout_id": request["rollout_id"],
            "status": summary["status"],
            "reward": summary["reward"]["reward"],
            "policy_version": summary["policy_version"],
            "completion_tokens": len(action["token_ids"]),
            "parse_mode": training_action_event["parse_mode"],
            "format_compliant": training_action_event["format_compliant"],
            "wall_seconds": training_seconds,
        },
        "provider_rollout": {
            "rollout_id": rollout_id,
            "status": started.json()["status"],
            "reward": reward["reward"],
            "reward_status": reward["status"],
            "parse_mode": provider_action["parse_mode"],
            "format_compliant": provider_action["format_compliant"],
            "token_capture": {
                "proxy_request_ids": capture["proxy_request_ids"],
                "policy_snapshot_id": capture["policy_snapshot_id"],
                "provenance": capture["provenance"],
            },
            "wall_seconds": provider_seconds,
        },
        "join": join,
        "files": sorted(path.name for path in out.iterdir()),
    }
    _write(out / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
