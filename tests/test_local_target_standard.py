"""Local target standard: Dockerfile + compose + code, never GHCR."""

from __future__ import annotations

import tomllib
from pathlib import Path

from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "targets"
REQUIRED_IDS = ("banking77", "healthbench", "craftax")
TELEMETRY = {"enabled": True, "transport": "poll", "retention": "run"}


def _catalog() -> dict:
    payload = tomllib.loads((TARGETS / "catalog.toml").read_text(encoding="utf-8"))
    assert payload["catalog"]["pull"] is False
    rows = {row["id"]: row for row in payload["target"]}
    return rows


def test_catalog_locks_three_local_targets_and_forbids_ghcr() -> None:
    rows = _catalog()
    assert tuple(sorted(rows)) == tuple(sorted(REQUIRED_IDS))
    for path in TARGETS.rglob("*"):
        if path.is_file() and path.suffix in {".toml", ".yaml", ".yml"}:
            text = path.read_text(encoding="utf-8")
            assert "ghcr.io" not in text.lower(), path
    for catalog_id, row in rows.items():
        assert row["contract"] == "synth.container.live-eval.v1"
        compose = ROOT / row["compose"]
        dockerfile = ROOT / row["dockerfile"]
        assert compose.is_file(), compose
        assert dockerfile.is_file(), dockerfile
        compose_text = compose.read_text(encoding="utf-8")
        docker_text = dockerfile.read_text(encoding="utf-8")
        assert "ghcr.io" not in compose_text.lower()
        from_lines = [
            line.strip().lower()
            for line in docker_text.splitlines()
            if line.strip().lower().startswith("from ")
        ]
        assert from_lines, dockerfile
        assert all("ghcr.io" not in line for line in from_lines), dockerfile
        assert f"127.0.0.1:{row['port']}:" in compose_text
        assert row["target_id"] in compose_text


def test_three_targets_advertise_live_eval_and_accept_prepare() -> None:
    rows = _catalog()
    for catalog_id, row in rows.items():
        client = TestClient(create_compat_app(row["target_id"]))
        info = client.get("/info")
        assert info.status_code == 200, catalog_id
        capabilities = info.json().get("capabilities") or {}
        assert capabilities.get("protocol") == "synth.container.live-eval.v1", catalog_id
        operations = capabilities.get("operations") or {}
        assert operations.get("rollouts.prepare") is True, catalog_id
        prepared = client.post(
            "/rollouts/prepare",
            json={"rollout_id": f"std-{catalog_id}", "telemetry": TELEMETRY},
        )
        assert prepared.status_code == 200, (catalog_id, prepared.status_code, prepared.text)
        stream = prepared.json().get("stream") or {}
        poll = ((stream.get("transports") or {}).get("poll") or {}).get("url")
        assert poll, catalog_id
        events = client.get(poll, params={"after": 0})
        assert events.status_code == 200, catalog_id
        kinds = [row["kind"] for row in events.json().get("events") or []]
        assert "stream.subscribed" in kinds, (catalog_id, kinds)
