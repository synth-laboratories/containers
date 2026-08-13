from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from synth_containers.tracing.capture.control_server import (
    DetachedCaptureConfig,
    DetachedCaptureSupervisor,
)
from synth_containers.tracing.projections.inspector import load_bundle
from synth_containers.tracing.store.bundle import LocalTraceBundle


def _open(client: httpx.Client, rollout_id: str) -> dict:
    response = client.post(
        "/captures",
        json={
            "rollout_id": rollout_id,
            "labels": {"test": rollout_id},
            "capture_mode": "required",
        },
    )
    response.raise_for_status()
    return response.json()


def test_detached_capture_seals_verifies_and_survives_restart(tmp_path: Path) -> None:
    service = DetachedCaptureSupervisor(
        DetachedCaptureConfig(output_root=tmp_path)
    ).start()
    with httpx.Client(base_url=service.base_url) as client:
        opened = _open(client, "rollout-a")
        capture_id = opened["capture_id"]
        for index in range(100):
            response = client.post(
                f"/captures/{capture_id}/events",
                json={
                    "event_type": "craftax.turn",
                    "payload": {"rollout_id": "rollout-a", "event_id": index},
                },
            )
            response.raise_for_status()
        artifact = client.post(
            f"/captures/{capture_id}/artifacts",
            json={
                "role": "observation",
                "media_type": "image/png",
                "logical_name": "frame.png",
                "content_base64": base64.b64encode(b"not-a-real-png").decode(),
            },
        )
        artifact.raise_for_status()
        sealed = client.post(
            f"/captures/{capture_id}/seal", json={"status": "completed"}
        )
        sealed.raise_for_status()
        result = sealed.json()
    service.stop()

    assert result["trace_v5_digest"].startswith("sha256:")
    assert result["event_count"] == 100
    assert LocalTraceBundle(Path(result["bundle_path"])).verify_self_contained() == (
        True,
        (),
    )

    restarted = DetachedCaptureSupervisor(
        DetachedCaptureConfig(output_root=tmp_path)
    ).start()
    try:
        response = httpx.get(f"{restarted.base_url}/captures/{capture_id}")
        response.raise_for_status()
        assert response.json()["trace_v5_digest"] == result["trace_v5_digest"]
        assert response.json()["status"] == "completed"
    finally:
        restarted.stop()


def test_ten_concurrent_captures_do_not_cross_talk(tmp_path: Path) -> None:
    service = DetachedCaptureSupervisor(
        DetachedCaptureConfig(output_root=tmp_path)
    ).start()
    try:
        with httpx.Client(base_url=service.base_url, timeout=30) as client:
            opened = [_open(client, f"rollout-{index}") for index in range(10)]

        def emit(item: tuple[int, dict]) -> None:
            index, capture = item
            with httpx.Client(base_url=service.base_url, timeout=30) as client:
                for event_index in range(20):
                    response = client.post(
                        f"/captures/{capture['capture_id']}/events",
                        json={
                            "event_type": "craftax.turn",
                            "payload": {
                                "rollout_id": f"rollout-{index}",
                                "event_id": f"{index}:{event_index}",
                            },
                        },
                    )
                    response.raise_for_status()

        with ThreadPoolExecutor(max_workers=10) as pool:
            tuple(pool.map(emit, enumerate(opened)))

        sealed = []
        with httpx.Client(base_url=service.base_url, timeout=30) as client:
            for capture in reversed(opened):
                response = client.post(
                    f"/captures/{capture['capture_id']}/seal",
                    json={"status": "completed"},
                )
                response.raise_for_status()
                sealed.append(response.json())

        for result in sealed:
            inspected = load_bundle(Path(result["bundle_path"]))
            assert len(inspected) == 1
            rollout_ids = {
                event.payload.get("rollout_id")
                for event in inspected[0].trace.events
                if event.event_type == "craftax.turn"
            }
            assert rollout_ids == {result["rollout_id"]}
            assert LocalTraceBundle(Path(result["bundle_path"])).verify_self_contained()[0]
    finally:
        service.stop()


def test_budget_refuses_or_evicts_without_touching_open_spools(tmp_path: Path) -> None:
    refuse_root = tmp_path / "refuse"
    refuse = DetachedCaptureSupervisor(
        DetachedCaptureConfig(
            output_root=refuse_root,
            capture_disk_budget_bytes=32 * 1024,
            budget_policy="refuse",
        )
    ).start()
    try:
        with httpx.Client(base_url=refuse.base_url) as client:
            first = _open(client, "first")
            response = client.post(
                f"/captures/{first['capture_id']}/events",
                json={"event_type": "craftax.turn", "payload": {"value": "x" * 1024}},
            )
            response.raise_for_status()
            sealed = client.post(
                f"/captures/{first['capture_id']}/seal", json={"status": "completed"}
            )
            sealed.raise_for_status()
            refused = client.post(
                "/captures",
                json={"rollout_id": "refused", "capture_mode": "best_effort"},
            )
            assert refused.status_code == 507
            assert LocalTraceBundle(Path(sealed.json()["bundle_path"])).verify_self_contained()[0]
    finally:
        refuse.stop()

    evict_root = tmp_path / "evict"
    evict = DetachedCaptureSupervisor(
        DetachedCaptureConfig(
            output_root=evict_root,
            capture_disk_budget_bytes=32 * 1024,
            budget_policy="evict_oldest_sealed",
        )
    ).start()
    try:
        with httpx.Client(base_url=evict.base_url) as client:
            oldest = _open(client, "oldest")
            sealed_oldest = client.post(
                f"/captures/{oldest['capture_id']}/seal", json={"status": "completed"}
            )
            sealed_oldest.raise_for_status()
            newest = _open(client, "newest")
            old_status = client.get(f"/captures/{oldest['capture_id']}")
            old_status.raise_for_status()
            assert old_status.json()["status"] == "evicted"
            assert not Path(sealed_oldest.json()["bundle_path"]).exists()
            new_status = client.get(f"/captures/{newest['capture_id']}")
            new_status.raise_for_status()
            assert new_status.json()["status"] == "open"
    finally:
        evict.stop()


def test_open_capture_is_interrupted_on_restart(tmp_path: Path) -> None:
    service = DetachedCaptureSupervisor(
        DetachedCaptureConfig(output_root=tmp_path)
    ).start()
    with httpx.Client(base_url=service.base_url) as client:
        opened = _open(client, "crashed")
        capture_id = opened["capture_id"]
        client.post(
            f"/captures/{capture_id}/events",
            json={"event_type": "craftax.turn", "payload": {"before": "crash"}},
        ).raise_for_status()

    # Model a process death after durable spool flush but before terminal authority.
    supervisor = service._captures[capture_id]
    supervisor._stop_capture_services(reason="test_crash")
    supervisor.session.close()
    service._server.shutdown()
    service._server.server_close()
    service._stopped = True

    restarted = DetachedCaptureSupervisor(
        DetachedCaptureConfig(output_root=tmp_path)
    ).start()
    try:
        response = httpx.get(f"{restarted.base_url}/captures/{capture_id}")
        response.raise_for_status()
        status = response.json()
        assert status["status"] == "interrupted"
        assert status["trace_v5_digest"].startswith("sha256:")
        assert LocalTraceBundle(Path(status["bundle_path"])).verify_self_contained()[0]
    finally:
        restarted.stop()
