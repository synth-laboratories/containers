"""An event can name the artifacts it carries, and the seal keeps the link."""

from __future__ import annotations

import base64
from pathlib import Path

import httpx

from synth_containers.tracing.capture.control_server import (
    DetachedCaptureConfig,
    DetachedCaptureSupervisor,
)
from synth_containers.tracing.projections.inspector import load_bundle


def test_an_event_keeps_the_artifact_it_declared(tmp_path: Path) -> None:
    service = DetachedCaptureSupervisor(DetachedCaptureConfig(output_root=tmp_path)).start()
    try:
        with httpx.Client(base_url=service.base_url) as client:
            opened = client.post(
                "/captures",
                json={"rollout_id": "rollout-link", "capture_mode": "required"},
            )
            opened.raise_for_status()
            capture_id = opened.json()["capture_id"]

            artifact = client.post(
                f"/captures/{capture_id}/artifacts",
                json={
                    "role": "observation",
                    "media_type": "image/png",
                    "logical_name": "frame.png",
                    "content_base64": base64.b64encode(b"\x89PNG\r\n\x1a\nframe").decode(),
                },
            )
            artifact.raise_for_status()
            artifact_id = artifact.json()["artifact_id"]

            # The event that rendered that frame says so.
            linked = client.post(
                f"/captures/{capture_id}/events",
                json={
                    "event_type": "frame",
                    "payload": {"step": 0},
                    "artifact_ids": [artifact_id],
                },
            )
            linked.raise_for_status()
            # An event that carries nothing stays unlinked.
            client.post(
                f"/captures/{capture_id}/events",
                json={"event_type": "step", "payload": {"step": 1}},
            ).raise_for_status()

            sealed = client.post(f"/captures/{capture_id}/seal", json={"status": "completed"})
            sealed.raise_for_status()
            bundle_path = sealed.json()["bundle_path"]
    finally:
        service.stop()

    [inspected] = load_bundle(Path(bundle_path))
    document = inspected.trace
    events = {str(event.event_type): event for event in document.events}
    assert events["frame"].artifact_ids == (artifact_id,)
    assert events["step"].artifact_ids == ()
    assert artifact_id in {item.artifact_id for item in document.artifacts}


def test_an_undeclarable_artifact_id_is_refused(tmp_path: Path) -> None:
    service = DetachedCaptureSupervisor(DetachedCaptureConfig(output_root=tmp_path)).start()
    try:
        with httpx.Client(base_url=service.base_url) as client:
            opened = client.post(
                "/captures",
                json={"rollout_id": "rollout-bad", "capture_mode": "required"},
            )
            opened.raise_for_status()
            capture_id = opened.json()["capture_id"]
            refused = client.post(
                f"/captures/{capture_id}/events",
                json={"event_type": "frame", "payload": {}, "artifact_ids": [7]},
            )
            assert refused.status_code == 400
            assert refused.json()["error"]["code"] == "invalid_event"
    finally:
        service.stop()
