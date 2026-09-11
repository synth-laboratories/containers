"""Exercise the pinned SDK seam without network or paid image builds."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("daytona")
from daytona_api_client_async.exceptions import NotFoundException

from synth_containers.harbor_daytona_build_provider import DaytonaSnapshotProvider
from synth_containers.harbor_environment import _tree_digest
from synth_containers.harbor_image_context import HarborImageContext


def provider():
    api = SimpleNamespace(
        create_snapshot=AsyncMock(), get_snapshot=AsyncMock(), remove_snapshot=AsyncMock()
    )
    service = SimpleNamespace(
        _AsyncSnapshotService__snapshots_api=api,
        _AsyncSnapshotService__object_storage_api=object(),
        _AsyncSnapshotService__default_region_id=None,
    )
    return DaytonaSnapshotProvider(SimpleNamespace(snapshot=service)), api


def test_api_creation_returns_without_sdk_polling_and_has_request_deadlines(tmp_path, monkeypatch):
    adapter, api = provider()
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text("FROM scratch\n")
    frozen = HarborImageContext(context, "sha256:" + "a" * 64, _tree_digest(context))
    upload = AsyncMock(return_value=[])
    monkeypatch.setattr(
        "synth_containers.harbor_daytona_build_provider.AsyncSnapshotService.process_image_context",
        upload,
    )
    prepared = asyncio.run(adapter.prepare_context(frozen))
    asyncio.run(adapter.create("owned", prepared))
    request = api.create_snapshot.await_args.args[0]
    assert request.build_info.dockerfile_content == "FROM scratch\n"
    assert request.cpu == 1 and request.memory == 1 and request.disk == 2
    assert api.create_snapshot.await_args.kwargs["_request_timeout"] == 15
    api.get_snapshot.assert_not_awaited()
    assert adapter.is_not_found(NotFoundException())
    assert not adapter.is_not_found(TimeoutError())


def test_copy_outside_frozen_context_never_uploads(tmp_path, monkeypatch):
    adapter, _ = provider()
    context = tmp_path / "context"
    context.mkdir()
    (tmp_path / "outside.txt").write_text("not an authorized build input")
    (context / "Dockerfile").write_text("FROM scratch\nCOPY ../outside.txt /app/\n")
    frozen = HarborImageContext(context, "sha256:" + "a" * 64, _tree_digest(context))
    upload = AsyncMock(return_value=[])
    monkeypatch.setattr(
        "synth_containers.harbor_daytona_build_provider.AsyncSnapshotService.process_image_context",
        upload,
    )
    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(adapter.prepare_context(frozen))
    upload.assert_not_awaited()
