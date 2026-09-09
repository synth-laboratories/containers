from __future__ import annotations

import importlib

from synth_containers.platform.targets import TARGETS


def test_image_main_passes_environment_storage_root_to_compat_app(monkeypatch, tmp_path) -> None:
    image_main = importlib.import_module("synth_containers.image_main")
    platform = importlib.import_module("synth_containers.platform")
    observed = {}

    def create(spec, *, storage_root=None, runtime_config=None):
        observed["spec"] = spec
        observed["storage_root"] = storage_root
        return object()

    def run_stack(*, app_factory, **kwargs):
        observed["app"] = app_factory()
        return 0

    monkeypatch.setenv("SYNTH_CONTAINER_STORAGE", str(tmp_path))
    monkeypatch.setattr(platform, "create_compat_app", create)
    monkeypatch.setattr(image_main, "run_stack", run_stack)

    assert image_main.image_main(
        image_id="test-image",
        targets={"openenv_echo": TARGETS["openenv_echo"]},
        default_target="openenv_echo",
        argv=[],
    ) == 0
    assert observed["storage_root"] == str(tmp_path)


def test_image_main_explicit_storage_root_overrides_environment(monkeypatch, tmp_path) -> None:
    image_main = importlib.import_module("synth_containers.image_main")
    platform = importlib.import_module("synth_containers.platform")
    observed = {}
    explicit = tmp_path / "explicit"

    monkeypatch.setenv("SYNTH_CONTAINER_STORAGE", str(tmp_path / "environment"))
    monkeypatch.setattr(
        platform,
        "create_compat_app",
        lambda spec, *, storage_root=None, runtime_config=None: observed.setdefault(
            "storage_root", storage_root
        ),
    )
    monkeypatch.setattr(
        image_main,
        "run_stack",
        lambda *, app_factory, **kwargs: (app_factory(), 0)[1],
    )

    assert image_main.image_main(
        image_id="test-image",
        targets={"openenv_echo": TARGETS["openenv_echo"]},
        default_target="openenv_echo",
        argv=["--storage-root", str(explicit)],
    ) == 0
    assert observed["storage_root"] == str(explicit)
