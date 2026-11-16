"""Harbor environment whose provider effects remain with Rhodes.

This adapter is qualified separately from the Harbor 0.22 native adapters. Local
task assets stay in the controller; only explicit file transfers reach a sandbox.
"""

from __future__ import annotations

import base64
import io
import re
import tarfile
from pathlib import Path

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities

from .harbor_serve_authority import trial_authority

TRANSFER_BOUND = 16 * 1024 * 1024


def pack_path(path: Path, *, directory: bool) -> str:
    output = io.BytesIO()
    paths = path.rglob("*") if directory else [path]
    with tarfile.open(fileobj=output, mode="w") as archive:
        total = 0
        for index, item in enumerate(paths):
            if index >= 4096:
                raise ValueError("harbor_transfer_file_count_invalid")
            if item.is_symlink() or not (item.is_file() or item.is_dir()):
                raise ValueError("harbor_transfer_special_file_refused")
            total += item.stat().st_size if item.is_file() else 0
            if total > TRANSFER_BOUND:
                raise ValueError("harbor_transfer_too_large")
            archive.add(
                item,
                arcname=str(item.relative_to(path)) if directory else item.name,
                recursive=False,
            )
    if output.tell() > TRANSFER_BOUND:
        raise ValueError("harbor_transfer_too_large")
    return base64.b64encode(output.getvalue()).decode()


def unpack_path(encoded: str, path: Path, *, directory: bool) -> None:
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) > TRANSFER_BOUND:
        raise ValueError("harbor_transfer_too_large")
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        members = archive.getmembers()
        if len(members) > 4096 or sum(member.size for member in members) > TRANSFER_BOUND:
            raise ValueError("harbor_transfer_too_large")
        for member in members:
            if not (member.isfile() or member.isdir()):
                raise ValueError("harbor_transfer_special_file_refused")
            target = (path / member.name).resolve() if directory else path.resolve()
            if directory and not target.is_relative_to(path.resolve()):
                raise ValueError("harbor_transfer_path_refused")
        if directory:
            path.mkdir(parents=True, exist_ok=True)
            archive.extractall(path, filter="data")
        else:
            files = [member for member in members if member.isfile()]
            if len(files) != 1:
                raise ValueError("harbor_transfer_file_count_invalid")
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_symlink():
                raise ValueError("harbor_transfer_symlink_refused")
            source = archive.extractfile(files[0])
            if source is None:
                raise ValueError("harbor_transfer_file_missing")
            with source:
                path.write_bytes(source.read(TRANSFER_BOUND + 1))


class RhodesEnvironment(BaseEnvironment):
    @staticmethod
    def type() -> str:
        return "rhodes"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True)

    def _validate_definition(self) -> None:
        image = self.task_env_config.docker_image or ""
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[a-f0-9]{64}", image):
            raise ValueError("rhodes_harbor_pinned_image_required")
        if self._uses_compose:
            raise ValueError("rhodes_harbor_compose_unsupported")

    async def _call(self, action: str, **payload):
        binding = trial_authority.get()
        return await binding.authority.call(
            "sandbox",
            {
                "native_id": binding.trial_id,
                "sandbox_id": self.session_id,
                "action": action,
                "payload": payload,
            },
        )

    async def start(self, force_build: bool) -> None:
        if force_build:
            raise ValueError("rhodes_harbor_build_not_admitted")
        await self._call(
            "start",
            image=self.task_env_config.docker_image,
            cpu_cores=self._effective_cpus,
            memory_mb=self._effective_memory_mb,
            storage_mb=self._effective_storage_mb,
            network=self.network_policy.model_dump(mode="json"),
            env=self._startup_env(),
        )
        await self.ensure_dirs(self._mount_targets())

    async def stop(self, delete: bool) -> None:
        if not delete:
            raise ValueError("rhodes_harbor_retained_trial_refused")
        await self._call("stop")

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None) -> ExecResult:
        value = await self._call(
            "exec",
            command=command,
            cwd=cwd,
            env=self._merge_env(env),
            timeout_seconds=timeout_sec,
            user=self._resolve_user(user),
        )
        return ExecResult.model_validate(value)

    async def upload_file(self, source_path, target_path):
        await self._call(
            "upload",
            path=target_path,
            directory=False,
            archive=pack_path(Path(source_path), directory=False),
        )

    async def upload_dir(self, source_dir, target_dir):
        await self._call(
            "upload",
            path=target_dir,
            directory=True,
            archive=pack_path(Path(source_dir), directory=True),
        )

    async def download_file(self, source_path, target_path):
        value = await self._call("download", path=source_path, directory=False)
        unpack_path(value["archive"], Path(target_path), directory=False)

    async def download_dir(self, source_dir, target_dir):
        value = await self._call("download", path=source_dir, directory=True)
        unpack_path(value["archive"], Path(target_dir), directory=True)
