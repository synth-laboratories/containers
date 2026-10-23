"""Bounded, symlink-free package input snapshots (container-package-v1.md)."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

from .package_models import PackageBuildPlan, PackageFileIdentity, portable_path

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_PACKAGE_BYTES = 256 * 1024 * 1024
MAX_ENTRIES = 10000


class ContainerPackageError(ValueError):
    """Stable refusal code without source contents, subprocess output or secrets."""


def _open(root: Path, relative: str) -> int:
    """Walk directory descriptors so a replaced parent cannot redirect a read."""
    portable_path(relative)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parts = relative.split("/")
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
    finally:
        os.close(descriptor)


def read_package_file(root: Path, relative: str, *, maximum: int = MAX_FILE_BYTES) -> bytes:
    try:
        with os.fdopen(_open(root, relative), "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ContainerPackageError("container_package_input_not_regular")
            if info.st_size > maximum:
                raise ContainerPackageError("container_package_input_too_large")
            content = handle.read(maximum + 1)
            if len(content) > maximum:
                raise ContainerPackageError("container_package_input_too_large")
            return content
    except OSError:
        raise ContainerPackageError("container_package_input_unavailable") from None


def snapshot_inputs(
    root: Path, paths: tuple[str, ...]
) -> tuple[tuple[PackageFileIdentity, ...], tuple[str, ...]]:
    files: dict[str, PackageFileIdentity] = {}
    directories: set[str] = set()
    total_bytes = 0

    def visit(relative: str) -> None:
        nonlocal total_bytes
        portable_path(relative)
        if relative in files or relative in directories:
            return
        if len(files) + len(directories) >= MAX_ENTRIES:
            raise ContainerPackageError("container_package_entry_limit")
        try:
            descriptor = _open(root, relative)
        except OSError:
            raise ContainerPackageError("container_package_input_unavailable") from None
        try:
            info = os.fstat(descriptor)
            if stat.S_ISDIR(info.st_mode):
                directories.add(relative)
                # Listing through the owned descriptor never follows a replacement path.
                children = []
                with os.scandir(descriptor) as entries:
                    for entry in entries:
                        if len(files) + len(directories) + len(children) >= MAX_ENTRIES:
                            raise ContainerPackageError("container_package_entry_limit")
                        children.append(entry.name)
                children.sort()
            elif stat.S_ISREG(info.st_mode):
                if info.st_size > min(MAX_FILE_BYTES, MAX_PACKAGE_BYTES - total_bytes):
                    raise ContainerPackageError("container_package_byte_limit")
                with os.fdopen(os.dup(descriptor), "rb") as handle:
                    data = handle.read(min(MAX_FILE_BYTES, MAX_PACKAGE_BYTES - total_bytes) + 1)
                if len(data) > min(MAX_FILE_BYTES, MAX_PACKAGE_BYTES - total_bytes):
                    raise ContainerPackageError("container_package_byte_limit")
                total_bytes += len(data)
                files[relative] = PackageFileIdentity(
                    path=relative,
                    digest="sha256:" + hashlib.sha256(data).hexdigest(),
                    size_bytes=len(data),
                    executable=bool(info.st_mode & 0o111),
                )
                return
            else:
                raise ContainerPackageError("container_package_input_not_regular")
        finally:
            os.close(descriptor)
        for child in children:
            visit(relative + "/" + child)

    for path in paths:
        visit(path)
    return tuple(files[path] for path in sorted(files)), tuple(sorted(directories))


def materialize_inputs(root: Path, plan: PackageBuildPlan, destination: Path) -> None:
    """Copy only acknowledged inputs, refusing a changed file before execution."""
    for directory in plan.directories:
        (destination / directory).mkdir(parents=True, exist_ok=True)
    for entry in plan.files:
        data = read_package_file(root, entry.path)
        if (
            len(data) != entry.size_bytes
            or "sha256:" + hashlib.sha256(data).hexdigest() != entry.digest
        ):
            raise ContainerPackageError("container_package_input_changed")
        target = destination / entry.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(0o755 if entry.executable else 0o644)
