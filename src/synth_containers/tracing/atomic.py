"""Local commit barriers. Callers still own writer fencing and directory custody."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_mkdir(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"refusing symlink directory: {path}")
    if path.is_dir():
        return
    durable_mkdir(path.parent)
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"directory publication conflict: {path}") from None
    sync_directory(path.parent)


def durable_write(path: Path, payload: bytes, *, replace: bool = False) -> bool:
    """Fsync bytes before exposing them; immutable writes never overwrite winners.

    A failed barrier is uncertain, even if the pathname is visible. The owning
    writer must stop and reopen/verify instead of continuing from memory.
    """
    durable_mkdir(path.parent)
    if path.is_symlink():
        raise ValueError(f"refusing symlink publication: {path}")
    fd, temporary = tempfile.mkstemp(prefix=".commit-", dir=path.parent)
    pending = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), 0o600 if replace else 0o400)
            os.fsync(handle.fileno())
        if replace:
            os.replace(pending, path)
            created = True
        else:
            try:
                os.link(pending, path)
                created = True
            except FileExistsError:
                if path.is_symlink() or path.read_bytes() != payload:
                    raise FileExistsError(f"immutable object has different bytes: {path}") from None
                created = False
        sync_directory(path.parent)
        sync_directory(path.parent.parent)
        return created
    finally:
        pending.unlink(missing_ok=True)
