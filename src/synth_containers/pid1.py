"""PID 1 helpers for platform images: baked children, one serve, one down.

An image's ``python -m <package>`` is PID 1 of its container. Anything the task
needs at runtime — a rust engine, a gold sidecar — is a **process child** of that
PID 1 inside the same container, never an operator object and never a second
``synth-containers up``.

Two rules this module enforces:

* No ``start_new_session``. A child in its own session survives ``docker stop``
  and host ``terminate()`` the same way stray rust engines used to.
* Fail closed. A child that never answers its health probe makes ``run_stack``
  raise, so the launcher's ``/health`` wait never sees a half-up container.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ChildFailed",
    "ChildProcess",
    "StartedChild",
    "free_port",
    "probe_http_health",
    "run_stack",
    "start_child",
]


class ChildFailed(RuntimeError):
    """A baked child exited, or never became healthy. The container is not up."""


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def probe_http_health(base_url: str, *, path: str = "/health", timeout: float = 1.0) -> dict[str, Any]:
    """GET a child's health. Raises if it is down. Missing is never ok."""

    url = f"{base_url.rstrip('/')}{path}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            if getattr(response, "status", 200) >= 400:
                raise ChildFailed(f"child_unhealthy:{getattr(response, 'status', 0)}")
            body = response.read().decode("utf-8") or "{}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ChildFailed(f"child_unreachable:{url}") from exc
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


@dataclass(frozen=True, slots=True)
class ChildProcess:
    """A baked executable PID 1 starts and owns.

    ``argv`` may contain ``{host}`` and ``{port}`` placeholders; the port is
    allocated here unless pinned. ``url_env`` is the variable the façade reads to
    reach this child — it is exported into ``os.environ`` before the app is built.
    """

    name: str
    argv: Sequence[str]
    url_env: str | None = None
    health_path: str | None = "/health"
    host: str = "127.0.0.1"
    port: int | None = None
    cwd: Path | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    startup_timeout_seconds: float = 60.0
    stop_timeout_seconds: float = 5.0

    def resolve_argv(self, port: int) -> list[str]:
        return [str(item).format(host=self.host, port=port) for item in self.argv]


@dataclass(slots=True)
class StartedChild:
    name: str
    url: str | None
    port: int
    process: subprocess.Popen[str]
    stop: Callable[[], None]

    def alive(self) -> bool:
        return self.process.poll() is None


def _stopper(process: subprocess.Popen[str], *, timeout: float) -> Callable[[], None]:
    def stop() -> None:
        if process.poll() is not None:
            return
        try:
            process.terminate()
        except (ProcessLookupError, OSError):
            return
        deadline = time.monotonic() + timeout
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if process.poll() is None:
            try:
                process.kill()
            except (ProcessLookupError, OSError):
                pass
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass

    return stop


def start_child(child: ChildProcess) -> StartedChild:
    """Spawn one baked child in PID 1's process group and wait for it."""

    port = int(child.port or free_port(child.host))
    argv = child.resolve_argv(port)
    executable = argv[0]
    if "/" in executable and not os.access(executable, os.X_OK):
        raise ChildFailed(f"child_binary_missing:{child.name}")
    env = {**os.environ, **{str(k): str(v) for k, v in child.env.items()}}
    # No start_new_session: the child must die with PID 1's pid namespace.
    process = subprocess.Popen(  # noqa: S603
        argv,
        cwd=str(child.cwd) if child.cwd is not None else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    stop = _stopper(process, timeout=child.stop_timeout_seconds)
    url = f"http://{child.host}:{port}" if child.health_path is not None else None

    threading.Thread(
        target=_pump_logs, args=(child.name, process), name=f"child-logs:{child.name}", daemon=True
    ).start()

    if child.health_path is None:
        time.sleep(0.1)
        if process.poll() is not None:
            raise ChildFailed(f"child_exited:{child.name}:{process.returncode}")
        return StartedChild(name=child.name, url=None, port=port, process=process, stop=stop)

    deadline = time.monotonic() + child.startup_timeout_seconds
    last = "unhealthy"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stop()
            raise ChildFailed(f"child_exited:{child.name}:{process.returncode}")
        try:
            probe_http_health(url or "", path=child.health_path, timeout=0.5)
            return StartedChild(name=child.name, url=url, port=port, process=process, stop=stop)
        except ChildFailed as exc:
            last = str(exc).split(":", 1)[0]
            time.sleep(0.1)
    stop()
    raise ChildFailed(f"child_unhealthy:{child.name}:{last}")


def _pump_logs(name: str, process: subprocess.Popen[str]) -> None:
    stream = process.stdout
    if stream is None:
        return
    for line in stream:
        sys.stderr.write(f"[{name}] {line.rstrip()}\n")
        sys.stderr.flush()


def run_stack(
    *,
    children: Sequence[ChildProcess],
    app_factory: Callable[[], Any],
    host: str | None = None,
    port: int | None = None,
    image_id: str = "",
    startup_timeout_seconds: float = 120.0,
    block: bool = True,
) -> int:
    """Start children, serve the façade, block until SIGTERM, take everything down.

    Returns the process exit code. Children stop in reverse start order.
    """

    from .sdk import ContainerRunner

    bind_host = host or os.environ.get("SYNTH_CONTAINER_HOST") or "127.0.0.1"
    bind_port = int(port or os.environ.get("SYNTH_CONTAINER_PORT") or 8080)

    started: list[StartedChild] = []
    restore_env: dict[str, str | None] = {}

    def _teardown() -> None:
        for child in reversed(started):
            child.stop()
        for key, value in restore_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    try:
        for spec in children:
            child = start_child(spec)
            started.append(child)
            if spec.url_env and child.url:
                restore_env[spec.url_env] = os.environ.get(spec.url_env)
                os.environ[spec.url_env] = child.url
        handle = ContainerRunner(
            app=app_factory(),
            host=bind_host,
            port=bind_port,
            startup_timeout_seconds=startup_timeout_seconds,
        ).serve()
    except Exception:
        _teardown()
        raise

    payload = {
        "url": handle.url,
        "image_id": image_id,
        "children": {child.name: child.url for child in started},
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)

    if not block:
        handle.down()
        _teardown()
        return 0

    _block_until_signal(handle, started)
    handle.down()
    _teardown()
    return 0


def _block_until_signal(handle: "Any", started: Sequence[StartedChild]) -> None:
    """Wait for SIGTERM/SIGINT, or for any child to die (fail closed)."""

    done = threading.Event()

    def _on_signal(_signum: int, _frame: object) -> None:
        done.set()

    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(signum, _on_signal)
        except (ValueError, OSError):
            pass
    del handle
    try:
        while not done.wait(timeout=1.0):
            dead = [child.name for child in started if not child.alive()]
            if dead:
                sys.stderr.write(f"[pid1] child exited, taking the container down: {dead}\n")
                return
    except KeyboardInterrupt:
        return
