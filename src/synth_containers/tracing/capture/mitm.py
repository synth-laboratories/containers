"""Opt-in, provider-scoped TLS interception for Trace V5 capture.

``ScopedMitmProxy`` owns one external ``mitmdump`` process and one ephemeral CA
directory. It does not alter machine trust. Only a public, combined trust bundle is
exposed to the workload, and the entire owned directory (including private CA
material) is removed when capture stops.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
import re
import shutil
import socket
import ssl
import subprocess
import tempfile
import time
from typing import Any, Sequence
from urllib.parse import urlparse

import certifi

from synth_containers.serde import JsonDataclassMixin

from ..canonical import (
    bytes_digest,
    canonical_bytes,
    content_digest,
    record_id,
    utc_now,
)
from .mitmproxy_addon import (
    MITM_ADDON_CONFIG_ENV,
    MITM_ADDON_CONFIG_SCHEMA_VERSION,
)
from .routes import ProviderEndpointConfig


MITM_LIFECYCLE_SCHEMA_VERSION = "synth.trace-mitm-lifecycle-receipt.v1"
_OWNERSHIP_FILE = ".synth-trace-mitm-owner"
_CONFIG_FILE = "synth-trace-mitm-config.json"
_EVENT_LOG_FILE = "synth-trace-mitm-events.jsonl"
_PROCESS_LOG_FILE = "synth-trace-mitmdump.log"
_PUBLIC_CA_FILE = "mitmproxy-ca-cert.pem"
_COMBINED_CA_FILE = "synth-trace-public-ca-bundle.pem"
_PRIVATE_CA_NAMES = ("mitmproxy-ca.pem", "mitmproxy-ca.p12")


class MitmStartupError(RuntimeError):
    """Raised when the selected MITM interception cannot prove readiness."""


@dataclass(frozen=True, slots=True)
class MitmRouteV1(JsonDataclassMixin):
    provider_host: str
    provider_port: int
    provider_path: str
    capture_route: str


@dataclass(frozen=True, slots=True)
class MitmLifecycleReceiptV1(JsonDataclassMixin):
    receipt_id: str
    capture_id: str
    allowed_hosts: tuple[str, ...]
    allowed_authorities: tuple[str, ...]
    allowlist_regex: str
    route_count: int
    addon_digest: str
    config_digest: str = ""
    started_at: str | None = None
    ready_at: str | None = None
    stopped_at: str | None = None
    readiness_ok: bool = False
    readiness_detail: str = "not started"
    public_ca_digest: str | None = None
    public_trust_bundle_digest: str | None = None
    private_key_material_observed: bool = False
    private_key_names: tuple[str, ...] = ()
    private_key_destroyed: bool = False
    confdir_destroyed: bool = False
    routed_requests: int = 0
    unmapped_provider_requests: int = 0
    unexpected_tls_hosts: int = 0
    malformed_addon_events: int = 0
    process_exit_code: int | None = None
    process_stopped: bool = False
    process_exited_before_stop: bool = False
    stop_reason: str | None = None
    failure: str | None = None
    schema_version: str = MITM_LIFECYCLE_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    content_digest: str = ""

    def sealed(self) -> "MitmLifecycleReceiptV1":
        return replace(self, content_digest=content_digest(self))


class ScopedMitmProxy:
    """Own one allowlisted regular-mode mitmdump process and ephemeral CA."""

    def __init__(
        self,
        *,
        capture_id: str,
        endpoints: Sequence[ProviderEndpointConfig],
        capture_proxy_host: str,
        capture_proxy_port: int,
        command: Sequence[str] = ("mitmdump",),
        host: str = "127.0.0.1",
        port: int = 0,
        allowed_hosts: Sequence[str] | None = None,
        temp_root: Path | None = None,
        base_ca_bundle_path: Path | None = None,
        startup_timeout: float = 10.0,
    ) -> None:
        if not command or not str(command[0]).strip():
            raise ValueError("scoped MITM command must not be empty")
        if startup_timeout <= 0:
            raise ValueError("scoped MITM startup timeout must be positive")
        if not (0 <= int(port) <= 65535):
            raise ValueError("scoped MITM listen port is invalid")
        if not (1 <= int(capture_proxy_port) <= 65535):
            raise ValueError("capture proxy port is invalid")
        self.capture_id = capture_id
        self.command = tuple(str(item) for item in command)
        self.host = str(host)
        self.port = int(port)
        self.capture_proxy_host = _loopback_for_wildcard(capture_proxy_host)
        self.capture_proxy_port = int(capture_proxy_port)
        self.temp_root = Path(temp_root).resolve() if temp_root is not None else None
        self.base_ca_bundle_path = (
            Path(base_ca_bundle_path).resolve()
            if base_ca_bundle_path is not None
            else None
        )
        self.startup_timeout = float(startup_timeout)
        all_routes = _routes_from_endpoints(endpoints)
        derived_hosts = tuple(sorted({item.provider_host for item in all_routes}))
        selected_hosts = (
            tuple(sorted({_normalize_host(item) for item in allowed_hosts}))
            if allowed_hosts is not None
            else derived_hosts
        )
        if not selected_hosts:
            raise ValueError("scoped MITM provider allowlist must not be empty")
        undeclared = tuple(item for item in selected_hosts if item not in derived_hosts)
        if undeclared:
            raise ValueError(
                "scoped MITM allowlist contains hosts without configured provider "
                f"endpoints: {', '.join(undeclared)}"
            )
        self.allowed_hosts = selected_hosts
        self.routes = tuple(
            item for item in all_routes if item.provider_host in set(selected_hosts)
        )
        if not self.routes:
            raise ValueError("scoped MITM allowlist selected no provider routes")
        self.allowed_authorities = tuple(
            sorted({f"{item.provider_host}:{item.provider_port}" for item in self.routes})
        )
        self.allowlist_regex = _allowlist_regex(self.routes)
        addon_path = Path(__file__).with_name("mitmproxy_addon.py").resolve()
        self.addon_path = addon_path
        addon_digest = bytes_digest(addon_path.read_bytes())
        receipt_id = record_id(
            "rcpt",
            kind="trace_mitm_lifecycle",
            scope=(capture_id,),
            key={
                "allowed_authorities": self.allowed_authorities,
                "routes": [item.to_dict() for item in self.routes],
                "addon_digest": addon_digest,
            },
        )
        self._receipt = MitmLifecycleReceiptV1(
            receipt_id=receipt_id,
            capture_id=capture_id,
            allowed_hosts=self.allowed_hosts,
            allowed_authorities=self.allowed_authorities,
            allowlist_regex=self.allowlist_regex,
            route_count=len(self.routes),
            addon_digest=addon_digest,
        )
        self._process: subprocess.Popen[bytes] | None = None
        self._process_log_handle: Any = None
        self._confdir: Path | None = None
        self._owner_token: str | None = None
        self._event_log: Path | None = None
        self._process_log: Path | None = None
        self._public_ca: Path | None = None
        self._public_trust_bundle: Path | None = None
        self._started = False
        self._stopped = False

    @property
    def ready(self) -> bool:
        return bool(
            self._receipt.readiness_ok
            and self._process is not None
            and self._process.poll() is None
            and self._public_trust_bundle is not None
            and self._public_trust_bundle.is_file()
        )

    @property
    def proxy_url(self) -> str:
        return self.proxy_url_for(None)

    def proxy_url_for(self, visible_host: str | None) -> str:
        host = visible_host or _loopback_for_wildcard(self.host)
        if self.port <= 0:
            raise MitmStartupError("scoped MITM has not allocated a listen port")
        return f"http://{host}:{self.port}"

    @property
    def public_ca_path(self) -> Path:
        """The public-only combined trust bundle to expose to a child."""

        if self._public_trust_bundle is None or not self._public_trust_bundle.is_file():
            raise MitmStartupError("scoped MITM public trust bundle is unavailable")
        return self._public_trust_bundle

    @property
    def authority_certificate_path(self) -> Path:
        if self._public_ca is None or not self._public_ca.is_file():
            raise MitmStartupError("scoped MITM public authority certificate is unavailable")
        return self._public_ca

    @property
    def confdir(self) -> Path | None:
        """Internal test/diagnostic surface; never inject this path into a workload."""

        return self._confdir

    def trust_mount(self, target: str) -> dict[str, Any]:
        if not target or not str(target).startswith("/"):
            raise ValueError("container MITM CA target must be an absolute path")
        return {
            "source": str(self.public_ca_path),
            "target": str(target),
            "read_only": True,
            "content_digest": self._receipt.public_trust_bundle_digest,
        }

    def lifecycle_receipt(self) -> MitmLifecycleReceiptV1:
        return self._receipt.sealed()

    def start(self) -> "ScopedMitmProxy":
        if self._stopped:
            raise MitmStartupError("a stopped scoped MITM cannot be restarted")
        if self._started:
            if not self.ready:
                raise MitmStartupError("scoped MITM process lost readiness")
            return self
        self._started = True
        self._receipt = replace(self._receipt, started_at=utc_now())
        try:
            executable = _resolve_executable(self.command[0])
            self._prepare_owned_directory()
            if self.port == 0:
                self.port = _reserve_port(self.host)
            config_path = self._write_addon_config()
            args = (
                executable,
                *self.command[1:],
                "--quiet",
                "--mode",
                "regular",
                "--listen-host",
                self.host,
                "--listen-port",
                str(self.port),
                "--allow-hosts",
                self.allowlist_regex,
                "--set",
                f"confdir={self._confdir}",
                "--set",
                "connection_strategy=lazy",
                "-s",
                str(self.addon_path),
            )
            if self._process_log is None:
                raise RuntimeError("scoped MITM process log was not prepared")
            self._process_log_handle = self._process_log.open("ab", buffering=0)
            process_environment = _mitmdump_environment()
            process_environment[MITM_ADDON_CONFIG_ENV] = str(config_path)
            self._process = subprocess.Popen(
                args,
                env=process_environment,
                stdin=subprocess.DEVNULL,
                stdout=self._process_log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self._await_readiness()
            self._receipt = replace(
                self._receipt,
                readiness_ok=True,
                readiness_detail=(
                    "mitmdump TCP listener, addon, ephemeral CA, and public trust "
                    "bundle are ready"
                ),
                ready_at=utc_now(),
            )
            return self
        except BaseException as exc:
            detail = f"{type(exc).__name__}: {exc}"
            self._receipt = replace(
                self._receipt,
                readiness_ok=False,
                readiness_detail=detail,
                failure=detail,
            )
            self.stop(reason="startup_failed")
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise MitmStartupError(detail) from exc

    def stop(self, *, reason: str = "normal") -> MitmLifecycleReceiptV1:
        if self._stopped:
            return self.lifecycle_receipt()
        self._stopped = True
        process_exited_before_stop = False
        process_stopped = self._process is None
        exit_code: int | None = None
        stop_error: str | None = None
        if self._process is not None:
            process_exited_before_stop = self._process.poll() is not None
            try:
                if self._process.poll() is None:
                    try:
                        self._process.terminate()
                    except ProcessLookupError:
                        process_exited_before_stop = True
                    try:
                        self._process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        try:
                            self._process.kill()
                        except ProcessLookupError:
                            process_exited_before_stop = True
                        self._process.wait(timeout=5.0)
            except BaseException as exc:
                stop_error = f"mitmdump shutdown raised {type(exc).__name__}: {exc}"
            exit_code = self._process.returncode
            process_stopped = self._process.poll() is not None
        if self._process_log_handle is not None:
            try:
                self._process_log_handle.close()
            except OSError as exc:
                stop_error = stop_error or (
                    f"mitmdump log close raised {type(exc).__name__}: {exc}"
                )
            self._process_log_handle = None
        event_counts, malformed_events = self._read_event_counts()
        private_names = self._private_key_names()
        private_observed = bool(private_names)
        destroyed = self._destroy_owned_directory()
        failure = self._receipt.failure
        if stop_error:
            failure = failure or stop_error
        if not process_stopped:
            failure = failure or "mitmdump remained alive after shutdown escalation"
        if process_exited_before_stop and self._receipt.readiness_ok:
            failure = failure or "mitmdump exited before capture finalization"
        if not destroyed:
            failure = failure or "ephemeral MITM CA directory could not be destroyed"
        self._receipt = replace(
            self._receipt,
            stopped_at=utc_now(),
            stop_reason=reason,
            process_exit_code=exit_code,
            process_stopped=process_stopped,
            process_exited_before_stop=process_exited_before_stop,
            private_key_material_observed=private_observed,
            private_key_names=private_names,
            private_key_destroyed=destroyed,
            confdir_destroyed=destroyed,
            routed_requests=event_counts.get("provider_routed", 0),
            unmapped_provider_requests=event_counts.get("unmapped_provider_route", 0),
            unexpected_tls_hosts=event_counts.get("unexpected_tls_host", 0),
            malformed_addon_events=malformed_events,
            failure=failure,
        )
        return self.lifecycle_receipt()

    def _prepare_owned_directory(self) -> None:
        if self.temp_root is not None:
            self.temp_root.mkdir(parents=True, exist_ok=True)
            directory = Path(
                tempfile.mkdtemp(prefix="synth-trace-mitm-", dir=self.temp_root)
            )
        else:
            directory = Path(tempfile.mkdtemp(prefix="synth-trace-mitm-"))
        directory = directory.resolve()
        directory.chmod(0o700)
        owner_token = os.urandom(32).hex()
        owner_path = directory / _OWNERSHIP_FILE
        owner_path.write_text(owner_token, encoding="utf-8")
        owner_path.chmod(0o400)
        event_log = directory / _EVENT_LOG_FILE
        event_log.touch(mode=0o600)
        process_log = directory / _PROCESS_LOG_FILE
        process_log.touch(mode=0o600)
        self._confdir = directory
        self._owner_token = owner_token
        self._event_log = event_log
        self._process_log = process_log
        self._public_ca = directory / _PUBLIC_CA_FILE
        self._public_trust_bundle = directory / _COMBINED_CA_FILE

    def _write_addon_config(self) -> Path:
        if self._confdir is None or self._event_log is None:
            raise RuntimeError("scoped MITM directory was not prepared")
        base_payload: dict[str, Any] = {
            "schema_version": MITM_ADDON_CONFIG_SCHEMA_VERSION,
            "capture_id": self.capture_id,
            "allowed_hosts": list(self.allowed_hosts),
            "allowed_authorities": list(self.allowed_authorities),
            "routes": [item.to_dict() for item in self.routes],
            "capture_proxy": {
                "host": self.capture_proxy_host,
                "port": self.capture_proxy_port,
            },
            "event_log": str(self._event_log),
        }
        config_digest = content_digest(base_payload)
        payload = {**base_payload, "config_digest": config_digest}
        path = self._confdir / _CONFIG_FILE
        path.write_bytes(canonical_bytes(payload))
        path.chmod(0o400)
        self._receipt = replace(self._receipt, config_digest=config_digest)
        return path

    def _await_readiness(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._process is None:
                raise RuntimeError("mitmdump process was not created")
            exit_code = self._process.poll()
            if exit_code is not None:
                raise RuntimeError(f"mitmdump exited during startup with code {exit_code}")
            addon_ready = self._addon_ready()
            public_ready = bool(self._public_ca and self._public_ca.is_file())
            private_ready = self._private_key_material_ready()
            listener_ready = _proxy_chain_ready(
                self.host,
                self.port,
                self.capture_proxy_host,
                self.capture_proxy_port,
            )
            if addon_ready and public_ready and private_ready and listener_ready:
                self._prepare_public_trust_bundle()
                return
            time.sleep(0.05)
        raise TimeoutError(
            "mitmdump readiness timed out before listener/addon/CA proof completed"
        )

    def _addon_ready(self) -> bool:
        if self._event_log is None:
            return False
        for payload in _read_json_lines(self._event_log)[0]:
            if (
                payload.get("event") == "addon_ready"
                and payload.get("config_digest") == self._receipt.config_digest
            ):
                return True
        return False

    def _prepare_public_trust_bundle(self) -> None:
        if self._public_ca is None or self._public_trust_bundle is None:
            raise RuntimeError("scoped MITM CA paths were not prepared")
        public_body = self._public_ca.read_bytes()
        if b"BEGIN CERTIFICATE" not in public_body or b"PRIVATE KEY" in public_body:
            raise RuntimeError("mitmdump public CA file is missing or contains private material")
        base_path = self.base_ca_bundle_path or _default_ca_bundle_path()
        if base_path is None or not base_path.is_file():
            raise RuntimeError(
                "a public base CA bundle is required so non-provider TLS can pass through"
            )
        base_body = base_path.read_bytes()
        if b"BEGIN CERTIFICATE" not in base_body or b"PRIVATE KEY" in base_body:
            raise RuntimeError(
                "base CA bundle is missing public certificates or contains private "
                "key material"
            )
        combined = base_body.rstrip() + b"\n" + public_body.lstrip()
        self._public_trust_bundle.write_bytes(combined)
        self._public_trust_bundle.chmod(0o444)
        self._public_ca.chmod(0o444)
        self._receipt = replace(
            self._receipt,
            public_ca_digest=bytes_digest(public_body),
            public_trust_bundle_digest=bytes_digest(combined),
        )

    def _private_key_names(self) -> tuple[str, ...]:
        if self._confdir is None or not self._confdir.is_dir():
            return ()
        names: list[str] = []
        for name in _PRIVATE_CA_NAMES:
            path = self._confdir / name
            if path.is_file():
                names.append(name)
        return tuple(sorted(names))

    def _private_key_material_ready(self) -> bool:
        if self._confdir is None:
            return False
        pem_path = self._confdir / "mitmproxy-ca.pem"
        if pem_path.is_file():
            try:
                if b"PRIVATE KEY" in pem_path.read_bytes():
                    return True
            except OSError:
                return False
        p12_path = self._confdir / "mitmproxy-ca.p12"
        try:
            return p12_path.is_file() and p12_path.stat().st_size > 0
        except OSError:
            return False

    def _read_event_counts(self) -> tuple[dict[str, int], int]:
        if self._event_log is None:
            return {}, 0
        payloads, malformed = _read_json_lines(self._event_log)
        counts: dict[str, int] = {}
        for payload in payloads:
            event = str(payload.get("event") or "")
            if event:
                counts[event] = counts.get(event, 0) + 1
        return counts, malformed

    def _destroy_owned_directory(self) -> bool:
        directory = self._confdir
        if directory is None:
            return True
        try:
            if not directory.exists():
                return True
            if (
                directory.is_symlink()
                or not directory.is_dir()
                or not directory.name.startswith("synth-trace-mitm-")
            ):
                return False
            owner_path = directory / _OWNERSHIP_FILE
            if (
                self._owner_token is None
                or not owner_path.is_file()
                or owner_path.read_text(encoding="utf-8") != self._owner_token
            ):
                return False
            for name in _PRIVATE_CA_NAMES:
                path = directory / name
                if path.is_file() and not path.is_symlink():
                    path.chmod(0o600)
                    path.unlink()
            shutil.rmtree(directory)
            return not directory.exists()
        except OSError:
            return False

def _routes_from_endpoints(
    endpoints: Sequence[ProviderEndpointConfig],
) -> tuple[MitmRouteV1, ...]:
    routes: dict[tuple[str, int, str], MitmRouteV1] = {}
    for endpoint in endpoints:
        parsed = urlparse(endpoint.upstream_url())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(
                f"provider endpoint {endpoint.route!r} has an invalid upstream URL"
            )
        host = _normalize_host(parsed.hostname)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        provider_path = parsed.path or "/"
        route = MitmRouteV1(
            provider_host=host,
            provider_port=port,
            provider_path=provider_path,
            capture_route=_normalize_path(endpoint.route),
        )
        key = (host, port, provider_path)
        existing = routes.get(key)
        if existing is not None and existing.capture_route != route.capture_route:
            raise ValueError(
                "provider endpoints map one upstream authority/path to multiple "
                "capture routes"
            )
        routes[key] = route
    return tuple(
        sorted(
            routes.values(),
            key=lambda item: (
                item.provider_host,
                item.provider_port,
                item.provider_path,
                item.capture_route,
            ),
        )
    )


def _allowlist_regex(routes: Sequence[MitmRouteV1]) -> str:
    by_host: dict[str, set[int]] = {}
    for route in routes:
        by_host.setdefault(route.provider_host, set()).add(route.provider_port)
    patterns: list[str] = []
    for host, ports in sorted(by_host.items()):
        port_pattern = "|".join(str(item) for item in sorted(ports))
        patterns.append(rf"{re.escape(host)}(?::(?:{port_pattern}))?")
    return r"^(?:" + "|".join(patterns) + r")$"


def _resolve_executable(command: str) -> str:
    value = str(command)
    if os.sep in value:
        path = Path(value)
        if not path.is_file() or not os.access(path, os.X_OK):
            raise FileNotFoundError(f"scoped MITM executable is unavailable: {value}")
        return str(path.resolve())
    resolved = shutil.which(value)
    if not resolved:
        raise FileNotFoundError(f"scoped MITM executable is unavailable: {value}")
    return resolved


def _reserve_port(host: str) -> int:
    bind_host = _loopback_for_wildcard(host) if host in {"localhost"} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((bind_host, 0))
        return int(listener.getsockname()[1])


def _proxy_chain_ready(
    listen_host: str,
    listen_port: int,
    capture_proxy_host: str,
    capture_proxy_port: int,
) -> bool:
    target = _loopback_for_wildcard(listen_host)
    request_target = (
        f"http://{capture_proxy_host}:{capture_proxy_port}/healthz"
    )
    request = (
        f"GET {request_target} HTTP/1.1\r\n"
        f"Host: {capture_proxy_host}:{capture_proxy_port}\r\n"
        "Connection: close\r\n\r\n"
    ).encode("ascii")
    try:
        with socket.create_connection((target, listen_port), timeout=0.5) as connection:
            connection.settimeout(0.5)
            connection.sendall(request)
            response = bytearray()
            while len(response) < 8192:
                chunk = connection.recv(8192 - len(response))
                if not chunk:
                    break
                response.extend(chunk)
        header, _, body = bytes(response).partition(b"\r\n\r\n")
        return (
            header.startswith(b"HTTP/1.1 200")
            or header.startswith(b"HTTP/1.0 200")
        ) and b"synth-trace-proxy/1" in body
    except (OSError, UnicodeError):
        return False


def _loopback_for_wildcard(host: str) -> str:
    return "127.0.0.1" if str(host) in {"0.0.0.0", "::", ""} else str(host)


def _normalize_host(value: str) -> str:
    normalized = str(value).strip().lower().rstrip(".")
    if not normalized:
        raise ValueError("provider host must not be empty")
    if "/" in normalized or "\\" in normalized or any(char.isspace() for char in normalized):
        raise ValueError("provider host is invalid")
    return normalized


def _normalize_path(value: str) -> str:
    path = str(value).strip()
    return path if path.startswith("/") else f"/{path}"


def _default_ca_bundle_path() -> Path | None:
    candidates = [
        ssl.get_default_verify_paths().cafile,
        os.environ.get("SSL_CERT_FILE"),
        "/etc/ssl/cert.pem",
        "/etc/ssl/certs/ca-certificates.crt",
        certifi.where(),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()
    return None


def _mitmdump_environment() -> dict[str, str]:
    """Pass only process-runtime variables, never ambient provider credentials."""

    allowed = (
        "PATH",
        "SYSTEMROOT",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PYTHONUTF8",
        "PYTHONIOENCODING",
    )
    return {name: os.environ[name] for name in allowed if name in os.environ}


def _read_json_lines(path: Path) -> tuple[list[dict[str, Any]], int]:
    if not path.is_file():
        return [], 0
    payloads: list[dict[str, Any]] = []
    malformed = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return [], 1
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
        else:
            malformed += 1
    return payloads, malformed


__all__ = [
    "MITM_LIFECYCLE_SCHEMA_VERSION",
    "MitmLifecycleReceiptV1",
    "MitmRouteV1",
    "MitmStartupError",
    "ScopedMitmProxy",
]
