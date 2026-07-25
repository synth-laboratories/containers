from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from synth_containers.tracing.capture.binding import CaptureMode, Interception
from synth_containers.tracing.capture.egress import mitm_environment
from synth_containers.tracing.capture.mitm import MitmStartupError, ScopedMitmProxy
from synth_containers.tracing.capture.mitmproxy_addon import (
    AddonConfig,
    AddonRoute,
    ScopedProviderAddon,
)
from synth_containers.tracing.capture.routes import ProviderEndpointConfig
from synth_containers.tracing.capture.supervisor import (
    CaptureNotReady,
    CaptureSupervisor,
    SupervisorConfig,
)
from synth_containers.tracing.models.identity import TraceProvenanceV5


def _endpoint() -> ProviderEndpointConfig:
    return ProviderEndpointConfig(
        route="/v1/responses",
        adapter_name="openai_responses",
        upstream_base_url="https://api.openai.com/v1",
        upstream_path="/responses",
    )


def _public_base_ca(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "-----BEGIN CERTIFICATE-----\nPUBLIC-BASE\n-----END CERTIFICATE-----\n",
        encoding="utf-8",
    )
    return path


def _fake_mitmdump(tmp_path: Path) -> tuple[tuple[str, ...], Path]:
    script = tmp_path / "fake_mitmdump.py"
    invocation = tmp_path / "fake_mitmdump_invocation.json"
    script.write_text(
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        "import socket\n"
        "import sys\n"
        "capture_path = Path(sys.argv[1])\n"
        "args = sys.argv[2:]\n"
        "host = args[args.index('--listen-host') + 1]\n"
        "port = int(args[args.index('--listen-port') + 1])\n"
        "confdir_value = next(item for item in args if item.startswith('confdir='))\n"
        "confdir = Path(confdir_value.split('=', 1)[1])\n"
        "config_path = Path(os.environ['SYNTH_TRACE_MITM_CONFIG'])\n"
        "config = json.loads(config_path.read_text(encoding='utf-8'))\n"
        "(confdir / 'mitmproxy-ca-cert.pem').write_text(\n"
        "    '-----BEGIN CERTIFICATE-----\\nPUBLIC-MITM\\n-----END CERTIFICATE-----\\n',\n"
        "    encoding='utf-8',\n"
        ")\n"
        "(confdir / 'mitmproxy-ca.pem').write_text(\n"
        "    '-----BEGIN PRIVATE KEY-----\\nPRIVATE\\n-----END PRIVATE KEY-----\\n',\n"
        "    encoding='utf-8',\n"
        ")\n"
        "with Path(config['event_log']).open('a', encoding='utf-8') as handle:\n"
        "    handle.write(json.dumps({\n"
        "        'event': 'addon_ready',\n"
        "        'config_digest': config['config_digest'],\n"
        "    }, sort_keys=True) + '\\n')\n"
        "capture_path.write_text(json.dumps({\n"
        "    'args': args,\n"
        "    'config': config,\n"
        "    'config_env': str(config_path),\n"
        "    'provider_credential_present': 'OPENAI_API_KEY' in os.environ,\n"
        "}, sort_keys=True), encoding='utf-8')\n"
        "listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
        "listener.bind((host, port))\n"
        "listener.listen(8)\n"
        "while True:\n"
        "    connection, _ = listener.accept()\n"
        "    request = connection.recv(8192)\n"
        "    if b'/healthz' in request:\n"
        "        body = b'{\"proxy\":\"synth-trace-proxy/1\"}'\n"
        "        connection.sendall(\n"
        "            b'HTTP/1.1 200 OK\\r\\nContent-Length: ' +\n"
        "            str(len(body)).encode('ascii') +\n"
        "            b'\\r\\nConnection: close\\r\\n\\r\\n' + body\n"
        "        )\n"
        "    connection.close()\n",
        encoding="utf-8",
    )
    return (sys.executable, str(script), str(invocation)), invocation


def _supervisor_config(
    tmp_path: Path,
    *,
    mode: CaptureMode | str = CaptureMode.REQUIRED,
    interception: Interception | str = Interception.PROVIDER_PROXY,
    command: tuple[str, ...] = ("mitmdump",),
) -> SupervisorConfig:
    return SupervisorConfig(
        bundle_root=tmp_path / "bundle",
        trace_key={"task": "scoped-mitm"},
        upstream_base_url="https://api.openai.com/v1",
        provenance=TraceProvenanceV5(producer="test", producer_version="1"),
        mode=mode,
        interception=interception,
        provider_endpoints=(_endpoint(),),
        mitmdump_command=command,
        mitm_temp_root=tmp_path / "mitm-temp",
        mitm_base_ca_bundle_path=_public_base_ca(tmp_path / "base-ca.pem"),
    )


def test_scoped_mitm_process_readiness_args_and_private_ca_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-mitmdump")
    command, invocation_path = _fake_mitmdump(tmp_path)
    mitm = ScopedMitmProxy(
        capture_id="capture-test",
        endpoints=(_endpoint(),),
        capture_proxy_host="127.0.0.1",
        capture_proxy_port=43123,
        command=command,
        temp_root=tmp_path / "owned",
        base_ca_bundle_path=_public_base_ca(tmp_path / "base-ca.pem"),
        startup_timeout=5,
    )

    mitm.start()
    confdir = mitm.confdir
    public_bundle = mitm.public_ca_path
    private_key = confdir / "mitmproxy-ca.pem" if confdir is not None else None
    assert mitm.ready
    assert confdir is not None and confdir.is_dir()
    assert public_bundle.is_file()
    assert private_key is not None and private_key.is_file()
    assert b"PRIVATE KEY" not in public_bundle.read_bytes()

    invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
    assert invocation["provider_credential_present"] is False
    assert invocation["config"]["allowed_hosts"] == ["api.openai.com"]
    assert invocation["config"]["routes"] == [
        {
            "capture_route": "/v1/responses",
            "provider_host": "api.openai.com",
            "provider_path": "/v1/responses",
            "provider_port": 443,
        }
    ]
    assert "--allow-hosts" in invocation["args"]
    regex = invocation["args"][invocation["args"].index("--allow-hosts") + 1]
    assert "api\\.openai\\.com" in regex
    assert "example.org" not in regex

    environment = mitm_environment(
        proxy_url=mitm.proxy_url,
        ca_bundle_path=str(public_bundle),
        base={"NO_PROXY": "*", "UNRELATED": "kept"},
    )
    assert environment["HTTPS_PROXY"] == mitm.proxy_url
    assert environment["SSL_CERT_FILE"] == str(public_bundle)
    assert environment["NO_PROXY"] == "127.0.0.1,localhost"
    assert environment["UNRELATED"] == "kept"
    assert all("mitmproxy-ca.pem" not in value for value in environment.values())

    receipt = mitm.stop(reason="test_complete")
    assert receipt.readiness_ok
    assert receipt.public_ca_digest
    assert receipt.public_trust_bundle_digest
    assert receipt.private_key_material_observed
    assert receipt.private_key_names == ("mitmproxy-ca.pem",)
    assert receipt.process_stopped
    assert receipt.private_key_destroyed
    assert receipt.confdir_destroyed
    assert not confdir.exists()


def test_scoped_mitm_addon_routes_only_declared_provider_paths(
    tmp_path: Path,
) -> None:
    event_log = tmp_path / "events.jsonl"
    config = AddonConfig(
        config_digest="sha256:config",
        allowed_hosts=("api.openai.com",),
        routes=(
            AddonRoute(
                provider_host="api.openai.com",
                provider_port=443,
                provider_path="/v1/responses",
                capture_route="/v1/responses",
            ),
        ),
        capture_proxy_host="127.0.0.1",
        capture_proxy_port=43123,
        event_log=event_log,
    )
    response_factory = lambda status, body, headers: SimpleNamespace(
        status=status,
        body=body,
        headers=dict(headers),
    )
    addon = ScopedProviderAddon(config, response_factory=response_factory)

    recognized = SimpleNamespace(
        request=SimpleNamespace(
            host="api.openai.com",
            port=443,
            scheme="https",
            path="/v1/responses",
            headers={},
        ),
        response=None,
    )
    addon.request(recognized)
    assert recognized.response is None
    assert recognized.request.scheme == "http"
    assert recognized.request.host == "127.0.0.1"
    assert recognized.request.port == 43123
    assert recognized.request.path == "/v1/responses"
    recognized.response = SimpleNamespace(stream=False)
    addon.responseheaders(recognized)
    assert recognized.response.stream is True

    query_request = SimpleNamespace(
        request=SimpleNamespace(
            host="api.openai.com",
            port=443,
            scheme="https",
            path="/v1/responses?beta=true",
            headers={},
        ),
        response=None,
    )
    addon.request(query_request)
    assert query_request.response.status == 421

    secret_path = "/v1/not-supported?token=must-not-appear"
    unknown_route = SimpleNamespace(
        request=SimpleNamespace(
            host="api.openai.com",
            port=443,
            scheme="https",
            path=secret_path,
            headers={},
        ),
        response=None,
    )
    addon.request(unknown_route)
    assert unknown_route.response.status == 421

    undeclared_tls = SimpleNamespace(
        request=SimpleNamespace(
            host="credentials.example.org",
            port=443,
            scheme="https",
            path="/secret",
            headers={},
        ),
        response=None,
    )
    addon.request(undeclared_tls)
    assert undeclared_tls.response.status == 421

    plain_http = SimpleNamespace(
        request=SimpleNamespace(
            host="example.org",
            port=80,
            scheme="http",
            path="/",
            headers={},
        ),
        response=None,
    )
    addon.request(plain_http)
    assert plain_http.response is None
    assert plain_http.request.host == "example.org"

    logged = event_log.read_text(encoding="utf-8")
    assert '"event":"provider_routed"' in logged
    assert '"event":"unmapped_provider_route"' in logged
    assert '"event":"unexpected_tls_host"' in logged
    assert "must-not-appear" not in logged
    assert "credentials.example.org" not in logged


def test_supervisor_tls_mitm_injects_public_trust_only_and_writes_receipt(
    tmp_path: Path,
) -> None:
    command, _ = _fake_mitmdump(tmp_path)
    config = _supervisor_config(
        tmp_path,
        interception=Interception.TLS_MITM,
        command=command,
    )

    with CaptureSupervisor(config) as supervisor:
        assert supervisor.mitm is not None
        confdir = supervisor.mitm.confdir
        environment = supervisor.environment()
        descriptor = supervisor.environment_descriptor()
        assert "OPENAI_BASE_URL" not in environment
        assert environment["HTTPS_PROXY"] == supervisor.mitm.proxy_url
        assert environment["SSL_CERT_FILE"].endswith(
            "synth-trace-public-ca-bundle.pem"
        )
        assert descriptor["mitm_allowlist"] == ("api.openai.com",)
        assert descriptor["capture_operational"] is True
        assert all(
            "mitmproxy-ca.pem" not in value
            for value in environment.values()
        )

    assert confdir is not None and not confdir.exists()
    receipts = tuple((config.bundle_root / "receipts").glob("mitm-lifecycle-*.json"))
    assert receipts
    lifecycle = json.loads(receipts[-1].read_text(encoding="utf-8"))
    assert lifecycle["readiness_ok"] is True
    assert lifecycle["private_key_destroyed"] is True
    assert lifecycle["confdir_destroyed"] is True
    assert lifecycle["allowed_hosts"] == ["api.openai.com"]


def test_supervisor_both_interception_exposes_explicit_proxy_and_public_ca_mount(
    tmp_path: Path,
) -> None:
    command, _ = _fake_mitmdump(tmp_path)
    config = _supervisor_config(
        tmp_path,
        interception=Interception.BOTH,
        command=command,
    )
    config.mitm_container_ca_path = "/trace/public-ca-bundle.pem"

    with CaptureSupervisor(config) as supervisor:
        environment = supervisor.environment("host.docker.internal")
        mount = supervisor.mitm_trust_mount()
        assert environment["OPENAI_BASE_URL"].startswith(
            "http://host.docker.internal:"
        )
        assert environment["HTTPS_PROXY"].startswith(
            "http://host.docker.internal:"
        )
        assert environment["SSL_CERT_FILE"] == "/trace/public-ca-bundle.pem"
        assert "host.docker.internal" in environment["NO_PROXY"]
        assert mount["target"] == "/trace/public-ca-bundle.pem"
        assert mount["read_only"] is True
        assert mount["source"].endswith("synth-trace-public-ca-bundle.pem")
        assert "mitmproxy-ca.pem" not in mount["source"]


def test_capture_modes_disable_or_suppress_dead_injection(tmp_path: Path) -> None:
    missing_command = (str(tmp_path / "missing-mitmdump"),)
    disabled = CaptureSupervisor(
        _supervisor_config(
            tmp_path / "disabled",
            mode=CaptureMode.DISABLED,
            interception=Interception.TLS_MITM,
            command=missing_command,
        )
    )
    disabled.start_capture()
    assert disabled.environment() == {}
    assert disabled.environment_descriptor()["variables"] == ()
    assert disabled.mitm is None
    assert disabled.proxy._started is False
    assert disabled.collector_server._started is False
    disabled.finalize()

    best_effort = CaptureSupervisor(
        _supervisor_config(
            tmp_path / "best-effort",
            mode=CaptureMode.BEST_EFFORT,
            interception=Interception.TLS_MITM,
            command=missing_command,
        )
    )
    best_effort.start_capture()
    assert best_effort.environment() == {}
    assert best_effort.receipt.readiness_ok is False
    assert best_effort.receipt.injected_variables == ()
    assert any(
        "startup failed" in item
        for item in best_effort.receipt.completeness_reasons
    )
    sealed = best_effort.finalize()
    assert any(
        "startup failed" in item
        for item in sealed.coverage.completeness_reasons
    )

    required = CaptureSupervisor(
        _supervisor_config(
            tmp_path / "required",
            mode=CaptureMode.REQUIRED,
            interception=Interception.TLS_MITM,
            command=missing_command,
        )
    )
    with pytest.raises(CaptureNotReady, match="startup failed"):
        required.start_capture()
    failed = required.finalize(status="failed")
    assert any(
        "scoped MITM lifecycle incomplete" in item
        for item in failed.coverage.completeness_reasons
    )


def test_observe_and_transform_requires_versioned_specification(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="transformation specification"):
        CaptureSupervisor(
            _supervisor_config(
                tmp_path,
                mode=CaptureMode.OBSERVE_AND_TRANSFORM,
            )
        )


def test_scoped_mitm_missing_binary_destroys_any_owned_state(tmp_path: Path) -> None:
    mitm = ScopedMitmProxy(
        capture_id="capture-test",
        endpoints=(_endpoint(),),
        capture_proxy_host="127.0.0.1",
        capture_proxy_port=43123,
        command=(str(tmp_path / "missing-mitmdump"),),
        temp_root=tmp_path / "owned",
        base_ca_bundle_path=_public_base_ca(tmp_path / "base-ca.pem"),
    )
    with pytest.raises(MitmStartupError, match="executable is unavailable"):
        mitm.start()
    receipt = mitm.lifecycle_receipt()
    assert receipt.readiness_ok is False
    assert receipt.private_key_destroyed is True
    assert receipt.confdir_destroyed is True
