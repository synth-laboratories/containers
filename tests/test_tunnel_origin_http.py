"""Real origin HTTP/TLS behavior without unused trust-store construction."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ssl
import threading
import urllib.error
import urllib.request

import pytest

from synth_containers.tunnels.relay import _open_upstream


@contextmanager
def origin(context=None):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.path)
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/target")
                self.send_header("Content-Length", "0")
            else:
                self.send_response(200)
                self.send_header("Content-Length", "2")
            self.end_headers()
            if self.path != "/redirect":
                self.wfile.write(b"ok")

        def log_message(self, *_args):
            pass

    class Server(ThreadingHTTPServer):
        request_queue_size = 128

    server = Server(("127.0.0.1", 0), Handler)
    if context is not None:
        server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
        assert not thread.is_alive()


def test_parallel_http_never_constructs_tls_authority(monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("plain HTTP parsed the TLS trust store")

    monkeypatch.setattr(http.client, "_create_https_context", unexpected)
    with origin() as (port, seen):
        def request(_index):
            with _open_upstream(urllib.request.Request(f"http://127.0.0.1:{port}/"), timeout=3) as response:
                return response.status, response.read()

        with ThreadPoolExecutor(max_workers=32) as pool:
            assert list(pool.map(request, range(64))) == [(200, b"ok")] * 64
        assert len(seen) == 64


def test_origin_redirect_cannot_forward_authority():
    with origin() as (port, seen):
        request = urllib.request.Request(f"http://127.0.0.1:{port}/redirect", headers={"X-Api-Key": "synthetic-origin-only"})
        with pytest.raises(urllib.error.HTTPError) as error:
            _open_upstream(request, timeout=3)
        assert error.value.code == 302
        error.value.close()
        assert seen == ["/redirect"]


def test_real_https_keeps_trust_and_hostname_verification(tmp_path, monkeypatch):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.now(timezone.utc)
    certificate = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
                   .public_key(key.public_key()).serial_number(x509.random_serial_number())
                   .not_valid_before(now - timedelta(minutes=1)).not_valid_after(now + timedelta(hours=1))
                   .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
                   .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
                   .sign(key, hashes.SHA256()))
    cert_path, key_path = tmp_path / "ca.pem", tmp_path / "server-key.pem"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    key_path.chmod(0o600)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    monkeypatch.setenv("SSL_CERT_FILE", str(cert_path))
    with origin(context) as (port, seen):
        with _open_upstream(urllib.request.Request(f"https://localhost:{port}/"), timeout=3) as response:
            assert response.status == 200 and response.read() == b"ok"
        with pytest.raises(urllib.error.URLError) as error:
            _open_upstream(urllib.request.Request(f"https://127.0.0.1:{port}/"), timeout=3)
        assert isinstance(error.value.reason, ssl.SSLCertVerificationError)
        # A new connection must still read current authority; do not cache or
        # silently disable verification to fix the plain-HTTP performance bug.
        empty = tmp_path / "empty-ca.pem"
        empty.write_bytes(b"")
        monkeypatch.setenv("SSL_CERT_FILE", str(empty))
        with pytest.raises(urllib.error.URLError) as error:
            _open_upstream(urllib.request.Request(f"https://localhost:{port}/"), timeout=3)
        assert isinstance(error.value.reason, ssl.SSLCertVerificationError)
        assert seen == ["/"]
