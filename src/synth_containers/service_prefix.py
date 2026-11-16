"""Authenticated reverse-proxy prefix for native docs and URL generation."""

import hmac

from .service_contracts import service_path


class ServicePrefixMiddleware:
    def __init__(self, app, *, api_key: str):
        self.app, self.api_key = app, api_key

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and not scope.get("synth.prefix.applied"):
            headers = dict(scope.get("headers", []))
            prefix = headers.get(b"x-synth-service-prefix")
            if prefix and hmac.compare_digest(
                headers.get(b"authorization", b""), f"Bearer {self.api_key}".encode()
            ):
                prefix = service_path(prefix.decode("ascii"))
                scope = {
                    **scope,
                    "root_path": prefix,
                    "path": prefix + scope["path"],
                    "raw_path": prefix.encode() + scope.get("raw_path", scope["path"].encode()),
                    "synth.prefix.applied": True,
                }
        await self.app(scope, receive, send)
