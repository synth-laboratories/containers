"""Launch the managed controller inside a Rhodes service deployment."""

from __future__ import annotations

import argparse
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn


def main(argv=None):
    parser = argparse.ArgumentParser(prog="synth-harbor-serve")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8390)
    parser.add_argument("--data-dir", type=Path, default=Path("/var/lib/harbor"))
    parser.add_argument("--task-root", type=Path, action="append", required=True)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--inference-ceiling-micros", type=int, default=1_000_000)
    args = parser.parse_args(argv)
    token = os.environ.get("HARBOR_API_KEY", "")
    authority_url = os.environ.get("SYNTH_SERVICE_AUTHORITY_URL", "")
    if len(token) < 24 or not authority_url:
        parser.error("Managed origin credentials and authority URL are required")
    from .harbor_serve import create_managed_harbor_app
    from .harbor_serve_authority import HttpHarborAuthority

    client = httpx.AsyncClient(timeout=httpx.Timeout(60, read=3600), trust_env=False)
    app = create_managed_harbor_app(
        data_dir=args.data_dir,
        api_key=token,
        authority=HttpHarborAuthority(base_url=authority_url, token=token, client=client),
        task_roots=tuple(args.task_root),
        concurrency=args.max_concurrent,
        port=args.port,
        inference={"models": args.model, "ceiling_micros": args.inference_ceiling_micros}
        if args.model
        else None,
    )
    original = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        try:
            async with original(application):
                yield
        finally:
            await client.aclose()

    app.router.lifespan_context = lifespan
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
