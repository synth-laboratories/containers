"""In-container annotation: mount the service beside the task platform.

Every terminal rollout already seals a portable Trace V5 bundle under
``{storage_root}/trace_bundles/*.zip``. This module turns that directory into a
trace source, exposes the annotation router on the container's own app, and —
when configured — runs a *post-rollout stage*: a watcher that submits the
configured deterministic annotators for every newly sealed bundle. It never
touches the platform's rollout state machine; it only reads sealed archives.

Environment (read by ``install_from_env``)::

    SYNTH_ANNOTATION=off                      disable entirely (default: on)
    SYNTH_ANNOTATION_POST_ROLLOUT=id1,id2     annotators to run after every seal (default: none)
    SYNTH_ANNOTATION_MAX_CONCURRENT=4         scheduler global cap
    SYNTH_ANNOTATION_DOMAINS=mod:fn,mod:fn    extra ``fn(registry)`` registrars to import if available
    SYNTH_ANNOTATION_PROMOTE=mod:fn           ``fn(document, sealed_digest) -> document`` promotion (e.g. Craftax lanes)
    SYNTH_ANNOTATION_BROKER_SECRET=...        enables host-signed reservations (SignedReservationBroker)
    SYNTH_ANNOTATION_BROKER_URL=...           where reconciliations are pushed (else pulled from /annotation/reservations)
    SYNTH_ANNOTATION_BROKER_TOKEN=...         bearer for the push
    SYNTH_ANNOTATION_PRICE_TABLE=path.json    per-model USD/1M-token prices (JSON or TOML); no default prices exist
    SYNTH_ANNOTATION_USD_PER_MILLION_TOKENS=  legacy flat price (token ceiling only, no USD reported)
    SYNTH_ANNOTATION_PROXY_ENFORCES=on        host asserts its provider proxy enforces reservations

Paid annotators are impossible in-container unless a broker is injected by the
host: the default broker denies. A paid model that is not priced (table or flat)
and not proxy-enforced is refused at submit; ``GET /annotation/pricing`` reports
which models are priced.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Iterable

from ..canonical import bytes_digest
from ..models.document import TraceDocumentV5
from ..store.bundle import LocalTraceBundle
from .broker import PaidComputeBroker
from .builtin import register_builtin_annotators
from .campaign import AnnotationCampaign, AnnotatorPlan, CampaignPlan, CampaignRun
from .definitions import DefinitionRegistry
from .persistence import AnnotationStore
from .pricing import PRICE_TABLE_ENV, PriceTable, PriceTableError
from .scheduler import AnnotationScheduler, ThroughputLimits
from .service import AnnotationService, AnnotatorRunner
from .sources import TraceLoader, bundle_trace_loader, bundle_trace_refs, chain_loaders

log = logging.getLogger("synth_containers.annotation")

Promote = Callable[[TraceDocumentV5, str], TraceDocumentV5]


def _archives(storage_root: Path) -> list[Path]:
    folder = Path(storage_root) / "trace_bundles"
    if not folder.exists():
        return []
    return sorted(path for path in folder.glob("*.zip") if path.is_file())


def _extract(archive: Path, cache_root: Path) -> Path | None:
    """Extract once per archive content; the cache key is the archive digest."""

    try:
        digest = bytes_digest(archive.read_bytes())
    except OSError:
        return None
    target = cache_root / digest.replace(":", "_")
    marker = target / ".extracted"
    if marker.exists():
        return target
    try:
        LocalTraceBundle.extract_archive(archive, target, require_self_contained=False)
        marker.write_text(json.dumps({"archive": str(archive), "digest": digest}))
    except Exception as error:  # noqa: BLE001 - a bad archive is skipped, never fatal
        log.warning("annotation: cannot extract %s: %s", archive, error)
        return None
    return target


class ContainerTraceSource:
    """Sealed bundles under the platform storage root, extracted on demand."""

    def __init__(self, storage_root: Path, *, cache_root: Path | None = None, promote: Promote | None = None) -> None:
        self.storage_root = Path(storage_root)
        self.cache_root = Path(cache_root or (self.storage_root / "annotation" / "bundles"))
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.promote = promote

    def bundle_dirs(self) -> list[Path]:
        found: list[Path] = []
        for archive in _archives(self.storage_root):
            extracted = _extract(archive, self.cache_root)
            if extracted is not None:
                found.append(extracted)
        return found

    def refs(self) -> list[dict[str, str]]:
        refs: list[dict[str, str]] = []
        for folder in self.bundle_dirs():
            try:
                refs.extend(bundle_trace_refs(folder, promote=self.promote))
            except Exception as error:  # noqa: BLE001
                log.warning("annotation: cannot read bundle %s: %s", folder, error)
        return refs

    def loader(self) -> TraceLoader:
        def load(trace_id: str, digest: str) -> TraceDocumentV5 | None:
            loaders = [bundle_trace_loader(folder, promote=self.promote) for folder in self.bundle_dirs()]
            return chain_loaders(loaders)(trace_id, digest)

        return load


class PostRolloutWatcher:
    """The optional pipelining stage: annotate every newly sealed bundle with a fixed plan."""

    def __init__(self, source: ContainerTraceSource, campaign: AnnotationCampaign, *, annotators: Iterable[AnnotatorPlan], session_id: str = "post_rollout", interval_seconds: float = 2.0, on_run: Callable[[CampaignRun], None] | None = None) -> None:
        self.source = source
        self.campaign = campaign
        self.annotators = tuple(annotators)
        self.session_id = session_id
        self.interval_seconds = interval_seconds
        self.on_run = on_run
        self._seen: set[str] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.runs: list[CampaignRun] = []

    def poll_once(self) -> CampaignRun | None:
        if not self.annotators:
            return None
        fresh = [ref for ref in self.source.refs() if ref["digest"] not in self._seen]
        if not fresh:
            return None
        plan = CampaignPlan(traces=tuple((ref["id"], ref["digest"]) for ref in fresh), annotators=self.annotators, session_id=self.session_id, label="post_rollout")
        run = self.campaign.submit(plan)
        self._seen.update(ref["digest"] for ref in fresh)
        self.runs.append(run)
        if self.on_run:
            self.on_run(run)
        return run

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception as error:  # noqa: BLE001 - the watcher is an evidence lane; it never crashes the container
                log.warning("annotation: post-rollout poll failed: %s", error)
            self._stop.wait(self.interval_seconds)

    def start(self) -> "PostRolloutWatcher":
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="annotation-post-rollout", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None


class ContainerAnnotation:
    """Everything mounted on one container: source, service, scheduler, watcher."""

    def __init__(self, *, storage_root: Path, registry: DefinitionRegistry, runners: dict[str, AnnotatorRunner] | None = None, broker: PaidComputeBroker | None = None, limits: ThroughputLimits | None = None, promote: Promote | None = None, post_rollout: Iterable[str] = (), price_table: PriceTable | None = None) -> None:
        self.storage_root = Path(storage_root)
        self.source = ContainerTraceSource(self.storage_root, promote=promote)
        self.registry = registry
        self.price_table = price_table
        self.service = AnnotationService(store=AnnotationStore(self.storage_root / "annotation" / "store"), registry=registry, runners=runners, trace_loader=self.source.loader(), broker=broker)
        self.scheduler = AnnotationScheduler(self.service, limits=limits or ThroughputLimits(max_concurrent_total=2, poll_seconds=0.5))
        self.campaign = AnnotationCampaign(self.service, self.scheduler)
        plans = [AnnotatorPlan(annotator_id=item) for item in post_rollout if registry.get(item) is not None]
        missing = [item for item in post_rollout if registry.get(item) is None]
        if missing:
            log.warning("annotation: post-rollout annotators not registered: %s", missing)
        self.watcher = PostRolloutWatcher(self.source, self.campaign, annotators=plans)

    def start(self) -> None:
        self.scheduler.start()
        if self.watcher.annotators:
            self.watcher.start()

    def stop(self) -> None:
        self.watcher.stop()
        self.scheduler.stop()

    def status(self) -> dict[str, Any]:
        return {
            "storage_root": str(self.storage_root),
            "annotators": [entry.annotator_id for entry in self.registry.list()],
            "post_rollout": [plan.annotator_id for plan in self.watcher.annotators],
            "post_rollout_runs": len(self.watcher.runs),
            "scheduler": self.scheduler.snapshot(),
            "broker": type(self.service.broker).__name__,
            "priced_models": list(self.price_table.models()) if self.price_table is not None else [],
        }

    def pricing(self) -> dict[str, Any]:
        """Which models are priced, per runner, and how each runner enforces dollars."""

        runners: dict[str, Any] = {}
        for kind, runner in self.service.runners.items():
            table = getattr(runner, "price_table", None)
            runners[kind] = {
                "priced_models": list(table.models()) if isinstance(table, PriceTable) else [],
                "flat_usd_per_million_tokens": getattr(runner, "usd_per_million_tokens", None),
                "proxy_enforces_reservation": bool(getattr(runner, "proxy_enforces_reservation", False)),
                "paid": kind != "deterministic",
            }
        table = self.price_table
        return {
            "env": PRICE_TABLE_ENV,
            "source": table.source if table is not None else None,
            "priced_models": list(table.models()) if table is not None else [],
            "prices": table.describe()["models"] if table is not None else [],
            "runners": runners,
        }


def default_registry(extra_registrars: Iterable[str] = ()) -> DefinitionRegistry:
    registry = DefinitionRegistry()
    register_builtin_annotators(registry)
    for spec in extra_registrars:
        module_name, _, function_name = spec.partition(":")
        if not module_name or not function_name:
            continue
        try:
            module = importlib.import_module(module_name)
            getattr(module, function_name)(registry)
        except Exception as error:  # noqa: BLE001 - optional domain packs must never block an image
            log.warning("annotation: registrar %s unavailable: %s", spec, error)
    return registry


def load_promote(spec: str) -> Promote | None:
    """Import an optional ``module:function`` promotion; unavailable means no promotion, not a crash."""

    module_name, _, function_name = spec.strip().partition(":")
    if not module_name or not function_name:
        return None
    try:
        return getattr(importlib.import_module(module_name), function_name)
    except Exception as error:  # noqa: BLE001
        log.warning("annotation: promotion %s unavailable: %s", spec, error)
        return None


def mount_annotation(app: Any, *, storage_root: Path, registry: DefinitionRegistry | None = None, runners: dict[str, AnnotatorRunner] | None = None, broker: PaidComputeBroker | None = None, limits: ThroughputLimits | None = None, promote: Promote | None = None, post_rollout: Iterable[str] = (), start: bool = True, price_table: PriceTable | None = None) -> ContainerAnnotation:
    """Attach the annotation router, scheduler, and optional post-rollout stage to a FastAPI app."""

    from .api import build_annotation_router

    mounted = ContainerAnnotation(storage_root=storage_root, registry=registry or default_registry(), runners=runners, broker=broker, limits=limits, promote=promote, post_rollout=post_rollout, price_table=price_table)
    app.include_router(build_annotation_router(mounted.service, scheduler=mounted.scheduler))

    @app.get("/annotation/traces")
    async def annotation_traces() -> dict[str, Any]:
        return {"traces": mounted.source.refs()}

    @app.get("/annotation/status")
    async def annotation_status() -> dict[str, Any]:
        return mounted.status()

    @app.get("/annotation/reservations")
    async def annotation_reservations() -> dict[str, Any]:
        reconciled = getattr(mounted.service.broker, "reconciled", None)
        return {"reconciled": reconciled() if callable(reconciled) else [], "push": getattr(mounted.service.broker, "reconcile_url", None), "priced_models": list(mounted.price_table.models()) if mounted.price_table is not None else []}

    @app.get("/annotation/pricing")
    async def annotation_pricing() -> dict[str, Any]:
        return mounted.pricing()

    @app.post("/annotation/campaigns")
    async def annotation_campaign(body: dict[str, Any]) -> dict[str, Any]:
        refs = body.get("traces") or mounted.source.refs()
        plans = [AnnotatorPlan(annotator_id=str(item)) if isinstance(item, str) else AnnotatorPlan(**item) for item in body.get("annotators") or []]
        from .campaign import plan_from_refs

        plan = plan_from_refs(refs, plans, session_id=body.get("session_id"), label=str(body.get("label") or ""))
        if body.get("estimate_only"):
            estimate = mounted.campaign.estimate(plan)
            return {"estimate": {**estimate.__dict__, "paid_jobs": list(estimate.paid_jobs), "notes": list(estimate.notes)}}
        reservations = body.get("reservations") or {}
        reservation_for = None
        if isinstance(reservations, dict) and reservations:
            # The host issues one reservation per paid job, keyed by trace digest, annotator, repeat.
            def reservation_for(request, session_id):  # noqa: ANN001
                key = f"{request.source_trace_digest}|{request.annotator_id}|{request.repeat_index}"
                token = reservations.get(key)
                if not isinstance(token, str) or not token:
                    raise LookupError(f"no reservation for {key}")
                return token

        run = mounted.campaign.submit(plan, reservation_for=reservation_for)
        job_bindings = [
            {
                "key": f"{job.request.source_trace_digest}|{job.request.annotator_id}|{job.request.repeat_index}",
                "job_id": job.job_id,
                "reservation_id": job.reservation_id,
            }
            for job in run.jobs
            if job.reservation_id
        ]
        return {"campaign_id": run.campaign_id, "jobs": run.job_ids, "job_bindings": job_bindings, "cache_hits": run.cache_hits, "enqueued": run.enqueued, "refused": run.refused}

    app.state.annotation = mounted
    if start:
        @app.on_event("startup")
        async def _start() -> None:
            mounted.start()

        @app.on_event("shutdown")
        async def _stop() -> None:
            mounted.stop()

    return mounted


def install_from_env(app: Any, *, storage_root: Path | None) -> ContainerAnnotation | None:
    """Image entrypoint hook. Fail-soft: annotation never prevents a container from serving."""

    if os.environ.get("SYNTH_ANNOTATION", "on").strip().lower() in {"off", "0", "false", "no"}:
        return None
    root = Path(storage_root) if storage_root else Path(os.environ.get("SYNTH_CONTAINER_STORAGE") or "/var/lib/synth/storage")
    post_rollout = [item.strip() for item in os.environ.get("SYNTH_ANNOTATION_POST_ROLLOUT", "").split(",") if item.strip()]
    registrars = [item.strip() for item in os.environ.get("SYNTH_ANNOTATION_DOMAINS", "").split(",") if item.strip()]
    try:
        limit = int(os.environ.get("SYNTH_ANNOTATION_MAX_CONCURRENT", "2"))
    except ValueError:
        limit = 2
    promote = load_promote(os.environ.get("SYNTH_ANNOTATION_PROMOTE", ""))
    broker: PaidComputeBroker | None = None
    secret = os.environ.get("SYNTH_ANNOTATION_BROKER_SECRET", "")
    if secret:
        from .signed_broker import SignedReservationBroker

        broker = SignedReservationBroker(root / "annotation" / "reservations", secret=secret.encode("utf-8"), reconcile_url=os.environ.get("SYNTH_ANNOTATION_BROKER_URL") or None, reconcile_token=os.environ.get("SYNTH_ANNOTATION_BROKER_TOKEN") or None)
    price_table: PriceTable | None = None
    try:
        price_table = PriceTable.from_env()
    except PriceTableError as error:
        # No table is safer than a wrong one: unpriced models fail closed at submit.
        log.warning("annotation: price table ignored: %s", error)
    runners: dict[str, AnnotatorRunner] | None = None
    if secret and os.environ.get("SYNTH_ANNOTATION_CODEX", "").strip().lower() in {"on", "1", "true", "yes"}:
        from .codex_app_server import CodexAppServerRunner

        price = os.environ.get("SYNTH_ANNOTATION_USD_PER_MILLION_TOKENS")
        runners = {"codex_app_server": CodexAppServerRunner(default_effort=os.environ.get("SYNTH_ANNOTATION_DEFAULT_EFFORT") or "medium", usd_per_million_tokens=float(price) if price else None, proxy_enforces_reservation=os.environ.get("SYNTH_ANNOTATION_PROXY_ENFORCES", "").lower() in {"on", "1", "true", "yes"}, price_table=price_table)}
    try:
        return mount_annotation(app, storage_root=root, registry=default_registry(registrars), runners=runners, broker=broker, limits=ThroughputLimits(max_concurrent_total=max(1, limit), poll_seconds=0.5), post_rollout=post_rollout, promote=promote, price_table=price_table)
    except Exception as error:  # noqa: BLE001
        log.warning("annotation: not mounted: %s", error)
        return None


__all__ = ["ContainerAnnotation", "ContainerTraceSource", "PostRolloutWatcher", "default_registry", "install_from_env", "load_promote", "mount_annotation"]
