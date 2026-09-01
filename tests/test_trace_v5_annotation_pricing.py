"""Price tables: explicit per-model prices, no defaults, fail closed when unpriced."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from synth_containers.tracing.annotation import (
    AnnotationJobErrorCode,
    AnnotationJobLimitsV1,
    AnnotationJobState,
    AnnotationService,
    AnnotationServiceError,
    AnnotationStore,
    AnnotatorProgramV1,
    CodexAppServerRunner,
    CompletionResult,
    DefinitionRegistry,
    LocalReservationBroker,
    ModelApiRunner,
    ModelPrice,
    PriceTable,
    PriceTableError,
    ReservationBindingV1,
    RunnerKind,
    ScriptedAppServer,
    build_craftax_smoke_trace,
)
from synth_containers.tracing.annotation.proposal import PROPOSAL_SCHEMA_VERSION
from synth_containers.tracing.models.standards import (
    AnnotationOutputContractV1,
    AnnotationTaskKind,
    AnnotationTaxonV1,
    TraceAnnotatorDefinitionV1,
)

MODEL = "gpt-5.6-luna"
TABLE = {"source": "unit", "models": {MODEL: {"input_usd_per_million": 1.0, "cached_input_usd_per_million": 0.1, "output_usd_per_million": 10.0}}}


def test_price_table_loads_only_explicit_prices(tmp_path: Path, monkeypatch) -> None:
    assert len(PriceTable()) == 0 and PriceTable().get(MODEL) is None  # no defaults, ever
    table = PriceTable.from_dict(TABLE)
    assert table.source == "unit" and table.models() == (MODEL,) and MODEL in table and "other" not in table
    price = table.get(MODEL)
    assert price == ModelPrice(model=MODEL, input_usd_per_million=1.0, cached_input_usd_per_million=0.1, output_usd_per_million=10.0)
    # Flat mapping and aliases are accepted; anything malformed is refused whole.
    flat = PriceTable.from_dict({"m": {"input": 2, "cached": 1, "output": 4}})
    assert flat.get("m").cached_input_usd_per_million == 1.0
    for bad in ({"m": {"input": 1, "output": 2}}, {"m": {"input": -1, "cached": 0, "output": 0}}, {"m": {"input": "x", "cached": 0, "output": 0}}, {"m": {"input": float("nan"), "cached": 0, "output": 0}}, {"m": 3}):
        with pytest.raises(PriceTableError):
            PriceTable.from_dict(bad)
    with pytest.raises(PriceTableError):
        PriceTable({"m": ModelPrice(model="other", input_usd_per_million=1, cached_input_usd_per_million=1, output_usd_per_million=1)})

    json_path = tmp_path / "prices.json"
    json_path.write_text(json.dumps(TABLE))
    toml_path = tmp_path / "prices.toml"
    toml_path.write_text(f'source = "toml"\n[models."{MODEL}"]\ninput_usd_per_million = 1.0\ncached_input_usd_per_million = 0.1\noutput_usd_per_million = 10.0\n')
    assert PriceTable.from_file(json_path).describe() == table.describe()
    assert PriceTable.from_file(toml_path).get(MODEL) == price and PriceTable.from_file(toml_path).source == "toml"
    with pytest.raises(PriceTableError):
        PriceTable.from_file(tmp_path / "missing.json")
    (tmp_path / "broken.json").write_text("{not json")
    with pytest.raises(PriceTableError):
        PriceTable.from_file(tmp_path / "broken.json")

    monkeypatch.delenv("SYNTH_ANNOTATION_PRICE_TABLE", raising=False)
    assert PriceTable.from_env() is None
    monkeypatch.setenv("SYNTH_ANNOTATION_PRICE_TABLE", str(json_path))
    assert PriceTable.from_env().models() == (MODEL,)


def test_model_price_arithmetic_and_conservative_billable_ceiling() -> None:
    price = PriceTable.from_dict(TABLE).get(MODEL)
    # 40k input of which 30k cached, 2k output: 10k x $1 + 30k x $0.1 + 2k x $10 per million
    assert price.cost_usd(input_tokens=40_000, cached_input_tokens=30_000, output_tokens=2_000) == pytest.approx(0.033)
    assert price.cost_usd(input_tokens=None, output_tokens=None) == 0.0
    assert price.cost_usd(input_tokens=100, cached_input_tokens=1_000, output_tokens=0) == pytest.approx(100 * 0.1 / 1e6)  # cached never exceeds input
    # Every billable token is charged at the dearest billable rate (output, $10/M): $0.05 buys 5,000.
    assert price.billable_token_ceiling(0.05) == 5_000 and price.billable_token_ceiling(None) is None
    free = ModelPrice(model="free", input_usd_per_million=0.0, cached_input_usd_per_million=0.0, output_usd_per_million=0.0)
    assert free.billable_token_ceiling(1.0) is None and free.cost_usd(input_tokens=10, output_tokens=10) == 0.0
    table = PriceTable.from_dict(TABLE)
    assert table.cost(MODEL, input_tokens=1_000_000, output_tokens=0) == (pytest.approx(1.0), "pinned_price")
    assert table.cost("unpriced", input_tokens=1_000_000, output_tokens=0) == (None, "unavailable")
    assert table.billable_token_ceiling("unpriced", 1.0) is None


# -- codex app-server runner ------------------------------------------------------------


def _definition(annotator_id: str = "test.priced") -> TraceAnnotatorDefinitionV1:
    return TraceAnnotatorDefinitionV1(annotator_id=annotator_id, name="p", purpose="p", taxonomy=("x",), required_subject_scope="message", model=MODEL, output_contract=AnnotationOutputContractV1(task_kind=AnnotationTaskKind.CLASSIFY, annotation_types=("t",), taxonomy=(AnnotationTaxonV1(label="x"),))).sealed()


def _empty_agent():
    manifest = yield ("trace_get_manifest", {})
    return json.dumps({"schema_version": PROPOSAL_SCHEMA_VERSION, "source_trace_id": manifest["trace_id"], "source_trace_digest": manifest["trace_digest"], "findings": [], "abstentions": []})


def _codex_service(tmp_path: Path, runner: CodexAppServerRunner):
    registry = DefinitionRegistry()
    registry.register(_definition(), AnnotatorProgramV1(program_id="test.priced.p", runner_kind=RunnerKind.CODEX_APP_SERVER, prompt="go", paid=True).sealed(), domain="test")
    broker = LocalReservationBroker(tmp_path / "broker")
    service = AnnotationService(store=AnnotationStore(tmp_path / "store"), registry=registry, runners={runner.kind: runner}, broker=broker)
    trace = build_craftax_smoke_trace()
    service.register_trace(trace)
    return service, trace, broker


def _reserve(broker, trace, *, cap_usd_micros: int = 1_000_000) -> str:
    return broker.issue(cap_usd_micros=cap_usd_micros, binding=ReservationBindingV1(trace_digest=trace.content_digest, annotator_id="test.priced", model=MODEL, session_id="s")).reservation_id


def test_codex_runner_reports_pinned_cost_and_holds_both_ceilings(tmp_path: Path) -> None:
    usage = {"inputTokens": 40_000, "cachedInputTokens": 30_000, "outputTokens": 2_000, "totalTokens": 42_000}
    runner = CodexAppServerRunner(lambda cwd: ScriptedAppServer(_empty_agent, token_usage=usage), poll_seconds=0.01, default_effort="low", price_table=PriceTable.from_dict(TABLE))
    service, trace, broker = _codex_service(tmp_path, runner)
    assert runner.cost_enforcement(MODEL) == "pinned_price" and runner.cost_enforcement("unpriced") is None
    limits = AnnotationJobLimitsV1(max_total_tokens=100_000, max_cost_usd=0.2)  # $0.2 at $10/M buys 20k billable tokens
    request = service.request_for(trace, "test.priced", limits=limits)
    assert request.model == MODEL and runner.token_ceiling(limits, MODEL) == 20_000
    job = service.submit_and_run(request, reservation_id=_reserve(broker, trace), session_id="s")
    assert str(job.state) == AnnotationJobState.SEALED, job.error
    assert job.usage.cost_usd == pytest.approx(0.033) and job.usage.cost_status == "pinned_price"
    assert job.usage.cached_input_tokens == 30_000 and job.usage.input_tokens == 40_000
    assert broker.get(job.reservation_id).actual_cost_usd_micros == 33_000  # reconciled from the pinned price
    assert service.store.ledger.get(job.job_id).metadata["cost_enforcement"] == "pinned_price"

    # Billable-token ceiling (uncached input + output = 12k) from the cost cap: $0.02 at $10/M buys 2,000.
    tight = service.request_for(trace, "test.priced", repeat_index=1, limits=AnnotationJobLimitsV1(max_total_tokens=100_000, max_cost_usd=0.02))
    capped = service.submit_and_run(tight, reservation_id=_reserve(broker, trace), session_id="s")
    assert capped.error.code == AnnotationJobErrorCode.TOKEN_LIMIT_EXCEEDED and "12000 billable" in capped.error.message and "ceiling 2000" in capped.error.message
    assert broker.get(capped.reservation_id).outcome == "failed"

    # Cost ceiling: cached input is priced but not billable-counted, so dollars are enforced too.
    cached_heavy = {"inputTokens": 1_000_000, "cachedInputTokens": 999_000, "outputTokens": 0, "totalTokens": 1_000_000}
    expensive_cache = PriceTable.from_dict({MODEL: {"input": 1.0, "cached": 1.0, "output": 1.0}})
    runner2 = CodexAppServerRunner(lambda cwd: ScriptedAppServer(_empty_agent, token_usage=cached_heavy), poll_seconds=0.01, default_effort="low", price_table=expensive_cache)
    service2, trace2, broker2 = _codex_service(tmp_path / "two", runner2)
    request2 = service2.request_for(trace2, "test.priced", limits=AnnotationJobLimitsV1(max_total_tokens=2_000_000, max_cost_usd=0.005))
    assert runner2.token_ceiling(request2.limits, MODEL) == 5_000  # 1,000 billable tokens fit...
    over = service2.submit_and_run(request2, reservation_id=_reserve(broker2, trace2), session_id="s")
    assert over.error.code == AnnotationJobErrorCode.COST_LIMIT_EXCEEDED  # ...but $1.00 of cached input does not
    assert over.usage.cost_usd == pytest.approx(1.0) and over.usage.cost_status == "pinned_price"
    assert broker2.get(over.reservation_id).actual_cost_usd_micros == 1_000_000


def test_unpriced_model_fails_closed_unless_the_proxy_enforces(tmp_path: Path) -> None:
    table = PriceTable.from_dict({"some-other-model": {"input": 1, "cached": 1, "output": 1}})
    runner = CodexAppServerRunner(lambda cwd: ScriptedAppServer(_empty_agent, token_usage={"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}), poll_seconds=0.01, default_effort="low", price_table=table)
    service, trace, broker = _codex_service(tmp_path, runner)
    request = service.request_for(trace, "test.priced", limits=AnnotationJobLimitsV1(max_total_tokens=1_000))
    reservation = _reserve(broker, trace)
    with pytest.raises(AnnotationServiceError) as refused:
        service.submit(request, reservation_id=reservation, session_id="s")
    assert refused.value.detail == {"reason": "cost_enforcement_unavailable", "model": MODEL}
    assert broker.get(reservation).claimed_by_job_id is None
    # The host asserting proxy enforcement lets it run; the usage stays honestly unpriced.
    runner.proxy_enforces_reservation = True
    job = service.submit_and_run(request, reservation_id=reservation, session_id="s")
    assert str(job.state) == AnnotationJobState.SEALED and job.usage.cost_usd is None and job.usage.cost_status == "unavailable"
    assert service.store.ledger.get(job.job_id).metadata["cost_enforcement"] == "provider_proxy"
    assert broker.get(reservation).actual_cost_usd_micros is None


def test_pricing_never_touches_the_idempotency_key(tmp_path: Path) -> None:
    unpriced = CodexAppServerRunner(lambda cwd: None, default_effort="low", proxy_enforces_reservation=True)
    priced = CodexAppServerRunner(lambda cwd: None, default_effort="low", price_table=PriceTable.from_dict(TABLE))
    service, trace, _ = _codex_service(tmp_path, unpriced)
    request = service.request_for(trace, "test.priced", limits=AnnotationJobLimitsV1(max_total_tokens=1_000, max_cost_usd=0.5))
    key_unpriced = service.estimate(request).idempotency_key
    service.runners[priced.kind] = priced
    assert service.estimate(request).idempotency_key == key_unpriced
    assert service.estimate(service.request_for(trace, "test.priced", limits=AnnotationJobLimitsV1(max_total_tokens=1_000, max_cost_usd=0.9))).idempotency_key == key_unpriced


# -- model API runner ---------------------------------------------------------------------


def _completion(**usage: Any):
    def complete(*, model, instructions, context, schema, max_output_tokens):
        trace_id = context.split("trace_id: ")[1].splitlines()[0]
        digest = context.split("trace_digest: ")[1].splitlines()[0]
        return CompletionResult(text=json.dumps({"schema_version": PROPOSAL_SCHEMA_VERSION, "source_trace_id": trace_id, "source_trace_digest": digest, "findings": [], "abstentions": []}), **usage)

    return complete


def _model_service(tmp_path: Path, runner: ModelApiRunner):
    registry = DefinitionRegistry()
    registry.register(_definition(), AnnotatorProgramV1(program_id="test.priced.p", runner_kind=RunnerKind.MODEL_API, prompt="go", paid=True).sealed(), domain="test")
    broker = LocalReservationBroker(tmp_path / "broker")
    service = AnnotationService(store=AnnotationStore(tmp_path / "store"), registry=registry, runners={runner.kind: runner}, broker=broker)
    trace = build_craftax_smoke_trace()
    service.register_trace(trace)
    return service, trace, broker


def test_model_api_runner_prices_completions_from_the_table(tmp_path: Path) -> None:
    table = PriceTable.from_dict(TABLE)
    runner = ModelApiRunner(_completion(input_tokens=40_000, cached_input_tokens=30_000, output_tokens=2_000, total_tokens=42_000), price_table=table)
    service, trace, broker = _model_service(tmp_path, runner)
    job = service.submit_and_run(service.request_for(trace, "test.priced", limits=AnnotationJobLimitsV1(max_total_tokens=100_000, max_cost_usd=0.2)), reservation_id=_reserve(broker, trace), session_id="s")
    assert str(job.state) == AnnotationJobState.SEALED, job.error
    assert job.usage.cost_usd == pytest.approx(0.033) and job.usage.cost_status == "pinned_price" and job.usage.cached_input_tokens == 30_000
    assert broker.get(job.reservation_id).actual_cost_usd_micros == 33_000
    # Provider-reported cost wins over the table, and the cost ceiling applies to it.
    reported = ModelApiRunner(_completion(input_tokens=100, output_tokens=10, total_tokens=110, cost_usd=0.2), price_table=table)
    service.runners[reported.kind] = reported
    job2 = service.submit_and_run(service.request_for(trace, "test.priced", repeat_index=1, limits=AnnotationJobLimitsV1(max_total_tokens=100_000, max_cost_usd=0.05)), reservation_id=_reserve(broker, trace), session_id="s")
    assert job2.error.code == AnnotationJobErrorCode.COST_LIMIT_EXCEEDED and job2.usage.cost_status == "reported"
    # Billable ceiling: 12k billable tokens against $0.02 at $10/M = 2,000.
    job3 = service.submit_and_run(service.request_for(trace, "test.priced", repeat_index=2, limits=AnnotationJobLimitsV1(max_total_tokens=100_000, max_cost_usd=0.02)), reservation_id=_reserve(broker, trace), session_id="s")
    service.runners[runner.kind] = runner
    job4 = service.submit_and_run(service.request_for(trace, "test.priced", repeat_index=3, limits=AnnotationJobLimitsV1(max_total_tokens=100_000, max_cost_usd=0.02)), reservation_id=_reserve(broker, trace), session_id="s")
    assert job3.error.code == AnnotationJobErrorCode.COST_LIMIT_EXCEEDED  # reported $0.2 > $0.02, only 110 tokens
    assert job4.error.code == AnnotationJobErrorCode.TOKEN_LIMIT_EXCEEDED and "ceiling 2000" in job4.error.message


# -- container wiring ---------------------------------------------------------------------


def test_container_honours_the_env_var_and_reports_priced_models(tmp_path: Path, monkeypatch) -> None:
    from synth_containers.platform import create_compat_app
    from synth_containers.tracing.annotation.container import install_from_env

    path = tmp_path / "prices.json"
    path.write_text(json.dumps(TABLE))
    monkeypatch.setenv("SYNTH_ANNOTATION", "on")
    monkeypatch.setenv("SYNTH_ANNOTATION_BROKER_SECRET", "secret")
    monkeypatch.setenv("SYNTH_ANNOTATION_CODEX", "on")
    monkeypatch.setenv("SYNTH_ANNOTATION_PRICE_TABLE", str(path))
    monkeypatch.delenv("SYNTH_ANNOTATION_USD_PER_MILLION_TOKENS", raising=False)
    app = create_compat_app("openenv_echo", storage_root=tmp_path)
    mounted = install_from_env(app, storage_root=tmp_path)
    assert mounted is not None and mounted.price_table.models() == (MODEL,)
    runner = mounted.service.runners["codex_app_server"]
    assert runner.price_table.get(MODEL) is not None and runner.cost_enforcement(MODEL) == "pinned_price" and runner.cost_enforcement("other") is None
    client = TestClient(app)
    pricing = client.get("/annotation/pricing").json()
    assert pricing["priced_models"] == [MODEL] and pricing["source"] == "unit" and pricing["env"] == "SYNTH_ANNOTATION_PRICE_TABLE"
    assert pricing["runners"]["codex_app_server"] == {"priced_models": [MODEL], "flat_usd_per_million_tokens": None, "proxy_enforces_reservation": False, "paid": True}
    assert pricing["prices"][0]["output_usd_per_million"] == 10.0
    assert client.get("/annotation/reservations").json()["priced_models"] == [MODEL]
    assert client.get("/annotation/status").json()["priced_models"] == [MODEL]

    # A malformed table is ignored (logged), leaving every model unpriced: fail closed, not fail open.
    (tmp_path / "broken.json").write_text("{")
    monkeypatch.setenv("SYNTH_ANNOTATION_PRICE_TABLE", str(tmp_path / "broken.json"))
    app2 = create_compat_app("openenv_echo", storage_root=tmp_path / "two")
    mounted2 = install_from_env(app2, storage_root=tmp_path / "two")
    assert mounted2 is not None and mounted2.price_table is None
    assert TestClient(app2).get("/annotation/pricing").json()["priced_models"] == []
    assert mounted2.service.runners["codex_app_server"].cost_enforcement(MODEL) is None
