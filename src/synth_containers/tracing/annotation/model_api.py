"""Single-shot model annotators: one structured completion, no tools.

The cheap middle tier between deterministic programs and agentic Codex tasks.
The runner is given a *completion function* by the host (Workshop's provider
proxy, a direct SDK call, a test stub); it never holds credentials itself. It
renders the same instructions a Codex task would get, plus a bounded, selector-
rich digest of the trace produced by the read-only tools, and asks for a
``synth.annotation-proposal.v1`` object. Validation is identical downstream.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ..canonical import utc_now
from ..models.standards import ProducerKind
from .definitions import RunnerKind
from .execution_trace import ExecutionCapture
from .jobs import RUNNER_VERSION, AnnotationJobErrorCode, AnnotationJobErrorV1, AnnotationJobUsageV1
from .pricing import COST_STATUS_UNAVAILABLE, ModelPrice, PriceTable
from .proposal import STRICT_PROPOSAL_JSON_SCHEMA, normalize_strict_proposal
from .validation import producer_for


@dataclass(frozen=True, slots=True)
class CompletionResult:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    provider_request_id: str | None = None
    cached_input_tokens: int | None = None


class CompletionFunction(Protocol):
    """``(model, instructions, digest_text, json_schema, max_output_tokens) -> CompletionResult``."""

    def __call__(self, *, model: str, instructions: str, context: str, schema: dict[str, Any], max_output_tokens: int) -> CompletionResult: ...


def render_trace_digest(tools: Any, *, max_messages: int = 60, max_chars_per_message: int = 1200) -> str:
    """A bounded, selector-bearing view of the trace built only through the read-only tools."""

    manifest = tools.call("trace_get_manifest", {})
    lines = [
        f"trace_id: {manifest['trace_id']}",
        f"trace_digest: {manifest['trace_digest']}",
        f"counts: {json.dumps(manifest['counts'])}",
        "",
        "## messages (cite with {kind: message, entity_id, quote})",
    ]
    listing = tools.call("trace_list_entities", {"kind": "message", "limit": min(max_messages, 200)})
    for item in listing["items"]:
        message = tools.call("trace_get_message", {"message_id": item["message_id"], "max_chars": max_chars_per_message})
        lines.append(f"- [{item['role']}] entity_id={item['message_id']} call_index={item.get('call_index')}")
        lines.append("  " + message["text"].replace("\n", "\n  "))
    steps = tools.call("trace_list_entities", {"kind": "span", "span_kind": "environment_step", "limit": 200})
    if steps["items"]:
        lines.append("")
        lines.append("## environment steps (cite with {kind: span, entity_id})")
        for span in steps["items"]:
            lines.append(f"- entity_id={span['span_id']} step={span.get('step_index')} action={span.get('action')} transition={span.get('transition')} reason={span.get('reason')}")
    events = tools.call("trace_list_entities", {"kind": "event", "event_type": "craftax.transcript", "limit": 200})
    unlocked = [event for event in events["items"] if event.get("kind") == "achievement_unlocked"]
    if unlocked:
        lines.append("")
        lines.append("## achievement events (cite with {kind: event, entity_id})")
        for event in unlocked:
            lines.append(f"- entity_id={event['event_id']} step={event.get('step_index')}")
    return "\n".join(lines) + "\n"


class ModelApiRunner:
    """``AnnotatorRunner`` for ``RunnerKind.MODEL_API``."""

    kind = RunnerKind.MODEL_API.value
    version = "model_api@1"

    def __init__(
        self,
        complete: CompletionFunction,
        *,
        default_model: str | None = None,
        default_effort: str | None = None,
        max_output_tokens: int = 8000,
        usd_per_million_tokens: float | None = None,
        proxy_enforces_reservation: bool = False,
        digest_renderer: Callable[[Any], str] = render_trace_digest,
        price_table: PriceTable | None = None,
    ) -> None:
        self.complete = complete
        self.default_model = default_model
        self.default_effort = default_effort
        self.max_output_tokens = max_output_tokens
        # Legacy flat price (token ceiling only) and the per-model price table,
        # which wins for any model it prices and also yields ``cost_usd``.
        self.usd_per_million_tokens = usd_per_million_tokens
        self.price_table = price_table
        self.proxy_enforces_reservation = proxy_enforces_reservation
        self.digest_renderer = digest_renderer

    def resolve_model(self, requested: str | None, definition_model: str | None) -> str | None:
        return requested or definition_model or self.default_model

    def resolve_effort(self, requested: str | None, program_default: str | None) -> str | None:
        return requested or program_default or self.default_effort

    def price_for(self, model: str | None) -> ModelPrice | None:
        return self.price_table.get(model) if self.price_table is not None else None

    def cost_enforcement(self, model: str | None = None) -> str | None:
        if self.price_for(model) is not None or self.usd_per_million_tokens:
            return "pinned_price"
        if self.proxy_enforces_reservation:
            return "provider_proxy"
        return None

    def token_ceiling(self, limits: Any, model: str | None = None) -> int | None:
        """Billable-token ceiling: the declared token cap, tightened by the cost cap at the dearest priced rate."""

        ceiling = limits.max_total_tokens
        by_cost: int | None = None
        if limits.max_cost_usd is not None:
            price = self.price_for(model)
            if price is not None:
                by_cost = price.billable_token_ceiling(limits.max_cost_usd)
            elif self.usd_per_million_tokens:
                by_cost = int(limits.max_cost_usd / self.usd_per_million_tokens * 1_000_000)
        if by_cost is not None:
            ceiling = by_cost if ceiling is None else min(ceiling, by_cost)
        return ceiling

    def run(self, context: Any) -> Any:
        from .service import RunOutcome

        job = context.job
        started = utc_now()
        clock = time.monotonic()
        model = job.request.model or self.default_model or ""
        error: AnnotationJobErrorV1 | None = None
        proposal: Any = None
        result: CompletionResult | None = None
        try:
            digest = self.digest_renderer(context.tools)
            result = self.complete(
                model=model,
                instructions=context.instructions_text,
                context=digest,
                schema=STRICT_PROPOSAL_JSON_SCHEMA,
                max_output_tokens=self.max_output_tokens,
            )
            try:
                proposal = normalize_strict_proposal(json.loads(result.text))
            except ValueError as bad:
                error = AnnotationJobErrorV1(code=AnnotationJobErrorCode.MALFORMED_OUTPUT, message=f"completion is not JSON: {bad}")
        except Exception as exc:  # noqa: BLE001 - provider/tool failures are typed job failures
            error = AnnotationJobErrorV1(code=AnnotationJobErrorCode.RUNNER_UNAVAILABLE, message=f"{type(exc).__name__}: {exc}")
        limits = job.request.limits
        total = result.total_tokens if result else None
        ceiling = self.token_ceiling(limits, model)
        # The ceiling bounds billable tokens (uncached input + output) when the
        # provider itemizes them; a bare total is counted in full.
        billable: int | None = None
        if result is not None and isinstance(result.input_tokens, int):
            billable = result.input_tokens - int(result.cached_input_tokens or 0) + int(result.output_tokens or 0)
        counted = billable if billable is not None else total
        if error is None and ceiling is not None and isinstance(counted, int) and counted > ceiling:
            error = AnnotationJobErrorV1(code=AnnotationJobErrorCode.TOKEN_LIMIT_EXCEEDED, message=f"completion used {counted} billable tokens, ceiling {ceiling}")
            proposal = None
        cost: float | None = None
        cost_status = COST_STATUS_UNAVAILABLE
        if result is not None:
            if result.cost_usd is not None:
                cost, cost_status = result.cost_usd, "reported"
            elif self.price_for(model) is not None and isinstance(result.input_tokens, int):
                cost, cost_status = self.price_table.cost(  # type: ignore[union-attr]
                    model,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    cached_input_tokens=result.cached_input_tokens,
                )
        if error is None and cost is not None and limits.max_cost_usd is not None and cost > limits.max_cost_usd:
            error = AnnotationJobErrorV1(code=AnnotationJobErrorCode.COST_LIMIT_EXCEEDED, message=f"completion cost ${cost:.6f} ({cost_status}), ceiling ${limits.max_cost_usd:.6f}")
            proposal = None
        usage = AnnotationJobUsageV1(
            tool_calls=len(context.tools.calls),
            tool_bytes=context.tools.total_bytes,
            input_tokens=result.input_tokens if result else None,
            output_tokens=result.output_tokens if result else None,
            cached_input_tokens=result.cached_input_tokens if result else None,
            total_tokens=total,
            cost_usd=cost,
            cost_status=cost_status,
            wall_time_seconds=time.monotonic() - clock,
        )
        capture = ExecutionCapture(
            started_at=started,
            ended_at=utc_now(),
            instructions_digest=context.instructions_digest,
            tool_calls=tuple(context.tools.calls),
            final_output=proposal if isinstance(proposal, dict) else None,
            final_output_text=None if isinstance(proposal, dict) else (result.text if result else None),
            usage=usage,
            runner_kind=self.kind,
            model=model,
            reasoning_effort=job.request.reasoning_effort,
            transport_events=(({"kind": "provider_request", "id": result.provider_request_id},) if result and result.provider_request_id else ()),
            error=error.message if error else None,
        )
        producer = producer_for(context.entry.definition, kind=ProducerKind.MODEL, name="model_api", version=RUNNER_VERSION, model=model, config_digest=context.entry.program.content_digest)
        return RunOutcome(proposal=proposal, capture=capture, error=error, producer=producer)


__all__ = ["CompletionFunction", "CompletionResult", "ModelApiRunner", "render_trace_digest"]
