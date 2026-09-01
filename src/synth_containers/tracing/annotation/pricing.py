"""Per-model price tables: the only way a runner turns tokens into dollars.

A paid annotation job is bounded twice: by a *billable-token ceiling* (uncached
input + output tokens, what the runner can observe mid-turn) and by a *cost
ceiling* (``limits.max_cost_usd``, what the broker reserved). Turning one into
the other needs a price, and the wrong price is worse than none: it would let a
job report a confident ``cost_usd`` that the reconciliation trusts.

So this module ships **no prices**. A ``PriceTable`` is loaded from an explicit
mapping, a JSON/TOML file, or the path in ``SYNTH_ANNOTATION_PRICE_TABLE``; a
model absent from the table is *unpriced* and the runner fails closed for it
(``cost_status="unavailable"``, paid submit refused unless the host's provider
proxy enforces the reservation).

File shape (JSON or TOML)::

    {
      "source": "openai-2026-08",             # optional label
      "models": {
        "gpt-5.6-luna": {
          "input_usd_per_million": 1.25,      # uncached input tokens
          "cached_input_usd_per_million": 0.125,
          "output_usd_per_million": 10.0
        }
      }
    }

Pricing never enters the idempotency key: a job's identity is what it computes
over which sealed trace, not what it was billed.
"""

from __future__ import annotations

import json
import math
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

PRICE_TABLE_ENV = "SYNTH_ANNOTATION_PRICE_TABLE"
COST_STATUS_PINNED = "pinned_price"
COST_STATUS_UNAVAILABLE = "unavailable"

_FIELDS = ("input_usd_per_million", "cached_input_usd_per_million", "output_usd_per_million")
_ALIASES = {
    "input": "input_usd_per_million",
    "uncached_input": "input_usd_per_million",
    "uncached_input_usd_per_million": "input_usd_per_million",
    "cached_input": "cached_input_usd_per_million",
    "cached": "cached_input_usd_per_million",
    "output": "output_usd_per_million",
}


class PriceTableError(ValueError):
    """The table is malformed; nothing about it is trusted."""


def _rate(value: Any, *, model: str, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PriceTableError(f"{model}: {field_name} must be a number, got {value!r}")
    rate = float(value)
    if not math.isfinite(rate) or rate < 0.0:
        raise PriceTableError(f"{model}: {field_name} must be finite and non-negative, got {value!r}")
    return rate


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """USD per million tokens for one resolved model name."""

    model: str
    input_usd_per_million: float
    cached_input_usd_per_million: float
    output_usd_per_million: float

    @classmethod
    def from_mapping(cls, model: str, payload: Mapping[str, Any]) -> "ModelPrice":
        if not isinstance(payload, Mapping):
            raise PriceTableError(f"{model}: price entry must be an object")
        normalized: dict[str, Any] = {}
        for key, value in payload.items():
            name = _ALIASES.get(str(key), str(key))
            if name in _FIELDS:
                normalized[name] = value
        missing = [name for name in _FIELDS if name not in normalized]
        if missing:
            raise PriceTableError(f"{model}: price entry is missing {', '.join(missing)}")
        return cls(
            model=model,
            input_usd_per_million=_rate(normalized["input_usd_per_million"], model=model, field_name="input_usd_per_million"),
            cached_input_usd_per_million=_rate(normalized["cached_input_usd_per_million"], model=model, field_name="cached_input_usd_per_million"),
            output_usd_per_million=_rate(normalized["output_usd_per_million"], model=model, field_name="output_usd_per_million"),
        )

    def cost_usd(
        self,
        *,
        input_tokens: int | None,
        output_tokens: int | None,
        cached_input_tokens: int | None = None,
    ) -> float:
        """Dollars for a usage report. ``input_tokens`` is the provider's total input (cached included)."""

        total_in = max(0, int(input_tokens or 0))
        cached = min(total_in, max(0, int(cached_input_tokens or 0)))
        uncached = total_in - cached
        out = max(0, int(output_tokens or 0))
        return (
            uncached * self.input_usd_per_million
            + cached * self.cached_input_usd_per_million
            + out * self.output_usd_per_million
        ) / 1_000_000

    @property
    def max_billable_usd_per_million(self) -> float:
        """The dearest rate a billable (uncached input or output) token can carry."""

        return max(self.input_usd_per_million, self.output_usd_per_million)

    def billable_token_ceiling(self, max_cost_usd: float | None) -> int | None:
        """Largest billable-token count guaranteed to cost at most ``max_cost_usd``.

        Conservative by construction: every billable token is charged at the
        dearest billable rate. Cached input is not billable-counted, so the
        runner also enforces the cost ceiling directly from priced usage.
        """

        if max_cost_usd is None:
            return None
        rate = self.max_billable_usd_per_million
        if rate <= 0.0:
            return None
        return int(math.floor(max(0.0, float(max_cost_usd)) * 1_000_000 / rate))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "input_usd_per_million": self.input_usd_per_million,
            "cached_input_usd_per_million": self.cached_input_usd_per_million,
            "output_usd_per_million": self.output_usd_per_million,
        }


class PriceTable:
    """Immutable mapping of resolved model name to ``ModelPrice``. Empty by default."""

    def __init__(self, prices: Mapping[str, ModelPrice] | None = None, *, source: str = "explicit") -> None:
        self._prices: dict[str, ModelPrice] = dict(prices or {})
        for model, price in self._prices.items():
            if not isinstance(price, ModelPrice) or price.model != model:
                raise PriceTableError(f"price for {model!r} must be a ModelPrice named {model!r}")
        self.source = source

    # -- construction -----------------------------------------------------------------

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, source: str = "dict") -> "PriceTable":
        if not isinstance(payload, Mapping):
            raise PriceTableError("price table must be an object")
        models = payload.get("models", payload)
        if not isinstance(models, Mapping):
            raise PriceTableError("price table 'models' must be an object")
        prices = {str(model): ModelPrice.from_mapping(str(model), entry) for model, entry in models.items()}
        label = payload.get("source") if "models" in payload else None
        return cls(prices, source=str(label) if isinstance(label, str) and label else source)

    @classmethod
    def from_file(cls, path: str | Path) -> "PriceTable":
        location = Path(path)
        try:
            raw = location.read_bytes()
        except OSError as error:
            raise PriceTableError(f"cannot read price table {location}: {error}") from error
        try:
            if location.suffix.lower() == ".toml":
                payload = tomllib.loads(raw.decode("utf-8"))
            else:
                payload = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as error:
            raise PriceTableError(f"price table {location} is not valid {location.suffix or 'json'}: {error}") from error
        return cls.from_dict(payload, source=str(location))

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None, *, variable: str = PRICE_TABLE_ENV) -> "PriceTable | None":
        """The table named by ``SYNTH_ANNOTATION_PRICE_TABLE``, or ``None`` when unset."""

        env = os.environ if environ is None else environ
        path = (env.get(variable) or "").strip()
        if not path:
            return None
        return cls.from_file(path)

    # -- queries ------------------------------------------------------------------------

    def get(self, model: str | None) -> ModelPrice | None:
        if not model:
            return None
        return self._prices.get(str(model))

    def __contains__(self, model: object) -> bool:
        return isinstance(model, str) and model in self._prices

    def __len__(self) -> int:
        return len(self._prices)

    def models(self) -> tuple[str, ...]:
        return tuple(sorted(self._prices))

    def cost(self, model: str | None, *, input_tokens: int | None, output_tokens: int | None, cached_input_tokens: int | None = None) -> tuple[float | None, str]:
        """``(cost_usd, cost_status)``: pinned when the model is priced, otherwise unavailable."""

        price = self.get(model)
        if price is None:
            return None, COST_STATUS_UNAVAILABLE
        return price.cost_usd(input_tokens=input_tokens, output_tokens=output_tokens, cached_input_tokens=cached_input_tokens), COST_STATUS_PINNED

    def billable_token_ceiling(self, model: str | None, max_cost_usd: float | None) -> int | None:
        price = self.get(model)
        return price.billable_token_ceiling(max_cost_usd) if price is not None else None

    def describe(self) -> dict[str, Any]:
        return {"source": self.source, "models": [self._prices[name].to_dict() for name in self.models()]}


__all__ = [
    "COST_STATUS_PINNED",
    "COST_STATUS_UNAVAILABLE",
    "PRICE_TABLE_ENV",
    "ModelPrice",
    "PriceTable",
    "PriceTableError",
]
