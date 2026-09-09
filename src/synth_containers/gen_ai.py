"""OpenTelemetry GenAI request attributes plus a Langfuse ``modelParameters`` bag.

OTel stores knobs as flat ``gen_ai.request.*`` span attributes (Development).
Langfuse stores the same knobs on a generation as an untyped ``modelParameters``
map. Prompts stay off this record: they are opt-in content events / ``input``.

The value of ``gen_ai.request.reasoning.level`` is the exact string sent to the
provider (``low`` / ``medium`` / ``high``, or whatever the wire used).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
GEN_AI_REQUEST_TOP_K = "gen_ai.request.top_k"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_SEED = "gen_ai.request.seed"
GEN_AI_REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
GEN_AI_REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
GEN_AI_REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"
GEN_AI_REQUEST_CHOICE_COUNT = "gen_ai.request.choice.count"
GEN_AI_REQUEST_REASONING_LEVEL = "gen_ai.request.reasoning.level"

_MAX_TOKEN_KEYS = ("max_tokens", "max_completion_tokens", "max_output_tokens")


def request_root(body: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Chat Completions body, or the nested Responses ``response`` object."""

    if not isinstance(body, Mapping):
        return {}
    if any(key in body for key in ("model", "messages", "input", "temperature", "reasoning")):
        return body
    nested = body.get("response")
    if isinstance(nested, Mapping):
        return nested
    return body


def request_observation(
    body: Mapping[str, Any] | None,
    *,
    operation: str = "chat",
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Filterable generation knobs. Omits unset keys. Never includes messages."""

    root = request_root(body)
    attributes = request_attributes(root, operation=operation)
    parameters = model_parameters(root, extras=extras)
    observation = dict(attributes)
    if parameters:
        observation["modelParameters"] = parameters
    return observation


def request_attributes(
    body: Mapping[str, Any] | None,
    *,
    operation: str = "chat",
) -> dict[str, Any]:
    """Flat OTel ``gen_ai.request.*`` attributes from a provider request body."""

    root = request_root(body)
    out: dict[str, Any] = {}
    if operation:
        out[GEN_AI_OPERATION_NAME] = str(operation)
    model = root.get("model")
    if isinstance(model, str) and model.strip():
        out[GEN_AI_REQUEST_MODEL] = model.strip()
    _put_float(out, GEN_AI_REQUEST_TEMPERATURE, root.get("temperature"))
    _put_float(out, GEN_AI_REQUEST_TOP_P, root.get("top_p") if "top_p" in root else root.get("topP"))
    top_k = root.get("top_k") if "top_k" in root else root.get("topK")
    if top_k is not None:
        _put_int(out, GEN_AI_REQUEST_TOP_K, top_k)
    for key in _MAX_TOKEN_KEYS:
        if key in root:
            _put_int(out, GEN_AI_REQUEST_MAX_TOKENS, root.get(key))
            break
    if "seed" in root:
        _put_int(out, GEN_AI_REQUEST_SEED, root.get("seed"))
    _put_float(out, GEN_AI_REQUEST_FREQUENCY_PENALTY, root.get("frequency_penalty"))
    _put_float(out, GEN_AI_REQUEST_PRESENCE_PENALTY, root.get("presence_penalty"))
    stop = _stop_sequences(root.get("stop"))
    if stop:
        out[GEN_AI_REQUEST_STOP_SEQUENCES] = stop
    if "n" in root:
        _put_int(out, GEN_AI_REQUEST_CHOICE_COUNT, root.get("n"))
    level = reasoning_level(root)
    if level:
        out[GEN_AI_REQUEST_REASONING_LEVEL] = level
    return out


def model_parameters(
    body: Mapping[str, Any] | None,
    *,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Langfuse-style generation ``modelParameters`` (wire names, camelCase extras)."""

    root = request_root(body)
    out: dict[str, Any] = {}
    _put_float(out, "temperature", root.get("temperature"))
    _put_float(out, "topP", root.get("top_p") if "top_p" in root else root.get("topP"))
    top_k = root.get("top_k") if "top_k" in root else root.get("topK")
    if top_k is not None:
        _put_int(out, "topK", top_k)
    for key, dest in (
        ("max_tokens", "maxTokens"),
        ("max_completion_tokens", "maxTokens"),
        ("max_output_tokens", "maxTokens"),
    ):
        if key in root and "maxTokens" not in out:
            _put_int(out, dest, root.get(key))
    if "seed" in root:
        _put_int(out, "seed", root.get("seed"))
    _put_float(out, "frequencyPenalty", root.get("frequency_penalty"))
    _put_float(out, "presencePenalty", root.get("presence_penalty"))
    stop = root.get("stop")
    if stop not in (None, "", []):
        out["stop"] = list(_stop_sequences(stop)) if not isinstance(stop, str) else stop
    if "n" in root:
        _put_int(out, "n", root.get("n"))
    if isinstance(root.get("reasoning"), Mapping):
        out["reasoning"] = dict(root["reasoning"])
    elif isinstance(root.get("reasoning_effort"), str) and root["reasoning_effort"].strip():
        out["reasoningEffort"] = root["reasoning_effort"].strip()
    thinking = root.get("thinking")
    if isinstance(thinking, Mapping):
        out["thinking"] = dict(thinking)
    if extras:
        for key, value in extras.items():
            if value is None or value == "":
                continue
            out[str(key)] = value
    return out


def reasoning_level(body: Mapping[str, Any] | None) -> str | None:
    """Exact provider reasoning/effort string, or ``None`` if the request omitted it."""

    root = request_root(body)
    reasoning = root.get("reasoning")
    if isinstance(reasoning, Mapping):
        for key in ("effort", "level"):
            value = reasoning.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in ("reasoning_effort", "reasoningEffort"):
        value = root.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    output_config = root.get("output_config")
    if isinstance(output_config, Mapping):
        value = output_config.get("effort")
        if isinstance(value, str) and value.strip():
            return value.strip()
    thinking = root.get("thinking")
    if isinstance(thinking, Mapping):
        value = thinking.get("effort")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def copy_observation(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Lift gen_ai / modelParameters keys off a journal or span payload."""

    if not isinstance(payload, Mapping):
        return {}
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "modelParameters" or str(key).startswith("gen_ai."):
            out[str(key)] = value
    return out


def _stop_sequences(value: Any) -> list[str]:
    if isinstance(value, str) and value:
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item not in (None, "")]
    return []


def _put_float(out: dict[str, Any], key: str, value: Any) -> None:
    if value is None or value == "":
        return
    try:
        out[key] = float(value)
    except (TypeError, ValueError):
        return


def _put_int(out: dict[str, Any], key: str, value: Any) -> None:
    if value is None or value == "":
        return
    try:
        out[key] = int(value)
    except (TypeError, ValueError):
        return


__all__ = [
    "GEN_AI_OPERATION_NAME",
    "GEN_AI_REQUEST_MAX_TOKENS",
    "GEN_AI_REQUEST_MODEL",
    "GEN_AI_REQUEST_REASONING_LEVEL",
    "GEN_AI_REQUEST_TEMPERATURE",
    "GEN_AI_REQUEST_TOP_K",
    "GEN_AI_REQUEST_TOP_P",
    "copy_observation",
    "model_parameters",
    "reasoning_level",
    "request_attributes",
    "request_observation",
    "request_root",
]
