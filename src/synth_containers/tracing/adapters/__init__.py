"""Adapters that import foreign trace and agent-event formats into V5."""

from .codex_jsonl import CodexImport, import_codex_jsonl
from .application import ApplicationEvent, ApplicationTraceAssembler
from .atif import atif_roundtrip_report, export_atif, import_atif, inspect_atif_import
from .anthropic_messages import AnthropicMessagesAdapter
from .base import (
    NormalizedMessage,
    NormalizedProviderResult,
    ProviderAdapter,
    ProviderAdapterRegistry,
)
from .experiments_v4 import import_experiments_trace_v4
from .openai_chat import (
    OpenAIChatAdapter,
    assemble_sse_frames,
    normalize_request_messages,
    normalize_unary_response,
    usage_from_provider,
)
from .openai_responses import OpenAIResponsesAdapter
from .v4 import import_rollout_trace_v4


def provider_adapters() -> ProviderAdapterRegistry:
    registry = ProviderAdapterRegistry()
    registry.register(OpenAIChatAdapter())
    registry.register(OpenAIResponsesAdapter())
    registry.register(AnthropicMessagesAdapter())
    return registry

__all__ = [
    "CodexImport",
    "ApplicationEvent",
    "ApplicationTraceAssembler",
    "AnthropicMessagesAdapter",
    "NormalizedMessage",
    "NormalizedProviderResult",
    "OpenAIChatAdapter",
    "OpenAIResponsesAdapter",
    "ProviderAdapter",
    "ProviderAdapterRegistry",
    "assemble_sse_frames",
    "atif_roundtrip_report",
    "export_atif",
    "import_codex_jsonl",
    "import_atif",
    "import_experiments_trace_v4",
    "import_rollout_trace_v4",
    "inspect_atif_import",
    "normalize_request_messages",
    "normalize_unary_response",
    "provider_adapters",
    "usage_from_provider",
]
