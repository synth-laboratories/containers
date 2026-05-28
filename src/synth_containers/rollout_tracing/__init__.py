"""Rollout trace format contracts for Synth container runtimes."""

from .v4 import (
    TRACE_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION_NAME,
    CanonicalChoice,
    CanonicalMessage,
    CanonicalRequest,
    CanonicalResponse,
    CanonicalUsage,
    ReasoningPart,
    RolloutTraceSpanV4,
    RolloutTraceV4,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    UnsupportedPart,
    chat_message_to_canonical,
    make_lm_span,
)

__all__ = [
    "TRACE_SCHEMA_VERSION",
    "TRACE_SCHEMA_VERSION_NAME",
    "CanonicalChoice",
    "CanonicalMessage",
    "CanonicalRequest",
    "CanonicalResponse",
    "CanonicalUsage",
    "ReasoningPart",
    "RolloutTraceSpanV4",
    "RolloutTraceV4",
    "TextPart",
    "ToolCallPart",
    "ToolResultPart",
    "UnsupportedPart",
    "chat_message_to_canonical",
    "make_lm_span",
]
