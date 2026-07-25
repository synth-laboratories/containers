"""Loss-declaring consumer projections derived only from sealed Trace V5."""

from __future__ import annotations

from typing import Any

from ..models.document import TraceDocumentV5
from ..models.messages import PartType


def transcript(document: TraceDocumentV5) -> dict[str, Any]:
    return {
        "trace_id": document.trace_id,
        "messages": [
            {
                "id": message.message_id,
                "role": str(message.role),
                "content": [
                    part.to_dict()
                    for part in message.parts
                    if str(part.type) != PartType.REASONING
                ],
            }
            for message in document.messages
        ],
        "losses": ["reasoning parts omitted"],
    }


def memory(document: TraceDocumentV5) -> dict[str, Any]:
    value = transcript(document)
    value["facts"] = [
        {"event_type": str(event.event_type), "payload": event.payload}
        for event in document.events
        if str(event.event_type).startswith(("environment.", "agent.", "react."))
    ]
    value["losses"].append("provider transport and token identity omitted")
    return value


def training(document: TraceDocumentV5) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    by_id = {message.message_id: message for message in document.messages}
    for span in document.spans:
        if str(span.span_kind) != "model_call":
            continue
        samples.append(
            {
                "span_id": span.span_id,
                "input": [by_id[item].to_dict() for item in span.input_message_ids if item in by_id],
                "output": [by_id[item].to_dict() for item in span.output_message_ids if item in by_id],
                "tokens": span.token_capture.to_dict() if span.token_capture else None,
                "usage": span.usage.to_dict() if span.usage else None,
            }
        )
    return {
        "trace_id": document.trace_id,
        "samples": samples,
        "losses": ["samples without captured token sequences have text-only supervision"],
    }


def logprobs(document: TraceDocumentV5) -> dict[str, Any]:
    return {
        "trace_id": document.trace_id,
        "calls": [
            {
                "span_id": span.span_id,
                "token_capture": span.token_capture.to_dict() if span.token_capture else None,
            }
            for span in document.spans
            if str(span.span_kind) == "model_call"
        ],
        "losses": ["provider did not expose logprobs" ]
        if not any(span.token_capture for span in document.spans)
        else [],
    }


def event_history(document: TraceDocumentV5) -> dict[str, Any]:
    return {
        "trace_id": document.trace_id,
        "events": [item.to_dict() for item in document.events],
        "losses": ["message graph and raw provider frames omitted"],
    }


PROJECTIONS = {
    "transcript": transcript,
    "memory": memory,
    "training": training,
    "logprobs": logprobs,
    "event_history": event_history,
}


__all__ = ["PROJECTIONS", "event_history", "logprobs", "memory", "training", "transcript"]
