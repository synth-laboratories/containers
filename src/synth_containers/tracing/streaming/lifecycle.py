"""Trace Streaming Profile checks over a durable Containers log.

See: docs/specs/trace-streaming-profile-v1.md
     workshop/docs/live_evals.md (TS-A…E)
"""

from __future__ import annotations

from typing import Any

CORE_KINDS = frozenset(
    {
        "trace.opened",
        "env.episode.opened",
        "env.episode.closed",
        "policy.session.opened",
        "policy.session.closed",
        "observation",
        "action",
        "reward_signal",
        "frame",
        "artifact.declared",
        "artifact.available",
        "span.policy.opened",
        "span.policy.data",
        "span.policy.plan",
        "span.policy.closed",
        "span.step.opened",
        "span.step.closed",
        "span.agent.opened",
        "span.agent.closed",
        "span.verifier.opened",
        "span.verifier.closed",
        "capture.high_water",
        "capture.closed",
        "status",
        "trial.planned",
        "trial.launched",
        "tools",
        "stdout",
        "verifier",
        "start_session",
        "session.opened",
        "legal_actions",
        "stats",
        "invalid_action",
    }
)

OPEN_CLOSE = (
    ("span.policy.opened", "span.policy.closed"),
    ("span.step.opened", "span.step.closed"),
    ("span.agent.opened", "span.agent.closed"),
    ("span.verifier.opened", "span.verifier.closed"),
    ("env.episode.opened", "env.episode.closed"),
    ("policy.session.opened", "policy.session.closed"),
)


def semantic(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in events if not item.get("control")]


def first_semantic_kind(events: list[dict[str, Any]]) -> str | None:
    rows = semantic(events)
    if not rows:
        return None
    kind = rows[0].get("kind")
    return kind if isinstance(kind, str) else None


def capture_closed_count(events: list[dict[str, Any]]) -> int:
    return sum(1 for item in semantic(events) if item.get("kind") == "capture.closed")


def missing_coerced_to_zero(events: list[dict[str, Any]]) -> bool:
    for item in semantic(events):
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        if item.get("kind") == "reward_signal" and payload.get("value") is None:
            return True if payload.get("value") == 0 else False
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else None
        if usage is not None:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if key in usage and usage[key] is None:
                    continue
    return False


def nested_span_violations(events: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    kinds = [str(item.get("kind")) for item in semantic(events)]
    for opened, closed in OPEN_CLOSE:
        depth = 0
        for kind in kinds:
            if kind == opened:
                depth += 1
            elif kind == closed:
                if depth == 0:
                    issues.append(f"orphan_close:{closed}")
                else:
                    depth -= 1
        if depth:
            issues.append(f"unclosed:{opened}")
    policy_open = kinds.index("span.policy.opened") if "span.policy.opened" in kinds else None
    step_open = kinds.index("span.step.opened") if "span.step.opened" in kinds else None
    if policy_open is not None and step_open is not None and step_open < policy_open:
        issues.append("child_before_parent_open")
    return issues


def unknown_namespaced_kinds(events: list[dict[str, Any]]) -> list[str]:
    found: list[str] = []
    for item in semantic(events):
        kind = item.get("kind")
        if isinstance(kind, str) and kind not in CORE_KINDS and "." in kind:
            found.append(kind)
    return found


def lifecycle_violations(events: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    rows = semantic(events)
    if not rows:
        return ["empty_stream"]
    if rows[0].get("kind") != "trace.opened":
        issues.append("first_semantic_not_trace.opened")
    if sum(1 for item in rows if item.get("kind") == "trace.opened") != 1:
        issues.append("duplicate_or_missing_trace.opened")
    closed = capture_closed_count(events)
    if closed > 1:
        issues.append("duplicate_capture.closed")
    kinds = [str(item.get("kind")) for item in rows]
    if "capture.closed" in kinds:
        closed_at = kinds.index("capture.closed")
        later = kinds[closed_at + 1 :]
        if any(kind not in {"capture.closed"} and not kind.startswith("trace.") for kind in later):
            # data after capture.closed is a lifecycle regression
            if any(kind.endswith(".data") or kind.endswith(".opened") for kind in later):
                issues.append("post_close_data")
    issues.extend(nested_span_violations(events))
    return issues
