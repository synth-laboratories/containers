"""Shared journal page/error handling, preserving the atomic cursor snapshot."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import re
from typing import Any

from ..event_log import RolloutEventLog, legacy_poll_cursor, poll_payload
from ..stream_cursor import StreamCursorError


def journal_events_payload(
    logs: Mapping[str, RolloutEventLog],
    retention: Callable[[RolloutEventLog], dict[str, Any]],
    rollout_id: str,
    after: int,
    limit: int,
    kinds: list[str] | tuple[str, ...] | None,
    *,
    ack: int | None,
    cursor: str | None,
    scoped: bool = False,
) -> dict[str, Any]:
    log = logs.get(rollout_id)
    if log is None:
        return {"error": "unknown_rollout", "status_code": 404}
    try:
        # Read/validate first: a malformed cursor must never mutate acknowledgment.
        payload = poll_payload(
            log, after=after, limit=limit, kinds=kinds, subject_id=rollout_id, cursor=cursor, scoped=True
        )
    except StreamCursorError as error:
        return {
            "error": error.code,
            "status_code": error.status_code,
            "recovery": "reattach_from_verified_snapshot",
        }
    except ValueError:
        return {"error": "invalid_page_limit", "status_code": 422}
    if ack is not None:
        try:
            with log._lock:
                log.record_ack(ack)
                payload = poll_payload(
                    log, after=after, limit=limit, kinds=kinds, subject_id=rollout_id, cursor=cursor, scoped=True
                )
        except StreamCursorError as error:
            return {"error": error.code, "status_code": error.status_code}
        except OSError:
            return {"error": "journal_ack_unavailable", "status_code": 503}
        except ValueError:
            return {"error": "invalid_ack", "status_code": 422}
        # Legacy ack remains stream-wide; it is not a per-view or per-client checkpoint.
    if cursor is None and not scoped:
        payload["cursor"] = legacy_poll_cursor(payload["cursor"], journal_state=True)
    payload["retention"] = retention(log)
    return payload


def sse_resume_position(
    log: RolloutEventLog,
    *,
    last_event_id: str | None,
    cursor: str | None,
    scoped: bool,
    kinds: list[str] | tuple[str, ...] | frozenset[str] | None,
) -> tuple[int, str | None, bool]:
    """Validate before streaming headers; Last-Event-ID supersedes the initial cursor."""
    selected = last_event_id if last_event_id is not None else cursor
    if selected is None:
        return 0, None, scoped
    if selected.startswith("sc1."):
        page = poll_payload(log, cursor=selected, kinds=kinds, limit=1)
        return page["cursor"]["after"], selected, True
    if scoped or cursor is not None or not re.fullmatch(r"(?:0|[1-9][0-9]{0,19})", selected):
        raise StreamCursorError("invalid_stream_cursor", status_code=422)
    after = int(selected)
    poll_payload(log, after=after, kinds=kinds, limit=1)
    return after, None, False
