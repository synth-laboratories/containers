"""Bounded read-only terminal projection; see docs/specs/rhodes-client-v1.md."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import math
import shutil
import unicodedata
from typing import Any, TextIO

from .pools import PoolClientError, PoolRolloutTimeout
from .rhodes_client import RhodesClient


def _cell(value: Any, width: int) -> str:
    """Never let event text move the terminal cursor or escape its column."""
    result = []
    used = 0
    for character in str(value)[:1024]:
        if unicodedata.category(character).startswith("C"):
            character = "?"
        size = 0 if unicodedata.combining(character) else (
            2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
        )
        if used + size > width:
            break
        result.append(character)
        used += size
    return "".join(result) + " " * max(0, width - used)


class RhodesTerminal:
    """At most ten visible lanes, no inferred progress or journal mutations."""

    def __init__(self, stream: TextIO, rollout_ids: Sequence[str]):
        self.stream = stream
        self.rows = {
            identity: {"status": "unobserved", "sequence": "?", "updated": "?", "latest": "?"}
            for identity in rollout_ids
        }
        self.previous_lines = 0

    def __call__(self, event: dict[str, Any]) -> None:
        identity = event.get("rollout_id")
        if identity not in self.rows:
            raise PoolClientError("Rhodes watch event has no selected rollout identity")
        row = self.rows[identity]
        kind = event.get("event_type")
        # Publication events also have a status, belonging to publication. Only
        # explicit execution transitions update the rollout's status column.
        if kind in {
            "rollout.queued", "rollout.running", "rollout.completed", "rollout.failed",
            "rollout.cancelled",
        }:
            row["status"] = kind.removeprefix("rollout.")
        row.update(sequence=event["sequence"], updated=event.get("created_at", "?"), latest=kind or "?")
        self.render()

    def render(self) -> None:
        width, height = shutil.get_terminal_size((100, 24))
        width = max(1, width - 1)
        count = min(10, max(0, height - 3))
        sidebar = min(46, max(12, width // 2))
        left = max(0, width - sidebar - 3)
        lines = [_cell(f"Rhodes  {len(self.rows)} rollouts", width)]
        # Retain selected order. Completed rows move below still-active rows so
        # a long cohort does not hide its active tail behind ten finished runs.
        rows = sorted(self.rows.items(), key=lambda item: item[1]["status"] in {
            "completed", "failed", "cancelled"
        })
        for identity, row in rows[:count]:
            label = identity if len(identity) <= 24 else identity[:10] + "…" + identity[-8:]
            main = f"{label}  event={row['sequence']}"
            try:
                updated = datetime.fromisoformat(str(row["updated"]))
                clock = updated.astimezone(timezone.utc).strftime("%H:%M:%SZ") if updated.tzinfo else "?"
            except ValueError:
                clock = "?"
            status = f"{row['status']}  {clock}  {str(row['latest']).removeprefix('rollout.')}"
            line = _cell(main, left) + " | " + _cell(status, sidebar)
            lines.append(_cell(line, width))
        if len(rows) > count:
            lines.append(_cell(f"+{len(rows) - count} additional rollouts", width))
        if self.previous_lines:
            self.stream.write(f"\x1b[{self.previous_lines}A")
        for line in lines:
            self.stream.write("\x1b[2K" + line + "\n")
        for _ in range(max(0, self.previous_lines - len(lines))):
            self.stream.write("\x1b[2K\n")
        self.previous_lines = max(len(lines), self.previous_lines)
        self.stream.flush()


async def watch_rollouts(
    client: RhodesClient,
    rollout_ids: Sequence[str],
    *,
    cursors: Mapping[str, int],
    timeout_seconds: float,
    emit: Callable[[dict[str, Any]], None],
) -> None:
    """Multiplex original per-rollout journals; disconnect never cancels runs."""
    if not 1 <= len(rollout_ids) <= 100 or len(set(rollout_ids)) != len(rollout_ids):
        raise ValueError("Watch requires between1 and100 distinct rollout IDs")
    if any(not isinstance(identity, str) or not 0 < len(identity) <= 256 for identity in rollout_ids):
        raise ValueError("Watch requires bounded nonempty rollout IDs")
    if set(cursors) - set(rollout_ids) or any(
        type(cursor) is not int or not 0 <= cursor <= 2**63 - 1 for cursor in cursors.values()
    ):
        raise ValueError("Watch cursors must belong to selected IDs and be nonnegative integers")
    if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 86400:
        raise ValueError("Watch timeout must be positive and at most one day")
    queue: asyncio.Queue[tuple[dict[str, Any] | None, Exception | None]] = asyncio.Queue(maxsize=32)
    readers = asyncio.Semaphore(32)

    async def read(identity: str) -> None:
        try:
            async with readers:
                async for event in client.follow_events(
                    identity, after_sequence=cursors.get(identity, 0), timeout_seconds=timeout_seconds
                ):
                    if event.get("rollout_id") != identity:
                        raise PoolClientError("Multiplexed Rhodes event has no exact rollout identity")
                    await queue.put((event, None))
        except Exception as error:
            await queue.put((None, error))
        else:
            await queue.put((None, None))

    tasks: list[asyncio.Task[None]] = []
    observation = asyncio.timeout(timeout_seconds)
    try:
        async with observation:
            tasks = [asyncio.create_task(read(identity)) for identity in rollout_ids]
            finished = 0
            while finished < len(tasks):
                event, error = await queue.get()
                if error is not None:
                    raise error
                if event is None:
                    finished += 1
                else:
                    emit(event)
    except TimeoutError as error:
        if not observation.expired():
            raise
        raise PoolRolloutTimeout("Rhodes watch timed out (not cancelled)") from error
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
