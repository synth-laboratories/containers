"""Incremental Server-Sent Events decoder preserving split network frames."""

from __future__ import annotations

from dataclasses import dataclass
import codecs


@dataclass(frozen=True, slots=True)
class SSEEvent:
    event: str | None
    data: str
    event_id: str | None = None


class SSEDecoder:
    def __init__(self) -> None:
        self._buffer = ""
        self._event: str | None = None
        self._event_id: str | None = None
        self._data: list[str] = []
        self._ready: list[SSEEvent] = []
        self._utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def feed(self, chunk: bytes | str) -> tuple[SSEEvent, ...]:
        text = self._utf8.decode(chunk, final=False) if isinstance(chunk, bytes) else chunk
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._line(line.rstrip("\r"))
        ready = tuple(self._ready)
        self._ready.clear()
        return ready

    def finish(self) -> tuple[SSEEvent, ...]:
        self._buffer += self._utf8.decode(b"", final=True)
        if self._buffer:
            self._line(self._buffer.rstrip("\r"))
            self._buffer = ""
        self._dispatch()
        ready = tuple(self._ready)
        self._ready.clear()
        return ready

    def _line(self, line: str) -> None:
        if not line:
            self._dispatch()
            return
        if line.startswith(":"):
            return
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "event":
            self._event = value
        elif field == "id":
            self._event_id = value
        elif field == "data":
            self._data.append(value)

    def _dispatch(self) -> None:
        if not self._data and self._event is None and self._event_id is None:
            return
        self._ready.append(
            SSEEvent(event=self._event, data="\n".join(self._data), event_id=self._event_id)
        )
        self._event = None
        self._event_id = None
        self._data = []


__all__ = ["SSEDecoder", "SSEEvent"]
