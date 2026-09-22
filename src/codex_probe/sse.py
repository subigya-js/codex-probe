"""Server-sent event parsing for streamed LLM responses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SseEvent:
    """One complete server-sent event."""

    data: str
    event: str | None = None


class SseParser:
    """Incrementally parse arbitrary byte chunks into SSE events."""

    _BOUNDARIES = (
        b"\r\n\r\n",
        b"\n\n",
        b"\r\r",
    )

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._finished = False

    def feed(self, chunk: bytes) -> list[SseEvent]:
        """Accept one network chunk and return newly completed events."""

        if self._finished:
            raise RuntimeError("cannot feed a finished SSE parser")

        self._buffer.extend(chunk)
        events: list[SseEvent] = []

        while True:
            boundary = self._find_boundary()

            if boundary is None:
                break

            position, boundary_length = boundary
            raw_event = bytes(self._buffer[:position])

            del self._buffer[
                : position + boundary_length
            ]

            event = self._parse_event(raw_event)

            if event is not None:
                events.append(event)

        return events

    def finish(self) -> list[SseEvent]:
        """Finish parsing and return a final unterminated event."""

        if self._finished:
            raise RuntimeError("SSE parser is already finished")

        self._finished = True

        if not self._buffer:
            return []

        raw_event = bytes(self._buffer)
        self._buffer.clear()

        event = self._parse_event(raw_event)

        if event is None:
            return []

        return [event]

    def _find_boundary(self) -> tuple[int, int] | None:
        """Find the earliest complete SSE event boundary."""

        matches: list[tuple[int, int]] = []

        for boundary in self._BOUNDARIES:
            position = self._buffer.find(boundary)

            if position != -1:
                matches.append((position, len(boundary)))

        if not matches:
            return None

        return min(matches)

    @staticmethod
    def _parse_event(raw_event: bytes) -> SseEvent | None:
        """Converts raw SSE bytes into a structured Python object"""

        text = raw_event.decode("utf-8")
        event_type: str | None = None
        data_lines: list[str] = []

        for line in text.splitlines():
            if not line or line.startswith(":"):
                continue

            field, separator, value = line.partition(":")

            if separator and value.startswith(" "):
                value = value[1:]

            if field == "event":
                event_type = value
            elif field == "data":
                data_lines.append(value)

        if not data_lines:
            return None

        return SseEvent(
            data="\n".join(data_lines),
            event=event_type,
        )
