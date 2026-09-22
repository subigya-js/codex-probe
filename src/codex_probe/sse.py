"""Parse and reconstruct streamed LLM responses."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .recording import JsonValue


@dataclass(frozen=True, slots=True)
class SseEvent:
    """One complete server-sent event."""

    data: str
    event: str | None = None


class SseParser:
    """Convert incoming byte chunks into complete SSE events."""

    BOUNDARIES = (
        b"\r\n\r\n",
        b"\n\n",
        b"\r\r",
    )

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._finished = False

    def feed(self, chunk: bytes) -> list[SseEvent]:
        """Add a network chunk and return completed events."""

        if self._finished:
            raise RuntimeError("cannot feed a finished SSE parser")

        self._buffer.extend(chunk)
        events: list[SseEvent] = []

        while True:
            boundary = self._find_boundary()

            if boundary is None:
                return events

            position, length = boundary
            raw_event = bytes(self._buffer[:position])

            del self._buffer[:position + length]

            event = self._parse_event(raw_event)

            if event is not None:
                events.append(event)

    def finish(self) -> list[SseEvent]:
        """Return any final event remaining in the buffer."""

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
        """Find the first blank line in the buffer."""

        matches: list[tuple[int, int]] = []

        for boundary in self.BOUNDARIES:
            position = self._buffer.find(boundary)

            if position != -1:
                matches.append((position, len(boundary)))

        if not matches:
            return None

        return min(matches)

    @staticmethod
    def _parse_event(
        raw_event: bytes,
    ) -> SseEvent | None:
        """Convert one SSE block into an SseEvent."""

        text = raw_event.decode("utf-8")
        event_name: str | None = None
        data_lines: list[str] = []

        for line in text.splitlines():
            if line.startswith(":"):
                continue

            if line.startswith("event:"):
                event_name = line[len("event:"):]

                if event_name.startswith(" "):
                    event_name = event_name[1:]

            if line.startswith("data:"):
                data = line[len("data:"):]

                if data.startswith(" "):
                    data = data[1:]

                data_lines.append(data)

        if not data_lines:
            return None

        return SseEvent(
            data="\n".join(data_lines),
            event=event_name,
        )


class ResponsesReassembler:
    """Extract a complete Responses API response."""

    def __init__(self) -> None:
        self._response: dict[str, JsonValue] | None = None

    def accept(self, event: SseEvent) -> None:
        """Process one parsed SSE event."""

        if event.data == "[DONE]":
            return

        payload = json.loads(event.data)

        if not isinstance(payload, dict):
            raise ValueError(
                "Responses stream event must be a JSON object"
            )

        event_type = payload.get("type")

        terminal_event_types = (
            "response.completed",
            "response.failed",
            "response.incomplete",
        )

        if event_type not in terminal_event_types:
            return

        response = payload.get("response")

        if not isinstance(response, dict):
            raise ValueError(
                "terminal event must contain a response object"
            )

        self._response = response

    def finish(self) -> dict[str, JsonValue]:
        """Return the complete response object."""

        if self._response is None:
            raise ValueError(
                "stream ended without a terminal response"
            )

        return self._response
