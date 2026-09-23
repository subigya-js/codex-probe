"""Parse and reconstruct streamed LLM responses."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

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

@dataclass
class _ToolCallParts:
    """Fragments collected for one streamed tool call."""

    id: str = ""
    type: str = "function"
    name: str = ""
    arguments: list[str] = field(default_factory=list)


@dataclass
class _ChoiceParts:
    """Fragments collected for one streamed response choice."""

    role: str = "assistant"
    content: list[str] = field(default_factory=list)
    refusal: list[str] = field(default_factory=list)
    tool_calls: dict[int, _ToolCallParts] = field(
        default_factory=dict
    )
    finish_reason: str | None = None


class ChatCompletionsReassembler:
    """Reconstruct a complete Chat Completions response."""

    def __init__(self) -> None:
        self._id: str | None = None
        self._created: int | None = None
        self._model: str | None = None
        self._system_fingerprint: str | None = None
        self._usage: dict[str, JsonValue] | None = None
        self._choices: dict[int, _ChoiceParts] = {}

    def accept(self, event: SseEvent) -> None:
        """Process one Chat Completions stream event."""

        if event.data == "[DONE]":
            return

        payload = json.loads(event.data)

        if not isinstance(payload, dict):
            raise ValueError(
                "Chat stream event must be a JSON object"
            )

        self._save_metadata(payload)

        choices = payload.get("choices", [])

        if not isinstance(choices, list):
            raise ValueError(
                "Chat stream choices must be a list"
            )

        for choice in choices:
            if not isinstance(choice, dict):
                raise ValueError(
                    "Chat stream choice must be an object"
                )

            self._accept_choice(choice)

    def _save_metadata(
        self,
        payload: dict[str, object],
    ) -> None:
        """Save response information shared by every chunk."""

        response_id = payload.get("id")

        if isinstance(response_id, str):
            self._id = response_id

        created = payload.get("created")

        if isinstance(created, int):
            self._created = created

        model = payload.get("model")

        if isinstance(model, str):
            self._model = model

        fingerprint = payload.get("system_fingerprint")

        if isinstance(fingerprint, str):
            self._system_fingerprint = fingerprint

        usage = payload.get("usage")

        if isinstance(usage, dict):
            self._usage = usage

    def _accept_choice(
        self,
        choice: dict[str, object],
    ) -> None:
        """Add one streamed choice fragment."""

        index = choice.get("index")

        if not isinstance(index, int):
            raise ValueError(
                "Chat stream choice must have an integer index"
            )

        parts = self._choices.setdefault(
            index,
            _ChoiceParts(),
        )

        finish_reason = choice.get("finish_reason")

        if isinstance(finish_reason, str):
            parts.finish_reason = finish_reason

        delta = choice.get("delta")

        if not isinstance(delta, dict):
            return

        role = delta.get("role")

        if isinstance(role, str):
            parts.role = role

        content = delta.get("content")

        if isinstance(content, str):
            parts.content.append(content)

        refusal = delta.get("refusal")

        if isinstance(refusal, str):
            parts.refusal.append(refusal)

        tool_calls = delta.get("tool_calls")

        if tool_calls is None:
            return

        if not isinstance(tool_calls, list):
            raise ValueError(
                "Chat stream tool_calls must be a list"
            )

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                raise ValueError(
                    "Chat stream tool call must be an object"
                )

            self._accept_tool_call(
                parts,
                tool_call,
            )

    def _accept_tool_call(
        self,
        choice: _ChoiceParts,
        tool_call: dict[str, object],
    ) -> None:
        """Add one streamed tool-call fragment."""

        index = tool_call.get("index")

        if not isinstance(index, int):
            raise ValueError(
                "Chat tool call must have an integer index"
            )

        parts = choice.tool_calls.setdefault(
            index,
            _ToolCallParts(),
        )

        tool_call_id = tool_call.get("id")

        if isinstance(tool_call_id, str):
            parts.id = tool_call_id

        tool_type = tool_call.get("type")

        if isinstance(tool_type, str):
            parts.type = tool_type

        function = tool_call.get("function")

        if not isinstance(function, dict):
            return

        name = function.get("name")

        if isinstance(name, str):
            parts.name = name

        arguments = function.get("arguments")

        if isinstance(arguments, str):
            parts.arguments.append(arguments)

    def finish(self) -> dict[str, JsonValue]:
        """Build and return the complete response."""

        if not self._choices:
            raise ValueError(
                "Chat stream ended without any choices"
            )

        completed_choices: list[JsonValue] = []

        for index in sorted(self._choices):
            parts = self._choices[index]
            message = self._build_message(parts)

            completed_choices.append(
                {
                    "index": index,
                    "message": message,
                    "finish_reason": parts.finish_reason,
                    "logprobs": None,
                }
            )

        response: dict[str, JsonValue] = {
            "id": self._id,
            "object": "chat.completion",
            "created": self._created,
            "model": self._model,
            "choices": completed_choices,
        }

        if self._system_fingerprint is not None:
            response["system_fingerprint"] = (
                self._system_fingerprint
            )

        if self._usage is not None:
            response["usage"] = self._usage

        return response

    @staticmethod
    def _build_message(
        parts: _ChoiceParts,
    ) -> dict[str, JsonValue]:
        """Build one complete assistant message."""

        content = (
            "".join(parts.content)
            if parts.content
            else None
        )

        message: dict[str, JsonValue] = {
            "role": parts.role,
            "content": content,
        }

        if parts.refusal:
            message["refusal"] = "".join(parts.refusal)

        if parts.tool_calls:
            completed_tool_calls: list[JsonValue] = []

            for index in sorted(parts.tool_calls):
                tool = parts.tool_calls[index]

                completed_tool_calls.append(
                    {
                        "id": tool.id,
                        "type": tool.type,
                        "function": {
                            "name": tool.name,
                            "arguments": "".join(
                                tool.arguments
                            ),
                        },
                    }
                )

            message["tool_calls"] = completed_tool_calls

        return message