"""Data structures for captured LLM requests and responses."""


import json
from pathlib import Path
from threading import Lock
from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias
from .config import WireApi


JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class RecordedRequest:
    """Complete request information safe to store in a recording."""

    method: str
    path: str
    body: JsonValue

    def to_dict(self) -> dict[str, JsonValue]:
        """Convert the request into a JSON serializable dictionary"""

        return {
            "method": self.method,
            "path": self.path,
            "body": self.body
        }


@dataclass(frozen=True, slots=True)
class RecordedResponse:
    """Complete response information returned by the backend."""

    status: int
    streamed: bool
    body: JsonValue

    def to_dict(self) -> dict[str, JsonValue]:
        """Convert the response into a JSON serializable dictionary"""

        return {
            "status": self.status,
            "streamed": self.streamed,
            "body": self.body
        }


@dataclass(frozen=True, slots=True)
class RecordedBackend:
    """Non sensitive identity of the backend that handled a call."""

    name: str
    base_url: str
    wire_api: WireApi

    def to_dict(self) -> dict[str, JsonValue]:
        """Convert the backend identity into a dictionary"""

        return {
            "name": self.name,
            "base_url": self.base_url,
            "wire_api": self.wire_api
        }


@dataclass(frozen=True, slots=True)
class RecordedCall:
    """One complete LLM call captured by Codex-probe"""

    session_id: str
    call_index: int
    started_at: str
    latency_ms: float
    backend: RecordedBackend
    request: RecordedRequest
    response: RecordedResponse | None
    error: str | None

    def __post_init__(self) -> None:
        """Reject internally inconsistent call records."""

        if self.call_index < 1:
            raise ValueError("call_index must be at least 1")

        if self.latency_ms < 0:
            raise ValueError("latency_ms must not be negative")

        if (self.response is None) == (self.error is None):
            raise ValueError(
                "exactly one of response or error must be provided")

    def to_dict(self) -> dict[str, JsonValue]:
        "Convert the complete call into it's public log representation."

        response = (
            self.response.to_dict()
            if self.response is not None
            else None
        )

        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "call_index": self.call_index,
            "started_at": self.started_at,
            "latency_ms": self.latency_ms,
            "backend": self.backend.to_dict(),
            "request": self.request.to_dict(),
            "response": response,
            "error": self.error
        }


class SessionLog:
    """Collect and persist all calls belonging to one recorder session."""

    def __init__(self, log_dir: Path, session_id: str) -> None:
        self._log_dir = log_dir
        self._session_id = session_id,
        self._path = log_dir / f"{session_id}.jsonl"
        self._calls: dict[int, RecordedCall] = {}
        self._lock = Lock()
        self._closed = False

    @property
    def path(self) -> Path:
        """Return the path of this session's JSONL file."""
        return self._path

    def add(self, call: RecordedCall) -> None:
        """Add one completed call to this session."""

        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot add a call to a closed session.")

            if call.session_id != self._session_id:
                raise ValueError(
                    "Call session_id does not match this session.")

            if call.call_index in self._calls:
                raise ValueError(
                    f"call_index {call.call_index} is already recorded.")

            self._calls[call.call_index] = call

    def close(self) -> list[dict[str, JsonValue]]:
        """Write the ordered session log and return it's public records."""

        with self._lock:
            if self._closed:
                raise RuntimeError("Session log is already closed")

            ordered_calls = [
                self._calls[index]
                for index in sorted(self._calls)
            ]

            serialized_calls = [
                call.to_dict()
                for call in ordered_calls
            ]

            self._log_dir.mkdir(parents=True, exist_ok=True)

            with self._path.open(
                mode="x",
                encoding="urf-8",
            ) as log_file:
                for call in serialized_calls:
                    line = json.dumps(
                        call,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    log_file.write(line)
                    log_file.write("\n")

            self._closed = True
            return serialized_calls
