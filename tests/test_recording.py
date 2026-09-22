"""Tests for the CodexProbe recording schema and session logs."""

import json
from pathlib import Path

import pytest

from codex_probe.recording import (
    RecordedBackend,
    RecordedCall,
    RecordedRequest,
    RecordedResponse,
    SessionLog,
)


def _make_successful_call(
    *,
    call_index: int,
    session_id: str = "session-123",
    latency_ms: float = 125.5,
) -> RecordedCall:
    """Create a valid successful call for recording tests."""

    return RecordedCall(
        session_id=session_id,
        call_index=call_index,
        started_at="2026-09-21T08:15:30+00:00",
        latency_ms=latency_ms,
        backend=RecordedBackend(
            name="ollama-qwen",
            base_url="http://127.0.0.1:11434/v1",
            wire_api="responses",
        ),
        request=RecordedRequest(
            method="POST",
            path="/responses",
            body={
                "model": "qwen2.5-coder",
                "call": call_index,
            },
        ),
        response=RecordedResponse(
            status=200,
            streamed=True,
            body={
                "output_text": f"Response {call_index}",
            },
        ),
        error=None,
    )


def test_successful_call_has_expected_public_schema() -> None:
    request = RecordedRequest(
        method="POST",
        path="/responses",
        body={
            "model": "qwen2.5-coder",
            "input": "Fix the failing test.",
        },
    )

    response = RecordedResponse(
        status=200,
        streamed=True,
        body={
            "id": "response-123",
            "output_text": "I fixed the test.",
        },
    )

    backend = RecordedBackend(
        name="ollama-qwen",
        base_url="http://127.0.0.1:11434/v1",
        wire_api="responses",
    )

    call = RecordedCall(
        session_id="session-123",
        call_index=1,
        started_at="2026-09-21T08:15:30+00:00",
        latency_ms=125.5,
        backend=backend,
        request=request,
        response=response,
        error=None,
    )

    assert call.to_dict() == {
        "schema_version": 1,
        "session_id": "session-123",
        "call_index": 1,
        "started_at": "2026-09-21T08:15:30+00:00",
        "latency_ms": 125.5,
        "backend": {
            "name": "ollama-qwen",
            "base_url": "http://127.0.0.1:11434/v1",
            "wire_api": "responses",
        },
        "request": {
            "method": "POST",
            "path": "/responses",
            "body": {
                "model": "qwen2.5-coder",
                "input": "Fix the failing test.",
            },
        },
        "response": {
            "status": 200,
            "streamed": True,
            "body": {
                "id": "response-123",
                "output_text": "I fixed the test.",
            },
        },
        "error": None,
    }


def test_failed_call_has_expected_public_schema() -> None:
    call = RecordedCall(
        session_id="session-123",
        call_index=1,
        started_at="2026-09-21T08:15:30+00:00",
        latency_ms=25.0,
        backend=RecordedBackend(
            name="ollama-qwen",
            base_url="http://127.0.0.1:11434/v1",
            wire_api="responses",
        ),
        request=RecordedRequest(
            method="POST",
            path="/responses",
            body={"model": "qwen2.5-coder"},
        ),
        response=None,
        error="Connection refused",
    )

    serialized = call.to_dict()

    assert serialized["response"] is None
    assert serialized["error"] == "Connection refused"


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (None, None),
        (
            RecordedResponse(
                status=200,
                streamed=False,
                body={"output_text": "Completed"},
            ),
            "Connection failed",
        ),
    ],
)
def test_call_requires_exactly_one_response_or_error(
    response: RecordedResponse | None,
    error: str | None,
) -> None:
    with pytest.raises(
        ValueError,
        match="exactly one of response or error must be provided",
    ):
        RecordedCall(
            session_id="session-123",
            call_index=1,
            started_at="2026-09-21T08:15:30+00:00",
            latency_ms=125.5,
            backend=RecordedBackend(
                name="ollama-qwen",
                base_url="http://127.0.0.1:11434/v1",
                wire_api="responses",
            ),
            request=RecordedRequest(
                method="POST",
                path="/responses",
                body={"model": "qwen2.5-coder"},
            ),
            response=response,
            error=error,
        )


@pytest.mark.parametrize(
    "call_index",
    [
        0,
        -1,
    ],
)
def test_non_positive_call_index_is_rejected(
    call_index: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="call_index must be at least 1",
    ):
        _make_successful_call(
            call_index=call_index,
        )


def test_negative_latency_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="latency_ms must not be negative",
    ):
        _make_successful_call(
            call_index=1,
            latency_ms=-0.1,
        )


def test_session_log_writes_calls_in_call_index_order(
    tmp_path: Path,
) -> None:
    log_dir = tmp_path / "nested" / "logs"
    session_log = SessionLog(
        log_dir=log_dir,
        session_id="session-123",
    )

    second_call = _make_successful_call(call_index=2)
    first_call = _make_successful_call(call_index=1)

    session_log.add(second_call)
    session_log.add(first_call)

    returned_calls = session_log.close()

    assert [
        call["call_index"]
        for call in returned_calls
    ] == [1, 2]

    assert session_log.path == (
        log_dir / "session-123.jsonl"
    )
    assert session_log.path.exists()

    lines = session_log.path.read_text(
        encoding="utf-8"
    ).splitlines()

    persisted_calls = [
        json.loads(line)
        for line in lines
    ]

    assert persisted_calls == returned_calls


def test_session_log_rejects_call_from_another_session(
    tmp_path: Path,
) -> None:
    session_log = SessionLog(
        log_dir=tmp_path,
        session_id="session-123",
    )

    another_session_call = _make_successful_call(
        call_index=1,
        session_id="session-999",
    )

    with pytest.raises(
        ValueError,
        match="does not match this session",
    ):
        session_log.add(another_session_call)


def test_session_log_rejects_duplicate_call_index(
    tmp_path: Path,
) -> None:
    session_log = SessionLog(
        log_dir=tmp_path,
        session_id="session-123",
    )

    session_log.add(
        _make_successful_call(call_index=1)
    )

    with pytest.raises(
        ValueError,
        match="call_index 1 is already recorded",
    ):
        session_log.add(
            _make_successful_call(call_index=1)
        )


def test_session_log_rejects_add_after_close(
    tmp_path: Path,
) -> None:
    session_log = SessionLog(
        log_dir=tmp_path,
        session_id="session-123",
    )

    session_log.close()

    with pytest.raises(
        RuntimeError,
        match="closed session",
    ):
        session_log.add(
            _make_successful_call(call_index=1)
        )


def test_session_log_rejects_second_close(
    tmp_path: Path,
) -> None:
    session_log = SessionLog(
        log_dir=tmp_path,
        session_id="session-123",
    )

    session_log.close()

    with pytest.raises(
        RuntimeError,
        match="already closed",
    ):
        session_log.close()
