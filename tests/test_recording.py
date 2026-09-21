"""Tests for the CodexProbe recording schema."""

from codex_probe.recording import (
    RecordedBackend,
    RecordedCall,
    RecordedRequest,
    RecordedResponse,
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