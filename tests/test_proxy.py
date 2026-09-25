"""Tests for HTTP proxy passthrough and recording."""

import asyncio
import gzip
import json
from pathlib import Path

import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer

from codex_probe.config import parse_config
from codex_probe.proxy import create_proxy_app
from codex_probe.recording import SessionLog


def _make_session_log(
    tmp_path: Path,
) -> SessionLog:
    """Create a temporary session log for one proxy test."""

    return SessionLog(
        log_dir=tmp_path / "logs",
        session_id="test-session",
    )


def test_non_streaming_request_and_response_passthrough(
    tmp_path: Path,
) -> None:
    asyncio.run(
        _test_non_streaming_request_and_response_passthrough(
            tmp_path
        )
    )


async def _test_non_streaming_request_and_response_passthrough(
    tmp_path: Path,
) -> None:
    received_request: dict[str, object] = {}

    async def backend_handler(
        request: web.Request,
    ) -> web.Response:
        received_request["method"] = request.method
        received_request["path"] = request.path
        received_request["query"] = request.query_string
        received_request["content_type"] = (
            request.headers.get("Content-Type")
        )
        received_request["test_header"] = (
            request.headers.get("X-Test")
        )
        received_request["authorization"] = (
            request.headers.get("Authorization")
        )
        received_request["cookie"] = (
            request.headers.get("Cookie")
        )
        received_request["body"] = await request.read()

        return web.Response(
            status=201,
            headers={
                "Content-Type": "application/json",
                "X-Backend": "mock-backend",
            },
            body=b'{"result":"created"}',
        )

    backend_app = web.Application()

    backend_app.router.add_post(
        "/v1/chat/completions",
        backend_handler,
    )

    backend_server = TestServer(backend_app)
    await backend_server.start_server()

    config = parse_config(
        {
            "backend": {
                "name": "mock",
                "base_url": str(
                    backend_server.make_url("")
                ).rstrip("/"),
                "wire_api": "chat",
                "auth": {
                    "mode": "forward",
                },
            }
        }
    )

    session_log = _make_session_log(tmp_path)

    proxy_app = create_proxy_app(
        config,
        session_log,
    )

    proxy_server = TestServer(proxy_app)
    await proxy_server.start_server()

    try:
        request_body = (
            b'{"model":"test-model","messages":[]}'
        )

        async with ClientSession() as client:
            async with client.post(
                proxy_server.make_url(
                    "/v1/chat/completions?trace=yes"
                ),
                headers={
                    "Content-Type": "application/json",
                    "X-Test": "preserve-me",
                    "Authorization": "Bearer test-token",
                    "Cookie": "session=private-cookie",
                },
                data=request_body,
            ) as response:
                response_body = await response.read()

                assert response.status == 201
                assert response_body == (
                    b'{"result":"created"}'
                )
                assert response.headers["Content-Type"] == (
                    "application/json"
                )
                assert response.headers["X-Backend"] == (
                    "mock-backend"
                )

        assert received_request == {
            "method": "POST",
            "path": "/v1/chat/completions",
            "query": "trace=yes",
            "content_type": "application/json",
            "test_header": "preserve-me",
            "authorization": "Bearer test-token",
            "cookie": "session=private-cookie",
            "body": request_body,
        }

    finally:
        await proxy_server.close()
        await backend_server.close()

    calls = session_log.close()

    assert len(calls) == 1

    recorded_call = calls[0]

    assert recorded_call["session_id"] == (
        "test-session"
    )
    assert recorded_call["call_index"] == 1
    assert isinstance(
        recorded_call["started_at"],
        str,
    )
    assert recorded_call["latency_ms"] >= 0

    assert recorded_call["backend"] == {
        "name": "mock",
        "base_url": config.backend.base_url,
        "wire_api": "chat",
    }

    assert recorded_call["request"] == {
        "method": "POST",
        "path": (
            "/v1/chat/completions?trace=yes"
        ),
        "body": {
            "model": "test-model",
            "messages": [],
        },
    }

    assert recorded_call["response"] == {
        "status": 201,
        "streamed": False,
        "body": {
            "result": "created",
        },
    }

    assert recorded_call["error"] is None

    serialized_calls = json.dumps(calls)

    assert "test-token" not in serialized_calls
    assert "private-cookie" not in serialized_calls


def test_backend_connection_failure_returns_502(
    tmp_path: Path,
) -> None:
    asyncio.run(
        _test_backend_connection_failure_returns_502(
            tmp_path
        )
    )


async def _test_backend_connection_failure_returns_502(
    tmp_path: Path,
) -> None:
    backend_app = web.Application()
    backend_server = TestServer(backend_app)

    await backend_server.start_server()

    unavailable_url = str(
        backend_server.make_url("")
    ).rstrip("/")

    await backend_server.close()

    config = parse_config(
        {
            "backend": {
                "name": "unavailable-backend",
                "base_url": unavailable_url,
                "wire_api": "responses",
                "auth": {
                    "mode": "none",
                },
            }
        }
    )

    session_log = _make_session_log(tmp_path)

    proxy_app = create_proxy_app(
        config,
        session_log,
    )

    proxy_server = TestServer(proxy_app)
    await proxy_server.start_server()

    try:
        async with ClientSession() as client:
            async with client.post(
                proxy_server.make_url(
                    "/v1/responses"
                ),
                json={
                    "model": "test-model",
                    "input": "Hello",
                },
            ) as response:
                response_body = await response.json()

                assert response.status == 502
                assert response_body == {
                    "error": "backend unavailable",
                }

    finally:
        await proxy_server.close()

    calls = session_log.close()

    assert len(calls) == 1

    recorded_call = calls[0]

    assert recorded_call["call_index"] == 1
    assert recorded_call["request"] == {
        "method": "POST",
        "path": "/v1/responses",
        "body": {
            "model": "test-model",
            "input": "Hello",
        },
    }
    assert recorded_call["response"] is None
    assert isinstance(
        recorded_call["error"],
        str,
    )
    assert recorded_call["error"].startswith(
        "backend request failed:"
    )


def test_none_auth_removes_authorization_header(
    tmp_path: Path,
) -> None:
    asyncio.run(
        _test_none_auth_removes_authorization_header(
            tmp_path
        )
    )


async def _test_none_auth_removes_authorization_header(
    tmp_path: Path,
) -> None:
    received_authorization: str | None = None

    async def backend_handler(
        request: web.Request,
    ) -> web.Response:
        nonlocal received_authorization

        received_authorization = request.headers.get(
            "Authorization"
        )

        return web.json_response(
            {
                "result": "ok",
            }
        )

    backend_app = web.Application()

    backend_app.router.add_post(
        "/v1/responses",
        backend_handler,
    )

    backend_server = TestServer(backend_app)
    await backend_server.start_server()

    config = parse_config(
        {
            "backend": {
                "name": "mock",
                "base_url": str(
                    backend_server.make_url("")
                ).rstrip("/"),
                "wire_api": "responses",
                "auth": {
                    "mode": "none",
                },
            }
        }
    )

    session_log = _make_session_log(tmp_path)

    proxy_server = TestServer(
        create_proxy_app(
            config,
            session_log,
        )
    )

    await proxy_server.start_server()

    try:
        async with ClientSession() as client:
            async with client.post(
                proxy_server.make_url(
                    "/v1/responses"
                ),
                headers={
                    "Authorization": (
                        "Bearer must-not-be-forwarded"
                    ),
                },
                json={
                    "model": "test-model",
                    "input": "Hello",
                },
            ) as response:
                assert response.status == 200

    finally:
        await proxy_server.close()
        await backend_server.close()

    assert received_authorization is None

    calls = session_log.close()
    serialized_calls = json.dumps(calls)

    assert (
        "must-not-be-forwarded"
        not in serialized_calls
    )


def test_environment_auth_uses_environment_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CODEX_PROBE_TEST_API_KEY",
        "environment-secret",
    )

    asyncio.run(
        _test_environment_auth_uses_environment_token(
            tmp_path
        )
    )


async def _test_environment_auth_uses_environment_token(
    tmp_path: Path,
) -> None:
    received_authorization: str | None = None

    async def backend_handler(
        request: web.Request,
    ) -> web.Response:
        nonlocal received_authorization

        received_authorization = request.headers.get(
            "Authorization"
        )

        return web.json_response(
            {
                "result": "ok",
            }
        )

    backend_app = web.Application()

    backend_app.router.add_post(
        "/v1/responses",
        backend_handler,
    )

    backend_server = TestServer(backend_app)
    await backend_server.start_server()

    config = parse_config(
        {
            "backend": {
                "name": "mock",
                "base_url": str(
                    backend_server.make_url("")
                ).rstrip("/"),
                "wire_api": "responses",
                "auth": {
                    "mode": "environment",
                    "environment_variable": (
                        "CODEX_PROBE_TEST_API_KEY"
                    ),
                },
            }
        }
    )

    session_log = _make_session_log(tmp_path)

    proxy_server = TestServer(
        create_proxy_app(
            config,
            session_log,
        )
    )

    await proxy_server.start_server()

    try:
        async with ClientSession() as client:
            async with client.post(
                proxy_server.make_url(
                    "/v1/responses"
                ),
                headers={
                    "Authorization": (
                        "Bearer incoming-secret"
                    ),
                },
                json={
                    "model": "test-model",
                    "input": "Hello",
                },
            ) as response:
                assert response.status == 200

    finally:
        await proxy_server.close()
        await backend_server.close()

    assert received_authorization == (
        "Bearer environment-secret"
    )

    calls = session_log.close()
    serialized_calls = json.dumps(calls)

    assert "environment-secret" not in serialized_calls
    assert "incoming-secret" not in serialized_calls


def test_missing_environment_token_returns_500(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "CODEX_PROBE_MISSING_API_KEY",
        raising=False,
    )

    asyncio.run(
        _test_missing_environment_token_returns_500(
            tmp_path
        )
    )


async def _test_missing_environment_token_returns_500(
    tmp_path: Path,
) -> None:
    backend_was_called = False

    async def backend_handler(
        request: web.Request,
    ) -> web.Response:
        nonlocal backend_was_called

        backend_was_called = True

        return web.json_response(
            {
                "result": "should not happen",
            }
        )

    backend_app = web.Application()

    backend_app.router.add_post(
        "/v1/responses",
        backend_handler,
    )

    backend_server = TestServer(backend_app)
    await backend_server.start_server()

    config = parse_config(
        {
            "backend": {
                "name": "mock",
                "base_url": str(
                    backend_server.make_url("")
                ).rstrip("/"),
                "wire_api": "responses",
                "auth": {
                    "mode": "environment",
                    "environment_variable": (
                        "CODEX_PROBE_MISSING_API_KEY"
                    ),
                },
            }
        }
    )

    session_log = _make_session_log(tmp_path)

    proxy_server = TestServer(
        create_proxy_app(
            config,
            session_log,
        )
    )

    await proxy_server.start_server()

    try:
        async with ClientSession() as client:
            async with client.post(
                proxy_server.make_url(
                    "/v1/responses"
                ),
                headers={
                    "Authorization": (
                        "Bearer incoming-secret"
                    ),
                },
                json={
                    "model": "test-model",
                    "input": "Hello",
                },
            ) as response:
                response_text = await response.text()

                assert response.status == 500
                assert (
                    "CODEX_PROBE_MISSING_API_KEY"
                    in response_text
                )
                assert (
                    "incoming-secret"
                    not in response_text
                )

    finally:
        await proxy_server.close()
        await backend_server.close()

    assert backend_was_called is False

    calls = session_log.close()

    assert len(calls) == 1
    assert calls[0]["response"] is None
    assert calls[0]["error"] == (
        "proxy request failed: "
        "HTTPInternalServerError"
    )

    serialized_calls = json.dumps(calls)

    assert "incoming-secret" not in serialized_calls


def test_backend_redirect_is_not_followed(
    tmp_path: Path,
) -> None:
    asyncio.run(
        _test_backend_redirect_is_not_followed(
            tmp_path
        )
    )


async def _test_backend_redirect_is_not_followed(
    tmp_path: Path,
) -> None:
    redirected_endpoint_was_called = False

    async def redirect_handler(
        request: web.Request,
    ) -> web.Response:
        return web.Response(
            status=307,
            headers={
                "Location": "/replacement",
            },
        )

    async def replacement_handler(
        request: web.Request,
    ) -> web.Response:
        nonlocal redirected_endpoint_was_called

        redirected_endpoint_was_called = True

        return web.json_response(
            {
                "result": "redirect followed",
            }
        )

    backend_app = web.Application()

    backend_app.router.add_post(
        "/v1/responses",
        redirect_handler,
    )

    backend_app.router.add_post(
        "/replacement",
        replacement_handler,
    )

    backend_server = TestServer(backend_app)
    await backend_server.start_server()

    config = parse_config(
        {
            "backend": {
                "name": "mock",
                "base_url": str(
                    backend_server.make_url("")
                ).rstrip("/"),
                "wire_api": "responses",
                "auth": {
                    "mode": "none",
                },
            }
        }
    )

    session_log = _make_session_log(tmp_path)

    proxy_server = TestServer(
        create_proxy_app(
            config,
            session_log,
        )
    )

    await proxy_server.start_server()

    try:
        async with ClientSession() as client:
            async with client.post(
                proxy_server.make_url(
                    "/v1/responses"
                ),
                json={
                    "model": "test-model",
                    "input": "Hello",
                },
                allow_redirects=False,
            ) as response:
                assert response.status == 307
                assert response.headers["Location"] == (
                    "/replacement"
                )

    finally:
        await proxy_server.close()
        await backend_server.close()

    assert redirected_endpoint_was_called is False

    calls = session_log.close()

    assert len(calls) == 1
    assert calls[0]["response"] == {
        "status": 307,
        "streamed": False,
        "body": None,
    }


def test_concurrent_calls_keep_request_start_order(
    tmp_path: Path,
) -> None:
    asyncio.run(
        _test_concurrent_calls_keep_request_start_order(
            tmp_path
        )
    )


async def _test_concurrent_calls_keep_request_start_order(
    tmp_path: Path,
) -> None:
    first_request_arrived = asyncio.Event()
    allow_first_response = asyncio.Event()

    async def backend_handler(
        request: web.Request,
    ) -> web.Response:
        body = await request.json()
        request_name = body["name"]

        if request_name == "first":
            first_request_arrived.set()
            await allow_first_response.wait()

        return web.json_response(
            {
                "name": request_name,
            }
        )

    backend_app = web.Application()

    backend_app.router.add_post(
        "/v1/responses",
        backend_handler,
    )

    backend_server = TestServer(backend_app)
    await backend_server.start_server()

    config = parse_config(
        {
            "backend": {
                "name": "mock",
                "base_url": str(
                    backend_server.make_url("")
                ).rstrip("/"),
                "wire_api": "responses",
                "auth": {
                    "mode": "none",
                },
            }
        }
    )

    session_log = _make_session_log(tmp_path)

    proxy_server = TestServer(
        create_proxy_app(
            config,
            session_log,
        )
    )

    await proxy_server.start_server()

    async def send_request(
        client: ClientSession,
        name: str,
    ) -> int:
        async with client.post(
            proxy_server.make_url(
                "/v1/responses"
            ),
            json={
                "name": name,
            },
        ) as response:
            await response.read()
            return response.status

    try:
        async with ClientSession() as client:
            first_task = asyncio.create_task(
                send_request(
                    client,
                    "first",
                )
            )

            await first_request_arrived.wait()

            try:
                second_status = await send_request(
                    client,
                    "second",
                )
            finally:
                allow_first_response.set()

            first_status = await first_task

            assert first_status == 200
            assert second_status == 200

    finally:
        allow_first_response.set()
        await proxy_server.close()
        await backend_server.close()

    calls = session_log.close()

    assert [
        call["call_index"]
        for call in calls
    ] == [1, 2]

    assert [
        call["request"]["body"]["name"]
        for call in calls
    ] == [
        "first",
        "second",
    ]


def _as_sse(payload: dict[str, object]) -> bytes:
    """Encode one dictionary as an SSE data event."""

    data = json.dumps(payload)

    return f"data: {data}\n\n".encode("utf-8")


def test_responses_stream_is_forwarded_live_and_recorded(
    tmp_path: Path,
) -> None:
    asyncio.run(
        _test_responses_stream_is_forwarded_live_and_recorded(
            tmp_path
        )
    )


async def _test_responses_stream_is_forwarded_live_and_recorded(
    tmp_path: Path,
) -> None:
    allow_stream_to_finish = asyncio.Event()

    expected_response = {
        "id": "resp-123",
        "object": "response",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Hello",
                    }
                ],
            }
        ],
    }

    first_event = _as_sse(
        {
            "type": "response.created",
            "response": {
                "id": "resp-123",
                "status": "in_progress",
            },
        }
    )

    terminal_event = _as_sse(
        {
            "type": "response.completed",
            "response": expected_response,
        }
    )

    done_event = b"data: [DONE]\n\n"

    async def backend_handler(
        request: web.Request,
    ) -> web.StreamResponse:
        await request.read()

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
            },
        )

        await response.prepare(request)

        midpoint = len(first_event) // 2

        await response.write(
            first_event[:midpoint]
        )
        await response.write(
            first_event[midpoint:]
        )

        await allow_stream_to_finish.wait()

        await response.write(terminal_event)
        await response.write(done_event)
        await response.write_eof()

        return response

    backend_app = web.Application()

    backend_app.router.add_post(
        "/v1/responses",
        backend_handler,
    )

    backend_server = TestServer(backend_app)
    await backend_server.start_server()

    config = parse_config(
        {
            "backend": {
                "name": "mock-responses",
                "base_url": str(
                    backend_server.make_url("")
                ).rstrip("/"),
                "wire_api": "responses",
                "auth": {
                    "mode": "none",
                },
            }
        }
    )

    session_log = _make_session_log(tmp_path)

    proxy_server = TestServer(
        create_proxy_app(
            config,
            session_log,
        )
    )

    await proxy_server.start_server()

    try:
        async with ClientSession() as client:
            async with client.post(
                proxy_server.make_url(
                    "/v1/responses"
                ),
                json={
                    "model": "test-model",
                    "input": "Hello",
                    "stream": True,
                },
            ) as response:
                assert response.status == 200
                assert response.headers[
                    "Content-Type"
                ].startswith("text/event-stream")

                try:
                    received_first_event = (
                        await asyncio.wait_for(
                            response.content.readexactly(
                                len(first_event)
                            ),
                            timeout=1,
                        )
                    )

                    assert (
                        received_first_event
                        == first_event
                    )

                finally:
                    allow_stream_to_finish.set()

                remaining_bytes = (
                    await response.read()
                )

                complete_stream = (
                    received_first_event
                    + remaining_bytes
                )

                assert complete_stream == (
                    first_event
                    + terminal_event
                    + done_event
                )

    finally:
        allow_stream_to_finish.set()
        await proxy_server.close()
        await backend_server.close()

    calls = session_log.close()

    assert len(calls) == 1

    recorded_call = calls[0]

    assert recorded_call["response"] == {
        "status": 200,
        "streamed": True,
        "body": expected_response,
    }

    assert recorded_call["error"] is None


def test_chat_tool_call_stream_is_reassembled_and_recorded(
    tmp_path: Path,
) -> None:
    asyncio.run(
        _test_chat_tool_call_stream_is_reassembled_and_recorded(
            tmp_path
        )
    )


async def _test_chat_tool_call_stream_is_reassembled_and_recorded(
    tmp_path: Path,
) -> None:
    first_event = _as_sse(
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-123",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":',
                                },
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        }
    )

    second_event = _as_sse(
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "arguments": (
                                        '"config.py"}'
                                    ),
                                },
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        }
    )

    final_event = _as_sse(
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls",
                }
            ],
        }
    )

    done_event = b"data: [DONE]\n\n"

    complete_stream = (
        first_event
        + second_event
        + final_event
        + done_event
    )

    async def backend_handler(
        request: web.Request,
    ) -> web.StreamResponse:
        await request.read()

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": (
                    "text/event-stream; charset=utf-8"
                ),
            },
        )

        await response.prepare(request)

        await response.write(first_event)
        await response.write(second_event)
        await response.write(final_event)
        await response.write(done_event)
        await response.write_eof()

        return response

    backend_app = web.Application()

    backend_app.router.add_post(
        "/v1/chat/completions",
        backend_handler,
    )

    backend_server = TestServer(backend_app)
    await backend_server.start_server()

    config = parse_config(
        {
            "backend": {
                "name": "mock-chat",
                "base_url": str(
                    backend_server.make_url("")
                ).rstrip("/"),
                "wire_api": "chat",
                "auth": {
                    "mode": "none",
                },
            }
        }
    )

    session_log = _make_session_log(tmp_path)

    proxy_server = TestServer(
        create_proxy_app(
            config,
            session_log,
        )
    )

    await proxy_server.start_server()

    try:
        async with ClientSession() as client:
            async with client.post(
                proxy_server.make_url(
                    "/v1/chat/completions"
                ),
                json={
                    "model": "test-model",
                    "messages": [],
                    "stream": True,
                },
            ) as response:
                received_stream = (
                    await response.read()
                )

                assert response.status == 200
                assert received_stream == complete_stream

    finally:
        await proxy_server.close()
        await backend_server.close()

    calls = session_log.close()

    assert len(calls) == 1

    assert calls[0]["response"] == {
        "status": 200,
        "streamed": True,
        "body": {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-123",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": (
                                        '{"path":"config.py"}'
                                    ),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                    "logprobs": None,
                }
            ],
        },
    }

    assert calls[0]["error"] is None


def test_incomplete_stream_is_forwarded_and_recorded_as_error(
    tmp_path: Path,
) -> None:
    asyncio.run(
        _test_incomplete_stream_is_forwarded_and_recorded_as_error(
            tmp_path
        )
    )


async def _test_incomplete_stream_is_forwarded_and_recorded_as_error(
    tmp_path: Path,
) -> None:
    incomplete_event = _as_sse(
        {
            "type": "response.created",
            "response": {
                "id": "resp-incomplete",
                "status": "in_progress",
            },
        }
    )

    done_event = b"data: [DONE]\n\n"

    incomplete_stream = (
        incomplete_event
        + done_event
    )

    async def backend_handler(
        request: web.Request,
    ) -> web.StreamResponse:
        await request.read()

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
            },
        )

        await response.prepare(request)
        await response.write(incomplete_event)
        await response.write(done_event)
        await response.write_eof()

        return response

    backend_app = web.Application()

    backend_app.router.add_post(
        "/v1/responses",
        backend_handler,
    )

    backend_server = TestServer(backend_app)
    await backend_server.start_server()

    config = parse_config(
        {
            "backend": {
                "name": "mock-incomplete",
                "base_url": str(
                    backend_server.make_url("")
                ).rstrip("/"),
                "wire_api": "responses",
                "auth": {
                    "mode": "none",
                },
            }
        }
    )

    session_log = _make_session_log(tmp_path)

    proxy_server = TestServer(
        create_proxy_app(
            config,
            session_log,
        )
    )

    await proxy_server.start_server()

    try:
        async with ClientSession() as client:
            async with client.post(
                proxy_server.make_url(
                    "/v1/responses"
                ),
                json={
                    "model": "test-model",
                    "input": "Hello",
                    "stream": True,
                },
            ) as response:
                received_stream = (
                    await response.read()
                )

                assert response.status == 200
                assert (
                    received_stream
                    == incomplete_stream
                )

    finally:
        await proxy_server.close()
        await backend_server.close()

    calls = session_log.close()

    assert len(calls) == 1
    assert calls[0]["response"] is None
    assert calls[0]["error"] == (
        "stream reconstruction failed: ValueError"
    )


def test_non_llm_request_is_forwarded_but_not_recorded(
    tmp_path: Path,
) -> None:
    asyncio.run(
        _test_non_llm_request_is_forwarded_but_not_recorded(
            tmp_path
        )
    )


async def _test_non_llm_request_is_forwarded_but_not_recorded(
    tmp_path: Path,
) -> None:
    received_request: dict[str, object] = {}

    async def backend_handler(
        request: web.Request,
    ) -> web.Response:
        received_request["method"] = request.method
        received_request["path"] = request.path
        received_request["query"] = request.query_string

        return web.json_response(
            {
                "object": "list",
                "data": [],
            }
        )

    backend_app = web.Application()

    backend_app.router.add_get(
        "/models",
        backend_handler,
    )

    backend_server = TestServer(backend_app)
    await backend_server.start_server()

    config = parse_config(
        {
            "backend": {
                "name": "mock",
                "base_url": str(
                    backend_server.make_url("")
                ).rstrip("/"),
                "wire_api": "responses",
                "auth": {
                    "mode": "none",
                },
            }
        }
    )

    session_log = _make_session_log(tmp_path)

    proxy_app = create_proxy_app(
        config,
        session_log,
    )

    proxy_server = TestServer(proxy_app)
    await proxy_server.start_server()

    try:
        async with ClientSession() as client:
            async with client.get(
                proxy_server.make_url(
                    "/models?client_version=0.153.0"
                )
            ) as response:
                response_body = await response.json()

                assert response.status == 200
                assert response_body == {
                    "object": "list",
                    "data": [],
                }

        assert received_request == {
            "method": "GET",
            "path": "/models",
            "query": "client_version=0.153.0",
        }

    finally:
        await proxy_server.close()
        await backend_server.close()

    calls = session_log.close()

    assert calls == []


def test_compressed_json_response_is_decoded_and_recorded(
    tmp_path: Path,
) -> None:
    asyncio.run(
        _test_compressed_json_response_is_decoded_and_recorded(
            tmp_path
        )
    )


async def _test_compressed_json_response_is_decoded_and_recorded(
    tmp_path: Path,
) -> None:
    response_data = {
        "status": "completed",
        "output": "OPENAI_PROXY_OK",
    }

    compressed_body = gzip.compress(
        json.dumps(response_data).encode("utf-8")
    )

    async def backend_handler(
        request: web.Request,
    ) -> web.Response:
        await request.read()

        return web.Response(
            body=compressed_body,
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
            },
        )

    backend_app = web.Application()
    backend_app.router.add_post(
        "/responses",
        backend_handler,
    )

    backend_server = TestServer(backend_app)
    await backend_server.start_server()

    config = parse_config(
        {
            "backend": {
                "name": "compressed-backend",
                "base_url": str(
                    backend_server.make_url("")
                ).rstrip("/"),
                "wire_api": "responses",
                "auth": {
                    "mode": "none",
                },
            }
        }
    )

    session_log = _make_session_log(tmp_path)
    proxy_app = create_proxy_app(
        config,
        session_log,
    )

    proxy_server = TestServer(proxy_app)
    await proxy_server.start_server()

    try:
        async with ClientSession() as client:
            async with client.post(
                proxy_server.make_url("/responses"),
                json={
                    "model": "test-model",
                    "input": "Hello",
                },
            ) as response:
                response_body = await response.json()

                assert response.status == 200
                assert response_body == response_data
                assert "Content-Encoding" not in response.headers

    finally:
        await proxy_server.close()
        await backend_server.close()

    calls = session_log.close()

    assert len(calls) == 1
    assert calls[0]["response"] == {
        "status": 200,
        "streamed": False,
        "body": response_data,
    }
