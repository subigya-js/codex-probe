"""HTTP forwarding, streaming, and recording for CodexProbe."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator, Mapping
from datetime import datetime, timezone
from time import perf_counter

from aiohttp import (
    ClientError,
    ClientResponse,
    ClientSession,
    ClientTimeout,
    web,
)

from .config import ProxyConfig, WireApi
from .recording import (
    JsonValue,
    RecordedBackend,
    RecordedCall,
    RecordedRequest,
    RecordedResponse,
    SessionLog,
)
from .sse import (
    ChatCompletionsReassembler,
    ResponsesReassembler,
    SseParser,
)


LOGGER = logging.getLogger(__name__)

_CONFIG_KEY = web.AppKey("config", ProxyConfig)
_CLIENT_KEY = web.AppKey("client", ClientSession)
_SESSION_LOG_KEY = web.AppKey("session_log", SessionLog)

_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def create_proxy_app(
    config: ProxyConfig,
    session_log: SessionLog,
) -> web.Application:
    """Create the internal aiohttp proxy application."""

    app = web.Application()

    app[_CONFIG_KEY] = config
    app[_SESSION_LOG_KEY] = session_log

    app.cleanup_ctx.append(
        _client_session_context
    )

    app.router.add_route(
        "*",
        "/{path:.*}",
        _forward_request,
    )

    return app


async def _client_session_context(
    app: web.Application,
) -> AsyncIterator[None]:
    """Create and close the shared backend HTTP client."""

    timeout = ClientTimeout(
        total=None,
        connect=10,
    )

    session = ClientSession(
        timeout=timeout,
        auto_decompress=False,
    )

    app[_CLIENT_KEY] = session

    yield

    await session.close()


async def _forward_request(
    request: web.Request,
) -> web.StreamResponse:
    """Forward and record one backend request."""

    config = request.app[_CONFIG_KEY]
    session = request.app[_CLIENT_KEY]
    session_log = request.app[_SESSION_LOG_KEY]

    call_index = session_log.reserve_call_index()

    started_at = datetime.now(
        timezone.utc
    ).isoformat()

    timer_started = perf_counter()

    request_body = await request.read()

    recorded_request = RecordedRequest(
        method=request.method,
        path=request.raw_path,
        body=_decode_body(request_body),
    )

    recorded_backend = RecordedBackend(
        name=config.backend.name,
        base_url=config.backend.base_url,
        wire_api=config.backend.wire_api,
    )

    def record_success(
        *,
        status: int,
        streamed: bool,
        body: JsonValue,
    ) -> None:
        """Record a successful backend response."""

        latency_ms = (
            perf_counter() - timer_started
        ) * 1000

        session_log.add(
            RecordedCall(
                session_id=session_log.session_id,
                call_index=call_index,
                started_at=started_at,
                latency_ms=latency_ms,
                backend=recorded_backend,
                request=recorded_request,
                response=RecordedResponse(
                    status=status,
                    streamed=streamed,
                    body=body,
                ),
                error=None,
            )
        )

    def record_error(error_message: str) -> None:
        """Record a call that did not produce a response."""

        latency_ms = (
            perf_counter() - timer_started
        ) * 1000

        session_log.add(
            RecordedCall(
                session_id=session_log.session_id,
                call_index=call_index,
                started_at=started_at,
                latency_ms=latency_ms,
                backend=recorded_backend,
                request=recorded_request,
                response=None,
                error=error_message,
            )
        )

    backend_url = (
        f"{config.backend.base_url}{request.raw_path}"
    )

    try:
        request_headers = _build_request_headers(
            request.headers,
            config,
        )

        async with session.request(
            method=request.method,
            url=backend_url,
            headers=request_headers,
            data=request_body,
            allow_redirects=False,
        ) as backend_response:
            if _is_sse_response(backend_response):
                (
                    client_response,
                    reconstructed_body,
                    stream_error,
                ) = await _forward_sse_response(
                    request=request,
                    backend_response=backend_response,
                    wire_api=config.backend.wire_api,
                )

                if stream_error is None:
                    record_success(
                        status=backend_response.status,
                        streamed=True,
                        body=reconstructed_body,
                    )

                else:
                    record_error(stream_error)

                return client_response

            response_body = await backend_response.read()

            response_headers = _copy_headers(
                backend_response.headers,
                extra_blocked={"content-length"},
            )

            record_success(
                status=backend_response.status,
                streamed=False,
                body=_decode_body(response_body),
            )

            return web.Response(
                status=backend_response.status,
                headers=response_headers,
                body=response_body,
            )

    except web.HTTPException as error:
        record_error(
            "proxy request failed: "
            f"{type(error).__name__}"
        )

        raise

    except (ClientError, asyncio.TimeoutError) as error:
        error_message = (
            "backend request failed: "
            f"{type(error).__name__}"
        )

        LOGGER.exception(
            "Could not communicate with backend %s",
            config.backend.name,
        )

        record_error(error_message)

        return web.json_response(
            {
                "error": "backend unavailable",
            },
            status=502,
        )


async def _forward_sse_response(
    *,
    request: web.Request,
    backend_response: ClientResponse,
    wire_api: WireApi,
) -> tuple[
    web.StreamResponse,
    JsonValue,
    str | None,
]:
    """Forward an SSE response while reconstructing its final body."""

    response_headers = _copy_headers(
        backend_response.headers,
        extra_blocked={"content-length"},
    )

    client_response = web.StreamResponse(
        status=backend_response.status,
        headers=response_headers,
    )

    parser = SseParser()
    reassembler = _create_reassembler(wire_api)

    reconstruction_error: str | None = None

    try:
        await client_response.prepare(request)

    except (ClientError, ConnectionError) as error:
        return (
            client_response,
            None,
            (
                "client connection failed: "
                f"{type(error).__name__}"
            ),
        )

    try:
        async for chunk in backend_response.content.iter_any():
            try:
                await client_response.write(chunk)

            except (ClientError, ConnectionError) as error:
                client_response.force_close()

                return (
                    client_response,
                    None,
                    (
                        "client disconnected: "
                        f"{type(error).__name__}"
                    ),
                )

            if reconstruction_error is not None:
                continue

            try:
                events = parser.feed(chunk)

                for event in events:
                    reassembler.accept(event)

            except (
                UnicodeDecodeError,
                ValueError,
                RuntimeError,
            ) as error:
                reconstruction_error = (
                    "stream reconstruction failed: "
                    f"{type(error).__name__}"
                )

    except (ClientError, asyncio.TimeoutError) as error:
        client_response.force_close()

        return (
            client_response,
            None,
            (
                "backend stream failed: "
                f"{type(error).__name__}"
            ),
        )

    reconstructed_body: JsonValue = None

    if reconstruction_error is None:
        try:
            for event in parser.finish():
                reassembler.accept(event)

            reconstructed_body = reassembler.finish()

        except (
            UnicodeDecodeError,
            ValueError,
            RuntimeError,
        ) as error:
            reconstruction_error = (
                "stream reconstruction failed: "
                f"{type(error).__name__}"
            )

    try:
        await client_response.write_eof()

    except (ClientError, ConnectionError):
        client_response.force_close()

    return (
        client_response,
        reconstructed_body,
        reconstruction_error,
    )


def _create_reassembler(
    wire_api: WireApi,
) -> (
    ResponsesReassembler
    | ChatCompletionsReassembler
):
    """Create the reassembler required by the backend API."""

    if wire_api == "responses":
        return ResponsesReassembler()

    return ChatCompletionsReassembler()


def _is_sse_response(
    response: ClientResponse,
) -> bool:
    """Return whether the backend response contains an SSE stream."""

    content_type = response.headers.get(
        "Content-Type",
        "",
    )

    media_type = content_type.split(
        ";",
        maxsplit=1,
    )[0]

    return (
        media_type.strip().lower()
        == "text/event-stream"
    )


def _decode_body(body: bytes) -> JsonValue:
    """Convert HTTP body bytes into a recordable JSON value."""

    if not body:
        return None

    try:
        return json.loads(body)

    except (json.JSONDecodeError, UnicodeDecodeError):
        return body.decode(
            "utf-8",
            errors="replace",
        )


def _build_request_headers(
    incoming_headers: Mapping[str, str],
    config: ProxyConfig,
) -> list[tuple[str, str]]:
    """Build backend headers according to the authentication mode."""

    headers = _copy_headers(
        incoming_headers,
        extra_blocked={
            "host",
            "content-length",
            "authorization",
        },
    )

    auth = config.backend.auth

    if auth.mode == "forward":
        authorization = incoming_headers.get(
            "Authorization"
        )

        if authorization is not None:
            headers.append(
                ("Authorization", authorization)
            )

    elif auth.mode == "environment":
        variable_name = auth.environment_variable

        if variable_name is None:
            raise RuntimeError(
                "environment auth requires a variable name"
            )

        token = os.environ.get(variable_name)

        if token is None:
            raise web.HTTPInternalServerError(
                text=(
                    "backend authentication variable "
                    f"{variable_name!r} is not set"
                )
            )

        headers.append(
            ("Authorization", f"Bearer {token}")
        )

    return headers


def _copy_headers(
    headers: Mapping[str, str],
    *,
    extra_blocked: set[str],
) -> list[tuple[str, str]]:
    """Copy headers that are safe to forward for one HTTP message."""

    blocked = set(_HOP_BY_HOP_HEADERS)
    blocked.update(extra_blocked)

    connection = headers.get("Connection")

    if connection is not None:
        connection_headers = connection.split(",")

        for header_name in connection_headers:
            blocked.add(
                header_name.strip().lower()
            )

    return [
        (name, value)
        for name, value in headers.items()
        if name.lower() not in blocked
    ]