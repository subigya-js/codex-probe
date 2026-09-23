"""HTTP request and response forwarding for CodexProbe."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Mapping

from aiohttp import (
    ClientError,
    ClientSession,
    ClientTimeout,
    web,
)

from .config import ProxyConfig


LOGGER = logging.getLogger(__name__)

_CONFIG_KEY = web.AppKey("config", ProxyConfig)
_CLIENT_KEY = web.AppKey("client", ClientSession)

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


def create_proxy_app(config: ProxyConfig) -> web.Application:
    """Create the internal aiohttp proxy application."""

    app = web.Application()
    app[_CONFIG_KEY] = config

    app.cleanup_ctx.append(_client_session_context)

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
) -> web.Response:
    """Forward one non-streaming request to the backend."""

    config = request.app[_CONFIG_KEY]
    session = request.app[_CLIENT_KEY]

    backend_url = (
        f"{config.backend.base_url}{request.raw_path}"
    )

    request_body = await request.read()

    request_headers = _build_request_headers(
        request.headers,
        config,
    )

    try:
        async with session.request(
            method=request.method,
            url=backend_url,
            headers=request_headers,
            data=request_body,
            allow_redirects=False,
        ) as backend_response:
            response_body = await backend_response.read()

            response_headers = _copy_headers(
                backend_response.headers,
                extra_blocked={"content-length"},
            )

            return web.Response(
                status=backend_response.status,
                headers=response_headers,
                body=response_body,
            )

    except (ClientError, asyncio.TimeoutError):
        LOGGER.exception(
            "Could not connect to backend %s",
            config.backend.name,
        )

        return web.json_response(
            {
                "error": "backend unavailable",
            },
            status=502,
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