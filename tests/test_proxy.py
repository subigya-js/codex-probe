"""Tests for HTTP proxy passthrough."""

import pytest
import asyncio

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer

from codex_probe.config import parse_config
from codex_probe.proxy import create_proxy_app


def test_non_streaming_request_and_response_passthrough() -> None:
    asyncio.run(
        _test_non_streaming_request_and_response_passthrough()
    )


async def _test_non_streaming_request_and_response_passthrough(
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

    proxy_app = create_proxy_app(config)
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
            "body": request_body,
        }

    finally:
        await proxy_server.close()
        await backend_server.close()


def test_backend_connection_failure_returns_502() -> None:
    asyncio.run(
        _test_backend_connection_failure_returns_502()
    )


async def _test_backend_connection_failure_returns_502(
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

    proxy_app = create_proxy_app(config)
    proxy_server = TestServer(proxy_app)

    await proxy_server.start_server()

    try:
        async with ClientSession() as client:
            async with client.post(
                proxy_server.make_url("/v1/responses"),
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


def test_none_auth_removes_authorization_header() -> None:
    asyncio.run(
        _test_none_auth_removes_authorization_header()
    )


async def _test_none_auth_removes_authorization_header(
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

    proxy_server = TestServer(
        create_proxy_app(config)
    )
    await proxy_server.start_server()

    try:
        async with ClientSession() as client:
            async with client.post(
                proxy_server.make_url("/v1/responses"),
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

        assert received_authorization is None

    finally:
        await proxy_server.close()
        await backend_server.close()


def test_environment_auth_uses_environment_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CODEX_PROBE_TEST_API_KEY",
        "environment-secret",
    )

    asyncio.run(
        _test_environment_auth_uses_environment_token()
    )


async def _test_environment_auth_uses_environment_token(
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

    proxy_server = TestServer(
        create_proxy_app(config)
    )
    await proxy_server.start_server()

    try:
        async with ClientSession() as client:
            async with client.post(
                proxy_server.make_url("/v1/responses"),
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

        assert received_authorization == (
            "Bearer environment-secret"
        )

    finally:
        await proxy_server.close()
        await backend_server.close()


def test_missing_environment_token_returns_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "CODEX_PROBE_MISSING_API_KEY",
        raising=False,
    )

    asyncio.run(
        _test_missing_environment_token_returns_500()
    )


async def _test_missing_environment_token_returns_500(
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

    proxy_server = TestServer(
        create_proxy_app(config)
    )
    await proxy_server.start_server()

    try:
        async with ClientSession() as client:
            async with client.post(
                proxy_server.make_url("/v1/responses"),
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

        assert backend_was_called is False

    finally:
        await proxy_server.close()
        await backend_server.close()


def test_backend_redirect_is_not_followed() -> None:
    asyncio.run(
        _test_backend_redirect_is_not_followed()
    )


async def _test_backend_redirect_is_not_followed(
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

    proxy_server = TestServer(
        create_proxy_app(config)
    )
    await proxy_server.start_server()

    try:
        async with ClientSession() as client:
            async with client.post(
                proxy_server.make_url("/v1/responses"),
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

        assert redirected_endpoint_was_called is False

    finally:
        await proxy_server.close()
        await backend_server.close()
