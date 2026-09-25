"""Public lifecycle API for starting and stopping CodexProbe."""

from __future__ import annotations

import asyncio
import logging
import socket
from collections.abc import Mapping
from threading import Event, Lock, Thread
from uuid import uuid4

from aiohttp import web

from .config import ProxyConfig, parse_config
from .proxy import create_proxy_app
from .recording import JsonValue, SessionLog


LOGGER = logging.getLogger(__name__)


class ProxyRecorder:
    """Run the CodexProbe proxy in a background thread."""

    def __init__(
        self,
        config: Mapping[str, object],
    ) -> None:
        self._config: ProxyConfig = parse_config(config)

        self._session_log = SessionLog(
            log_dir=self._config.log_dir,
            session_id=_create_session_id(),
        )

        self._state_lock = Lock()
        self._ready = Event()

        self._thread: Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: web.AppRunner | None = None
        self._endpoint: str | None = None
        self._startup_error: Exception | None = None

        self._started = False
        self._stopped = False
        self._cleanup_finished = False

    def start(self) -> str:
        """Start the proxy and return its local HTTP endpoint."""

        with self._state_lock:
            if self._started:
                raise RuntimeError(
                    "ProxyRecorder is already started"
                )

            self._started = True

            self._thread = Thread(
                target=self._run_server,
                name=(
                    "codex-probe-"
                    f"{self._session_log.session_id}"
                ),
                daemon=True,
            )

            self._thread.start()

        self._ready.wait()

        if self._startup_error is not None:
            thread = self._thread

            if thread is not None:
                thread.join()

            with self._state_lock:
                self._stopped = True

            raise RuntimeError(
                "could not start ProxyRecorder"
            ) from self._startup_error

        if self._endpoint is None:
            raise RuntimeError(
                "ProxyRecorder started without an endpoint"
            )

        return self._endpoint

    def stop(self) -> list[dict[str, JsonValue]]:
        """Stop the proxy and return its ordered recorded calls."""

        with self._state_lock:
            if not self._started:
                raise RuntimeError(
                    "ProxyRecorder has not been started"
                )

            if self._stopped:
                raise RuntimeError(
                    "ProxyRecorder is already stopped"
                )

            self._stopped = True

        loop = self._loop
        thread = self._thread

        if loop is None or thread is None:
            raise RuntimeError(
                "ProxyRecorder has no running server"
            )

        cleanup_future = asyncio.run_coroutine_threadsafe(
            self._cleanup_server(),
            loop,
        )

        cleanup_error: Exception | None = None

        try:
            cleanup_future.result()

        except Exception as error:
            cleanup_error = error

        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join()

        if cleanup_error is not None:
            raise RuntimeError(
                "could not stop ProxyRecorder cleanly"
            ) from cleanup_error

        return self._session_log.close()

    def _run_server(self) -> None:
        """Own and run the proxy's asynchronous event loop."""

        loop = asyncio.new_event_loop()
        self._loop = loop

        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(
                self._start_server()
            )

            self._ready.set()
            loop.run_forever()

        except Exception as error:
            if not self._ready.is_set():
                self._startup_error = error
            else:
                LOGGER.exception(
                    "ProxyRecorder server thread failed"
                )

        finally:
            self._ready.set()

            if (
                self._runner is not None
                and not self._cleanup_finished
            ):
                try:
                    loop.run_until_complete(
                        self._cleanup_server()
                    )

                except Exception:
                    LOGGER.exception(
                        "ProxyRecorder cleanup failed"
                    )

            pending_tasks = asyncio.all_tasks(loop)

            for task in pending_tasks:
                task.cancel()

            if pending_tasks:
                loop.run_until_complete(
                    asyncio.gather(
                        *pending_tasks,
                        return_exceptions=True,
                    )
                )

            asyncio.set_event_loop(None)
            loop.close()

    async def _start_server(self) -> None:
        """Create and bind the internal aiohttp server."""

        app = create_proxy_app(
            self._config,
            self._session_log,
        )

        runner = web.AppRunner(app)
        self._runner = runner

        await runner.setup()

        server_socket = _create_server_socket(
            host=self._config.listen.host,
            port=self._config.listen.port,
        )

        try:
            site = web.SockSite(
                runner,
                server_socket,
            )

            await site.start()

        except Exception:
            server_socket.close()
            raise

        actual_port = server_socket.getsockname()[1]

        endpoint_host = _format_endpoint_host(
            self._config.listen.host
        )

        self._endpoint = (
            f"http://{endpoint_host}:{actual_port}"
        )

    async def _cleanup_server(self) -> None:
        """Stop accepting requests and clean up the server."""

        if self._cleanup_finished:
            return

        if self._runner is not None:
            await self._runner.cleanup()

        self._cleanup_finished = True


def _create_server_socket(
    *,
    host: str,
    port: int,
) -> socket.socket:
    """Create a non-blocking socket for the local proxy."""

    family = (
        socket.AF_INET6
        if ":" in host
        else socket.AF_INET
    )

    server_socket = socket.socket(
        family=family,
        type=socket.SOCK_STREAM,
    )

    server_socket.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1,
    )

    server_socket.bind(
        (
            host,
            port,
        )
    )

    server_socket.listen(128)
    server_socket.setblocking(False)

    return server_socket


def _format_endpoint_host(host: str) -> str:
    """Format an IPv4, IPv6, or hostname for an HTTP URL."""

    if ":" in host:
        return f"[{host}]"

    return host


def _create_session_id() -> str:
    """Create a unique identifier for one proxy session."""

    return uuid4().hex
