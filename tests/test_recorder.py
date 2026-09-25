"""Tests for the public ProxyRecorder lifecycle."""

import json
import socket
from collections.abc import Iterator
from http.client import HTTPConnection
from http.server import (
    BaseHTTPRequestHandler,
    ThreadingHTTPServer,
)
from pathlib import Path
from threading import Thread
from urllib.parse import urlsplit

import pytest

from codex_probe import ProxyRecorder


class _MockBackendHandler(BaseHTTPRequestHandler):
    """Handle requests for the recorder lifecycle tests."""

    received_requests: list[dict[str, object]] = []

    def do_POST(self) -> None:
        """Return one predictable JSON response."""

        content_length = int(
            self.headers.get(
                "Content-Length",
                "0",
            )
        )

        request_body = self.rfile.read(
            content_length
        )

        self.received_requests.append(
            {
                "method": self.command,
                "path": self.path,
                "body": request_body,
            }
        )

        response_body = json.dumps(
            {
                "result": "completed",
            }
        ).encode("utf-8")

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "application/json",
        )

        self.send_header(
            "Content-Length",
            str(len(response_body)),
        )

        self.end_headers()
        self.wfile.write(response_body)

    def log_message(
        self,
        format: str,
        *args: object,
    ) -> None:
        """Disable noisy test-server access logging."""


@pytest.fixture
def backend_url() -> Iterator[str]:
    """Run a local mock backend for one test."""

    _MockBackendHandler.received_requests = []

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _MockBackendHandler,
    )

    server.daemon_threads = True

    thread = Thread(
        target=server.serve_forever,
        daemon=True,
    )

    thread.start()

    host, port = server.server_address[:2]

    try:
        yield f"http://{host}:{port}"

    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _make_config(
    *,
    backend_url: str,
    log_dir: Path,
    listen_port: int = 0,
) -> dict[str, object]:
    """Create a valid recorder configuration."""

    return {
        "backend": {
            "name": "mock-backend",
            "base_url": backend_url,
            "wire_api": "responses",
            "auth": {
                "mode": "none",
            },
        },
        "listen": {
            "host": "127.0.0.1",
            "port": listen_port,
        },
        "log_dir": str(log_dir),
    }


def test_start_request_and_stop(
    tmp_path: Path,
    backend_url: str,
) -> None:
    log_dir = tmp_path / "logs"

    recorder = ProxyRecorder(
        _make_config(
            backend_url=backend_url,
            log_dir=log_dir,
        )
    )

    endpoint = recorder.start()
    parsed_endpoint = urlsplit(endpoint)

    assert parsed_endpoint.scheme == "http"
    assert parsed_endpoint.hostname == "127.0.0.1"
    assert parsed_endpoint.port is not None
    assert parsed_endpoint.port > 0

    connection = HTTPConnection(
        host=parsed_endpoint.hostname,
        port=parsed_endpoint.port,
        timeout=5,
    )

    request_body = json.dumps(
        {
            "model": "test-model",
            "input": "Hello",
        }
    ).encode("utf-8")

    try:
        connection.request(
            method="POST",
            url="/v1/responses",
            body=request_body,
            headers={
                "Content-Type": "application/json",
            },
        )

        response = connection.getresponse()
        response_body = response.read()

        assert response.status == 200
        assert json.loads(response_body) == {
            "result": "completed",
        }

    finally:
        connection.close()

    calls = recorder.stop()

    assert _MockBackendHandler.received_requests == [
        {
            "method": "POST",
            "path": "/v1/responses",
            "body": request_body,
        }
    ]

    assert len(calls) == 1

    assert calls[0]["call_index"] == 1
    assert calls[0]["request"] == {
        "method": "POST",
        "path": "/v1/responses",
        "body": {
            "model": "test-model",
            "input": "Hello",
        },
    }

    assert calls[0]["response"] == {
        "status": 200,
        "streamed": False,
        "body": {
            "result": "completed",
        },
    }

    assert calls[0]["error"] is None

    log_files = list(
        log_dir.glob("*.jsonl")
    )

    assert len(log_files) == 1

    persisted_calls = [
        json.loads(line)
        for line in log_files[0].read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert persisted_calls == calls
    assert log_files[0].stem == (
        calls[0]["session_id"]
    )


def test_stop_before_start_is_rejected(
    tmp_path: Path,
) -> None:
    recorder = ProxyRecorder(
        _make_config(
            backend_url="http://127.0.0.1:1",
            log_dir=tmp_path,
        )
    )

    with pytest.raises(
        RuntimeError,
        match="has not been started",
    ):
        recorder.stop()


def test_second_start_is_rejected(
    tmp_path: Path,
) -> None:
    recorder = ProxyRecorder(
        _make_config(
            backend_url="http://127.0.0.1:1",
            log_dir=tmp_path,
        )
    )

    recorder.start()

    try:
        with pytest.raises(
            RuntimeError,
            match="already started",
        ):
            recorder.start()

    finally:
        recorder.stop()


def test_second_stop_is_rejected(
    tmp_path: Path,
) -> None:
    recorder = ProxyRecorder(
        _make_config(
            backend_url="http://127.0.0.1:1",
            log_dir=tmp_path,
        )
    )

    recorder.start()
    calls = recorder.stop()

    assert calls == []

    with pytest.raises(
        RuntimeError,
        match="already stopped",
    ):
        recorder.stop()


def test_start_fails_when_port_is_already_in_use(
    tmp_path: Path,
) -> None:
    occupied_socket = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    )

    occupied_socket.bind(("127.0.0.1", 0))
    occupied_socket.listen(1)

    occupied_port = occupied_socket.getsockname()[1]

    recorder = ProxyRecorder(
        _make_config(
            backend_url="http://127.0.0.1:1",
            log_dir=tmp_path,
            listen_port=occupied_port,
        )
    )

    try:
        with pytest.raises(
            RuntimeError,
            match="could not start ProxyRecorder",
        ):
            recorder.start()

    finally:
        occupied_socket.close()
