"""Tests for CodexProbe configuration validation."""

from pathlib import Path

import pytest

from codex_probe.config import ConfigError, parse_config


def _make_config(
    *,
    base_url: object = "http://127.0.0.1:11434/v1/",
    wire_api: object = "responses",
    auth: object | None = None,
    listen_host: object = "127.0.0.1",
    listen_port: object = 0,
) -> dict[str, object]:
    """Create a fresh configuration dictionary for each test."""

    if auth is None:
        auth = {"mode": "none"}

    return {
        "backend": {
            "name": "ollama-qwen",
            "base_url": base_url,
            "wire_api": wire_api,
            "auth": auth,
        },
        "listen": {
            "host": listen_host,
            "port": listen_port,
        },
        "log_dir": ".codex-probe/logs",
    }


def test_parse_valid_config() -> None:
    config = parse_config(_make_config())

    assert config.backend.name == "ollama-qwen"
    assert config.backend.base_url == "http://127.0.0.1:11434/v1"
    assert config.backend.wire_api == "responses"
    assert config.backend.auth.mode == "none"
    assert config.backend.auth.environment_variable is None
    assert config.listen.host == "127.0.0.1"
    assert config.listen.port == 0
    assert config.log_dir == Path(".codex-probe/logs")


def test_defaults_are_applied() -> None:
    raw_config = {
        "backend": {
            "name": "ollama",
            "base_url": "http://localhost:11434/v1",
            "wire_api": "responses",
            "auth": {
                "mode": "none",
            },
        }
    }

    config = parse_config(raw_config)

    assert config.listen.host == "127.0.0.1"
    assert config.listen.port == 0
    assert config.log_dir == Path(".codex-probe/logs")


def test_environment_auth_is_parsed() -> None:
    raw_config = _make_config(
        auth={
            "mode": "environment",
            "environment_variable": "OPENAI_API_KEY",
        }
    )

    config = parse_config(raw_config)

    assert config.backend.auth.mode == "environment"
    assert (
        config.backend.auth.environment_variable
        == "OPENAI_API_KEY"
    )


def test_forward_auth_is_parsed() -> None:
    raw_config = _make_config(
        auth={
            "mode": "forward",
        }
    )

    config = parse_config(raw_config)

    assert config.backend.auth.mode == "forward"
    assert config.backend.auth.environment_variable is None


def test_missing_backend_is_rejected() -> None:
    with pytest.raises(
        ConfigError,
        match="config is missing required fields: backend",
    ):
        parse_config({})


def test_unknown_top_level_field_is_rejected() -> None:
    raw_config = _make_config()
    raw_config["unexpected"] = True

    with pytest.raises(
        ConfigError,
        match="config contains unknown fields: unexpected",
    ):
        parse_config(raw_config)


def test_none_auth_rejects_environment_variable() -> None:
    raw_config = _make_config(
        auth={
            "mode": "none",
            "environment_variable": "OPENAI_API_KEY",
        }
    )

    with pytest.raises(
        ConfigError,
        match="environment_variable is only valid",
    ):
        parse_config(raw_config)


def test_invalid_environment_variable_name_is_rejected() -> None:
    raw_config = _make_config(
        auth={
            "mode": "environment",
            "environment_variable": "9INVALID-NAME",
        }
    )

    with pytest.raises(
        ConfigError,
        match="not a valid environment variable name",
    ):
        parse_config(raw_config)


def test_invalid_wire_api_is_rejected() -> None:
    raw_config = _make_config(
        wire_api="completions",
    )

    with pytest.raises(
        ConfigError,
        match="wire_api must be either 'responses' or 'chat'",
    ):
        parse_config(raw_config)


def test_remote_plain_http_backend_is_rejected() -> None:
    raw_config = _make_config(
        base_url="http://api.example.com/v1",
    )

    with pytest.raises(
        ConfigError,
        match="plain HTTP only for a loopback host",
    ):
        parse_config(raw_config)


def test_embedded_backend_credentials_are_rejected() -> None:
    raw_config = _make_config(
        base_url="https://user:password@api.example.com/v1",
    )

    with pytest.raises(
        ConfigError,
        match="must not contain embedded credentials",
    ):
        parse_config(raw_config)


def test_non_loopback_listener_is_rejected() -> None:
    raw_config = _make_config(
        listen_host="0.0.0.0",
    )

    with pytest.raises(
        ConfigError,
        match="listen.host must be a loopback address",
    ):
        parse_config(raw_config)


@pytest.mark.parametrize(
    ("port", "expected_message"),
    [
        (True, "listen.port must be an integer"),
        ("8000", "listen.port must be an integer"),
        (-1, "listen.port must be between 0 and 65535"),
        (65536, "listen.port must be between 0 and 65535"),
    ],
)
def test_invalid_listen_port_is_rejected(
    port: object,
    expected_message: str,
) -> None:
    raw_config = _make_config(
        listen_port=port,
    )

    with pytest.raises(
        ConfigError,
        match=expected_message,
    ):
        parse_config(raw_config)
