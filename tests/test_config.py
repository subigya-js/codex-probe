"""Tests for CodexProbe configuration validation."""

from pathlib import Path

from codex_probe.config import parse_config


VALID_CONFIG: dict[str, object] = {
    "backend": {
        "name": "ollama-qwen",
        "base_url": "http://127.0.0.1:11434/v1/",
        "wire_api": "responses",
        "auth": {
            "mode": "none",
        },
    },
    "listen": {
        "host": "127.0.0.1",
        "port": 0,
    },
    "log_dir": ".codex-probe/logs",
    "seed": 42,
}


def test_parse_valid_config() -> None:
    config = parse_config(VALID_CONFIG)

    assert config.backend.name == "ollama-qwen"
    assert config.backend.base_url == "http://127.0.0.1:11434/v1"
    assert config.backend.wire_api == "responses"
    assert config.backend.auth.mode == "none"
    assert config.backend.auth.environment_variable is None
    assert config.listen.host == "127.0.0.1"
    assert config.listen.port == 0
    assert config.log_dir == Path(".codex-probe/logs")
    assert config.seed == 42


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
    assert config.seed is None
