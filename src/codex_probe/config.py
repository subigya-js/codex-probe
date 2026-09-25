"""Configuration parsing and validation for CodexProbe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit


WireApi = Literal["responses", "chat"]
AuthMode = Literal["forward", "environment", "none"]


class ConfigError(ValueError):
    """Raised when CodexProbe receives an invalid configuration."""


@dataclass(frozen=True, slots=True)
class AuthConfig:
    """Validated backend authentication configuration."""

    mode: AuthMode
    environment_variable: str | None = None


@dataclass(frozen=True, slots=True)
class BackendConfig:
    """Validated configuration for the real LLM backend."""

    name: str
    base_url: str
    wire_api: WireApi
    auth: AuthConfig


@dataclass(frozen=True, slots=True)
class ListenConfig:
    """Validated local proxy listener configuration."""

    host: str = "127.0.0.1"
    port: int = 0


@dataclass(frozen=True, slots=True)
class ProxyConfig:
    """Complete validated configuration used by ProxyRecorder."""

    backend: BackendConfig
    listen: ListenConfig
    log_dir: Path


def parse_config(config: Mapping[str, object]) -> ProxyConfig:
    """Validate a dictionary-like configuration and return typed values."""

    config_mapping = _require_mapping(config, "config")
    _validate_keys(
        config_mapping,
        allowed={"backend", "listen", "log_dir"},
        required={"backend"},
        field="config",
    )

    backend = _parse_backend(config_mapping["backend"])
    listen = _parse_listen(config_mapping.get("listen", {}))
    log_dir = _parse_log_dir(
        config_mapping.get("log_dir", ".codex-probe/logs")
    )

    return ProxyConfig(
        backend=backend,
        listen=listen,
        log_dir=log_dir,
    )


def _parse_backend(value: object) -> BackendConfig:
    backend = _require_mapping(value, "backend")
    _validate_keys(
        backend,
        allowed={"name", "base_url", "wire_api", "auth"},
        required={"name", "base_url", "wire_api", "auth"},
        field="backend",
    )

    name = _require_non_empty_string(backend["name"], "backend.name")
    base_url = _parse_base_url(backend["base_url"])
    wire_api = _parse_wire_api(backend["wire_api"])
    auth = _parse_auth(backend["auth"])

    return BackendConfig(
        name=name,
        base_url=base_url,
        wire_api=wire_api,
        auth=auth,
    )


def _parse_auth(value: object) -> AuthConfig:
    auth = _require_mapping(value, "backend.auth")
    _validate_keys(
        auth,
        allowed={"mode", "environment_variable"},
        required={"mode"},
        field="backend.auth",
    )

    raw_mode = _require_non_empty_string(auth["mode"], "backend.auth.mode")
    allowed_modes = {"forward", "environment", "none"}

    if raw_mode not in allowed_modes:
        raise ConfigError(
            "backend.auth.mode must be one of: "
            "'forward', 'environment', or 'none'"
        )

    environment_variable = auth.get("environment_variable")

    if raw_mode == "environment":
        variable_name = _require_non_empty_string(
            environment_variable,
            "backend.auth.environment_variable",
        )
        _validate_environment_variable_name(variable_name)
        return AuthConfig(
            mode="environment",
            environment_variable=variable_name,
        )

    if environment_variable is not None:
        raise ConfigError(
            "backend.auth.environment_variable is only valid when "
            "backend.auth.mode is 'environment'"
        )

    if raw_mode == "forward":
        return AuthConfig(mode="forward")

    return AuthConfig(mode="none")


def _parse_listen(value: object) -> ListenConfig:
    listen = _require_mapping(value, "listen")
    _validate_keys(
        listen,
        allowed={"host", "port"},
        required=set(),
        field="listen",
    )

    host = _require_non_empty_string(
        listen.get("host", "127.0.0.1"),
        "listen.host",
    )
    port = listen.get("port", 0)

    if not _is_loopback_host(host):
        raise ConfigError(
            "listen.host must be a loopback address such as "
            "'127.0.0.1', '::1', or 'localhost'"
        )

    if isinstance(port, bool) or not isinstance(port, int):
        raise ConfigError("listen.port must be an integer")

    if not 0 <= port <= 65535:
        raise ConfigError("listen.port must be between 0 and 65535")

    return ListenConfig(host=host, port=port)


def _parse_base_url(value: object) -> str:
    base_url = _require_non_empty_string(value, "backend.base_url")
    parsed = urlsplit(base_url)

    if parsed.scheme not in {"http", "https"}:
        raise ConfigError("backend.base_url must use http or https")

    if parsed.hostname is None:
        raise ConfigError("backend.base_url must include a hostname")

    if parsed.username is not None or parsed.password is not None:
        raise ConfigError(
            "backend.base_url must not contain embedded credentials"
        )

    if parsed.query:
        raise ConfigError("backend.base_url must not contain a query string")

    if parsed.fragment:
        raise ConfigError("backend.base_url must not contain a fragment")

    if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname):
        raise ConfigError(
            "backend.base_url may use plain HTTP only for a loopback host"
        )

    return base_url.rstrip("/")


def _parse_wire_api(value: object) -> WireApi:
    wire_api = _require_non_empty_string(value, "backend.wire_api")

    if wire_api == "responses":
        return "responses"

    if wire_api == "chat":
        return "chat"

    raise ConfigError(
        "backend.wire_api must be either 'responses' or 'chat'"
    )


def _parse_log_dir(value: object) -> Path:
    raw_path = _require_non_empty_string(value, "log_dir")
    return Path(raw_path).expanduser()


def _require_mapping(
    value: object,
    field: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field} must be an object")

    if not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{field} keys must be strings")

    return value


def _require_non_empty_string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{field} must be a string")

    normalized = value.strip()

    if not normalized:
        raise ConfigError(f"{field} must not be empty")

    return normalized


def _validate_keys(
    value: Mapping[str, object],
    *,
    allowed: set[str],
    required: set[str],
    field: str,
) -> None:
    unknown = set(value) - allowed
    missing = required - set(value)

    if unknown:
        names = ", ".join(sorted(unknown))
        raise ConfigError(f"{field} contains unknown fields: {names}")

    if missing:
        names = ", ".join(sorted(missing))
        raise ConfigError(f"{field} is missing required fields: {names}")


def _validate_environment_variable_name(name: str) -> None:
    first_character = name[0]

    if not (first_character.isalpha() or first_character == "_"):
        raise ConfigError(
            "backend.auth.environment_variable is not a valid "
            "environment variable name"
        )

    if not all(character.isalnum() or character == "_" for character in name):
        raise ConfigError(
            "backend.auth.environment_variable is not a valid "
            "environment variable name"
        )


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True

    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False
