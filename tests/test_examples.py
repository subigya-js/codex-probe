"""Tests for the example configuration files."""

import json
from pathlib import Path

import pytest

from codex_probe.config import parse_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"


@pytest.mark.parametrize(
    (
        "filename",
        "expected_backend",
        "expected_auth_mode",
    ),
    [
        (
            "ollama-qwen.json",
            "ollama-qwen2.5-coder",
            "none",
        ),
        (
            "openai.json",
            "openai",
            "environment",
        ),
    ],
)
def test_example_config_is_valid(
    filename: str,
    expected_backend: str,
    expected_auth_mode: str,
) -> None:
    config_path = EXAMPLES_DIR / filename

    with config_path.open(
        mode="r",
        encoding="utf-8",
    ) as config_file:
        config = json.load(config_file)

    assert isinstance(config, dict)

    parsed_config = parse_config(config)

    assert parsed_config.backend.name == expected_backend
    assert parsed_config.backend.auth.mode == expected_auth_mode
