"""Run CodexProbe using a JSON configuration file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from codex_probe import ProxyRecorder


def main() -> None:
    """Start CodexProbe and wait until the user stops it."""

    arguments = _parse_arguments()
    config = _load_config(arguments.config)

    recorder = ProxyRecorder(config)
    endpoint = recorder.start()

    print(f"CodexProbe is running at: {endpoint}")
    print("Press Enter or Ctrl+C to stop.")

    try:
        input()

    except (KeyboardInterrupt, EOFError):
        pass

    calls = recorder.stop()

    print(
        f"CodexProbe stopped after recording "
        f"{len(calls)} call(s)."
    )


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run CodexProbe with a JSON configuration."
        )
    )

    parser.add_argument(
        "config",
        type=Path,
        help="Path to a CodexProbe JSON configuration file.",
    )

    return parser.parse_args()


def _load_config(
    config_path: Path,
) -> dict[str, object]:
    """Load a JSON configuration file."""

    with config_path.open(
        mode="r",
        encoding="utf-8",
    ) as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError(
            "configuration file must contain a JSON object"
        )

    return config


if __name__ == "__main__":
    main()
