#!/usr/bin/env python3
"""Collect the FastCode config DTO JSON schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fastcode.main._config_schema_root import FastCodeConfigDTO


def collect_schema() -> dict[str, Any]:
    """Return the generated JSON schema for the root runtime config DTO."""
    return FastCodeConfigDTO.model_json_schema()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        "-o",
        help="Optional path to write the collected schema JSON.",
    )
    args = parser.parse_args()

    payload = json.dumps(collect_schema(), indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(f"{payload}\n")
        return
    print(payload)


if __name__ == "__main__":
    main()
