"""Availability-gated command smokes for stable SCIP tool profiles."""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from fastcode.scip.indexers import SUPPORTED_LANGUAGES, get_scip_indexer_profile


def _stable_profiles() -> list[tuple[str, str]]:
    requested = {
        item.strip()
        for item in os.environ.get("FASTCODE_SCIP_SMOKE_LANGUAGES", "").split(",")
        if item.strip()
    }
    seen_binaries: set[str] = set()
    profiles: list[tuple[str, str]] = []
    for language in sorted(SUPPORTED_LANGUAGES):
        if requested and language not in requested:
            continue
        profile = get_scip_indexer_profile(language)
        if profile is None or profile.experimental:
            continue
        if profile.binary_name in seen_binaries:
            continue
        seen_binaries.add(profile.binary_name)
        profiles.append((language, profile.binary_name))
    return profiles


@pytest.mark.parametrize(("language", "binary_name"), _stable_profiles())
def test_stable_scip_tool_binary_reports_version(
    language: str, binary_name: str
) -> None:
    binary_path = shutil.which(binary_name)
    if binary_path is None:
        pytest.skip(f"{binary_name} not installed for {language} SCIP smoke")

    result = subprocess.run(
        [binary_path, "--version"],
        check=False,
        text=True,
        capture_output=True,
        timeout=20,
    )

    unavailable = f"Unknown binary '{binary_name}'"
    if result.returncode != 0 and unavailable in result.stderr:
        pytest.skip(f"{binary_name} rustup proxy is installed but unavailable")

    assert result.returncode == 0, result.stderr or result.stdout
    assert (result.stdout or result.stderr).strip()
