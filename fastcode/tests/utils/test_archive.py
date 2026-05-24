from __future__ import annotations

import io
import stat
import zipfile
from pathlib import Path

import pytest

from fastcode.utils.archive import (
    UnsafeArchiveError,
    ZipExtractionLimits,
    safe_extract_zip,
    safe_repo_name_from_archive,
)


def _zip_with(entries: dict[str, bytes]) -> zipfile.ZipFile:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zip_ref:
        for name, payload in entries.items():
            zip_ref.writestr(name, payload)
    buffer.seek(0)
    return zipfile.ZipFile(buffer, "r")


def test_safe_extract_zip_rejects_path_traversal(tmp_path: Path) -> None:
    with (
        _zip_with({"../escape.py": b"bad"}) as zip_ref,
        pytest.raises(UnsafeArchiveError),
    ):
        safe_extract_zip(zip_ref, tmp_path)

    assert not (tmp_path.parent / "escape.py").exists()


def test_safe_extract_zip_rejects_absolute_paths(tmp_path: Path) -> None:
    with (
        _zip_with({"/tmp/escape.py": b"bad"}) as zip_ref,
        pytest.raises(UnsafeArchiveError),
    ):
        safe_extract_zip(zip_ref, tmp_path)


def test_safe_extract_zip_rejects_symlinks(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zip_ref:
        info = zipfile.ZipInfo("link")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zip_ref.writestr(info, "target")
    buffer.seek(0)

    with zipfile.ZipFile(buffer, "r") as zip_ref, pytest.raises(UnsafeArchiveError):
        safe_extract_zip(zip_ref, tmp_path)


def test_safe_extract_zip_enforces_expanded_size_limit(tmp_path: Path) -> None:
    with (
        _zip_with({"large.txt": b"x" * 20}) as zip_ref,
        pytest.raises(UnsafeArchiveError),
    ):
        safe_extract_zip(
            zip_ref,
            tmp_path,
            limits=ZipExtractionLimits(max_total_uncompressed_bytes=10),
        )


def test_safe_extract_zip_extracts_valid_archive(tmp_path: Path) -> None:
    with _zip_with({"repo/main.py": b"print('ok')\n"}) as zip_ref:
        safe_extract_zip(zip_ref, tmp_path)

    assert (tmp_path / "repo" / "main.py").read_text(
        encoding="utf-8"
    ) == "print('ok')\n"


def test_safe_repo_name_from_archive_sanitizes_upload_name() -> None:
    assert safe_repo_name_from_archive("../../bad repo-main.zip") == "bad_repo"


def test_zip_extraction_limits_coerces_foundation_types() -> None:
    limits = ZipExtractionLimits(
        max_members=5,
        max_total_uncompressed_bytes=10,
        max_member_uncompressed_bytes=3,
    )

    assert limits.max_members.as_int() == 5
    assert limits.max_total_uncompressed_bytes.as_bytes() == 10
    assert limits.max_member_uncompressed_bytes.as_bytes() == 3
