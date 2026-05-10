from __future__ import annotations

from pathlib import Path

import pytest

from fastcode.indexing.loader import RepositoryLoader


def test_scan_files_can_emit_fingerprints(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "a.py"
    source.write_text("print('a')\n", encoding="utf-8")
    loader = RepositoryLoader(
        {
            "repo_root": str(tmp_path / "workspace"),
            "repository": {"supported_extensions": [".py"]},
        }
    )
    loader.repo_path = str(repo)
    loader.repo_name = "repo"

    files = loader.scan_files(include_fingerprints=True)

    assert len(files) == 1
    assert files[0]["relative_path"] == "a.py"
    assert files[0]["content_hash"]
    assert files[0]["blob_oid"] == files[0]["content_hash"]
    assert files[0]["mtime"] > 0


def test_get_repository_info_uses_supplied_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    loader = RepositoryLoader(
        {
            "repo_root": str(tmp_path / "workspace"),
            "repository": {"supported_extensions": [".py"]},
        }
    )
    loader.repo_path = str(repo)
    loader.repo_name = "repo"

    monkeypatch.setattr(
        loader,
        "scan_files",
        lambda: (_ for _ in ()).throw(AssertionError("must use supplied files")),
    )

    info = loader.get_repository_info(
        files=[{"relative_path": "a.py", "size": 1024 * 1024}]
    )

    assert info["file_count"] == 1
    assert info["total_size_mb"] == 1.0
