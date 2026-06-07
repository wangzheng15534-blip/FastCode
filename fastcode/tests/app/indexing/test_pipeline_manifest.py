from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastcode.app.indexing.pipeline.manifest import detect_file_changes


def _write_manifest(
    persist_dir: Path,
    repo_name: str,
    files: dict[str, dict[str, Any]],
) -> None:
    (persist_dir / f"{repo_name}_manifest.json").write_text(
        json.dumps({"repo_name": repo_name, "files": files}),
        encoding="utf-8",
    )


def test_detect_file_changes_reports_each_deleted_file_once(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    kept = repo / "kept.py"
    deleted = repo / "deleted.py"
    kept.write_text("print('kept')\n", encoding="utf-8")
    deleted.write_text("print('deleted')\n", encoding="utf-8")
    kept_stat = kept.stat()
    deleted_stat = deleted.stat()
    _write_manifest(
        tmp_path,
        "repo",
        {
            "kept.py": {"mtime": kept_stat.st_mtime, "size": kept_stat.st_size},
            "deleted.py": {
                "mtime": deleted_stat.st_mtime,
                "size": deleted_stat.st_size,
            },
        },
    )
    deleted.unlink()

    result = detect_file_changes(
        "repo",
        [{"relative_path": "kept.py", "path": str(kept)}],
        str(tmp_path),
    )

    assert result is not None
    assert result["unchanged"] == ["kept.py"]
    assert result["deleted"] == ["deleted.py"]


def test_detect_file_changes_marks_manifest_files_deleted_when_inventory_empty(
    tmp_path: Path,
) -> None:
    _write_manifest(
        tmp_path,
        "repo",
        {
            "a.py": {"mtime": 1.0, "size": 1},
            "b.py": {"mtime": 2.0, "size": 2},
        },
    )

    result = detect_file_changes("repo", [], str(tmp_path))

    assert result is not None
    assert result["deleted"] == ["a.py", "b.py"]
    assert result["added"] == []
    assert result["modified"] == []
    assert result["unchanged"] == []
