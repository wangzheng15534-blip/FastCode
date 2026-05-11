from __future__ import annotations

from pathlib import Path

import pytest

from fastcode.indexing.loader import RepositoryLoader
from fastcode.indexing.pipeline import IndexPipeline


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


def test_load_from_path_uses_in_place_mode_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("print('a')\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    loader = RepositoryLoader(
        {
            "repo_root": str(workspace),
            "repository": {"supported_extensions": [".py"]},
        }
    )
    monkeypatch.setattr(
        "fastcode.indexing.loader.shutil.copytree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("in-place local loading must not copy the repository")
        ),
    )

    loaded_path = loader.load_from_path(str(repo))

    assert loaded_path == str(repo)
    assert loader.repo_path == str(repo)
    assert loader.repo_load_mode == "in_place"
    assert loader.repo_is_workspace_copy is False
    assert loader.last_load_stats["copied_bytes"] == 0
    assert loader.get_repository_info()["load_mode"] == "in_place"
    assert loader.get_repository_info()["workspace_copy"] is False
    assert not (workspace / "repo").exists()


def test_load_from_path_copy_mode_preserves_workspace_copy(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("print('a')\n", encoding="utf-8")
    loader = RepositoryLoader(
        {
            "repo_root": str(tmp_path / "workspace"),
            "repository": {
                "supported_extensions": [".py"],
                "local_source_mode": "copy",
            },
        }
    )

    loaded_path = Path(loader.load_from_path(str(repo)))

    assert loaded_path != repo
    assert loaded_path == tmp_path / "workspace" / "repo"
    assert (loaded_path / "a.py").read_text(encoding="utf-8") == "print('a')\n"
    assert loader.repo_load_mode == "copy"
    assert loader.repo_is_workspace_copy is True
    assert loader.last_load_stats["copied_files"] == 1
    assert loader.last_load_stats["copied_bytes"] > 0
    info = loader.get_repository_info()
    assert info["load_mode"] == "copy"
    assert info["workspace_copy"] is True
    assert info["copied_files"] == 1


def test_pipeline_refuses_checkout_on_in_place_local_repo() -> None:
    pipeline = IndexPipeline.__new__(IndexPipeline)
    pipeline.loader = type(
        "Loader",
        (),
        {"repo_path": "/tmp/source-repo", "repo_load_mode": "in_place"},
    )()

    with pytest.raises(RuntimeError, match="would mutate an in-place local repository"):
        pipeline._checkout_target_ref(ref="feature")


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
